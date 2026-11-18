"""Modelado reproducible: Silver -> evaluación y modelo entrenado."""

from __future__ import annotations

import importlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import BRONZE, MODELS, SILVER, SILVER_NAME

log = get_logger(__name__)

TARGET = "Survived"
NUM_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Familia"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "TTitulo", "Alone"]
CABIN_NUM = ["HasCabin"]
CABIN_CAT = ["TCabin"]
CANDIDATE_COLUMNS = NUM_FEATURES + CAT_FEATURES + CABIN_NUM + CABIN_CAT
PRIMARY_METRIC = "roc_auc"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

MODEL_NAME = "titanic_model_v1"
FINAL_ESTIMATOR = "hgb"
FINAL_PARAMS = {}
USE_CABIN_FEATURES = False
THRESHOLD = 0.5


def _silver_path(path: Path | None) -> Path:
    if path is None:
        return SILVER / SILVER_NAME
    path = Path(path)
    return path if path.suffix == ".parquet" else path / SILVER_NAME


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Carga las variables candidatas y la edad sin imputar para evitar fuga."""
    df = pd.read_parquet(_silver_path(path))
    missing = sorted(set(CANDIDATE_COLUMNS + ["AgeOriginal", TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Silver no contiene columnas requeridas por el modelo: {missing}")
    X = df[CANDIDATE_COLUMNS].copy()
    X["Age"] = df["AgeOriginal"]
    return X, df[TARGET].astype(int)


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
):
    """Divide de forma estratificada y reproducible."""
    return train_test_split(
        X,
        y,
        test_size=settings.test_size if test_size is None else test_size,
        random_state=settings.random_state if random_state is None else random_state,
        stratify=y,
    )


class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Imputa Age con promedios aprendidos exclusivamente durante fit."""

    def __init__(self, group_cols=("Sex",), target="Age"):
        self.group_cols = group_cols
        self.target = target

    def fit(self, X, y=None):
        if self.target not in X.columns:
            raise ValueError(f"Falta la columna objetivo de imputación {self.target!r}.")
        self.group_cols_ = [column for column in self.group_cols if column in X.columns]
        if not self.group_cols_:
            raise ValueError("No hay columnas de grupo disponibles para imputar Age.")
        group_key = self.group_cols_[0] if len(self.group_cols_) == 1 else self.group_cols_
        self.group_means_ = X.groupby(group_key, dropna=False)[self.target].mean()
        self.global_mean_ = X[self.target].mean()
        if pd.isna(self.global_mean_):
            raise ValueError("No se puede imputar Age: todas las edades de entrenamiento faltan.")
        return self

    def transform(self, X):
        if not hasattr(self, "group_means_"):
            raise ValueError("GroupAgeImputer debe ajustarse antes de transformar.")
        result = X.copy()
        means = self.group_means_.rename("_fill").reset_index()
        fill_values = (
            result[self.group_cols_]
            .merge(means, on=self.group_cols_, how="left")["_fill"]
            .to_numpy(dtype=float)
        )
        fill_values = np.where(np.isnan(fill_values), self.global_mean_, fill_values)
        ages = result[self.target].to_numpy(dtype=float, copy=True)
        missing = np.isnan(ages)
        ages[missing] = fill_values[missing]
        result[self.target] = ages
        return result


def build_pipeline(
    model,
    num_features: list[str] | None = None,
    cat_features: list[str] | None = None,
) -> Pipeline:
    """Construye imputación, codificación y estimador en un pipeline."""
    num_features = NUM_FEATURES if num_features is None else num_features
    cat_features = CAT_FEATURES if cat_features is None else cat_features
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, num_features),
            ("cat", categorical, cat_features),
        ],
        sparse_threshold=0,
    )
    return Pipeline(
        [
            ("age_imputer", GroupAgeImputer(group_cols=("Sex", "Pclass"))),
            ("prep", preprocessor),
            ("model", model),
        ]
    )


def get_candidates(random_state: int = 42) -> dict:
    """Devuelve estimadores sin entrenar para comparar familias de modelos."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000),
        "tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        "forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
    }


def cross_validate_candidates(
    X,
    y,
    num_features=None,
    cat_features=None,
    random_state=None,
    cv_folds=None,
) -> pd.DataFrame:
    """Evalúa candidatos en CV y devuelve medias/desviaciones de train y validación."""
    seed = settings.random_state if random_state is None else random_state
    folds = settings.cv_folds if cv_folds is None else cv_folds
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name, estimator in get_candidates(seed).items():
        pipeline = build_pipeline(estimator, num_features, cat_features)
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,
            error_score="raise",
        )
        row = {"model": name}
        for metric in SCORING:
            row[f"test_{metric}_mean"] = float(result[f"test_{metric}"].mean())
            row[f"test_{metric}_std"] = float(result[f"test_{metric}"].std())
            row[f"train_{metric}_mean"] = float(result[f"train_{metric}"].mean())
        rows.append(row)
        log.info(
            "%s CV %s: %.3f ± %.3f",
            name,
            PRIMARY_METRIC,
            row[f"test_{PRIMARY_METRIC}_mean"],
            row[f"test_{PRIMARY_METRIC}_std"],
        )
    return pd.DataFrame(rows).sort_values(
        f"test_{PRIMARY_METRIC}_mean", ascending=False, ignore_index=True
    )


def get_param_grids() -> dict[str, dict]:
    """Define espacios de búsqueda de hiperparámetros para los finalistas."""
    return {
        "logreg": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
        },
        "forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "hgb": {
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__max_iter": [100, 200],
            "model__l2_regularization": [0.0, 1.0],
        },
    }


def metrics_from_probabilities(y_test, probabilities, threshold: float = 0.5) -> dict:
    """Calcula métricas a partir de probabilidades ya obtenidas."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "n_test": len(y_test),
        "threshold": float(threshold),
    }


def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evalúa un pipeline entrenado y devuelve métricas serializables."""
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    return metrics_from_probabilities(y_test, probabilities, threshold)


def _bootstrap_roc_auc_interval(y_true, probabilities, random_state: int, n_boot: int = 1000):
    """Calcula IC percentil para ROC-AUC remuestreando pares con reemplazo."""
    rng = np.random.default_rng(random_state)
    y_values = np.asarray(y_true)
    scores = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(y_values), size=len(y_values))
        if np.unique(y_values[indices]).size < 2:
            continue
        scores.append(roc_auc_score(y_values[indices], probabilities[indices]))
    if not scores:
        raise ValueError("No se pudo calcular el intervalo: remuestras sin ambas clases.")
    lower, upper = np.quantile(scores, [0.025, 0.975])
    return float(lower), float(upper)


def train_final(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Entrena el pipeline congelado con todos los datos recibidos."""
    numeric = NUM_FEATURES + (CABIN_NUM if USE_CABIN_FEATURES else [])
    categorical = CAT_FEATURES + (CABIN_CAT if USE_CABIN_FEATURES else [])
    estimator = get_candidates(settings.random_state)[FINAL_ESTIMATOR]
    pipeline = build_pipeline(estimator, numeric, categorical)
    pipeline.set_params(**FINAL_PARAMS)
    return pipeline.fit(X, y)


def save_model(
    pipeline: Pipeline,
    metadata: dict,
    name: str = MODEL_NAME,
    models_dir: Path = MODELS,
) -> Path:
    """Guarda el pipeline y metadatos de auditoría."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, model_path)
    meta = {
        **metadata,
        "name": name,
        "saved_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "model_sha256": sha256_of(model_path),
    }
    model_path.with_suffix(".metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log.info("Modelo guardado en %s", model_path)
    return model_path


def load_model(path: Path) -> tuple[Pipeline, dict]:
    """Carga un modelo confiable y advierte si cambió scikit-learn."""
    path = Path(path)
    pipeline = joblib.load(path)
    metadata_path = path.with_suffix(".metadata.json")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning(
            "Modelo entrenado con scikit-learn %s; versión actual %s",
            meta.get("sklearn_version"),
            sklearn.__version__,
        )
    return pipeline, meta


def predict(
    pipeline: Pipeline, X: pd.DataFrame, threshold: float = THRESHOLD
) -> pd.DataFrame:
    """Devuelve probabilidad de supervivencia y etiqueta binaria."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")
    probabilities = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "proba_survive": probabilities,
            "pred": (probabilities >= threshold).astype(int),
        },
        index=X.index,
    )


def run(in_dir: Path | None = None, models_dir: Path = MODELS) -> Path:
    """Evalúa en holdout, entrena con datos etiquetados y guarda artefactos."""
    silver_path = _silver_path(in_dir)
    X, y = load_dataset(silver_path)
    X_train, X_test, y_train, y_test = split(X, y)

    evaluation_pipeline = train_final(X_train, y_train)
    probabilities = evaluation_pipeline.predict_proba(X_test)[:, 1]
    metrics = metrics_from_probabilities(y_test, probabilities, THRESHOLD)
    lower, upper = _bootstrap_roc_auc_interval(
        y_test, probabilities, settings.random_state
    )
    metrics["roc_auc_ci_95"] = [lower, upper]
    log.info("Evaluación holdout (uso único): %s", metrics)

    pipeline = train_final(X, y)
    features_num = NUM_FEATURES + (CABIN_NUM if USE_CABIN_FEATURES else [])
    features_cat = CAT_FEATURES + (CABIN_CAT if USE_CABIN_FEATURES else [])
    metadata = {
        "estimator": FINAL_ESTIMATOR,
        "params": FINAL_PARAMS,
        "features": {"numeric": features_num, "categorical": features_cat},
        "target": TARGET,
        "threshold": THRESHOLD,
        "random_state": settings.random_state,
        "n_train": len(X),
        "data_sha256": sha256_of(silver_path),
        "metrics": metrics,
        "evaluation": "single stratified holdout; bootstrap CI for ROC-AUC",
    }
    model_path = save_model(pipeline, metadata, models_dir=models_dir)

    unlabeled_path = BRONZE / "titanic_unlabeled.csv"
    if unlabeled_path.exists():
        unlabeled = pd.read_csv(unlabeled_path)
        if not unlabeled.empty:
            unlabeled["Survived"] = 0
            engineering = importlib.import_module("flows.01_data_engineering")
            silver_unlabeled = engineering.build_silver(unlabeled)
            features = silver_unlabeled[CANDIDATE_COLUMNS].copy()
            features["Age"] = silver_unlabeled["AgeOriginal"]
            predictions = predict(pipeline, features, THRESHOLD)
            predictions.to_csv(
                Path(models_dir) / "unlabeled_predictions.csv", index=False
            )
            log.info(
                "Prueba de humo completada: %d pasajeros sin etiqueta",
                len(predictions),
            )
    return model_path
