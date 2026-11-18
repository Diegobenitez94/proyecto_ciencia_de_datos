"""Ingeniería de datos: Bronze -> Silver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_LABELED_NAME,
    SILVER,
    SILVER_FINAL_NAME,
    SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
DECKS = list("ABCDEFGTX")
TITLE_MAP = {
    "Master": "Mr",
    "the Countess": "Mrs",
    "Dona": "Mrs",
    "Mme": "Mrs",
    "Lady": "Miss",
    "Mlle": "Miss",
    "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Valida las columnas mínimas y unicidad de PassengerId."""
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en Bronze: {missing}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("PassengerId contiene valores duplicados.")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte variables numéricas a tipos estables sin modificar la entrada."""
    result = df.copy()
    for column in ("Survived", "Pclass", "SibSp", "Parch"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(int)
    for column in ("Age", "Fare"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result


def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega Person: child si Age < 16; si no, usa Sex, incluso con Age nula."""
    result = df.copy()
    result["Person"] = np.where(result["Age"] < 16, "child", result["Sex"])
    return result


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega cubierta y marca binaria de cabina registrada."""
    result = df.copy()
    result["TCabin"] = result["Cabin"].str[0].fillna("X")
    result["HasCabin"] = result["Cabin"].notna().astype(int)
    return result


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega tamaño familiar y etiqueta si el pasajero viaja solo."""
    result = df.copy()
    result["Familia"] = result["SibSp"] + result["Parch"]
    result["Alone"] = np.where(result["Familia"] > 0, "with Family", "Alone")
    return result


def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega etiqueta legible de supervivencia para visualizaciones."""
    result = df.copy()
    result["Survivor"] = result["Survived"].map({0: "No", 1: "Yes"})
    return result


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título del nombre del pasajero."""
    result = df.copy()
    result["TTitulo"] = (
        result["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    )
    return result


def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza títulos frecuentes y agrupa el resto como Otro."""
    result = df.copy()
    titles = result["TTitulo"].replace(TITLE_MAP)
    result["TTitulo"] = titles.where(titles.isin(MAIN_TITLES), "Otro")
    return result


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa Embarked con la moda observada."""
    result = df.copy()
    modes = result["Embarked"].mode()
    if modes.empty:
        raise ValueError("No se puede imputar Embarked: no hay valores observados.")
    result["Embarked"] = result["Embarked"].fillna(modes.iloc[0])
    return result


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa Age con el promedio por género y conserva edad original e indicador."""
    result = df.copy()
    result["AgeOriginal"] = result["Age"]
    result["AgeImputada"] = result["Age"].isna()
    before_std = result["Age"].std()
    means = result.groupby("Sex", dropna=False)["Age"].mean()
    result["Age"] = result["Age"].fillna(result["Sex"].map(means))
    after_std = result["Age"].std()
    log.info("Desviación estándar de Age antes/después de imputar: %s / %s", before_std, after_std)
    return result


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Crea codificaciones numéricas estables para análisis correlacional."""
    result = df.copy()
    result["NSex"] = result["Sex"].map({"female": 0, "male": 1})
    result["NEmbarked"] = result["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    result["NCabin"] = result["TCabin"].map(
        {deck: index for index, deck in enumerate(DECKS)}
    )
    result["NTitulo"] = result["TTitulo"].map(
        {"Mr": 0, "Mrs": 1, "Miss": 2, "Otro": 3}
    )
    return result


def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia sin las columnas categóricas originales seleccionadas."""
    return df.drop(columns=["Name", "Sex", "Embarked", "Cabin"]).copy()


def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Valida los campos requeridos no nulos en Silver."""
    required = [
        "Survived",
        "Pclass",
        "Age",
        "Fare",
        "Embarked",
        "Person",
        "Alone",
        "TCabin",
        "TTitulo",
        "NSex",
        "NEmbarked",
        "NCabin",
        "NTitulo",
    ]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en Silver: {missing_columns}")
    null_columns = df[required].columns[df[required].isna().any()].tolist()
    if null_columns:
        raise ValueError(f"Silver contiene nulos en columnas requeridas: {null_columns}")
    return df


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta transformaciones en orden, manteniendo AgeOriginal para auditoría."""
    return (
        bronze.pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    """Lee Bronze y guarda Silver completo y la vista numérica final."""
    bronze_path = Path(in_dir) / BRONZE_LABELED_NAME
    bronze = pd.read_csv(bronze_path)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver_path = out_dir / SILVER_NAME
    silver.to_parquet(silver_path, index=False)
    drop_original_columns(silver).to_parquet(
        out_dir / SILVER_FINAL_NAME, index=False
    )
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return silver_path
