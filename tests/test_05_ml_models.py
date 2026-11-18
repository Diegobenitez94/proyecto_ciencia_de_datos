import importlib

import numpy as np
import pandas as pd
import pytest

ml = importlib.import_module("flows.05_ml_models")


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(
        {
            "Age": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 70, n)),
            "SibSp": rng.integers(0, 4, n),
            "Parch": rng.integers(0, 3, n),
            "Fare": rng.uniform(5, 100, n),
            "Familia": rng.integers(0, 6, n),
            "Pclass": rng.choice([1, 2, 3], n),
            "Sex": rng.choice(["male", "female"], n),
            "Embarked": rng.choice(["S", "C", "Q"], n),
            "TTitulo": rng.choice(["Mr", "Mrs", "Miss", "Otro"], n),
            "Alone": rng.choice(["Alone", "with Family"], n),
            "HasCabin": rng.integers(0, 2, n),
            "TCabin": rng.choice(list("XXXXCB"), n),
        }
    )
    y = pd.Series(((X["Sex"] == "female") & (rng.random(n) < 0.9)).astype(int))
    return X, y


def test_group_age_imputer_uses_only_training_statistics():
    train = pd.DataFrame(
        {"Sex": ["male", "male", "female"], "Age": [10.0, 20.0, 40.0]}
    )
    test = pd.DataFrame({"Sex": ["male", "other"], "Age": [np.nan, np.nan]})
    out = ml.GroupAgeImputer(group_cols=("Sex",)).fit(train).transform(test)
    assert out.loc[0, "Age"] == 15.0
    assert out.loc[1, "Age"] == pytest.approx(70 / 3)


def test_pipeline_fits_and_predicts_probabilities(toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    probabilities = pipe.predict_proba(X)[:, 1]
    assert probabilities.shape == (len(X),)
    assert ((0 <= probabilities) & (probabilities <= 1)).all()


def test_split_is_stratified_and_reproducible(toy):
    X, y = toy
    first = ml.split(X, y, test_size=0.25, random_state=1)
    second = ml.split(X, y, test_size=0.25, random_state=1)
    pd.testing.assert_index_equal(first[0].index, second[0].index)
    assert abs(first[2].mean() - first[3].mean()) < 0.03


def test_candidates_include_baseline_and_are_seeded():
    candidates = ml.get_candidates(42)
    assert {"dummy", "logreg", "tree", "forest", "hgb"} <= set(candidates)


def test_evaluate_on_test_returns_metrics(toy):
    X, y = toy
    X_train, X_test, y_train, y_test = ml.split(
        X, y, test_size=0.25, random_state=1
    )
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X_train, y_train)
    metrics = ml.evaluate_on_test(pipe, X_test, y_test, threshold=0.5)
    assert {"accuracy", "f1", "roc_auc", "n_test"} <= set(metrics)
    assert 0 <= metrics["accuracy"] <= 1 and metrics["n_test"] == len(X_test)


def test_save_and_load_roundtrip(tmp_path, toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    path = ml.save_model(
        pipe,
        {"estimator": "logreg", "n_train": len(X)},
        name="toy_model",
        models_dir=tmp_path,
    )
    loaded, meta = ml.load_model(path)
    assert np.array_equal(pipe.predict(X), loaded.predict(X))
    assert {"sklearn_version", "python_version", "model_sha256", "saved_at"} <= set(meta)
    assert meta["n_train"] == len(X)


def test_predict_returns_probabilities_and_labels(toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    result = ml.predict(pipe, X, threshold=0.5)
    assert list(result.columns) == ["proba_survive", "pred"]
    assert set(result["pred"].unique()) <= {0, 1}


def test_predict_rejects_invalid_threshold(toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    with pytest.raises(ValueError, match="threshold"):
        ml.predict(pipe, X, threshold=1.1)
