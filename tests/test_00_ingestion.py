import importlib
import json

import pandas as pd
import pytest

ing = importlib.import_module("flows.00_ingestion")


def test_split_labeled_keeps_source_rows_unchanged():
    df = pd.DataFrame({"Survived": [0, 1, None, None]})
    before = df.copy(deep=True)

    labeled, unlabeled = ing.split_labeled(df, 2)

    assert len(labeled) == 2 and len(unlabeled) == 2
    assert unlabeled["Survived"].isna().all()
    pd.testing.assert_frame_equal(df, before)


def test_split_labeled_rejects_invalid_count():
    with pytest.raises(ValueError):
        ing.split_labeled(pd.DataFrame({"x": [1]}), 2)


def test_run_writes_bronze_and_auditable_metadata(tmp_path):
    original = pd.DataFrame(
        {"PassengerId": range(1, 6), "Survived": [0, 1, 1, None, None]}
    )
    source = tmp_path / "source.csv"
    original.to_csv(source, index=False)
    out = tmp_path / "bronze"

    metadata = ing.run(source=source, n_labeled=3, out_dir=out)

    labeled_path = out / "titanic_labeled.csv"
    unlabeled_path = out / "titanic_unlabeled.csv"
    assert labeled_path.exists() and unlabeled_path.exists()
    assert metadata["rows_labeled"] == 3 and metadata["rows_unlabeled"] == 2
    assert len(metadata["sha256_labeled"]) == 64
    saved_metadata = json.loads((out / "_ingestion_metadata.json").read_text())
    assert saved_metadata["sha256_unlabeled"] == metadata["sha256_unlabeled"]
    pd.testing.assert_frame_equal(
        pd.read_csv(labeled_path), original.iloc[:3].reset_index(drop=True)
    )
