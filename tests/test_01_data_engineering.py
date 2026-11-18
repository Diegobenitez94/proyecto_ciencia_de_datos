import importlib

import pandas as pd


data_engineering = importlib.import_module("flows.01_data_engineering")


def test_add_person_classifies_children_and_preserves_sex_for_adults():
    df = pd.DataFrame(
        {
            "Age": [2, 15, 16, 40],
            "Sex": ["male", "female", "male", "female"],
        }
    )

    result = data_engineering.add_person(df)

    assert result["Person"].tolist() == ["child", "child", "male", "female"]


def test_add_person_uses_sex_for_unknown_age_without_mutating_input():
    df = pd.DataFrame({"Age": [None], "Sex": ["female"]})
    original = df.copy(deep=True)

    result = data_engineering.add_person(df)

    assert result["Person"].tolist() == ["female"]
    pd.testing.assert_frame_equal(df, original)
