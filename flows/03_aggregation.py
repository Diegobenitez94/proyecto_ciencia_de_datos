"""Agregaciones analíticas: Silver -> Gold."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calcula estadísticas descriptivas de edad y supervivencia por grupos."""
    return {
        "age_by_survival": df.groupby("Survived")["Age"].describe().reset_index(),
        "survival_by_sex": (
            df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"]).reset_index()
        ),
        "survival_by_pclass": (
            df.groupby("Pclass")["Survived"]
            .agg(["mean", "sum", "count"])
            .reset_index()
        ),
        "survival_by_alone": (
            df.groupby("Alone")["Survived"]
            .agg(["mean", "sum", "count"])
            .reset_index()
        ),
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict[str, float | int]:
    """Prueba independencia entre sexo y supervivencia mediante chi-cuadrado."""
    table = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p_value, degrees, _ = stats.chi2_contingency(table)
    return {"chi2": float(chi2), "p_value": float(p_value), "dof": int(degrees)}


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las tasas de supervivencia por clase y sexo."""
    return (
        df.pivot_table(
            values="Survived", index="Pclass", columns="Sex", aggfunc="mean"
        )
        .reset_index()
        .rename(columns=str)
    )


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Resume tasa y cantidad de pasajeros por puerto."""
    return (
        df.groupby("Embarked")["Survived"]
        .agg(["mean", "count"])
        .reset_index()
    )


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Compara tasas de supervivencia por puerto, estratificadas por clase."""
    return (
        df.pivot_table(
            values="Survived", index="Embarked", columns="Pclass", aggfunc="mean"
        )
        .reset_index()
        .rename(columns=str)
    )


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Añade cuartil de tarifa calculado independientemente dentro de cada clase."""
    result = df.copy()
    result["FareQ"] = result.groupby("Pclass")["Fare"].transform(
        lambda values: pd.qcut(
            values, 4, labels=False, duplicates="drop"
        )
    )
    return result


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve tasas por cuartil de tarifa y clase."""
    quartiles = add_fare_quartile(df)
    return (
        quartiles.pivot_table(
            values="Survived", index="FareQ", columns="Pclass", aggfunc="mean"
        )
        .reset_index()
        .rename(columns=str)
    )


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Resume supervivencia por tamaño de familia."""
    return (
        df.groupby("Familia")["Survived"]
        .agg(["mean", "count"])
        .reset_index()
    )


def outlier_mask_iqr(series: pd.Series) -> pd.Series:
    """Marca valores fuera de 1.5 IQR, ignorando ausencias."""
    if series.dropna().empty:
        return pd.Series(False, index=series.index, dtype=bool)
    lower_quartile, upper_quartile = series.quantile([0.25, 0.75])
    spread = upper_quartile - lower_quartile
    lower = lower_quartile - 1.5 * spread
    upper = upper_quartile + 1.5 * spread
    return series.lt(lower).fillna(False) | series.gt(upper).fillna(False)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta atípicos por variable y compara supervivencia en tarifas atípicas."""
    variables = {
        "Fare": df["Fare"],
        "Age": df["AgeOriginal"] if "AgeOriginal" in df else df["Age"],
        "Familia": df["Familia"],
    }
    rows = []
    fare_mask = outlier_mask_iqr(variables["Fare"])
    for name, values in variables.items():
        mask = outlier_mask_iqr(values)
        row = {
            "feature": name,
            "n": int(values.notna().sum()),
            "n_outliers": int(mask.sum()),
            "outlier_rate": float(mask.sum() / values.notna().sum())
            if values.notna().any()
            else 0.0,
            "survival_rate_outliers": float(df.loc[fare_mask, "Survived"].mean())
            if name == "Fare" and fare_mask.any()
            else float("nan"),
            "survival_rate_others": float(df.loc[~fare_mask, "Survived"].mean())
            if name == "Fare" and (~fare_mask).any()
            else float("nan"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    """Lee Silver y guarda tablas de análisis en Gold."""
    df = pd.read_parquet((Path(in_dir) if in_dir is not None else SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(
        json.dumps(chi2_sex_survival(df), indent=2), encoding="utf-8"
    )
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
