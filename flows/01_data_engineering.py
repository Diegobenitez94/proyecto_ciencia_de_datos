"""Transformaciones de ingeniería de datos para el conjunto del Titanic."""

import numpy as np
import pandas as pd


def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``Person``: ``child`` si Age < 16; en otro caso conserva Sex.

    Si la edad es desconocida, se conserva ``Sex``. La función no modifica
    el DataFrame recibido.
    """
    result = df.copy()
    result["Person"] = np.where(result["Age"] < 16, "child", result["Sex"])
    return result
