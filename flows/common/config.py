"""Configuración del proyecto: valores por defecto más `.env` opcional."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from flows.common.paths import ROOT

load_dotenv(ROOT / ".env")

DEFAULT_URL = "https://paste.c-net.org/MarvelJinxed"


@dataclass(frozen=True)
class Settings:
    data_url: str
    local_path: str
    n_labeled: int
    random_state: int
    test_size: float
    cv_folds: int


def load_settings() -> Settings:
    settings = Settings(
        data_url=os.getenv("TITANIC_URL", DEFAULT_URL),
        local_path=os.getenv("TITANIC_LOCAL_PATH", ""),
        n_labeled=int(os.getenv("TITANIC_N_LABELED", "891")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        cv_folds=int(os.getenv("CV_FOLDS", "5")),
    )
    if settings.n_labeled <= 0:
        raise ValueError("TITANIC_N_LABELED debe ser mayor que cero.")
    if not 0 < settings.test_size < 1:
        raise ValueError("TEST_SIZE debe estar entre 0 y 1.")
    if settings.cv_folds < 2:
        raise ValueError("CV_FOLDS debe ser al menos 2.")
    return settings


settings = load_settings()
