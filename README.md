# Proyecto Titanic: análisis y predicción reproducibles

Pipeline de ciencia de datos que prepara los datos del Titanic en capas Bronze,
Silver y Gold, genera tablas analíticas y entrena un clasificador reproducible.
Las transformaciones y el preprocesamiento del modelo están cubiertos por pruebas.

## Requisitos

- Python 3.11 o posterior
- `make`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

En Windows/PowerShell, activa el entorno con `.venv\Scripts\Activate.ps1`.

## Ejecutar

```bash
make test
make pipeline
```

El pipeline descarga la fuente configurada en `.env` (por defecto, una copia CSV
del Titanic), separa las primeras 891 filas etiquetadas y produce:

- `data/bronze/`: copias CSV originales y huellas SHA-256 de ingesta.
- `data/silver/`: datos enriquecidos en Parquet y vista numérica final.
- `data/gold/`: tablas de supervivencia, grupos y atípicos.
- `models/`: modelo final, metadatos de auditoría y predicciones de humo.

Los artefactos generados y la configuración local no se versionan. Para usar
una copia local, define `TITANIC_LOCAL_PATH` en `.env`.

## Análisis

Los cuadernos [02_exploracion.ipynb](notebooks/02_exploracion.ipynb) y
[04_modelado.ipynb](notebooks/04_modelado.ipynb) contienen análisis reproducibles.
Las conclusiones y limitaciones están en `reports/`.

El modelo describe patrones de un único desastre histórico. No establece
causalidad y no debe usarse para tomar decisiones actuales sobre personas.