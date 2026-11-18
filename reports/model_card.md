# Ficha del modelo: Titanic v1

## Propósito y uso previsto

Clasificador experimental para estimar la supervivencia histórica de pasajeros
del Titanic a partir de características disponibles al embarcar. Su propósito
es pedagógico y analítico: revisar preprocesamiento, métricas y limitaciones.
No está aprobado para decisiones de seguros, evacuación, acceso a servicios ni
ninguna decisión actual sobre personas.

## Datos y entrenamiento

- Fuente: CSV Titanic indicada por `TITANIC_URL` (por defecto, la fuente de
  `.env.example`); ingesta del 17 de noviembre de 2026.
- Datos etiquetados: 891 filas. División estratificada 80/20 con semilla 42:
  712 para desarrollo y 179 para evaluación final. Después de esa evaluación,
  el artefacto se ajustó con las 891 filas etiquetadas.
- Etiqueta: `Survived` (0/1). Los 418 pasajeros sin etiqueta se usaron solo en
  la prueba de humo, no en entrenamiento ni cálculo de métricas.
- SHA-256 de Silver utilizado: `1169b3fa24d4b86ae51e4b7402d10c24d464c234da3560e691dd56bf02073a0e`.
- Tipo: HistGradientBoosting; parámetros del estimador por defecto; umbral 0.5.
- Variables numéricas: edad original (imputada dentro del pipeline), hermanos/
  cónyuge, padres/hijos, tarifa y tamaño familiar.
- Variables categóricas: clase, sexo, puerto, título y si viaja solo. Se usa
  one-hot. Cabina se excluyó del modelo final por riesgo de sesgo de registro;
  en CV su adición a la regresión logística cambió ROC-AUC de 0.858 a 0.863,
  diferencia pequeña frente a la variabilidad entre pliegues.
- Objetivo, identificadores, nombre libre, ticket y codificaciones ordinales
  se excluyen para reducir fuga, memorización o un orden artificial.

## Desempeño

| Evaluación | Resultado |
|---|---:|
| ROC-AUC CV estratificada (entrenamiento) | 0.887 ± 0.019 |
| ROC-AUC test (n=179) | 0.823 |
| IC bootstrap 95 % ROC-AUC | 0.748–0.893 |
| Exactitud / precisión / recall / F1 test | 0.810 / 0.769 / 0.725 / 0.746 |
| Regla de género: exactitud / F1 test | 0.777 / 0.692 |
| Mayoría: exactitud test | 0.615 |

La brecha promedio train-validación de ROC-AUC en CV fue 0.107 para
HistGradientBoosting; la regresión logística fue 0.858 ± 0.024 con brecha
0.006. Una búsqueda de hiperparámetros no mejoró de forma convincente la
estimación anidada: diferencia pareada −0.0016 ± 0.0078, por lo que se conservó
el estimador por defecto. El intervalo de test es ancho y no permite concluir
que diferencias pequeñas entre modelos sean reales.

Las predicciones OOF de entrenamiento muestran diferencias entre subgrupos:
recall 0.888 en mujeres (n=253) y 0.353 en hombres (n=459); 0.820 en primera
clase (n=171), 0.881 en segunda (n=150) y 0.495 en tercera (n=391). Son
diagnósticos sobre datos históricos, no garantías de equidad ni métricas
independientes por subgrupo.

## Limitaciones, riesgos y reproducción

Un único desastre histórico no representa otros contextos. Hay faltantes,
sesgos de registro de cabina, categorías pequeñas y variables sensibles (`Sex`,
`Pclass`). Predicción no implica causalidad: no usar el modelo para estimar el
efecto de cambiar una política ni para tomar decisiones sobre individuos.

Reproducir con `python -m pip install -e ".[dev]"`, `make test` y `make pipeline`.
El artefacto y su metadata registran huellas y versiones; un archivo Joblib solo
debe cargarse si proviene de una fuente confiable.

Entorno de esta ejecución: Python 3.14.4, pandas 3.1.1, scikit-learn 1.9.1.
La ficha completa y las métricas se guardan junto al modelo en
`models/titanic_model_v1.metadata.json` (artefacto generado, no versionado).

El umbral 0.5 es una referencia convencional, no un costo optimizado: no se
definió una operación de despliegue ni una función de costos aprobada.
