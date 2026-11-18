# Formulación y evaluación del modelo

## Problema y criterio previo

| Decisión | Definición |
|---|---|
| Tipo de problema | Clasificación binaria supervisada |
| Objetivo | `Survived` (1 = sobrevivió) |
| Unidad y momento | Un pasajero, usando solo variables disponibles al embarcar |
| Métrica principal | ROC-AUC: mide ordenamiento de probabilidades sin fijar umbral |
| Métricas secundarias | Exactitud, precisión, recall y F1; la exactitud sola oculta el desbalance |
| Costo de errores | En un protocolo de evacuación, un falso negativo sería especialmente grave; en seguros ambos errores pueden distorsionar el riesgo y afectar a grupos |
| Criterio de éxito | Superar el baseline de mayoría y la regla de género en ROC-AUC de validación cruzada, con estabilidad comparable |

Las métricas no convierten una predicción histórica en una regla de evacuación o
de suscripción. `Sex` y `Pclass` se usan para investigar patrones históricos,
no para recomendar decisiones actuales.

## Auditoría de variables

- `Survived` y `Survivor` se excluyen: contienen directamente la respuesta.
- `PassengerId`, `Ticket` y `Name` se excluyen como identificadores; solo se
  conserva el título extraído (`TTitulo`).
- Se usa `AgeOriginal`, con sus ausencias, no la edad imputada con estadísticas
  calculadas sobre todo Silver. La imputación del modelo se ajusta dentro del
  pipeline y solo sobre los datos de entrenamiento de cada partición.
- Las columnas categóricas originales se codifican con one-hot, no con códigos
  ordinales `N*`.
- `Person` se excluye por redundancia con edad y sexo.
- Las variables de cabina se mantienen fuera del modelo final por su posible
  sesgo de registro; la duda debe evaluarse por ablación antes de incorporarlas.

## Baselines y selección

El baseline de mayoría predice siempre la clase más frecuente. La regla de género
predice supervivencia para `female` y no supervivencia para `male`. Las tablas de
validación cruzada de los candidatos y sus brechas de entrenamiento se generan
en el cuaderno de modelado.

En el entrenamiento, el baseline de mayoría obtuvo 61.7 % de exactitud CV y la
regla de género obtuvo 78.9 % de exactitud, 71.5 % de F1 y ROC-AUC 0.770. La
comparación debe priorizar ROC-AUC y estabilidad, no solo exactitud.

HistGradientBoosting fue el mejor candidato en CV (ROC-AUC 0.887 ± 0.019), pero
presentó una brecha train-validación (0.107). La regresión logística obtuvo
0.858 ± 0.024 con una brecha mucho menor (0.006), por lo que ambos resultados
son los finalistas: HistGradientBoosting prioriza ROC-AUC; logística aporta
estabilidad e interpretabilidad. La búsqueda de hiperparámetros tuvo una
diferencia pareada anidada de −0.0016 ± 0.0078 frente al estimador por defecto:
no hay evidencia de una mejora y se conserva la configuración predeterminada
de HistGradientBoosting (`FINAL_PARAMS = {}`), evitando complejidad sin respaldo.
La evaluación final usa una división estratificada y su conjunto de prueba se
abre una sola vez en `run()`.

## Resultado y límites

En el holdout de 179 pasajeros, el modelo final obtuvo exactitud 0.810,
precisión 0.769, recall 0.725, F1 0.746 y ROC-AUC 0.823 (IC bootstrap 95 %:
0.748–0.893). Como referencia en la misma partición, la regla de género tuvo
exactitud 0.777 y F1 0.692; el baseline de mayoría tuvo exactitud 0.615. El
intervalo amplio refleja la incertidumbre de una muestra pequeña; no se usa
para volver a seleccionar el modelo.

`make pipeline` registra en `models/titanic_model_v1.metadata.json` las métricas
de prueba, el intervalo bootstrap de ROC-AUC, la huella de Silver y las versiones
del entorno. Se debe interpretar ese resultado considerando el tamaño reducido
del conjunto, un solo naufragio histórico, datos ausentes y la sensibilidad de
sexo y clase. No se debe usar este modelo para decisiones de elegibilidad,
precio, evacuación ni trato de personas.

El umbral 0.5 se mantiene como referencia convencional porque no hay una función
de costos aprobada para el uso hipotético. El cuaderno informa el umbral que
maximiza F1 en predicciones OOF, pero no se interpreta como una decisión de
política. Aunque un falso negativo sería grave en una evacuación, no hay una
operación actual definida para traducir ese costo a un umbral; cualquier uso
real requeriría decidirlo con datos de validación y una revisión de impacto.
