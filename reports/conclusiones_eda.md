# Informe EDA: supervivencia del Titanic

## Hallazgos principales

El análisis usa los **891 pasajeros etiquetados** de la fuente configurada. La
supervivencia observada fue **38.4 % (342/891)**. Son asociaciones dentro de un
solo naufragio, no estimaciones causales ni riesgos aplicables a otros viajes.

1. **Sexo y supervivencia:** sobrevivió el **74.2 % de las mujeres (233/314)** y
   el **18.9 % de los hombres (109/577)**. La prueba de independencia dio
   χ²(1) = 260.72, p = 1.20 × 10⁻⁵⁸. La diferencia también aparece dentro de
   cada clase: tasas mujer/hombre de 96.8/36.9 % en primera, 92.1/15.7 % en
   segunda y 50.0/13.5 % en tercera. La asociación es marcada, pero no prueba
   que el sexo por sí solo causara el resultado.
2. **Clase del tiquete:** sobrevivió el **63.0 % en primera (136/216)**, el
   **47.3 % en segunda (87/184)** y el **24.2 % en tercera (119/491)**. Clase
   también resume diferencias de acceso, ubicación y condiciones de evacuación;
   no debe interpretarse como efecto aislado.
3. **Tipo de persona:** la tasa observada fue 59.0 % para `child` (49/83),
   75.7 % para `female` (205/271) y 16.4 % para `male` (88/537). `child`
   significa edad conocida menor de 16; las edades faltantes no se clasifican
   como niños.
4. **Composición familiar:** sobrevivió el 50.6 % de quienes viajaban con
   familia (179/354), frente al 30.4 % de quienes viajaban solos (163/537).
   Los grupos familiares grandes son pequeños: por ejemplo, Familia=7 tiene
   n=6 y Familia=10 n=7, así que sus tasas son inestables.
5. **Puerto y confusión por clase:** la tasa sin estratificar fue 55.4 % en
   Cherbourg (93/168), 39.0 % en Queenstown (30/77) y 33.9 % en Southampton
   (219/646). Pero la composición varía: Cherbourg tenía 50.6 % de pasajeros
   de primera clase; Southampton, 54.6 % de tercera; Queenstown, 93.5 % de
   tercera. Al comparar dentro de clase, las tasas de Queenstown se basan en
   solo 2 pasajeros de primera y 3 de segunda. Por tanto, el contraste global
   entre puertos está confundido por la mezcla de clases y tiene incertidumbre
   importante en grupos pequeños.

## Calidad y decisiones de limpieza

- En las 891 filas faltaban **177 edades (19.9 %)**, **687 cabinas (77.1 %)** y
  **2 puertos (0.2 %)**. La edad de Silver se imputó con el promedio por sexo;
  la desviación estándar bajó de 14.53 a 13.01 años. El promedio reduce la
  dispersión y genera una forma de distribución artificialmente concentrada.
  `AgeOriginal` conserva los faltantes y `AgeImputada` los señala.
- Los dos puertos faltantes se rellenaron con la moda (`S`). Esto asigna a esas
  personas el perfil de la categoría más común, no una procedencia conocida.
- La cabina ausente se representa como cubierta `X`, y se conserva `HasCabin`.
  La ausencia no se interpreta como una cubierta real.
- `Master` se agrupa con `Mr` para reducir categorías escasas; `Person` conserva
  por separado la regla de niño basada en edad conocida. Esa columna compensa,
  solo parcialmente, la mezcla de niños con hombres adultos.
- `Person` usa el sexo cuando `Age` es nula. La regla es reproducible, pero
  subcuenta potencialmente a niños con edad no registrada.
- Las tarifas están sesgadas a la derecha (media 32.20, mediana 14.45,
  asimetría 4.79). Los 116 atípicos IQR de tarifa (13.0 %) tuvieron una tasa de
  supervivencia de 68.1 %, frente a 33.9 % en los demás. Esto describe grupos
  distintos —incluida la clase—, no un efecto causal de pagar más.

## Asociaciones, límites y uso

La matriz de correlación usa códigos ordinales para categorías y por eso es
solo una aproximación descriptiva; el modelado usa one-hot. Las tasas de puerto
se contrastaron dentro de clase para hacer visible la confusión, sin eliminar
la incertidumbre de las celdas pequeñas. Chi-cuadrado indica asociación entre
sexo y supervivencia, pero no su tamaño causal.

Este conjunto representa un único evento de 1912 y contiene datos incompletos
y sesgos de registro. No permite afirmar qué habría ocurrido bajo otro
protocolo de evacuación, que una característica causó la supervivencia, ni que
los patrones se generalicen a otros desastres. No debe usarse para asignar
precios de seguro ni decidir a quién evacuar o brindar un servicio.
