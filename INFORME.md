# Informe — Predicción del consumo de Frío del día siguiente

## 1. Datos recibidos
Tres archivos horarios solapados (`2020_2022`, `2021_2023`, `2022_2023`) que, unificados, cubren **2020-07-01 → 2023-10-26**
(1.213 días de calendario, 1.165 válidos). Los nombres y años no coinciden con los del enunciado; el código busca
`Totalizadores*Planta*.xlsx` y funciona con ambos.

## 2. Decisiones principales

| Decisión | Justificación (número medido) |
|---|---|
| **Total diario = lectura de las 23:59** | El 23:00 pierde la última hora: 926 kW de media (4 % del día). La documentación define el 23:59 como total. Se usa igual en features, target, baselines y `predict.py`. |
| **Si falta la 23:59, se estima** desde las 23:00 con el ritmo de las últimas 3 horas | Validado en 1.129 días sanos fingiendo que falta: MAE 155 kW, contra 949 kW de usar las 23:00 (6 h: 157 kW; día completo: 171 kW). |
| **Frio = suma de incrementos horarios corregida** | Un salto (incremento < 0 o > 3.000 kW, incluidos los del millón) se reemplaza por la mediana horaria del día. Validado inyectando saltos en 1.129 días sanos (MAE 163 kW, sesgo −10 kW) y contra 8 días reales que tienen su valor sano en otro archivo (MAE 283 kW). Con más de 2 saltos el día se descarta. |
| Días sin reinicio a las 00:00 se **conservan** | La suma de incrementos descuenta el arrastre del día anterior. |
| **Jornadas apiladas: se descartan** (la reasignación queda como opción) | Ver 2.1. |
| Desempate entre archivos: día válido > archivo más reciente | 498 fechas en más de un archivo; de 493 comparables, 488 idénticas. Las 5 distintas son errores del archivo más viejo. |
| `shift(-1)` sobre calendario completo | Se reindexa a todos los días antes del shift; `build_target` falla si el índice tiene huecos. |
| Selección ampliada a **todas las columnas de *Consolidado EE*** | Para cada una: total del día, consumo de las últimas 6 h y media de 7 días. Filtro sin target (≤ 5 % faltantes, > 10 valores distintos) y luego LASSO, Random Forest y RFE solo con train; entra lo elegido por ≥ 2 de 3. Quedan 11 variables, **todas de una sola hoja**. |
| Escalado e imputación dentro del `Pipeline` | Ajustados solo con train. `dataset_final.csv` se guarda sin escalar. |
| Modelo elegido solo por MAE de validación | Lasso (α = 300): 2.103 kW en validación, prácticamente empatado con Random Forest (2.104) y Ridge (2.112). |

### 2.1 Jornadas apiladas: por qué no se reasignan
Las fechas faltantes de cada archivo coinciden con la cantidad de bloques sobrantes, y una regla del EDA los reasigna
(30/MM → 31/(MM-1); fin de febrero → 31/12, 30/08 y 29/08 del año anterior). La probamos de dos formas:

- **Continuidad.** Para cada bloque comparamos su total con la media de los días vecinos de la fecha destino y de la fecha original.
  La mediana del error es 4.983 kW contra el destino y 2.154 kW contra el origen, igual a la de un día normal (2.202 kW). El destino
  gana solo en 6 de 21 casos. El ejemplo más claro: bloques de ~21.000 kW asignados al 31/12, que es una parada de fin de año con
  vecinos de ~3.000 kW.
- **Modelo.** Sobre las mismas 174 fechas de validación, reasignar da 2.080 kW contra 2.103 kW. Son 23 kW, del orden del error estándar
  de una comparación pareada (~32 kW).

Una mejora no significativa no justifica cargar datos que la validación por continuidad dice que están en la fecha equivocada: un 31/12
con 21.000 kW contamina los lags y el target de los días de parada. La regla queda implementada en forma general (sin fechas escritas a
mano) y se activa con `REASIGNAR_APILADAS = True`.

## 3. Conclusiones

### 3.1 ¿Su modelo agrega valor?

| Mismo conjunto de test (2023-04-26 → 2023-10-24, 175 días) | MAE (kW) | RMSE | R² |
|---|---|---|---|
| Persistencia (ŷ[D+1] = y[D]) | 3.357 | 4.323 | 0,42 |
| Media móvil 7 días | 3.048 | 3.976 | 0,51 |
| **Lasso v1.0.0** | **2.724** | **3.572** | **0,60** |

La mejora sobre el mejor baseline es **10,6 %**. En validación es de 18 % (2.103 vs 2.571 kW). Lo que se agrega es un `Pipeline` con un
modelo lineal que entrena en milisegundos y usa 8 coeficientes distintos de cero. El costo grande del proyecto (leer, validar, limpiar y
recuperar el Excel) lo pagan también los baselines: sin esa limpieza la persistencia usaría totales de millones de kW. **Sí lo
compraríamos.** El margen contra 3.000 kW es de ~280 kW: la próxima mejora tiene que venir de información nueva sobre D+1 (clima).

### 3.2 ¿Qué variables se parecen demasiado al objetivo?
- La **suma de los compresores Mycom 1-7** (*Totalizadores Energia*) correlaciona 0,98-0,99 con `Frio` del mismo día en 2020, 2022 y
  2023 (0,96 en 2021): es el mismo sistema medido desde otro tablero. **No entra**: predice D+1 igual o peor que `Frio` de hoy
  (r = 0,83 vs 0,85) y exigiría 7 columnas más.
- **Sala de Máquinas** contiene al frío. Su correlación de Pearson con Frio se ve baja por un valor extremo, pero en rango es ~0,98.
  Entra su consumo de las últimas 6 horas (`sala_maq_ult6h`), con peso chico.

Nuestro caso es el **legítimo**: todo lo que entra es del día D o anterior.
- El único lugar donde aparece D+1 es `build_target`, con `shift(-1)` sobre un calendario sin huecos (si hay huecos, falla).
- Las features usan `shift(k ≥ 0)` o ventanas hacia atrás.
- Las "últimas 6 horas" son de 18:00 a 23:59 **del día D**.
- Lo único que se mira de D+1 es su fecha (día de la semana y mes), que se conoce de antemano.

### 3.3 Un resultado que contradice la intuición
El enunciado afirma un patrón semanal. **Medido, no aparece:**
- sobre el nivel, el día de la semana explica el **0,5 %** de la varianza (ANOVA p = 0,42; Kruskal-Wallis p = 0,47);
- quitando la tendencia de 28 días, el efecto es detectable pero chico (eta² = 0,024), sin corte laborable/fin de semana;
- agregar el día de la semana al modelo **empeora** el MAE de validación; el indicador de fin de semana no lo cambia.

El frío es una carga base 7×24: el perfil horario medio varía 3,9 %. Envasado, que sí trabaja por turnos, no se mueve con el frío
(r = 0,00). La estacionalidad anual (julio = 2,4 × diciembre; el mes explica el 59 % de la varianza) domina todo. **Qué hicimos:** construimos las
variables por trazabilidad y las descartamos como candidatas.

### 3.4 Umbral para reemplazar un modelo desplegado: **300 kW de MAE**
- El error estándar de la diferencia de MAE entre el modelo y el mejor baseline, medido sobre ~6 meses de validación con bootstrap por
  bloques semanales (los errores de días consecutivos no son independientes), es de **156 kW**: dos errores estándar dan ~300 kW.
  Esa diferencia hoy es de 468 kW (IC 95 % [158; 776]): el modelo sí supera el umbral contra la media móvil.
- El desvío del MAE entre folds de `TimeSeriesSplit` es de 1.131 kW: dos modelos solo se comparan **sobre el mismo período**,
  nunca contra un número reportado en otro tramo.
- 300 kW es ~10 % del MAE actual, del mismo orden que el margen que piden para justificar complejidad.

Por debajo de 300 kW no compensa redeplegar: hay que re-certificar el contrato de datos y el pipeline. Las cinco mejores
configuraciones de validación están a menos de 10 kW entre sí: ninguna justificaría reemplazar a otra.

### 3.5 Lo que no usamos
- **PCA.** Con 11 variables, la multicolinealidad la resuelve Lasso. Las componentes mezclan todas las columnas originales, así que el
  archivo de test tendría que traerlas todas y el contrato de `predict.py` crecería. Además se pierde la lectura de los coeficientes.
- **MICE.** Los faltantes no son celdas sueltas al azar: son días enteros inválidos o columnas que no existen en un archivo entero.
  MICE imputaría con variables que también faltan en esos casos. Sobre las features elegidas casi no quedan faltantes, y la mediana de
  train dentro del `Pipeline` alcanza sin riesgo de fuga.

## 4. Limitaciones
- **Sin clima.** La temperatura de Tecate (Open-Meteo) es la mejora más prometedora, pero haría que `predict.py` dependa de internet.
- **Test en verano.** El test interno cae en la época de más consumo y más error. Para un año completo, como 2024-2025, esperamos un
  MAE menor (en validación, otoño-invierno, fue 2.103 kW). El margen respecto de 3.000 no es grande.
- **Distancia temporal.** Entrenado hasta abril de 2023 y aplicado a 2024-2025: un cambio de equipos o de régimen no está cubierto.
- **Arranque del archivo de test.** Los primeros días no tienen historia: los lags caen al último valor conocido y las medias usan los
  días disponibles.
- **Columnas sin peso.** Lasso deja en 0 tres de las 11 variables (`frio_delta1`, `frio_x_cos`, `pta_agua_eflu_rm7`). Por esta última,
  el contrato todavía exige `Pta Agua / Eflu (Kw)`.
- **Días sin dato.** 48 de los 1.213 días del calendario quedan sin dato válido, sobre todo por fechas apiladas o faltantes. Si la
  planta confirmara a qué fecha corresponde cada bloque apilado, se recuperarían.
- **Días anómalos imprevisibles.** Paradas y arranques que nada en el día D anticipa generan los errores más grandes.
