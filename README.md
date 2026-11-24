# Predicción del consumo de Frío (kW) del día siguiente — Planta cervecera

Pipeline de machine learning reproducible que predice el consumo eléctrico diario del sistema de
refrigeración (`Frio (Kw)`, hoja *Consolidado EE*) del día **D+1** a partir de los totalizadores horarios del día **D**.

| | MAE test (kW) |
|---|---|
| Baseline persistencia | 3.357 |
| Baseline media móvil 7 días | 3.048 |
| **Modelo v1.0.0 (Lasso)** | **2.724** |

Test: 175 días, del 2023-04-26 al 2023-10-24. Mejora de 10,6 % sobre el mejor baseline.

Las decisiones y su justificación están en [`INFORME.md`](INFORME.md); el detalle, en los notebooks.

---

## 1. Clonar el repositorio
```bash
git clone <URL-DEL-REPO> tp_final
cd tp_final
```

## 2. Crear el entorno (venv + uv, Python 3.12)
Desde la carpeta `tp_final`, crear un entorno virtual aislado con Python 3.12:
```bash
py -3.12 -m venv cervecera_env
```
(En Mac o Linux: `python3.12 -m venv cervecera_env`.)

Activarlo:
```bash
# Windows, PowerShell
cervecera_env\Scripts\Activate.ps1
# Windows, Git Bash
source cervecera_env/Scripts/activate
# Mac o Linux
source cervecera_env/bin/activate
```

Instalar las dependencias con uv:
```bash
pip install uv
uv pip install -r requirements.txt
```

La carpeta `cervecera_env/` está en `.gitignore`: el entorno no se sube al repositorio, se recrea con estos pasos.

Alternativa con conda, la que sugiere la consigna:
```bash
conda create -n cervecera_env python=3.12 -y
conda activate cervecera_env
pip install uv
uv pip install -r requirements.txt
```

## 3. Datos
Los Excel **no están en el repositorio** (pesan ~30 MB cada uno y están en `.gitignore`). Se obtienen del aula virtual
de la materia y se copian dentro de `data/` **sin renombrarlos**:

```
data/Totalizadores Planta 2020_2022.xlsx
data/Totalizadores Planta - 2021_2023.xlsx
data/Totalizadores Planta 2022_2023.xlsx
```

El código busca `data/Totalizadores*Planta*.xlsx`, así que también funciona con los nombres
`Totalizadores Planta de Cerveza ….xlsx`. Un archivo cuyo nombre contenga `2024_2025` **nunca** se usa para entrenar.

Verificar que son exactamente los mismos archivos que usamos nosotros:
```bash
python src/auxiliar_functions.py      # compara contra data/checksums.json (SHA-256)
```
(Las líneas de `data/interim/…` y `data/processed/…` dan OK después de correr el paso 4.)

## 4. Ejecutar el pipeline completo
```bash
python src/preprocessing_pipeline.py   # ~30 s -> data/processed/dataset_final.csv, data_lineage.json, checksums
python src/train_model.py              # ~1 min -> models/modelo_v1.0.0.pkl, model_registry.json, results/experiment_logs.csv
```
Se usa `SEMILLA = 42` en toda partición y estimador: dos ejecuciones dan los mismos números.

## 5. Predecir sobre un archivo nuevo
```bash
python src/predict.py "data/Totalizadores Planta de Cerveza 2024_2025.xlsx"
# opciones: --version 1.0.0  --output results/predicciones.csv   (por defecto: la versión más alta del registro)
```
Salida: `results/predicciones.csv`, una fila por día. `fecha` es el día D y `prediccion_frio_kw` es el consumo
predicho para D+1.

**Contrato del archivo de entrada.** Hojas y columnas se buscan por nombre (sin importar orden, mayúsculas ni espacios
de más). Puede tener hojas y columnas extra. El modelo v1.0.0 necesita **una sola hoja**:

| Hoja | Columnas |
|---|---|
| Consolidado EE | `DIA`, `HORA`, `Frio (Kw)`, `Envasado (Kw)`, `Sala Maq (Kw)`, `Pta Agua / Eflu (Kw)`, `KW Gral Planta` |

Si falta algo, el script termina con código 2 y un mensaje como:
```
ERROR en el archivo de entrada: El archivo 'x.xlsx' no cumple el contrato de entrada. Falta:
  - columna 'Sala Maq (Kw)' en la hoja 'Consolidado EE'
```

## 6. Notebooks
```bash
jupyter notebook notebooks/
```
Orden: `eda.ipynb` → `preprocesamiento.ipynb` → `modelado.ipynb` (este último lee los artefactos del paso 4).

## Estructura
```
tp_final/
├── data/                       Excel (no versionados), checksums.json
│   └── processed/              dataset_final.csv, data_lineage.json, selected_features.json
├── models/                     modelo_v1.0.0.pkl, model_registry.json
├── notebooks/                  eda, preprocesamiento, modelado
├── src/
│   ├── auxiliar_functions.py   lectura robusta del Excel, total diario, calidad, huella()
│   ├── preprocessing_pipeline.py  unificación, limpieza, features, target, selección
│   ├── train_model.py          baselines, búsqueda, tracking, modelo final, registry
│   └── predict.py              predicción sobre un Excel nuevo
├── results/                    experiment_logs.csv, predicciones*.csv
├── reports/figures/            gráficos de los notebooks
├── INFORME.md                  decisiones, limitaciones y conclusiones
├── GIT_WORKFLOW.md             ramas, PRs y tag v1.0.0
├── consigna.md
├── requirements.txt
└── .gitignore
```

## Versionado del modelo
El modelo se guarda como `models/modelo_vX.Y.Z.pkl` y queda registrado en `models/model_registry.json` con sus
hiperparámetros, features, columnas requeridas, métricas, SHA-256 y el **hash del commit de Git**. El commit
correspondiente se etiqueta con `git tag -a vX.Y.Z` (ver `GIT_WORKFLOW.md`). `predict.py` carga por defecto la versión
semántica más alta del registro.
