"""Pipeline de predicción: consumo de Frio (kW) del día siguiente.

Uso:
    python src/predict.py "data/Totalizadores Planta de Cerveza 2024_2025.xlsx"
    python src/predict.py archivo.xlsx --version 1.0.0 --output results/predicciones.csv

Salida: CSV con una fila por día. 'fecha' es el día D y
'prediccion_frio_kw' es el consumo predicho para el día D+1.

Contrato con el archivo de entrada: las hojas y columnas se buscan por
NOMBRE (no por posición). Puede haber hojas y columnas de más; si falta
alguna de las que el modelo necesita, el script termina con un mensaje que
dice exactamente qué falta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auxiliar_functions import MODELS_DIR, RESULTS_DIR, ROOT, DataContractError  # noqa: E402
from preprocessing_pipeline import build_features, load_daily_for_prediction  # noqa: E402

REGISTRY_PATH = MODELS_DIR / "model_registry.json"


def _semver(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def load_model(version: str = "latest"):
    """Carga el modelo desde el registro. 'latest' = versión semántica más alta.
    Devuelve (modelo, entrada_del_registro)."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"No existe {REGISTRY_PATH.relative_to(ROOT)}. "
                                "Ejecutar antes: python src/train_model.py")
    registry = json.loads(REGISTRY_PATH.read_text())
    if not registry:
        raise ValueError("El model registry está vacío.")
    if version == "latest":
        version = max(registry, key=_semver)
    if version not in registry:
        raise KeyError(f"La versión '{version}' no está en el registro. "
                       f"Disponibles: {sorted(registry, key=_semver)}")
    entry = registry[version]
    model_path = ROOT / entry["path"]
    if not model_path.exists():
        raise FileNotFoundError(f"El registro apunta a {entry['path']} pero el archivo no existe.")
    return joblib.load(model_path), entry


def load_and_preprocess(filepath, features: list[str]):
    """Carga el Excel y aplica el MISMO preprocesamiento del entrenamiento
    (funciones de preprocessing_pipeline.py, no una reimplementación).
    Devuelve la matriz de características y las fechas."""
    daily = load_daily_for_prediction(filepath, features)
    X = build_features(daily)
    X = X[X["frio_hoy"].notna()] if "frio_hoy" in X else X
    return X[features], X.index


def predict_consumption(filepath, version: str = "latest") -> pd.DataFrame:
    """Función principal de predicción."""
    model, entry = load_model(version)
    X, fechas = load_and_preprocess(filepath, entry["features"])
    if len(X) == 0:
        raise DataContractError("El archivo no tiene ningún día válido para predecir.")
    predictions = model.predict(X)
    # La predicción es DIARIA: una fila por día, no por hora.
    # 'fecha' es el día D; la predicción corresponde al consumo del día D+1.
    return pd.DataFrame({
        "fecha": pd.to_datetime(fechas).date,
        "prediccion_frio_kw": predictions.round(1),
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Predice Frio (kW) del día siguiente.")
    ap.add_argument("archivo", help="Excel con la misma estructura que los de entrenamiento")
    ap.add_argument("--version", default="latest", help="versión del modelo (default: latest)")
    ap.add_argument("--output", default=str(RESULTS_DIR / "predicciones.csv"))
    args = ap.parse_args(argv)
    try:
        results = predict_consumption(args.archivo, args.version)
    except DataContractError as e:
        print(f"ERROR en el archivo de entrada: {e}", file=sys.stderr)
        return 2
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"Predicciones generadas en {out}  ({len(results)} días)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
