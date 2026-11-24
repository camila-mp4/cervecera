"""Entrenamiento, comparación, tracking de experimentos y registro del modelo.

Uso:
    python src/train_model.py                 # versión 1.0.0
    python src/train_model.py --version 1.1.0

Protocolo (sin fuga de datos):
  * Toda configuración se ajusta en TRAIN y se evalúa en VALIDATION.
  * El mejor modelo se elige por MAE de validación (nunca mirando test).
  * Se reentrena con TRAIN+VALIDATION y recién ahí se evalúa TEST, una vez.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auxiliar_functions import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ROOT, SEMILLA, huella  # noqa: E402
from preprocessing_pipeline import (  # noqa: E402
    TARGET_Y, DeltaRegressor, make_preprocessor, required_raw,
)

LOG_PATH = RESULTS_DIR / "experiment_logs.csv"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# --------------------------------------------------------------------------
# Espacio de búsqueda (documentado en modelado.ipynb)
# --------------------------------------------------------------------------
SEARCH_SPACE = {
    "Ridge": (Ridge, {"alpha": [0.1, 1, 3, 10, 30, 100]}),
    "Lasso": (Lasso, {"alpha": [1, 10, 30, 100, 300], "max_iter": [50000]}),
    "RandomForest": (RandomForestRegressor, {
        "n_estimators": [400], "max_depth": [None, 8], "min_samples_leaf": [3, 10],
        "max_features": [1.0, 0.5], "random_state": [SEMILLA], "n_jobs": [-1]}),
    "XGBoost": (XGBRegressor, {
        "n_estimators": [300, 600], "learning_rate": [0.03], "max_depth": [2, 4],
        "subsample": [0.8], "colsample_bytree": [0.8], "min_child_weight": [5],
        "random_state": [SEMILLA], "n_jobs": [1]}),
    "LightGBM": (LGBMRegressor, {
        "n_estimators": [300, 600], "learning_rate": [0.03], "num_leaves": [7, 15],
        "min_child_samples": [20], "subsample": [0.8], "subsample_freq": [1],
        "colsample_bytree": [0.8], "random_state": [SEMILLA], "verbose": [-1], "n_jobs": [1]}),
}
# Cada configuración se prueba prediciendo el nivel (y) y el cambio (y - frio_hoy).
OBJETIVOS = ["nivel", "delta"]


def metrics(y, p) -> dict:
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def make_model(name: str, params: dict, objetivo: str):
    est = SEARCH_SPACE[name][0](**params)
    pipe = Pipeline([("pre", make_preprocessor()), ("model", est)])
    return DeltaRegressor(pipe) if objetivo == "delta" else pipe


def baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Persistencia: y_hat[D+1] = y[D]. Media móvil: promedio de D-6..D.
    Se calculan con las MISMAS columnas del dataset final que ve el modelo."""
    return pd.DataFrame({"persistencia": df["frio_hoy"], "media_movil_7d": df["frio_rm7"]},
                        index=df.index)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "sin-repositorio-git"


def log_experiment(row: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


def cv_on_train(model, X, y, n_splits: int = 5) -> list[float]:
    """MAE por fold con TimeSeriesSplit sobre train (para medir dispersión)."""
    out = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        out.append(float(mean_absolute_error(y.iloc[te], m.predict(X.iloc[te]))))
    return out


def main(version: str = "1.0.0") -> dict:
    ds = pd.read_csv(PROCESSED_DIR / "dataset_final.csv", index_col="fecha", parse_dates=True)
    feats = json.loads((PROCESSED_DIR / "selected_features.json").read_text())["seleccionadas"]
    tr, va, te = (ds[ds["split"] == s] for s in ("train", "val", "test"))
    Xtr, ytr, Xva, yva = tr[feats], tr[TARGET_Y], va[feats], va[TARGET_Y]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"Train {Xtr.shape}  Val {Xva.shape}  Test {te[feats].shape}")

    # ---------------- baselines en validación ----------------
    rows = []
    for b, pred in baselines(va).items():
        m = metrics(yva, pred)
        rows.append({"modelo": b, "objetivo": "-", "params": "{}", **{f"val_{k}": v for k, v in m.items()}})
        log_experiment({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "run_id": run_id, "modelo": b, "objetivo": "-", "params": "{}",
                        "n_features": 0, "split_eval": "val", **m, "fit_seconds": 0.0})

    # ---------------- búsqueda: train -> val ----------------
    for name, (_, grid) in SEARCH_SPACE.items():
        keys = list(grid)
        for values in itertools.product(*grid.values()):
            params = dict(zip(keys, values))
            for obj in OBJETIVOS:
                t0 = time.perf_counter()
                model = make_model(name, params, obj).fit(Xtr, ytr)
                secs = time.perf_counter() - t0
                m = metrics(yva, model.predict(Xva))
                shown = {k: v for k, v in params.items() if k not in ("random_state", "n_jobs", "verbose")}
                rows.append({"modelo": name, "objetivo": obj, "params": json.dumps(shown),
                             **{f"val_{k}": v for k, v in m.items()}})
                log_experiment({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "run_id": run_id, "modelo": name, "objetivo": obj,
                                "params": json.dumps(shown), "n_features": len(feats),
                                "split_eval": "val", **m, "fit_seconds": round(secs, 3)})
    res = pd.DataFrame(rows).sort_values("val_MAE").reset_index(drop=True)
    res.to_csv(RESULTS_DIR / "comparacion_validacion.csv", index=False)
    print("\nTop 10 en validación:\n", res.head(10).to_string(index=False))

    # ---------------- selección (solo validación) ----------------
    best = res[~res["modelo"].isin(["persistencia", "media_movil_7d"])].iloc[0]
    name, obj = best["modelo"], best["objetivo"]
    params = {**json.loads(best["params"]),
              **{k: v[0] for k, v in SEARCH_SPACE[name][1].items()
                 if k in ("random_state", "n_jobs", "verbose")}}
    print(f"\nMejor modelo por MAE de validación: {name} ({obj}) {params}")

    best_on_train = make_model(name, params, obj).fit(Xtr, ytr)
    val_pred = best_on_train.predict(Xva)
    cv_folds = cv_on_train(make_model(name, params, obj), Xtr, ytr)

    # ---------------- reentrenar con train+val y evaluar test UNA vez ----------------
    trva = pd.concat([tr, va])
    final = make_model(name, params, obj).fit(trva[feats], trva[TARGET_Y])
    test_pred = final.predict(te[feats])
    test_tab = {"modelo_final": metrics(te[TARGET_Y], test_pred)}
    for b, pred in baselines(te).items():
        test_tab[b] = metrics(te[TARGET_Y], pred)
    mejor_base = min(test_tab["persistencia"]["MAE"], test_tab["media_movil_7d"]["MAE"])
    mejora = 1 - test_tab["modelo_final"]["MAE"] / mejor_base
    for k, m in test_tab.items():
        log_experiment({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "run_id": run_id, "modelo": f"{k}", "objetivo": obj if k == "modelo_final" else "-",
                        "params": json.dumps(json.loads(best["params"])) if k == "modelo_final" else "{}",
                        "n_features": len(feats), "split_eval": "test", **m, "fit_seconds": None})

    pd.DataFrame({"fecha": va.index, "y_real": yva.values, "y_pred": val_pred,
                  **{b: p.values for b, p in baselines(va).items()}}).to_csv(
        RESULTS_DIR / "predicciones_validacion.csv", index=False)
    pd.DataFrame({"fecha": te.index, "y_real": te[TARGET_Y].values, "y_pred": test_pred,
                  **{b: p.values for b, p in baselines(te).items()}}).to_csv(
        RESULTS_DIR / "predicciones_test.csv", index=False)

    # ---------------- serialización y registry ----------------
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"modelo_v{version}.pkl"
    joblib.dump(final, model_path)
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}
    registry[version] = {
        "version": version,
        "modelo": name,
        "objetivo": obj,
        "hiperparametros": params,
        "features": feats,
        "columnas_requeridas": required_raw(feats),
        "metricas_validacion": metrics(yva, val_pred),
        "cv_train_mae_folds": cv_folds,
        "metricas_test": test_tab,
        "mejora_vs_mejor_baseline_test": mejora,
        "entrenado_con": "train+val",
        "rango_entrenamiento": [str(trva.index.min().date()), str(trva.index.max().date())],
        "path": model_path.relative_to(ROOT).as_posix(),
        "sha256_modelo": huella(model_path),
        "sha256_dataset": huella(PROCESSED_DIR / "dataset_final.csv"),
        "git_commit": git_commit(),
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semilla": SEMILLA,
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))

    print("\nTEST (una sola evaluación):")
    for k, m in test_tab.items():
        print(f"  {k:16s} MAE={m['MAE']:8.1f}  RMSE={m['RMSE']:8.1f}  R2={m['R2']:.3f}")
    print(f"  Mejora sobre el mejor baseline: {mejora:.1%}")
    print(f"\nModelo guardado en {model_path.relative_to(ROOT)} y registrado como v{version}")
    return registry[version]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0", help="versión semántica X.Y.Z")
    args = ap.parse_args()
    main(args.version)
