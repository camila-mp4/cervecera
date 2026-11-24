"""Pipeline de preprocesamiento completo y reproducible.

Uso:
    python src/preprocessing_pipeline.py

Pasos:
 1. Lee todos los Excel de data/ y los lleva a un total diario por variable
    (lectura de las 23:59 de cada jornada, ver auxiliar_functions).
 2. Unifica los archivos solapados con una regla de desempate explícita.
 3. Limpia: descarta días incompletos, apilados, sin reinicio o con totales
    imposibles; corrige saltos espurios del totalizador de Frio.
 4. Construye features (lags, medias móviles, calendario, ratios, eficiencia).
 5. Construye el target y = Frio(D+1) con shift(-1) sobre un calendario
    completo (así un día faltante no desalinea el shift).
 6. Partición temporal 70/15/15 y selección de variables SOLO con train.
 7. Guarda data/processed/dataset_final.csv (sin escalar), el linaje y los
    checksums.

predict.py importa `build_features`, `load_daily_for_prediction` y
`QuantileClipper` de este módulo: el código de features es uno solo.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auxiliar_functions as _af  # noqa: E402
from auxiliar_functions import (  # noqa: E402
    DATA_DIR, PROCESSED_DIR, ROOT, SEMILLA, SHEETS, TARGET, DataContractError,

    daily_from_excel, find_excels, update_checksums,
)

INTERIM_DIR = DATA_DIR / "interim"
TARGET_Y = "y"           # Frio(D+1)
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

# --------------------------------------------------------------------------
# Variables crudas candidatas (salen del EDA). {hoja: [columnas]}
# --------------------------------------------------------------------------
RAW_COLUMNS: dict[str, list[str]] = {
    # Todas las columnas de Consolidado EE (la selección decide cuáles quedan)
    "Consolidado EE": ["Frio (Kw)", "Planta (Kw)", "Elaboracion (Kw)", "Bodega (Kw)", "Cocina (Kw)",
                       "Envasado (Kw)", "Linea 2 (Kw)", "Linea 3 (Kw)", "Linea 4 (Kw)",
                       "Servicios (Kw)", "Sala Maq (Kw)", "Aire (Kw)", "Calderas (Kw)",
                       "Efluentes (Kw)", "Pta Agua / Eflu (Kw)", "Prod Agua (Kw)",
                       "Resto Serv (Kw)", "Restos Planta (Kw)", "KW Gral Planta"],
    "Totalizadores Energia": ["KW Cond 5. 6 y 9"],
    "Consolidado Produccion": ["Hl de Mosto", "Hl Producido Bodega",
                               "Hl Cerveza Envasada", "Hl Cerveza Filtrada"],
    "Consolidado Agua": ["Agua Planta (Hl)"],
    "Consolidado Aire": ["Aire Producido (M3)"],
    "Consolidado KPI": ["EE Frio / Hl", "Meta EE Frio"],
}


def col(sheet: str, name: str) -> str:
    """Nombre de la columna diaria para (hoja, columna cruda)."""
    return f"{SHEETS[sheet]}__{name}"


F = TARGET  # alias corto: Frio del día D (23:59, saltos corregidos)
C = {  # alias legibles de las columnas diarias crudas
    "servicios": col("Consolidado EE", "Servicios (Kw)"),
    "planta": col("Consolidado EE", "Planta (Kw)"),
    "elab": col("Consolidado EE", "Elaboracion (Kw)"),
    "salamaq": col("Consolidado EE", "Sala Maq (Kw)"),
    "bodega": col("Consolidado EE", "Bodega (Kw)"),
    "envasado": col("Consolidado EE", "Envasado (Kw)"),
    "cond": col("Totalizadores Energia", "KW Cond 5. 6 y 9"),
    "mosto": col("Consolidado Produccion", "Hl de Mosto"),
    "hl_bodega": col("Consolidado Produccion", "Hl Producido Bodega"),
    "hl_envasada": col("Consolidado Produccion", "Hl Cerveza Envasada"),
    "hl_filtrada": col("Consolidado Produccion", "Hl Cerveza Filtrada"),
    "agua": col("Consolidado Agua", "Agua Planta (Hl)"),
    "aire": col("Consolidado Aire", "Aire Producido (M3)"),
    "ee_frio_hl": col("Consolidado KPI", "EE Frio / Hl"),
    "meta_frio": col("Consolidado KPI", "Meta EE Frio"),
}

# Feature -> columnas diarias de las que depende (para el contrato de predict).
_SELF = [F]
FEATURE_DEPS: dict[str, list[str]] = {
    "frio_hoy": _SELF, "frio_lag1": _SELF, "frio_lag2": _SELF, "frio_lag7": _SELF,
    "frio_rm3": _SELF, "frio_rm7": _SELF, "frio_rm14": _SELF, "frio_rm28": _SELF,
    "frio_std7": _SELF, "frio_delta1": _SELF,
    "frio_ult6h": ["EE__Frio_ult6h"], "frio_ult12h": ["EE__Frio_ult12h"],
    "doy_sin": [], "doy_cos": [], "mes": [], "dia_semana": [], "fin_de_semana": [],
    "servicios": [C["servicios"]], "planta": [C["planta"]], "elaboracion": [C["elab"]],
    "sala_maq": [C["salamaq"]], "bodega": [C["bodega"]], "envasado": [C["envasado"]],
    "condensadores": [C["cond"]], "hl_mosto": [C["mosto"]], "hl_bodega": [C["hl_bodega"]],
    "hl_envasada": [C["hl_envasada"]], "hl_filtrada": [C["hl_filtrada"]],
    "agua_planta": [C["agua"]], "aire_producido": [C["aire"]],
    "hl_mosto_rm7": [C["mosto"]],
    "ratio_frio_planta": [F, C["planta"]], "ratio_frio_elab": [F, C["elab"]],
    "eficiencia_frio": [C["ee_frio_hl"], C["meta_frio"]],
    "frio_x_cos": _SELF, "mosto_x_cos": [C["mosto"]],
}


def slug(nombre: str) -> str:
    """'Sala Maq (Kw)' -> 'sala_maq'; 'KW Gral Planta' -> 'kw_gral_planta'."""
    t = re.sub(r"\(\s*kw\s*\)", "", nombre.lower())
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


# Para cada columna de Consolidado EE (salvo Frio, que tiene sus propias):
# total del día, consumo de las últimas 6 horas y media móvil de 7 días.
EE_COLS = [c for c in RAW_COLUMNS["Consolidado EE"] if c != "Frio (Kw)"]
EE_FEATURES: dict[str, tuple[str, str]] = {}   # feature -> (columna diaria, tipo)
for _c in EE_COLS:
    _d = col("Consolidado EE", _c)
    EE_FEATURES[slug(_c)] = (_d, "dia")
    EE_FEATURES[f"{slug(_c)}_ult6h"] = (f"{_d}__ult6h", "dia")
    EE_FEATURES[f"{slug(_c)}_rm7"] = (_d, "rm7")
for _f, (_d, _) in EE_FEATURES.items():
    FEATURE_DEPS.setdefault(_f, [_d])
FEATURE_DEPS.update({"frio_min7": _SELF, "frio_max7": _SELF})

# Medido en el EDA: efecto del día de la semana despreciable (eta² ~ 0,03,
# ANOVA p ~ 0,34). Se construyen para trazabilidad pero NO son candidatas.
DESCARTADAS_POR_EDA = ["dia_semana", "fin_de_semana"]
CANDIDATAS = [f for f in FEATURE_DEPS if f not in DESCARTADAS_POR_EDA]

# columnas diarias -> (hoja, columna cruda), para pedir solo lo necesario
DAILY_TO_RAW = {col(s, c): (s, c) for s, cs in RAW_COLUMNS.items() for c in cs}
for _c in RAW_COLUMNS["Consolidado EE"]:
    DAILY_TO_RAW[f"{col('Consolidado EE', _c)}__ult6h"] = ("Consolidado EE", _c)
DAILY_TO_RAW["EE__Frio_ult6h"] = ("Consolidado EE", "Frio (Kw)")
DAILY_TO_RAW["EE__Frio_ult12h"] = ("Consolidado EE", "Frio (Kw)")


def required_raw(features: list[str]) -> dict[str, list[str]]:
    """{hoja: [columnas]} que el Excel debe tener para calcular `features`."""
    req: dict[str, list[str]] = {"Consolidado EE": ["Frio (Kw)"]}
    for f in features:
        if f not in FEATURE_DEPS:
            raise KeyError(f"Feature desconocida en el registro del modelo: '{f}'")
        for d in FEATURE_DEPS[f]:
            s, c = DAILY_TO_RAW[d]
            req.setdefault(s, [])
            if c not in req[s]:
                req[s].append(c)
    return req


# --------------------------------------------------------------------------
# 1-3. Carga, unificación y limpieza
# --------------------------------------------------------------------------
def unify_files(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Une los Excel solapados.

    Regla de desempate (por fecha): entre los archivos en los que el día es
    VÁLIDO, gana el archivo cuya última fecha es más reciente (extracción
    posterior, que incorpora correcciones). Se mide antes cuántas fechas se
    solapan y si los totales coinciden.
    """
    dailies, qcs, ends = {}, {}, {}
    for f in files:
        d, q = daily_from_excel(f)  # modo exploración: todas las columnas
        dailies[f.name], qcs[f.name], ends[f.name] = d, q, d.index.max()

    # --- informe de solapamiento (sobre el total diario crudo de Frio) ---
    raw = pd.DataFrame({k: q["valor_2359"] for k, q in qcs.items()})
    presentes = raw.notna().sum(axis=1)
    multi = raw[presentes > 1]
    rango = multi.max(axis=1) - multi.min(axis=1)
    overlap = {
        "fechas_en_mas_de_un_archivo": int((pd.DataFrame(
            {k: pd.Series(1, index=q.index) for k, q in qcs.items()}).notna().sum(axis=1) > 1).sum()),
        "fechas_comparables": int(len(multi)),
        "fechas_con_total_identico": int((rango == 0).sum()),
        "fechas_con_total_distinto": int((rango > 0).sum()),
        "detalle_distintas": {str(k.date()): {c: (None if pd.isna(v) else float(v))
                                              for c, v in row.items()}
                              for k, row in multi[rango > 0].iterrows()},
        "regla_desempate": "dia valido > archivo con fecha final mas reciente",
    }

    # --- unificación fila a fila ---
    order = sorted(files, key=lambda f: ends[f.name], reverse=True)
    parts = []
    for prio, f in enumerate(order):
        d = dailies[f.name].copy()
        d["_archivo"], d["_prioridad"] = f.name, prio
        d["_valido"] = qcs[f.name]["valido"].reindex(d.index).fillna(False).astype(bool)
        parts.append(d)
    allp = pd.concat(parts).reset_index()
    allp = allp.sort_values(["DIA", "_valido", "_prioridad"], ascending=[True, False, True])
    unified = allp.drop_duplicates("DIA").set_index("DIA").sort_index()

    calendario = pd.date_range(unified.index.min(), unified.index.max(), freq="D", name="DIA")
    unified = unified.reindex(calendario)
    unified["_valido"] = unified["_valido"].fillna(False).astype(bool)

    qc = pd.concat({k: q for k, q in qcs.items()}, names=["archivo", "DIA"])
    return unified, qc, overlap


def load_daily_for_prediction(path: str | Path, features: list[str]) -> pd.DataFrame:
    """Carga un Excel nuevo pidiendo SOLO las hojas/columnas que necesitan
    `features`; si falta alguna, DataContractError dice cuál."""
    daily, _ = daily_from_excel(path, required=required_raw(features))
    cal = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="DIA")
    return daily.reindex(cal)


# --------------------------------------------------------------------------
# 4-5. Feature engineering y target
# --------------------------------------------------------------------------
def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Features del día D a partir del dataframe diario (índice = calendario
    completo). Solo usa información disponible al cierre del día D."""
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise TypeError("build_features espera un índice de fechas diarias")
    d = daily
    g = lambda c: d[c] if c in d.columns else pd.Series(np.nan, index=d.index)  # noqa: E731
    s = g(F)
    X = pd.DataFrame(index=d.index)

    # nivel actual con respaldo: si el día D es inválido, se usa el último
    # valor disponible de los 7 días previos (sin mirar el futuro)
    X["frio_hoy"] = s.fillna(s.ffill(limit=7))
    X["frio_lag1"] = s.shift(1).fillna(X["frio_hoy"])
    X["frio_lag2"] = s.shift(2).fillna(X["frio_lag1"])
    X["frio_lag7"] = s.shift(7).fillna(X["frio_lag2"])
    for w in (3, 7, 14, 28):
        X[f"frio_rm{w}"] = s.rolling(w, min_periods=1).mean()
    X["frio_std7"] = s.rolling(7, min_periods=2).std()
    X["frio_delta1"] = X["frio_hoy"] - X["frio_lag1"]
    # ritmo de consumo al final del día, llevado a 24 h
    X["frio_ult6h"] = g("EE__Frio_ult6h") * 4
    X["frio_ult12h"] = g("EE__Frio_ult12h") * 2

    # calendario. La estacionalidad anual se codifica cíclicamente (sin/cos)
    # porque diciembre y enero son vecinos; el mes queda para trazabilidad.
    doy = d.index.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # calendario del día que se predice (D+1): la fecha de mañana se conoce hoy
    manana = d.index + pd.Timedelta(days=1)
    X["mes"] = manana.month
    X["dia_semana"] = manana.dayofweek
    X["fin_de_semana"] = (manana.dayofweek >= 5).astype(int)
    X["frio_min7"] = s.rolling(7, min_periods=1).min()
    X["frio_max7"] = s.rolling(7, min_periods=1).max()

    # todas las áreas de Consolidado EE (día D): total, últimas 6 h y media 7 d
    for f, (dcol, tipo) in EE_FEATURES.items():
        v = g(dcol)
        X[f] = v.rolling(7, min_periods=1).mean() if tipo == "rm7" else v
    X["condensadores"] = g(C["cond"])
    X["hl_mosto"], X["hl_bodega"] = g(C["mosto"]), g(C["hl_bodega"])
    X["hl_envasada"], X["hl_filtrada"] = g(C["hl_envasada"]), g(C["hl_filtrada"])
    X["agua_planta"], X["aire_producido"] = g(C["agua"]), g(C["aire"])
    X["hl_mosto_rm7"] = g(C["mosto"]).rolling(7, min_periods=1).mean()

    # ratios entre áreas (división segura)
    def ratio(a, b):
        b = b.where(b > 0)
        return a / b
    X["ratio_frio_planta"] = ratio(s, X["planta"])
    X["ratio_frio_elab"] = ratio(s, X["elaboracion"])
    # eficiencia energética: kW de frío por Hl real contra la meta por Hl
    X["eficiencia_frio"] = ratio(g(C["ee_frio_hl"]), g(C["meta_frio"]))

    # interacciones: nivel actual y producción modulados por la estación
    X["frio_x_cos"] = X["frio_hoy"] * X["doy_cos"]
    X["mosto_x_cos"] = X["hl_mosto"] * X["doy_cos"]

    return X.replace([np.inf, -np.inf], np.nan)


def build_target(daily: pd.DataFrame) -> pd.Series:
    """y[D] = Frio[D+1]. El índice es un calendario completo, por eso
    shift(-1) siempre trae el día calendario siguiente (o NaN si falta)."""
    idx = daily.index
    if len(idx) > 1 and not (pd.Series(idx).diff().dropna() == pd.Timedelta("1D")).all():
        raise ValueError("El índice diario debe ser un calendario completo antes del shift(-1)")
    return daily[F].shift(-1).rename(TARGET_Y)


# --------------------------------------------------------------------------
# Transformador: recorte por cuantiles aprendidos en train
# --------------------------------------------------------------------------
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Recorta cada columna a [q_low, q_high] aprendidos en el fit (train).
    Trata los outliers extremos de las features sin mirar val/test."""

    def __init__(self, low: float = 0.005, high: float = 0.995):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_ = X.quantile(self.low).to_numpy()
        self.hi_ = X.quantile(self.high).to_numpy()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        A = np.asarray(X, dtype=float)
        return np.clip(A, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


class DeltaRegressor(BaseEstimator):
    """Envuelve un regresor para que aprenda el CAMBIO respecto del consumo
    de hoy: y_hat = frio_hoy + f(X). Ayuda a los árboles, que no extrapolan
    niveles fuera del rango de train, y no cambia nada para los lineales."""

    def __init__(self, estimator=None, base_col: str = "frio_hoy"):
        self.estimator, self.base_col = estimator, base_col

    def _base(self, X):
        b = pd.DataFrame(X)[self.base_col].astype(float)
        return b.fillna(self.base_fill_).to_numpy()

    def fit(self, X, y):
        from sklearn.base import clone
        self.base_fill_ = float(pd.DataFrame(X)[self.base_col].median())
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y) - self._base(X))
        return self

    def predict(self, X):
        return self._base(X) + self.estimator_.predict(X)


def make_preprocessor() -> Pipeline:
    """Recorte -> imputación por mediana -> estandarización. Se ajusta
    únicamente con train (dentro del Pipeline del modelo)."""
    return Pipeline([
        ("clip", QuantileClipper()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# --------------------------------------------------------------------------
# 6. Partición temporal y selección de variables
# --------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame) -> pd.Series:
    """Etiqueta train/val/test por orden temporal (70/15/15)."""
    n = len(df)
    i1, i2 = int(round(n * TRAIN_FRAC)), int(round(n * (TRAIN_FRAC + VAL_FRAC)))
    lab = np.array(["train"] * i1 + ["val"] * (i2 - i1) + ["test"] * (n - i2))
    return pd.Series(lab, index=df.index, name="split")


def select_features(X: pd.DataFrame, y: pd.Series, n_keep: int = 15) -> dict:
    """Tres métodos, todos ajustados SOLO con train:
    LASSO (coef != 0), importancia de Random Forest (top n) y RFE con Ridge
    (top n). Se queda con las variables elegidas por al menos 2 de 3.
    Antes, un filtro que no mira el target: se descartan candidatas con más
    de 5 % de faltantes o con 10 valores distintos o menos en train."""
    ok = (X.isna().mean() <= 0.05) & (X.nunique() > 10)
    X = X.loc[:, ok]
    pre = make_preprocessor()
    Xs = pd.DataFrame(pre.fit_transform(X), columns=X.columns, index=X.index)
    tss = TimeSeriesSplit(n_splits=5)

    lasso = LassoCV(cv=tss, random_state=SEMILLA, max_iter=50000).fit(Xs, y)
    coef = pd.Series(lasso.coef_, index=X.columns)

    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                               random_state=SEMILLA, n_jobs=-1).fit(Xs, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    rfe = RFE(Ridge(alpha=1.0), n_features_to_select=n_keep).fit(Xs, y)
    rfe_rank = pd.Series(rfe.ranking_, index=X.columns)

    votos = pd.DataFrame({
        "lasso": coef.abs() > 1e-9,
        "random_forest": X.columns.isin(imp.index[:n_keep]),
        "rfe_ridge": rfe_rank == 1,
    }, index=X.columns)
    votos["votos"] = votos.sum(axis=1)
    elegidas = votos.index[votos["votos"] >= 2].tolist()
    return {
        "seleccionadas": elegidas,
        "votos": votos,
        "lasso_coef": coef,
        "lasso_alpha": float(lasso.alpha_),
        "rf_importancia": imp,
        "rfe_ranking": rfe_rank,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run(verbose: bool = True, guardar: bool = True) -> dict:
    """guardar=False arma todo en memoria sin escribir archivos (para comparar
    variantes desde los notebooks sin pisar los artefactos versionados)."""
    files = find_excels()
    unified, qc, overlap = unify_files(files)

    daily = unified[[c for c in unified.columns if not c.startswith("_")]]
    X = build_features(daily)
    y = build_target(daily)

    ds = X.join(y)
    ds = ds[ds[TARGET_Y].notna()]          # sin target no hay fila de entrenamiento
    ds = ds[ds["frio_hoy"].notna()]        # ni fila sin ningún nivel reciente de Frio
    ds["split"] = temporal_split(ds)
    ds.index.name = "fecha"

    tr = ds[ds["split"] == "train"]
    sel = select_features(tr[CANDIDATAS], tr[TARGET_Y])

    if not guardar:
        return {"dataset": ds, "seleccion": sel, "unified": unified, "qc": qc, "overlap": overlap}
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    unified_path = INTERIM_DIR / "dataset_unificado.csv"
    unified.to_csv(unified_path, float_format="%.6f")
    qc.to_csv(INTERIM_DIR / "calidad_target.csv")
    final_path = PROCESSED_DIR / "dataset_final.csv"
    ds.to_csv(final_path, float_format="%.6f")
    (PROCESSED_DIR / "selected_features.json").write_text(json.dumps({
        "candidatas": CANDIDATAS,
        "descartadas_por_eda": DESCARTADAS_POR_EDA,
        "seleccionadas": sel["seleccionadas"],
        "votos": sel["votos"].astype(int).to_dict(orient="index"),
        "lasso_alpha": sel["lasso_alpha"],
    }, indent=2, ensure_ascii=False))

    shapes = {k: list(ds[ds["split"] == k][sel["seleccionadas"]].shape) for k in ("train", "val", "test")}
    rangos = {k: [str(ds[ds["split"] == k].index.min().date()), str(ds[ds["split"] == k].index.max().date())]
              for k in ("train", "val", "test")}
    lineage = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/preprocessing_pipeline.py",
        "entradas": [f.name for f in files],
        "pasos": [
            "lectura por nombre de hojas/columnas; HORA truncada al minuto",
            ("jornadas apiladas: reasignadas con la regla del EDA" if _af.REASIGNAR_APILADAS
             else "jornadas apiladas: fechas invalidas (la regla de reasignacion no paso la validacion)"),
            "total diario = lectura de las 23:59; si falta, 23:00 + ritmo de las ultimas 3 h x 59 min",
            "Frio: total = suma de incrementos horarios; saltos (<0 o >3000 kW, incluidos los del millon) "
            "reemplazados por la mediana del dia; arrastre sin reinicio descontado",
            "otras columnas: valores >= 1e6 -> NaN",
            "dias invalidos: <24 lecturas, arranque despues de 00:00, apilados no reasignados, >2 saltos",
            "consumo de las ultimas 6 h (18:00 -> cierre) para todas las columnas de Consolidado EE",
            "unificacion: dia valido > archivo con fecha final mas reciente",
            "reindexado a calendario completo",
            "features: lags, medias moviles, sin/cos dia del anio, areas, ratios, eficiencia, interacciones",
            "target y = Frio.shift(-1) sobre calendario completo",
            "filas sin target o sin nivel de Frio reciente eliminadas",
            "split temporal 70/15/15",
            "filtro sin target (<=5% NaN, >10 valores distintos) + seleccion LASSO/RF/RFE solo con train, >=2 votos",
            "dataset_final.csv NO escalado (el escalado vive en el Pipeline del modelo)",
        ],
        "solapamiento": overlap,
        "dias_calendario": int(len(unified)),
        "dias_validos": int(unified["_valido"].sum()),
        "filas_dataset_final": int(len(ds)),
        "shapes": shapes,
        "rangos": rangos,
        "salidas": ["data/interim/dataset_unificado.csv", "data/processed/dataset_final.csv",
                    "data/processed/selected_features.json"],
    }
    (PROCESSED_DIR / "data_lineage.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=False))
    update_checksums(files + [unified_path, final_path])

    if verbose:
        print(f"Dataset final: {final_path.relative_to(ROOT)}  {ds.shape}")
        print("Shapes:", shapes)
        print("Variables seleccionadas:", sel["seleccionadas"])
    return {"dataset": ds, "seleccion": sel, "lineage": lineage, "unified": unified, "qc": qc,
            "overlap": overlap}


if __name__ == "__main__":
    try:
        run()
    except (DataContractError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
