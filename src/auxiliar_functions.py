"""Funciones auxiliares compartidas por todo el proyecto.

Todo lo que toca el Excel crudo vive acá, para que el entrenamiento
(preprocessing_pipeline.py) y la predicción (predict.py) usen
exactamente el mismo código.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constantes del proyecto
# --------------------------------------------------------------------------
SEMILLA = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "reports" / "figures"

EXCEL_GLOB = "Totalizadores*Planta*.xlsx"
# El archivo de test oculto (2024_2025) nunca debe entrar al entrenamiento,
# aunque alguien lo copie dentro de data/.
EXCLUIR_DE_ENTRENAMIENTO = ("2024_2025", "2024-2025")

# Hoja -> prefijo corto que se antepone a cada columna para evitar choques de
# nombres entre hojas (p. ej. "KW Gral Planta" existe en dos hojas).
SHEETS = {
    "Consolidado EE": "EE",
    "Totalizadores Energia": "TEN",
    "Consolidado Produccion": "PROD",
    "Totalizadores Produccion": "TPROD",
    "Consolidado KPI": "KPI",
    "Consolidado Agua": "AGUA",
    "Totalizadores Agua": "TAGUA",
    "Consolidado GasVapor": "GV",
    "Totalizadores Gas y Vapor": "TGV",
    "Consolidado Aire": "AIRE",
    "Totalizadores Aire": "TAIRE",
    "Totalizadores CO2": "CO2",
    "Totalizadores Efluentes": "EFL",
    "Totalizadores Glicol": "GLI",
}
# Hojas auxiliares de la planilla: no contienen mediciones, se ignoran.
IGNORED_SHEETS = {"Auxiliar", "Seguimiento Dia"}

TARGET_SHEET = "Consolidado EE"
TARGET_RAW_COL = "Frio (Kw)"
TARGET = "EE__Frio (Kw)"  # nombre de la columna diaria (valor de las 23:59)

# Definición del total diario: lectura de las 23:59 (minuto 1439).
END_OF_DAY_MIN = 23 * 60 + 59
MIN_LECTURAS = 24          # menos de esto = día incompleto
VALOR_IMPOSIBLE = 1e6      # totales diarios del orden del millón no son físicos
# Incremento horario máximo físicamente plausible del totalizador de Frio.
# El percentil 99,9 de los incrementos horarios del archivo más limpio es
# ~2.000 kW; 3.000 kW/h deja margen y sigue aislando los saltos espurios
# (hay saltos de +28.000 kW en una hora).
MAX_INC_HORARIO = 3000
MAX_SALTOS_CORREGIBLES = 2  # con más saltos que esto, el día se descarta


class DataContractError(ValueError):
    """Error claro cuando el archivo de entrada no cumple el contrato."""


# --------------------------------------------------------------------------
# Hash de archivos (checksums)
# --------------------------------------------------------------------------
def huella(path: str | Path, bloque: int = 1 << 20) -> str:
    """SHA-256 de un archivo leído por bloques de 1 MiB (no carga todo en RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bloque), b""):
            h.update(chunk)
    return h.hexdigest()


def update_checksums(paths, checksum_file: Path = DATA_DIR / "checksums.json") -> dict:
    """Agrega/actualiza {archivo: sha256} en data/checksums.json."""
    checksum_file = Path(checksum_file)
    data = json.loads(checksum_file.read_text()) if checksum_file.exists() else {}
    for p in paths:
        p = Path(p)
        data[p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name] = huella(p)
    checksum_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def find_excels(data_dir: Path = DATA_DIR) -> list[Path]:
    files = sorted(f for f in Path(data_dir).glob(EXCEL_GLOB)
                   if not any(p in f.name for p in EXCLUIR_DE_ENTRENAMIENTO)
                   and not f.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos '{EXCEL_GLOB}' en {data_dir}. "
            "Ver README.md, sección 'Datos'."
        )
    return files


# --------------------------------------------------------------------------
# Lectura robusta del Excel (por nombre, nunca por posición)
# --------------------------------------------------------------------------
def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def check_contract(nombre: str, raw: dict[str, pd.DataFrame],
                   required: dict[str, list[str]]) -> None:
    """Verifica TODAS las hojas y columnas requeridas y, si falta algo,
    levanta un único error que lista exactamente qué falta."""
    faltan = []
    for sheet, cols in required.items():
        if sheet not in raw:
            faltan.append(f"  - hoja '{sheet}' (completa)")
            continue
        df = raw[sheet]
        for c in ["DIA", "HORA", *cols]:
            if _col_lookup(df, c) is None:
                faltan.append(f"  - columna '{c}' en la hoja '{sheet}'")
    if faltan:
        raise DataContractError(
            f"El archivo '{nombre}' no cumple el contrato de entrada. Falta:\n" + "\n".join(faltan))


def read_excel_sheets(path: str | Path, sheets=None, strict: bool = True) -> dict[str, pd.DataFrame]:
    """Lee las hojas pedidas (por nombre, tolerando mayúsculas/espacios).

    Si `sheets` es None lee todas las hojas conocidas que estén presentes.
    Las hojas que falten se informan con DataContractError solo si fueron
    pedidas explícitamente.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
    try:
        xls = pd.ExcelFile(path, engine="calamine")
    except Exception:  # calamine no instalado -> openpyxl
        xls = pd.ExcelFile(path)
    available = {_norm(s): s for s in xls.sheet_names}

    wanted = list(SHEETS) if sheets is None else list(sheets)
    out, missing = {}, []
    for s in wanted:
        real = available.get(_norm(s))
        if real is None:
            missing.append(s)
            continue
        df = xls.parse(real)
        df.columns = [str(c).strip() for c in df.columns]
        out[s] = df
    if strict and sheets is not None and missing:
        raise DataContractError(
            f"El archivo '{path.name}' no tiene las hojas requeridas: {missing}. "
            f"Hojas disponibles: {xls.sheet_names}"
        )
    return out


def _hora_a_minutos(h) -> float:
    """Convierte la columna HORA (str 'HH:MM:SS', time o datetime) a minutos
    desde medianoche, truncando los segundos (07:00:01 -> 420, 23:59:59 -> 1439)."""
    if h is None or (isinstance(h, float) and np.isnan(h)):
        return np.nan
    if isinstance(h, (time, datetime)):
        return h.hour * 60 + h.minute
    if isinstance(h, pd.Timedelta):
        return int(h.total_seconds() // 60) % 1440
    if isinstance(h, (int, float)):  # fracción de día de Excel
        return int(round(h * 86400) // 60) % 1440
    m = re.match(r"^\s*(\d{1,2}):(\d{2})", str(h))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan


def _col_lookup(df: pd.DataFrame, name: str) -> str | None:
    norm = {_norm(c): c for c in df.columns}
    return norm.get(_norm(name))


# --------------------------------------------------------------------------
# Jornadas apiladas: reasignación opcional
# --------------------------------------------------------------------------
# La planilla tiene 31 renglones de día por mes. Algunas fechas traen bloques
# extra: el 30 de los meses de 30 días y el último día de febrero. La regla
# del EDA del equipo los reasigna así (solo si la fecha destino falta):
#   30/MM, bloque extra        -> 31/(MM-1)
#   fin de febrero, bloques 2-4 -> 31/12, 30/08 y 29/08 del año anterior
# La validación por continuidad (preprocesamiento.ipynb) muestra que los
# bloques se parecen a los vecinos de su fecha ORIGINAL y no a los del
# destino, y la validación del modelo no mejora al reasignar. Por eso por
# defecto la regla queda apagada y esas fechas se descartan.
REASIGNAR_APILADAS = False


def destinos_apilados(fecha: pd.Timestamp, bloque: int) -> pd.Timestamp | None:
    """Fecha destino de un bloque extra (bloque >= 1) según la regla del EDA."""
    fin_feb = fecha.month == 2 and (fecha + pd.Timedelta(days=1)).month == 3
    if fin_feb:
        orden = [pd.Timestamp(fecha.year - 1, 12, 31), pd.Timestamp(fecha.year - 1, 8, 30),
                 pd.Timestamp(fecha.year - 1, 8, 29)]
        return orden[bloque - 1] if bloque - 1 < len(orden) else None
    if fecha.day == 30 and bloque == 1:
        destino = fecha.replace(day=1) - pd.Timedelta(days=1)
        return destino if destino.day == 31 else None
    return None


def prepare_hourly(df: pd.DataFrame, sheet: str, reasignar: bool | None = None) -> pd.DataFrame:
    """Normaliza una hoja horaria: DIA (fecha), MIN (minuto del día), bloque.

    `bloque` numera las jornadas que comparten la misma etiqueta DIA: cada vez
    que la hora retrocede dentro de una misma fecha empieza una jornada nueva.
    Con `reasignar`, los bloques extra se mueven a su fecha destino si esa
    fecha no existe en la hoja (regla general, sin fechas escritas a mano).
    """
    reasignar = REASIGNAR_APILADAS if reasignar is None else reasignar
    c_dia, c_hora = _col_lookup(df, "DIA"), _col_lookup(df, "HORA")
    if c_dia is None or c_hora is None:
        raise DataContractError(f"La hoja '{sheet}' no tiene las columnas 'DIA' y 'HORA'.")
    out = df.rename(columns={c_dia: "DIA", c_hora: "HORA"}).copy()
    out["DIA"] = pd.to_datetime(out["DIA"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["DIA"]).reset_index(drop=True)
    out["MIN"] = out["HORA"].map(_hora_a_minutos)
    prev = out.groupby("DIA")["MIN"].shift()
    out["bloque"] = (out["MIN"] < prev).astype(int).groupby(out["DIA"]).cumsum()
    out["reasignado"] = False
    if reasignar and (out["bloque"] > 0).any():
        presentes = set(out.loc[out["bloque"] == 0, "DIA"])
        for (fecha, b), idx in out[out["bloque"] > 0].groupby(["DIA", "bloque"]).groups.items():
            destino = destinos_apilados(fecha, int(b))
            if destino is not None and destino not in presentes:
                out.loc[idx, ["DIA", "bloque", "reasignado"]] = [destino, 0, True]
                presentes.add(destino)
        out = out.sort_values(["DIA", "bloque"], kind="stable").reset_index(drop=True)
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    skip = {"dia", "hora", "min", "bloque", "id", "reasignado"}
    return [c for c in df.columns
            if _norm(c) not in skip and not str(c).startswith("Unnamed") and str(c).strip()]


# --------------------------------------------------------------------------
# Total diario
# --------------------------------------------------------------------------
HORAS_RITMO = 3   # ritmo para estimar los 59 min faltantes (validado: ver abajo)
HORA_COLA = 18    # "últimas 6 horas" = consumo de 18:00 al cierre


def _lectura(p: pd.DataFrame, minuto: int) -> pd.Series:
    return p[minuto] if minuto in p.columns else pd.Series(np.nan, index=p.index)


def cierre_diario(p: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Total del día a partir de una tabla (día x minuto) de un totalizador.

    Usa la lectura de las 23:59. Si falta pero están las 23:00 y las
    (23-HORAS_RITMO):00, estima los 59 minutos con el ritmo de esas horas.
    Devuelve (total, estimado?)."""
    v2359, v23 = _lectura(p, END_OF_DAY_MIN), _lectura(p, 23 * 60)
    vr = _lectura(p, (23 - HORAS_RITMO) * 60)
    est = v23 + (v23 - vr) / (HORAS_RITMO * 60) * 59
    usar_est = v2359.isna() & est.notna()
    return v2359.where(~usar_est, est), usar_est


def daily_quality(hourly: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Diagnóstico y total corregido por fecha para el totalizador del target.

    El total se arma con la SUMA DE INCREMENTOS horarios de la jornada:
      * si el contador no se reinició a las 00:00, el arrastre no cuenta;
      * los saltos espurios (incremento < 0 o > MAX_INC_HORARIO, incluidos los
        totales del orden del millón) se reemplazan por la mediana del día;
      * si falta la lectura de las 23:59, los 59 minutos se estiman con el
        ritmo de las últimas HORAS_RITMO horas (corregidas).
    """
    h = hourly[hourly["bloque"] == 0].copy()
    h["_v"] = pd.to_numeric(h[value_col], errors="coerce")
    h = h.dropna(subset=["MIN"])
    p = h.pivot_table(index="DIA", columns="MIN", values="_v", aggfunc="first")
    g_all = hourly.groupby("DIA")
    idx = pd.Index(sorted(hourly["DIA"].unique()), name="DIA")

    q = pd.DataFrame(index=idx)
    q["n_registros"] = g_all.size()
    q["n_jornadas"] = g_all["bloque"].max() + 1
    q["reasignado"] = hourly.groupby("DIA")["reasignado"].any() if "reasignado" in hourly else False
    q["n_lecturas"] = h.groupby("DIA").size().reindex(idx).fillna(0).astype(int)
    q["primera_hora_min"] = h.groupby("DIA")["MIN"].min()
    q["valor_0000"] = _lectura(p, 0).reindex(idx)
    q["valor_2300"] = _lectura(p, 23 * 60).reindex(idx)
    q["valor_2359"] = _lectura(p, END_OF_DAY_MIN).reindex(idx)
    q["max_valor"] = p.max(axis=1).reindex(idx)

    # serie acumulada corregida: incrementos con los saltos reemplazados
    horas = [m for m in sorted(p.columns) if m % 60 == 0 or m == END_OF_DAY_MIN]
    pp = p[horas]
    # incremento entre lecturas consecutivas disponibles (un hueco no borra consumo)
    inc = pp.ffill(axis=1).diff(axis=1).where(pp.notna()).iloc[:, 1:]
    salto = (inc < 0) | (inc > MAX_INC_HORARIO)
    med = inc.where(~salto).median(axis=1)
    reemplazo = pd.DataFrame(np.repeat(med.to_numpy()[:, None], inc.shape[1], axis=1),
                             index=inc.index, columns=inc.columns)
    inc_c = inc.mask(salto, reemplazo)
    acum = pd.concat([pd.Series(0.0, index=pp.index), inc_c.fillna(0).cumsum(axis=1)], axis=1)
    acum.columns = pp.columns
    acum = acum.where(pp.notna())
    total, estimado = cierre_diario(acum)
    q["n_saltos"] = salto.sum(axis=1).reindex(idx).fillna(0).astype(int)
    q["valor_corregido"] = total.reindex(idx)
    q["estimado_2359"] = estimado.reindex(idx).fillna(False).astype(bool)
    q["ult6h"] = (total - _lectura(acum, HORA_COLA * 60)).reindex(idx)
    q["ult12h"] = (total - _lectura(acum, 12 * 60)).reindex(idx)

    q["sin_reinicio"] = q["valor_0000"].fillna(0) > 0          # se corrige, no invalida
    q["incompleto"] = (q["n_lecturas"] < MIN_LECTURAS) | (q["primera_hora_min"] > 0) \
        | q["valor_corregido"].isna()
    q["imposible"] = q["max_valor"] >= VALOR_IMPOSIBLE          # se reconstruye si se puede
    q["apilado"] = q["n_jornadas"] > 1
    q["danado"] = q["n_saltos"] > MAX_SALTOS_CORREGIBLES
    q["valido"] = ~(q["incompleto"] | q["apilado"] | q["danado"])
    return q


def daily_from_sheet(df: pd.DataFrame, sheet: str, prefix: str,
                     columns: list[str] | None = None, colas: bool = False) -> pd.DataFrame:
    """Total diario de cada columna = lectura de las 23:59 de la jornada
    (o su estimación desde las 23:00). Con `colas`, agrega además el consumo
    de las últimas 6 horas (`<col>__ult6h`). Fechas con jornadas apiladas que
    no se pudieron reasignar -> NaN."""
    h = prepare_hourly(df, sheet)
    cols = numeric_columns(h) if columns is None else columns
    missing = [c for c in cols if _col_lookup(h, c) is None]
    if missing:
        raise DataContractError(f"La hoja '{sheet}' no tiene las columnas requeridas: {missing}")
    real = [_col_lookup(h, c) for c in cols]
    vals = h[real].apply(pd.to_numeric, errors="coerce")
    # valores del orden del millón no son físicos en ningún totalizador diario
    vals = vals.mask(vals.abs() >= VALOR_IMPOSIBLE)
    vals.columns = cols
    vals[["DIA", "MIN", "bloque"]] = h[["DIA", "MIN", "bloque"]]
    all_days = pd.Index(sorted(vals["DIA"].unique()), name="DIA")
    stacked = vals.groupby("DIA")["bloque"].max() > 0
    b0 = vals[vals["bloque"] == 0].dropna(subset=["MIN"])
    salida = {}
    for c in cols:
        p = b0.pivot_table(index="DIA", columns="MIN", values=c, aggfunc="first")
        total, _ = cierre_diario(p)
        salida[f"{prefix}__{c}"] = total
        if colas:
            salida[f"{prefix}__{c}__ult6h"] = total - _lectura(p, HORA_COLA * 60)
    daily = pd.DataFrame(salida).reindex(all_days)
    daily.loc[stacked[stacked].index.intersection(daily.index)] = np.nan
    return daily


def daily_from_excel(path: str | Path, required: dict[str, list[str]] | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convierte un Excel horario en (diario, calidad_del_target).

    required: {hoja: [columnas]}. None = todas las hojas/columnas conocidas
    (modo exploración). En modo predicción se pasa el subconjunto exacto que
    necesita el modelo y cualquier faltante produce DataContractError.
    """
    if required is None:
        raw = read_excel_sheets(path)
    else:
        required = {s: list(c) for s, c in required.items()}
        required.setdefault(TARGET_SHEET, [])
        if TARGET_RAW_COL not in required[TARGET_SHEET]:
            required[TARGET_SHEET].append(TARGET_RAW_COL)
        raw = read_excel_sheets(path, sheets=list(required), strict=False)
        check_contract(Path(path).name, raw, required)
    if TARGET_SHEET not in raw:
        raise DataContractError(f"El archivo '{Path(path).name}' no tiene la hoja '{TARGET_SHEET}'.")
    parts = []
    for sheet, df in raw.items():
        cols = None if required is None else list(dict.fromkeys(
            required.get(sheet, []) + ([TARGET_RAW_COL] if sheet == TARGET_SHEET else [])))
        parts.append(daily_from_sheet(df, sheet, SHEETS[sheet], cols, colas=(sheet == TARGET_SHEET)))
    daily = pd.concat(parts, axis=1).sort_index()
    q = daily_quality(prepare_hourly(raw[TARGET_SHEET], TARGET_SHEET),
                      _col_lookup(raw[TARGET_SHEET], TARGET_RAW_COL) or TARGET_RAW_COL)
    q = q.reindex(daily.index)
    # Frio del día: suma de incrementos corregida (ver daily_quality).
    daily[TARGET] = q["valor_corregido"]
    daily[f"{TARGET}__ult6h"] = q["ult6h"]
    daily["EE__Frio_ult6h"] = q["ult6h"]       # nombres de las features de fin de día de Frio
    daily["EE__Frio_ult12h"] = q["ult12h"]
    daily.loc[~q["valido"].fillna(False).astype(bool)] = np.nan
    return daily, q


# --------------------------------------------------------------------------
# Validaciones de los métodos de recuperación (se muestran en el notebook)
# --------------------------------------------------------------------------
def dias_sanos(hourly: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Tabla (día x minuto) de los días con las 25 lecturas, arranque en 0 y
    sin saltos: tienen el valor real de las 23:59 contra el cual comparar."""
    h = hourly[hourly["bloque"] == 0].assign(_v=lambda d: pd.to_numeric(d[value_col], errors="coerce"))
    p = h.pivot_table(index="DIA", columns="MIN", values="_v", aggfunc="first")
    need = [60 * i for i in range(24)] + [END_OF_DAY_MIN]
    if any(m not in p.columns for m in need):
        return p.iloc[0:0]
    p = p[need].dropna()
    inc = p.diff(axis=1).iloc[:, 1:]
    return p[((inc >= 0) & (inc <= MAX_INC_HORARIO)).all(axis=1) & (p[0] == 0)]


def validar_estimacion_2359(sanos: pd.DataFrame) -> pd.DataFrame:
    """Compara métodos para estimar las 23:59 desde las 23:00 en días sanos."""
    real, v23 = sanos[END_OF_DAY_MIN], sanos[23 * 60]
    metodos = {
        "23:00 sin corregir": v23,
        "ritmo últimas 3 h (elegido)": v23 + (v23 - sanos[20 * 60]) / 180 * 59,
        "ritmo últimas 6 h": v23 + (v23 - sanos[17 * 60]) / 360 * 59,
        "ritmo del día completo": v23 + v23 / 1380 * 59,
    }
    return pd.DataFrame({m: {"MAE_kW": (e - real).abs().mean(), "sesgo_kW": (e - real).mean()}
                         for m, e in metodos.items()}).T


def validar_reconstruccion_saltos(sanos: pd.DataFrame, semilla: int = SEMILLA) -> dict:
    """Inyecta un salto (25 mil, 1 millón o 5 millones de kW) en una hora al
    azar de cada día sano, lo corrige con la regla de daily_quality y mide el
    error contra el total real."""
    rng = np.random.default_rng(semilla)
    errores = []
    for _, r in sanos.iterrows():
        v = r.to_numpy(dtype=float).copy()
        j = rng.integers(1, len(v))
        v[j:] += rng.choice([2.5e4, 1e6, 5e6])
        inc = np.diff(v)
        malo = (inc < 0) | (inc > MAX_INC_HORARIO)
        bueno = inc[~malo]
        errores.append(bueno.sum() + malo.sum() * np.median(bueno) - r.iloc[-1])
    e = np.array(errores)
    return {"dias": int(len(e)), "MAE_kW": float(np.abs(e).mean()), "sesgo_kW": float(e.mean())}


def verify_checksums(checksum_file: Path = DATA_DIR / "checksums.json") -> bool:
    """Compara los archivos presentes con data/checksums.json.
    Uso: python src/auxiliar_functions.py"""
    data = json.loads(Path(checksum_file).read_text())
    ok = True
    for rel, esperado in data.items():
        p = ROOT / rel
        if not p.exists():
            print(f"FALTA     {rel}")
            ok = False
            continue
        real = huella(p)
        estado = "OK" if real == esperado else "DISTINTO"
        ok &= real == esperado
        print(f"{estado:9s} {rel}")
    return ok


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(0 if verify_checksums() else 1)
