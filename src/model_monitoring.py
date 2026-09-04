"""Monitoreo del modelo — detección de data drift.

Compara la distribución de cada variable en una ventana reciente ("actual")
contra la ventana de referencia (la población con la que se entrenó el modelo).
Muestreo periódico: mensual, usando fecha_prestamo.

Métricas: KS (numéricas), PSI, Jensen-Shannon, chi-cuadrado (categóricas).
Umbrales PSI convencionales: <0.10 estable, 0.10–0.25 alerta, >0.25 crítico.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.ft_engineering import cargar_datos, limpiar, crear_features, CATEGORICAS

EPS = 1e-6  # evita divisiones y logaritmos con cero


def _proporciones_numericas(ref, cur, bins=10):
    """Discretiza ambas series con cuantiles de la REFERENCIA y devuelve
    las proporciones por bin. La referencia define los cortes: lo que se
    mide es cuánto se movió la población actual respecto de esos cortes."""
    ref, cur = ref.dropna(), cur.dropna()
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:  # variable casi constante
        return None, None
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(ref, bins=cortes)[0] / len(ref)
    q = np.histogram(cur, bins=cortes)[0] / len(cur)
    return p + EPS, q + EPS


def _proporciones_categoricas(ref, cur):
    cats = sorted(set(ref.dropna()) | set(cur.dropna()))
    p = ref.value_counts(normalize=True).reindex(cats).fillna(0).values
    q = cur.value_counts(normalize=True).reindex(cats).fillna(0).values
    return p + EPS, q + EPS


def calcular_psi(p, q):
    """Population Stability Index: suma de (q-p)*ln(q/p) sobre los bins."""
    return float(np.sum((q - p) * np.log(q / p)))


def drift_variable(ref, cur, es_categorica):
    """Métricas de drift de una variable entre referencia y ventana actual."""
    if es_categorica:
        p, q = _proporciones_categoricas(ref, cur)
        chi2_stat, chi2_p = stats.chisquare(f_obs=q * len(cur.dropna()),
                                            f_exp=p * len(cur.dropna()))
        ks_stat, ks_p = np.nan, np.nan
    else:
        p, q = _proporciones_numericas(ref, cur)
        if p is None:
            return None
        ks_stat, ks_p = stats.ks_2samp(ref.dropna(), cur.dropna())
        chi2_stat, chi2_p = np.nan, np.nan

    return {
        "psi": calcular_psi(p, q),
        "js": float(jensenshannon(p, q)),
        "ks_stat": ks_stat, "ks_pvalor": ks_p,
        "chi2_stat": chi2_stat, "chi2_pvalor": chi2_p,
    }


def nivel_alerta(psi):
    if psi < 0.10:
        return "estable"
    if psi < 0.25:
        return "alerta"
    return "critico"


def monitorear(df, col_fecha="fecha_prestamo", meses_referencia=12, n_minimo=100):
    """Muestreo periódico mensual: los primeros `meses_referencia` meses son la
    población de referencia; cada mes posterior se compara contra ella."""
    df = df.copy()
    df["mes"] = df[col_fecha].dt.to_period("M")
    meses = sorted(df["mes"].unique())

    ref = df[df["mes"].isin(meses[:meses_referencia])]
    variables = [c for c in df.columns if c not in (col_fecha, "mes", "Pago_atiempo")]

    filas = []
    for mes in meses[meses_referencia:]:
        actual = df[df["mes"] == mes]
        if len(actual) < n_minimo:
            continue
        for col in variables:
            r = drift_variable(ref[col], actual[col], col in CATEGORICAS)
            if r is None:
                continue
            filas.append({"mes": str(mes), "variable": col, "n_mes": len(actual),
                          **r, "alerta": nivel_alerta(r["psi"])})

    return pd.DataFrame(filas)


if __name__ == "__main__":
    crudo = cargar_datos()
    fechas = crudo["fecha_prestamo"]          # se preserva para el monitoreo
    df = crear_features(limpiar(crudo))
    df["fecha_prestamo"] = fechas             # el índice se conserva en la limpieza

    metricas = monitorear(df)
    salida = RAIZ / "data" / "processed" / "drift_metrics.csv"
    metricas.to_csv(salida, index=False)

    print(f"Métricas guardadas en {salida}  ({len(metricas)} filas)")
    print("\nVariables con más meses en nivel crítico:")
    print(metricas[metricas["alerta"] == "critico"]["variable"]
          .value_counts().head(10))
    print("\nPSI promedio por mes (drift global):")
    print(metricas.groupby("mes")["psi"].mean().round(3))