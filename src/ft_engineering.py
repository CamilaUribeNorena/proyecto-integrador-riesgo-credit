"""Ingeniería de características — riesgo crediticio.

Primera componente del flujo de modelos: toma el dataset crudo,
aplica la limpieza definida en el EDA y genera las features derivadas.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CRUDO = RAIZ / "data" / "raw" / "base_datos.xlsx"


def cargar_datos(ruta: Path = RUTA_CRUDO) -> pd.DataFrame:
    """Lee el dataset crudo. En producción, esta función consultaría el DWH."""
    return pd.read_excel(ruta)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza derivadas del EDA (secciones 1.3–1.5)."""
    df = df.copy()

    # Centinelas -> NaN
    df.loc[df["edad_cliente"] >= 120, "edad_cliente"] = np.nan
    df.loc[df["puntaje_datacredito"] == 0, "puntaje_datacredito"] = np.nan

    # Nulo estructural: sin codeudor -> sin mora de codeudor
    df["saldo_mora_codeudor"] = df["saldo_mora_codeudor"].fillna(0)

    # tendencia_ingresos: valores numericos corruptos -> NaN
    es_numero = pd.to_numeric(df["tendencia_ingresos"], errors="coerce").notna()
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(~es_numero)

    # Variables sin capacidad discriminante o no predictoras
    df = df.drop(columns=["puntaje", "fecha_prestamo"])

    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los atributos derivados candidatos (EDA seccion 5.3)."""
    df = df.copy()

    # Carga de la cuota sobre el ingreso (proxy de capacidad de pago)
    df["carga_cuota"] = df["cuota_pactada"] / df["salario_cliente"].replace(0, np.nan)

    # Proporcion del endeudamiento en mora
    df["prop_mora"] = df["saldo_mora"] / df["saldo_total"].replace(0, np.nan)

    # Exposicion total en centrales
    df["total_creditos_sectores"] = (
        df["creditos_sectorFinanciero"]
        + df["creditos_sectorCooperativo"]
        + df["creditos_sectorReal"]
    )

    return df


if __name__ == "__main__":
    df = cargar_datos()
    print("Crudo:", df.shape)
    df = limpiar(df)
    print("Limpio:", df.shape)
    df = crear_features(df)
    print("Con features:", df.shape)
    print(df[["carga_cuota", "prop_mora", "total_creditos_sectores"]].describe().round(3))