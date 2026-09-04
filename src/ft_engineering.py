"""Ingeniería de características — riesgo crediticio.

Primera componente del flujo de modelos: toma el dataset crudo,
aplica la limpieza definida en el EDA, genera las features derivadas
y produce los conjuntos de entrenamiento y evaluación.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CRUDO = RAIZ / "data" / "raw" / "base_datos.xlsx"

OBJETIVO = "Pago_atiempo"
CATEGORICAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]
MONETARIAS = [
    "capital_prestado", "salario_cliente", "cuota_pactada",
    "total_otros_prestamos", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor",
    "promedio_ingresos_datacredito", "carga_cuota",
]


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


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """Preprocesador del EDA (seccion 5.2): imputacion, log1p en montos,
    escalado y one-hot. Se ajusta SOLO con train (dentro del pipeline del
    modelo) para evitar fuga de datos."""
    monetarias = [c for c in MONETARIAS if c in X.columns]
    categoricas = [c for c in CATEGORICAS if c in X.columns]
    otras_num = [c for c in X.columns if c not in monetarias + categoricas]

    pipe_monetarias = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escalar", StandardScaler()),
    ])
    pipe_numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    pipe_categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("monetarias", pipe_monetarias, monetarias),
        ("numericas", pipe_numericas, otras_num),
        ("categoricas", pipe_categoricas, categoricas),
    ])


def split_datos(df: pd.DataFrame, test_size: float = 0.2, semilla: int = 42):
    """Split estratificado por el objetivo (desbalance ~20:1)."""
    X = df.drop(columns=[OBJETIVO])
    y = df[OBJETIVO].astype(int)
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=semilla)


if __name__ == "__main__":
    df = cargar_datos()
    df = crear_features(limpiar(df))
    X_train, X_test, y_train, y_test = split_datos(df)

    salida = RAIZ / "data" / "processed"
    X_train.assign(**{OBJETIVO: y_train}).to_csv(salida / "train.csv", index=False)
    X_test.assign(**{OBJETIVO: y_test}).to_csv(salida / "test.csv", index=False)

    print("Train:", X_train.shape, "| tasa de pago:", round(y_train.mean(), 4))
    print("Test: ", X_test.shape, "| tasa de pago:", round(y_test.mean(), 4))