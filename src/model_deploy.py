"""Despliegue del modelo — API de predicción por lotes con FastAPI.

Uso:
    python src/model_deploy.py --entrenar   # entrena y guarda models/modelo.joblib
    uvicorn src.model_deploy:app --port 8000  # levanta la API

Endpoints:
    GET  /salud    -> estado del servicio
    POST /predict  -> {"registros": [{...}, {...}]} devuelve predicción y
                      probabilidad de no pago por registro (batch).
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.ft_engineering import (cargar_datos, limpiar, crear_features,
                                construir_preprocesador, split_datos, OBJETIVO)

RUTA_MODELO = RAIZ / "models" / "modelo.joblib"


def entrenar_y_guardar():
    """Entrena el modelo seleccionado (logística balanceada) y serializa el
    pipeline completo (preprocesador + clasificador) junto con las columnas
    esperadas, para que la API pueda validar y ordenar la entrada."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    crudo = cargar_datos()
    columnas_crudas = [c for c in crudo.columns if c != OBJETIVO]

    df = crear_features(limpiar(crudo))
    X_train, X_test, y_train, y_test = split_datos(df)

    pipe = Pipeline([
        ("preprocesador", construir_preprocesador(X_train)),
        ("clasificador", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)

    RUTA_MODELO.parent.mkdir(exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "columnas_crudas": columnas_crudas,
        "columnas_modelo": list(X_train.columns),
    }, RUTA_MODELO)
    print(f"Modelo guardado en {RUTA_MODELO}")
    print(f"Score de prueba (accuracy de referencia): {pipe.score(X_test, y_test):.4f}")


# ----------------------------- API -----------------------------
app = FastAPI(title="API de riesgo crediticio",
              description="Predicción por lotes de Pago_atiempo")

_estado = {}


@app.on_event("startup")
def cargar_modelo():
    if RUTA_MODELO.exists():
        _estado.update(joblib.load(RUTA_MODELO))


def preparar(registros):
    """Convierte el batch JSON en la matriz que el pipeline espera:
    mismas columnas crudas (faltantes -> NaN, extras se descartan),
    misma limpieza y mismas features que en entrenamiento."""
    df = pd.DataFrame(registros).reindex(columns=_estado["columnas_crudas"])
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")
    df = crear_features(limpiar(df))
    return df.reindex(columns=_estado["columnas_modelo"])


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": bool(_estado)}


@app.post("/predict")
def predict(payload: dict):
    if not _estado:
        raise HTTPException(503, "Modelo no disponible: ejecutar --entrenar primero")
    registros = payload.get("registros")
    if not isinstance(registros, list) or not registros:
        raise HTTPException(400, "Formato esperado: {'registros': [{...}, {...}]}")

    X = preparar(registros)
    pipe = _estado["pipeline"]
    proba_no_pago = pipe.predict_proba(X)[:, list(pipe.classes_).index(0)]
    predicciones = pipe.predict(X)

    return {
        "n_registros": len(registros),
        "resultados": [
            {"pago_atiempo_pred": int(p), "prob_no_pago": round(float(q), 4)}
            for p, q in zip(predicciones, proba_no_pago)
        ],
    }


if __name__ == "__main__":
    if "--entrenar" in sys.argv:
        entrenar_y_guardar()
    else:
        print("Uso: python src/model_deploy.py --entrenar  |  "
              "uvicorn src.model_deploy:app --port 8000")
