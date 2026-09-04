
# Proyecto Integrador — Riesgo Crediticio

Pipeline completo de machine learning para predecir el pago a tiempo de créditos:
desde el análisis exploratorio hasta el monitoreo de data drift en operación.

## Caso de negocio

Una entidad de crédito necesita anticipar qué clientes no pagarán a tiempo para
priorizar la gestión de cobranza preventiva. El evento es raro (~4.75% de los
10,763 créditos históricos), por lo que el problema se trata como clasificación
severamente desbalanceada: las métricas y los modelos se eligen en función de la
clase minoritaria, no del accuracy global.

## Estructura del repositorio

| Componente | Rol |
|---|---|
| `notebooks/cargar_datos.ipynb` | Carga del dataset (simula la extracción del DWH) |
| `notebooks/comprension_eda.ipynb` | EDA: calidad de datos, reglas de validación, transformaciones |
| `notebooks/modelamiento.ipynb` | Entrenamiento y evaluación comparativa de 4 modelos |
| `src/ft_engineering.py` | Limpieza, features derivadas, preprocesador (pipelines) y split estratificado |
| `src/model_monitoring.py` | Métricas de data drift (KS, PSI, Jensen-Shannon, chi²) con muestreo mensual |
| `app.py` | Dashboard Streamlit de monitoreo |

## Principales hallazgos

**EDA.** El dataset presenta nulos centinela (edades 121–123, puntaje de central
en 0), una variable casi constante (`puntaje`, eliminada), una columna categórica
contaminada con valores numéricos (`tendencia_ingresos`) y un bloque de ~27% de
faltantes proveniente de la central de riesgo, cuya ausencia demostró no ser
informativa respecto del objetivo. El aparente deterioro temporal de la tasa de
pago resultó ser confusión por mezcla de productos (ρ de Spearman = −0.45 entre
plazo mediano mensual y tasa de pago), por lo que se descartaron variables
temporales como predictoras.

**Modelamiento.** Se compararon regresión logística, árbol de decisión, random
forest y gradient boosting dentro de pipelines (imputación y escalado ajustados
solo con entrenamiento). El modelo seleccionado es la **regresión logística con
`class_weight="balanced"`**: mejor F1 sobre la clase de interés (0.136), recall
de 0.57 y AUC-ROC de 0.67, además de interpretabilidad (coeficientes como
log-odds). Random forest mostró el mejor AUC-PR (0.138), lo que sugiere que con
ajuste de umbral podría superarla; queda como mejora identificada.

**Monitoreo.** El sistema compara cada mes contra los primeros 12 meses
(población de entrenamiento). Hallazgo principal: **drift creciente en las
variables monetarias** — `promedio_ingresos_datacredito` en nivel crítico los
3 meses monitoreados, seguida de `total_otros_prestamos` y `capital_prestado`.
El PSI global pasó de 0.140 (2025-11) a 0.218 (2026-01), acercándose al umbral
crítico de 0.25: la población que llega hoy ya no es la que el modelo conoció,
y el dashboard recomienda reentrenamiento.

## Cómo ejecutar

```bash
# entorno
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# pipeline de datos y monitoreo
python src/ft_engineering.py    # genera train.csv / test.csv
python src/model_monitoring.py  # genera drift_metrics.csv

# dashboard
streamlit run app.py
```

## Versionamiento

Flujo de ramas `developer` → `main` mediante pull requests con aprobación de
revisor. Versiones: `V1.0.0` (estructura), `V1.0.1` (carga + EDA), `V1.1.0`
(features + modelamiento), `V1.2.0` (monitoreo + dashboard).