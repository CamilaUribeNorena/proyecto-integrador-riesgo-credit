"""Dashboard de monitoreo de data drift — riesgo crediticio (Streamlit)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.append(str(RAIZ))

from src.ft_engineering import cargar_datos, limpiar, crear_features, CATEGORICAS
from src.model_monitoring import monitorear

st.set_page_config(page_title="Monitoreo de drift", layout="wide")

UMBRAL_ALERTA, UMBRAL_CRITICO = 0.10, 0.25
COLORES = {"estable": "🟢", "alerta": "🟡", "critico": "🔴"}


@st.cache_data
def cargar_todo():
    crudo = cargar_datos()
    fechas = crudo["fecha_prestamo"]
    df = crear_features(limpiar(crudo))
    df["fecha_prestamo"] = fechas
    df["mes"] = df["fecha_prestamo"].dt.to_period("M").astype(str)
    metricas = pd.read_csv(RAIZ / "data" / "processed" / "drift_metrics.csv")
    return df, metricas


df, metricas = cargar_todo()
meses_monitoreados = sorted(metricas["mes"].unique())
meses_ref = sorted(df["mes"].unique())[:12]

st.title("Monitoreo del modelo — data drift")
st.caption(
    f"Referencia: {meses_ref[0]} a {meses_ref[-1]} (población de entrenamiento). "
    f"Monitoreo mensual posterior. Umbrales PSI: <{UMBRAL_ALERTA} estable, "
    f"{UMBRAL_ALERTA}–{UMBRAL_CRITICO} alerta, >{UMBRAL_CRITICO} crítico.")

# ---------------- barra lateral
mes_sel = st.sidebar.selectbox("Mes a analizar", meses_monitoreados,
                               index=len(meses_monitoreados) - 1)
m_mes = metricas[metricas["mes"] == mes_sel].set_index("variable")

# ---------------- indicadores globales
psi_prom = m_mes["psi"].mean()
n_criticas = (m_mes["alerta"] == "critico").sum()
estado = ("🔴 CRÍTICO" if psi_prom > UMBRAL_CRITICO or n_criticas >= 3
          else "🟡 ALERTA" if psi_prom > UMBRAL_ALERTA or n_criticas >= 1
          else "🟢 ESTABLE")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estado global", estado)
c2.metric("PSI promedio", f"{psi_prom:.3f}")
c3.metric("Variables críticas", int(n_criticas))
c4.metric("Registros del mes", int(m_mes["n_mes"].iloc[0]))

# ---------------- recomendaciones automáticas
if estado.startswith("🔴"):
    st.error(
        "**Drift crítico.** La población actual difiere sustancialmente de la de "
        "entrenamiento. Recomendación: reentrenar el modelo con datos recientes y "
        "revisar las variables críticas listadas abajo antes de confiar en las "
        "predicciones de este período.")
elif estado.startswith("🟡"):
    st.warning(
        "**Drift moderado.** Hay variables desplazándose respecto de la referencia. "
        "Recomendación: vigilar la tendencia; si persiste el próximo período, "
        "planificar reentrenamiento y validar el desempeño sobre datos recientes.")
else:
    st.success("**Población estable.** No se requiere acción.")

# ---------------- tabla semáforo por variable
st.subheader(f"Métricas de drift por variable — {mes_sel}")
tabla = m_mes[["psi", "js", "ks_stat", "ks_pvalor",
               "chi2_stat", "chi2_pvalor", "alerta"]].copy()
tabla["semáforo"] = tabla["alerta"].map(COLORES)
tabla = tabla.sort_values("psi", ascending=False).round(4)
st.dataframe(tabla, use_container_width=True)

# ---------------- evolución temporal
st.subheader("Evolución del drift en el tiempo")
evol = metricas.groupby("mes")["psi"].mean().reset_index()
fig = px.line(evol, x="mes", y="psi", markers=True,
              title="PSI promedio por mes (drift global)")
fig.add_hline(y=UMBRAL_ALERTA, line_dash="dash", line_color="orange")
fig.add_hline(y=UMBRAL_CRITICO, line_dash="dash", line_color="red")
st.plotly_chart(fig, use_container_width=True)

top_vars = (metricas.groupby("variable")["psi"].mean()
            .sort_values(ascending=False).head(6).index)
fig2 = px.line(metricas[metricas["variable"].isin(top_vars)],
               x="mes", y="psi", color="variable", markers=True,
               title="PSI por mes — 6 variables con mayor drift promedio")
fig2.add_hline(y=UMBRAL_CRITICO, line_dash="dash", line_color="red")
st.plotly_chart(fig2, use_container_width=True)

# ---------------- distribución histórica vs actual
st.subheader("Distribución: referencia vs mes actual")
var_sel = st.selectbox("Variable", sorted(m_mes.index))
ref = df[df["mes"].isin(meses_ref)]
act = df[df["mes"] == mes_sel]

if var_sel in CATEGORICAS:
    comp = pd.concat([
        ref[var_sel].value_counts(normalize=True).rename("referencia"),
        act[var_sel].value_counts(normalize=True).rename(mes_sel),
    ], axis=1).fillna(0).reset_index(names=var_sel)
    fig3 = px.bar(comp.melt(id_vars=var_sel, var_name="período",
                            value_name="proporción"),
                  x=var_sel, y="proporción", color="período", barmode="group")
else:
    largo = pd.concat([
        ref[[var_sel]].assign(período="referencia"),
        act[[var_sel]].assign(período=str(mes_sel)),
    ])
    fig3 = px.histogram(largo, x=var_sel, color="período", barmode="overlay",
                        histnorm="percent", nbins=40, opacity=0.6)
st.plotly_chart(fig3, use_container_width=True)
psi_var = m_mes.loc[var_sel, "psi"]
st.caption(f"PSI de `{var_sel}` en {mes_sel}: **{psi_var:.3f}** "
           f"({COLORES[m_mes.loc[var_sel, 'alerta']]} {m_mes.loc[var_sel, 'alerta']})")