#Es el espacio principal de la aplicación, para poder iniciar la aplicación
#es necesario de ejecutar el comando: streamlit run app.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st #libreria en linea y que corre directamente en el navegador

from config import PredictionConfig
from pipeline import clean, progress_callback, read_csv, run_pipeline, save_predictions

st.set_page_config(page_title="Productividad agrícola · Sucre", page_icon="🌾", layout="wide")
st.title("🌾 Predicción de rendimiento agrícola — Sucre")
st.caption("Perceptrón multicapa (TensorFlow) · entrenamiento 2021-2024 · predicción 2025")

# ------------------------------------------------------------------ 1) configuración
with st.sidebar:
    st.header("⚙️ Configuración del modelo")
    layers_txt = st.text_input("Neuronas por capa oculta", "128,64,32", help="Separadas por coma")
    activation = st.selectbox("Activación", ["relu", "elu", "selu", "tanh"])
    dropout = st.slider("Dropout", 0.0, 0.6, 0.2, 0.05)
    l2 = st.select_slider("Regularización L2", [0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
    lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
    epochs = st.number_input("Épocas máximas", 10, 2000, 300, 10)
    batch = st.select_slider("Batch size", [16, 32, 64, 128], value=32)
    patience = st.number_input("Paciencia (early stopping)", 3, 100, 25)
    val_strategy = st.radio("Validación", ["temporal", "random"],
                            format_func={"temporal": "Temporal (último año del histórico)",
                                            "random": "Aleatoria (15 %)"}.get)
    refit = st.checkbox("Reentrenar con todo el histórico", True)
    log_target = st.checkbox("Aplicar log1p al rendimiento", True)
    seed = st.number_input("Semilla", 0, 9999, 42)

try:
    hidden = tuple(int(x) for x in layers_txt.split(",") if x.strip())
    assert hidden and all(h > 0 for h in hidden)
except Exception:
    st.sidebar.error("Capas inválidas. Ejemplo: 128,64,32")
    st.stop()

cfg = PredictionConfig(hidden_layers=hidden, activation=activation, dropout=dropout, l2=l2,
                        learning_rate=lr, epochs=int(epochs), batch_size=int(batch),
                        patience=int(patience), val_strategy=val_strategy, refit_full=refit,
                        log_target=log_target, seed=int(seed))

tab_data, tab_run, tab_res, tab_dash = st.tabs(
    ["1 · Datos", "2 · Pipeline", "3 · Resultados (CSV)", "4 · Dashboard"])

# ------------------------------------------------------------------ datos
with tab_data:
    c1, c2 = st.columns(2)
    f_train = c1.file_uploader("CSV de entrenamiento (2021-2024)", type="csv", key="train")
    f_test = c2.file_uploader("CSV de test (2025)", type="csv", key="test")
    st.caption("Columnas esperadas: Año, Cultivo, Periodo (Anual / Semestre A / Semestre B / Permanente), "
                "Municipios, Subregión, Área Sembrada (Has). El CSV de entrenamiento además necesita "
                "Rendimiento promedio (Ton/Ha). En el de 2025, Rendimiento, Área Cosechada y Producción son "
                "opcionales (si vienen, se evalúa el modelo).")
    raw = {}
    for name, f, col in [("train", f_train, c1), ("test", f_test, c2)]:
        if f is None:
            continue
        try:
            raw[name] = read_csv(f)
            ok = clean(raw[name], require_target=(name == "train"))
            with col:
                st.success(f"{len(raw[name])} filas leídas · {len(ok)} válidas · "
                            f"años {int(ok['Anio'].min())}-{int(ok['Anio'].max())} · "
                            f"{ok['Municipio'].nunique()} municipios · {ok['Cultivo'].nunique()} cultivos")
                st.dataframe(raw[name].head(8), width="stretch")
        except Exception as e:
            col.error(str(e))

# ------------------------------------------------------------------ pipeline
with tab_run:
    ready = "train" in raw and "test" in raw
    if not ready:
        st.info("Carga ambos CSV en la pestaña **Datos**.")
    if st.button("🚀 Ejecutar pipeline", type="primary", disabled=not ready):
        bar, status = st.progress(0.0), st.empty()

        def on_epoch(ep, logs):
            bar.progress(min(ep / cfg.epochs, 1.0))
            status.text(f"Época {ep} · loss {logs['loss']:.4f} · val_loss {logs['val_loss']:.4f}")

        try:
            with st.spinner("Entrenando MLP y generando predicciones..."):
                st.session_state["result"] = run_pipeline(
                    raw["train"], raw["test"], cfg, callbacks=[progress_callback(on_epoch)])
            bar.progress(1.0)
            status.text("Listo ✅")
        except Exception as e:
            st.error(f"Error en el pipeline: {e}")

    res = st.session_state.get("result")
    if res:
        p = res["predictor"]
        st.subheader("Métricas")
        m = res["val_metrics"]
        cols = st.columns(4)
        cols[0].metric("Val. MAE (t/ha)", f"{m['MAE']:.3f}",
                        f"{m['MAE'] - res['val_baseline']['MAE']:+.3f} vs. baseline", delta_color="inverse")
        cols[1].metric("Val. RMSE", f"{m['RMSE']:.3f}")
        cols[2].metric("Val. R²", f"{m['R2']:.3f}")
        cols[3].metric("Mejor época", p.best_epoch)
        if res["test_metrics"]:
            t = res["test_metrics"]
            cols = st.columns(3)
            cols[0].metric("Test 2025 · MAE", f"{t['MAE']:.3f}",
                            f"{t['MAE'] - res['test_baseline']['MAE']:+.3f} vs. baseline", delta_color="inverse")
            cols[1].metric("Test 2025 · RMSE", f"{t['RMSE']:.3f}")
            cols[2].metric("Test 2025 · R²", f"{t['R2']:.3f}")
        st.caption("Baseline = promedio histórico de rendimiento por cultivo.")

        h = p.history
        fig = go.Figure()
        fig.add_scatter(y=h["loss"], name="Entrenamiento")
        fig.add_scatter(y=h["val_loss"], name="Validación")
        fig.update_layout(title="Curva de aprendizaje (MSE)", xaxis_title="Época", yaxis_title="Loss",
                            height=350, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ resultados
# en este apartado se pueden obtener los resultados de la predicción
with tab_res:
    res = st.session_state.get("result")
    if not res:
        st.info("Ejecuta el pipeline para generar el CSV de predicciones.")
    else:
        out = res["predictions"]
        st.dataframe(out, width="stretch", height=420)
        st.download_button("⬇️ Descargar predicciones (CSV)",
                            out.to_csv(index=False).encode("utf-8-sig"),
                            "predicciones_2025.csv", "text/csv")
        if st.button("💾 Guardar también en output/"):
            st.success(f"Guardado en {save_predictions(out)}")
        st.caption("Este mismo DataFrame alimenta el dashboard. Producción estimada = rendimiento "
                    "predicho × área cosechada (la del CSV si existe; si no, sembrada × razón histórica del cultivo).")

# ------------------------------------------------------------------ dashboard
def wyield(g):
    """Rendimiento ponderado = producción / área cosechada (usa la sembrada si el CSV es antiguo)."""
    col = "Area_Cosechada_Usada" if "Area_Cosechada_Usada" in g.columns else "Area_Sembrada"
    return g["Produccion_Estimada_t"].sum() / g[col].sum() if g[col].sum() else np.nan

with tab_dash:
    src = st.radio("Fuente de datos", ["Resultado de esta sesión", "Cargar un CSV de predicciones previo"],
                    horizontal=True)
    df = None
    if src.startswith("Resultado"):
        if "result" in st.session_state:
            df = st.session_state["result"]["predictions"]
    else:
        up = st.file_uploader("CSV generado por esta aplicación", type="csv", key="dash_up")
        if up:
            df = read_csv(up)

    need = ["Cultivo", "Municipio", "Subregion", "Area_Sembrada", "Rendimiento_Predicho", "Produccion_Estimada_t"]
    if df is None:
        st.info("Aún no hay predicciones para visualizar.")
    elif any(c not in df.columns for c in need):
        st.error(f"El CSV debe incluir: {need}")
    else:
        f1, f2 = st.columns(2)
        subs = f1.multiselect("Subregión", sorted(df["Subregion"].unique()), default=sorted(df["Subregion"].unique()))
        crops = f2.multiselect("Cultivo", sorted(df["Cultivo"].unique()), default=sorted(df["Cultivo"].unique()))
        d = df[df["Subregion"].isin(subs) & df["Cultivo"].isin(crops)]
        if d.empty:
            st.warning("Sin datos con esos filtros.")
            st.stop()

        k = st.columns(4)
        k[0].metric("Producción estimada (t)", f"{d['Produccion_Estimada_t'].sum():,.0f}")
        k[1].metric("Área sembrada (ha)", f"{d['Area_Sembrada'].sum():,.0f}")
        k[2].metric("Rendimiento ponderado (t/ha)", f"{wyield(d):.2f}")
        k[3].metric("Registros", f"{len(d):,}")

        a, b = st.columns(2)
        prod_crop = d.groupby("Cultivo", as_index=False)["Produccion_Estimada_t"].sum().sort_values("Produccion_Estimada_t")
        a.plotly_chart(px.bar(prod_crop, x="Produccion_Estimada_t", y="Cultivo", orientation="h",
                                title="Producción estimada por cultivo (t)"), width="stretch")
        prod_sub = d.groupby("Subregion", as_index=False)["Produccion_Estimada_t"].sum()
        b.plotly_chart(px.pie(prod_sub, values="Produccion_Estimada_t", names="Subregion", hole=0.35,
                                title="Participación subregional en la producción"), width="stretch")

        a, b = st.columns(2)
        a.plotly_chart(px.histogram(d, x="Rendimiento_Predicho", color="Subregion", nbins=30,
                                    title="Distribución del rendimiento predicho (t/ha)"), width="stretch")
        y_sub = d.groupby("Subregion").apply(wyield, include_groups=False).rename("Rendimiento").reset_index()
        b.plotly_chart(px.bar(y_sub.sort_values("Rendimiento"), x="Subregion", y="Rendimiento",
                                title="Productividad por subregión (t/ha ponderado por área)"), width="stretch")

        a, b = st.columns(2)
        a.plotly_chart(px.box(d, x="Cultivo", y="Rendimiento_Predicho", color="Cultivo",
                                title="Rendimiento predicho por cultivo").update_layout(showlegend=False),
                        width="stretch")
        top = d.groupby("Municipio", as_index=False)["Produccion_Estimada_t"].sum().nlargest(10, "Produccion_Estimada_t")
        b.plotly_chart(px.bar(top.sort_values("Produccion_Estimada_t"), x="Produccion_Estimada_t", y="Municipio",
                                orientation="h", title="Top 10 municipios por producción (t)"), width="stretch")

        st.plotly_chart(px.treemap(d, path=[px.Constant("Sucre"), "Subregion", "Municipio", "Cultivo"],
                                    values="Produccion_Estimada_t", title="Producción: subregión → municipio → cultivo"),
                        width="stretch")

        pivot = d.pivot_table(index="Municipio", columns="Cultivo", values="Rendimiento_Predicho", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * pivot.shape[1] + 3), max(4, 0.35 * pivot.shape[0] + 1)))
        sns.heatmap(pivot, cmap="YlGn", annot=pivot.shape[0] * pivot.shape[1] <= 200, fmt=".1f",
                    linewidths=0.4, cbar_kws={"label": "t/ha"}, ax=ax)
        ax.set_title("Rendimiento predicho promedio: municipio × cultivo")
        st.pyplot(fig)

        # histórico real vs. predicción (usa el histórico de la sesión, si existe)
        res = st.session_state.get("result")
        if res is not None:
            hist = res["predictor"].train_df
            hist = hist[hist["Cultivo"].isin(crops)]
            t1 = hist.groupby(["Anio", "Cultivo"], as_index=False)["Rendimiento"].mean().assign(Fuente="Histórico real")
            t2 = (d.groupby(["Anio", "Cultivo"], as_index=False)["Rendimiento_Predicho"].mean()
                    .rename(columns={"Rendimiento_Predicho": "Rendimiento"}).assign(Fuente="Predicción"))
            trend = pd.concat([t1, t2])
            st.plotly_chart(px.line(trend, x="Anio", y="Rendimiento", color="Cultivo", line_dash="Fuente",
                                    markers=True, title="Rendimiento promedio por año: histórico vs. predicción"
                                    ).update_xaxes(dtick=1), width="stretch")

        if "Rendimiento_Real" in d.columns and d["Rendimiento_Real"].notna().any():
            st.subheader("Evaluación contra valores reales de 2025")
            a, b = st.columns(2)
            lim = float(max(d["Rendimiento_Real"].max(), d["Rendimiento_Predicho"].max()))
            sc = px.scatter(d, x="Rendimiento_Real", y="Rendimiento_Predicho", color="Cultivo",
                            hover_data=["Municipio"], title="Real vs. predicho (t/ha)")
            sc.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="gray"))
            a.plotly_chart(sc, width="stretch")
            err = d.assign(Error=d["Rendimiento_Predicho"] - d["Rendimiento_Real"])
            b.plotly_chart(px.histogram(err, x="Error", nbins=40, title="Distribución del error (predicho − real)"),
                            width="stretch") 