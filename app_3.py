"""
GeoMarket-VLC · Plataforma de Inteligencia Comercial Urbana para Valencia
Asignatura: Evaluación, Despliegue y Monitorización de Modelos (EDM)
"""
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
import shap

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GENERAL
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GeoMarket-VLC · Inteligencia Comercial Urbana",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta urbanística profesional
COLORS = {
    "bg_primary":    "#0F1B2D",   # azul muy oscuro
    "bg_secondary":  "#1A2B45",
    "bg_card":       "#FFFFFF",
    "bg_subtle":     "#F4F6FA",
    "accent":        "#E8833A",   # naranja terracota (acento)
    "accent_dark":   "#C2641E",
    "neutral":       "#5C6B7F",
    "border":        "#D8DEE8",
    "text_primary":  "#0F1B2D",
    "text_secondary":"#5C6B7F",
    "success":       "#2D7D5A",
    "warning":       "#D4A04C",
    "danger":        "#B23B3B",
}

# CSS personalizado
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['bg_primary']} 0%, {COLORS['bg_secondary']} 100%);
    }}
    [data-testid="stSidebar"] * {{ color: #E8EDF5 !important; }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: rgba(255,255,255,0.08);
        border-color: rgba(255,255,255,0.15);
    }}

    /* Header del proyecto */
    .project-header {{
        background: linear-gradient(135deg, {COLORS['bg_primary']} 0%, {COLORS['bg_secondary']} 100%);
        color: white;
        padding: 28px 36px;
        border-radius: 12px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }}
    .project-header::before {{
        content: "";
        position: absolute;
        right: -40px; top: -40px;
        width: 200px; height: 200px;
        background: {COLORS['accent']};
        opacity: 0.08;
        border-radius: 50%;
    }}
    .project-header h1 {{
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 6px 0;
        color: white;
    }}
    .project-header .subtitle {{
        font-size: 14px;
        color: rgba(255,255,255,0.7);
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    .project-header .tagline {{
        font-size: 15px;
        margin-top: 12px;
        color: rgba(255,255,255,0.85);
        max-width: 720px;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {COLORS['bg_subtle']};
        padding: 6px;
        border-radius: 10px;
        border: 1px solid {COLORS['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 15px;
        font-weight: 600;
        padding: 10px 22px;
        background: transparent;
        border-radius: 8px;
        color: {COLORS['text_secondary']};
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['bg_primary']} !important;
        color: white !important;
    }}

    /* Métricas */
    [data-testid="stMetric"] {{
        background: white;
        padding: 18px 22px;
        border-radius: 10px;
        border: 1px solid {COLORS['border']};
        border-left: 4px solid {COLORS['accent']};
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {COLORS['text_secondary']} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 26px !important;
        font-weight: 700 !important;
        color: {COLORS['text_primary']} !important;
    }}

    /* Section headers */
    h2 {{
        color: {COLORS['text_primary']};
        font-size: 22px !important;
        font-weight: 700;
        border-bottom: 2px solid {COLORS['bg_primary']};
        padding-bottom: 8px;
        margin-top: 24px !important;
    }}
    h3 {{
        color: {COLORS['text_primary']};
        font-size: 17px !important;
        font-weight: 600;
    }}

    /* Cards */
    .info-card {{
        background: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid {COLORS['border']};
        margin-bottom: 16px;
    }}

    .footer {{
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid {COLORS['border']};
        text-align: center;
        color: {COLORS['text_secondary']};
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
    }}

    /* Botones */
    .stButton button {{
        background: {COLORS['bg_primary']};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }}
    .stButton button:hover {{
        background: {COLORS['accent']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS Y MODELOS (caché)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_datos():
    df = pd.read_csv("geomarket_vlc_features.csv", encoding="utf-8-sig")
    gdf = gpd.read_file("geomarket_vlc_barrios.geojson")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf["coddistbar"] = pd.to_numeric(gdf["coddistbar"], errors="coerce").astype("Int64")
    df["coddistbar"]  = pd.to_numeric(df["coddistbar"], errors="coerce").astype("Int64")
    return df, gdf


@st.cache_resource
def cargar_modelos():
    modelos = {
        "rf":         joblib.load("models/modelo_rf.pkl"),
        "xgb":        joblib.load("models/modelo_xgb.pkl"),
        "stacking":   joblib.load("models/modelo_stacking.pkl"),
        "explainer":  joblib.load("models/shap_explainer.pkl"),
        "features":   joblib.load("models/feature_names.pkl"),
    }
    return modelos


# Etiquetas amigables para sectores comerciales
SECTOR_LABELS = {
    "bakery": "Panadería",
    "bank": "Banco / Sucursal",
    "bar": "Bar",
    "cafe": "Cafetería",
    "clothes": "Ropa / Moda",
    "convenience": "Tienda de conveniencia",
    "fast_food": "Comida rápida",
    "hairdresser": "Peluquería",
    "health_care": "Servicios de salud",
    "home_craft_repairs": "Reformas / Bricolaje",
    "leisure_education": "Ocio / Educación",
    "pharmacy": "Farmacia",
    "professional_services": "Servicios profesionales",
    "restaurant": "Restaurante",
    "retail_commerce": "Comercio minorista",
    "supermarket_food": "Supermercado / Alimentación",
}

# Carga inicial
try:
    df, gdf = cargar_datos()
    modelos = cargar_modelos()
    APP_OK = True
except Exception as e:
    APP_OK = False
    error_msg = str(e)


# ════════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════
def calcular_score_emprendedor(df_input, sector, pesos):
    """Score multi-criterio 0-100 para abrir un negocio del sector dado."""
    col_sector = f"n_locales_{sector}"
    if col_sector not in df_input.columns:
        return pd.DataFrame()

    out = df_input[["nombre", "codbarrio", "coddistbar", "coddistrit",
                    col_sector, "poblacion", "accesibilidad_tp", "Ind_Global"]].copy()
    out.columns = ["nombre", "codbarrio", "coddistbar", "coddistrit",
                   "competencia", "poblacion", "accesibilidad", "vulnerabilidad"]

    out_valid = out[out["poblacion"] > 0].copy()
    scaler = MinMaxScaler()

    out_valid["competencia_n"]   = scaler.fit_transform(out_valid[["competencia"]])
    out_valid["demanda_n"]       = scaler.fit_transform(out_valid[["poblacion"]])
    out_valid["accesibilidad_n"] = scaler.fit_transform(out_valid[["accesibilidad"]])
    out_valid["vuln_n"] = scaler.fit_transform(
        out_valid[["vulnerabilidad"]].fillna(out_valid["vulnerabilidad"].mean())
    )

    out_valid["score"] = (
        pesos["competencia"]   * (1 - out_valid["competencia_n"]) +
        pesos["demanda"]       * out_valid["demanda_n"] +
        pesos["accesibilidad"] * out_valid["accesibilidad_n"] +
        pesos["calidad"]       * (1 - out_valid["vuln_n"])
    ) * 100

    return out_valid.sort_values("score", ascending=False)


def crear_mapa_coropletico(gdf, df_valores, columna_valor, columna_etiqueta="nombre",
                            colorscale="YlOrRd", titulo_leyenda="", revertir=False):
    """Mapa choropleth de Valencia con folium."""
    gdf_plot = gdf.merge(df_valores[["coddistbar", columna_valor]],
                         on="coddistbar", how="left", suffixes=("", "_v"))
    if f"{columna_valor}_v" in gdf_plot.columns:
        gdf_plot[columna_valor] = gdf_plot[f"{columna_valor}_v"]

    gdf_plot["coddistbar"] = gdf_plot["coddistbar"].fillna(0).astype(int)
    geo_data = json.loads(gdf_plot.to_json())

    m = folium.Map(
        location=[39.4699, -0.3763],
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Choropleth
    folium.Choropleth(
        geo_data=geo_data,
        data=gdf_plot,
        columns=["coddistbar", columna_valor],
        key_on="feature.properties.coddistbar",
        fill_color=colorscale,
        fill_opacity=0.78,
        line_opacity=0.5,
        line_color="white",
        line_weight=1,
        legend_name=titulo_leyenda,
        nan_fill_color="#E0E0E0",
        nan_fill_opacity=0.5,
        highlight=True,
    ).add_to(m)

    # Tooltip overlay
    folium.GeoJson(
        geo_data,
        style_function=lambda x: {"fillOpacity": 0, "color": "#444", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=[columna_etiqueta, columna_valor],
            aliases=["Barrio:", f"{titulo_leyenda}:"],
            sticky=True,
            style="background-color: white; color: #0F1B2D; font-family: Inter; "
                  "font-size: 12px; padding: 6px;"
        )
    ).add_to(m)

    return m


def grafico_radar_comparativo(df, nombre_barrio):
    """Gráfico radar que superpone el barrio seleccionado (color acento) sobre
    la media de todos los barrios (relleno gris).
    """
    # Cambiamos la búsqueda para que sea por la columna 'nombre'
    row = df[df["nombre"] == nombre_barrio].iloc[0] if (df["nombre"] == nombre_barrio).any() else None
    if row is None:
        return None

    dims_def = {
        "Comercio total":      ("n_locales_total",            False, "locales"),
        "Conexión transporte": ("accesibilidad_tp",            False, "pts accesibilidad"),
        "Equipamiento":        ("n_equipamiento_municipal",    False, "equipamientos"),
        "Demanda (pob.)":      ("poblacion",                   False, "hab."),
        "Calidad zona":        ("Ind_Global",                  True,  "índice (invertido)"),
        "Presión turística":   ("indice_turistico",            False, "índice turístico"),
    }
    dims_def = {k: v for k, v in dims_def.items() if v[0] in df.columns}

    vals_barrio = []
    vals_media  = []
    hover_barrio = []
    hover_media  = []

    for label, (col, invertir, unidad) in dims_def.items():
        serie   = df[col].fillna(df[col].mean())
        s_min, s_max = serie.min(), serie.max()
        rango = s_max - s_min if s_max != s_min else 1

        # Valor del barrio
        val_real = row[col] if pd.notna(row[col]) else serie.mean()
        norm_b = (val_real - s_min) / rango * 100
        if invertir:
            norm_b = 100 - norm_b
        norm_b = float(np.clip(norm_b, 0, 100))
        vals_barrio.append(norm_b)
        hover_barrio.append(f"{val_real:,.2f} {unidad}" if invertir else f"{val_real:,.0f} {unidad}")

        # Media del dataset
        val_mean = serie.mean()
        norm_m = (val_mean - s_min) / rango * 100
        if invertir:
            norm_m = 100 - norm_m
        norm_m = float(np.clip(norm_m, 0, 100))
        vals_media.append(norm_m)
        hover_media.append(f"{val_mean:,.2f} {unidad}" if invertir else f"{val_mean:,.0f} {unidad}")

    labels = list(dims_def.keys())

    fig = go.Figure()

    # Traza media (gris de fondo)
    fig.add_trace(go.Scatterpolar(
        r=vals_media + [vals_media[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(92, 107, 127, 0.18)",
        line=dict(color="rgba(92, 107, 127, 0.55)", width=2, dash="dot"),
        name="Media Valencia",
        hovertemplate="<b>%{theta}</b><br>Media: %{customdata}<extra></extra>",
        customdata=hover_media + [hover_media[0]],
    ))

    # Traza barrio seleccionado (acento naranja)
    fig.add_trace(go.Scatterpolar(
        r=vals_barrio + [vals_barrio[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(232, 131, 58, 0.22)",
        line=dict(color=COLORS["accent"], width=2.5),
        name=row["nombre"],
        hovertemplate="<b>%{theta}</b><br>%{customdata}<extra></extra>",
        customdata=hover_barrio + [hover_barrio[0]],
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["mín", "25%", "50%", "75%", "máx"],
                tickfont=dict(size=9, color="#5C6B7F"),
                gridcolor="#E8EDF5",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS["text_primary"]),
                gridcolor="#E8EDF5",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.12,
            font=dict(size=11, color=COLORS["text_primary"]),
        ),
        height=360,
        margin=dict(l=50, r=50, t=30, b=50),
        paper_bgcolor="white",
    )
    return fig

def grafico_radar_barrio(df, codbarrio, sector=None):
    """Gráfico radar con los KPIs del barrio.
    
    Cada eje se normaliza de forma independiente (0 = mínimo del dataset,
    100 = máximo del dataset para ESA variable), de modo que un barrio con
    el valor máximo en cualquier dimensión llega siempre al borde exterior,
    independientemente de la escala real de cada variable.
    """
    row = df[df["codbarrio"] == codbarrio].iloc[0] if (df["codbarrio"] == codbarrio).any() else None
    if row is None:
        return None

    # Definir dimensiones: (columna_df, invertir, label_hover)
    # invertir=True cuando menor valor = mejor (ej: vulnerabilidad)
    dims_def = {
        "Comercio total":      ("n_locales_total",             False, "locales"),
        "Conexión transporte": ("accesibilidad_tp",            False, "pts accesibilidad"),
        "Equipamiento":        ("n_equipamiento_municipal",    False, "equipamientos"),
        "Demanda (pob.)":      ("poblacion",                   False, "hab."),
        "Calidad zona":        ("Ind_Global",                  True,  "índice (invertido)"),
        "Presión turística":   ("indice_turistico",            False, "índice turístico"),
    }
    dims_def = {k: v for k, v in dims_def.items() if v[0] in df.columns}

    valores_norm  = []   # valor normalizado 0-100 para cada eje
    valores_hover = []   # texto con el valor real para el tooltip

    for label, (col, invertir, unidad) in dims_def.items():
        serie = df[col].fillna(df[col].mean())
        s_min, s_max = serie.min(), serie.max()
        val_real = row[col] if pd.notna(row[col]) else serie.mean()

        if s_max - s_min == 0:
            norm = 50.0
        else:
            norm = (val_real - s_min) / (s_max - s_min) * 100
            if invertir:
                norm = 100 - norm   # invertir: menor Ind_Global → mejor → más hacia el exterior

        norm = float(np.clip(norm, 0, 100))
        valores_norm.append(norm)

        # Texto de hover: valor real + rango del dataset
        if invertir:
            hover_txt = f"{val_real:.2f} {unidad}<br>rango dataset: [{s_min:.2f}, {s_max:.2f}]"
        else:
            hover_txt = f"{val_real:,.0f} {unidad}<br>rango dataset: [{s_min:,.0f}, {s_max:,.0f}]"
        valores_hover.append(hover_txt)

    labels = list(dims_def.keys())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores_norm + [valores_norm[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(232, 131, 58, 0.25)",
        line=dict(color=COLORS["accent"], width=2.5),
        name=row["nombre"],
        hovertemplate="<b>%{theta}</b><br>%{customdata}<extra></extra>",
        customdata=valores_hover + [valores_hover[0]],
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["mín", "25%", "50%", "75%", "máx"],
                tickfont=dict(size=9, color="#5C6B7F"),
                gridcolor="#E8EDF5",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS["text_primary"]),
                gridcolor="#E8EDF5",
            ),
        ),
        showlegend=False,
        height=320,
        margin=dict(l=50, r=50, t=30, b=30),
        paper_bgcolor="white",
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="project-header">
    <div class="subtitle">⏐ GeoMarket-VLC · Decision Support System</div>
    <h1>Inteligencia Comercial Urbana de Valencia</h1>
    <div class="tagline">
        Plataforma analítica para diagnosticar la oferta comercial de los barrios,
        recomendar localizaciones a emprendedores y orientar políticas de equidad
        territorial al planificador urbano.
    </div>
</div>
""", unsafe_allow_html=True)

if not APP_OK:
    st.error(f"❌ Error cargando datos o modelos: {error_msg}")
    st.info("Asegúrate de que los archivos `geomarket_vlc_features.csv`, "
            "`geomarket_vlc_barrios.geojson` y la carpeta `models/` estén en el directorio.")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⏐ Panel de control")
    st.markdown("---")
    st.markdown(f"**Barrios analizados:** `{len(df)}`")
    st.markdown(f"**Distritos:** `{df['coddistrit'].nunique()}`")
    n_locales = int(df["n_locales_total"].sum())
    st.markdown(f"**Locales geolocalizados:** `{n_locales:,}`")
    st.markdown("---")
    st.markdown("### Stack analítico")
    st.markdown(
        "- Clasificador: Random Forest + XGBoost  \n"
        "- Ensemble: Stacking + LogReg  \n"
        "- Explicabilidad: SHAP  \n"
        "- Score multi-criterio normalizado"
    )
    st.markdown("---")
    st.caption("Asignatura: Evaluación, Despliegue y Monitorización de Modelos")


# ════════════════════════════════════════════════════════════════════════════
#  PESTAÑAS PRINCIPALES
# ════════════════════════════════════════════════════════════════════════════
tab_diag, tab_emp, tab_plan = st.tabs([
    "🗺️ Diagnóstico Territorial",
    "💼 Recomendador de Emprendimiento",
    "🏛️ Panel del Planificador",
])


# ────────────────────────────────────────────────────────────────────────────
#  PESTAÑA 1 · DIAGNÓSTICO TERRITORIAL (visión general)
# ────────────────────────────────────────────────────────────────────────────
with tab_diag:
    st.markdown("## Visión general del comercio en Valencia")
    st.markdown(
        "Diagnóstico inicial cruzando 17 categorías comerciales con población, "
        "accesibilidad en transporte público y vulnerabilidad socioeconómica."
    )

    # KPIs globales
    col1, col2, col3, col4 = st.columns(4)
    pob_total = int(df["poblacion"].sum())
    n_insuf = int((df["oferta_insuficiente"] == 1).sum())
    densidad_media = df["densidad_locales_hab"].mean() * 1000  # locales por 1000 hab
    pct_insuf = n_insuf / len(df) * 100

    with col1:
        st.metric("Locales totales", f"{n_locales:,}")
    with col2:
        st.metric("Población atendida", f"{pob_total:,}")
    with col3:
        st.metric("Densidad media", f"{densidad_media:.1f}", help="Locales por cada 1.000 habitantes")
    with col4:
        st.metric("Barrios desabastecidos", f"{n_insuf} · {pct_insuf:.0f}%")

    st.markdown("---")

    # Mapa principal + selector de capa
    col_mapa, col_panel = st.columns([2.2, 1])

    with col_panel:
        st.markdown("### Capa visualizada")
        capa = st.radio(
            "Selecciona la métrica a mostrar:",
            options=[
                "Densidad comercial",
                "Oferta insuficiente (target)",
                "Vulnerabilidad social",
                "Accesibilidad transporte",
                "Presión turística",
            ],
            label_visibility="collapsed",
        )

        st.markdown("### Distribución por sector")
        sector_cols = [c for c in df.columns if c.startswith("n_locales_") and c != "n_locales_total" and c != "n_locales_otro"]
        totales_sector = df[sector_cols].sum().sort_values(ascending=True)
        totales_sector.index = [SECTOR_LABELS.get(c.replace("n_locales_", ""), c) for c in totales_sector.index]

        fig_bar = go.Figure(go.Bar(
            x=totales_sector.values,
            y=totales_sector.index,
            orientation="h",
            marker=dict(color=COLORS["accent"], line=dict(color=COLORS["accent_dark"], width=0.5)),
        ))
        fig_bar.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(title="Nº locales", showgrid=True, gridcolor="#E8EDF5",
                    tickfont=dict(color="#0F1B2D")),
            yaxis=dict(title="", tickfont=dict(color="#0F1B2D", size=11)),
            font=dict(family="Inter", size=11, color="#0F1B2D"),
        )
        st.plotly_chart(fig_bar, use_container_width=True) 
        
    with col_mapa:
        st.markdown("### Mapa coroplético de Valencia")

        if capa == "Densidad comercial":
            m = crear_mapa_coropletico(gdf, df, "densidad_locales_hab",
                                       colorscale="YlOrRd",
                                       titulo_leyenda="Locales/hab")
        elif capa == "Oferta insuficiente (target)":
            df_t = df.copy()
            df_t["target_label"] = df_t["oferta_insuficiente"].fillna(-1).astype(int)
            m = crear_mapa_coropletico(gdf, df_t, "target_label",
                                       colorscale="RdYlGn_r",
                                       titulo_leyenda="0 Suficiente · 1 Insuficiente")
        elif capa == "Vulnerabilidad social":
            m = crear_mapa_coropletico(gdf, df, "Ind_Global",
                                       colorscale="RdYlGn",
                                       titulo_leyenda="Índice Global (mayor=mejor)")
        elif capa == "Presión turística":
            m = crear_mapa_coropletico(gdf, df, "indice_turistico",
                                       colorscale="PuRd",
                                       titulo_leyenda="Índice de presión turística")
        else:
            m = crear_mapa_coropletico(gdf, df, "accesibilidad_tp",
                                       colorscale="Blues",
                                       titulo_leyenda="Paradas EMT + Metro×2")

        st_folium(m, width=None, height=540, returned_objects=[])

    # ── Rankings: vulnerabilidad, desabastecimiento y presión turística ──────
    st.markdown("---")
    col_vuln, col_ratio, col_tur = st.columns(3)

    with col_vuln:
        st.markdown("### 🔴 Barrios más vulnerables")
        st.markdown(
            "Ordenados por el **Índice Global de Vulnerabilidad** "
            "(valores bajos = mayor vulnerabilidad socioeconómica)."
        )
        df_vuln = (
            df[["nombre", "Ind_Global", "poblacion"]]
            .dropna(subset=["Ind_Global"])
            .sort_values("Ind_Global")
            .head(10)
            .reset_index(drop=True)
        )
        df_vuln.index += 1
        df_vuln.columns = ["Barrio", "Índice Global", "Población"]
        df_vuln["Índice Global"] = df_vuln["Índice Global"].round(2)
        df_vuln["Población"] = df_vuln["Población"].apply(lambda x: f"{int(x):,}")

        # Colorear filas según nivel de vulnerabilidad
        def color_vuln(val):
            try:
                v = float(val)
                if v < 2:   return "color: #B23B3B; font-weight:600"
                elif v < 3: return "color: #D4A04C; font-weight:600"
                else:        return "color: #2D7D5A"
            except Exception:
                return ""

        st.dataframe(
            df_vuln,
            use_container_width=True,
            column_config={
                "Índice Global": st.column_config.NumberColumn(
                    "Índice Global ↑ mejor",
                    format="%.2f",
                ),
                "Población": st.column_config.TextColumn("Población"),
            },
        )

    with col_ratio:
        st.markdown("### 🏪 Barrios con mayor déficit comercial")
        st.markdown(
            "Ratio **habitantes por local comercial** — valores altos indican "
            "escasez de oferta relativa a la demanda potencial."
        )
        df_ratio = df[["nombre", "poblacion", "n_locales_total"]].copy()
        df_ratio = df_ratio[df_ratio["n_locales_total"] > 0].copy()
        df_ratio["hab_por_local"] = (
            df_ratio["poblacion"] / df_ratio["n_locales_total"]
        ).round(1)
        df_ratio = (
            df_ratio.sort_values("hab_por_local", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        df_ratio.index += 1
        df_ratio = df_ratio[["nombre", "hab_por_local", "poblacion", "n_locales_total"]]
        df_ratio.columns = ["Barrio", "Hab./local", "Población", "Locales"]
        df_ratio["Población"] = df_ratio["Población"].apply(lambda x: f"{int(x):,}")
        df_ratio["Locales"] = df_ratio["Locales"].apply(lambda x: f"{int(x)}")

        st.dataframe(
            df_ratio,
            use_container_width=True,
            column_config={
                "Hab./local": st.column_config.NumberColumn(
                    "Hab./local ↑ más déficit",
                    format="%.1f",
                ),
                "Población": st.column_config.TextColumn("Población"),
                "Locales": st.column_config.TextColumn("Locales"),
            },
        )

    with col_tur:
        st.markdown("### 🧳 Barrios más turísticos")
        st.markdown(
            "Ordenados por el **Índice de Presión Turística** "
            "(oferta Airbnb/VUT, actividad de reservas y presión reciente)."
        )
        if "indice_turistico" in df.columns:
            df_tur = (
                df[["nombre", "indice_turistico", "poblacion"]]
                .dropna(subset=["indice_turistico"])
                .sort_values("indice_turistico", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )
            df_tur.index += 1
            df_tur.columns = ["Barrio", "Índice Turístico", "Población"]
            df_tur["Índice Turístico"] = df_tur["Índice Turístico"].round(3)
            df_tur["Población"] = df_tur["Población"].apply(lambda x: f"{int(x):,}")

            st.dataframe(
                df_tur,
                use_container_width=True,
                column_config={
                    "Índice Turístico": st.column_config.ProgressColumn(
                        "Índice Turístico ↑ mayor presión",
                        format="%.3f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    "Población": st.column_config.TextColumn("Población"),
                },
            )
        else:
            st.info("No hay datos de índice turístico disponibles en el dataset.")


# ────────────────────────────────────────────────────────────────────────────
#  PESTAÑA 2 · RECOMENDADOR EMPRENDEDOR
# ────────────────────────────────────────────────────────────────────────────
with tab_emp:
    st.markdown("## Recomendador de localización comercial")
    st.markdown(
        "Selecciona el tipo de negocio que quieres abrir. El sistema calcula un "
        "score 0-100 para cada barrio combinando competencia, demanda potencial, "
        "accesibilidad y calidad socioeconómica de la zona."
    )

    # Controles
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1.2, 1.2])

    with col_ctrl1:
        sector_disp = sorted([
            (SECTOR_LABELS.get(c.replace("n_locales_", ""), c),
             c.replace("n_locales_", ""))
            for c in df.columns
            if c.startswith("n_locales_") and c not in ["n_locales_total", "n_locales_otro"]
        ])
        sector_label = st.selectbox(
            "Tipo de negocio:",
            options=[s[0] for s in sector_disp],
            index=[s[0] for s in sector_disp].index("Cafetería") if any(s[0] == "Cafetería" for s in sector_disp) else 0,
        )
        sector_id = dict(sector_disp)[sector_label]

    with col_ctrl2:
        modo_pesos = st.selectbox(
            "Estrategia:",
            ["Equilibrada", "Anti-competencia", "Maximizar demanda", "Personalizada"],
        )

    with col_ctrl3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:{COLORS['bg_subtle']};padding:10px 14px;"
        f"border-radius:8px;border-left:3px solid {COLORS['accent']};font-size:12px;"
        f"color:{COLORS['text_primary']};'>"
        f"<b>Sector activo:</b><br/>{sector_label}</div>",
        unsafe_allow_html=True
    )

    # Pesos según estrategia
    presets = {
        "Equilibrada":       {"competencia": 0.40, "demanda": 0.25, "accesibilidad": 0.20, "calidad": 0.15},
        "Anti-competencia":  {"competencia": 0.60, "demanda": 0.20, "accesibilidad": 0.10, "calidad": 0.10},
        "Maximizar demanda": {"competencia": 0.20, "demanda": 0.50, "accesibilidad": 0.20, "calidad": 0.10},
    }

    if modo_pesos == "Personalizada":
        with st.expander("⚙️ Ajustar pesos del score", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1: w_comp = st.slider("Anti-competencia", 0.0, 1.0, 0.40, 0.05)
            with c2: w_dem  = st.slider("Demanda",          0.0, 1.0, 0.25, 0.05)
            with c3: w_acc  = st.slider("Accesibilidad",    0.0, 1.0, 0.20, 0.05)
            with c4: w_cal  = st.slider("Calidad zona",     0.0, 1.0, 0.15, 0.05)
            suma = w_comp + w_dem + w_acc + w_cal
            if suma == 0: suma = 1
            pesos = {"competencia": w_comp/suma, "demanda": w_dem/suma,
                     "accesibilidad": w_acc/suma, "calidad": w_cal/suma}
            st.caption(f"Suma de pesos normalizada a 1.0 (original: {suma:.2f})")
    else:
        pesos = presets[modo_pesos]

    # Calcular ranking
    ranking = calcular_score_emprendedor(df, sector_id, pesos)

    st.markdown("---")

    # Resultados
    col_map, col_top = st.columns([2, 1])

    with col_map:
        st.markdown("### Mapa de idoneidad")
        ranking_map = ranking[["coddistbar", "nombre", "score"]].copy()
        m_emp = crear_mapa_coropletico(gdf, ranking_map, "score",
                                        colorscale="RdYlGn",
                                        titulo_leyenda=f"Score idoneidad · {sector_label}")
        st_folium(m_emp, width=None, height=500, returned_objects=[])

    with col_top:
        st.markdown(f"### Top 10 barrios")
        st.markdown(
            f"<div style='color:{COLORS['text_secondary']};font-size:13px;margin-bottom:8px;'>"
            f"Estrategia: <b style='color:{COLORS['text_primary']}'>{modo_pesos}</b></div>",
            unsafe_allow_html=True
        )

        top10 = ranking.head(10).reset_index(drop=True)
        for i, row in top10.iterrows():
            score_color = "#2D7D5A" if row["score"] >= 70 else (
                          "#D4A04C" if row["score"] >= 50 else "#B23B3B")
            st.markdown(f"""
            <div style='background:white;padding:10px 14px;border-radius:8px;
                        border:1px solid {COLORS["border"]};
                        border-left:4px solid {score_color};margin-bottom:6px;
                        display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <div style='font-weight:600;font-size:13px;color:{COLORS["text_primary"]};'>
                        {i+1}. {row['nombre']}
                    </div>
                    <div style='font-size:11px;color:{COLORS["text_secondary"]};'>
                        Pob. {int(row['poblacion']):,} · Competidores: {int(row['competencia'])}
                    </div>
                </div>
                <div style='font-size:20px;font-weight:700;color:{score_color};'>
                    {row['score']:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Análisis detallado del barrio Top 1
    st.markdown("---")
    st.markdown(f"### Análisis del barrio recomendado: **{top10.iloc[0]['nombre']}**")
    barrio_top = top10.iloc[0]

    col_radar, col_desglose = st.columns([1, 1])
    with col_radar:
        fig_r = grafico_radar_barrio(df, int(barrio_top["codbarrio"]), sector_id)
        if fig_r: st.plotly_chart(fig_r, use_container_width=True)

    with col_desglose:
        st.markdown("##### Desglose del score")
        componentes = {
            "Baja competencia":   pesos["competencia"]   * (1 - barrio_top["competencia_n"]) * 100,
            "Demanda potencial":  pesos["demanda"]       * barrio_top["demanda_n"] * 100,
            "Accesibilidad TP":   pesos["accesibilidad"] * barrio_top["accesibilidad_n"] * 100,
            "Calidad zona":       pesos["calidad"]       * (1 - barrio_top["vuln_n"]) * 100,
        }
        fig_desg = go.Figure(go.Bar(
            x=list(componentes.values()),
            y=list(componentes.keys()),
            orientation="h",
            marker=dict(color=[COLORS["accent"], COLORS["bg_primary"], COLORS["neutral"], COLORS["success"]]),
            text=[f"{v:.1f}" for v in componentes.values()],
            textposition="outside",
        ))
        fig_desg.update_layout(
            height=280,
            margin=dict(l=10, r=40, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Contribución al score", range=[0, max(componentes.values())*1.25],
                    showgrid=True, gridcolor="#E8EDF5",
                    tickfont=dict(color="#0F1B2D")),
            yaxis=dict(tickfont=dict(color="#0F1B2D", size=11)),
            font=dict(family="Inter", size=11, color="#0F1B2D"),
        )
        st.plotly_chart(fig_desg, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
#  PESTAÑA 3 · PLANIFICADOR URBANO
# ────────────────────────────────────────────────────────────────────────────
with tab_plan:
    st.markdown("## Panel de Equidad Territorial")
    st.markdown(
        "Identifica barrios con oferta comercial insuficiente, entiende **por qué** "
        "lo son y simula intervenciones urbanísticas para evaluar su impacto."
    )

    # Predicciones del modelo
    feat_names = modelos["features"]
    X_pred = df[feat_names].copy()
    for c in X_pred.columns:
        if X_pred[c].isnull().any():
            X_pred[c] = X_pred[c].fillna(X_pred[c].median())

    probas = modelos["stacking"].predict_proba(X_pred)[:, 1]
    df_pred = df.copy()
    df_pred["prob_insuficiente"] = probas

    n_alerta = int((probas > 0.5).sum())
    n_critico = int((probas > 0.75).sum())

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Barrios en alerta", f"{n_alerta}", help="Prob > 0.5 de oferta insuficiente")
    with c2: st.metric("Barrios críticos", f"{n_critico}", help="Prob > 0.75")
    with c3:
        pob_afectada = int(df_pred[df_pred["prob_insuficiente"] > 0.5]["poblacion"].sum())
        st.metric("Población afectada", f"{pob_afectada:,}")
    with c4:
        vuln_media = df_pred[df_pred["prob_insuficiente"] > 0.5]["Ind_Global"].mean()
        st.metric("Vuln. media barrios alerta", f"{vuln_media:.2f}", help="Menor = más vulnerable")

    st.markdown("---")

    col_m, col_sel = st.columns([2, 1])

    with col_m:
        st.markdown("### Mapa de riesgo predictivo")
        m_plan = crear_mapa_coropletico(gdf, df_pred, "prob_insuficiente",
                                         colorscale="YlOrRd",
                                         titulo_leyenda="Prob. oferta insuficiente")
        st_folium(m_plan, width=None, height=500, returned_objects=[])

    with col_sel:
        st.markdown("### Inspección de barrio")
        barrios_alerta = df_pred[df_pred["prob_insuficiente"] > 0.4].sort_values(
            "prob_insuficiente", ascending=False)
        opciones = barrios_alerta["nombre"].tolist() + ["─" * 10] + \
                   df_pred[df_pred["prob_insuficiente"] <= 0.4]["nombre"].tolist()
        barrio_sel = st.selectbox(
            "Selecciona un barrio:",
            opciones,
            label_visibility="collapsed",
        )

        if barrio_sel and "─" not in barrio_sel:
            row_b = df_pred[df_pred["nombre"] == barrio_sel].iloc[0]
            prob = row_b["prob_insuficiente"]
            color = "#B23B3B" if prob > 0.75 else ("#D4A04C" if prob > 0.5 else "#2D7D5A")
            tur_val = row_b["indice_turistico"] if ("indice_turistico" in row_b.index and pd.notna(row_b["indice_turistico"])) else None
            tur_str = f"{tur_val:.3f}" if tur_val is not None else "—"
            st.markdown(f"""
            <div style='background:white;padding:14px;border-radius:8px;
                        border:1px solid {COLORS["border"]};
                        border-left:4px solid {color};'>
                <div style='font-size:11px;color:{COLORS["text_secondary"]};text-transform:uppercase;'>
                    Diagnóstico
                </div>
                <div style='font-size:22px;font-weight:700;color:{color};margin:4px 0;'>
                    {prob*100:.0f}% prob. insuficiente
                </div>
                <hr style='border:none;border-top:1px solid {COLORS["border"]};margin:8px 0;'>
                <div style='font-size:12px;line-height:1.7;color:{COLORS["text_primary"]};'>
                    <b>Población:</b> {int(row_b['poblacion']):,}<br/>
                    <b>Locales totales:</b> {int(row_b['n_locales_total'])}<br/>
                    <b>Paradas EMT:</b> {int(row_b['n_paradas_emt'])}<br/>
                    <b>Bocas metro:</b> {int(row_b['n_paradas_metro'])}<br/>
                    <b>Equipamiento:</b> {int(row_b['n_equipamiento_municipal'])}<br/>
                    <b>Índice turístico:</b> {tur_str}<br/>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Gráfico de araña comparativo ────────────────────────────────────────
    # ── Gráfico de araña comparativo ────────────────────────────────────────
    if barrio_sel and "─" not in barrio_sel:
        st.markdown("---")
        st.markdown("### 🕸️ Perfil del barrio vs. media de Valencia")
        st.markdown(
            "Las dimensiones son las mismas que en el análisis del barrio recomendado. "
            "El área **gris** representa la media de todos los barrios de Valencia; "
            "el área **naranja** muestra el perfil del barrio seleccionado."
        )
        
        # LLAMADA CORREGIDA: Pasamos 'barrio_sel' (el nombre estricto del selectbox)
        fig_radar_plan = grafico_radar_comparativo(df_pred, barrio_sel)
        
        if fig_radar_plan:
            col_radar_plan, col_radar_info = st.columns([1.2, 1])
            with col_radar_plan:
                st.plotly_chart(fig_radar_plan, use_container_width=True)
            with col_radar_info:
                dims_info = {
                    "Comercio total":      ("n_locales_total",          False),
                    "Conexión transporte": ("accesibilidad_tp",         False),
                    "Equipamiento":        ("n_equipamiento_municipal", False),
                    "Demanda (pob.)":      ("poblacion",                False),
                    "Calidad zona":        ("Ind_Global",               True),
                    "Presión turística":   ("indice_turistico",         False),
                }
                dims_info = {k: v for k, v in dims_info.items() if v[0] in df.columns}
                filas = []
                for dim_label, (col_r, invertir) in dims_info.items():
                    val_b = row_b[col_r] if pd.notna(row_b[col_r]) else df[col_r].mean()
                    val_m = df[col_r].mean()
                    diferencia = val_b - val_m
                    if invertir:
                        mejor = "✅" if diferencia < 0 else ("➖" if abs(diferencia) < 0.5 else "⚠️")
                    else:
                        mejor = "✅" if diferencia > 0 else ("➖" if abs(diferencia / (val_m + 1e-9)) < 0.05 else "⚠️")
                    filas.append({
                        "Dimensión": dim_label,
                        "Barrio": f"{val_b:,.1f}",
                        "Media VLC": f"{val_m:,.1f}",
                        "vs. media": mejor,
                    })
                df_tabla = pd.DataFrame(filas)
                st.markdown("##### Comparativa de valores reales")
                st.dataframe(
                    df_tabla,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "vs. media": st.column_config.TextColumn("vs. media", width="small"),
                    },
                )
    # Explicabilidad SHAP
    if barrio_sel and "─" not in barrio_sel:
        st.markdown("---")
        st.markdown("### 🔬 Explicación del modelo (SHAP)")
        st.markdown(
            "*¿Por qué el modelo clasifica este barrio como insuficiente?* "
            "Cada barra muestra cuánto empuja cada variable la predicción "
            "hacia 'insuficiente' (rojo) o 'suficiente' (verde)."
        )

        # Contexto turístico antes del SHAP
        if "indice_turistico" in df.columns:
            tur_idx = row_b["indice_turistico"] if pd.notna(row_b.get("indice_turistico")) else 0.0
            tur_pct = int((df["indice_turistico"].fillna(0) <= tur_idx).mean() * 100)
            if tur_idx > df["indice_turistico"].quantile(0.75):
                tur_label = "🔴 Alta presión turística"
            elif tur_idx > df["indice_turistico"].quantile(0.40):
                tur_label = "🟡 Presión turística media"
            else:
                tur_label = "🟢 Baja presión turística"
            tur_idx_str = f"{tur_idx:.3f}"
            st.markdown(
                f"<div style='background:{COLORS['bg_subtle']};padding:8px 14px;"
                f"border-radius:8px;border-left:3px solid {COLORS['accent']};"
                f"font-size:12px;color:{COLORS['text_primary']};margin-bottom:10px;'>"
                f"<b>Contexto turístico:</b> {tur_label} &nbsp;·&nbsp; "
                f"Índice: <b>{tur_idx_str}</b> (percentil {tur_pct}% de Valencia)</div>",
                unsafe_allow_html=True
            )

        idx_b = df_pred[df_pred["nombre"] == barrio_sel].index[0]
        pos_b = X_pred.index.get_loc(idx_b)
        shap_values = modelos["explainer"].shap_values(X_pred.iloc[[pos_b]])[0]

        # Top contribuciones (mayor magnitud absoluta)
        contribs = pd.DataFrame({
            "feature": X_pred.columns,
            "valor":   X_pred.iloc[pos_b].values,
            "shap":    shap_values,
        }).sort_values("shap", key=abs, ascending=False).head(12)

        contribs = contribs.sort_values("shap")
        fig_shap = go.Figure(go.Bar(
            x=contribs["shap"],
            y=contribs["feature"],
            orientation="h",
            marker=dict(color=["#B23B3B" if v > 0 else "#2D7D5A" for v in contribs["shap"]]),
            text=[f"{v:+.2f}" for v in contribs["shap"]],
            textposition="outside",
        ))
        fig_shap.update_layout(
            height=420,
            margin=dict(l=150, r=40, t=10, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Impacto SHAP (→ insuficiente)", zeroline=True,
                    zerolinecolor=COLORS["text_primary"], showgrid=True, gridcolor="#E8EDF5",
                    tickfont=dict(color="#0F1B2D")),
            yaxis=dict(title="", tickfont=dict(color="#0F1B2D", size=11)),
            font=dict(family="Inter", size=11, color="#0F1B2D"),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Simulador de escenarios
        st.markdown("---")
        st.markdown("### 🎚️ Simulador de intervención urbanística")
        st.markdown(
            f"Modifica las variables del barrio **{barrio_sel}** para ver cómo "
            "cambiaría la predicción del modelo. Análisis causal vs correlación (Tema 3)."
        )

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        emt_actual = int(row_b["n_paradas_emt"])
        metro_actual = int(row_b["n_paradas_metro"])
        equip_actual = int(row_b["n_equipamiento_municipal"])
        loc_actual = int(row_b["n_locales_total"])

        with col_s1:
            emt_nuevo = st.slider("Paradas EMT", 0, max(60, emt_actual+30),
                                  emt_actual, key=f"emt_{barrio_sel}")
        with col_s2:
            metro_nuevo = st.slider("Bocas metro", 0, max(10, metro_actual+5),
                                    metro_actual, key=f"metro_{barrio_sel}")
        with col_s3:
            equip_nuevo = st.slider("Equipamientos", 0, max(120, equip_actual+30),
                                    equip_actual, key=f"equip_{barrio_sel}")
        with col_s4:
            loc_nuevo = st.slider("Locales nuevos", loc_actual, loc_actual + 50,
                                  loc_actual, key=f"loc_{barrio_sel}")

        # Recalcular con valores simulados
        X_sim = X_pred.iloc[[pos_b]].copy()
        if "n_paradas_emt" in X_sim.columns: X_sim["n_paradas_emt"] = emt_nuevo
        if "n_paradas_metro" in X_sim.columns: X_sim["n_paradas_metro"] = metro_nuevo
        if "n_equipamiento_municipal" in X_sim.columns: X_sim["n_equipamiento_municipal"] = equip_nuevo
        if "n_locales_total" in X_sim.columns: X_sim["n_locales_total"] = loc_nuevo
        if "accesibilidad_tp" in X_sim.columns: X_sim["accesibilidad_tp"] = emt_nuevo + metro_nuevo * 2

        prob_sim = modelos["stacking"].predict_proba(X_sim)[0, 1]
        delta = prob_sim - prob

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Prob. actual", f"{prob*100:.1f}%")
        with col_res2:
            st.metric("Prob. simulada", f"{prob_sim*100:.1f}%",
                      delta=f"{delta*100:+.1f}%", delta_color="inverse")
        with col_res3:
            if abs(delta) < 0.02:
                impacto = "Sin cambio significativo"
                color_imp = COLORS["neutral"]
            elif delta < -0.1:
                impacto = "Mejora notable"
                color_imp = COLORS["success"]
            elif delta < 0:
                impacto = "Mejora ligera"
                color_imp = COLORS["success"]
            else:
                impacto = "Empeora"
                color_imp = COLORS["danger"]
            st.markdown(f"""
            <div style='background:white;padding:14px;border-radius:8px;
                        border:1px solid {COLORS["border"]};
                        border-left:4px solid {color_imp};margin-top:8px;'>
                <div style='font-size:11px;color:{COLORS["text_secondary"]};
                            text-transform:uppercase;'>Impacto estimado</div>
                <div style='font-size:18px;font-weight:700;color:{color_imp};margin-top:4px;'>
                    {impacto}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
    GeoMarket-VLC · Universidad · EDM 2025/26 · Datos: Valencia Open Data + OpenStreetMap
</div>
""", unsafe_allow_html=True)