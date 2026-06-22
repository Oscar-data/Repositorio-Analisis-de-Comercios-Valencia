# GeoMarket-VLC · Plataforma de Inteligencia Comercial Urbana

> Proyecto académico para la asignatura **Evaluación, Despliegue y Monitorización de Modelos (EDM)**  
> Universitat Politècnica de València · Grado en Ciencia de Datos

[![App desplegada](https://img.shields.io/badge/Streamlit-App%20en%20vivo-red?logo=streamlit)](https://9ppde4t6haz4lkufhqrgah.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

🔗 **[Acceder a la aplicación](https://9ppde4t6haz4lkufhqrgah.streamlit.app)**

---

## ¿Qué es GeoMarket-VLC?

GeoMarket-VLC es una aplicación web de inteligencia comercial que combina **análisis geoespacial**, **machine learning** y **explicabilidad XAI** para responder a una pregunta concreta: ¿qué barrios de Valencia tienen una oferta comercial insuficiente respecto a su población, y cuáles son las mejores zonas para abrir un negocio?

La plataforma integra datos abiertos del Ayuntamiento de Valencia (locales comerciales, transporte público, equipamiento municipal, demografía, vulnerabilidad socioeconómica) y los procesa end-to-end: desde el pipeline de datos hasta una app Streamlit interactiva con mapas coropléticos, modelos predictivos y recomendaciones personalizadas por sector.

---

## Estructura del repositorio

El repositorio está organizado en **3 ramas** separando código, datos y documentación:

### `main` — Documentación
Contiene únicamente este README.

### `Código` — Notebooks y aplicación
```
├── app_3.py                          # App Streamlit interactiva
├── geomarket_vlc_pipeline_v5.ipynb   # Fase 1: ingesta y unión de fuentes geoespaciales
├── geomarket_vlc_modelado.ipynb      # Fase 2: entrenamiento de modelos y serialización
├── datos_comercios.ipynb             # Exploración inicial de locales comerciales
├── preprocesar_aparcamientos.py      # Script de preprocesado de aparcamientos
└── requirements.txt                  # Dependencias del proyecto
```

### `Datos` — Fuentes de datos y modelos serializados
```
├── models/
│   ├── modelo_rf.pkl                 # Random Forest final
│   ├── modelo_xgb.pkl                # XGBoost final
│   ├── modelo_stacking.pkl           # Stacking Ensemble (RF + XGBoost → LogReg)
│   ├── shap_explainer.pkl            # TreeExplainer SHAP
│   └── feature_names.pkl             # Lista de features del modelo
│
├── geomarket_vlc_features.csv        # Tabla de features (output del pipeline)
├── geomarket_vlc_barrios.geojson     # Geometría de barrios (output del pipeline)
├── indice_turistico_por_barrio.csv   # Índice de presión turística por barrio
│
├── barris.json                       # Geometría base de barrios de Valencia
├── locales_valencia.json             # Locales comerciales (fuente principal)
├── emt_paradas.json                  # Paradas de autobús EMT
├── fgv_bocas.json                    # Bocas de metro / FGV
├── equipament_municipal.json         # Equipamiento municipal
├── aparcamientos_barrios.csv         # Aparcamientos por barrio
├── demografia_distritos.csv          # Demografía por distrito
├── vulnerabilidad-por-barrios.csv    # Índices de vulnerabilidad socioeconómica
└── airbnb_listings_limpio.csv.gz     # Listings Airbnb (base del índice turístico)
```

El flujo es estrictamente secuencial: el **pipeline** exporta el CSV y el GeoJSON que consume el **notebook de modelado**, que a su vez serializa los `.pkl` que carga la **app** en tiempo de ejecución sin reentrenar nada.

---

## Fase 1 · Pipeline de datos (`geomarket_vlc_pipeline_v5.ipynb`)

Parte de la geometría base de los 87 barrios de Valencia y realiza un **join maestro** con siete fuentes de datos abiertas:

| Dataset | Fuente | Join |
|---------|--------|------|
| Geometría de barrios | Ayuntamiento VLC | Base |
| Vulnerabilidad socioeconómica | Ayuntamiento VLC | `barrio_key` (nombre normalizado) |
| Demografía por distritos | Ayuntamiento VLC | `coddistrit` |
| Locales comerciales | Ayuntamiento VLC (JSON) | Spatial join geométrico |
| Paradas EMT (autobús) | EMT Valencia | Spatial join → agregación por barrio |
| Bocas de metro / FGV | FGV | Spatial join → agregación por barrio |
| Equipamiento municipal | Ayuntamiento VLC | Spatial join → agregación por barrio |
| Aparcamientos | Ayuntamiento VLC | `barrio_key` |
| Índice turístico | Calculado externamente | `coddistbar` |

Tras la limpieza e imputación con la mediana, el notebook construye las variables derivadas clave (`densidad_locales_hab`, `accesibilidad_tp`) y define el **target** `oferta_insuficiente` como los barrios por debajo del percentil 25 en densidad de locales por habitante (22 barrios positivos / 65 negativos).

**Salidas:**
- `output/geomarket_vlc_features.csv` — 87 barrios × 64 columnas
- `output/geomarket_vlc_barrios.geojson` — geometría limpia para mapas

---

## Fase 2 · Modelado (`geomarket_vlc_modelado.ipynb`)

### Datos de entrenamiento
87 barrios, 51 features numéricas, desbalanceo 75/25 (suficiente / insuficiente).

### Modelos entrenados

| Modelo | Configuración | ROC-AUC (CV 5-fold) | F1 macro |
|--------|--------------|---------------------|----------|
| Random Forest | 300 árboles, `class_weight='balanced'` | 0.850 | 0.726 |
| XGBoost | 200 estimadores, `scale_pos_weight=2.95` | — | — |
| **Stacking (RF + XGBoost → LogReg)** | Meta-modelo logístico | **0.859** | **0.780** |

Se usa `StratifiedKFold` con 5 folds en toda la evaluación. Se reporta F1/ROC-AUC en lugar de accuracy por el desbalanceo de clases (un clasificador trivial alcanzaría 75% de accuracy).

### Score de idoneidad para emprendedores

Función multi-criterio (0-100) por barrio y sector que combina cuatro componentes con pesos ajustables:

- **Baja competencia** — escasez de locales del mismo sector
- **Demanda potencial** — población del barrio
- **Accesibilidad TP** — paradas de autobús + bocas de metro
- **Calidad de zona** — inverso del índice de vulnerabilidad

### Explicabilidad (SHAP / XAI)

Se entrena un XGBoost final en todo el dataset y se construye un `TreeExplainer` SHAP para:
- **Importancia global** — qué variables pesan más en el modelo
- **Explicación local** — por qué un barrio concreto se clasifica como insuficiente (waterfall plot en la app)

### Serialización
Los cinco artefactos se guardan en `models/` con `joblib` y son cargados directamente por la app sin reentrenar.

---

## Fase 3 · App Streamlit (`app_3.py`)

Interfaz web con tres pestañas:

### 🗺️ Diagnóstico Comercial
Mapa coroplético interactivo de Valencia con cinco capas seleccionables (densidad de locales, clasificación de oferta, vulnerabilidad socioeconómica, presión turística, accesibilidad TP). Panel lateral con estadísticas del barrio seleccionado, tabla de los 10 barrios más vulnerables y distribución por sector comercial.

### 🏪 Recomendador para Emprendedores
Selección de sector (16 categorías: restaurante, cafetería, farmacia, peluquería, etc.) y sliders para ajustar los pesos del score multi-criterio. Devuelve un ranking de barrios con mapa de idoneidad, análisis detallado del barrio top (gráfico radar + desglose del score) y la predicción del modelo Stacking con probabilidad de oferta insuficiente.

### 🏙️ Planificación Urbana
Mapa de riesgo predictivo (probabilidad de oferta insuficiente por barrio) usando el modelo Stacking. Selección de barrio con perfil radar vs. media de Valencia, tabla comparativa de KPIs y panel de explicabilidad SHAP con las 12 variables más influyentes para ese barrio concreto.

---

## Instalación y ejecución local

> La app está desplegada y disponible directamente en **[https://9ppde4t6haz4lkufhqrgah.streamlit.app](https://9ppde4t6haz4lkufhqrgah.streamlit.app)** sin necesidad de instalación local. Las instrucciones siguientes son para reproducir el pipeline completo o desarrollar sobre el proyecto.

### Requisitos

```bash
pip install streamlit geopandas folium streamlit-folium \
            scikit-learn xgboost shap joblib plotly \
            pandas numpy "numpy<2"
```

> **Nota:** Es necesario `numpy<2` por compatibilidad con la versión actual de `fiona` (usada por geopandas para leer/escribir GeoJSON). Si usas `numpy>=2`, sustituye el motor con `engine="pyogrio"` o instala `pyogrio`.

### Ejecutar el pipeline

```bash
# 1. Colocar los datos en data/
# 2. Abrir y ejecutar todas las celdas de:
jupyter notebook geomarket_vlc_pipeline_v5.ipynb
# Genera output/geomarket_vlc_features.csv y output/geomarket_vlc_barrios.geojson
```

### Entrenar los modelos

```bash
jupyter notebook geomarket_vlc_modelado.ipynb
# Genera models/*.pkl
```

### Lanzar la app

```bash
# Los siguientes archivos deben estar en el directorio de ejecución:
# - geomarket_vlc_features.csv
# - geomarket_vlc_barrios.geojson
# - models/ (con los 5 .pkl)

streamlit run app_3.py
```

---

## Fuentes de datos

Todos los datos son de acceso abierto:

- [Portal de datos abiertos del Ayuntamiento de Valencia](https://www.valencia.es/dadesobertes/)
- [EMT Valencia — API de paradas](https://www.emtvalencia.es/)
- [FGV — Ferrocarils de la Generalitat Valenciana](https://www.fgv.es/)

---

## Tecnologías utilizadas

`Python` · `GeoPandas` · `Shapely` · `scikit-learn` · `XGBoost` · `SHAP` · `Streamlit` · `Folium` · `Plotly` · `joblib` · `pandas` · `NumPy`

---

## Contexto académico

Este proyecto cubre los siguientes temas de la asignatura EDM:

| Tema | Concepto | Aplicación |
|------|----------|------------|
| 1 | Model Evaluation con datos desbalanceados | F1 / ROC-AUC, StratifiedKFold, coste asimétrico de falsos negativos |
| 2 | Ensemble y Hybrid Models | Stacking (RF + XGBoost → LogisticRegression) |
| 3 | Explicabilidad XAI | SHAP TreeExplainer, importancia global y explicaciones locales |
| 4 | Despliegue | App Streamlit lista para producción, carga de modelos serializados |

---

## Autores

**Oscar Muñoz** · **David Llacer** · **Pablo Arnau**  
Estudiantes de Ciencia de Datos · ETSInf, Universitat Politècnica de València
