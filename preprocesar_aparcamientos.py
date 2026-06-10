"""
Preprocesador de datos: aparcamientos por distritos y barrios de Valencia
=========================================================================
El CSV contiene tres tipos de filas mezcladas:
  - GRUP = "Grup"       → 19 filas: resumen agregado por DISTRICTE
  - GRUP = "Grup 2"     → 88 filas: datos por BARRI (sin franja de edad)
  - GRUP = grupos edad  → 361 filas: demografía por franja de edad y DISTRICTE

Genera tres DataFrames limpios y separados:
  1. df_distritos   → datos de aparcamiento por distrito (agregados)
  2. df_barrios     → datos de aparcamiento por barrio
  3. df_demografico → pirámide de edad por distrito
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# 1. CARGA BRUTA
# ──────────────────────────────────────────────
ruta = "aparcamientos-por-distritos-barrios-val.csv"

df_raw = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")

# Limpiar nombre de la primera columna (viene con BOM \ufeff)
df_raw.columns = df_raw.columns.str.strip().str.replace("\ufeff", "", regex=False)

print(f"Filas totales: {len(df_raw)}")
print(f"Columnas: {df_raw.columns.tolist()}\n")


# ──────────────────────────────────────────────
# 2. FUNCIÓN AUXILIAR: convertir comas decimales a float
# ──────────────────────────────────────────────
def coma_a_float(serie: pd.Series) -> pd.Series:
    """Convierte '0,66' → 0.66, dejando NaN donde no hay valor."""
    return (
        serie.astype(str)
        .str.replace(",", ".", regex=False)
        .replace("nan", np.nan)
        .pipe(pd.to_numeric, errors="coerce")
    )


# Columnas de ratios que usan coma decimal (pandas las leyó como str)
cols_ratio = [
    "PLACES/HABITANTS",
    "PLACES/HABITANTS 20-70",
    "PLACES/TURISMES",
    "HABITANTS/TURISMES",
    "HABITANTS 20-70/TURISMES",
    "HABITANTS/HABITANTS 20-70",
]


# ──────────────────────────────────────────────
# 3. SEPARAR LOS TRES TIPOS DE FILAS
# ──────────────────────────────────────────────

# ---- 3a. Resúmenes por DISTRICTE ----------------------------------------
df_distritos = (
    df_raw[df_raw["GRUP"] == "Grup"]
    .copy()
    .drop(columns=["GRUP", "HABITANTS GRUP"])  # columnas sin sentido aquí
    .reset_index(drop=True)
)

# Convertir ratios de texto a float
for col in cols_ratio:
    df_distritos[col] = coma_a_float(df_distritos[col])

# Columnas numéricas que ya son float64 pero tienen NaN inesperados
cols_num_dist = [
    "HABITANTS", "HABITANTS 20-70", "TOTAL", "LLIURES", "ORA",
    "GUALS", "PÀRQUINGS", "SOLARS", "ALTRES", "TURISMES",
]
for col in cols_num_dist:
    df_distritos[col] = pd.to_numeric(df_distritos[col], errors="coerce")

print("── df_distritos ──────────────────────────────────────────")
print(f"  Forma: {df_distritos.shape}")
print(f"  Nulos por columna:\n{df_distritos.isnull().sum().to_string()}\n")


# ---- 3b. Datos por BARRI ------------------------------------------------
df_barrios = (
    df_raw[df_raw["GRUP"] == "Grup 2"]
    .copy()
    .drop(columns=["GRUP", "HABITANTS GRUP"])
    .reset_index(drop=True)
)

# Convertir ratios a float (solo los que tienen datos en este nivel)
cols_ratio_barrio = [c for c in cols_ratio if c in df_barrios.columns]
for col in cols_ratio_barrio:
    df_barrios[col] = coma_a_float(df_barrios[col])

# HABITANTS 20-70 y TURISMES son NaN en TODOS los barrios → las marcamos
cols_sin_dato_barrio = ["HABITANTS 20-70", "TURISMES",
                        "PLACES/HABITANTS 20-70", "PLACES/TURISMES",
                        "HABITANTS/TURISMES", "HABITANTS 20-70/TURISMES",
                        "HABITANTS/HABITANTS 20-70"]
# (ya son NaN, no hace falta eliminarlas; se documentan para el usuario)

print("── df_barrios ────────────────────────────────────────────")
print(f"  Forma: {df_barrios.shape}")
print(f"  Nulos por columna:\n{df_barrios.isnull().sum().to_string()}\n")


# ---- 3c. Datos demográficos por franja de edad --------------------------
GRUPOS_EDAD = [
    "0 - 4", "05-sep", "oct-14", "15 - 19", "20 - 24", "25 - 29",
    "30 -34", "35 - 39", "40 - 44", "45 - 49", "50 - 54", "55 - 59",
    "60 - 64", "65 - 69", "70 - 74", "75 - 79", "80 - 84", "85 - 89",
    "? 90",
]

df_demografico = (
    df_raw[df_raw["GRUP"].isin(GRUPOS_EDAD)]
    .copy()
    [["DISTRICTE", "BARRI", "GRUP", "HABITANTS GRUP"]]  # solo lo útil
    .rename(columns={"GRUP": "GRUP_EDAD", "HABITANTS GRUP": "HABITANTS_GRUP"})
    .reset_index(drop=True)
)

# Normalizar etiquetas de grupo (los rangos vienen sucios por Excel)
mapa_grupos = {
    "05-sep": "05 - 09",
    "oct-14": "10 - 14",
    "30 -34": "30 - 34",
    "? 90":   "90+",
}
df_demografico["GRUP_EDAD"] = df_demografico["GRUP_EDAD"].replace(mapa_grupos)

print("── df_demografico ────────────────────────────────────────")
print(f"  Forma: {df_demografico.shape}")
print(f"  Grupos de edad únicos:\n  {sorted(df_demografico['GRUP_EDAD'].unique())}")
print(f"  Nulos: {df_demografico.isnull().sum().to_dict()}\n")


# ──────────────────────────────────────────────
# 4. VERIFICACIÓN RÁPIDA
# ──────────────────────────────────────────────
print("── Verificación cruzada ──────────────────────────────────")
print(f"  Distritos en df_distritos:  {df_distritos['DISTRICTE'].nunique()}")
print(f"  Distritos en df_barrios:    {df_barrios['DISTRICTE'].nunique()}")
print(f"  Distritos en df_demografico:{df_demografico['DISTRICTE'].nunique()}")
print(f"  Barrios únicos:             {df_barrios['BARRI'].nunique()}")
total = len(df_distritos) + len(df_barrios) + len(df_demografico)
print(f"  Filas recuperadas:          {total} / {len(df_raw)} ({'OK' if total == len(df_raw) else 'ERROR'})")


# ──────────────────────────────────────────────
# 5. EXPORTAR CSVs LIMPIOS
# ──────────────────────────────────────────────
import os
out = "/mnt/user-data/outputs"
os.makedirs(out, exist_ok=True)

df_distritos.to_csv(f"{out}/aparcamientos_distritos.csv", index=False, encoding="utf-8-sig")
df_barrios.to_csv(f"{out}/aparcamientos_barrios.csv",     index=False, encoding="utf-8-sig")
df_demografico.to_csv(f"{out}/demografia_distritos.csv",  index=False, encoding="utf-8-sig")

print("\n✓ Archivos exportados:")
print(f"  → {out}/aparcamientos_distritos.csv")
print(f"  → {out}/aparcamientos_barrios.csv")
print(f"  → {out}/demografia_distritos.csv")
