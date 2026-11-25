"""Definición de distritos, barrios e ingeniería de características tabulares."""

import unicodedata

import numpy as np
import pandas as pd

# Referencia base (€/m²) para los 21 distritos oficiales de Madrid.
MADRID_DISTRICTS: dict[str, int] = {
    "Salamanca": 7800,
    "Chamberí": 6900,
    "Retiro": 6400,
    "Centro": 6300,
    "Chamartín": 6200,
    "Moncloa-Aravaca": 5400,
    "Arganzuela": 5150,
    "Tetuán": 5000,
    "Hortaleza": 4700,
    "Fuencarral-El Pardo": 4550,
    "Ciudad Lineal": 4200,
    "Barajas": 4100,
    "San Blas-Canillejas": 3550,
    "Moratalaz": 3400,
    "Latina": 3250,
    "Villa de Vallecas": 3200,
    "Carabanchel": 3150,
    "Vicálvaro": 3100,
    "Usera": 2850,
    "Puente de Vallecas": 2450,
    "Villaverde": 2250,
}

PROPERTY_TYPES: list[str] = ["Piso", "Ático", "Dúplex", "Casa o chalet"]
DISTRICT_WIDE_BARRIO = "__distrito__"

CATEGORICAL_FEATURES: list[str] = ["barrio", "micro_barrio", "tipo_vivienda"]

NUMERIC_FEATURES: list[str] = [
    "metros_cuadrados",
    "habitaciones",
    "banos",
    "planta",
    "ascensor",
    "exterior",
    "terraza",
    "garaje",
    "aire_acondicionado",
    "trastero",
    "necesita_reforma",
    "obra_nueva",
    "m2_por_habitacion",
    "sin_ascensor_planta_alta",
    "es_bajo",
]

ALL_MODEL_FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


_CANONICAL_LOOKUP: dict[str, str] = {
    **{_strip_accents(name).lower(): name for name in MADRID_DISTRICTS},
    "moncloa": "Moncloa-Aravaca",
    "fuencarral": "Fuencarral-El Pardo",
    "san blas": "San Blas-Canillejas",
}


def normalize_district_name(value: str) -> str:
    """Normaliza el nombre del distrito hacia su forma canónica si pertenece a Madrid."""
    cleaned = " ".join(str(value).strip().split())
    key = _strip_accents(cleaned).lower()
    return _CANONICAL_LOOKUP.get(key, cleaned)


def normalize_micro_barrio(value: object) -> str:
    """Normaliza el micro-barrio o devuelve marcador de media de distrito."""
    if value is None or pd.isna(value):
        return DISTRICT_WIDE_BARRIO
    cleaned = " ".join(str(value).strip().split())
    return cleaned if cleaned else DISTRICT_WIDE_BARRIO


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz exacta de características usada en entrenamiento e inferencia.

    Mantener esta transformación fuera del ``Pipeline`` serializado evita problemas
    de ``pickle`` al cargar el modelo en distintos módulos o puntos de entrada.
    """
    out = pd.DataFrame(index=df.index)
    out["barrio"] = df["barrio"].astype(str).map(normalize_district_name)
    if "micro_barrio" in df.columns:
        out["micro_barrio"] = df["micro_barrio"].map(normalize_micro_barrio)
    else:
        out["micro_barrio"] = DISTRICT_WIDE_BARRIO

    if "tipo_vivienda" in df.columns:
        out["tipo_vivienda"] = (
            df["tipo_vivienda"].fillna("Piso").astype(str).str.strip().replace({"": "Piso"})
        )
    else:
        out["tipo_vivienda"] = "Piso"

    out["metros_cuadrados"] = pd.to_numeric(df["metros_cuadrados"], errors="raise").astype(float)
    out["habitaciones"] = pd.to_numeric(df["habitaciones"], errors="raise").astype(float)
    out["banos"] = pd.to_numeric(df["banos"], errors="raise").astype(float)
    out["planta"] = pd.to_numeric(df["planta"], errors="raise").astype(float)

    optional_bools = {"aire_acondicionado", "trastero", "necesita_reforma", "obra_nueva"}
    for bool_col in (
        "ascensor",
        "exterior",
        "terraza",
        "garaje",
        "aire_acondicionado",
        "trastero",
        "necesita_reforma",
        "obra_nueva",
    ):
        if bool_col in df.columns:
            out[bool_col] = df[bool_col].fillna(False).astype(bool).astype(int)
        elif bool_col in optional_bools:
            out[bool_col] = 0
        else:
            out[bool_col] = 1

    out["m2_por_habitacion"] = out["metros_cuadrados"] / np.maximum(out["habitaciones"], 1.0)
    out["sin_ascensor_planta_alta"] = ((out["ascensor"] == 0) & (out["planta"] >= 3)).astype(int)
    out["es_bajo"] = (out["planta"] == 0).astype(int)

    return out[ALL_MODEL_FEATURES]
