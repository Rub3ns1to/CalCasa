"""Generador reproducible de un dataset realista de viviendas en Madrid.

Permite entrenar y probar el sistema de principio a fin sin depender de descargas
manuales externas, manteniendo compatibilidad total con datasets públicos de Kaggle.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from tasador.ml.features import DISTRICT_WIDE_BARRIO, MADRID_DISTRICTS  # noqa: E402

# Pesos relativos de oferta residencial aproximada por distrito en Madrid
DISTRICT_WEIGHTS: dict[str, float] = {
    "Salamanca": 1.35,
    "Chamberí": 1.25,
    "Retiro": 1.10,
    "Centro": 1.45,
    "Chamartín": 1.25,
    "Moncloa-Aravaca": 1.05,
    "Arganzuela": 1.15,
    "Tetuán": 1.20,
    "Hortaleza": 1.15,
    "Fuencarral-El Pardo": 1.10,
    "Ciudad Lineal": 1.15,
    "Barajas": 0.65,
    "San Blas-Canillejas": 1.00,
    "Moratalaz": 0.75,
    "Latina": 1.10,
    "Villa de Vallecas": 0.85,
    "Carabanchel": 1.20,
    "Vicálvaro": 0.65,
    "Usera": 0.90,
    "Puente de Vallecas": 1.15,
    "Villaverde": 0.85,
}


def generate_dataset(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Genera ``n_rows`` viviendas sintéticas calibradas al mercado madrileño."""
    rng = np.random.default_rng(seed)

    districts = list(MADRID_DISTRICTS.keys())
    raw_weights = np.array([DISTRICT_WEIGHTS[d] for d in districts], dtype=float)
    probs = raw_weights / raw_weights.sum()

    chosen_districts = rng.choice(districts, size=n_rows, p=probs)

    rows: list[dict[str, object]] = []
    for barrio in chosen_districts:
        base_m2 = MADRID_DISTRICTS[barrio]

        # Distribución de superficie según perfil del distrito
        if barrio in {"Salamanca", "Chamartín", "Moncloa-Aravaca", "Retiro"}:
            mean_area, std_area = 118.0, 44.0
        elif barrio in {"Centro", "Tetuán", "Chamberí"}:
            mean_area, std_area = 84.0, 34.0
        elif barrio in {"Fuencarral-El Pardo", "Hortaleza", "Arganzuela", "Barajas"}:
            mean_area, std_area = 96.0, 30.0
        else:
            mean_area, std_area = 78.0, 24.0

        area = float(np.clip(rng.normal(mean_area, std_area), 26.0, 290.0))
        area = round(area, 1)

        # Habitaciones coherentes con la superficie
        if area < 42:
            habitaciones = int(rng.choice([0, 1], p=[0.35, 0.65]))
        elif area < 66:
            habitaciones = int(rng.choice([1, 2], p=[0.42, 0.58]))
        elif area < 95:
            habitaciones = int(rng.choice([2, 3], p=[0.45, 0.55]))
        elif area < 140:
            habitaciones = int(rng.choice([3, 4], p=[0.55, 0.45]))
        else:
            habitaciones = int(rng.choice([4, 5], p=[0.65, 0.35]))

        # Baños coherentes con habitaciones y superficie
        if area < 75:
            banos = int(rng.choice([1, 2], p=[0.85, 0.15]))
        elif area < 135:
            banos = int(rng.choice([1, 2], p=[0.28, 0.72]))
        else:
            banos = int(rng.choice([2, 3, 4], p=[0.52, 0.38, 0.10]))

        planta = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], p=[0.10, 0.19, 0.20, 0.18, 0.13, 0.09, 0.06, 0.03, 0.02]))

        # Ascensor más común en plantas altas y distritos nuevos/prime
        lift_prob = 0.88 if barrio in {"Salamanca", "Chamartín", "Retiro", "Fuencarral-El Pardo", "Hortaleza"} else 0.72
        if planta >= 5:
            lift_prob = max(lift_prob, 0.90)
        elif planta == 0:
            lift_prob *= 0.75
        ascensor = bool(rng.random() < lift_prob)

        exterior_prob = 0.68 if barrio in {"Centro", "Chamberí", "Tetuán"} else 0.84
        exterior = bool(rng.random() < exterior_prob)

        terraza_prob = 0.22 + (0.14 if planta >= 5 else 0.0) + (0.08 if base_m2 < 4800 else 0.0)
        terraza = bool(rng.random() < terraza_prob)

        garage_prob = (
            0.14
            if barrio in {"Centro", "Chamberí"}
            else 0.28
            if barrio in {"Salamanca", "Retiro", "Tetuán"}
            else 0.52
        )
        garaje = bool(rng.random() < garage_prob)

        # Precio de mercado latente con efectos no lineales realistas
        price_m2 = float(base_m2)

        # Prima de liquidez en pisos compactos y ligero ajuste en grandes superficies
        if area < 50:
            price_m2 *= 1.06
        elif area > 170:
            price_m2 *= 1.03

        # Factor planta
        floor_factors = {0: 0.92, 1: 0.985, 2: 1.00, 3: 1.015, 4: 1.03, 5: 1.045}
        price_m2 *= floor_factors.get(planta, 1.065)

        # Penalización por falta de ascensor en plantas altas
        if not ascensor:
            if planta >= 5:
                price_m2 *= 0.81
            elif planta >= 3:
                price_m2 *= 0.86
            elif planta == 2:
                price_m2 *= 0.95

        # Exterior / interior
        price_m2 *= 1.055 if exterior else 0.91

        # Terraza y baños adicionales
        if terraza:
            price_m2 *= 1.06
        if banos >= 2 and area >= 75:
            price_m2 *= 1.025

        # Garaje suma prima fija escalada por zona
        garage_value = (
            32000.0 if base_m2 >= 6000 else 21000.0 if base_m2 >= 4000 else 13000.0
        ) if garaje else 0.0

        noise = float(np.exp(rng.normal(0.0, 0.06)))
        raw_price = (price_m2 * area + garage_value) * noise
        precio = int(max(55000, round(raw_price / 500.0) * 500))

        rows.append(
            {
                "barrio": barrio,
                "micro_barrio": DISTRICT_WIDE_BARRIO,
                "tipo_vivienda": "Ático" if (planta >= 6 and terraza) else "Piso",
                "metros_cuadrados": area,
                "habitaciones": habitaciones,
                "banos": banos,
                "planta": planta,
                "ascensor": ascensor,
                "exterior": exterior,
                "terraza": terraza,
                "garaje": garaje,
                "aire_acondicionado": False,
                "trastero": False,
                "necesita_reforma": False,
                "obra_nueva": False,
                "precio": precio,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el dataset de viviendas de Madrid.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "madrid_housing_sample.csv",
        help="Ruta del archivo CSV de salida.",
    )
    parser.add_argument("--rows", type=int, default=4000, help="Número de viviendas a generar.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria reproducible.")
    args = parser.parse_args()

    df = generate_dataset(n_rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Dataset generado con {len(df)} viviendas en: {args.output}")
    print(f"Distritos incluidos: {df['barrio'].nunique()} | Precio mediano: {int(df['precio'].median()):,} €")


if __name__ == "__main__":
    main()
