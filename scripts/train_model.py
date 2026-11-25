"""Entrenamiento y benchmark de modelos tabulares para tasación inmobiliaria en Madrid."""

import argparse
import gzip
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from tasador.ml.features import (  # noqa: E402
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    DISTRICT_WIDE_BARRIO,
    MADRID_DISTRICTS,
    NUMERIC_FEATURES,
    build_feature_dataframe,
    normalize_district_name,
    normalize_micro_barrio,
)

COLUMN_ALIASES: dict[str, list[str]] = {
    "barrio": ["barrio", "district", "neighborhood_id", "neighborhood", "zona"],
    "micro_barrio": ["micro_barrio", "neighborhood_name"],
    "tipo_vivienda": ["tipo_vivienda", "house_type_id", "property_type"],
    "metros_cuadrados": ["metros_cuadrados", "sq_mt_built", "sq_mt_useful", "area", "surface"],
    "habitaciones": ["habitaciones", "n_rooms", "rooms", "bedrooms"],
    "banos": ["banos", "baños", "n_bathrooms", "bathrooms"],
    "planta": ["planta", "floor"],
    "ascensor": ["ascensor", "has_lift", "lift", "elevator"],
    "exterior": ["exterior", "is_exterior"],
    "terraza": ["terraza", "has_terrace"],
    "garaje": ["garaje", "has_parking", "is_parking_included_in_price"],
    "aire_acondicionado": ["aire_acondicionado", "has_ac"],
    "trastero": ["trastero", "has_storage_room"],
    "necesita_reforma": ["necesita_reforma", "is_renewal_needed"],
    "obra_nueva": ["obra_nueva", "is_new_development"],
    "precio": ["precio", "buy_price", "price"],
}

KAGGLE_NEIGHBORHOOD_RE = re.compile(
    r"Neighborhood\s+\d+:\s+(.+?)\s+\([^)]*\)\s+-\s+District\s+\d+:\s+(.+)"
)


def _find_column(columns: pd.Index, candidates: list[str]) -> str | None:
    lower_map = {col.lower().strip(): col for col in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _parse_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(int(default)).astype(float).gt(0.5)
    true_vals = {"1", "true", "si", "sí", "yes", "y", "s", "t"}
    return series.fillna(str(default)).astype(str).str.strip().str.lower().isin(true_vals)


def _parse_house_type_value(val: Any) -> str:
    if pd.isna(val):
        return "Piso"
    txt = str(val).strip().lower()
    if "ático" in txt or "atico" in txt:
        return "Ático"
    if "dúplex" in txt or "duplex" in txt:
        return "Dúplex"
    if "casa" in txt or "chalet" in txt:
        return "Casa o chalet"
    return "Piso"


def _parse_floor_value(val: Any, default_floor: float = 2.0) -> float:
    if pd.isna(val):
        return default_floor
    if isinstance(val, (int, float, np.number)):
        return float(max(0, min(40, int(round(val)))))
    txt = str(val).strip().lower()
    if any(token in txt for token in ("bajo", "entreplanta", "sotano", "sótano", "ground")):
        return 0.0
    if "atico" in txt or "ático" in txt:
        return 6.0
    match = re.search(r"-?\d+", txt)
    if match:
        return float(max(0, min(40, int(match.group(0)))))
    return default_floor


def _parse_location_columns(raw_df: pd.DataFrame, district_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    explicit_micro_col = _find_column(raw_df.columns, COLUMN_ALIASES["micro_barrio"])
    districts: list[str] = []
    micro_barrios: list[str] = []

    for idx, val in district_series.items():
        txt = "" if pd.isna(val) else str(val).strip()
        match = KAGGLE_NEIGHBORHOOD_RE.search(txt)
        if match:
            micro = normalize_micro_barrio(match.group(1))
            dist = normalize_district_name(match.group(2))
        else:
            paren_match = re.search(r"\(([^()]+)\)\s*$", txt)
            if paren_match and normalize_district_name(paren_match.group(1)) in MADRID_DISTRICTS:
                dist = normalize_district_name(paren_match.group(1))
                micro = DISTRICT_WIDE_BARRIO
            else:
                if ":" in txt:
                    txt = txt.split(":", maxsplit=1)[1].strip()
                dist = normalize_district_name(txt)
                micro = DISTRICT_WIDE_BARRIO

        if explicit_micro_col is not None:
            micro = normalize_micro_barrio(raw_df.at[idx, explicit_micro_col])

        districts.append(dist)
        micro_barrios.append(micro)

    return pd.Series(districts, index=raw_df.index), pd.Series(micro_barrios, index=raw_df.index)


def normalize_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza tanto el CSV sintético como el dataset real de Kaggle Madrid."""
    work_df = raw_df.copy()
    if "operation" in work_df.columns:
        work_df = work_df[work_df["operation"].astype(str).str.lower().str.strip() == "sale"].copy()

    resolved: dict[str, pd.Series] = {}
    for required in ("barrio", "metros_cuadrados", "habitaciones", "precio"):
        col = _find_column(work_df.columns, COLUMN_ALIASES[required])
        if col is None:
            raise ValueError(
                f"El dataset no contiene ninguna columna equivalente a '{required}'. "
                f"Columnas disponibles: {list(work_df.columns)}"
            )
        resolved[required] = work_df[col]

    distritos, micro_barrios = _parse_location_columns(work_df, resolved["barrio"])

    tipo_col = _find_column(work_df.columns, COLUMN_ALIASES["tipo_vivienda"])
    tipo_series = (
        work_df[tipo_col].map(_parse_house_type_value)
        if tipo_col
        else pd.Series("Piso", index=work_df.index)
    )

    banos_col = _find_column(work_df.columns, COLUMN_ALIASES["banos"])
    planta_col = _find_column(work_df.columns, COLUMN_ALIASES["planta"])
    if planta_col:
        planta_series = pd.Series(
            [
                _parse_floor_value(
                    work_df.at[idx, planta_col],
                    default_floor=0.0 if tipo_series.at[idx] == "Casa o chalet" else 2.0,
                )
                for idx in work_df.index
            ],
            index=work_df.index,
        )
    else:
        planta_series = pd.Series(2.0, index=work_df.index)

    ascensor_col = _find_column(work_df.columns, COLUMN_ALIASES["ascensor"])
    if ascensor_col:
        ascensor_vals: list[bool] = []
        raw_lift = work_df[ascensor_col]
        for idx in work_df.index:
            val = raw_lift.at[idx]
            if pd.isna(val):
                ascensor_vals.append(
                    not (tipo_series.at[idx] == "Casa o chalet" or planta_series.at[idx] == 0.0)
                )
            else:
                ascensor_vals.append(bool(_parse_bool_series(pd.Series([val])).iloc[0]))
        ascensor_series = pd.Series(ascensor_vals, index=work_df.index)
    else:
        ascensor_series = pd.Series(True, index=work_df.index)

    exterior_col = _find_column(work_df.columns, COLUMN_ALIASES["exterior"])
    terraza_col = _find_column(work_df.columns, COLUMN_ALIASES["terraza"])
    garaje_col = _find_column(work_df.columns, COLUMN_ALIASES["garaje"])
    ac_col = _find_column(work_df.columns, COLUMN_ALIASES["aire_acondicionado"])
    trastero_col = _find_column(work_df.columns, COLUMN_ALIASES["trastero"])
    reforma_col = _find_column(work_df.columns, COLUMN_ALIASES["necesita_reforma"])
    obra_col = _find_column(work_df.columns, COLUMN_ALIASES["obra_nueva"])

    df = pd.DataFrame(
        {
            "barrio": distritos,
            "micro_barrio": micro_barrios,
            "tipo_vivienda": tipo_series,
            "metros_cuadrados": pd.to_numeric(resolved["metros_cuadrados"], errors="coerce"),
            "habitaciones": pd.to_numeric(resolved["habitaciones"], errors="coerce"),
            "banos": pd.to_numeric(work_df[banos_col], errors="coerce").fillna(1).clip(1, 10)
            if banos_col
            else 1,
            "planta": planta_series,
            "ascensor": ascensor_series,
            "exterior": _parse_bool_series(work_df[exterior_col], default=True)
            if exterior_col
            else True,
            "terraza": _parse_bool_series(work_df[terraza_col], default=False)
            if terraza_col
            else False,
            "garaje": _parse_bool_series(work_df[garaje_col], default=False)
            if garaje_col
            else False,
            "aire_acondicionado": _parse_bool_series(work_df[ac_col], default=False)
            if ac_col
            else False,
            "trastero": _parse_bool_series(work_df[trastero_col], default=False)
            if trastero_col
            else False,
            "necesita_reforma": _parse_bool_series(work_df[reforma_col], default=False)
            if reforma_col
            else False,
            "obra_nueva": _parse_bool_series(work_df[obra_col], default=False)
            if obra_col
            else False,
            "precio": pd.to_numeric(resolved["precio"], errors="coerce"),
        }
    )

    df = df.dropna(subset=["barrio", "metros_cuadrados", "habitaciones", "precio"]).copy()
    price_m2 = df["precio"] / np.maximum(df["metros_cuadrados"], 1.0)
    df = df[
        (df["metros_cuadrados"] >= 15.0)
        & (df["metros_cuadrados"] <= 1000.0)
        & (df["habitaciones"] >= 0)
        & (df["habitaciones"] <= 15)
        & (df["banos"] >= 1)
        & (df["banos"] <= 10)
        & (df["precio"] >= 30000.0)
        & (df["precio"] <= 5_000_000.0)
        & (price_m2 >= 900.0)
        & (price_m2 <= 15_000.0)
    ].reset_index(drop=True)
    return df


def _build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    num_transformer: Any = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", num_transformer, NUMERIC_FEATURES),
        ]
    )


def build_ridge_pipeline() -> Pipeline:
    """Modelo lineal regularizado (baseline) con objetivo logarítmico."""
    return Pipeline(
        [
            ("prep", _build_preprocessor(scale_numeric=True)),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=Ridge(alpha=1.0),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def build_histgb_pipeline(seed: int = 42) -> Pipeline:
    """Histogram Gradient Boosting de scikit-learn para comparativa."""
    return Pipeline(
        [
            ("prep", _build_preprocessor(scale_numeric=False)),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=HistGradientBoostingRegressor(
                        max_iter=350,
                        max_depth=8,
                        learning_rate=0.06,
                        random_state=seed,
                    ),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def build_training_pipeline(seed: int = 42) -> Pipeline:
    """Construye el Pipeline principal XGBoost con target logarítmico."""
    xgb = XGBRegressor(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=2.0,
        min_child_weight=3,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("prep", _build_preprocessor(scale_numeric=False)),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=xgb,
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_pct_err = np.abs((y_true - y_pred) / np.maximum(y_true, 1.0)) * 100.0
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae_eur": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse_eur": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "mape_pct": round(float(np.mean(abs_pct_err)), 2),
        "medape_pct": round(float(np.median(abs_pct_err)), 2),
    }


def _compute_grouped_importances(pipeline: Pipeline) -> list[dict[str, Any]]:
    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pipeline.named_steps["model"].regressor_.feature_importances_

    group_map = {
        "cat__barrio": "Ubicación (Distrito y Barrio)",
        "cat__micro_barrio": "Ubicación (Distrito y Barrio)",
        "cat__tipo_vivienda": "Tipo de vivienda",
        "num__metros_cuadrados": "Superficie y distribución",
        "num__habitaciones": "Superficie y distribución",
        "num__m2_por_habitacion": "Superficie y distribución",
        "num__banos": "Baños",
        "num__planta": "Planta y ascensor",
        "num__ascensor": "Planta y ascensor",
        "num__sin_ascensor_planta_alta": "Planta y ascensor",
        "num__es_bajo": "Planta y ascensor",
        "num__necesita_reforma": "Estado / Obra nueva",
        "num__obra_nueva": "Estado / Obra nueva",
        "num__garaje": "Garaje",
        "num__exterior": "Orientación exterior",
        "num__terraza": "Terraza",
        "num__aire_acondicionado": "Aire acondicionado",
        "num__trastero": "Trastero",
    }

    totals: dict[str, float] = {}
    for fname, imp in zip(feature_names, importances, strict=True):
        label = "Otras"
        for prefix, grp in group_map.items():
            if fname.startswith(prefix):
                label = grp
                break
        totals[label] = totals.get(label, 0.0) + float(imp)

    grand_total = sum(totals.values()) or 1.0
    grouped = [
        {"grupo": k, "importancia_pct": round((v / grand_total) * 100.0, 1)}
        for k, v in totals.items()
    ]
    return sorted(grouped, key=lambda item: item["importancia_pct"], reverse=True)


def train_and_save(
    dataset_path: Path,
    output_dir: Path,
    seed: int = 42,
    market_factor: float | None = None,
) -> dict[str, Any]:
    """Entrena, compara modelos, evalúa y persiste el pipeline ganador."""
    raw_df = pd.read_csv(dataset_path)
    df = normalize_raw_dataframe(raw_df)

    if market_factor is None:
        market_factor = 1.35 if "houses_madrid" in dataset_path.name.lower() else 1.0

    X = build_feature_dataframe(df)
    y = df["precio"].to_numpy(dtype=float)

    can_stratify = X["barrio"].value_counts().min() >= 5
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=X["barrio"] if can_stratify else None,
    )

    # Benchmark de 3 familias de modelos sobre el mismo split holdout
    benchmark_specs = [
        ("Ridge (Baseline)", build_ridge_pipeline()),
        ("HistGradientBoosting", build_histgb_pipeline(seed=seed)),
        ("XGBoost (Seleccionado)", build_training_pipeline(seed=seed)),
    ]
    benchmark_results: list[dict[str, Any]] = []
    xgb_test_metrics: dict[str, float] = {}

    for model_name, pipe in benchmark_specs:
        pipe.fit(X_train, y_train)
        preds = np.maximum(pipe.predict(X_test), 20000.0)
        m = _compute_regression_metrics(y_test, preds)
        benchmark_results.append({"modelo": model_name, **m})
        if model_name.startswith("XGBoost"):
            xgb_test_metrics = m

    # Validación cruzada 5-fold del modelo elegido sobre X_train (sin fuga de test)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_preds = cross_val_predict(build_training_pipeline(seed=seed), X_train, y_train, cv=cv)
    cv_metrics = _compute_regression_metrics(y_train, np.maximum(cv_preds, 20000.0))

    # Reentrenamiento final con el 100 % de las muestras limpias para despliegue
    final_pipeline = build_training_pipeline(seed=seed)
    final_pipeline.fit(X, y)

    price_per_m2_current = (df["precio"] / df["metros_cuadrados"]) * market_factor
    district_medians = {
        str(k): int(round(float(v)))
        for k, v in price_per_m2_current.groupby(df["barrio"]).median().items()
    }

    neighborhoods_by_district: dict[str, list[dict[str, Any]]] = {}
    grouped_micro = df.assign(pm2=price_per_m2_current).groupby(["barrio", "micro_barrio"])["pm2"]
    for (dist, micro), series in grouped_micro:
        if micro == DISTRICT_WIDE_BARRIO:
            continue
        neighborhoods_by_district.setdefault(str(dist), []).append(
            {
                "nombre": str(micro),
                "mediana_m2": int(round(float(series.median()))),
                "muestras": int(len(series)),
            }
        )
    for dist in neighborhoods_by_district:
        neighborhoods_by_district[dist].sort(key=lambda item: item["nombre"])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "price_model.joblib"
    metadata_path = output_dir / "model_metadata.json"
    comparables_path = output_dir / "comparables_index.json.gz"

    joblib.dump(final_pipeline, model_path, compress=3)

    # Índice compacto y 100 % anonimizado de testigos reales para consulta en inferencia
    districts_vocab = sorted(df["barrio"].astype(str).unique().tolist())
    micro_vocab = sorted(
        m for m in df["micro_barrio"].astype(str).unique().tolist() if m != DISTRICT_WIDE_BARRIO
    )
    tipos_vocab = sorted(df["tipo_vivienda"].astype(str).unique().tolist())

    dist_idx_map = {name: idx for idx, name in enumerate(districts_vocab)}
    micro_idx_map = {name: idx for idx, name in enumerate(micro_vocab)}
    tipo_idx_map = {name: idx for idx, name in enumerate(tipos_vocab)}

    scaled_prices = np.round((df["precio"].to_numpy(dtype=float) * market_factor) / 500.0) * 500.0
    comp_rows: list[list[int]] = []
    for row_idx, row in enumerate(df.itertuples(index=False)):
        comp_rows.append(
            [
                dist_idx_map[str(row.barrio)],
                micro_idx_map.get(str(row.micro_barrio), -1),
                tipo_idx_map[str(row.tipo_vivienda)],
                int(round(float(row.metros_cuadrados))),
                int(row.habitaciones),
                int(row.banos),
                int(round(float(row.planta))),
                1 if bool(row.ascensor) else 0,
                1 if bool(row.exterior) else 0,
                int(scaled_prices[row_idx]),
            ]
        )

    comparables_payload = {
        "districts": districts_vocab,
        "micro_barrios": micro_vocab,
        "tipos": tipos_vocab,
        "rows": comp_rows,
    }
    with gzip.open(comparables_path, "wt", encoding="utf-8") as gz_file:
        json.dump(comparables_payload, gz_file, ensure_ascii=False, separators=(",", ":"))

    year_note = (
        f"Dataset Kaggle Madrid ({len(df):,} viviendas) actualizado con factor de mercado ×{market_factor:.2f}"
        if market_factor != 1.0
        else f"Dataset calibrado ({len(df):,} viviendas)"
    )

    metadata: dict[str, Any] = {
        "algorithm": "XGBRegressor + TransformedTargetRegressor(log1p)",
        "dataset_source": str(dataset_path.name),
        "dataset_year_note": year_note,
        "market_update_factor": round(float(market_factor), 4),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "total_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_districts": int(df["barrio"].nunique()),
        "n_neighborhoods": int(
            sum(len(items) for items in neighborhoods_by_district.values())
        ),
        "features": ALL_MODEL_FEATURES,
        "metrics": {
            "test_r2": xgb_test_metrics["r2"],
            "test_mae_eur": xgb_test_metrics["mae_eur"],
            "test_rmse_eur": xgb_test_metrics["rmse_eur"],
            "test_mape_pct": xgb_test_metrics["mape_pct"],
            "test_medape_pct": xgb_test_metrics["medape_pct"],
            "cv5_r2": cv_metrics["r2"],
            "cv5_mape_pct": cv_metrics["mape_pct"],
        },
        "benchmark": benchmark_results,
        "feature_importances": _compute_grouped_importances(final_pipeline),
        "district_median_m2": district_medians,
        "neighborhoods_by_district": neighborhoods_by_district,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo tabular de tasación en Madrid.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "madrid_housing_sample.csv",
        help="Ruta al CSV de entrenamiento.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "models",
        help="Carpeta donde guardar price_model.joblib y model_metadata.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument(
        "--market-factor",
        type=float,
        default=None,
        help="Factor de actualización temporal sobre los precios del CSV (por defecto 1.35 para houses_Madrid.csv).",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"No existe el dataset '{args.dataset}'. Ejecuta primero scripts/generate_madrid_dataset.py."
        )

    metadata = train_and_save(
        args.dataset,
        args.output_dir,
        seed=args.seed,
        market_factor=args.market_factor,
    )
    m = metadata["metrics"]
    print("=== Entrenamiento completado ===")
    print(
        f"Dataset           : {metadata['dataset_source']} | Registros={metadata['total_rows']} "
        f"(train={metadata['training_rows']}, test={metadata['test_rows']})"
    )
    print(
        f"Cobertura         : {metadata['n_districts']} distritos | "
        f"{metadata['n_neighborhoods']} barrios | Factor mercado=x{metadata['market_update_factor']}"
    )
    print(f"Test R²           : {m['test_r2']} | 5-Fold CV R²: {m['cv5_r2']}")
    print(f"Test MAE / RMSE   : {m['test_mae_eur']:,.0f} € / {m['test_rmse_eur']:,.0f} €")
    print(f"Test MAPE / MedAPE: {m['test_mape_pct']}% / {m['test_medape_pct']}% | CV MAPE: {m['cv5_mape_pct']}%")
    print("Benchmark holdout :")
    for row in metadata["benchmark"]:
        print(
            f"  - {row['modelo']:<24} R²={row['r2']:<6} MAPE={row['mape_pct']:<5}% "
            f"MedAPE={row['medape_pct']:<5}% MAE={row['mae_eur']:,.0f} €"
        )
    print(f"Artefactos en     : {args.output_dir}")


if __name__ == "__main__":
    main()
