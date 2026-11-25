"""Carga e inferencia del modelo tabular de valoración inmobiliaria."""

import gzip
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from tasador.ml.features import (
    MADRID_DISTRICTS,
    build_feature_dataframe,
    normalize_district_name,
    normalize_micro_barrio,
)
from tasador.schemas import ComparableProperty, PropertyInput, TabularFactor


class TabularPricePredictor:
    """Envuelve el pipeline XGBoost entrenado, metadatos y testigos comparables."""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        comparables_path: Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.comparables_path = (
            Path(comparables_path)
            if comparables_path is not None
            else self.model_path.parent / "comparables_index.json.gz"
        )
        self._pipeline: Any = None
        self._metadata: dict[str, Any] = {}
        self._comparables_index: dict[str, Any] = {}

    def load(self) -> None:
        """Carga el modelo serializado, metadatos e índice de comparables desde disco."""
        if self.model_path.exists():
            self._pipeline = joblib.load(self.model_path)
        else:
            self._pipeline = None

        if self.metadata_path.exists():
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        else:
            self._metadata = {}

        if self.comparables_path.exists():
            with gzip.open(self.comparables_path, "rt", encoding="utf-8") as gz_file:
                self._comparables_index = json.load(gz_file)
        else:
            self._comparables_index = {}

    @property
    def is_loaded(self) -> bool:
        """Indica si el pipeline tabular está listo para realizar predicciones."""
        return self._pipeline is not None

    @property
    def market_update_factor(self) -> float:
        """Factor de actualización temporal aplicado sobre la predicción base."""
        return float(self._metadata.get("market_update_factor", 1.0))

    def get_metadata(self) -> dict[str, Any]:
        """Devuelve una copia de los metadatos del modelo tabular."""
        return dict(self._metadata)

    def get_district_median_m2(
        self,
        barrio: str,
        micro_barrio: str | None = None,
    ) -> int | None:
        """Devuelve la mediana de €/m² del micro-barrio (si existe) o del distrito."""
        canonical = normalize_district_name(barrio)
        if micro_barrio:
            clean_micro = normalize_micro_barrio(micro_barrio)
            neighborhoods = self._metadata.get("neighborhoods_by_district", {}).get(canonical, [])
            for item in neighborhoods:
                if item.get("nombre", "").lower() == clean_micro.lower():
                    return int(round(float(item["mediana_m2"])))

        medians = self._metadata.get("district_median_m2", {})
        if canonical in medians:
            return int(round(float(medians[canonical])))
        if canonical in MADRID_DISTRICTS:
            return MADRID_DISTRICTS[canonical]
        return None

    def get_available_districts(self) -> list[dict[str, Any]]:
        """Lista distritos ordenados alfabéticamente con su mediana €/m² y barrios."""
        medians = self._metadata.get("district_median_m2", {})
        neighborhoods_map = self._metadata.get("neighborhoods_by_district", {})
        district_names = sorted(set(MADRID_DISTRICTS.keys()) | set(medians.keys()))
        return [
            {
                "nombre": name,
                "mediana_m2": int(round(float(medians.get(name, MADRID_DISTRICTS.get(name, 0))))),
                "barrios": neighborhoods_map.get(name, []),
            }
            for name in district_names
        ]

    @staticmethod
    def _to_row_dict(prop: PropertyInput, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        row = prop.model_dump()
        # Baseline en estado estándar para no duplicar el ajuste visual de reforma
        row["necesita_reforma"] = False
        row["obra_nueva"] = False
        if overrides:
            row.update(overrides)
        return row

    def predict_base_price(self, prop: PropertyInput) -> tuple[float, float]:
        """Predice el precio base (EUR) y la banda de incertidumbre relativa.

        Returns:
            Tupla ``(precio_base_eur, banda_incertidumbre_pct_decimal)``.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"El modelo tabular no está cargado. Ejecuta 'python scripts/train_model.py' "
                f"para generar {self.model_path}."
            )

        raw_df = pd.DataFrame([self._to_row_dict(prop)])
        features_df = build_feature_dataframe(raw_df)
        raw_predicted = float(self._pipeline.predict(features_df)[0])
        predicted_price = max(raw_predicted * self.market_update_factor, 25000.0)

        medape_pct = self._metadata.get("metrics", {}).get("test_medape_pct")
        if medape_pct is None:
            uncertainty_band = 0.06
        else:
            uncertainty_band = max(0.045, min(0.085, float(medape_pct) / 100.0))

        return predicted_price, round(uncertainty_band, 4)

    def explain_base_price(self, prop: PropertyInput, base_price: float) -> list[TabularFactor]:
        """Estima la contribución local de cada característica clave por contrafactuales."""
        if not self.is_loaded:
            return []

        scenarios: list[tuple[str, dict[str, Any]]] = []
        if prop.micro_barrio:
            scenarios.append((f"Barrio {prop.micro_barrio}", {"micro_barrio": None}))
        if prop.tipo_vivienda != "Piso":
            scenarios.append((f"Tipología: {prop.tipo_vivienda}", {"tipo_vivienda": "Piso"}))

        scenarios.append(
            (
                "Vivienda exterior" if prop.exterior else "Orientación interior",
                {"exterior": not prop.exterior},
            )
        )
        if prop.planta >= 2 and prop.tipo_vivienda != "Casa o chalet":
            scenarios.append(
                (
                    "Con ascensor" if prop.ascensor else f"Sin ascensor (planta {prop.planta})",
                    {"ascensor": not prop.ascensor},
                )
            )
        if prop.terraza:
            scenarios.append(("Terraza", {"terraza": False}))
        if prop.garaje:
            scenarios.append(("Plaza de garaje", {"garaje": False}))
        if prop.aire_acondicionado:
            scenarios.append(("Aire acondicionado", {"aire_acondicionado": False}))
        if prop.trastero:
            scenarios.append(("Trastero", {"trastero": False}))

        if not scenarios:
            return []

        alt_rows = [self._to_row_dict(prop, overrides) for _, overrides in scenarios]
        alt_features = build_feature_dataframe(pd.DataFrame(alt_rows))
        alt_preds = self._pipeline.predict(alt_features)

        factors: list[TabularFactor] = []
        factor_scale = self.market_update_factor
        for (label, _), alt_raw in zip(scenarios, alt_preds, strict=True):
            alt_price = max(float(alt_raw) * factor_scale, 25000.0)
            diff = base_price - alt_price
            impact_eur = int(round(diff / 500.0) * 500)
            if abs(impact_eur) < 1500:
                continue
            impact_pct = round((diff / max(alt_price, 1.0)) * 100.0, 1)
            factors.append(
                TabularFactor(
                    factor=label,
                    impacto_eur=impact_eur,
                    impacto_pct=impact_pct,
                )
            )

        factors.sort(key=lambda item: abs(item.impacto_eur), reverse=True)
        return factors[:5]

    def find_comparables(self, prop: PropertyInput, limit: int = 4) -> list[ComparableProperty]:
        """Busca los ``limit`` testigos anonimizados más similares de la zona."""
        rows = self._comparables_index.get("rows")
        if not rows:
            return []

        districts: list[str] = self._comparables_index.get("districts", [])
        micro_barrios: list[str] = self._comparables_index.get("micro_barrios", [])
        tipos: list[str] = self._comparables_index.get("tipos", [])

        canonical_dist = normalize_district_name(prop.barrio)
        if canonical_dist not in districts:
            return []
        target_dist_idx = districts.index(canonical_dist)

        district_rows = [r for r in rows if r[0] == target_dist_idx]
        if not district_rows:
            return []

        candidates = district_rows
        if prop.micro_barrio:
            clean_micro = normalize_micro_barrio(prop.micro_barrio).lower()
            matching_micro_idxs = {
                idx for idx, name in enumerate(micro_barrios) if name.lower() == clean_micro
            }
            if matching_micro_idxs:
                micro_rows = [r for r in district_rows if r[1] in matching_micro_idxs]
                if len(micro_rows) >= limit:
                    candidates = micro_rows

        area_ref = max(prop.metros_cuadrados, 30.0)
        scored: list[tuple[float, list[int]]] = []
        for r in candidates:
            _, _, tipo_idx, m2, hab, banos, planta, asc, ext, _precio = r
            tipo_name = tipos[tipo_idx] if 0 <= tipo_idx < len(tipos) else "Piso"
            dist = (
                3.0 * abs(m2 - prop.metros_cuadrados) / area_ref
                + 0.55 * abs(hab - prop.habitaciones)
                + 0.35 * abs(banos - prop.banos)
                + (0.0 if tipo_name == prop.tipo_vivienda else 0.7)
                + (0.0 if bool(ext) == prop.exterior else 0.35)
                + (0.0 if bool(asc) == prop.ascensor else 0.4)
                + 0.06 * abs(planta - prop.planta)
            )
            scored.append((dist, r))

        scored.sort(key=lambda item: item[0])

        results: list[ComparableProperty] = []
        seen_signatures: set[tuple[Any, ...]] = set()
        for dist, r in scored:
            _, micro_idx, tipo_idx, m2, hab, banos, planta, asc, ext, precio = r
            micro_name = micro_barrios[micro_idx] if 0 <= micro_idx < len(micro_barrios) else None
            tipo_name = tipos[tipo_idx] if 0 <= tipo_idx < len(tipos) else "Piso"

            sig = (micro_name, tipo_name, m2, hab, banos, planta, precio)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            similitud_pct = int(round(max(55.0, min(99.0, 100.0 / (1.0 + dist * 0.75)))))
            results.append(
                ComparableProperty(
                    distrito=canonical_dist,
                    micro_barrio=micro_name,
                    tipo_vivienda=tipo_name,
                    metros_cuadrados=m2,
                    habitaciones=hab,
                    banos=banos,
                    planta=planta,
                    ascensor=bool(asc),
                    exterior=bool(ext),
                    precio_estimado=precio,
                    precio_m2=round(precio / max(m2, 1)),
                    similitud_pct=similitud_pct,
                )
            )
            if len(results) >= limit:
                break

        return results
