"""Orquestación de la tasación multimodal (modelo tabular + ajuste visual)."""

from tasador.config import Settings
from tasador.ml.predictor import TabularPricePredictor
from tasador.schemas import (
    AppraisalResponse,
    PropertyInput,
    TabularFactor,
    ValuationBreakdown,
    VisionAssessment,
)
from tasador.vision.heuristic_fallback import prepare_image_bytes
from tasador.vision.openrouter_client import OpenRouterVisionClient


def round_to_nearest_500(value: float) -> int:
    """Redondea importes inmobiliarios al múltiplo de 500 € más cercano."""
    return int(round(value / 500.0) * 500)


def calculate_visual_adjustment(
    reforma: int,
    luminosidad: int,
    calidades: int,
    max_adjustment_pct: float = 15.0,
) -> tuple[float, float]:
    """Calcula el score ponderado (1-10) y el ajuste porcentual sobre el precio base.

    Ponderación:
        - Reforma: 45 %
        - Calidades: 30 %
        - Luminosidad: 25 %

    Un score de 5.5 actúa como punto neutro (0 % de ajuste). El extremo inferior
    (1.0) aplica ``-max_adjustment_pct`` y el superior (10.0) aplica ``+max_adjustment_pct``.
    """
    weighted_score = round(0.45 * reforma + 0.30 * calidades + 0.25 * luminosidad, 2)
    delta = max(-1.0, min(1.0, (weighted_score - 5.5) / 4.5))
    adjustment_pct = round(delta * max_adjustment_pct, 2)
    return weighted_score, adjustment_pct


def compute_valuation_breakdown(
    base_price: float,
    uncertainty_band_pct: float,
    prop: PropertyInput,
    vision: VisionAssessment,
    max_adjustment_pct: float,
    district_median_m2: int | None = None,
    market_update_factor: float = 1.0,
    factores_precio_base: list[TabularFactor] | None = None,
) -> ValuationBreakdown:
    """Combina el precio base ML con la evaluación visual para producir el rango final."""
    precio_base_int = max(10000, round_to_nearest_500(base_price))
    weighted_score, ajuste_pct = calculate_visual_adjustment(
        reforma=vision.reforma,
        luminosidad=vision.luminosidad,
        calidades=vision.calidades,
        max_adjustment_pct=max_adjustment_pct,
    )

    raw_adjusted = base_price * (1.0 + ajuste_pct / 100.0)
    precio_ajustado = max(10000, round_to_nearest_500(raw_adjusted))
    ajuste_eur = precio_ajustado - precio_base_int

    rango_min = min(precio_ajustado, max(10000, round_to_nearest_500(raw_adjusted * (1.0 - uncertainty_band_pct))))
    rango_max = max(precio_ajustado, round_to_nearest_500(raw_adjusted * (1.0 + uncertainty_band_pct)))

    area = prop.metros_cuadrados
    precio_m2_base = round(precio_base_int / area)
    precio_m2_estimado = round(precio_ajustado / area)
    precio_m2_rango_min = round(rango_min / area)
    precio_m2_rango_max = round(rango_max / area)

    comparativa_barrio_pct: float | None = None
    if district_median_m2 and district_median_m2 > 0:
        comparativa_barrio_pct = round(
            ((precio_m2_estimado - district_median_m2) / district_median_m2) * 100.0,
            1,
        )

    return ValuationBreakdown(
        precio_base=precio_base_int,
        precio_m2_base=precio_m2_base,
        score_visual_ponderado=weighted_score,
        ajuste_visual_pct=ajuste_pct,
        ajuste_visual_eur=ajuste_eur,
        precio_ajustado=precio_ajustado,
        rango_min=rango_min,
        rango_max=rango_max,
        precio_m2_estimado=precio_m2_estimado,
        precio_m2_rango_min=precio_m2_rango_min,
        precio_m2_rango_max=precio_m2_rango_max,
        banda_incertidumbre_pct=round(uncertainty_band_pct * 100.0, 1),
        mediana_barrio_m2=district_median_m2,
        comparativa_barrio_pct=comparativa_barrio_pct,
        factor_actualizacion_mercado=round(market_update_factor, 4),
        factores_precio_base=factores_precio_base or [],
    )


class AppraisalService:
    """Coordina validación de imágenes, predicción tabular y evaluación multimodal."""

    def __init__(
        self,
        settings: Settings,
        predictor: TabularPricePredictor,
        vision_client: OpenRouterVisionClient,
    ) -> None:
        self.settings = settings
        self.predictor = predictor
        self.vision_client = vision_client

    async def appraise(
        self,
        prop: PropertyInput,
        uploaded_files: list[tuple[str, bytes]],
        byok_api_key: str | None = None,
        byok_provider: str = "auto",
        byok_model: str | None = None,
    ) -> AppraisalResponse:
        """Ejecuta el pipeline completo de tasación inmobiliaria multimodal."""
        if not (self.settings.min_images <= len(uploaded_files) <= self.settings.max_images):
            raise ValueError(
                f"Debes subir entre {self.settings.min_images} y {self.settings.max_images} fotografías "
                f"(recibidas: {len(uploaded_files)})."
            )

        targets_override = None
        if byok_api_key and byok_api_key.strip():
            targets_override = self.settings.resolve_byok_targets(
                byok_api_key,
                provider=byok_provider,
                model_override=byok_model,
            )

        max_edge = 1024 if len(uploaded_files) <= 4 else 768
        prepared_images = []
        for filename, raw_bytes in uploaded_files:
            if len(raw_bytes) > self.settings.max_image_size_bytes:
                raise ValueError(
                    f"La imagen '{filename}' supera el límite de {self.settings.max_image_size_mb} MB."
                )
            prepared_images.append(
                prepare_image_bytes(raw_bytes, filename=filename, max_edge=max_edge)
            )

        base_price, uncertainty_band_pct = self.predictor.predict_base_price(prop)
        factores = self.predictor.explain_base_price(prop, base_price)
        comparables = self.predictor.find_comparables(prop, limit=4)
        vision_assessment = await self.vision_client.analyze(
            prepared_images,
            prop,
            targets_override=targets_override,
        )
        district_median_m2 = self.predictor.get_district_median_m2(
            prop.barrio,
            micro_barrio=prop.micro_barrio,
        )

        breakdown = compute_valuation_breakdown(
            base_price=base_price,
            uncertainty_band_pct=uncertainty_band_pct,
            prop=prop,
            vision=vision_assessment,
            max_adjustment_pct=self.settings.max_visual_adjustment_pct,
            district_median_m2=district_median_m2,
            market_update_factor=self.predictor.market_update_factor,
            factores_precio_base=factores,
        )

        meta = self.predictor.get_metadata()
        tabular_summary = {
            "algoritmo": meta.get("algorithm", "XGBRegressor"),
            "dataset": meta.get("dataset_source"),
            "r2_test": meta.get("metrics", {}).get("test_r2"),
            "mape_test_pct": meta.get("metrics", {}).get("test_mape_pct"),
            "registros_entrenamiento": meta.get("training_rows"),
            "registros_totales": meta.get("total_rows"),
        }

        return AppraisalResponse(
            propiedad=prop,
            valoracion=breakdown,
            analisis_visual=vision_assessment,
            fotos_analizadas=len(prepared_images),
            modelo_tabular=tabular_summary,
            comparables=comparables,
        )
