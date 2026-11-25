"""Pruebas unitarias de la lógica de ajuste visual y construcción de rangos."""

from tasador.schemas import PropertyInput, VisionAssessment
from tasador.services.appraisal_service import (
    calculate_visual_adjustment,
    compute_valuation_breakdown,
)


def test_neutral_scores_produce_zero_adjustment() -> None:
    # 0.45*6 + 0.30*6 + 0.25*4 = 5.50 (punto neutro exacto)
    score, adj_pct = calculate_visual_adjustment(
        reforma=6,
        luminosidad=4,
        calidades=6,
        max_adjustment_pct=15.0,
    )
    assert score == 5.5
    assert adj_pct == 0.0


def test_extreme_scores_hit_max_bounds() -> None:
    high_score, high_adj = calculate_visual_adjustment(10, 10, 10, max_adjustment_pct=15.0)
    low_score, low_adj = calculate_visual_adjustment(1, 1, 1, max_adjustment_pct=15.0)

    assert high_score == 10.0
    assert high_adj == 15.0
    assert low_score == 1.0
    assert low_adj == -15.0


def test_reforma_has_higher_weight_than_luminosidad() -> None:
    score_reforma, adj_reforma = calculate_visual_adjustment(reforma=9, luminosidad=5, calidades=5)
    score_lum, adj_lum = calculate_visual_adjustment(reforma=5, luminosidad=9, calidades=5)

    assert score_reforma > score_lum
    assert adj_reforma > adj_lum


def test_valuation_breakdown_rounding_and_bounds() -> None:
    prop = PropertyInput(
        barrio="Arganzuela",
        metros_cuadrados=100.0,
        habitaciones=3,
        banos=2,
        planta=3,
        ascensor=True,
    )
    vision = VisionAssessment(
        reforma=9,
        luminosidad=8,
        calidades=8,
        estado_general="Reformado",
        puntos_fuertes=["Cocina nueva", "Buena luz natural"],
        puntos_mejora=[],
        justificacion="Piso reformado con buenas calidades.",
        mode="demo",
        model_used="heuristica-local-pillow",
    )

    breakdown = compute_valuation_breakdown(
        base_price=500_000.0,
        uncertainty_band_pct=0.06,
        prop=prop,
        vision=vision,
        max_adjustment_pct=15.0,
        district_median_m2=5150,
    )

    assert breakdown.precio_base % 500 == 0
    assert breakdown.precio_ajustado % 500 == 0
    assert breakdown.rango_min % 500 == 0
    assert breakdown.rango_max % 500 == 0
    assert breakdown.precio_ajustado > breakdown.precio_base
    assert breakdown.rango_min < breakdown.precio_ajustado < breakdown.rango_max
    assert breakdown.precio_m2_estimado == round(breakdown.precio_ajustado / 100.0)
    assert breakdown.comparativa_barrio_pct is not None
