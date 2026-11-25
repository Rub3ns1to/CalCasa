"""Pruebas de coherencia económica y explicabilidad del modelo tabular entrenado."""

from pathlib import Path

from tasador.config import Settings, _resolve_base_dir
from tasador.ml.predictor import TabularPricePredictor
from tasador.schemas import PropertyInput


def _load_predictor() -> TabularPricePredictor:
    settings = Settings(_env_file=None)
    predictor = TabularPricePredictor(
        model_path=settings.model_path,
        metadata_path=settings.metadata_path,
    )
    predictor.load()
    return predictor


def test_trained_model_exists_and_loads() -> None:
    predictor = _load_predictor()
    assert predictor.is_loaded
    meta = predictor.get_metadata()
    assert meta["metrics"]["test_r2"] >= 0.85


def test_district_hierarchy_salamanca_above_vallecas() -> None:
    predictor = _load_predictor()
    salamanca = PropertyInput(
        barrio="Salamanca",
        metros_cuadrados=90.0,
        habitaciones=3,
        banos=2,
        planta=3,
        ascensor=True,
    )
    vallecas = salamanca.model_copy(update={"barrio": "Puente de Vallecas"})

    price_salamanca, _ = predictor.predict_base_price(salamanca)
    price_vallecas, _ = predictor.predict_base_price(vallecas)

    assert price_salamanca > price_vallecas * 2.0


def test_micro_barrio_hierarchy_recoletos_above_guindalera() -> None:
    predictor = _load_predictor()
    recoletos = PropertyInput(
        barrio="Salamanca",
        micro_barrio="Recoletos",
        tipo_vivienda="Piso",
        metros_cuadrados=106.0,
        habitaciones=2,
        banos=2,
        planta=3,
        ascensor=True,
        exterior=False,
    )
    guindalera = recoletos.model_copy(update={"micro_barrio": "Guindalera"})
    vallecas = recoletos.model_copy(update={"barrio": "Puente de Vallecas", "micro_barrio": None})

    p_recoletos, _ = predictor.predict_base_price(recoletos)
    p_guindalera, _ = predictor.predict_base_price(guindalera)
    p_vallecas, _ = predictor.predict_base_price(vallecas)

    assert p_recoletos > p_guindalera * 1.3 > p_vallecas


def test_property_type_atico_above_piso() -> None:
    predictor = _load_predictor()
    piso = PropertyInput(
        barrio="Salamanca",
        micro_barrio="Recoletos",
        tipo_vivienda="Piso",
        metros_cuadrados=106.0,
        habitaciones=2,
        banos=2,
        planta=3,
        ascensor=True,
        exterior=False,
    )
    atico = piso.model_copy(update={"tipo_vivienda": "Ático"})

    p_piso, _ = predictor.predict_base_price(piso)
    p_atico, _ = predictor.predict_base_price(atico)

    assert p_atico > p_piso


def test_explain_base_price_returns_factors() -> None:
    predictor = _load_predictor()
    prop = PropertyInput(
        barrio="Salamanca",
        micro_barrio="Recoletos",
        tipo_vivienda="Piso",
        metros_cuadrados=106.0,
        habitaciones=2,
        banos=2,
        planta=3,
        ascensor=True,
        exterior=False,
    )
    base_price, _ = predictor.predict_base_price(prop)
    factors = predictor.explain_base_price(prop, base_price)

    assert len(factors) >= 1
    assert any("Recoletos" in f.factor and f.impacto_eur > 0 for f in factors)


def test_area_monotonicity() -> None:
    predictor = _load_predictor()
    small = PropertyInput(
        barrio="Arganzuela",
        metros_cuadrados=55.0,
        habitaciones=1,
        banos=1,
        planta=2,
        ascensor=True,
    )
    large = PropertyInput(
        barrio="Arganzuela",
        metros_cuadrados=120.0,
        habitaciones=3,
        banos=2,
        planta=2,
        ascensor=True,
    )

    p_small, _ = predictor.predict_base_price(small)
    p_large, _ = predictor.predict_base_price(large)

    assert p_large > p_small * 1.6


def test_elevator_impact_on_high_floor() -> None:
    predictor = _load_predictor()
    with_lift = PropertyInput(
        barrio="Chamberí",
        metros_cuadrados=85.0,
        habitaciones=2,
        banos=1,
        planta=5,
        ascensor=True,
    )
    without_lift = with_lift.model_copy(update={"ascensor": False})

    p_lift, _ = predictor.predict_base_price(with_lift)
    p_no_lift, _ = predictor.predict_base_price(without_lift)

    assert p_lift > p_no_lift


def test_district_normalization_handles_missing_accent() -> None:
    predictor = _load_predictor()
    p1, _ = predictor.predict_base_price(
        PropertyInput(barrio="Chamberí", metros_cuadrados=80.0, habitaciones=2, planta=2, ascensor=True)
    )
    p2, _ = predictor.predict_base_price(
        PropertyInput(barrio="chamberi", metros_cuadrados=80.0, habitaciones=2, planta=2, ascensor=True)
    )
    assert abs(p1 - p2) < 1e-6


def test_find_comparables_returns_similar_recoletos_witnesses() -> None:
    predictor = _load_predictor()
    prop = PropertyInput(
        barrio="Salamanca",
        micro_barrio="Recoletos",
        tipo_vivienda="Piso",
        metros_cuadrados=106.0,
        habitaciones=2,
        banos=2,
        planta=3,
        ascensor=True,
        exterior=False,
    )
    comparables = predictor.find_comparables(prop, limit=4)
    assert len(comparables) == 4
    assert all(c.micro_barrio == "Recoletos" for c in comparables)
    assert all(c.precio_m2 > 0 and 55 <= c.similitud_pct <= 99 for c in comparables)


def test_resolve_base_dir_falls_back_to_cwd_for_wheel_install(tmp_path: Path) -> None:
    fake_site_pkg = tmp_path / "venv" / "lib" / "python3.11" / "site-packages" / "tasador" / "config.py"
    fake_site_pkg.parent.mkdir(parents=True)
    fake_site_pkg.write_text("", encoding="utf-8")

    app_dir = tmp_path / "app"
    (app_dir / "models").mkdir(parents=True)

    resolved = _resolve_base_dir(module_file=fake_site_pkg, cwd=app_dir)
    assert resolved == app_dir.resolve()
