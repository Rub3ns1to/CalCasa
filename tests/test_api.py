"""Pruebas de integración de los endpoints FastAPI."""

import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from tests.conftest import make_test_image_bytes


def _default_form_data() -> dict[str, str]:
    return {
        "barrio": "Arganzuela",
        "metros_cuadrados": "92",
        "habitaciones": "3",
        "banos": "2",
        "planta": "3",
        "ascensor": "true",
        "exterior": "true",
        "terraza": "false",
        "garaje": "true",
    }


def test_root_serves_html(demo_client: TestClient) -> None:
    res = demo_client.get("/")
    assert res.status_code == 200
    assert "CalCasa" in res.text
    assert "Tasador Inmobiliario Multimodal" in res.text
    assert "favicon.svg" in res.text
    assert "app.js?v=" in res.text
    assert "no-cache" in res.headers.get("cache-control", "")

    favicon_res = demo_client.get("/static/favicon.svg")
    assert favicon_res.status_code == 200
    assert "<svg" in favicon_res.text

    logo_res = demo_client.get("/static/img/calcasa-logo.svg")
    assert logo_res.status_code == 200
    assert "<svg" in logo_res.text


def test_static_js_served_with_no_cache(demo_client: TestClient) -> None:
    res = demo_client.get("/static/js/app.js")
    assert res.status_code == 200
    assert "no-cache" in res.headers.get("cache-control", "")
    for fn_name in ("downloadPdfReport", "renderComparables", "openModelModal", "updateSimulator"):
        assert fn_name in res.text


def test_health_and_metadata_endpoints(demo_client: TestClient) -> None:
    health = demo_client.get("/api/health")
    assert health.status_code == 200
    health_data = health.json()
    assert health_data["model_loaded"] is True
    assert health_data["vision_mode"] == "demo"
    assert health_data["byok_supported"] is True

    meta = demo_client.get("/api/metadata")
    assert meta.status_code == 200
    meta_data = meta.json()
    assert len(meta_data["districts"]) >= 20
    assert "Ático" in meta_data["property_types"]
    assert meta_data["limits"]["max_images"] == 10


def test_appraise_with_one_photo(demo_client: TestClient) -> None:
    img_bytes = make_test_image_bytes()
    data = {
        **_default_form_data(),
        "barrio": "Salamanca",
        "micro_barrio": "Recoletos",
        "tipo_vivienda": "Piso",
        "aire_acondicionado": "true",
    }
    res = demo_client.post(
        "/api/appraise",
        data=data,
        files=[("fotos", ("salon.jpg", img_bytes, "image/jpeg"))],
    )
    assert res.status_code == 200
    body = res.json()
    assert body["fotos_analizadas"] == 1
    assert body["propiedad"]["micro_barrio"] == "Recoletos"
    val = body["valoracion"]
    assert val["rango_min"] <= val["precio_ajustado"] <= val["rango_max"]
    assert val["precio_m2_estimado"] > 0
    assert len(val["factores_precio_base"]) >= 1
    assert len(body["comparables"]) == 4
    assert body["analisis_visual"]["mode"] == "demo"
    assert len(body["analisis_visual"]["justificacion"]) > 15


def test_pdf_report_endpoint_returns_valid_pdf(demo_client: TestClient) -> None:
    img_bytes = make_test_image_bytes()
    appraise_res = demo_client.post(
        "/api/appraise",
        data={
            **_default_form_data(),
            "barrio": "Salamanca",
            "micro_barrio": "Recoletos",
            "tipo_vivienda": "Piso",
        },
        files=[("fotos", ("salon.jpg", img_bytes, "image/jpeg"))],
    )
    assert appraise_res.status_code == 200

    pdf_res = demo_client.post("/api/report/pdf", json=appraise_res.json())
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"].startswith("application/pdf")
    assert pdf_res.content.startswith(b"%PDF")
    assert len(pdf_res.content) > 1500


def test_appraise_with_ten_photos(demo_client: TestClient) -> None:
    files = [
        ("fotos", (f"foto_{idx}.png", make_test_image_bytes(fmt="PNG"), "image/png"))
        for idx in range(10)
    ]
    res = demo_client.post("/api/appraise", data=_default_form_data(), files=files)
    assert res.status_code == 200
    assert res.json()["fotos_analizadas"] == 10


def test_appraise_rejects_eleven_photos(demo_client: TestClient) -> None:
    files = [
        ("fotos", (f"foto_{idx}.jpg", make_test_image_bytes(), "image/jpeg"))
        for idx in range(11)
    ]
    res = demo_client.post("/api/appraise", data=_default_form_data(), files=files)
    assert res.status_code == 400


def test_appraise_rejects_zero_photos(demo_client: TestClient) -> None:
    res = demo_client.post("/api/appraise", data=_default_form_data())
    assert res.status_code == 422


def test_appraise_rejects_corrupt_image(demo_client: TestClient) -> None:
    res = demo_client.post(
        "/api/appraise",
        data=_default_form_data(),
        files=[("fotos", ("falso.jpg", b"no-es-una-imagen", "image/jpeg"))],
    )
    assert res.status_code == 400


def test_appraise_rejects_invalid_byok_header(demo_client: TestClient) -> None:
    res = demo_client.post(
        "/api/appraise",
        data=_default_form_data(),
        files=[("fotos", ("salon.jpg", make_test_image_bytes(), "image/jpeg"))],
        headers={"X-Vision-Api-Key": "sk-or-..."},
    )
    assert res.status_code == 400


def test_appraise_accepts_byok_headers_mocked(demo_client: TestClient) -> None:
    gemini_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "reforma": 9,
                                    "luminosidad": 8,
                                    "calidades": 9,
                                    "estado_general": "Excelente / Diseño",
                                    "puntos_fuertes": ["Reforma actual"],
                                    "puntos_mejora": [],
                                    "justificacion": "Evaluación ejecutada con API key del usuario.",
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }

    async def fake_post(self, url, headers=None, json=None):
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=200, json=gemini_payload, request=req)

    with patch.object(httpx.AsyncClient, "post", fake_post):
        res = demo_client.post(
            "/api/appraise",
            data=_default_form_data(),
            files=[("fotos", ("salon.jpg", make_test_image_bytes(), "image/jpeg"))],
            headers={"X-Vision-Api-Key": "AIzaSyFakeByokTest123456"},
        )

    assert res.status_code == 200
    body = res.json()
    assert body["analisis_visual"]["mode"] == "gemini"
    assert body["analisis_visual"]["model_used"] == "gemini-3.5-flash-lite"

    # El cliente vuelve a operar en Modo Demo Local después de la petición puntual BYOK
    health = demo_client.get("/api/health")
    assert health.json()["vision_mode"] == "demo"
