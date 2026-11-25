"""Pruebas de parseo JSON multimodal, resolución de proveedores `.env` y fallback."""

import asyncio
import json

import httpx
import pytest

from tasador.config import Settings
from tasador.schemas import PropertyInput
from tasador.vision.heuristic_fallback import prepare_image_bytes
from tasador.vision.openrouter_client import (
    OpenRouterVisionClient,
    VisionClientError,
    _format_error_reason,
    parse_vision_json,
)
from tests.conftest import make_test_image_bytes


def _sample_property() -> PropertyInput:
    return PropertyInput(
        barrio="Chamberí",
        metros_cuadrados=85.0,
        habitaciones=2,
        banos=1,
        planta=4,
        ascensor=True,
    )


def test_parse_plain_and_fenced_json() -> None:
    raw_fenced = """Aquí tienes la valoración:
    ```json
    {
      "reforma": 8,
      "luminosidad": 9,
      "calidades": 7,
      "estado_general": "Reformado",
      "puntos_fuertes": ["Suelo de tarima nuevo", "Ventanales amplios"],
      "puntos_mejora": ["Baño de tamaño compacto"],
      "justificacion": "Vivienda luminosa y actualizada recientemente."
    }
    ```
    """
    parsed = parse_vision_json(raw_fenced)
    assert parsed["reforma"] == 8
    assert parsed["luminosidad"] == 9
    assert parsed["calidades"] == 7
    assert parsed["estado_general"] == "Reformado"
    assert len(parsed["puntos_fuertes"]) == 2


def test_parse_clamps_out_of_range_scores() -> None:
    raw = json.dumps(
        {
            "reforma": 14,
            "luminosidad": -3,
            "calidades": 0,
            "estado_general": "A reformar",
            "justificacion": "Estado heterogéneo.",
        }
    )
    parsed = parse_vision_json(raw)
    assert parsed["reforma"] == 10
    assert parsed["luminosidad"] == 1
    assert parsed["calidades"] == 1


def test_placeholder_sk_or_resolves_to_demo() -> None:
    settings = Settings(_env_file=None, openrouter_api_key="sk-or-...")
    target = settings.resolve_vision_target()
    assert target.protocol == "demo"
    assert target.provider == "demo"


def test_openrouter_key_uses_gemini_model_by_default() -> None:
    settings = Settings(_env_file=None, openrouter_api_key="sk-or-v1-real-key-123")
    target = settings.resolve_vision_target()
    assert target.provider == "openrouter"
    assert target.protocol == "openai_chat"
    assert target.model == "google/gemini-3.8-flash"


def test_openai_env_reference_interpolation() -> None:
    settings = Settings(
        _env_file=None,
        openrouter_api_key="sk-or-v1-shared-999",
        openai_api_key="${OPENROUTER_API_KEY}",
        vision_provider="openai",
    )
    target = settings.resolve_vision_target()
    assert target.api_key == "sk-or-v1-shared-999"
    assert target.provider == "openrouter"
    assert target.model == "openai/gpt-5.4-mini"


def test_openrouter_client_runs_demo_when_no_key() -> None:
    settings = Settings(_env_file=None, openrouter_api_key="")
    client = OpenRouterVisionClient(settings)
    prepared = [prepare_image_bytes(make_test_image_bytes(), "salon.jpg")]

    result = asyncio.run(client.analyze(prepared, _sample_property()))
    assert result.mode == "demo"
    assert 1 <= result.reforma <= 10
    assert 1 <= result.luminosidad <= 10
    assert 1 <= result.calidades <= 10
    assert "Modo Demo Local" in result.justificacion


def test_openrouter_client_mocked_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "reforma": 9,
                            "luminosidad": 8,
                            "calidades": 9,
                            "estado_general": "Excelente / Diseño",
                            "puntos_fuertes": ["Cocina equipada de alta gama"],
                            "puntos_mejora": [],
                            "justificacion": "Reforma integral reciente con acabados de calidad.",
                        }
                    )
                }
            }
        ]
    }

    async def fake_post(self, url, headers=None, json=None):
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=200, json=payload, request=req)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    settings = Settings(_env_file=None, openrouter_api_key="sk-or-v1-valid-test-token-123456")
    client = OpenRouterVisionClient(settings)
    prepared = [prepare_image_bytes(make_test_image_bytes(), "salon.jpg")]

    result = asyncio.run(client.analyze(prepared, _sample_property()))
    assert result.mode == "openrouter"
    assert result.reforma == 9
    assert result.calidades == 9
    assert result.warning is None


def test_gemini_native_strips_prefix_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_urls: list[str] = []
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "reforma": 8,
                                    "luminosidad": 9,
                                    "calidades": 8,
                                    "estado_general": "Reformado",
                                    "puntos_fuertes": ["Mucha luz natural"],
                                    "puntos_mejora": [],
                                    "justificacion": "Estancias luminosas y reforma actual.",
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }

    async def fake_post(self, url, headers=None, json=None):
        captured_urls.append(str(url))
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=200, json=payload, request=req)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    settings = Settings(
        _env_file=None,
        gemini_api_key="AIzaSyFakeTestKey123456",
        gemini_model="google/gemini-3.5-flash-lite",
    )
    target = settings.resolve_vision_target()
    assert target.provider == "gemini"
    assert target.model == "gemini-3.5-flash-lite"

    client = OpenRouterVisionClient(settings)
    prepared = [prepare_image_bytes(make_test_image_bytes(), "salon.jpg")]
    result = asyncio.run(client.analyze(prepared, _sample_property()))

    assert result.mode == "gemini"
    assert result.model_used == "gemini-3.5-flash-lite"
    assert result.luminosidad == 9
    assert captured_urls[0].endswith("/models/gemini-3.5-flash-lite:generateContent")


def test_anthropic_direct_mode_strips_prefix_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_urls: list[str] = []
    payload = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "reforma": 7,
                        "luminosidad": 7,
                        "calidades": 8,
                        "estado_general": "Reformado",
                        "puntos_fuertes": ["Carpintería cuidada"],
                        "puntos_mejora": ["Cocina compacta"],
                        "justificacion": "Buen estado general y acabados sólidos.",
                    }
                ),
            }
        ]
    }

    async def fake_post(self, url, headers=None, json=None):
        captured_urls.append(str(url))
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=200, json=payload, request=req)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    settings = Settings(
        _env_file=None,
        anthropic_base_url="",
        anthropic_api_key="sk-ant-fake-key-123456",
        anthropic_model="anthropic/claude-haiku-4.5",
        vision_provider="anthropic",
    )
    target = settings.resolve_vision_target()
    assert target.provider == "anthropic"
    assert target.model == "claude-haiku-4.5"

    client = OpenRouterVisionClient(settings)
    prepared = [prepare_image_bytes(make_test_image_bytes(), "salon.jpg")]
    result = asyncio.run(client.analyze(prepared, _sample_property()))

    assert result.mode == "anthropic"
    assert result.calidades == 8
    assert captured_urls[0].endswith("/v1/messages")


def test_gemini_503_fails_over_to_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    openrouter_content = json.dumps(
        {
            "reforma": 10,
            "luminosidad": 8,
            "calidades": 9,
            "estado_general": "Excelente / Diseño",
            "puntos_fuertes": ["Reforma de diseño", "Vestidor acristalado"],
            "puntos_mejora": ["Orientación interior"],
            "justificacion": "Acabados de alta gama tras reforma integral reciente.",
        }
    )

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fake_post(self, url, headers=None, json=None):
        url_str = str(url)
        calls.append(url_str)
        req = httpx.Request("POST", url)
        if "generativelanguage.googleapis.com" in url_str:
            return httpx.Response(
                status_code=503,
                json={"error": {"code": 503, "message": "This model is currently experiencing high demand."}},
                request=req,
            )
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": openrouter_content}}]},
            request=req,
        )

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    settings = Settings(
        _env_file=None,
        gemini_api_key="AIzaSyFakeTestKey123456",
        gemini_model="google/gemini-3.5-flash-lite",
        openrouter_api_key="sk-or-v1-valid-test-token-123456",
    )
    client = OpenRouterVisionClient(settings)
    prepared = [prepare_image_bytes(make_test_image_bytes(), "salon.jpg")]
    result = asyncio.run(client.analyze(prepared, _sample_property()))

    assert result.mode == "openrouter"
    assert result.model_used == "google/gemini-3.5-flash-lite"
    assert result.reforma == 10
    assert result.warning is not None
    assert "503" in result.warning
    assert len(calls) == 3  # 2 intentos en Gemini (reintento 503) + 1 en OpenRouter


def test_openrouter_client_fallback_and_strict_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_post(self, url, headers=None, json=None):
        raise httpx.ConnectError("Error de red simulado")

    monkeypatch.setattr(httpx.AsyncClient, "post", failing_post)
    prepared = [prepare_image_bytes(make_test_image_bytes(), "salon.jpg")]

    client_fallback = OpenRouterVisionClient(
        Settings(
            _env_file=None,
            openrouter_api_key="sk-or-v1-valid-test-token-123456",
            allow_fallback_on_error=True,
        )
    )
    res = asyncio.run(client_fallback.analyze(prepared, _sample_property()))
    assert res.mode == "demo"
    assert res.warning is not None
    assert "sin clave" not in res.justificacion

    client_strict = OpenRouterVisionClient(
        Settings(
            _env_file=None,
            openrouter_api_key="sk-or-v1-valid-test-token-123456",
            allow_fallback_on_error=False,
        )
    )
    with pytest.raises(VisionClientError):
        asyncio.run(client_strict.analyze(prepared, _sample_property()))


def test_byok_resolves_providers_and_auto_detection() -> None:
    settings = Settings(_env_file=None)

    gemini_target = settings.resolve_byok_targets("AIzaSyFakeGeminiKey123456")[0]
    assert gemini_target.provider == "gemini"
    assert gemini_target.protocol == "gemini_native"
    assert gemini_target.model == "gemini-3.5-flash-lite"

    or_target = settings.resolve_byok_targets("sk-or-v1-fake-openrouter-123456")[0]
    assert or_target.provider == "openrouter"
    assert or_target.protocol == "openai_chat"
    assert or_target.model == "google/gemini-3.5-flash-lite"

    ant_target = settings.resolve_byok_targets("sk-ant-fake-claude-123456")[0]
    assert ant_target.provider == "anthropic"
    assert ant_target.protocol == "anthropic_messages"

    oai_target = settings.resolve_byok_targets("sk-fake-openai-123456")[0]
    assert oai_target.provider == "openai"
    assert oai_target.protocol == "openai_chat"

    override_target = settings.resolve_byok_targets(
        "AIzaSyFakeGeminiKey123456",
        provider="gemini",
        model_override="google/gemini-2.5-flash",
    )[0]
    assert override_target.model == "gemini-2.5-flash"


def test_byok_rejects_invalid_key_provider_or_model() -> None:
    settings = Settings(_env_file=None)

    assert settings.resolve_byok_targets("   ") == []

    with pytest.raises(ValueError, match="no parece válida"):
        settings.resolve_byok_targets("sk-or-...")

    with pytest.raises(ValueError, match="detectar el proveedor"):
        settings.resolve_byok_targets("token_sin_prefijo_conocido")

    with pytest.raises(ValueError, match="no soportado"):
        settings.resolve_byok_targets("AIzaSyFakeKey123", provider="otro")

    with pytest.raises(ValueError, match="modelo especificado"):
        settings.resolve_byok_targets("AIzaSyFakeKey123", model_override="../secret")


def test_format_error_reason_scrubs_secret() -> None:
    secret = "AIzaSySuperSecretKey999"
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/test")
    resp = httpx.Response(
        status_code=403,
        json={"error": {"message": f"API key not valid: {secret}"}},
        request=req,
    )
    exc = httpx.HTTPStatusError("403 Forbidden", request=req, response=resp)
    formatted = _format_error_reason(exc, secret=secret)
    assert secret not in formatted
    assert "***" in formatted
    assert "403" in formatted
