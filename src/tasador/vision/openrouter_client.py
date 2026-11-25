"""Cliente multimodal asíncrono (OpenRouter, Gemini nativo, OpenAI, Anthropic) + modo demo."""

import asyncio
import base64
import json
import logging
import re
from typing import Any

import httpx
from PIL import Image

from tasador.config import Settings, VisionTarget
from tasador.schemas import PropertyInput, VisionAssessment
from tasador.vision.heuristic_fallback import analyze_images_heuristic
from tasador.vision.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

TRANSIENT_STATUS_CODES = {429, 502, 503, 504}


class VisionClientError(Exception):
    """Error al consultar o interpretar la respuesta del modelo multimodal."""


def _clamp_1_10(value: Any, field_name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"El campo '{field_name}' debe ser numérico, recibido: {value!r}") from exc
    return max(1, min(10, int(round(numeric))))


def _clean_str_list(value: Any, max_items: int = 4) -> list[str]:
    if isinstance(value, str):
        items = [value.strip()] if value.strip() else []
    elif isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = []
    return items[:max_items]


def _format_error_reason(exc: Exception, secret: str = "") -> str:
    """Resume el motivo técnico de fallo sin exponer claves ni cabeceras."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        detail = ""
        try:
            body = exc.response.json()
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    detail = str(err.get("message") or err.get("status") or "").strip()
                elif err:
                    detail = str(err).strip()
        except Exception:
            detail = ""
        if detail:
            short = detail[:110] + ("..." if len(detail) > 110 else "")
            reason = f"HTTP {status} ({short})"
        else:
            reason = f"HTTP {status}"
    else:
        reason = type(exc).__name__

    if secret:
        reason = reason.replace(secret, "***")
    return reason


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> httpx.Response:
    """Ejecuta POST y reintenta una vez tras errores transitorios (429/502/503/504)."""
    response = await client.post(url, headers=headers, json=payload)
    if response.status_code in TRANSIENT_STATUS_CODES:
        await asyncio.sleep(1.0)
        response = await client.post(url, headers=headers, json=payload)
    return response


def parse_vision_json(raw_text: str) -> dict[str, Any]:
    """Extrae y valida el JSON estructurado devuelto por un LLM multimodal.

    Soporta JSON puro, bloques Markdown ```json ... ``` o respuestas con texto extra
    rodeando al objeto JSON principal.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("La respuesta del modelo visual está vacía.")

    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("No se encontró ningún objeto JSON válido en la respuesta del modelo.")

    json_slice = cleaned[start_idx : end_idx + 1]
    data = json.loads(json_slice)
    if not isinstance(data, dict):
        raise ValueError("La respuesta JSON debe ser un diccionario de propiedades.")

    reforma = _clamp_1_10(data.get("reforma"), "reforma")
    luminosidad = _clamp_1_10(data.get("luminosidad"), "luminosidad")
    calidades = _clamp_1_10(data.get("calidades"), "calidades")

    mean_score = (reforma + luminosidad + calidades) / 3.0
    default_estado = (
        "Excelente / Diseño"
        if mean_score >= 8.5
        else "Reformado"
        if mean_score >= 7.0
        else "Buen estado"
        if mean_score >= 5.3
        else "Necesita actualización"
        if mean_score >= 3.8
        else "A reformar"
    )

    estado_general = str(data.get("estado_general") or default_estado).strip() or default_estado
    justificacion = str(data.get("justificacion") or "").strip()
    if not justificacion:
        justificacion = (
            f"Evaluación visual automática: reforma {reforma}/10, "
            f"luminosidad {luminosidad}/10 y calidades {calidades}/10."
        )

    return {
        "reforma": reforma,
        "luminosidad": luminosidad,
        "calidades": calidades,
        "estado_general": estado_general,
        "puntos_fuertes": _clean_str_list(data.get("puntos_fuertes")),
        "puntos_mejora": _clean_str_list(data.get("puntos_mejora")),
        "justificacion": justificacion,
    }


def _extract_openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("La respuesta del proveedor no contiene 'choices'.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(parts)
    raise ValueError("Formato inesperado en 'message.content'.")


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError("La respuesta de Gemini no contiene 'candidates'.")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict) and "text" in part]
    combined = "\n".join(texts).strip()
    if not combined:
        raise ValueError("La respuesta de Gemini no contiene texto.")
    return combined


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    blocks = payload.get("content") or []
    texts = [
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    combined = "\n".join(texts).strip()
    if not combined:
        raise ValueError("La respuesta de Anthropic no contiene texto.")
    return combined


class OpenRouterVisionClient:
    """Cliente multimodal con reintento, failover entre proveedores y modo demo local."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def analyze(
        self,
        images: list[tuple[bytes, Image.Image]],
        prop: PropertyInput,
        targets_override: list[VisionTarget] | None = None,
    ) -> VisionAssessment:
        """Evalúa de 1 a 10 fotografías y devuelve puntuaciones y justificación."""
        pil_images = [pil_img for _, pil_img in images]
        targets = (
            targets_override if targets_override is not None else self.settings.resolve_vision_targets()
        )

        if not targets or targets[0].protocol == "demo":
            return analyze_images_heuristic(pil_images, prop)

        failed_attempts: list[str] = []
        last_exc: Exception | None = None

        for target in targets:
            try:
                if target.protocol == "gemini_native":
                    raw_content = await self._call_gemini_native(target, images, prop)
                elif target.protocol == "anthropic_messages":
                    raw_content = await self._call_anthropic_native(target, images, prop)
                else:
                    raw_content = await self._call_openai_compatible(target, images, prop)

                parsed = parse_vision_json(raw_content)
                failover_note = None
                if failed_attempts:
                    failover_note = (
                        f"Respaldo automático activado ({'; '.join(failed_attempts)}): "
                        f"análisis completado con {target.provider} ({target.model})."
                    )
                return VisionAssessment(
                    reforma=parsed["reforma"],
                    luminosidad=parsed["luminosidad"],
                    calidades=parsed["calidades"],
                    estado_general=parsed["estado_general"],
                    puntos_fuertes=parsed["puntos_fuertes"],
                    puntos_mejora=parsed["puntos_mejora"],
                    justificacion=parsed["justificacion"],
                    mode=target.provider,
                    model_used=target.model,
                    warning=failover_note,
                )
            except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
                last_exc = exc
                reason = _format_error_reason(exc, secret=target.api_key)
                failed_attempts.append(f"{target.provider} ({target.model}): {reason}")
                logger.warning(
                    "Fallo al analizar fotos con %s (%s): %s",
                    target.provider,
                    target.model,
                    reason,
                )

        summary = "; ".join(failed_attempts)
        if self.settings.allow_fallback_on_error:
            if targets_override is not None:
                warning_msg = (
                    f"No se pudo completar la llamada con la API key proporcionada ({summary}). "
                    "Se aplicó análisis heurístico local como respaldo."
                )
            else:
                warning_msg = (
                    f"No se pudo completar la llamada con los proveedores configurados ({summary}). "
                    "Se aplicó análisis heurístico local como respaldo."
                )
            return analyze_images_heuristic(pil_images, prop, warning=warning_msg)
        raise VisionClientError(f"Error en proveedor visual ({summary})") from last_exc

    async def _call_openai_compatible(
        self,
        target: VisionTarget,
        images: list[tuple[bytes, Image.Image]],
        prop: PropertyInput,
    ) -> str:
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": build_user_prompt(prop)},
        ]
        for jpeg_bytes, _ in images:
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        headers = {
            "Authorization": f"Bearer {target.api_key}",
            "Content-Type": "application/json",
        }
        if target.provider == "openrouter":
            headers["HTTP-Referer"] = self.settings.openrouter_app_url
            headers["X-Title"] = self.settings.openrouter_app_name

        base_payload: dict[str, Any] = {
            "model": target.model,
            "temperature": 0.2,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        endpoint = f"{target.base_url.rstrip('/')}/chat/completions"

        async with httpx.AsyncClient(timeout=self.settings.openrouter_timeout_seconds) as client:
            payload_with_json = {**base_payload, "response_format": {"type": "json_object"}}
            response = await _post_with_retry(client, endpoint, headers, payload_with_json)
            if response.status_code == 400:
                response = await _post_with_retry(client, endpoint, headers, base_payload)

            response.raise_for_status()
            return _extract_openai_text(response.json())

    async def _call_gemini_native(
        self,
        target: VisionTarget,
        images: list[tuple[bytes, Image.Image]],
        prop: PropertyInput,
    ) -> str:
        parts: list[dict[str, Any]] = [{"text": build_user_prompt(prop)}]
        for jpeg_bytes, _ in images:
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
            },
        }
        headers = {
            "x-goog-api-key": target.api_key,
            "Content-Type": "application/json",
        }
        endpoint = f"{target.base_url.rstrip('/')}/models/{target.model}:generateContent"

        async with httpx.AsyncClient(timeout=self.settings.openrouter_timeout_seconds) as client:
            response = await _post_with_retry(client, endpoint, headers, payload)
            response.raise_for_status()
            return _extract_gemini_text(response.json())

    async def _call_anthropic_native(
        self,
        target: VisionTarget,
        images: list[tuple[bytes, Image.Image]],
        prop: PropertyInput,
    ) -> str:
        content_blocks: list[dict[str, Any]] = [{"type": "text", "text": build_user_prompt(prop)}]
        for jpeg_bytes, _ in images:
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                }
            )

        payload: dict[str, Any] = {
            "model": target.model,
            "system": SYSTEM_PROMPT,
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": content_blocks}],
        }
        headers = {
            "x-api-key": target.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        base = target.base_url.rstrip("/")
        endpoint = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"

        async with httpx.AsyncClient(timeout=self.settings.openrouter_timeout_seconds) as client:
            response = await _post_with_retry(client, endpoint, headers, payload)
            response.raise_for_status()
            return _extract_anthropic_text(response.json())
