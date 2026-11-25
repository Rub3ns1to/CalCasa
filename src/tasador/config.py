"""Configuración centralizada de la aplicación mediante pydantic-settings."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_base_dir(module_file: Path | None = None, cwd: Path | None = None) -> Path:
    """Resuelve la raíz del proyecto tanto en editable (`src/`) como en wheel/Docker (`/app`)."""
    resolved_file = (module_file or Path(__file__)).resolve()
    source_root = (
        resolved_file.parents[2] if len(resolved_file.parents) > 2 else resolved_file.parent
    )
    if (source_root / "models").exists() or (source_root / "static").exists():
        return source_root
    current_dir = (cwd or Path.cwd()).resolve()
    if (current_dir / "models").exists() or (current_dir / "static").exists():
        return current_dir
    return source_root


BASE_DIR = _resolve_base_dir()
BYOK_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,79}$")

VisionProviderMode = Literal["openrouter", "gemini", "openai", "anthropic", "demo"]
VisionProtocol = Literal["openai_chat", "gemini_native", "anthropic_messages", "demo"]


@dataclass(frozen=True)
class VisionTarget:
    """Destino resuelto de proveedor, protocolo, credenciales y modelo visual."""

    provider: VisionProviderMode
    api_key: str
    base_url: str
    model: str
    protocol: VisionProtocol


def _is_valid_api_key(value: str) -> bool:
    """Devuelve True si ``value`` parece una clave real y no un placeholder."""
    key = value.strip()
    if not key:
        return False
    lowered = key.lower()
    placeholders = {"your_api_key", "your_key_here", "changeme", "sk-or-v1-", "sk-or-"}
    if lowered in placeholders or "..." in key or lowered.endswith("xxx") or "${" in key:
        return False
    return True


class Settings(BaseSettings):
    """Configuración cargada desde variables de entorno y archivo `.env`."""

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_app_url: str = Field(default="http://localhost:8000")
    openrouter_app_name: str = Field(default="CalCasa - Tasador Inmobiliario Multimodal")
    openrouter_timeout_seconds: float = Field(default=45.0, ge=5.0, le=180.0)

    openai_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="openai/gpt-5.4-mini")

    anthropic_base_url: str = Field(default="https://openrouter.ai/api")
    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="anthropic/claude-haiku-4.5")

    gemini_model: str = Field(default="google/gemini-3.8-flash")
    gemini_api_key: str = Field(default="")
    byok_gemini_model: str = Field(default="gemini-3.5-flash-lite")
    byok_openrouter_model: str = Field(default="google/gemini-3.5-flash-lite")

    vision_provider: Literal["auto", "gemini", "openrouter", "openai", "anthropic"] = Field(
        default="auto"
    )

    max_visual_adjustment_pct: float = Field(default=15.0, ge=0.0, le=35.0)
    min_images: int = Field(default=1, ge=1, le=4)
    max_images: int = Field(default=10, ge=1, le=20)
    max_image_size_mb: int = Field(default=8, ge=1, le=25)
    allow_fallback_on_error: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    model_path: Path = Field(default=BASE_DIR / "models" / "price_model.joblib")
    metadata_path: Path = Field(default=BASE_DIR / "models" / "model_metadata.json")
    comparables_path: Path = Field(default=BASE_DIR / "models" / "comparables_index.json.gz")
    static_dir: Path = Field(default=BASE_DIR / "static")

    @model_validator(mode="after")
    def _resolve_env_references(self) -> "Settings":
        """Expande referencias ``${OPENROUTER_API_KEY}`` habituales en `.env`."""
        or_key = self.openrouter_api_key.strip()
        for field_name in ("openai_api_key", "anthropic_api_key"):
            val = getattr(self, field_name).strip()
            if val == "${OPENROUTER_API_KEY}":
                setattr(self, field_name, or_key)
            elif "${OPENROUTER_API_KEY}" in val:
                setattr(self, field_name, val.replace("${OPENROUTER_API_KEY}", or_key))
        return self

    def _demo_target(self) -> VisionTarget:
        return VisionTarget(
            provider="demo",
            api_key="",
            base_url="",
            model="heuristica-local-pillow",
            protocol="demo",
        )

    def _resolve_gemini_native_only(self) -> VisionTarget | None:
        if not _is_valid_api_key(self.gemini_api_key):
            return None
        model = self.gemini_model.strip().removeprefix("google/") or "gemini-2.0-flash"
        return VisionTarget(
            provider="gemini",
            api_key=self.gemini_api_key.strip(),
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model=model,
            protocol="gemini_native",
        )

    def _resolve_openrouter_for_model(self, model: str) -> VisionTarget | None:
        base_url = self.openrouter_base_url.strip() or "https://openrouter.ai/api/v1"
        if _is_valid_api_key(self.openrouter_api_key):
            key = self.openrouter_api_key.strip()
        elif _is_valid_api_key(self.openai_api_key) and "openrouter.ai" in self.openai_base_url:
            key = self.openai_api_key.strip()
        else:
            return None
        clean_model = model.strip() or "google/gemini-3.8-flash"
        return VisionTarget(
            provider="openrouter",
            api_key=key,
            base_url=base_url,
            model=clean_model,
            protocol="openai_chat",
        )

    def _resolve_gemini_target(self) -> VisionTarget | None:
        native = self._resolve_gemini_native_only()
        if native is not None:
            return native
        return self._resolve_openrouter_for_model(
            self.gemini_model.strip() or self.openrouter_model.strip() or "google/gemini-3.8-flash"
        )

    def _resolve_openrouter_target(self) -> VisionTarget | None:
        model = (
            self.openrouter_model.strip()
            or self.gemini_model.strip()
            or self.openai_model.strip()
            or "google/gemini-3.8-flash"
        )
        return self._resolve_openrouter_for_model(model)

    def _resolve_openai_target(self) -> VisionTarget | None:
        base_url = self.openai_base_url.strip() or "https://api.openai.com/v1"
        if _is_valid_api_key(self.openai_api_key):
            key = self.openai_api_key.strip()
        elif _is_valid_api_key(self.openrouter_api_key) and "openrouter.ai" in base_url:
            key = self.openrouter_api_key.strip()
        else:
            return None

        model = self.openai_model.strip() or "openai/gpt-5.4-mini"
        if "api.openai.com" in base_url:
            model = model.removeprefix("openai/")
        provider: VisionProviderMode = "openrouter" if "openrouter.ai" in base_url else "openai"
        return VisionTarget(
            provider=provider,
            api_key=key,
            base_url=base_url,
            model=model,
            protocol="openai_chat",
        )

    def _resolve_anthropic_target(self) -> VisionTarget | None:
        base_url = self.anthropic_base_url.strip() or "https://api.anthropic.com"
        if _is_valid_api_key(self.anthropic_api_key):
            key = self.anthropic_api_key.strip()
        elif _is_valid_api_key(self.openrouter_api_key) and "openrouter.ai" in base_url:
            key = self.openrouter_api_key.strip()
        else:
            return None

        model = self.anthropic_model.strip() or "anthropic/claude-haiku-4.5"
        if "openrouter.ai" in base_url:
            return VisionTarget(
                provider="openrouter",
                api_key=key,
                base_url=self.openrouter_base_url.strip() or "https://openrouter.ai/api/v1",
                model=model,
                protocol="openai_chat",
            )
        return VisionTarget(
            provider="anthropic",
            api_key=key,
            base_url=base_url,
            model=model.removeprefix("anthropic/"),
            protocol="anthropic_messages",
        )

    @staticmethod
    def _dedup_targets(candidates: list[VisionTarget | None]) -> list[VisionTarget]:
        unique: list[VisionTarget] = []
        seen: set[tuple[str, str, str, str]] = set()
        for target in candidates:
            if target is None:
                continue
            sig = (target.provider, target.protocol, target.base_url, target.model)
            if sig not in seen:
                seen.add(sig)
                unique.append(target)
        return unique

    def resolve_vision_targets(self) -> list[VisionTarget]:
        """Devuelve la cadena ordenada de proveedores/modelos para failover automático."""
        mode = self.vision_provider
        candidates: list[VisionTarget | None] = []

        if mode == "gemini":
            candidates.append(self._resolve_gemini_native_only())
            candidates.append(
                self._resolve_openrouter_for_model(
                    self.gemini_model.strip()
                    or self.openrouter_model.strip()
                    or "google/gemini-3.8-flash"
                )
            )
        elif mode == "openrouter":
            candidates.append(self._resolve_openrouter_target())
            if self.openai_model.strip():
                candidates.append(self._resolve_openrouter_for_model(self.openai_model.strip()))
        elif mode == "openai":
            candidates.append(self._resolve_openai_target())
        elif mode == "anthropic":
            candidates.append(self._resolve_anthropic_target())
        else:
            # Modo automático: Gemini nativo -> OpenRouter primario -> OpenRouter secundario (GPT)
            candidates.append(self._resolve_gemini_native_only())
            candidates.append(self._resolve_openrouter_target())
            if _is_valid_api_key(self.openrouter_api_key) and self.openai_model.strip():
                candidates.append(self._resolve_openrouter_for_model(self.openai_model.strip()))
            if not self.has_openrouter_key:
                candidates.append(self._resolve_openai_target())
                candidates.append(self._resolve_anthropic_target())

        resolved = self._dedup_targets(candidates)
        return resolved or [self._demo_target()]

    def resolve_vision_target(self) -> VisionTarget:
        """Resuelve el proveedor, protocolo y modelo visual principal según `.env`."""
        return self.resolve_vision_targets()[0]

    def resolve_byok_targets(
        self,
        api_key: str,
        provider: str = "auto",
        model_override: str | None = None,
    ) -> list[VisionTarget]:
        """Resuelve un destino visual puntual a partir de una API key aportada por el usuario."""
        key = api_key.strip()
        if not key:
            return []
        if not _is_valid_api_key(key):
            raise ValueError("La API key introducida no parece válida.")

        clean_model = (model_override or "").strip()
        if clean_model and (".." in clean_model or not BYOK_MODEL_RE.match(clean_model)):
            raise ValueError("El nombre del modelo especificado no es válido.")

        prov = (provider or "auto").strip().lower()
        valid_providers = {"auto", "gemini", "openrouter", "openai", "anthropic"}
        if prov not in valid_providers:
            raise ValueError(
                f"Proveedor visual no soportado: '{prov}'. Usa auto, gemini, openrouter, openai o anthropic."
            )

        if prov == "auto":
            if key.startswith("AIza"):
                prov = "gemini"
            elif key.startswith("sk-or-"):
                prov = "openrouter"
            elif key.startswith("sk-ant-"):
                prov = "anthropic"
            elif key.startswith("sk-"):
                prov = "openai"
            else:
                raise ValueError(
                    "No se pudo detectar el proveedor automáticamente por el formato de la clave. "
                    "Selecciona manualmente Gemini, OpenRouter, OpenAI o Anthropic."
                )

        if prov == "gemini":
            model = (
                clean_model.removeprefix("google/")
                if clean_model
                else self.byok_gemini_model.strip().removeprefix("google/") or "gemini-3.5-flash-lite"
            )
            return [
                VisionTarget(
                    provider="gemini",
                    api_key=key,
                    base_url="https://generativelanguage.googleapis.com/v1beta",
                    model=model,
                    protocol="gemini_native",
                )
            ]

        if prov == "openrouter":
            model = (
                clean_model
                or self.byok_openrouter_model.strip()
                or "google/gemini-3.5-flash-lite"
            )
            return [
                VisionTarget(
                    provider="openrouter",
                    api_key=key,
                    base_url=self.openrouter_base_url.strip() or "https://openrouter.ai/api/v1",
                    model=model,
                    protocol="openai_chat",
                )
            ]

        if prov == "openai":
            model = clean_model.removeprefix("openai/") if clean_model else "gpt-4o-mini"
            return [
                VisionTarget(
                    provider="openai",
                    api_key=key,
                    base_url="https://api.openai.com/v1",
                    model=model,
                    protocol="openai_chat",
                )
            ]

        model = clean_model.removeprefix("anthropic/") if clean_model else "claude-haiku-4-5"
        return [
            VisionTarget(
                provider="anthropic",
                api_key=key,
                base_url="https://api.anthropic.com",
                model=model,
                protocol="anthropic_messages",
            )
        ]

    @property
    def has_openrouter_key(self) -> bool:
        """Indica si existe una clave de OpenRouter aparentemente válida."""
        return _is_valid_api_key(self.openrouter_api_key)

    @property
    def has_vision_api(self) -> bool:
        """Indica si existe al menos un proveedor de visión real configurado."""
        return self.resolve_vision_target().protocol != "demo"

    @property
    def max_image_size_bytes(self) -> int:
        """Límite máximo por imagen en bytes."""
        return self.max_image_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Devuelve la instancia cacheada de configuración."""
    return Settings()
