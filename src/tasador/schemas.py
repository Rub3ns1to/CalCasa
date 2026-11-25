"""Esquemas Pydantic de entrada y salida de la API de tasación."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PropertyInput(BaseModel):
    """Características tabulares de la vivienda introducidas por el usuario."""

    barrio: str = Field(..., min_length=2, max_length=80, description="Distrito de Madrid")
    micro_barrio: str | None = Field(
        default=None,
        max_length=100,
        description="Barrio específico dentro del distrito (ej. Recoletos, Goya)",
    )
    tipo_vivienda: Literal["Piso", "Ático", "Dúplex", "Casa o chalet"] = Field(
        default="Piso",
        description="Tipología del inmueble",
    )
    metros_cuadrados: float = Field(..., ge=15.0, le=1000.0, description="Superficie útil/construida en m²")
    habitaciones: int = Field(..., ge=0, le=15, description="Número de dormitorios (0 = estudio)")
    banos: int = Field(default=1, ge=1, le=10, description="Número de baños")
    planta: int = Field(default=1, ge=0, le=40, description="Planta (0 = bajo)")
    ascensor: bool = Field(default=True, description="Dispone de ascensor")
    exterior: bool = Field(default=True, description="Vivienda exterior")
    terraza: bool = Field(default=False, description="Dispone de terraza o balcón amplio")
    garaje: bool = Field(default=False, description="Incluye plaza de garaje")
    aire_acondicionado: bool = Field(default=False, description="Dispone de aire acondicionado")
    trastero: bool = Field(default=False, description="Incluye trastero")

    @field_validator("barrio")
    @classmethod
    def clean_barrio(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if len(cleaned) < 2:
            raise ValueError("El nombre del barrio debe tener al menos 2 caracteres.")
        return cleaned

    @field_validator("micro_barrio")
    @classmethod
    def clean_micro_barrio(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.strip().split())
        if not cleaned or cleaned == "__distrito__":
            return None
        return cleaned


class TabularFactor(BaseModel):
    """Contribución local estimada de una variable sobre el precio base ML."""

    factor: str
    impacto_eur: int
    impacto_pct: float


class ComparableProperty(BaseModel):
    """Testigo real anonimizado procedente del dataset de mercado."""

    distrito: str
    micro_barrio: str | None = None
    tipo_vivienda: str
    metros_cuadrados: int
    habitaciones: int
    banos: int
    planta: int
    ascensor: bool
    exterior: bool
    precio_estimado: int
    precio_m2: int
    similitud_pct: int


class VisionAssessment(BaseModel):
    """Resultado estructurado del análisis visual sobre las fotografías."""

    reforma: int = Field(..., ge=1, le=10, description="Estado de conservación y reforma (1-10)")
    luminosidad: int = Field(..., ge=1, le=10, description="Luminosidad natural percibida (1-10)")
    calidades: int = Field(..., ge=1, le=10, description="Calidad de acabados y materiales (1-10)")
    estado_general: str = Field(..., description="Etiqueta resumida del estado del inmueble")
    puntos_fuertes: list[str] = Field(default_factory=list)
    puntos_mejora: list[str] = Field(default_factory=list)
    justificacion: str = Field(..., description="Explicación breve del ajuste aplicado")
    mode: Literal["openrouter", "gemini", "openai", "anthropic", "demo"] = Field(
        ...,
        description="Motor utilizado para analizar las fotos",
    )
    model_used: str = Field(..., description="Identificador del modelo o heurística aplicada")
    warning: str | None = Field(default=None, description="Aviso si hubo degradación a modo demo")


class ValuationBreakdown(BaseModel):
    """Desglose transparente desde el precio base ML hasta el rango final."""

    precio_base: int
    precio_m2_base: int
    score_visual_ponderado: float
    ajuste_visual_pct: float
    ajuste_visual_eur: int
    precio_ajustado: int
    rango_min: int
    rango_max: int
    precio_m2_estimado: int
    precio_m2_rango_min: int
    precio_m2_rango_max: int
    banda_incertidumbre_pct: float
    mediana_barrio_m2: int | None = None
    comparativa_barrio_pct: float | None = None
    factor_actualizacion_mercado: float = 1.0
    factores_precio_base: list[TabularFactor] = Field(default_factory=list)


class AppraisalResponse(BaseModel):
    """Respuesta completa del endpoint de tasación multimodal."""

    propiedad: PropertyInput
    valoracion: ValuationBreakdown
    analisis_visual: VisionAssessment
    fotos_analizadas: int
    modelo_tabular: dict[str, Any]
    comparables: list[ComparableProperty] = Field(default_factory=list)
