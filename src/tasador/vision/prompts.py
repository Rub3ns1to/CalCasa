"""Plantillas de prompts para el tasador visual multimodal vía OpenRouter."""

from tasador.schemas import PropertyInput

SYSTEM_PROMPT = """Eres un tasador inmobiliario sénior especializado en el mercado residencial de Madrid.
Analiza con objetividad y rigor técnico las fotografías aportadas de la vivienda (de 1 a 10 fotos).
Debes responder ÚNICAMENTE con un objeto JSON válido, sin texto adicional ni bloques Markdown, siguiendo exactamente este esquema:
{
  "reforma": <entero del 1 al 10>,
  "luminosidad": <entero del 1 al 10>,
  "calidades": <entero del 1 al 10>,
  "estado_general": "<A reformar | Necesita actualización | Buen estado | Reformado | Excelente / Diseño>",
  "puntos_fuertes": ["<observación visual concreta 1>", "<observación visual concreta 2>"],
  "puntos_mejora": ["<aspecto visual mejorable 1>", "<aspecto visual mejorable 2>"],
  "justificacion": "<Resumen en español de 2 o 3 frases explicando por qué estas fotos suben, mantienen o bajan la valoración>"
}

Criterios de puntuación (1 = muy deficiente / reforma integral obligatoria; 5-6 = estándar habitable de mercado; 9-10 = reforma reciente de revista o calidades premium):
- reforma: estado de cocina, baños, pintura, suelos, instalaciones visibles y sensación de actualización.
- luminosidad: entrada de luz natural, tamaño de huecos/ventanas, claridad de las estancias.
- calidades: carpintería exterior (climalit/PVC/madera antigua), tarima/parquet/terrazo, puertas, mobiliario fijo y acabados.
No inventes detalles que no aparezcan en las imágenes."""


def build_user_prompt(prop: PropertyInput) -> str:
    """Construye el prompt contextual con los datos declarados del inmueble."""
    ascensor_txt = "con ascensor" if prop.ascensor else "sin ascensor"
    orientacion_txt = "exterior" if prop.exterior else "interior"
    extras = []
    if prop.terraza:
        extras.append("terraza")
    if prop.garaje:
        extras.append("garaje")
    if prop.aire_acondicionado:
        extras.append("aire acondicionado")
    if prop.trastero:
        extras.append("trastero")
    extras_txt = ", ".join(extras) if extras else "sin extras adicionales"
    ubicacion = (
        f"barrio de {prop.micro_barrio} (distrito de {prop.barrio}, Madrid)"
        if prop.micro_barrio
        else f"distrito de {prop.barrio} (Madrid)"
    )

    return (
        f"Evalúa estas fotografías de un inmueble ({prop.tipo_vivienda.lower()}) en el {ubicacion}: "
        f"{prop.metros_cuadrados:.0f} m², {prop.habitaciones} habitaciones, {prop.banos} baños, "
        f"planta {prop.planta}, {ascensor_txt}, {orientacion_txt}, {extras_txt}. "
        "Devuelve estrictamente el JSON solicitado en español."
    )
