"""Normalización de imágenes con Pillow y analizador heurístico para Modo Demo Local."""

import io

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from tasador.schemas import PropertyInput, VisionAssessment


def prepare_image_bytes(
    raw_bytes: bytes,
    filename: str = "foto.jpg",
    max_edge: int = 1024,
) -> tuple[bytes, Image.Image]:
    """Valida una imagen subida, corrige orientación EXIF, redimensiona y convierte a JPEG.

    Returns:
        Tupla ``(jpeg_bytes, rgb_pil_image)``.

    Raises:
        ValueError: Si los bytes no corresponden a una imagen válida.
    """
    if not raw_bytes:
        raise ValueError(f"El archivo '{filename}' está vacío.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            transposed = ImageOps.exif_transpose(img)
            rgb_img = transposed.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"El archivo '{filename}' no es una imagen válida (formatos soportados: JPG, PNG, WEBP)."
        ) from exc

    if max(rgb_img.size) > max_edge:
        rgb_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    rgb_img.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue(), rgb_img


def _clamp_score(value: float) -> int:
    return max(1, min(10, int(round(value))))


def _estado_label(score_medio: float) -> str:
    if score_medio >= 8.5:
        return "Excelente / Diseño"
    if score_medio >= 7.0:
        return "Reformado"
    if score_medio >= 5.3:
        return "Buen estado"
    if score_medio >= 3.8:
        return "Necesita actualización"
    return "A reformar"


def analyze_images_heuristic(
    images: list[Image.Image],
    prop: PropertyInput,
    warning: str | None = None,
) -> VisionAssessment:
    """Genera una valoración visual determinista local cuando no hay clave de OpenRouter.

    Calcula métricas objetivas de brillo, contraste, rango dinámico y nitidez sobre las
    imágenes reales subidas para ofrecer una demostración reproducible sin llamadas externas.
    """
    brightness_values: list[float] = []
    contrast_values: list[float] = []
    edge_values: list[float] = []
    saturation_values: list[float] = []

    for img in images:
        sample = img.copy()
        sample.thumbnail((256, 256), Image.Resampling.BILINEAR)
        gray = sample.convert("L")
        stat_gray = ImageStat.Stat(gray)
        brightness_values.append(stat_gray.mean[0] / 255.0)
        contrast_values.append(stat_gray.stddev[0] / 128.0)

        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        edge_values.append(edge_stat.mean[0] / 64.0)

        hsv = sample.convert("HSV")
        sat_stat = ImageStat.Stat(hsv)
        saturation_values.append(sat_stat.mean[1] / 255.0)

    avg_brightness = sum(brightness_values) / len(brightness_values)
    avg_contrast = sum(contrast_values) / len(contrast_values)
    avg_edges = sum(edge_values) / len(edge_values)
    avg_saturation = sum(saturation_values) / len(saturation_values)

    # Luminosidad: combina brillo medio y bonus moderado si es vivienda exterior
    lum_raw = 2.0 + avg_brightness * 8.2 + (0.6 if prop.exterior else -0.4)
    luminosidad = _clamp_score(lum_raw)

    # Reforma: combina limpieza tonal, contraste y nitidez de líneas arquitectónicas
    ref_raw = 2.5 + avg_brightness * 4.5 + min(avg_contrast, 0.75) * 3.8 + min(avg_edges, 0.8) * 1.5
    reforma = _clamp_score(ref_raw)

    # Calidades: contraste equilibrado, nitidez y saturación controlada (sin amarilleo excesivo)
    sat_penalty = max(0.0, avg_saturation - 0.45) * 2.5
    cal_raw = 2.8 + min(avg_contrast, 0.75) * 4.2 + min(avg_edges, 0.8) * 3.0 + avg_brightness * 2.2 - sat_penalty
    calidades = _clamp_score(cal_raw)

    score_medio = (reforma + luminosidad + calidades) / 3.0
    estado_general = _estado_label(score_medio)

    puntos_fuertes: list[str] = []
    puntos_mejora: list[str] = []

    if luminosidad >= 7:
        puntos_fuertes.append("Buena luminosidad general y claridad en las estancias analizadas")
    else:
        puntos_mejora.append("Iluminación natural moderada o estancias con menor claridad")

    if reforma >= 7:
        puntos_fuertes.append("Aspecto cuidado y acabados visualmente actuales")
    else:
        puntos_mejora.append("Potencial margen de actualización estética o de acabados")

    if calidades >= 7:
        puntos_fuertes.append("Buena definición de líneas, contraste y superficies limpias")
    else:
        puntos_mejora.append("Acabados estándar sin elementos diferenciales destacados en las fotos")

    if prop.terraza and len(puntos_fuertes) < 3:
        puntos_fuertes.append("Valor añadido por presencia declarada de terraza")

    if warning:
        justificacion = (
            f"Modo Demo Local (respaldo automático tras error temporal del proveedor IA sobre "
            f"{len(images)} foto(s)): se estima un estado '{estado_general.lower()}' con "
            f"reforma {reforma}/10, luminosidad {luminosidad}/10 y calidades {calidades}/10."
        )
    else:
        justificacion = (
            f"Modo Demo Local (análisis heurístico de {len(images)} foto(s) sin clave API "
            f"configurada): se estima un estado '{estado_general.lower()}' con puntuación de "
            f"reforma {reforma}/10, luminosidad {luminosidad}/10 y calidades {calidades}/10. "
            "Configura tus claves en el archivo .env para activar la auditoría semántica multimodal real."
        )

    return VisionAssessment(
        reforma=reforma,
        luminosidad=luminosidad,
        calidades=calidades,
        estado_general=estado_general,
        puntos_fuertes=puntos_fuertes[:3],
        puntos_mejora=puntos_mejora[:3],
        justificacion=justificacion,
        mode="demo",
        model_used="heuristica-local-pillow",
        warning=warning,
    )
