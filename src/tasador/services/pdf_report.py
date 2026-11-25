"""Generación de informe ejecutivo de tasación inmobiliaria en PDF."""

from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from tasador.schemas import AppraisalResponse

_FONT_CANDIDATES: list[tuple[Path, Path]] = [
    (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
    (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ),
]


def _fmt_eur(amount: int) -> str:
    return f"{amount:,}".replace(",", ".") + " €"


class _AppraisalPDF(FPDF):
    """Documento PDF con soporte Unicode automático y fallback seguro."""

    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.unicode_font = False
        self.font_family_name = "Helvetica"
        self._configure_fonts()
        self.set_auto_page_break(auto=True, margin=14)

    def _configure_fonts(self) -> None:
        for regular_path, bold_path in _FONT_CANDIDATES:
            if regular_path.exists() and bold_path.exists():
                try:
                    self.add_font("AppFont", "", str(regular_path))
                    self.add_font("AppFont", "B", str(bold_path))
                    self.unicode_font = True
                    self.font_family_name = "AppFont"
                    return
                except Exception:
                    continue

    def txt(self, value: str) -> str:
        """Garantiza compatibilidad de caracteres si no hay fuente TTF Unicode."""
        if self.unicode_font:
            return value
        normalized = (
            value.replace("€", "EUR")
            .replace("–", "-")
            .replace("—", "-")
            .replace("·", "|")
            .replace("±", "+/-")
        )
        return normalized.encode("latin-1", errors="replace").decode("latin-1")

    def section_title(self, title: str) -> None:
        self.ln(3)
        self.set_font(self.font_family_name, "B", 11)
        self.set_text_color(52, 78, 65)
        self.cell(0, 7, self.txt(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        y = self.get_y()
        self.set_draw_color(163, 177, 138)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(2.5)


def generate_appraisal_pdf(appraisal: AppraisalResponse) -> bytes:
    """Construye un PDF multipropósito limpio con el informe completo de tasación."""
    pdf = _AppraisalPDF()
    pdf.set_margins(14, 14, 14)
    pdf.add_page()

    prop = appraisal.propiedad
    val = appraisal.valoracion
    vis = appraisal.analisis_visual
    ff = pdf.font_family_name

    # Cabecera corporativa CalCasa (#344E41)
    pdf.set_fill_color(52, 78, 65)
    pdf.rect(0, 0, pdf.w, 28, style="F")
    pdf.set_xy(14, 7)
    pdf.set_font(ff, "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(
        0,
        7,
        pdf.txt("CalCasa · Informe de Tasación Inmobiliaria Multimodal (Madrid)"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font(ff, "", 9)
    pdf.set_text_color(218, 215, 205)
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(
        0,
        5,
        pdf.txt(f"Autor: Rubén García Núñez  |  Fecha de emisión: {fecha}"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    pdf.set_y(33)

    # Caja Hero de valoración
    pdf.set_fill_color(246, 245, 240)
    pdf.set_draw_color(163, 177, 138)
    box_y = pdf.get_y()
    pdf.rect(14, box_y, pdf.w - 28, 30, style="DF")
    pdf.set_xy(18, box_y + 3)
    pdf.set_font(ff, "B", 8)
    pdf.set_text_color(88, 129, 87)
    pdf.cell(
        0,
        4,
        pdf.txt("RANGO DE VALORACIÓN ESTIMADO"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_x(18)
    pdf.set_font(ff, "B", 16)
    pdf.set_text_color(52, 78, 65)
    pdf.cell(
        0,
        9,
        pdf.txt(f"{_fmt_eur(val.rango_min)} – {_fmt_eur(val.rango_max)}"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_x(18)
    pdf.set_font(ff, "", 9.5)
    pdf.set_text_color(51, 65, 85)
    comp_txt = ""
    if val.comparativa_barrio_pct is not None and val.mediana_barrio_m2:
        sign_c = "+" if val.comparativa_barrio_pct >= 0 else ""
        comp_txt = f"  |  {sign_c}{val.comparativa_barrio_pct}% vs mediana ({val.mediana_barrio_m2:,} €/m²)".replace(",", ".")
    pdf.cell(
        0,
        5.5,
        pdf.txt(
            f"Estimación central: {_fmt_eur(val.precio_ajustado)}  |  "
            f"Precio/m²: {val.precio_m2_estimado:,} €/m²{comp_txt}".replace(",", ".")
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_x(18)
    pdf.set_font(ff, "", 8.5)
    pdf.set_text_color(71, 85, 105)
    sign_v = "+" if val.ajuste_visual_pct > 0 else ""
    pdf.cell(
        0,
        5,
        pdf.txt(
            f"Base ML: {_fmt_eur(val.precio_base)}  ·  Ajuste visual: {sign_v}{val.ajuste_visual_pct}% "
            f"({sign_v}{_fmt_eur(val.ajuste_visual_eur)})  ·  Banda: ±{val.banda_incertidumbre_pct}%  ·  "
            f"Factor mercado: ×{val.factor_actualizacion_mercado}"
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    pdf.set_y(box_y + 33)

    # Datos del inmueble
    pdf.section_title("1. Características de la vivienda")
    pdf.set_font(ff, "", 9.5)
    pdf.set_text_color(30, 41, 59)
    ubicacion = f"{prop.micro_barrio} ({prop.barrio})" if prop.micro_barrio else prop.barrio
    extras = [
        lbl
        for lbl, enabled in (
            ("Ascensor", prop.ascensor),
            ("Exterior", prop.exterior),
            ("Terraza", prop.terraza),
            ("Garaje", prop.garaje),
            ("Aire acondicionado", prop.aire_acondicionado),
            ("Trastero", prop.trastero),
        )
        if enabled
    ]
    extras_str = ", ".join(extras) if extras else "Sin equipamiento adicional destacado"
    pdf.multi_cell(
        0,
        5.2,
        pdf.txt(
            f"Ubicación: {ubicacion}  |  Tipología: {prop.tipo_vivienda}\n"
            f"Superficie: {prop.metros_cuadrados:.0f} m²  |  Dormitorios: {prop.habitaciones}  |  "
            f"Baños: {prop.banos}  |  Planta: {prop.planta}\n"
            f"Equipamiento: {extras_str}"
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    # Factores XGBoost
    if val.factores_precio_base:
        pdf.section_title("2. Factores clave del modelo tabular (XGBoost)")
        pdf.set_font(ff, "", 9)
        for factor in val.factores_precio_base:
            sign_f = "+" if factor.impacto_eur >= 0 else ""
            pdf.cell(
                0,
                5.2,
                pdf.txt(f"• {factor.factor}: {sign_f}{_fmt_eur(factor.impacto_eur)} ({sign_f}{factor.impacto_pct}%)"),
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )

    # Auditoría visual
    pdf.section_title(f"3. Auditoría visual ({appraisal.fotos_analizadas} fotos)")
    pdf.set_font(ff, "B", 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(
        0,
        5.5,
        pdf.txt(
            f"Estado general: {vis.estado_general}  |  Score ponderado: {val.score_visual_ponderado}/10  |  "
            f"Motor: {vis.mode} ({vis.model_used})"
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font(ff, "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(
        0,
        5.2,
        pdf.txt(
            f"Puntuaciones -> Reforma (45%): {vis.reforma}/10  |  "
            f"Calidades (30%): {vis.calidades}/10  |  Luminosidad (25%): {vis.luminosidad}/10"
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(1)
    pdf.multi_cell(0, 5, pdf.txt(f"Justificación: {vis.justificacion}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if vis.puntos_fuertes:
        pdf.ln(1)
        pdf.set_font(ff, "B", 8.8)
        pdf.cell(0, 4.8, pdf.txt("Puntos fuertes:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(ff, "", 8.8)
        for pt in vis.puntos_fuertes:
            pdf.multi_cell(0, 4.6, pdf.txt(f"  + {pt}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if vis.puntos_mejora:
        pdf.ln(1)
        pdf.set_font(ff, "B", 8.8)
        pdf.cell(0, 4.8, pdf.txt("Aspectos mejorables:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(ff, "", 8.8)
        for pm in vis.puntos_mejora:
            pdf.multi_cell(0, 4.6, pdf.txt(f"  - {pm}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Testigos comparables
    if appraisal.comparables:
        pdf.section_title("4. Testigos comparables reales del mercado (anonimizados)")
        col_widths = [44, 24, 15, 18, 15, 29, 22, 15]
        headers = ["Barrio", "Tipo", "m²", "Hab/B", "Planta", "Precio est.", "€/m²", "Sim."]

        pdf.set_fill_color(241, 245, 249)
        pdf.set_font(ff, "B", 8.2)
        pdf.set_text_color(15, 23, 42)
        for width, hdr in zip(col_widths, headers, strict=True):
            pdf.cell(width, 6, pdf.txt(hdr), border=1, fill=True)
        pdf.ln()

        pdf.set_font(ff, "", 8)
        pdf.set_text_color(51, 65, 85)
        for comp in appraisal.comparables:
            barrio_txt = comp.micro_barrio or comp.distrito
            row_vals = [
                barrio_txt[:22],
                comp.tipo_vivienda[:12],
                f"{comp.metros_cuadrados}",
                f"{comp.habitaciones}/{comp.banos}",
                f"{comp.planta}",
                _fmt_eur(comp.precio_estimado),
                f"{comp.precio_m2:,}".replace(",", "."),
                f"{comp.similitud_pct}%",
            ]
            for width, cell_val in zip(col_widths, row_vals, strict=True):
                pdf.cell(width, 5.5, pdf.txt(cell_val), border=1)
            pdf.ln()

    # Pie informativo
    pdf.ln(4)
    pdf.set_font(ff, "", 7.8)
    pdf.set_text_color(100, 116, 139)
    r2_val = appraisal.modelo_tabular.get("r2_test", "-")
    mape_val = appraisal.modelo_tabular.get("mape_test_pct", "-")
    n_rows = appraisal.modelo_tabular.get("registros_totales") or appraisal.modelo_tabular.get("registros_entrenamiento", "-")
    pdf.multi_cell(
        0,
        4.2,
        pdf.txt(
            f"Modelo tabular: XGBoost (R² test {r2_val}, MAPE {mape_val}%, {n_rows} registros). "
            "Nota: informe orientativo generado con fines de portfolio; no constituye una tasación oficial ECO."
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    return bytes(pdf.output())
