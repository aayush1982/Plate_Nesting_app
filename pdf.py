# pdf.py
#
# Final management PDF generator.
#
# Part 1: Summary (A4 portrait)
#   - Header / timestamp (+ Tiers/Fabs covered if provided via bhct_kpis)
#   - BH/CT KPIs
#   - Connection MR KPIs
#   - Nesting Methodology & Assumptions  <-- NEW
#   - Final Consolidated Order table (ALL rows)
#   - Exceptions / Notes
#
# Part 2: Plate Marking Layouts (A3 landscape)
#   - Up to 3 plates per page
#   - Each block has header with Util %, Conn Use kg, Final Util %
#
# Part 3: Splice Joint Views (A3 landscape)
#   - 5 splice views per page
#
# Then all parts are merged into one PDF using PyPDF2.
#
# Requirements:
#   pip install reportlab svglib PyPDF2
#
# Public entry point:
#   build_pdf_report(...)

import io
from datetime import datetime
from typing import Dict, List, Any, Iterable, Tuple

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, A3, landscape
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Flowable,
    PageBreak,
)

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

from PyPDF2 import PdfReader, PdfWriter


# -------------------------------------------------
# Small helper: convert full DataFrame -> Table
# -------------------------------------------------
def _df_as_table_for_pdf(df: pd.DataFrame) -> Table:
    if df is None or len(df) == 0:
        data = [["—"]]
    else:
        nice = df.copy()
        for col in nice.columns:
            if pd.api.types.is_numeric_dtype(nice[col]):
                nice[col] = nice[col].map(
                    lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
                )
        data = [list(nice.columns)] + nice.astype(str).values.tolist()

    tbl = Table(
        data,
        repeatRows=1,
    )
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 7),
        ("FONTSIZE", (0,1), (-1,-1), 7),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
    ]))
    return tbl


# -------------------------------------------------
# Flowable wrapper so we can scale SVG drawings
# -------------------------------------------------
class ScaledDrawing(Flowable):
    def __init__(self, drawing, target_w_mm: float, target_h_mm: float):
        super().__init__()
        self.drawing = drawing
        self.target_w_mm = target_w_mm
        self.target_h_mm = target_h_mm

        self.orig_w_pt = drawing.width
        self.orig_h_pt = drawing.height

        self.box_w_pt = target_w_mm * mm
        self.box_h_pt = target_h_mm * mm

        sx = self.box_w_pt / self.orig_w_pt if self.orig_w_pt > 0 else 1.0
        sy = self.box_h_pt / self.orig_h_pt if self.orig_h_pt > 0 else 1.0
        self.scale = min(sx, sy)

        self.width = self.box_w_pt
        self.height = self.box_h_pt

    def draw(self):
        scaled_w = self.orig_w_pt * self.scale
        scaled_h = self.orig_h_pt * self.scale

        off_x = (self.box_w_pt - scaled_w) / 2.0
        off_y = (self.box_h_pt - scaled_h) / 2.0

        self.canv.saveState()
        self.canv.translate(off_x, off_y)
        self.canv.scale(self.scale, self.scale)
        renderPDF.draw(self.drawing, self.canv, 0, 0)
        self.canv.restoreState()


# -------------------------------------------------
# Utils to render "Tiers covered" / "Fabs covered"
# -------------------------------------------------
def _normalize_id_list(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        parts = [p.strip() for p in val.replace(";", ",").replace("|", ",").split(",")]
        parts = [p for p in parts if p]
    elif isinstance(val, Iterable):
        try:
            parts = [str(x).strip() for x in list(val)]
        except Exception:
            parts = [str(val)]
    else:
        parts = [str(val)]

    def keyer(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    uniq = sorted(set(parts), key=keyer)
    return ", ".join(uniq)


# -------------------------------------------------
# 1. Build SUMMARY PDF (A4 portrait)
# -------------------------------------------------
def _build_summary_pdf(
    project_name: str,
    bhct_kpis: dict,
    conn_summary: dict,
    final_order_df: pd.DataFrame,
    notes_text: str,
    kerf: int,                      # <-- NEW: to print in methodology section
    trim: int,                      # <-- NEW: to print in methodology section
) -> bytes:

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=16,
        spaceAfter=6,
    )

    style_subhead = ParagraphStyle(
        "Subhead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.darkblue,
        spaceBefore=8,
        spaceAfter=4,
    )

    style_body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        spaceAfter=4,
    )

    style_bullets = ParagraphStyle(
        "Bullets",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        leftIndent=12,
        bulletIndent=6,
        spaceAfter=2,
    )

    now_str = datetime.now().strftime("%d-%b-%Y %H:%M")

    tiers_txt = _normalize_id_list(bhct_kpis.get("tiers_covered"))
    fabs_txt  = _normalize_id_list(bhct_kpis.get("fabs_covered"))

    story = []

    # header
    story.append(Paragraph(f"{project_name} — Plate Optimization Summary", style_title))
    story.append(Paragraph(f"Generated: {now_str}", style_body))
    if tiers_txt or fabs_txt:
        lines = []
        if tiers_txt:
            lines.append(f"Tiers covered: <b>{tiers_txt}</b>")
        if fabs_txt:
            lines.append(f"Fabs covered: <b>{fabs_txt}</b>")
        story.append(Paragraph(" | ".join(lines), style_body))
    story.append(Spacer(1, 4 * mm))

    # BH/CT KPIs
    story.append(Paragraph("BH/CT Nesting KPIs", style_subhead))
    story.append(Paragraph(
        f"Total BH/CT Required Weight (kg): "
        f"<b>{bhct_kpis.get('total_bh_weight_kg', 0):,.1f}</b><br/>"
        f"Total Plate Weight Planned (kg): "
        f"<b>{bhct_kpis.get('total_plate_weight_kg', 0):,.1f}</b><br/>"
        f"Material Utilization (%): "
        f"<b>{bhct_kpis.get('utilization_pct_overall', 0):,.1f}%</b>",
        style_body
    ))
    story.append(Spacer(1, 2 * mm))

    # Connection KPIs
    story.append(Paragraph("Connection MR KPIs (after BH/CT leftover reuse)", style_subhead))
    story.append(Paragraph(
        f"Total MR Weight (kg): "
        f"<b>{conn_summary.get('total_mr', 0):,.1f}</b><br/>"
        f"Covered by BH/CT Leftover (kg): "
        f"<b>{conn_summary.get('total_available', 0):,.1f}</b><br/>"
        f"Net Balance to Procure (kg): "
        f"<b>{conn_summary.get('total_net_after', 0):,.1f}</b><br/>"
        f"Total Ordered Plate Weight (kg): "
        f"<b>{conn_summary.get('total_order', 0):,.1f}</b><br/>"
        f"Extra Weight over Net (kg): "
        f"<b>{conn_summary.get('extra_total', 0):,.1f}</b><br/>"
        f"% Extra vs Net Balance: "
        f"<b>{conn_summary.get('pct_extra_total', 0):,.2f}%</b>",
        style_body
    ))
    story.append(Spacer(1, 2 * mm))

        # -------- NEW: Methodology paragraph --------
    story.append(Paragraph("Nesting Methodology & Assumptions", style_subhead))

    meth_lines = [
        f"<b>Kerf / Trim:</b> kerf = <b>{kerf} mm</b>, trim on each edge = <b>{trim} mm</b> (deducted from usable plate size).",
        "• <b>Splice location:</b> A common practice is to place welding joints in the <b>middle third</b> of the member length (beam/column).",
        "• <b>Staggering:</b> Flange and web splices are <b>staggered</b> to avoid alignment of joints.",
        "• <b>Clearance:</b> Maintain at least <b>300 mm</b> between web splice and flange weld locations.",
        "• <b>Splice count:</b> Maximum <b>one</b> lengthwise splice per member (flange or web).",
        "• <b>Transverse (width-wise) splice:</b> Allowed <b>only for webs</b> when required due to width limitation; both sides marked 'WELD JOINT'.",
        "• <b>Plate width preference:</b> Prefer ≤ <b>2500 mm</b>; extend up to <b>3500 mm</b> for wide web or thick plates as per standard stock menu.",
        "• <b>Leftover plate reuse:</b> Plates with < 80 % utilization are reused for connection nesting; up to <b>95 %</b> of leftover is credited.",
        "• <b>Final Utilization:</b> Base nesting utilization + credited reuse (≤ 95 % of leftover).",
    ]
    for line in meth_lines:
        story.append(Paragraph(line, style_bullets))
    story.append(Spacer(1, 3 * mm))
    # -------- /NEW --------

    # -------- /NEW --------

    # Final consolidated order table (FULL)
    story.append(Paragraph("Final Consolidated Order (BH/CT + Connection)", style_subhead))
    story.append(Paragraph(
        "This includes reuse of <80% utilized BH/CT plates against Connection MR first.",
        style_body
    ))
    story.append(_df_as_table_for_pdf(final_order_df))
    story.append(Spacer(1, 4 * mm))

    # Notes / Exceptions
    clean_notes = [ln.strip() for ln in (notes_text or "").splitlines() if ln.strip()]
    if clean_notes:
        story.append(Paragraph("Exceptions / Notes", style_subhead))
        for line in clean_notes:
            story.append(Paragraph(f"• {line}", style_body))

    doc.build(story)
    out = buf.getvalue()
    buf.close()
    return out


# -------------------------------------------------
# 2. Build DRAWINGS PDF (A3 landscape)
# -------------------------------------------------
def _build_drawings_pdf(
    plates: List[Any],
    kerf: int,
    trim: int,
    conn_usage_map: Dict[str, float],
) -> bytes:

    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Flowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from svglib.svglib import svg2rlg
    from drawing import svg_marking_2d
    from data_io import estimate_full_plate_weight_kg

    class ScaledDrawingFlowable(Flowable):
        def __init__(self, drawing, max_w_mm: float, max_h_mm: float):
            super().__init__()
            self.drawing = drawing
            self.max_w = max_w_mm * mm
            self.max_h = max_h_mm * mm
            w, h = drawing.width, drawing.height
            sx = self.max_w / float(w) if w else 1.0
            sy = self.max_h / float(h) if h else 1.0
            self.scale = min(sx, sy)
            self.width = w * self.scale
            self.height = h * self.scale
            self.keepWithNext = True

        def wrap(self, availWidth, availHeight):
            return (min(self.width, availWidth), min(self.height, availHeight))

        def draw(self):
            self.canv.saveState()
            self.canv.scale(self.scale, self.scale)
            self.drawing.drawOn(self.canv, 0, 0)
            self.canv.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A3),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()
    head = ParagraphStyle(
        "PlateHead",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.black,
        spaceAfter=2,
    )

    story = []

    blocks_per_page = 3
    block_count = 0
    box_w_mm = 390.0
    box_h_mm = 78.0

    for idx, sp in enumerate(plates):
        plate_id = str(getattr(sp, "plate_id", ""))
        src      = str(getattr(sp, "source", ""))
        thk      = int(getattr(sp, "thickness_mm", 0))
        Wmm      = int(getattr(sp, "stock_width_mm", 0))
        Lmm      = int(getattr(sp, "stock_length_mm", 0))

        try:
            base_util_pct = float(sp.utilization() * 100.0)
        except Exception:
            base_util_pct = 0.0

        plate_wt_kg = estimate_full_plate_weight_kg(thk, Wmm, Lmm) if (thk and Wmm and Lmm) else 0.0
        leftover_capacity_kg = max(0.0, plate_wt_kg * (1.0 - base_util_pct / 100.0))

        conn_planned_kg = float(conn_usage_map.get(plate_id, 0.0))
        max_usable_from_leftover = leftover_capacity_kg * 0.95
        conn_used_effective_kg = min(conn_planned_kg, max_usable_from_leftover)

        add_pct = (conn_used_effective_kg / plate_wt_kg * 100.0) if plate_wt_kg > 0 else 0.0
        final_util_pct = base_util_pct + add_pct

        heading = (
            f"<b>{plate_id}</b> [{src}]  t{thk}  {Wmm}×{Lmm} mm  "
            f"Util {base_util_pct:.1f}%  |  Conn Use {conn_used_effective_kg:,.1f} kg  "
            f"|  Final Util {final_util_pct:.1f}%"
        )
        story.append(Paragraph(heading, head))

        svg_xml = svg_marking_2d(sp, kerf, trim)
        drawing = svg2rlg(io.BytesIO(svg_xml.encode("utf-8")))
        story.append(ScaledDrawingFlowable(drawing, max_w_mm=box_w_mm, max_h_mm=box_h_mm))
        story.append(Spacer(1, 2 * mm))

        block_count += 1
        if block_count == blocks_per_page and idx != len(plates) - 1:
            story.append(PageBreak())
            block_count = 0

    doc.build(story)
    out = buf.getvalue()
    buf.close()
    return out


# -------------------------------------------------
# 2b. Build SPLICE VIEWS PDF (A3 landscape, 5 per page)
# -------------------------------------------------
def _build_splice_views_pdf(splice_views: List[Tuple[str, str]]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A3),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()
    head = ParagraphStyle(
        "SpliceHead",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.black,
        spaceAfter=1,
    )

    story: List[Any] = []

    blocks_per_page = 5
    block_count = 0
    box_w_mm = 390.0
    box_h_mm = 48.0
    gap_mm = 1.0

    for idx, (title, svg_html) in enumerate(splice_views):
        story.append(Paragraph(title, head))
        drawing = svg2rlg(io.BytesIO(svg_html.encode("utf-8")))
        story.append(ScaledDrawing(drawing, target_w_mm=box_w_mm, target_h_mm=box_h_mm))
        story.append(Spacer(1, gap_mm * mm))

        block_count += 1
        if block_count == blocks_per_page and idx != len(splice_views) - 1:
            story.append(PageBreak())
            block_count = 0

    doc.build(story)
    out = buf.getvalue()
    buf.close()
    return out


# -------------------------------------------------
# 3. Merge SUMMARY + DRAWINGS + SPLICE VIEWS into one PDF
# -------------------------------------------------
def build_pdf_report(
    project_name: str,
    bhct_kpis: dict,
    conn_summary: dict,
    final_order_df: pd.DataFrame,
    notes_text: str,
    plates: List[Any],
    kerf: int,
    trim: int,
    conn_usage_map: Dict[str, float] = None,
    splice_views: List[Tuple[str, str]] = None,
) -> bytes:

    if conn_usage_map is None:
        conn_usage_map = {}

    # Part 1: Summary PDF bytes
    summary_bytes = _build_summary_pdf(
        project_name=project_name,
        bhct_kpis=bhct_kpis,
        conn_summary=conn_summary,
        final_order_df=final_order_df,
        notes_text=notes_text,
        kerf=kerf,          # pass through for methodology text
        trim=trim,          # pass through for methodology text
    )

    # Part 2: Drawings PDF bytes
    drawings_bytes = _build_drawings_pdf(
        plates=plates,
        kerf=kerf,
        trim=trim,
        conn_usage_map=conn_usage_map,
    )

    # Part 2b: Splice Views PDF bytes (optional)
    splice_bytes = None
    if splice_views:
        splice_bytes = _build_splice_views_pdf(splice_views)

    # Merge them
    writer = PdfWriter()

    reader1 = PdfReader(io.BytesIO(summary_bytes))
    for page in reader1.pages:
        writer.add_page(page)

    reader2 = PdfReader(io.BytesIO(drawings_bytes))
    for page in reader2.pages:
        writer.add_page(page)

    if splice_bytes:
        reader3 = PdfReader(io.BytesIO(splice_bytes))
        for page in reader3.pages:
            writer.add_page(page)

    out_buf = io.BytesIO()
    writer.write(out_buf)
    return out_buf.getvalue()
