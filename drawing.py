# drawing.py
#
# 1. SVG generation for plate layouts and splice views
# 2. ZIP builder for all plate marking SVGs
# 3. PDF helper: convert plate SVGs to ReportLab flowables
#
# Requires:
#   pip install reportlab svglib

import io, zipfile
from typing import List, Tuple, DefaultDict
from collections import defaultdict

from plate_rules import StockPlate, SubPiece
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm


############################
# Plate marking SVG
############################

def svg_marking_2d(plate: StockPlate, kerf: int, trim: int) -> str:
    """
    Build an SVG string for a single plate.
    Shows:
      - full plate outline
      - dashed usable window = after trim
      - each placed rectangle with label
    """
    SCALE = 0.14
    margin = 40
    svg_w = int(plate.stock_length_mm * SCALE) + 2 * margin
    svg_h = int(plate.stock_width_mm  * SCALE) + 2 * margin

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">'
    ]

    # plate background
    parts.append(
        f'<rect x="{margin}" y="{margin}" '
        f'width="{int(plate.stock_length_mm*SCALE)}" '
        f'height="{int(plate.stock_width_mm*SCALE)}" '
        f'fill="#f6f6f6" stroke="#111" stroke-width="2"/>'
    )

    # header text
    util_pct = plate.utilization() * 100.0
    parts.append(
        f'<text x="{margin}" y="{margin-12}" '
        f'font-family="monospace" font-size="16">'
        f'{plate.plate_id} [{plate.source}] '
        f't{plate.thickness_mm} '
        f'{plate.stock_width_mm}×{plate.stock_length_mm} '
        f'Util {util_pct:.1f}%'
        f'</text>'
    )

    # usable inner window (trimmed)
    x0 = margin + int(trim * SCALE)
    y0 = margin + int(trim * SCALE)
    win_w = int((plate.stock_length_mm - 2*trim) * SCALE)
    win_h = int((plate.stock_width_mm  - 2*trim) * SCALE)
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{win_w}" height="{win_h}" '
        f'fill="none" stroke="#888" stroke-dasharray="6,6" stroke-width="1.5"/>'
    )

    # each nested part
    for plc in plate.placements:
        rx = margin + int(plc.x * SCALE)
        ry = margin + int(plc.y * SCALE)
        rw = int(plc.w * SCALE)
        rh = int(plc.h * SCALE)

        parts.append(
            f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" '
            f'fill="#dff1ff" stroke="#0b79d0" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{rx+6}" y="{ry+18}" font-size="12" font-family="monospace">'
            f'{plc.label}</text>'
        )
        parts.append(
            f'<text x="{rx+6}" y="{ry+34}" font-size="12" font-family="monospace">'
            f'{plc.annotate}</text>'
        )

    parts.append('</svg>')
    return "\n".join(parts)


############################
# Splice joint SVG
############################

def svg_splice_view(
    bh_profile: str,
    kind: str,
    total_len: int,
    seg1_len: int,
    seg2_len: int,
    joint_pos: int,
    plate1_id: str,
    plate2_id: str
) -> str:
    """
    Draw a 1D splice bar with weld joint marker.
    """
    SCALE = 0.14
    margin = 30
    title_h = 24
    bar_h_mm = 250

    svg_w = int(total_len * SCALE) + 2 * margin
    svg_h = int(bar_h_mm * SCALE) + title_h + 2 * margin

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">'
    ]

    # title
    parts.append(
        f'<text x="{margin}" y="{margin+title_h-6}" '
        f'font-family="monospace" font-size="16">'
        f'{bh_profile} — {kind.upper()} — Total {total_len} mm'
        f'</text>'
    )

    x0 = margin
    y0 = margin + title_h
    bar_h = int(bar_h_mm * SCALE)

    # first segment
    w1 = int(seg1_len * SCALE)
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{w1}" height="{bar_h}" '
        f'fill="#dff1ff" stroke="#0b79d0"/>'
    )
    parts.append(
        f'<text x="{x0+6}" y="{y0+18}" font-size="12" font-family="monospace">'
        f'Seg1 {seg1_len} mm | Plate {plate1_id}'
        f'</text>'
    )

    # weld joint line
    joint_x = x0 + int(joint_pos * SCALE)
    parts.append(
        f'<line x1="{joint_x}" y1="{y0}" x2="{joint_x}" y2="{y0+bar_h}" '
        f'stroke="#cc0000" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{joint_x+6}" y="{y0+bar_h//2}" font-size="12" font-family="monospace">'
        f'WELD @ {joint_pos} mm'
        f'</text>'
    )

    # second segment
    w2 = int(seg2_len * SCALE)
    parts.append(
        f'<rect x="{x0+w1}" y="{y0}" width="{w2}" height="{bar_h}" '
        f'fill="#e8ffe1" stroke="#2b8a3e"/>'
    )
    parts.append(
        f'<text x="{x0+w1+6}" y="{y0+18}" font-size="12" font-family="monospace">'
        f'Seg2 {seg2_len} mm | Plate {plate2_id}'
        f'</text>'
    )

    parts.append('</svg>')
    return "\n".join(parts)


# ---- replace your existing collect_splice_views with this ----
from typing import List, Tuple, Dict, Any
import html

def collect_splice_views(plates: List[Any], subs_all: List[Any]) -> List[Tuple[str, str]]:
    """
    Build simple SVG mini-views for members that have a splice.
    Supports:
      - Length splices: subpieces index 1/2 with joint_pos_mm set (red line at position)
      - Width splices:  subpieces index 91/92 (transverse split); no joint_pos -> mark as WIDTH SPLICE
    Returns: list of (title, svg_html)
    """
    views: List[Tuple[str, str]] = []

    # Group subpieces by original member (parent_id + kind + profile)
    groups: Dict[Tuple[int, str, str], List[Any]] = {}
    for s in subs_all or []:
        # s has: parent_id, index, total_len_mm, length_mm, width_mm, thickness_mm,
        #        kind ('flange'/'web'), bh_profile, splice_joint_here, joint_pos_mm
        key = (getattr(s, "parent_id", -1),
               (getattr(s, "kind", "") or "").lower(),
               getattr(s, "bh_profile", ""))
        groups.setdefault(key, []).append(s)

    for (parent_id, kind, bhp), chunks in groups.items():
        # Only draw if any subpiece indicates a splice (length or width)
        if not any(getattr(c, "splice_joint_here", False) for c in chunks):
            continue

        # Determine total length along member axis (best available)
        total_len = 0
        # Prefer an explicit total_len_mm if present in any chunk
        for c in chunks:
            if getattr(c, "total_len_mm", 0):
                total_len = max(total_len, int(getattr(c, "total_len_mm", 0)))
        if total_len <= 0:
            # Fallback: sum of length_mm of chunks that look like a length split
            maybe_len = sum(int(getattr(c, "length_mm", 0)) for c in chunks)
            total_len = max(maybe_len, max((int(getattr(c, "length_mm", 0)) for c in chunks), default=0))

        # Basic scaling to a fixed pixel width
        PX_W = 640
        PX_H = 90
        PAD_X = 20
        PAD_Y = 20
        bar_y = PAD_Y + 25
        bar_h = 18

        scale = (PX_W - 2 * PAD_X) / float(max(total_len, 1))

        # Decide splice type(s)
        has_length_splice = any((getattr(c, "index", 0) in (1, 2)) and getattr(c, "joint_pos_mm", None) is not None
                                for c in chunks)
        has_width_splice  = any(getattr(c, "index", 0) in (91, 92) for c in chunks)

        # Build SVG elements
        svg_elems = []

        # Bar (overall member length reference)
        svg_elems.append(
            f'<rect x="{PAD_X}" y="{bar_y}" width="{PX_W-2*PAD_X}" height="{bar_h}" '
            f'rx="3" ry="3" fill="#f0f3f6" stroke="#c7ced6"/>'
        )

        # If length splice: draw vertical red line at joint_pos_mm (use the first valid one)
        joint_pos = None
        for c in chunks:
            jp = getattr(c, "joint_pos_mm", None)
            if jp is not None:
                joint_pos = int(jp)
                break
        if has_length_splice and joint_pos is not None and total_len > 0:
            x_px = PAD_X + joint_pos * scale
            svg_elems.append(
                f'<line x1="{x_px:.1f}" y1="{bar_y-6}" x2="{x_px:.1f}" y2="{bar_y+bar_h+6}" '
                f'stroke="#cc0000" stroke-width="2"/>'
            )
            svg_elems.append(
                f'<text x="{x_px+4:.1f}" y="{bar_y-10}" font-size="11" fill="#cc0000">WELD JOINT</text>'
            )

        # If width splice only (91/92), show a center tag
        if has_width_splice and not has_length_splice:
            # Indicate a transverse (width-wise) joint; position label at mid
            x_mid = PAD_X + (PX_W - 2*PAD_X) / 2
            svg_elems.append(
                f'<text x="{x_mid-60:.1f}" y="{bar_y+bar_h+18}" font-size="11" fill="#cc0000">'
                f'WIDTH SPLICE — WELD JOINT</text>'
            )

        # Labels: left shows BH profile & kind; right shows length
        left_lbl  = f"{bhp}  |  {kind.upper()}  |  Member #{parent_id}"
        right_lbl = f"L = {total_len} mm"

        svg_elems.append(
            f'<text x="{PAD_X}" y="{PAD_Y}" font-size="12" fill="#111">{html.escape(left_lbl)}</text>'
        )
        svg_elems.append(
            f'<text x="{PX_W-PAD_X-120}" y="{PAD_Y}" font-size="12" fill="#555">{html.escape(right_lbl)}</text>'
        )

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PX_W}" height="{PX_H}" viewBox="0 0 {PX_W} {PX_H}">'
            + "".join(svg_elems) +
            '</svg>'
        )

        title = f"{bhp} — {kind.upper()} — Member #{parent_id} (splice view)"
        views.append((title, svg))

    return views



############################
# ZIP of all markings
############################

def build_markings_zip(plates: List[StockPlate], kerf: int, trim: int) -> bytes:
    """
    Bundle all plate SVGs into a ZIP for download.
    """
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for sp in plates:
            name = (
                f"markings/{sp.plate_id}_{sp.source}_"
                f"t{sp.thickness_mm}_{sp.stock_width_mm}_{sp.stock_length_mm}.svg"
            )
            z.writestr(name, svg_marking_2d(sp, kerf, trim))

        z.writestr(
            "markings/README.txt",
            "SVGs with usable-window dashed; labels show L×W×t & BH/CT.\n"
            "Title shows source [inventory|standard]."
        )
    mem.seek(0)
    return mem.read()


############################
# PDF flowables for all plate drawings
############################

def _plate_svg_flowable(plate: StockPlate, kerf: int, trim: int):
    """
    Convert one plate's SVG into a ReportLab flowable (Drawing wrapped as Flowable).
    We'll return a list of [Paragraph(header), Drawing(), Spacer].
    """
    # make SVG string
    svg_str = svg_marking_2d(plate, kerf, trim)

    # convert to ReportLab drawing
    drawing_obj = svg2rlg(io.BytesIO(svg_str.encode("utf-8")))

    # label
    header_style = ParagraphStyle(
        "PlateHeader",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.black,
        spaceAfter=2,
    )

    header_para = Paragraph(
        f"{plate.plate_id} | t{plate.thickness_mm} | "
        f"{plate.stock_width_mm}×{plate.stock_length_mm} | "
        f"Util {plate.utilization()*100:.1f}%",
        header_style,
    )

    # return a small stack to insert into the PDF story
    return [header_para, drawing_obj, Spacer(1, 4 * mm)]


def flowables_for_all_plates(plates: List[StockPlate], kerf: int, trim: int, max_plates: int = None):
    """
    Returns a flat list of Flowables:
    header+svg+spacer for each (optionally truncated after max_plates).
    We'll handle 2-per-page or natural pagination in pdf.py using frames if we want
    BUT simplest is just sequential flow, and ReportLab will paginate naturally.
    """
    flow = []
    count = 0
    for p in plates:
        if max_plates is not None and count >= max_plates:
            break
        flow.extend(_plate_svg_flowable(p, kerf, trim))
        count += 1
    return flow
