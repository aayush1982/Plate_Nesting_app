# dfx.py — Auto-calibrated Plate Layout DXF
# Landscape view • 10× text • Rotated labels • Centered text
# In-part info now shows ONLY:  BH/CT: <BH_Profile>   (no F/W line)

import io, os, zipfile, tempfile
from datetime import datetime
import pandas as pd
import streamlit as st

try:
    import ezdxf
    from ezdxf.math import Vec2
except Exception:
    st.error("Please install ezdxf: pip install ezdxf")
    st.stop()

APP_TITLE   = "🧱 Plate Layout DXF (Landscape + Rotated & Centered Labels)"
DXF_VERSION = "R2000"

BASE_LAYERS = [
    ("PLATE_OUTLINE", 7), ("CENTER", 3), ("TRIM", 8),
    ("TEXT", 7), ("TITLE", 6), ("ANNOT", 5), ("DIM", 5), ("OOB", 6),
]
KIND_COLOR = {"WEB": 3, "FLANGE": 1, "STIFFENER": 2, "STIFF": 2, "RIB": 6}

# ---------- helpers ----------
def _setup_doc():
    doc = ezdxf.new(DXF_VERSION, setup=True)
    for name, color in BASE_LAYERS:
        if name not in doc.layers:
            doc.layers.add(name, color=color)
    return doc

def _ensure_kind_layer(doc, kind: str):
    k = (kind or "").strip().upper()
    lname = f"CUT_{k}" if k else "CUT"
    if lname not in doc.layers:
        doc.layers.add(lname, color=KIND_COLOR.get(k, 2))
    return lname

def _sanitize_text(s: str) -> str:
    if s is None: return ""
    # Replace unicode symbols that some CAD fonts lack
    s = str(s)
    s = s.replace("×", "x").replace("–", "-").replace("·", ".")
    s = s.replace(" ", " ").replace("\u00A0", " ")  # thin/nb spaces
    return s

def _add_text(msp, text, pos, height=30.0, layer="TEXT", rotation_deg=0.0, centered=True):
    """
    Rotation-aware TEXT with optional true center anchoring.
    Uses DXF halign/valign so text is centered at 'pos' regardless of rotation.
    """
    txt = msp.add_text(_sanitize_text(text), dxfattribs={
        "height": float(height),
        "layer": layer,
        "rotation": float(rotation_deg),
    })
    # Set alignment to middle-center (works across ezdxf versions)
    if centered:
        try:
            txt.dxf.halign = 1  # center
            txt.dxf.valign = 2  # middle
            txt.dxf.align_point = (float(pos[0]), float(pos[1]))
        except Exception:
            # Fallback: place at pos; some old viewers may left-align
            pass
    txt.dxf.insert = (float(pos[0]), float(pos[1]))
    return txt

def _draw_plate(msp, pw, pl):
    """
    Draw outer plate boundary and short dashed centerlines for reference.
    """
    # Main plate outline (solid white)
    msp.add_lwpolyline(
        [(0, 0), (pw, 0), (pw, pl), (0, pl), (0, 0)],
        dxfattribs={"layer": "PLATE_OUTLINE", "closed": True},
    )

    # Short dashed centerlines (20% of full width/height)
    cx, cy = pw / 2, pl / 2
    x_span = pw * 0.1
    y_span = pl * 0.2

    # Register dashed linetype (if not already present)
    try:
        if "CENTER" not in msp.doc.linetypes:
            msp.doc.linetypes.new("CENTER", dxfattribs={"description": "Dashed line"})
    except Exception:
        pass  # safe fallback for older ezdxf

    # Horizontal short line
    msp.add_line(
        (cx - x_span / 2, cy),
        (cx + x_span / 2, cy),
        dxfattribs={"layer": "CENTER", "linetype": "CENTER", "lineweight": 2},
    )

    # Vertical short line
    msp.add_line(
        (cx, cy - y_span / 2),
        (cx, cy + y_span / 2),
        dxfattribs={"layer": "CENTER", "linetype": "CENTER", "lineweight": 2},
    )


def _fake_dim(msp, p1, p2, offset=(0,300), label=None):
    p1, p2 = Vec2(p1), Vec2(p2)
    d = p2 - p1
    L = (d.x*d.x + d.y*d.y) ** 0.5
    if L < 1e-6: return
    ux, uy = d.x/L, d.y/L
    nx, ny = -uy, ux
    ox, oy = float(offset[0]), float(offset[1])
    dist = ox*nx + oy*ny
    q1, q2 = p1 + Vec2(nx*dist, ny*dist), p2 + Vec2(nx*dist, ny*dist)
    msp.add_line(tuple(p1), tuple(q1), dxfattribs={"layer":"DIM"})
    msp.add_line(tuple(p2), tuple(q2), dxfattribs={"layer":"DIM"})
    msp.add_line(tuple(q1), tuple(q2), dxfattribs={"layer":"DIM"})
    lab = label if label else f"{int(round(L))} mm"
    mid = (q1 + q2) / 2
    _add_text(msp, lab, (mid.x, mid.y + 30), height=40.0, layer="DIM", rotation_deg=0.0, centered=True)

def _save_bytes(doc) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tf:
        path = tf.name
    try:
        doc.saveas(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try: os.remove(path)
        except: pass

# ---------- geometry / scoring ----------
def _rect_overlap(a, b, eps=1e-6):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    return (ix * iy) > eps

def _transform_piece(x, y, w, h, pw, pl, swap_axes, rotate90, y_top):
    if swap_axes:
        x, y = y, x
        pw, pl = pl, pw
    if rotate90:
        widthX, heightY = h, w
    else:
        widthX, heightY = w, h
    if y_top:
        y = pl - y - heightY
    return float(x), float(y), float(widthX), float(heightY), float(pw), float(pl)

def _score_transform(rows, pw, pl, swap_axes, rotate90, y_top):
    rects = []
    inside = 0; oob = 0; spill = 0.0
    for _, r in rows.iterrows():
        tx, ty, tw, th, _, _ = _transform_piece(
            float(r["X (mm)"]), float(r["Y (mm)"]), float(r["W (mm)"]), float(r["H (mm)"]),
            pw, pl, swap_axes, rotate90, y_top
        )
        rects.append((tx, ty, tw, th))
        dx = max(0.0, -tx) + max(0.0, tx + tw - pw)
        dy = max(0.0, -ty) + max(0.0, ty + th - pl)
        if dx == 0.0 and dy == 0.0: inside += 1
        else: oob += 1; spill += dx + dy
    overlaps = 0
    for i in range(len(rects)):
        for j in range(i+1, len(rects)):
            if _rect_overlap(rects[i], rects[j]): overlaps += 1
    score = inside - 5*oob - 2*overlaps - 0.001*spill
    return score, overlaps, oob, spill, rects

def _best_transform(rows, pw, pl):
    best = None
    for swap in (False, True):
        for rot in (False, True):
            for top in (False, True):
                s, ov, oo, sp, rects = _score_transform(rows, pw, pl, swap, rot, top)
                cand = {"flags": (swap, rot, top), "score": s, "ov": ov, "oob": oo, "spill": sp, "rects": rects}
                if (best is None) or (cand["score"] > best["score"]): best = cand
    return best

def _rotate_rects_CW(rects, pw, pl):
    rotated = []
    for (x, y, w, h) in rects:
        x2 = y
        y2 = pw - (x + w)
        rotated.append((x2, y2, h, w))
    return rotated, pl, pw

# ---------- drawing ----------
def _draw_piece(msp, doc, rect, label, annot, kind, profile, out_of_bounds=False):
    x, y, w, h = rect
    layer = "OOB" if out_of_bounds else _ensure_kind_layer(doc, kind or "")

    # Rectangle
    msp.add_lwpolyline([(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)],
                       dxfattribs={"layer": layer, "closed": True})

    # Center
    cx, cy = x + w/2, y + h/2

    # Rotate texts along longest side
    horiz = (w >= h)
    rot_deg = 0.0 if horiz else 90.0

    # Main label (use existing 'Label' if available; sanitize)
    main = str(label).strip() if (pd.notna(label) and str(label).strip()) else f"{int(round(w))}x{int(round(h))}"
    main = _sanitize_text(main)
    _add_text(msp, main, (cx, cy), height=30.0, layer="TEXT", rotation_deg=rot_deg, centered=True)

    # In-part info: only BH/CT (Kind line removed as requested)
    bhct = _sanitize_text(str(profile).strip() if pd.notna(profile) else "")
    if bhct:
        gap = max(min(w, h) * 0.12, 40.0)  # offset away from center
        if horiz:
            info_pos = (cx, cy - gap)   # below center
        else:
            info_pos = (cx - gap, cy)   # left of center
        _add_text(msp, f"BH/CT: {bhct}", info_pos, height=22.0,
                  layer="ANNOT", rotation_deg=rot_deg, centered=True)

def build_plate_dxf_auto_landscape(plate_id, g, show_dims=True, show_trim=True, force_landscape=True):
    pw = float(g["Plate Width (mm)"].iloc[0])
    pl = float(g["Plate Length (mm)"].iloc[0])
    th = float(g["Thickness (mm)"].iloc[0])
    src = str(g["Source"].iloc[0])

    best = _best_transform(g, pw, pl)
    rects = best["rects"]
    swap, rot, top = best["flags"]

    display_pw, display_pl = pw, pl
    if force_landscape and pl > pw:
        rects, display_pw, display_pl = _rotate_rects_CW(rects, pw, pl)

    doc = _setup_doc()
    msp = doc.modelspace()

    _draw_plate(msp, display_pw, display_pl)

    _add_text(msp, f"Plate ID: {plate_id} | {src} | T={int(round(th))} mm",
              (0, -280), height=50.0, layer="TITLE", rotation_deg=0.0, centered=False)
    _add_text(msp,
              f"Plate: {int(round(display_pw))}x{int(round(display_pl))} mm | Parts: {len(g)} | "
              f"Fit: swap={swap}, rot90={rot}, yTop={top} | Score={best['score']:.3f} | Ovl={best['ov']}, OOB={best['oob']}",
              (0, -360), height=35.0, layer="TITLE", rotation_deg=0.0, centered=False)

    if show_dims:
        _fake_dim(msp, (0,0), (display_pw,0), offset=(0,-300), label=f"{int(round(display_pw))} mm")
        _fake_dim(msp, (display_pw,0), (display_pw,display_pl), offset=(300,0), label=f"{int(round(display_pl))} mm")

    if show_trim:
        try:
            tminx = max(0.0, float(g["X (mm)"].min()))
            tminy = max(0.0, float(g["Y (mm)"].min()))
            tx0, ty0 = tminx, tminy
            tw0, th0 = pw - 2*tminx, pl - 2*tminy
            if force_landscape and pl > pw:
                x2 = ty0
                y2 = pw - (tx0 + tw0)
                tw, th = th0, tw0
                msp.add_lwpolyline([(x2,y2),(x2+tw,y2),(x2+tw,y2+th),(x2,y2+th),(x2,y2)],
                                   dxfattribs={"layer":"TRIM","closed":True})
            else:
                msp.add_lwpolyline([(tx0,ty0),(tx0+tw0,ty0),(tx0+tw0,ty0+th0),(tx0,ty0+th0),(tx0,ty0)],
                                   dxfattribs={"layer":"TRIM","closed":True})
        except Exception:
            pass

    for rect, (_, r) in zip(rects, g.iterrows()):
        x, y, w, h = rect
        oob = (x < -1e-6) or (y < -1e-6) or (x + w > display_pw + 1e-6) or (y + h > display_pl + 1e-6)
        _draw_piece(msp, doc, rect,
                    r.get("Label",""), r.get("Annotation",""),
                    r.get("Kind",""), r.get("BH_Profile",""),
                    out_of_bounds=oob)

    return _save_bytes(doc)

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload Shop Marking (Excel/CSV). The app auto-detects axes/origin/rotation, forces landscape view, centers/rotates labels, and prints BH/CT inside each piece.")

up = st.file_uploader("Upload Shop Marking (.xlsx or .csv)", type=["xlsx","csv"])
prefix = st.text_input("DXF filename prefix", value="PLATE")
c1, c2 = st.columns(2)
with c1: show_dims = st.checkbox("Add overall plate dimensions", value=True)
with c2: show_trim = st.checkbox("Show trim rectangle (min X/Y)", value=True)
btn = st.button("Generate DXFs", type="primary", use_container_width=True)

if up is not None:
    try:
        df = pd.read_excel(up) if up.name.lower().endswith(".xlsx") else pd.read_csv(up)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.info(f"Loaded {len(df)} rows from {up.name}")

    required = ["Plate ID","Source","Thickness (mm)","Plate Width (mm)","Plate Length (mm)",
                "X (mm)","Y (mm)","W (mm)","H (mm)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    for col in ["Thickness (mm)","Plate Width (mm)","Plate Length (mm)","X (mm)","Y (mm)","W (mm)","H (mm)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    good = df[(df["Plate Width (mm)"]>0) & (df["Plate Length (mm)"]>0) &
              (df["W (mm)"]>0) & (df["H (mm)"]>0)].copy()

    if btn:
        plates = good["Plate ID"].dropna().astype(str).unique()
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pid, g in good.groupby("Plate ID"):
                try:
                    dxf_bytes = build_plate_dxf_auto_landscape(str(pid), g,
                                                               show_dims=show_dims,
                                                               show_trim=show_trim,
                                                               force_landscape=True)
                    zf.writestr(f"{prefix}_{pid}.dxf", dxf_bytes)
                except Exception as ex:
                    st.warning(f"Skipping Plate {pid}: {type(ex).__name__}: {ex}")

        st.success(f"DXFs ready for {len(plates)} plates.")
        st.download_button("📦 Download DXF ZIP", data=zip_buf.getvalue(),
                           file_name=f"plate_layouts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                           mime="application/zip", use_container_width=True)
