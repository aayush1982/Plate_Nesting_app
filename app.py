# app.py
#
# Final Plate Optimizer UI (modular version with PDF tab).
# Uses:
#   data_io.py, optimizer.py, drawing.py, pdf.py
#
# External deps for full PDF with drawings:
#   pip install reportlab svglib PyPDF2

import io
import pandas as pd
import streamlit as st

from data_io import (
    load_mill_offer,
    summarize_weight_by_thickness,
)
from optimizer import run_full_optimization
from drawing import (
    svg_marking_2d,
    collect_splice_views,
    build_markings_zip,
)
from pdf import build_pdf_report  # final combined PDF (summary + drawings)
from drawing import collect_splice_views


# ---------- larger, compact KPI styling ----------
_COMPACT_CSS = """
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 1.0rem; }

/* KPI cards: bigger, denser, nicer */
div[data-testid="stMetric"] {
  background: #f8fafc;
  padding: 14px 16px;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  font-size: 0.95rem;
  color: #4b5563; /* gray-600 */
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-size: 1.6rem;
  line-height: 1.3;
}
div[data-testid="stExpander"] > div > div { padding-top: .25rem; padding-bottom: .25rem; }
hr { margin: 0.6rem 0 0.8rem 0; }
</style>
"""


def _build_shop_marking_df(plates) -> pd.DataFrame:
    """
    Flatten all placements into a 'shop marking' table.
    Columns include plate details + piece geometry + labels.
    """
    rows = []
    for sp in plates:
        for plc in getattr(sp, "placements", []):
            rows.append({
                "Plate ID": getattr(sp, "plate_id", ""),
                "Source": getattr(sp, "source", ""),
                "Thickness (mm)": getattr(sp, "thickness_mm", ""),
                "Plate Width (mm)": getattr(sp, "stock_width_mm", ""),
                "Plate Length (mm)": getattr(sp, "stock_length_mm", ""),
                "X (mm)": getattr(plc, "x", ""),
                "Y (mm)": getattr(plc, "y", ""),
                "W (mm)": getattr(plc, "w", ""),
                "H (mm)": getattr(plc, "h", ""),
                "Label": getattr(plc, "label", ""),
                "Annotation": getattr(plc, "annotate", ""),
                "Kind": getattr(plc, "kind", ""),
                "BH_Profile": getattr(plc, "bh_profile", ""),
                "ParentID": getattr(plc, "parent_id", ""),
                "SubIndex": getattr(plc, "sub_index", ""),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "Plate ID","Source","Thickness (mm)","Plate Width (mm)","Plate Length (mm)",
            "X (mm)","Y (mm)","W (mm)","H (mm)","Label","Annotation",
            "Kind","BH_Profile","ParentID","SubIndex"
        ])
    df = pd.DataFrame(rows)
    return df.sort_values(by=["Plate ID", "Y (mm)", "X (mm)"]).reset_index(drop=True)


def app():
    st.set_page_config(page_title="Plate Optimizer", layout="wide")
    st.markdown(_COMPACT_CSS, unsafe_allow_html=True)

    st.title("Plate Optimizer")
    st.caption(
        "• BH/CT stock-first nesting • staggered splice joints • weld joint map • "
        "Connection MR reuse • final consolidated order • tail merge optimizer • "
        "Mill offer control • CT support • PDF summary with drawings"
    )

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.subheader("Settings")
        kerf = st.number_input("Cutting kerf (mm)", 0, 10, 4)
        trim = st.number_input("Trim on each edge (mm)", 0, 200, 5)
        # min-util slider removed per request (leftover is reused anyway)
        priority = st.selectbox("Stock priority", ["Largest area", "Closest fit"])
        st.caption("Flange / Web splices are staggered ≥300 mm when both need splicing.")

        st.markdown("---")
        unified_file = st.file_uploader(
            "Upload Unified Excel (Sheet1 = BH/CT, Sheet2 = Connection MR)",
            type=["xlsx", "xls"]
        )
        stock_file = st.file_uploader("Upload Stock Plates (optional)", type=["xlsx", "xls", "csv"])
        mill_file = st.file_uploader("Mill Plate Offer (optional)", type=["xlsx", "xls", "csv"])
        st.caption("Mill Offer columns: Thickness (mm), Width (mm), Length (mm)")

    if not unified_file:
        st.info("Upload the unified Excel (BH/CT + Connection MR) to begin.")
        return

    # -------------- Load Input Data --------------
    try:
        df_bhct = pd.read_excel(unified_file, sheet_name=0)
        df_conn = pd.read_excel(unified_file, sheet_name=1)
    except Exception as e:
        st.error(f"Error reading BH/CT + Connection file: {e}")
        return

    if stock_file is not None:
        try:
            df_stock = pd.read_csv(stock_file) if stock_file.name.lower().endswith(".csv") else pd.read_excel(stock_file)
        except Exception as e:
            st.error(f"Error reading Stock file: {e}")
            return
    else:
        df_stock = None

    mill_sizes_dict = None
    df_mill_display = None
    if mill_file is not None:
        try:
            df_mill_raw = pd.read_csv(mill_file) if mill_file.name.lower().endswith(".csv") else pd.read_excel(mill_file)
        except Exception as e:
            st.error(f"Error reading Mill Offer file: {e}")
            return
        try:
            mill_sizes_dict = load_mill_offer(df_mill_raw)
            df_mill_display = df_mill_raw
        except Exception as e:
            st.error(f"Mill offer error: {e}")
            return

    # -------------- Run Optimizer Core Logic --------------
    MIN_UTIL_PCT_FOR_STOCK = 1  # effectively allow using stock; leftover reuse handles efficiency
    result = run_full_optimization(
        df_bhct=df_bhct,
        df_conn=df_conn,
        df_stock=df_stock,
        mill_sizes=mill_sizes_dict,
        kerf=kerf,
        trim=trim,
        min_util_pct=MIN_UTIL_PCT_FOR_STOCK,
        priority=priority,
    )

    # Unpack
    plates = result.get("plates_for_svg", result.get("plates", []))
    subs_all = result.get("subs_all", [])

    order_df               = result.get("order_df", pd.DataFrame())
    procurement_df         = result.get("procurement_df", pd.DataFrame())
    bhct_pieces_df         = result.get("bhct_pieces_df", pd.DataFrame())
    thickness_summary_df   = result.get("thickness_summary_df", pd.DataFrame())

    bhct_kpis              = result.get("bhct_kpis", {})
    conn_summary           = result.get("conn_summary", {})
    final_order_df         = result.get("final_order_df", pd.DataFrame())
    messages               = result.get("messages", [])
    xls_bytes              = result.get("xls_bytes", b"")

    grouped_conn_df        = result.get("grouped_conn_df", pd.DataFrame())
    conn_plan_df           = result.get("conn_plan_df", pd.DataFrame())
    conn_usage_map         = result.get("conn_usage_map", {})

    # KPI values
    total_bh_weight_kg      = float(bhct_kpis.get("total_bh_weight_kg", 0.0))
    total_plate_weight_kg   = float(bhct_kpis.get("total_plate_weight_kg", 0.0))
    utilization_pct_overall = float(bhct_kpis.get("utilization_pct_overall", 0.0))

    total_mr          = float(conn_summary.get("total_mr", 0.0))
    total_available   = float(conn_summary.get("total_available", 0.0))
    total_net_after   = float(conn_summary.get("total_net_after", 0.0))
    total_order       = float(conn_summary.get("total_order", 0.0))
    # pct_extra_total omitted from UI per request

    # Combined metrics (unchanged math)
    covered_from_leftovers = total_available
    net_conn_proc_kg       = total_order

    combined_total_mr = total_bh_weight_kg + total_mr
    combined_net_proc = total_plate_weight_kg + net_conn_proc_kg
    combined_util_pct = (
        (total_bh_weight_kg + covered_from_leftovers + net_conn_proc_kg)
        / (total_plate_weight_kg + net_conn_proc_kg)
        * 100.0
    ) if (total_plate_weight_kg + net_conn_proc_kg) > 0 else 0.0

    # -------------- KPIs (primary, larger) --------------
    st.markdown("### 1. BH/CT MR")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total BH/CT MR Weight (kg)", f"{total_bh_weight_kg:,.1f}")
    c2.metric("Total Plate Weight (kg)", f"{total_plate_weight_kg:,.1f}")
    c3.metric("Utilization (%)", f"{utilization_pct_overall:,.1f}%")
    c4.metric("Usable Leftover (kg)", f"{covered_from_leftovers:,.1f}")
    

    st.markdown("### 2. Connection Material MR")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total MR (kg)", f"{total_mr:,.1f}")
    d2.metric("Covered from Leftovers (kg)", f"{covered_from_leftovers:,.1f}")
    d3.metric("Net Procurement Requirement (kg)", f"{total_net_after:,.1f}")
    

    st.markdown("### 3. Combined — BH/CT + Connection")
    e1, e2, e3 = st.columns(3)
    e1.metric("Total MR (kg)", f"{combined_total_mr:,.1f}")
    e2.metric("Net Procurement (kg)", f"{combined_net_proc:,.1f}")
    e3.metric("% Utilization", f"{combined_util_pct:,.1f}%")
    

    # -------------- Input Preview Expanders --------------
    with st.expander("Input Preview — BH / CT (Sheet 1)", expanded=False):
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**BH/CT Input (first 50 rows)**")
            st.dataframe(df_bhct.head(50), use_container_width=True)
        with cB:
            st.markdown("**Thickness-wise Approx Weight Requirement**")
            if not thickness_summary_df.empty:
                st.dataframe(thickness_summary_df, use_container_width=True)
            else:
                st.dataframe(summarize_weight_by_thickness([]), use_container_width=True)

    with st.expander("Input Preview — Connection MR (Sheet 2)", expanded=False):
        st.dataframe(df_conn.head(50), use_container_width=True)

    if df_stock is not None:
        with st.expander("Input Preview — Stock Plates", expanded=False):
            st.dataframe(df_stock.head(50), use_container_width=True)

    if df_mill_display is not None:
        with st.expander("Input Preview — Mill Plate Offer", expanded=False):
            st.dataframe(df_mill_display.head(100), use_container_width=True)

    # -------------- Build PDF bytes --------------
    notes_text = "\n".join(messages) if messages else ""
    views = collect_splice_views(plates, subs_all)  # returns List[Tuple[str, svg_html]]
    pdf_bytes = build_pdf_report(
        project_name="Plate Optimization Report",
        bhct_kpis=bhct_kpis,
        conn_summary=conn_summary,
        final_order_df=final_order_df,
        notes_text=notes_text,
        plates=plates,
        kerf=kerf,
        trim=trim,
        conn_usage_map=conn_usage_map,
        splice_views=views,     # NEW
    )

    # -------------- Tabs --------------
    tabs = st.tabs([
        "Summary PDF Report",   # Tab 0
        "BH/CT Planner",        # Tab 1
        "Connection Plates",    # Tab 2
        "Final Order Sheet",    # Tab 3
        "Splice Joint Views",   # Tab 4
        "Marking Drawings",     # Tab 5
        "Shop Marking (Excel)", # Tab 6 - NEW
        "Exceptions / Notes",   # Tab 7
    ])

    # --- Tab 0: Summary PDF Report (KPIs removed here) ---
    with tabs[0]:
        st.markdown("### Executive Summary PDF")
        st.download_button(
            "📄 Download Full PDF Report (Summary + Drawings)",
            data=pdf_bytes,
            file_name="Plate_Optimization_Summary.pdf",
            mime="application/pdf"
        )
        st.markdown("#### Final Consolidated Order (Preview)")
        st.dataframe(
            final_order_df.head(12).style.format({
                "Final Qty": "{:,.0f}",
                "Final Total Weight (kg)": "{:,.1f}",
            }),
            use_container_width=True
        )

    # --- Tab 1: BH/CT Planner ---
    with tabs[1]:
        st.markdown("### Plate Orders (BH/CT Nesting Result)")
        st.dataframe(order_df, use_container_width=True)

        st.markdown("### Procurement Summary (BH/CT)")
        st.dataframe(procurement_df, use_container_width=True)

        st.download_button(
            "📥 Download Excel (Orders + Shop + BH/CT)",
            data=xls_bytes,
            file_name="BH_CT_plate_orders_and_shop_stock_first.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("### BH/CT Pieces (Exploded from Sections)")
        st.dataframe(bhct_pieces_df.head(200), use_container_width=True)

        # -------------------- DXF Export block (NEW) --------------------
        st.markdown("#### Export for DXF Generator")
        def _build_dxf_export_df(bhct_pieces_df, shop_df=None):
            needed = ["PieceUID","Type","T_mm","W_req","L_req","BH_Profile","Segments","Joints","PlateIDs","RID"]
            out = pd.DataFrame()

            if not bhct_pieces_df.empty:
                out = bhct_pieces_df.copy()

                # Harmonize possible alternate names
                rename_map = {
                    "Thickness (mm)": "T_mm",
                    "W (mm)": "W_req",
                    "H (mm)": "L_req",
                    "Kind": "Type",            # FLANGE/WEB sometimes stored as 'Kind'
                    "Label": "RID",            # optional: you may encode RID here
                }
                for k, v in rename_map.items():
                    if k in out.columns and v not in out.columns:
                        out[v] = out[k]

                # Ensure required columns exist
                for c in ["PieceUID","Type","T_mm","W_req","L_req","BH_Profile"]:
                    if c not in out.columns:
                        out[c] = ""

                # Normalize
                out["Type"] = out["Type"].astype(str).str.upper().replace({
                    "FLG": "FLANGE",
                    "FLANGE": "FLANGE",
                    "WEB": "WEB",
                })

                if "PieceUID" not in out.columns or out["PieceUID"].isna().all():
                    out["PieceUID"] = (out.index + 1).astype(str)

                keep_cols = [c for c in needed if c in out.columns]
                out = out[keep_cols]

                # Drop rows with missing geometry
                for c in ["T_mm", "W_req", "L_req"]:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
                out = out.dropna(subset=["W_req","L_req","T_mm"])

            # If empty, derive from placements (shop_df)
            if (out.empty) and (shop_df is not None) and (not shop_df.empty):
                tmp = shop_df.copy()
                tmp["PieceUID"] = tmp.get("ParentID", "").astype(str) + "_" + tmp.get("SubIndex","").astype(str)
                tmp["Type"] = tmp.get("Kind","").astype(str).str.upper().replace({"FLG":"FLANGE"})
                tmp["T_mm"] = pd.to_numeric(tmp.get("Thickness (mm)", None), errors="coerce")
                tmp["W_req"] = pd.to_numeric(tmp.get("W (mm)", None), errors="coerce")
                tmp["L_req"] = pd.to_numeric(tmp.get("H (mm)", None), errors="coerce")
                tmp["BH_Profile"] = tmp.get("BH_Profile","")
                out = tmp[["PieceUID","Type","T_mm","W_req","L_req","BH_Profile"]].dropna(subset=["W_req","L_req","T_mm"])

            return out.reset_index(drop=True)

        shop_df_for_dxf = _build_shop_marking_df(plates)
        dxf_export_df = _build_dxf_export_df(bhct_pieces_df, shop_df=shop_df_for_dxf)

        st.dataframe(dxf_export_df.head(50), use_container_width=True)

        buf_dxf = io.BytesIO()
        dxf_export_df.to_csv(buf_dxf, index=False)
        buf_dxf.seek(0)
        st.download_button(
            "📤 Download CSV for DXF Generator",
            data=buf_dxf.getvalue(),
            file_name="bh_pieces.csv",
            mime="text/csv",
        )
        st.caption("Download this CSV and upload it in the separate dfx.py app to generate DXF drawings with dimensions.")
        # ------------------ End DXF Export block ------------------------

    # --- Tab 2: Connection Plates (KPIs removed here) ---
    with tabs[2]:
        st.markdown("### Connection MR Summary")
        st.write("Grouped by thickness with BH/CT leftover reuse applied.")
        if not grouped_conn_df.empty:
            st.dataframe(grouped_conn_df, use_container_width=True)
        if not conn_plan_df.empty:
            st.markdown("#### Proposed Connection Plate Ordering Plan")
            st.dataframe(conn_plan_df, use_container_width=True)

    # --- Tab 3: Final Order Sheet ---
    with tabs[3]:
        st.markdown("### Final Consolidated Ordering Sheet")
        st.dataframe(
            final_order_df.style.format({
                "Final Qty": "{:,.0f}",
                "Final Total Weight (kg)": "{:,.1f}",
            }),
            use_container_width=True
        )

    # --- Tab 4: Splice Joint Views ---
    with tabs[4]:
        st.markdown("### Splice Joint Views (Staggered Weld Joints)")
        views = collect_splice_views(plates, subs_all)
        if not views:
            st.info("No spliced members found.")
        else:
            for title, svg_html in views:
                st.markdown(f"**{title}**")
                st.components.v1.html(svg_html, height=200, scrolling=False)
                st.markdown("---")

    # --- Tab 5: Marking Drawings ---
    with tabs[5]:
        st.markdown("### Plate Marking Drawings")
        zbytes = build_markings_zip(plates, kerf, trim)
        st.download_button(
            "📦 Download All Markings (ZIP)",
            data=zbytes,
            file_name="markings_stock_first.zip",
            mime="application/zip"
        )

        st.caption("Preview (first few plates):")
        for sp in plates[:min(300, len(plates))]:
            try:
                util_pct = sp.utilization() * 100.0
            except Exception:
                util_pct = 0.0
            st.markdown(
                f"**{getattr(sp,'plate_id','')}** "
                f"[{getattr(sp,'source','?')}] — "
                f"t{getattr(sp,'thickness_mm','?')} | "
                f"{getattr(sp,'stock_width_mm','?')}×{getattr(sp,'stock_length_mm','?')} | "
                f"Util {util_pct:.1f}%"
            )
            svg_html = svg_marking_2d(sp, kerf, trim)
            st.components.v1.html(svg_html, height=520, scrolling=True)

    # --- Tab 6: Shop Marking (Excel) — NEW ---
    with tabs[6]:
        st.markdown("### Shop Marking — Placement Detail (Excel)")
        shop_df = _build_shop_marking_df(plates)
        st.dataframe(shop_df.head(200), use_container_width=True)

        # build Excel on the fly
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            shop_df.to_excel(writer, index=False, sheet_name="Shop Marking")
        buf.seek(0)
        st.download_button(
            "📥 Download Shop Marking (Excel)",
            data=buf.read(),
            file_name="shop_marking.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # --- Tab 7: Exceptions / Notes ---
    with tabs[7]:
        if messages:
            st.warning("\n".join(messages))
        else:
            st.success("No exceptions.")


if __name__ == "__main__":
    app()
