# data_io.py
import math
import pandas as pd
from typing import List, Dict, Tuple, Optional, DefaultDict
from collections import defaultdict

from plate_rules import (
    BHRow, PlatePiece, SubPiece, InvPlate,
    usable_cap_for_thickness, plan_staggered_splits_for_bh,
    STEEL_DENSITY_KG_PER_M3, DENSITY
)

############################
# FILE PARSERS / LOADERS
############################

def load_bhct_rows(df_bh: pd.DataFrame) -> List[BHRow]:
    """
    Expect columns:
    PROFILE, LENGTH (mm), UNIT WEIGHT(Kg), QTY., TOTAL WEIGHT(Kg)
    """
    req = ["PROFILE","LENGTH (mm)","UNIT WEIGHT(Kg)","QTY.","TOTAL WEIGHT(Kg)"]
    missing = [c for c in req if c not in df_bh.columns]
    if missing:
        raise ValueError(f"BH/CT sheet missing columns: {missing}")

    rows=[]
    for _, r in df_bh.iterrows():
        br = BHRow(
            profile=str(r["PROFILE"]).strip(),
            length_mm=int(r["LENGTH (mm)"]),
            unit_weight_kg=float(r["UNIT WEIGHT(Kg)"]),
            qty=int(r["QTY."]),
            total_weight_kg=float(r["TOTAL WEIGHT(Kg)"]),
        )
        br.parse()  # fills sec_type/H/B/tw/tf
        rows.append(br)
    return rows

def load_stock_inventory(df_stock: pd.DataFrame) -> List[InvPlate]:
    """
    Expect columns:
    T (mm), W (mm), L (mm), Qty
    (optional) Weight (Kg)
    """
    req = ["T (mm)","W (mm)","L (mm)","Qty"]
    missing = [c for c in req if c not in df_stock.columns]
    if missing:
        raise ValueError(f"Stock sheet missing columns: {missing}")

    out=[]
    for _, r in df_stock.iterrows():
        if int(r["Qty"]) <= 0:
            continue
        out.append(
            InvPlate(
                t=int(r["T (mm)"]),
                w=int(r["W (mm)"]),
                l=int(r["L (mm)"]),
                qty=int(r["Qty"]),
                weight=float(r["Weight (Kg)"]) if "Weight (Kg)" in df_stock.columns and pd.notna(r["Weight (Kg)"]) else 0.0
            )
        )
    return out

def load_mill_offer(df_mill: pd.DataFrame) -> Dict[int, List[Tuple[int,int]]]:
    """
    Mill Offer columns:
    Thickness (mm), Width (mm), Length (mm)

    Returns:
      { thickness_mm : [(W,L), ... unique sorted] }
    """
    req = ["Thickness (mm)","Width (mm)","Length (mm)"]
    missing = [c for c in req if c not in df_mill.columns]
    if missing:
        raise ValueError(f"Mill Offer missing columns: {missing}")

    temp: DefaultDict[int, List[Tuple[int,int]]] = defaultdict(list)
    for _,r in df_mill.iterrows():
        t = int(r["Thickness (mm)"])
        W = int(r["Width (mm)"])
        L = int(r["Length (mm)"])
        temp[t].append((W,L))

    mill_sizes = {}
    for t, pairs in temp.items():
        mill_sizes[t] = sorted(list(set(pairs)))
    return mill_sizes

############################
# BH/CT piece explosion
############################

def explode_rows_to_unit_pieces(rows: List[BHRow]) -> List[PlatePiece]:
    """
    Turn each BHRow (BH or CT section) into plate pieces:
      BH: flange x2 + web x1 per qty
      CT: flange x1 + web x1 per qty
    """
    pieces=[]
    for br in rows:
        if br.sec_type == "CT":
            # CT flange (single flange)
            for _ in range(br.qty):
                pieces.append(
                    PlatePiece("flange", br.tf, br.flange_width(), br.length_mm, 1, br.profile)
                )
            # CT web
            for _ in range(br.qty):
                pieces.append(
                    PlatePiece("web", br.tw, br.web_width(), br.length_mm, 1, br.profile)
                )
        else:
            # BH flange: 2 per qty
            for _ in range(br.qty*2):
                pieces.append(
                    PlatePiece("flange", br.tf, br.flange_width(), br.length_mm, 1, br.profile)
                )
            # BH web: 1 per qty
            for _ in range(br.qty):
                pieces.append(
                    PlatePiece("web", br.tw, br.web_width(), br.length_mm, 1, br.profile)
                )
    return pieces

def build_all_subpieces_with_stagger(
    unit_pieces: List[PlatePiece],
    trim_mm:int,
    kerf:int,
    mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
) -> List[SubPiece]:
    """
    Apply splice logic + stagger rule on each BH/CT member.
    Output SubPieces (1 or 2 segments per piece).
    """
    # Group by (profile, length, kind='flange'/'web')
    by_key: DefaultDict[Tuple[str,int,str], List[PlatePiece]] = defaultdict(list)
    for p in unit_pieces:
        by_key[(p.bh_profile, p.length_mm, p.kind)].append(p)

    # Find splice joint position per member length/type
    joint_pos_map: Dict[Tuple[str,int,str], List[Optional[int]]] = {}
    bh_lengths = sorted({(p.bh_profile, p.length_mm) for p in unit_pieces})

    for bh, L in bh_lengths:
        fl_list = by_key.get((bh, L, "flange"), [])
        wb_list = by_key.get((bh, L, "web"), [])

        if fl_list and wb_list:
            tf = fl_list[0].thickness_mm
            tw = wb_list[0].thickness_mm
            pos_f, pos_w = plan_staggered_splits_for_bh(
                length_mm=L,
                flange_t=tf,
                web_t=tw,
                trim_mm=trim_mm,
                kerf=kerf,
                mill_sizes=mill_sizes,
                min_stagger_mm=300
            )
        else:
            any_list = fl_list or wb_list
            tt = any_list[0].thickness_mm if any_list else 0
            cap = usable_cap_for_thickness(tt, trim_mm, kerf, mill_sizes)
            if L > cap:
                lower = math.floor(L/3); upper = math.ceil(2*L/3)
                pos = min(cap, max(lower, L-cap))
            else:
                pos = None
            pos_f = pos if fl_list else None
            pos_w = pos if wb_list else None

        joint_pos_map[(bh, L, "flange")] = [pos_f]*len(fl_list)
        joint_pos_map[(bh, L, "web")]    = [pos_w]*len(wb_list)

    # Now actually split the pieces
    subs: List[SubPiece] = []
    for (bh, L, kind), items in by_key.items():
        for i,p in enumerate(items):
            splice_pos = joint_pos_map[(bh,L,kind)][i]
            cap = usable_cap_for_thickness(p.thickness_mm, trim_mm, kerf, mill_sizes)
            pid = id(p)

            if splice_pos is None or L <= cap:
                subs.append(SubPiece(
                    parent_id=pid,
                    index=1,
                    total_len_mm=L,
                    length_mm=L,
                    width_mm=p.width_mm,
                    thickness_mm=p.thickness_mm,
                    kind=kind,
                    bh_profile=bh,
                    splice_joint_here=False,
                    joint_pos_mm=None
                ))
            else:
                a = splice_pos
                b = L - a
                subs.append(SubPiece(
                    parent_id=pid,
                    index=1,
                    total_len_mm=L,
                    length_mm=a,
                    width_mm=p.width_mm,
                    thickness_mm=p.thickness_mm,
                    kind=kind,
                    bh_profile=bh,
                    splice_joint_here=False,
                    joint_pos_mm=splice_pos
                ))
                subs.append(SubPiece(
                    parent_id=pid,
                    index=2,
                    total_len_mm=L,
                    length_mm=b,
                    width_mm=p.width_mm,
                    thickness_mm=p.thickness_mm,
                    kind=kind,
                    bh_profile=bh,
                    splice_joint_here=True,
                    joint_pos_mm=splice_pos
                ))
    return subs

############################
# BH/CT thickness weight summary
############################

def summarize_weight_by_thickness(unit_pieces: List[PlatePiece]) -> pd.DataFrame:
    """
    Approximate weight by thickness from BH/CT requirements
    (before splicing).
    """
    rows_weight=[]
    for p in unit_pieces:
        thk_m   = p.thickness_mm / 1000.0
        w_m     = p.width_mm     / 1000.0
        L_m     = p.length_mm    / 1000.0
        vol_m3  = thk_m * w_m * L_m
        wt_kg   = vol_m3 * STEEL_DENSITY_KG_PER_M3
        rows_weight.append({
            "Thickness (mm)": p.thickness_mm,
            "Approx Weight (kg)": wt_kg
        })
    if not rows_weight:
        return pd.DataFrame(columns=["Thickness (mm)","Total Weight (kg)","% Share"])

    df_tmp = pd.DataFrame(rows_weight)
    grouped = (
        df_tmp
        .groupby("Thickness (mm)", as_index=False)["Approx Weight (kg)"]
        .sum()
        .rename(columns={"Approx Weight (kg)":"Total Weight (kg)"})
    )
    total_all = grouped["Total Weight (kg)"].sum()
    grouped["% Share"] = (
        grouped["Total Weight (kg)"] /
        (total_all if total_all>0 else 1)
        * 100.0
    )

    total_row = {
        "Thickness (mm)": "TOTAL",
        "Total Weight (kg)": total_all,
        "% Share": 100.0 if total_all>0 else 0.0
    }
    grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)
    return grouped

############################
# Connection MR helpers
############################

def clean_connection_mr(df_conn_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns:
    Thickness, Weight (kg)
    We normalize Thickness -> Thickness_mm int
    """
    df_conn = df_conn_raw.copy()
    df_conn["Thickness_mm"] = (
        df_conn["Thickness"]
        .astype(str).str.upper().str.replace("PL","", regex=False)
        .astype(float).astype(int)
    )
    return df_conn

def group_connection_weight(df_conn: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df_conn
        .groupby("Thickness_mm", as_index=False)
        .agg({"Weight (kg)": "sum"})
        .rename(columns={"Weight (kg)": "Req_Wt_Original(kg)"})
    )
    return grouped

############################
# Leftover calculation from low-util plates
############################

def estimate_full_plate_weight_kg(thk_mm:int, w_mm:int, l_mm:int) -> float:
    """
    Full plate weight (kg) for given thickness and size.
    """
    area_m2 = (w_mm * l_mm) * 1e-6
    return area_m2 * (thk_mm/1000.0) * DENSITY

def leftover_weight_dict_from_low_util(plates, util_threshold:float=0.80) -> Dict[int, float]:
    """
    For each planned plate, if utilization < threshold,
    consider leftover weight = unused % of that plate.
    Sum leftover by thickness.
    """
    leftover = {}
    for sp in plates:
        util = sp.utilization()
        if util < util_threshold:
            full_wt = estimate_full_plate_weight_kg(sp.thickness_mm, sp.stock_width_mm, sp.stock_length_mm)
            bal = full_wt * (1.0 - util)
            leftover[sp.thickness_mm] = leftover.get(sp.thickness_mm, 0.0) + bal
    return leftover

def load_bhct_and_conn(unified_file) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the uploaded unified Excel:
      Sheet1 = BH/CT requirement table
      Sheet2 = Connection MR requirement table

    Returns:
        df_bhct (DataFrame)
        df_conn (DataFrame)
    """
    # read_excel with sheet_name=None gives dict of all sheets,
    # but here we explicitly want first two sheets in order
    # because users may rename them.
    xls = pd.ExcelFile(unified_file)

    if len(xls.sheet_names) < 2:
        raise ValueError("Unified Excel must have at least two sheets: [BH/CT, Connection MR].")

    sheet_bhct = xls.sheet_names[0]
    sheet_conn = xls.sheet_names[1]

    df_bhct = pd.read_excel(xls, sheet_name=sheet_bhct)
    df_conn = pd.read_excel(xls, sheet_name=sheet_conn)

    return df_bhct, df_conn
