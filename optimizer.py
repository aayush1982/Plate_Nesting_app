# optimizer.py
import io, math
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set, DefaultDict, Any
from collections import defaultdict
from rectpack import newPacker, PackingMode, PackingBin, MaxRectsBssf, SORT_AREA

from plate_rules import (
    BHRow, PlatePiece, SubPiece, StockPlate, Placement, InvPlate,
    STEEL_DENSITY_KG_PER_M3, DENSITY,
    get_allowed_plate_sizes_for_thickness,
    standard_plate_options_for_thickness_conn,
    conditional_split_wide_webs,   # WEB-only width split (only if necessary)
)

from data_io import (
    explode_rows_to_unit_pieces,
    build_all_subpieces_with_stagger,
    summarize_weight_by_thickness,
    clean_connection_mr,
    group_connection_weight,
    leftover_weight_dict_from_low_util,
    estimate_full_plate_weight_kg,
)

##########################################
# Helpers to prefer ≤2500 mm plates unless necessary
##########################################
def _needs_wider_than_limit(subs: List[SubPiece], usable_limit_mm: int) -> bool:
    """
    True if any subpiece width exceeds usable_limit_mm (i.e., cannot fit in a plate
    of width ≤ SOFT limit after trim).
    """
    for s in subs:
        if s.width_mm > usable_limit_mm:
            return True
    return False


def allowed_size_options_for_t(
    t: int,
    subs: List[SubPiece],
    kerf: int,
    trim: int,
    mill_sizes: Optional[Dict[int, List[Tuple[int, int]]]],
    soft_w_limit: int = 2500,
) -> List[Tuple[int, int]]:
    """
    Get plate (W, L) options for this thickness, but restrict to W ≤ soft_w_limit
    unless it's necessary to go wider (i.e., at least one subpiece width exceeds
    the usable width on a ≤ soft limit plate after trim).

    If wider is necessary, we keep all options, but we still sort so ≤ soft limit
    widths are preferred when score ties.
    """
    opts = get_allowed_plate_sizes_for_thickness(t, mill_sizes)
    if not opts:
        return []

    usable_limit_mm = soft_w_limit - 2 * trim

    need_wide = _needs_wider_than_limit(subs, usable_limit_mm)

    if not need_wide:
        restricted = [wl for wl in opts if wl[0] <= soft_w_limit]
        if restricted:
            return restricted

    # either we need wide, or nothing under soft limit exists for this thickness
    return sorted(opts, key=lambda wl: (wl[0] > soft_w_limit, wl[0], wl[1]))

##########################################
# RECTPACK UTILITIES
##########################################
def _rectpack_place(rects: List[SubPiece], stock_w:int, stock_l:int, kerf:int, trim:int):
    """
    Packs rects into a single plate (stock_w x stock_l).
    Returns placement list and set of used indices.
    """
    if not rects:
        return [], set()

    usable_h = stock_w - 2*trim  # treat stock width as Y
    usable_w = stock_l - 2*trim  # treat stock length as X
    if usable_h <= 0 or usable_w <= 0:
        return [], set()

    indexed_rects = list(enumerate(rects))
    indexed_rects.sort(
        key=lambda kv: kv[1].length_mm * kv[1].width_mm,
        reverse=True
    )

    packer = newPacker(
        PackingMode.Offline,
        PackingBin.BBF,
        MaxRectsBssf,
        SORT_AREA,
        True,
    )
    packer.add_bin(usable_w, usable_h, count=1)

    for rid, sp in indexed_rects:
        need_x = sp.length_mm + kerf
        need_y = sp.width_mm  + kerf
        packer.add_rect(need_x, need_y, rid=rid)

    packer.pack()

    used_idx: Set[int] = set()
    placements_raw = []
    for (bin_idx, x, y, w, h, rid) in packer.rect_list():
        if bin_idx != 0:
            continue
        s = rects[rid]

        not_rot = (w == s.length_mm + kerf and h == s.width_mm  + kerf)
        rot_90  = (w == s.width_mm  + kerf and h == s.length_mm + kerf)

        if not_rot:
            final_w = s.length_mm
            final_h = s.width_mm
        elif rot_90:
            final_w = s.width_mm
            final_h = s.length_mm
        else:
            diff_notrot = abs(w - (s.length_mm+kerf)) + abs(h - (s.width_mm+kerf))
            diff_rot    = abs(w - (s.width_mm+kerf))   + abs(h - (s.length_mm+kerf))
            if diff_notrot <= diff_rot:
                final_w = s.length_mm
                final_h = s.width_mm
            else:
                final_w = s.width_mm
                final_h = s.length_mm

        placements_raw.append((
            int(x + trim),
            int(y + trim),
            int(final_w),
            int(final_h),
            rid
        ))
        used_idx.add(rid)

    return placements_raw, used_idx


def build_plate_from_rectpack(rects: List[SubPiece],
                              stock_w:int,
                              stock_l:int,
                              kerf:int,
                              trim:int,
                              source:str) -> Tuple[StockPlate, Set[int]]:
    """
    Convert rectpack packing result into StockPlate + used idx set
    """
    sp = StockPlate(
        plate_id="TEMP",
        thickness_mm= rects[0].thickness_mm if rects else 0,
        stock_width_mm=stock_w,
        stock_length_mm=stock_l,
        placements=[],
        trim_mm=trim,
        kerf_mm=kerf,
        source=source
    )

    placements_raw, used_idx = _rectpack_place(rects, stock_w, stock_l, kerf, trim)
    for (x, y, w, h, rid) in placements_raw:
        s = rects[rid]
        sp.placements.append(
            Placement(
                x=x,
                y=y,
                w=w,
                h=h,
                label=f"{s.kind[:1].upper()} {s.length_mm}×{s.width_mm}×{s.thickness_mm}",
                annotate=f"{s.bh_profile}{' | WELD JOINT' if s.splice_joint_here else ''}",
                parent_id=s.parent_id,
                sub_index=s.index,
                bh_profile=s.bh_profile,
                kind=s.kind,
            )
        )
    return sp, used_idx


def _pack_once(subs:List[SubPiece],
               W:int, L:int,
               kerf:int, trim:int,
               source:str="standard"):
    """
    Try to pack `subs` into one (W,L) plate. Return (plate, used_idxs, remaining_subs)
    """
    if not subs:
        return None, set(), []
    plate, used = build_plate_from_rectpack(subs, W, L, kerf, trim, source)
    if not used:
        return None, set(), subs[:]
    rem = [s for i,s in enumerate(subs) if i not in used]
    return plate, used, rem


def _plate_waste_area(pl:StockPlate)->float:
    """
    Waste area = full plate area - utilized area
    """
    if pl is None: return 0.0
    plate_area = pl.stock_width_mm * pl.stock_length_mm
    util_frac  = pl.utilization()
    used_area  = plate_area * util_frac
    return plate_area - used_area


def explain_unplaceable(
    t:int,
    subs:List[SubPiece],
    trim:int,
    kerf:int,
    mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
)->List[str]:
    """
    Diagnostic message if a subpiece can't be packed in allowed sizes
    """
    opts = get_allowed_plate_sizes_for_thickness(t, mill_sizes)
    if not opts:
        return [f"No plate sizes available for t={t}mm."]

    max_w = max(W for (W,_) in opts)
    max_l = max(L for (_,L) in opts)
    usable_w_max = max_w - 2*trim - kerf
    usable_l_max = max_l - 2*trim - kerf

    notes=[]
    for s in subs:
        too_wide  = s.width_mm  > usable_w_max
        too_long  = s.length_mm > usable_l_max
        if too_wide or too_long:
            msg = f"{s.kind.upper()} {s.length_mm}×{s.width_mm}×{s.thickness_mm} ({s.bh_profile})"
            reasons=[]
            if too_wide:
                reasons.append(f"width {s.width_mm} > max usable {usable_w_max}")
            if too_long:
                reasons.append(f"length {s.length_mm} > max usable {usable_l_max}")
            notes.append(msg + " — " + ", ".join(reasons))
    return notes

##########################################
# INVENTORY-FIRST PLANNING
##########################################
def plan_on_inventory(
    subs_by_t:Dict[int,List[SubPiece]],
    inventory:List[InvPlate],
    kerf:int,
    trim:int,
    min_util_pct:float,
    priority:str,
    start_serial:int=1
):
    """
    Use site stock first.
    Only keep plates if utilization >= min_util_pct.
    """
    plates:List[StockPlate] = []
    remaining = {t:list(v) for t,v in subs_by_t.items()}
    serial = start_serial
    errors:List[str] = []

    # group inventory plates by thickness
    inv_by_t: DefaultDict[int, List[InvPlate]] = defaultdict(list)
    for pl in inventory:
        inv_by_t[pl.t].append(pl)

    for t, inv_list in inv_by_t.items():
        subs = remaining.get(t, [])
        if not subs:
            continue

        if priority == "Largest area":
            inv_list.sort(key=lambda r: r.w*r.l, reverse=True)
        elif priority == "Closest fit":
            inv_list.sort(key=lambda r: (r.w, r.l))

        for inv in inv_list:
            count = inv.qty
            while count>0 and subs:
                plate, used = build_plate_from_rectpack(subs, inv.w, inv.l, kerf, trim, source="inventory")
                if not used:
                    break

                util = plate.utilization()*100
                if util < min_util_pct:
                    errors.append(
                        f"Skipped stock {inv.w}×{inv.l} t{t} "
                        f"(util {util:.1f}% < {min_util_pct}%)"
                    )
                    break

                subs = [s for i,s in enumerate(subs) if i not in used]
                plate.plate_id = f"S{serial:04d}"
                serial += 1
                plates.append(plate)
                count -= 1

            remaining[t] = subs
    return plates, remaining, serial, errors

##########################################
# STANDARD PLANNING + TAIL MERGE
##########################################
def plan_on_standard(
    remaining_by_t:Dict[int,List[SubPiece]],
    kerf:int,
    trim:int,
    start_serial:int,
    mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
):
    """
    Choose best standard plate size each time based on
    waste + 'tail penalty' heuristic.
    """
    plates_all:List[StockPlate] = []
    plates_by_t: Dict[int,List[StockPlate]] = defaultdict(list)
    serial = start_serial
    notes: List[str] = []

    TARGET_UTIL_FRACTION = 0.70

    for t, subs in sorted(remaining_by_t.items()):
        if not subs:
            continue

        # <<< Use width soft-limit (≤2500) unless necessary to go wider >>>
        size_options = allowed_size_options_for_t(
            t=t,
            subs=subs,
            kerf=kerf,
            trim=trim,
            mill_sizes=mill_sizes,
            soft_w_limit=2500,
        )

        while subs:
            best_choice = None
            for (W, L) in size_options:
                plA, usedA, remA = _pack_once(subs, W, L, kerf, trim, source="standard")
                if not usedA:
                    continue

                wasteA = _plate_waste_area(plA)

                remaining_area = sum(piece.length_mm * piece.width_mm for piece in remA)
                plate_area = W * L
                effective_capacity = plate_area * TARGET_UTIL_FRACTION
                approx_more_plates = (
                    remaining_area / effective_capacity
                    if effective_capacity>0 else 999999
                )
                tail_penalty_area = approx_more_plates * plate_area * (1 - TARGET_UTIL_FRACTION)
                score = wasteA + tail_penalty_area

                cand = {
                    "score": score,
                    "plA": plA,
                    "usedA": usedA,
                    "plate_area": plate_area,
                    "remA": remA,
                    "W": W,
                    "L": L,
                }
                if (
                    best_choice is None or
                    cand["score"] < best_choice["score"] or
                    (
                        cand["score"] == best_choice["score"]
                        and (cand["W"], cand["L"]) < (best_choice["W"], best_choice["L"])
                    )
                ):
                    best_choice = cand

            if best_choice is None:
                msg = f"No further packing possible for t={t}mm; remaining pieces: {len(subs)}"
                diag = explain_unplaceable(t, subs, trim, kerf, mill_sizes)
                if diag:
                    msg += "\n" + "\n".join(diag)
                notes.append(msg)
                break

            pl_final = best_choice["plA"]
            used_final = best_choice["usedA"]
            subs = [s for i,s in enumerate(subs) if i not in used_final]

            pl_final.plate_id = f"N{serial:04d}"
            serial += 1
            plates_all.append(pl_final)
            plates_by_t[t].append(pl_final)

        remaining_by_t[t] = subs

    return plates_all, serial, notes, plates_by_t


def optimize_tail_plates(
    plates_by_t:Dict[int,List[StockPlate]],
    kerf:int,
    trim:int,
    mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
):
    """
    Merge last 1-2 worst-util standard plates (per thickness)
    into 1 better-util plate, if possible.
    """
    messages: List[str] = []
    final_list: List[StockPlate] = []

    def placements_to_subpieces(pl:StockPlate)->List[SubPiece]:
        subs_local=[]
        for plc in pl.placements:
            subs_local.append(
                SubPiece(
                    parent_id = plc.parent_id,
                    index = plc.sub_index,
                    total_len_mm = plc.w if plc.w >= plc.h else plc.h,
                    length_mm = plc.w,
                    width_mm = plc.h,
                    thickness_mm = pl.thickness_mm,
                    kind = plc.kind,
                    bh_profile = plc.bh_profile,
                    splice_joint_here = ("WELD JOINT" in plc.annotate),
                    joint_pos_mm = None
                )
            )
        return subs_local

    for t, plist in plates_by_t.items():
        if not plist:
            continue

        std_plates = [p for p in plist if p.source=="standard"]
        inv_plates = [p for p in plist if p.source!="standard"]

        if len(std_plates) < 2:
            final_list.extend(inv_plates + std_plates)
            continue

        # sort by utilization ascending
        std_sorted = sorted(std_plates, key=lambda p: p.utilization())
        tail_group = std_sorted[:2]

        # if both tails >30%, don't bother
        if all(p.utilization() >= 0.30 for p in tail_group):
            final_list.extend(inv_plates + std_plates)
            continue

        # collect pieces from those tail plates
        tail_subpieces=[]
        for tp in tail_group:
            tail_subpieces.extend(placements_to_subpieces(tp))

        # attempt repack into one plate
        size_options = get_allowed_plate_sizes_for_thickness(t, mill_sizes)

        best_merge = None
        for (W, L) in size_options:
            merged_plate, used_idx = build_plate_from_rectpack(
                tail_subpieces, W, L, kerf, trim, source="standard"
            )
            if len(used_idx) != len(tail_subpieces):
                # couldn't place everything
                continue

            util_new = merged_plate.utilization()
            old_waste = sum(_plate_waste_area(p) for p in tail_group)
            new_waste = _plate_waste_area(merged_plate)
            if (best_merge is None) or (new_waste < best_merge["new_waste"]):
                best_merge = {
                    "merged_plate": merged_plate,
                    "new_waste": new_waste,
                    "old_waste": old_waste,
                    "util_new": util_new,
                    "W": W,
                    "L": L,
                }

        if best_merge is None:
            final_list.extend(inv_plates + std_plates)
            continue

        if best_merge["new_waste"] < best_merge["old_waste"]:
            kept_std = [p for p in std_plates if p not in tail_group]
            merged_plate = best_merge["merged_plate"]
            merged_plate.plate_id = "TO_RENUMBER"
            final_list.extend(inv_plates + kept_std + [merged_plate])
            messages.append(
                f"Thickness {t}mm: merged {len(tail_group)} low-util plates "
                f"into {best_merge['W']}×{best_merge['L']} "
                f"(util {best_merge['util_new']*100:.1f}%)."
            )
        else:
            final_list.extend(inv_plates + std_plates)

    return final_list, messages

##########################################
# MASTER BH/CT PLAN
##########################################
def master_plan_bhct(
    bh_rows: List[BHRow],
    inventory: List[InvPlate],
    kerf:int,
    trim:int,
    min_util_pct:float,
    priority:str,
    mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
):
    """
    Full BH/CT nesting pipeline.
    Returns:
      plates (final list StockPlate),
      per-plate order_df,
      procurement_df,
      bh_pieces_df,
      subs_all,
      messages,
      kpis...
    """
    # explode BH/CT rows -> individual flange/web pieces
    unit_pieces = explode_rows_to_unit_pieces(bh_rows)

    # splice logic & stagger (length-wise)
    subs_all = build_all_subpieces_with_stagger(
        unit_pieces,
        trim_mm=trim,
        kerf=kerf,
        mill_sizes=mill_sizes
    )

    # NEW: conditional width-split for **WEBs** only if no available plate width can fit
    subs_all = conditional_split_wide_webs(
        subs_all,
        trim_mm=trim,
        kerf=kerf,
        mill_sizes=mill_sizes
    )

    # organize by thickness
    by_t: DefaultDict[int,List[SubPiece]] = defaultdict(list)
    for s in subs_all:
        by_t[s.thickness_mm].append(s)

    # 1. use inventory plates
    plates_inv, remaining, serial, inv_errors = plan_on_inventory(
        by_t,
        inventory,
        kerf,
        trim,
        min_util_pct,
        priority,
        start_serial=1
    )

    # 2. fill balance with standard plates (respect soft width limit unless required)
    plates_std, serial, std_notes, plates_by_t = plan_on_standard(
        remaining,
        kerf,
        trim,
        serial,
        mill_sizes
    )

    # 3. tail merge cleanup
    merged_list, tail_msgs = optimize_tail_plates(
        plates_by_t,
        kerf,
        trim,
        mill_sizes
    )

    # rebuild + renumber standard plates from tail merge
    inventory_final = [p for p in plates_inv]
    standard_final  = [p for p in merged_list if p.source=="standard"]
    from_merge_inv  = [p for p in merged_list if p.source!="standard"]
    for p in from_merge_inv:
        if p not in inventory_final:
            inventory_final.append(p)

    new_standard_final=[]
    for p in standard_final:
        p.plate_id = f"N{serial:04d}"
        serial += 1
        new_standard_final.append(p)

    all_plates = inventory_final + new_standard_final

    # build order_df (= one row per physical plate)
    order_rows=[]
    total_plate_weight_kg = 0.0
    for sp in all_plates:
        wt = estimate_full_plate_weight_kg(sp.thickness_mm, sp.stock_width_mm, sp.stock_length_mm)
        total_plate_weight_kg += wt
        order_rows.append({
            "Plate ID": sp.plate_id,
            "Source": sp.source.title(),
            "Thickness (mm)": sp.thickness_mm,
            "Width (mm)": sp.stock_width_mm,
            "Length (mm)": sp.stock_length_mm,
            "Est. Weight (kg)": round(wt,1),
            "Utilization %": round(sp.utilization()*100,1),
            "Weld Cuts?": any("WELD JOINT" in plc.annotate for plc in sp.placements),
        })
    order_df = pd.DataFrame(order_rows)

    # procurement summary (grouped by t,W,L,Source)
    procurement_rows=[]
    if len(all_plates)>0:
        df_temp = pd.DataFrame([{
            "Source": p.source.title(),
            "Thickness (mm)": p.thickness_mm,
            "Width (mm)": p.stock_width_mm,
            "Length (mm)": p.stock_length_mm
        } for p in all_plates])

        for (src,t,W,L), grp in df_temp.groupby(["Source","Thickness (mm)","Width (mm)","Length (mm)"]):
            qty = len(grp)
            tot_wt = qty * estimate_full_plate_weight_kg(t, W, L)
            avg_util = pd.Series([
                pl.utilization()*100
                for pl in all_plates
                if pl.source.title()==src
                and pl.thickness_mm==t
                and pl.stock_width_mm==W
                and pl.stock_length_mm==L
            ]).mean()
            procurement_rows.append({
                "Source": src,
                "Thickness (mm)": t,
                "Chosen Width (mm)": W,
                "Chosen Length (mm)": L,
                "No. of Plates": qty,
                "Total Weight (kg)": round(tot_wt,1),
                "Avg Utilization %": round(avg_util,1)
            })
    procurement_df = pd.DataFrame(procurement_rows)

    # BH piece listing (before cutting)
    bh_pieces_df = pd.DataFrame([{
        "BH/CT": p.bh_profile,
        "Kind": p.kind,
        "t (mm)": p.thickness_mm,
        "w (mm)": p.width_mm,
        "L (mm)": p.length_mm,
        "Qty": 1
    } for p in unit_pieces])

    # KPIs
    total_bh_weight_kg = sum(r.total_weight_kg for r in bh_rows)
    util_pct_overall = (
        (total_bh_weight_kg / total_plate_weight_kg * 100.0)
        if total_plate_weight_kg>0 else 0.0
    )

    # thickness summary for sidebar view
    thickness_summary_df = summarize_weight_by_thickness(unit_pieces)

    # messages
    messages = []
    messages.extend(inv_errors)
    messages.extend(std_notes)
    messages.extend(tail_msgs)

    return (
        all_plates,
        order_df,
        procurement_df,
        bh_pieces_df,
        subs_all,
        messages,
        total_bh_weight_kg,
        total_plate_weight_kg,
        util_pct_overall,
        thickness_summary_df
    )

##########################################
# CONNECTION MR OPTIMIZER (reuse BH/CT leftover)
##########################################
def _pick_best_plate_option_for_connection(req_wt, thk_mm, catalog_df):
    """
    Find best (W,L) plate for given thickness for remaining connection weight.
    Minimizes %extra, then number of plates, then wider plate pref.
    """
    options=[]
    if catalog_df is not None:
        sub = catalog_df[catalog_df["Thickness(mm)"] == thk_mm]
        if len(sub)>0:
            for _,r in sub.iterrows():
                options.append((int(r["Width(mm)"]), int(r["Length(mm)"])))
    if not options:
        widths, lengths = standard_plate_options_for_thickness_conn(thk_mm)
        for w in widths:
            for L in lengths:
                options.append((w,L))

    best=None
    for (w,L) in options:
        pw_each = estimate_full_plate_weight_kg(thk_mm, w, L)
        plates_needed = math.ceil(req_wt / pw_each) if pw_each>0 else 0
        total_wt = plates_needed * pw_each
        extra = total_wt - req_wt
        pct_extra = (extra/req_wt) if req_wt>0 else 0

        cand = {
            "Thickness(mm)": thk_mm,
            "Req_Wt(kg)": req_wt,
            "Width(mm)": w,
            "Length(mm)": L,
            "Plate_Wt(kg_per_plate)": pw_each,
            "Plates_To_Order": plates_needed,
            "Ordered_Wt(kg)": total_wt,
            "Extra_Wt(kg)": extra,
            "Pct_Extra": pct_extra
        }

        if best is None:
            best = cand
        else:
            if (
                cand["Pct_Extra"] < best["Pct_Extra"] or
                (
                    abs(cand["Pct_Extra"] - best["Pct_Extra"]) < 1e-9 and
                    cand["Plates_To_Order"] < best["Plates_To_Order"]
                ) or
                (
                    abs(cand["Pct_Extra"] - best["Pct_Extra"]) < 1e-9 and
                    cand["Plates_To_Order"] == best["Plates_To_Order"] and
                    cand["Width(mm)"] > best["Width(mm)"]
                )
            ):
                best = cand
    return best


def optimize_connection_after_reuse(
    df_conn_raw: pd.DataFrame,
    leftover_dict: Dict[int,float],
    df_mill_for_conn: Optional[pd.DataFrame]
):
    """
    1. Clean/normalize Connection MR.
    2. Compute net balance per thickness after BH/CT leftover reuse.
    3. Suggest best plate sizes to buy for that net.
    Also build grouped order lines for final procurement merge.
    """
    df_conn = clean_connection_mr(df_conn_raw)
    grouped_req = group_connection_weight(df_conn)

    # leftover_dict: { thickness_mm: available_kg }
    avail_list=[]
    net_list=[]
    for _, row in grouped_req.iterrows():
        t = int(row["Thickness_mm"])
        orig_req = float(row["Req_Wt_Original(kg)"])
        avail = float(leftover_dict.get(t, 0.0))
        net = max(0.0, orig_req - avail)
        avail_list.append(avail)
        net_list.append(net)

    grouped_req["Available_From_BHCT(kg)"] = avail_list
    grouped_req["Net_Req_Wt(kg)"] = net_list

    # catalog_df for allowed connection sizes (mill sheet, if any)
    catalog_df = None
    if df_mill_for_conn is not None and len(df_mill_for_conn)>0:
        catalog_df = df_mill_for_conn.copy()
        catalog_df["Thickness(mm)"] = catalog_df["Thickness(mm)"].astype(int)
        catalog_df["Width(mm)"]     = catalog_df["Width(mm)"].astype(int)
        catalog_df["Length(mm)"]    = catalog_df["Length(mm)"].astype(int)

    # choose final ordering for each thickness
    best_rows=[]
    for _, row in grouped_req.iterrows():
        thk = int(row["Thickness_mm"])
        req_net = float(row["Net_Req_Wt(kg)"])
        if req_net <= 0:
            best_rows.append({
                "Thickness(mm)": thk,
                "Req_Wt(kg)": 0.0,
                "Width(mm)": None,
                "Length(mm)": None,
                "Plate_Wt(kg_per_plate)": 0.0,
                "Plates_To_Order": 0,
                "Ordered_Wt(kg)": 0.0,
                "Extra_Wt(kg)": 0.0,
                "Pct_Extra": 0.0,
            })
        else:
            best_rows.append(_pick_best_plate_option_for_connection(req_net, thk, catalog_df))

    disp_df = pd.DataFrame(best_rows)

    # pretty display version
    display_df = disp_df.copy()
    display_df["Chosen Plate (mm)"] = display_df.apply(
        lambda r: (
            f"{r['Width(mm)']} x {r['Length(mm)']}"
            if pd.notna(r["Width(mm)"]) and pd.notna(r["Length(mm)"])
            else ""
        ),
        axis=1
    )
    display_df = display_df[[
        "Thickness(mm)",
        "Req_Wt(kg)",
        "Chosen Plate (mm)",
        "Plate_Wt(kg_per_plate)",
        "Plates_To_Order",
        "Ordered_Wt(kg)",
        "Extra_Wt(kg)",
        "Pct_Extra",
    ]]
    display_df["Pct_Extra"] = display_df["Pct_Extra"] * 100.0

    # KPI summary for connection stage
    total_mr          = grouped_req["Req_Wt_Original(kg)"].sum()
    total_available   = grouped_req["Available_From_BHCT(kg)"].sum()
    total_net_after   = grouped_req["Net_Req_Wt(kg)"].sum()

    total_order       = disp_df["Ordered_Wt(kg)"].sum()
    extra_total       = total_order - total_net_after
    pct_extra_total   = (extra_total / total_net_after * 100) if total_net_after>0 else 0.0

    conn_summary = {
        "total_mr": total_mr,
        "total_available": total_available,
        "total_net_after": total_net_after,
        "total_order": total_order,
        "extra_total": extra_total,
        "pct_extra_total": pct_extra_total,
    }

    # also build a grouped "order lines" table for connections:
    # Thickness, Width, Length, Qty, Total Weight
    conn_order_lines=[]
    for _, r in disp_df.iterrows():
        if (
            r["Plates_To_Order"] > 0
            and pd.notna(r["Width(mm)"])
            and pd.notna(r["Length(mm)"])
        ):
            t  = int(r["Thickness(mm)"])
            W  = int(r["Width(mm)"])
            L  = int(r["Length(mm)"])
            q  = int(r["Plates_To_Order"])
            wt_each = estimate_full_plate_weight_kg(t, W, L)
            conn_order_lines.append({
                "Thickness (mm)": t,
                "Width (mm)": W,
                "Length (mm)": L,
                "Qty": q,
                "Total Weight (kg)": wt_each * q,
            })
    conn_order_df = pd.DataFrame(conn_order_lines)

    return grouped_req, display_df, conn_order_df, conn_summary

##########################################
# FINAL ORDER MERGE
##########################################
def collapse_bhct_order_lines(all_plates: List[StockPlate]) -> pd.DataFrame:
    """
    Group BH/CT final plates: Thickness, Width, Length, Qty, Total Weight
    """
    raw=[]
    for sp in all_plates:
        raw.append({
            "Thickness (mm)": sp.thickness_mm,
            "Width (mm)": sp.stock_width_mm,
            "Length (mm)": sp.stock_length_mm,
            "Qty": 1,
            "Total Weight (kg)": estimate_full_plate_weight_kg(
                sp.thickness_mm,
                sp.stock_width_mm,
                sp.stock_length_mm
            ),
        })
    if not raw:
        return pd.DataFrame(columns=["Thickness (mm)","Width (mm)","Length (mm)","Qty","Total Weight (kg)"])

    df = pd.DataFrame(raw)
    grouped = (
        df.groupby(["Thickness (mm)","Width (mm)","Length (mm)"], as_index=False)
          .agg({"Qty":"sum", "Total Weight (kg)":"sum"})
    )
    return grouped


def merge_final_orders(
    bhct_df: pd.DataFrame,
    conn_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge BH/CT and Connection plate needs to produce final consolidated order.
    Adds a 'Place used' column with categories:
      - 'BH/CT'          (only BH/CT plates)
      - 'Conn'           (only Connection plates)
      - 'BH/CT + Conn'   (same size used in both)
    """
    if bhct_df is None or len(bhct_df) == 0:
        left = pd.DataFrame(columns=["Thickness (mm)","Width (mm)","Length (mm)","Qty","Total Weight (kg)"])
    else:
        left = bhct_df.copy()

    if conn_df is None or len(conn_df) == 0:
        right = pd.DataFrame(columns=["Thickness (mm)","Width (mm)","Length (mm)","Qty","Total Weight (kg)"])
    else:
        right = conn_df.copy()

    merged = pd.merge(
        left,
        right,
        on=["Thickness (mm)", "Width (mm)", "Length (mm)"],
        how="outer",
        suffixes=("_BHCT", "_CONN")
    )

    # fill NaNs for arithmetic
    for c in ["Qty_BHCT", "Qty_CONN", "Total Weight (kg)_BHCT", "Total Weight (kg)_CONN"]:
        if c not in merged.columns:
            # if column missing (edge case), create it
            merged[c] = 0
        merged[c] = merged[c].fillna(0)

    # Determine “Place used”
    def _place_used(row):
        q_b = int(row.get("Qty_BHCT", 0))
        q_c = int(row.get("Qty_CONN", 0))
        if q_b > 0 and q_c > 0:
            return "BH/CT + Conn"
        elif q_b > 0:
            return "BH/CT"
        elif q_c > 0:
            return "Conn"
        return "—"

    merged["Place used"] = merged.apply(_place_used, axis=1)

    # Final totals
    merged["Final Qty"] = merged["Qty_BHCT"] + merged["Qty_CONN"]
    merged["Final Total Weight (kg)"] = merged["Total Weight (kg)_BHCT"] + merged["Total Weight (kg)_CONN"]

    # Output frame (now includes Place used)
    final_out = merged[[
        "Thickness (mm)",
        "Width (mm)",
        "Length (mm)",
        "Place used",
        "Final Qty",
        "Final Total Weight (kg)"
    ]].copy()

    final_out = final_out.sort_values(
        by=["Thickness (mm)", "Width (mm)", "Length (mm)"]
    ).reset_index(drop=True)

    return final_out


##########################################
# WRAPPER FUNCTION (used by app.py)
##########################################
def build_conn_usage_map_by_plate(
    plates: List[StockPlate],
    grouped_conn_df: pd.DataFrame,
    reuse_cap: float = 0.95,
    util_threshold: float = 0.80,
) -> Dict[str, float]:
    """
    Allocate the 'Available_From_BHCT(kg)' to specific plates, per thickness.
    We consume leftover starting from the lowest-utilization plates.

    Returns: { plate_id (str) : kg_allocated_to_connections }
    """
    # How much (kg) we can use per thickness as per connection stage
    need_by_t: Dict[int, float] = {}
    if grouped_conn_df is not None and len(grouped_conn_df) > 0:
        for _, r in grouped_conn_df.iterrows():
            t = int(r["Thickness_mm"])
            need_by_t[t] = float(r.get("Available_From_BHCT(kg)", 0.0))

    # Build list of candidate leftover by plate (only low-util plates per earlier logic)
    cand_list = []  # (util, thickness, plate_id, leftover95_kg)
    for sp in plates:
        try:
            util = float(sp.utilization())
        except Exception:
            util = 0.0
        if util < util_threshold:
            full_wt = estimate_full_plate_weight_kg(sp.thickness_mm, sp.stock_width_mm, sp.stock_length_mm)
            leftover_kg_95 = max(0.0, (1.0 - util) * full_wt * reuse_cap)
            if leftover_kg_95 > 0:
                cand_list.append((util, int(sp.thickness_mm), str(sp.plate_id), leftover_kg_95))

    # Consume per thickness: lowest-util plates first
    cand_list.sort(key=lambda x: x[0])  # ascending utilization
    conn_usage_map: Dict[str, float] = {}
    for util, t, pid, avail_kg in cand_list:
        rem = need_by_t.get(t, 0.0)
        if rem <= 1e-9:
            continue
        take = min(avail_kg, rem)
        if take > 0:
            conn_usage_map[pid] = conn_usage_map.get(pid, 0.0) + take
            need_by_t[t] = rem - take

    return conn_usage_map


def run_full_optimization(
    df_bhct: pd.DataFrame,
    df_conn: pd.DataFrame,
    df_stock: Optional[pd.DataFrame],
    mill_sizes: Optional[Dict[int, List[Tuple[int, int]]]],
    kerf: int,
    trim: int,
    min_util_pct: float,
    priority: str,
) -> Dict:
    """
    Top-level orchestrator used by Streamlit app.
    Combines BH/CT optimizer + Connection MR reuse + Final order.
    """

    # --- Parse BH/CT rows ---
    bh_rows: List[BHRow] = []
    if df_bhct is not None and len(df_bhct) > 0:
        for _, r in df_bhct.iterrows():
            try:
                br = BHRow(
                    profile=str(r["PROFILE"]).strip(),
                    length_mm=int(r["LENGTH (mm)"]),
                    unit_weight_kg=float(r["UNIT WEIGHT(Kg)"]),
                    qty=int(r["QTY."]),
                    total_weight_kg=float(r["TOTAL WEIGHT(Kg)"]),
                )
                br.parse()
                bh_rows.append(br)
            except Exception:
                continue

    # --- Load stock list ---
    inventory: List[InvPlate] = []
    if df_stock is not None and len(df_stock) > 0:
        from data_io import load_stock_inventory
        inventory = load_stock_inventory(df_stock)

    # --- Run BH/CT nesting plan ---
    (
        plates,
        order_df,
        procurement_df,
        bhct_pieces_df,
        subs_all,
        messages,
        total_bh_weight_kg,
        total_plate_weight_kg,
        util_pct_overall,
        thickness_summary_df,
    ) = master_plan_bhct(
        bh_rows,
        inventory,
        kerf,
        trim,
        min_util_pct,
        priority,
        mill_sizes,
    )

    # --- Compute leftover (<80% utilized plates) ---
    leftover_dict = leftover_weight_dict_from_low_util(plates, util_threshold=0.80)

    # --- Optimize Connection MR (reusing BH/CT leftovers) ---
    grouped_conn_df, conn_plan_df, conn_order_df, conn_summary = optimize_connection_after_reuse(
        df_conn_raw=df_conn,
        leftover_dict=leftover_dict,
        df_mill_for_conn=None if mill_sizes is None else pd.DataFrame(
            [(t, W, L) for t, lst in mill_sizes.items() for (W, L) in lst],
            columns=["Thickness(mm)", "Width(mm)", "Length(mm)"]
        ),
    )

    # NEW: per-plate allocation of leftover to connections (for PDF drawing headers)
    conn_usage_map = build_conn_usage_map_by_plate(
        plates=plates,
        grouped_conn_df=grouped_conn_df,
        reuse_cap=0.95,        # your rule: use max 95% of leftover
        util_threshold=0.80,   # only <80% util plates are considered leftover
    )

    # --- Merge Final Orders ---
    bhct_order_df = collapse_bhct_order_lines(plates)
    final_order_df = merge_final_orders(bhct_order_df, conn_order_df)

    # --- KPI dicts ---
    bhct_kpis = {
        "total_bh_weight_kg": total_bh_weight_kg,
        "total_plate_weight_kg": total_plate_weight_kg,
        "utilization_pct_overall": util_pct_overall,
        "tiers_covered": sorted(df_bhct["Tier"].dropna().unique().tolist()),
        "fabs_covered": sorted(df_bhct["Fab"].dropna().unique().tolist()),
    }

    # --- Save Excel ---
    out_xls = io.BytesIO()
    with pd.ExcelWriter(out_xls, engine="xlsxwriter") as writer:
        order_df.to_excel(writer, sheet_name="PlateOrders", index=False)
        procurement_df.to_excel(writer, sheet_name="ProcurementSummary", index=False)
        bhct_pieces_df.to_excel(writer, sheet_name="BHPieces", index=False)
        thickness_summary_df.to_excel(writer, sheet_name="ThicknessSummary", index=False)
        grouped_conn_df.to_excel(writer, sheet_name="ConnGrouped", index=False)
        conn_plan_df.to_excel(writer, sheet_name="ConnPlan", index=False)
        final_order_df.to_excel(writer, sheet_name="FinalOrder", index=False)
    out_xls.seek(0)
    xls_bytes = out_xls.read()

    return {
        "plates": plates,
        "subs_all": subs_all,
        "order_df": order_df,
        "procurement_df": procurement_df,
        "bhct_pieces_df": bhct_pieces_df,
        "thickness_summary_df": thickness_summary_df,
        "bhct_kpis": bhct_kpis,
        "conn_summary": conn_summary,
        "grouped_conn_df": grouped_conn_df,
        "conn_plan_df": conn_plan_df,
        "final_order_df": final_order_df,
        "messages": messages,
        "xls_bytes": xls_bytes,
        "conn_usage_map": conn_usage_map,
    }
