# plate_rules.py
import math, re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, DefaultDict
from collections import defaultdict

STEEL_DENSITY_KG_PER_M3 = 7850.0  # kg/m³
DENSITY = 7850.0                  # same constant for connection calc

# BH/CT profile pattern: BH990X640X28X45 or CT475X600X20X32
SEC_PATTERN = re.compile(
    r"(BH|CT)\s*(\d+)\s*[Xx]\s*(\d+)\s*[Xx]\s*(\d+)\s*[Xx]\s*(\d+)",
    re.IGNORECASE
)

# Soft width limit (we prefer ≤ this; > is last-resort in optimizer)
SOFT_WIDTH_LIMIT_MM = 2500

# -------------------- Dataclasses --------------------
@dataclass
class BHRow:
    profile: str
    length_mm: int
    unit_weight_kg: float
    qty: int
    total_weight_kg: float

    sec_type: str = "BH"  # "BH" or "CT"
    H: int = 0
    B: int = 0
    tw: int = 0
    tf: int = 0

    def parse(self):
        """
        Parse BH/CT profile string:
        BH990X640X28X45 -> sec_type=BH, H=990, B=640, tw=28, tf=45
        CT475X600X20X32 -> sec_type=CT, ...
        """
        m = SEC_PATTERN.match(self.profile.strip())
        if not m:
            raise ValueError(f"Invalid profile: {self.profile}")
        self.sec_type = m.group(1).upper()
        self.H  = int(m.group(2))
        self.B  = int(m.group(3))
        self.tw = int(m.group(4))
        self.tf = int(m.group(5))

    def flange_width(self) -> int:
        return self.B

    def web_width(self) -> int:
        # CT = single flange T-profile (one flange), BH = I-profile (two flanges)
        if self.sec_type == "CT":
            return self.H - self.tf
        else:
            return self.H - 2*self.tf


@dataclass
class PlatePiece:
    kind: str            # 'flange' or 'web'
    thickness_mm: int
    width_mm: int
    length_mm: int
    qty: int
    bh_profile: str      # BH... or CT...


@dataclass
class SubPiece:
    parent_id:int
    index:int              # 1 or 2 when length-spliced; 91/92 if width-split halves
    total_len_mm:int
    length_mm:int
    width_mm:int
    thickness_mm:int
    kind:str               # 'flange'/'web'
    bh_profile:str
    splice_joint_here:bool
    joint_pos_mm: Optional[int]   # for length-wise splice joint only (None for width split)


@dataclass
class Placement:
    x:int; y:int; w:int; h:int
    label:str; annotate:str
    parent_id:int; sub_index:int; bh_profile:str; kind:str


@dataclass
class StockPlate:
    plate_id:str
    thickness_mm:int
    stock_width_mm:int
    stock_length_mm:int
    placements:List[Placement]=field(default_factory=list)
    trim_mm:int=0; kerf_mm:int=0
    source:str="standard"  # "inventory" or "standard"

    def utilization(self)->float:
        """
        Nesting utilization based on bounding boxes inside usable window.
        """
        usable_w = self.stock_width_mm - 2*self.trim_mm
        usable_l = self.stock_length_mm - 2*self.trim_mm
        if usable_w<=0 or usable_l<=0:
            return 0.0
        used_area = 0
        x_max = self.trim_mm + usable_l
        y_max = self.trim_mm + usable_w
        for p in self.placements:
            x1 = min(p.x + p.w, x_max)
            y1 = min(p.y + p.h, y_max)
            if x1>p.x and y1>p.y:
                used_area += (x1 - p.x) * (y1 - p.y)
        return used_area / (usable_w * usable_l)


@dataclass
class InvPlate:
    t:int; w:int; l:int; qty:int; weight:float=0.0


# -------------------- Plate size menus --------------------
def get_allowed_plate_sizes_for_thickness(
    t: int,
    mill_sizes: Optional[Dict[int, List[Tuple[int, int]]]]
) -> List[Tuple[int, int]]:
    """
    Menu of width/length for BH/CT nesting.
    Priority:
    1. Mill offer for that thickness (if uploaded)
    2. Built-in defaults
    """
    if mill_sizes and t in mill_sizes:
        return mill_sizes[t]

    if t <= 45:
        # ✅ Custom width list up to 3550 mm for thinner plates
        widths  = [1500, 2000, 2200, 2500, 2800, 3000, 3100, 3200, 3300, 3400, 3500, 3550]
        lengths = [10000, 10500, 11000, 11500, 12000, 12500, 13000]
        return [(W, L) for W in widths for L in lengths]
    else:
        # ✅ Keep thicker plates rule (up to 3600 mm, 6–13 m)
        widths  = [w for w in range(1200, 3601, 50)]
        lengths = list(range(6000, 13001, 50))
        return [(W, L) for W in widths for L in lengths]




def standard_plate_options_for_thickness_conn(t:int) -> Tuple[List[int], List[int]]:
    """
    Menu used by connection MR optimizer.
    """
    if t <= 45:
        widths  = [2000, 2200, 2500]
        lengths = [10000, 10500, 11000, 11500, 12000]
    else:
        widths  = [
            1000,1100,1200,1300,1400,1450,1500,1550,1600,
            1650,1700,1750,1800,1900,2000,2100,2200,2300,
            2400,2500,  # prefer up to 2500 for connections
        ]
        lengths = list(range(5000, 12000, 50))
    return widths, lengths


# -------------------- Capacity / splice helpers --------------------
def usable_cap_for_thickness(
    t:int,
    trim:int,
    kerf:int,
    mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]=None
)->int:
    """
    Maximum usable one-piece length for a given thickness,
    based on the best length available for that thickness.
    """
    opts = get_allowed_plate_sizes_for_thickness(t, mill_sizes)
    if not opts:
        return 0
    max_L = max(L for (_,L) in opts)
    return max_L - 2*trim - kerf


def plan_staggered_splits_for_bh(
    length_mm:int,
    flange_t:int,
    web_t:int,
    trim_mm:int,
    kerf:int,
    mill_sizes:Optional[Dict[int,List[Tuple[int,int]]]],
    min_stagger_mm:int=300
)->Tuple[Optional[int], Optional[int]]:
    """
    Decide where to splice flange and web, and keep them staggered ≥ min_stagger_mm.
    Returns (flange_splice_pos_mm, web_splice_pos_mm).
    None means full length in single piece.
    """
    cap_f = usable_cap_for_thickness(flange_t, trim_mm, kerf, mill_sizes)
    cap_w = usable_cap_for_thickness(web_t,    trim_mm, kerf, mill_sizes)

    need_f, need_w = length_mm > cap_f, length_mm > cap_w
    if not need_f and not need_w:
        return None, None

    lower = math.floor(length_mm/3)
    upper = math.ceil(2*length_mm/3)

    def choose_pos(cap:int)->Optional[int]:
        # try nice ~1/3–2/3 split
        for a in range(lower, upper+1):
            b = length_mm - a
            if 0 < a <= cap and 0 < b <= cap:
                return a
        # fallback
        a = min(cap, max(lower, length_mm - cap))
        b = length_mm - a
        if lower<=a<=upper and 0<a<=cap and 0<b<=cap:
            return a
        return None

    pos_f = choose_pos(cap_f) if need_f else None
    pos_w = choose_pos(cap_w) if need_w else None

    if need_f and need_w and pos_f is not None and pos_w is not None:
        if abs(pos_f - pos_w) < min_stagger_mm:
            # Try moving web splice first
            def try_shift(base:int, fixed:int, cap:int)->Optional[int]:
                for delta in range(min_stagger_mm, (upper-lower)+1):
                    for cand in (base - delta, base + delta):
                        if lower<=cand<=upper and abs(cand - fixed) >= min_stagger_mm:
                            a=cand; b=length_mm-a
                            if 0<a<=cap and 0<b<=cap:
                                return cand
                return None

            shifted_web = try_shift(pos_w, pos_f, cap_w)
            if shifted_web is not None:
                return pos_f, shifted_web

            shifted_f   = try_shift(pos_f, pos_w, cap_f)
            if shifted_f is not None:
                return shifted_f, pos_w

    return pos_f, pos_w


# -------------------- Conditional width-split (WEB only) --------------------
def _max_usable_width_for_t(
    t:int, trim:int, mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
) -> int:
    """
    Return the maximum usable *piece width* for any available plate at thickness t,
    i.e., max(W) - 2*trim. If no options, return 0.
    """
    opts = get_allowed_plate_sizes_for_thickness(t, mill_sizes)
    if not opts:
        return 0
    max_W = max(W for (W, _) in opts)
    return max_W - 2*trim


def conditional_split_wide_webs(
    subs: List['SubPiece'],
    trim_mm: int,
    kerf:int,
    mill_sizes: Optional[Dict[int, List[Tuple[int,int]]]],
) -> List['SubPiece']:
    """
    Apply ONE transverse (width-wise) split to WEB subpieces ONLY IF
    even the *widest available* plate for that thickness (from mill_sizes or defaults)
    cannot accommodate the subpiece width after trim.

    - If a wider plate (even >2500) exists that can fit -> NO split (optimizer
      will treat >2500 as last resort).
    - If nothing can fit it (even the max width available), split into two halves
      (indices 91/92), mark BOTH with splice_joint_here=True, joint_pos_mm=None.
    """
    out: List[SubPiece] = []
    for s in subs:
        if (s.kind or "").lower() != "web":
            out.append(s)
            continue

        usable_max_w = _max_usable_width_for_t(s.thickness_mm, trim_mm, mill_sizes)
        if usable_max_w <= 0 or s.width_mm <= usable_max_w:
            # Either no menu (edge case) or it fits on some plate width -> no width split.
            out.append(s)
            continue

        # Not fit on ANY available width -> split once across width.
        total_w = s.width_mm
        w1 = total_w // 2
        w2 = total_w - w1
        w1 = max(1, int(w1))
        w2 = max(1, int(w2))

        a = SubPiece(
            parent_id = s.parent_id,
            index = 91,
            total_len_mm = s.total_len_mm,
            length_mm = s.length_mm,
            width_mm = w1,
            thickness_mm = s.thickness_mm,
            kind = s.kind,
            bh_profile = s.bh_profile,
            splice_joint_here = True,
            joint_pos_mm = None
        )
        b = SubPiece(
            parent_id = s.parent_id,
            index = 92,
            total_len_mm = s.total_len_mm,
            length_mm = s.length_mm,
            width_mm = w2,
            thickness_mm = s.thickness_mm,
            kind = s.kind,
            bh_profile = s.bh_profile,
            splice_joint_here = True,
            joint_pos_mm = None
        )
        out.extend([a, b])

    return out
