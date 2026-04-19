
# Intelligent Mould Scheduling Dashboard

# Streamlit application for precast panel production scheduling.

# local usage:
    # pip install streamlit pandas numpy matplotlib plotly ortools scikit-learn
    # streamlit run app.py


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import itertools
from collections import defaultdict
import time
import io
import warnings
warnings.filterwarnings("ignore")


# Page configuraton

st.set_page_config(
    page_title="PBU Mould Scheduler",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS

st.markdown("""
<style>
    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .kpi-card h2 { margin: 0; font-size: 2rem; font-weight: 700; }
    .kpi-card p { margin: 0.2rem 0 0; font-size: 0.85rem; opacity: 0.85; }
    .kpi-improvement {
        background: linear-gradient(135deg, #1a7a4c 0%, #27ae60 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .kpi-improvement h2 { margin: 0; font-size: 2rem; font-weight: 700; }
    .kpi-improvement p { margin: 0.2rem 0 0; font-size: 0.85rem; opacity: 0.85; }
    /* Section headers */
    .section-header {
        border-left: 4px solid #2d5986;
        padding-left: 12px;
        margin: 1.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Constants

CAST_TIME_H = 6.0
T_LEN_H = 3.5
T_CEIL_H = 3.5
T_RECESS_H = 3.0
T_OPEN_H = 2.0

SCALE = 2
CAST_T = int(CAST_TIME_H * SCALE)
T_LEN = int(T_LEN_H * SCALE)
T_CEIL = int(T_CEIL_H * SCALE)
T_RECESS = int(T_RECESS_H * SCALE)
T_OPEN = int(T_OPEN_H * SCALE)

SPACING_MM = 300
CORNERS = ["NW", "NE", "SW", "SE"]
DIAGONAL_PAIRS = [("NW", "SE"), ("NE", "SW")]
QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]


# Panel handednes model (V1/V2)


CORNER_MAP_V1 = {
    "C02": {"SW": "ba", "SE": "ab", "NW": "ba", "NE": "ab"},
    "C03": {"SW": "ba", "SE": "ab", "NW": "ab", "NE": "ba"},
    "C04": {"SW": "ab", "SE": "ba", "NW": "ba", "NE": "ab"},
}

CORNER_MAP_V2 = {}
for _code, _mapping in CORNER_MAP_V1.items():
    CORNER_MAP_V2[_code] = {
        "SW": _mapping["SE"], "SE": _mapping["SW"],
        "NW": _mapping["NE"], "NE": _mapping["NW"],
    }

QUADRANT_TO_CORNER = {"Q1": "NW", "Q2": "NE", "Q3": "SW", "Q4": "SE"}

V2_PBU_TYPES = {
    "1a", "1b", "1c", "1d", "1e",
    "2b-m",
    "3a-m", "3a-m(a)",
    "3b-m", "3b-m(a)",
    "4b-m", "4b-m(a)",
    "5a", "5a(a)",
}


def get_pbu_version(panel):
    pbu_type = str(panel.get("pbu_type", "")).strip().lower()
    for v2_type in V2_PBU_TYPES:
        if pbu_type == v2_type.lower():
            return "V2"
    return "V1"


def get_hv_for_corner(panel, corner):
    a, b = int(panel["a_len"]), int(panel["b_len"])
    code = panel.get("panel_code", "")
    version = get_pbu_version(panel)
    mapping = (CORNER_MAP_V2 if version == "V2" else CORNER_MAP_V1).get(code)
    if mapping is None:
        return (a, b)
    return (b, a) if mapping[corner] == "ba" else (a, b)


def get_hv_for_quadrant(panel, quadrant):
    return get_hv_for_corner(panel, QUADRANT_TO_CORNER[quadrant])



# Core placement functions

def find_placement_square(combo, mould):
    n = len(combo)
    if n == 0 or n > 4:
        return None
    L = int(mould["max_length"])
    for p in combo:
        if max(int(p["a_len"]), int(p["b_len"])) > L:
            return None
    for chosen in itertools.combinations(CORNERS, n):
        for perm in itertools.permutations(combo):
            placed = dict(zip(chosen, perm))
            # Get fixed (h, v) per panel-corner pair
            hv = {}
            for c in chosen:
                hv[c] = get_hv_for_corner(placed[c], c)
            # Diagonal ceiling constraint (binary)
            diag_ok = True
            for c1, c2 in DIAGONAL_PAIRS:
                if c1 in placed and c2 in placed:
                    if int(placed[c1]["has_ceiling"]) and int(placed[c2]["has_ceiling"]):
                        diag_ok = False
                        break
            if not diag_ok:
                continue

            # Panel adjacency spacing
            ok = True
            if ok and "NW" in hv and "NE" in hv:
                if hv["NW"][0] + hv["NE"][0] + SPACING_MM > L:
                    ok = False
            if ok and "SW" in hv and "SE" in hv:
                if hv["SW"][0] + hv["SE"][0] + SPACING_MM > L:
                    ok = False
            if ok and "NW" in hv and "SW" in hv:
                if hv["NW"][1] + hv["SW"][1] + SPACING_MM > L:
                    ok = False
            if ok and "NE" in hv and "SE" in hv:
                if hv["NE"][1] + hv["SE"][1] + SPACING_MM > L:
                    ok = False
            if ok:
                result = {}
                for c in chosen:
                    p = placed[c]
                    horiz_leg, vert_leg = hv[c]
                    result[c] = {
                        "panel": p, "horiz": horiz_leg, "vert": vert_leg,
                        "a_len": int(p["a_len"]), "b_len": int(p["b_len"]),
                    }
                return result
    return None


def find_placement_cross(combo, mould):
    n = len(combo)
    if n == 0 or n > 4:
        return None
    q = int(mould["quadrant_limit"])
    for p in combo:
        if int(p["a_len"]) > q or int(p["b_len"]) > q:
            return None
    for chosen in itertools.combinations(QUADRANTS, n):
        for perm in itertools.permutations(combo):
            assigned = dict(zip(chosen, perm))
            ok = True
            placements = {}
            for qname in chosen:
                p = assigned[qname]
                h, v = get_hv_for_quadrant(p, qname)
                if h > q or v > q:
                    ok = False
                    break
                placements[qname] = {
                    "panel": p, "horiz": h, "vert": v,
                    "a_len": int(p["a_len"]), "b_len": int(p["b_len"]),
                }
            if ok:
                return placements
    return None


def find_placement(combo, mould):
    if mould["mould_type"] == "square":
        return find_placement_square(combo, mould)
    if mould["mould_type"] == "cross":
        return find_placement_cross(combo, mould)
    return None


def is_feasible(combo, mould):
    return find_placement(combo, mould) is not None


def recipe_for_cycle(panels_in_cycle):
    return (
        max(int(p["b_len"]) for p in panels_in_cycle),
        max(int(p["has_ceiling"]) for p in panels_in_cycle),
        max(int(p["has_recess"]) for p in panels_in_cycle),
        max(int(p["num_openings"]) for p in panels_in_cycle),
    )


def setup_time_scaled(prev_recipe, new_recipe):
    if prev_recipe is None:
        return 0
    t = 0
    if prev_recipe[0] != new_recipe[0]: t += T_LEN
    if prev_recipe[1] != new_recipe[1]: t += T_CEIL
    if prev_recipe[2] != new_recipe[2]: t += T_RECESS
    if prev_recipe[3] != new_recipe[3]: t += T_OPEN
    return t


def panel_fits_cross(panel, qlimit):
    return int(panel["a_len"]) <= qlimit and int(panel["b_len"]) <= qlimit


def build_cycles_balanced(panels_records, moulds_records):
    """Balanced batching strategy — same as your notebook."""
    cross_moulds = [m for m in moulds_records if m["mould_type"] == "cross"]
    square_moulds = sorted(
        [m for m in moulds_records if m["mould_type"] == "square"],
        key=lambda m: int(m["max_length"]), reverse=True
    )
    has_cross = len(cross_moulds) > 0
    cross = cross_moulds[0] if has_cross else None
    qlimit = int(cross["quadrant_limit"]) if cross else 0

    if has_cross:
        big_panels = [p for p in panels_records if not panel_fits_cross(p, qlimit)]
        small_panels = [p for p in panels_records if panel_fits_cross(p, qlimit)]
    else:
        big_panels = list(panels_records)
        small_panels = []

    big_panels.sort(key=lambda p: (int(p["b_len"]), int(p["has_ceiling"])), reverse=True)
    small_panels.sort(key=lambda p: int(p["b_len"]), reverse=True)

    all_mould_ids_local = [m["mould_id"] for m in moulds_records]
    moulds_by_id = {m["mould_id"]: m for m in moulds_records}

    cycles = []
    open_cycle_ids = []
    cid = 0

    def get_cycle_counts():
        counts = {mid: 0 for mid in all_mould_ids_local}
        for c in cycles:
            counts[c["mould_id"]] += 1
        return counts

    # STEP 1: Place big panels on square moulds
    sq_ptr = 0
    for panel in big_panels:
        placed = False
        for idx in sorted(open_cycle_ids,
                          key=lambda i: len(cycles[i]["panels"]), reverse=True):
            cyc = cycles[idx]
            if moulds_by_id[cyc["mould_id"]]["mould_type"] != "square":
                continue
            m = moulds_by_id[cyc["mould_id"]]
            test = cyc["panels"] + [panel]
            if len(test) <= 4 and is_feasible(test, m):
                cyc["panels"].append(panel)
                cyc["recipe"] = recipe_for_cycle(cyc["panels"])
                if len(cyc["panels"]) == 4:
                    open_cycle_ids.remove(idx)
                placed = True
                break
        if not placed:
            for attempt in range(len(square_moulds)):
                m = square_moulds[(sq_ptr + attempt) % len(square_moulds)]
                if is_feasible([panel], m):
                    new_cyc = {
                        "cycle_id": cid, "mould_id": m["mould_id"],
                        "panels": [panel], "recipe": recipe_for_cycle([panel]),
                    }
                    cycles.append(new_cyc)
                    open_cycle_ids.append(cid)
                    cid += 1
                    sq_ptr = (sq_ptr + attempt + 1) % len(square_moulds)
                    placed = True
                    break

    # STEP 2: Fill open square cycles with small panels
    remaining_small = []
    for panel in small_panels:
        placed = False
        for idx in sorted(open_cycle_ids,
                          key=lambda i: len(cycles[i]["panels"]), reverse=True):
            cyc = cycles[idx]
            m = moulds_by_id[cyc["mould_id"]]
            test = cyc["panels"] + [panel]
            if len(test) <= 4 and is_feasible(test, m):
                cyc["panels"].append(panel)
                cyc["recipe"] = recipe_for_cycle(cyc["panels"])
                if len(cyc["panels"]) == 4:
                    open_cycle_ids.remove(idx)
                placed = True
                break
        if not placed:
            remaining_small.append(panel)

    # STEP 3: Distribute remaining across all moulds
    for panel in remaining_small:
        placed = False
        candidates = sorted(open_cycle_ids,
                            key=lambda i: (get_cycle_counts()[cycles[i]["mould_id"]],
                                           -len(cycles[i]["panels"])))
        for idx in candidates:
            cyc = cycles[idx]
            m = moulds_by_id[cyc["mould_id"]]
            test = cyc["panels"] + [panel]
            if len(test) <= 4 and is_feasible(test, m):
                cyc["panels"].append(panel)
                cyc["recipe"] = recipe_for_cycle(cyc["panels"])
                if len(cyc["panels"]) == 4:
                    open_cycle_ids.remove(idx)
                placed = True
                break
        if not placed:
            counts = get_cycle_counts()
            sorted_moulds = sorted(moulds_records, key=lambda m: counts[m["mould_id"]])
            for m in sorted_moulds:
                if is_feasible([panel], m):
                    new_cyc = {
                        "cycle_id": cid, "mould_id": m["mould_id"],
                        "panels": [panel], "recipe": recipe_for_cycle([panel]),
                    }
                    cycles.append(new_cyc)
                    open_cycle_ids.append(cid)
                    cid += 1
                    placed = True
                    break

    # STEP 4: Consolidate underfilled cycles
    for mid in all_mould_ids_local:
        m = moulds_by_id[mid]
        changed_merge = True
        while changed_merge:
            changed_merge = False
            mid_cycles = [c for c in cycles if c["mould_id"] == mid]
            mid_cycles.sort(key=lambda c: len(c["panels"]))
            for i in range(len(mid_cycles)):
                for j in range(i + 1, len(mid_cycles)):
                    combined = mid_cycles[i]["panels"] + mid_cycles[j]["panels"]
                    if len(combined) <= 4 and is_feasible(combined, m):
                        mid_cycles[i]["panels"] = combined
                        mid_cycles[i]["recipe"] = recipe_for_cycle(combined)
                        cycles.remove(mid_cycles[j])
                        changed_merge = True
                        break
                if changed_merge:
                    break
    open_cycle_ids = [i for i, c in enumerate(cycles) if len(c["panels"]) < 4]

    # STEP 5: Rebalance
    for _ in range(200):
        counts = get_cycle_counts()
        max_mid = max(counts, key=counts.get)
        min_mid = min(counts, key=counts.get)
        if counts[max_mid] - counts[min_mid] < 2:
            break
        target_m = moulds_by_id[min_mid]
        moved = False
        for c in sorted([c for c in cycles if c["mould_id"] == max_mid],
                        key=lambda c: len(c["panels"])):
            if is_feasible(c["panels"], target_m):
                c["mould_id"] = min_mid
                moved = True
                break
        if not moved:
            break

    for i, c in enumerate(cycles):
        c["cycle_id"] = i
    return cycles


def optimize_sequences_cpsat(cycles, mould_ids, cast_time_scaled=CAST_T):
    # CP-SAT sequencing — same as your notebook.
    from ortools.sat.python import cp_model

    cycles_by_mould = defaultdict(list)
    for c in cycles:
        cycles_by_mould[c["mould_id"]].append(c)
    for mid in mould_ids:
        _ = cycles_by_mould[mid]

    model = cp_model.CpModel()
    mould_total_time = {}
    arcs_by_mould = {}

    for mid in mould_ids:
        mc = cycles_by_mould[mid]
        k = len(mc)
        if k == 0:
            mould_total_time[mid] = 0
            arcs_by_mould[mid] = []
            continue
        if k == 1:
            mould_total_time[mid] = cast_time_scaled
            arcs_by_mould[mid] = []
            continue

        arcs = []
        setup_cost = {}
        for i in range(k + 1):
            for j in range(k + 1):
                if i == j:
                    continue
                lit = model.NewBoolVar(f"arc_{mid}_{i}_{j}")
                arcs.append((i, j, lit))
                if i == 0 or j == 0:
                    setup_cost[(i, j)] = 0
                else:
                    setup_cost[(i, j)] = setup_time_scaled(
                        mc[i - 1]["recipe"], mc[j - 1]["recipe"]
                    )
        model.AddCircuit(arcs)

        setup_terms = []
        for (i, j, lit) in arcs:
            if i > 0 and j > 0 and setup_cost[(i, j)] > 0:
                setup_terms.append(setup_cost[(i, j)] * lit)

        total_setup = model.NewIntVar(0, 1_000_000, f"setup_{mid}")
        if setup_terms:
            model.Add(total_setup == sum(setup_terms))
        else:
            model.Add(total_setup == 0)

        total_time = model.NewIntVar(0, 1_000_000, f"time_{mid}")
        model.Add(total_time == k * cast_time_scaled + total_setup)
        mould_total_time[mid] = total_time
        arcs_by_mould[mid] = arcs

    makespan = model.NewIntVar(0, 1_000_000, "makespan")
    for mid in mould_ids:
        tt = mould_total_time[mid]
        model.Add(makespan >= tt)

    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed: {solver.StatusName(status)}")

    sequences = {}
    times = {}
    for mid in mould_ids:
        mc = cycles_by_mould[mid]
        k = len(mc)
        if k == 0:
            sequences[mid] = []
            times[mid] = 0
            continue
        if k == 1:
            sequences[mid] = list(mc)
            times[mid] = cast_time_scaled
            continue
        arcs = arcs_by_mould[mid]
        next_node = {}
        for (i, j, lit) in arcs:
            if solver.Value(lit) == 1:
                next_node[i] = j
        order = []
        cur = 0
        for _ in range(k):
            nxt = next_node.get(cur)
            if nxt is None or nxt == 0:
                break
            order.append(nxt - 1)
            cur = nxt
        sequences[mid] = [mc[i] for i in order]
        tt = mould_total_time[mid]
        times[mid] = solver.Value(tt) if not isinstance(tt, int) else tt

    return sequences, solver.Value(makespan), times


def build_timeline(sequences_by_mould, all_mould_ids):
    rows = []
    for mid in all_mould_ids:
        seq = sequences_by_mould.get(mid, [])
        t = 0
        prev_recipe = None
        for step, cyc in enumerate(seq, start=1):
            st = setup_time_scaled(prev_recipe, cyc["recipe"])
            start = t
            end = start + st + CAST_T
            rows.append({
                "mould_id": mid,
                "step": step,
                "cycle_id": cyc["cycle_id"],
                "n_panels": len(cyc["panels"]),
                "panel_ids": ", ".join(p["panel_id"] for p in cyc["panels"]),
                "recipe": str(cyc["recipe"]),
                "setup_h": st / SCALE,
                "cast_h": CAST_T / SCALE,
                "start_h": start / SCALE,
                "end_h": end / SCALE,
            })
            t = end
            prev_recipe = cyc["recipe"]
    return pd.DataFrame(rows)



# Plotly Gantt chart


def create_gantt_chart(schedule_df, all_mould_ids, title="Production Schedule"):
    colors = {
        "M1a": "#1f77b4", "M1b": "#2ca02c", "M1c": "#ff7f0e",
        "M1d": "#d62728", "M2": "#9467bd", "M3": "#8c564b",
        "C1": "#e377c2", "C2": "#7f7f7f", "C3": "#bcbd22",
        "M1e": "#17becf", "M1f": "#aec7e8",
    }
    fig = go.Figure()

    for mid in all_mould_ids:
        mdf = schedule_df[schedule_df["mould_id"] == mid]
        if mdf.empty:
            continue
        for _, r in mdf.iterrows():
            color = colors.get(mid, "#888888")
            # Setup bar
            if r["setup_h"] > 0:
                fig.add_trace(go.Bar(
                    y=[mid], x=[r["setup_h"]], base=[r["start_h"]],
                    orientation="h", name="Setup",
                    marker=dict(color=color, opacity=0.3,
                                pattern=dict(shape="/")),
                    hovertemplate=(
                        f"<b>{mid} — Step {int(r['step'])}</b><br>"
                        f"Setup: {r['setup_h']:.1f}h<br>"
                        f"Start: {r['start_h']:.1f}h<extra></extra>"
                    ),
                    showlegend=False,
                ))
            # Casting bar
            cast_start = r["start_h"] + r["setup_h"]
            fig.add_trace(go.Bar(
                y=[mid], x=[r["cast_h"]], base=[cast_start],
                orientation="h", name="Casting",
                marker=dict(color=color, opacity=0.85,
                            line=dict(color="black", width=0.5)),
                text=f"S{int(r['step'])} ({r['n_panels']}p)",
                textposition="inside",
                textfont=dict(size=10, color="white"),
                hovertemplate=(
                    f"<b>{mid} — Step {int(r['step'])}</b><br>"
                    f"Panels: {r['panel_ids']}<br>"
                    f"Cast: {cast_start:.1f}–{r['end_h']:.1f}h<br>"
                    f"Recipe: {r['recipe']}<extra></extra>"
                ),
                showlegend=False,
            ))

    # Makespan line
    if not schedule_df.empty:
        makespan = schedule_df["end_h"].max()
        fig.add_vline(x=makespan, line_dash="dash", line_color="red",
                      annotation_text=f"Makespan: {makespan:.1f}h",
                      annotation_position="top right",
                      annotation_font=dict(color="red", size=12))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Time (hours)",
        yaxis=dict(categoryorder="array",
                   categoryarray=list(reversed(all_mould_ids))),
        barmode="overlay",
        height=max(300, 80 * len(all_mould_ids)),
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee", zeroline=True),
        yaxis_title="Mould",
    )
    return fig



# Mould layout visualisation


def draw_mould_layout_plotly(placement, mould, step_label):
    """Draw L-shaped panels in a mould using Plotly shapes."""
    fig = go.Figure()

    if mould["mould_type"] == "square":
        L = int(mould["max_length"])
        # Mould boundary
        fig.add_shape(type="rect", x0=0, y0=0, x1=L, y1=L,
              line=dict(color="black", width=3), fillcolor="rgba(245,245,245,0.3)")

        corner_colors = {"NW": "#4C72B0", "NE": "#DD8452",
                         "SW": "#55A868", "SE": "#C44E52"}
        
        corner_cfg = {
            "NW": {"origin": (0, L), "h_dir": +1, "v_dir": -1},
            "NE": {"origin": (L, L), "h_dir": -1, "v_dir": -1},
            "SW": {"origin": (0, 0), "h_dir": +1, "v_dir": +1},
            "SE": {"origin": (L, 0), "h_dir": -1, "v_dir": +1},
        }
        thick = L * 0.04

        for corner, info in (placement or {}).items():
            cfg = corner_cfg[corner]
            ox, oy = cfg["origin"]
            h_dir, v_dir = cfg["h_dir"], cfg["v_dir"]
            h_leg, v_leg = info["horiz"], info["vert"]
            color = corner_colors[corner]

            # L-shape as SVG path
            pts = [
                (ox, oy),
                (ox + h_dir * h_leg, oy),
                (ox + h_dir * h_leg, oy + v_dir * thick),
                (ox + h_dir * thick, oy + v_dir * thick),
                (ox + h_dir * thick, oy + v_dir * v_leg),
                (ox, oy + v_dir * v_leg),
            ]
            path = "M " + " L ".join(f"{x} {y}" for x, y in pts) + " Z"
            fig.add_shape(type="path", path=path,
                          fillcolor=color, opacity=0.6,
                          line=dict(color="black", width=1.5))

            # Label
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            p = info["panel"]
            fig.add_annotation(x=cx, y=cy, text=f"{p['panel_id']}<br>{info['a_len']}×{info['b_len']}",
                               showarrow=False, font=dict(size=9, color=black),
                               bgcolor="rgba(255,255,255,0.95)", bordercolor="gray", borderwidth=1, borderpad=3, opacity=0.85)

        # Empty corner labels
        for corner in CORNERS:
            if corner not in (placement or {}):
                cfg = corner_cfg[corner]
                ox, oy = cfg["origin"]
                fig.add_annotation(x=cx, y=cy, text=f"{p['panel_id']}<br>a={info['a_len']} b={info['b_len']}",
                   showarrow=False, font=dict(size=9, color="black"),
                   bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1,
                   borderpad=3)

        fig.update_layout(
            xaxis=dict(range=[-L * 0.1, L * 1.1], scaleanchor="y",
                       showgrid=False, zeroline=False),
            yaxis=dict(range=[-L * 0.1, L * 1.1],
                       showgrid=False, zeroline=False),
        )

    elif mould["mould_type"] == "cross":
        Q = int(mould["quadrant_limit"])
        gap = Q * 0.12
        total = 2 * Q + gap
        thick = Q * 0.06

        quad_boxes = {
            "Q1": (0, Q + gap), "Q2": (Q + gap, Q + gap),
            "Q3": (0, 0), "Q4": (Q + gap, 0),
        }
        quad_colors = {"Q1": "#4C72B0", "Q2": "#DD8452",
                       "Q3": "#55A868", "Q4": "#C44E52"}
        quad_cfg = {
            "Q1": {"L_origin": (Q, Q + gap), "h_dir": -1, "v_dir": +1},
            "Q2": {"L_origin": (Q + gap, Q + gap), "h_dir": +1, "v_dir": +1},
            "Q3": {"L_origin": (Q, Q), "h_dir": -1, "v_dir": -1},
            "Q4": {"L_origin": (Q + gap, Q), "h_dir": +1, "v_dir": -1},
        }

        # Quadrant boxes
        for qname, (bx, by) in quad_boxes.items():
            fig.add_shape(type="rect", x0=bx, y0=by, x1=bx + Q, y1=by + Q,
                          line=dict(color="black", width=1.5, dash="dash"),
                          fillcolor="rgba(0,0,0,0)")

        # Centre cross
        fig.add_shape(type="rect", x0=Q, y0=0, x1=Q + gap, y1=total,
                      fillcolor="rgba(200,200,200,0.2)", line=dict(width=0))
        fig.add_shape(type="rect", x0=0, y0=Q, x1=total, y1=Q + gap,
                      fillcolor="rgba(200,200,200,0.2)", line=dict(width=0))

        for qname in QUADRANTS:
            bx, by = quad_boxes[qname]
            cfg = quad_cfg[qname]
            if qname in (placement or {}):
                info = placement[qname]
                color = quad_colors[qname]
                lx, ly = cfg["L_origin"]
                h_dir, v_dir = cfg["h_dir"], cfg["v_dir"]
                h_leg, v_leg = info["horiz"], info["vert"]

                pts = [
                    (lx, ly),
                    (lx + h_dir * h_leg, ly),
                    (lx + h_dir * h_leg, ly + v_dir * thick),
                    (lx + h_dir * thick, ly + v_dir * thick),
                    (lx + h_dir * thick, ly + v_dir * v_leg),
                    (lx, ly + v_dir * v_leg),
                ]
                path = "M " + " L ".join(f"{x} {y}" for x, y in pts) + " Z"
                fig.add_shape(type="path", path=path,
                              fillcolor=color, opacity=0.6,
                              line=dict(color="black", width=1))

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (pts[0][0] + pts[1][0]) / 2
                cy = (pts[0][1] + pts[2][1]) / 2
                p = info["panel"]
                fig.add_annotation(x=cx, y=cy,
                   text=f"{p['panel_id']}<br>a={info['a_len']} b={info['b_len']}",
                   showarrow=False, font=dict(size=9, color="black"),
                   bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1,
                   borderpad=3)
            else:
                fig.add_annotation(x=bx + Q * 0.5, y=by + Q * 0.5,
                                   text=f"{qname}<br>(empty)",
                                   showarrow=False, font=dict(size=8, color="gray"))

        fig.update_layout(
            xaxis=dict(range=[-total * 0.08, total * 1.08], scaleanchor="y"),
            yaxis=dict(range=[-total * 0.08, total * 1.08]),
        )

    fig.update_layout(
        title=dict(text=step_label, font=dict(size=14)),
        xaxis_title="mm", yaxis_title="mm",
        width=500, height=500,
        margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    return fig



# Sidebar


with st.sidebar:
    st.markdown("## 🏗️ PBU Mould Scheduler")
    st.markdown("---")

    # Panel data input
    st.markdown("### 📦 Panel Data")
    data_source = st.radio("Data source", ["Upload CSV", "Sample data"],
                           horizontal=True)

    panels_df = None
    selected_pbus = []
    n_pbus = 0
    moulds_list = []
    run_button = False

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload panels.csv", type=["csv"])
        if uploaded:
            panels_df = pd.read_csv(uploaded)
        else:
            # Auto-load default dataset if available (for Streamlit Cloud)
            import os
            default_csv = os.path.join(os.path.dirname(__file__), "panels.csv")
            if os.path.exists(default_csv):
                panels_df = pd.read_csv(default_csv)
                st.success("Loaded default dataset from repository")
    else:
        # Generate sample data for demo
        st.info("Using built-in sample data (25 PBUs)")
        sample_rows = []
        np.random.seed(42)
        sample_pbu_types = ["1a", "1b", "1c-m", "2a", "2b", "3a", "3a(a)",
                            "3b", "3b(a)", "4a", "4a(a)", "4b", "4b(a)",
                            "1a", "1b", "1a", "2a", "3a", "3b", "4a",
                            "1b", "2b", "3a(a)", "4b", "4b(a)"]
        for pbu_idx in range(1, 26):
            pbu_id = f"PBU{pbu_idx:03d}"
            pbu_type = sample_pbu_types[pbu_idx - 1] if pbu_idx <= len(sample_pbu_types) else "1a"
            for code in ["C02", "C03", "C04"]:
                a_len = np.random.choice([1200, 1500, 1800, 2100, 2400, 2700])
                b_len = np.random.choice([800, 1000, 1200, 1500, 1800])
                # Ensure a_len <= b_len (a = shorter, b = longer)
                if a_len > b_len:
                    a_len, b_len = b_len, a_len
                sample_rows.append({
                    "panel_id": f"{pbu_id}/{code}",
                    "pbu_type": pbu_type,
                    "a_len": a_len,
                    "b_len": b_len,
                    "has_recess": np.random.choice([0, 1]),
                    "has_ceiling": np.random.choice([0, 1]),
                    "num_openings": np.random.choice([0, 1, 2, 3]),
                })
        panels_df = pd.DataFrame(sample_rows)

    if panels_df is not None:
        # Parse
        panels_df["pbu_id"] = panels_df["panel_id"].str.split("/").str[0]
        panels_df["panel_code"] = panels_df["panel_id"].str.split("/").str[1]
        panels_df["a_len"] = panels_df["a_len"].astype(int)
        panels_df["b_len"] = panels_df["b_len"].astype(int)
        panels_df["has_recess"] = panels_df["has_recess"].astype(int)
        panels_df["has_ceiling"] = panels_df["has_ceiling"].astype(int)
        panels_df["num_openings"] = panels_df["num_openings"].astype(int)
        # Ensure pbu_type exists (default to "1a" / V1 if not in CSV)
        if "pbu_type" not in panels_df.columns:
            panels_df["pbu_type"] = "1a"

        all_pbus = sorted(panels_df["pbu_id"].unique())
        n_pbus = st.slider("Number of PBUs", 1, len(all_pbus), min(25, len(all_pbus)))
        selected_pbus = all_pbus[:n_pbus]

        st.markdown("---")

        # Mould configuration
        st.markdown("### 🔧 Mould Fleet")
        mould_preset = st.selectbox("Mould preset", [
            "Default (4×M1 + M2 + M3 cross)",
            "Square only (5×square)",
            "Minimal (2×M1 + 1×cross)",
            "Large fleet (6×M1 + 2×cross)",
            "Custom",
        ])

        if mould_preset == "Custom":
            n_m1 = st.number_input("Large square moulds (3900mm)", 0, 8, 4)
            n_m2 = st.number_input("Medium square moulds (3000mm)", 0, 4, 1)
            n_cross = st.number_input("Cross moulds (2000mm/quad)", 0, 4, 1)
        elif mould_preset == "Square only (5×square)":
            n_m1, n_m2, n_cross = 4, 1, 0
        elif mould_preset == "Minimal (2×M1 + 1×cross)":
            n_m1, n_m2, n_cross = 2, 0, 1
        elif mould_preset == "Large fleet (6×M1 + 2×cross)":
            n_m1, n_m2, n_cross = 6, 0, 2
        else:
            n_m1, n_m2, n_cross = 4, 1, 1

        # Build mould list
        moulds_list = []
        for i in range(n_m1):
            suffix = chr(ord('a') + i)
            moulds_list.append({
                "mould_id": f"M1{suffix}", "mould_type": "square", "max_length": 3900
            })
        for i in range(n_m2):
            moulds_list.append({
                "mould_id": f"M2{'.' + str(i+1) if n_m2 > 1 else ''}",
                "mould_type": "square", "max_length": 3000
            })
        for i in range(n_cross):
            moulds_list.append({
                "mould_id": f"M3{'.' + str(i+1) if n_cross > 1 else ''}",
                "mould_type": "cross", "max_length": 2000, "quadrant_limit": 2000
            })

        st.markdown(f"**Fleet:** {len(moulds_list)} moulds "
                    f"({n_m1} large sq + {n_m2} med sq + {n_cross} cross)")

        st.markdown("---")
        run_button = st.button("🚀 **Optimise Schedule**", type="primary",
                               use_container_width=True)



# Main content


st.markdown("# 🏗️ Intelligent Mould Scheduling")
st.markdown("*Constraint Programming + ML Framework for Precast Panel Production*")

if panels_df is None:
    st.warning("⬅️ Please upload panel data or select **Sample data** in the sidebar to get started.")
    st.stop()

# Filter panels
panels_test = panels_df[panels_df["pbu_id"].isin(selected_pbus)].copy().reset_index(drop=True)

# Summary cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="kpi-card">
        <h2>{n_pbus}</h2><p>PBUs Selected</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="kpi-card">
        <h2>{len(panels_test)}</h2><p>Total Panels</p></div>""", unsafe_allow_html=True)
with col3:
    cross_fit = ((panels_test["a_len"] <= 2000) & (panels_test["b_len"] <= 2000)).sum()
    st.markdown(f"""<div class="kpi-card">
        <h2>{cross_fit}</h2><p>Cross-Compatible</p></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="kpi-card">
        <h2>{len(moulds_list)}</h2><p>Moulds in Fleet</p></div>""", unsafe_allow_html=True)

st.markdown("---")

# Panel data preview
with st.expander("📋 Panel Data Preview", expanded=False):
    st.dataframe(panels_test[["panel_id", "a_len", "b_len",
                               "has_ceiling", "has_recess", "num_openings"]],
                 use_container_width=True, height=300)


# Run optimisation


if run_button or "schedule_df" in st.session_state:
    if run_button:
        with st.spinner("Running optimisation..."):
            t_start = time.time()

            # Batching
            moulds_records = moulds_list
            panels_records = panels_test.to_dict("records")
            all_mould_ids = [m["mould_id"] for m in moulds_list]

            try:
                cycles = build_cycles_balanced(panels_records, moulds_records)
            except Exception as e:
                st.error(f"Batching failed: {e}")
                st.stop()

            n_assigned = sum(len(c["panels"]) for c in cycles)
            if n_assigned < len(panels_test):
                st.warning(f"Only {n_assigned}/{len(panels_test)} panels could be assigned. "
                           f"Some panels may be too large for available moulds.")

            # Baseline (FIFO)
            baseline = defaultdict(list)
            for c in cycles:
                baseline[c["mould_id"]].append(c)
            ms_baseline = 0
            for mid in all_mould_ids:
                t, prev = 0, None
                for c in baseline.get(mid, []):
                    t += setup_time_scaled(prev, c["recipe"]) + CAST_T
                    prev = c["recipe"]
                ms_baseline = max(ms_baseline, t)

            # CP-SAT optimisation
            try:
                sequences, makespan_scaled, time_by_mould = optimize_sequences_cpsat(
                    cycles, mould_ids=all_mould_ids
                )
            except Exception as e:
                st.error(f"CP-SAT optimisation failed: {e}")
                st.stop()

            schedule_df = build_timeline(sequences, all_mould_ids)
            elapsed = time.time() - t_start

            # Store in session state
            st.session_state["schedule_df"] = schedule_df
            st.session_state["cycles"] = cycles
            st.session_state["sequences"] = sequences
            st.session_state["all_mould_ids"] = all_mould_ids
            st.session_state["moulds_list"] = moulds_list
            st.session_state["makespan_scaled"] = makespan_scaled
            st.session_state["ms_baseline"] = ms_baseline
            st.session_state["elapsed"] = elapsed
            st.session_state["time_by_mould"] = time_by_mould

    # Retrieve from session state
    schedule_df = st.session_state["schedule_df"]
    cycles = st.session_state["cycles"]
    sequences = st.session_state["sequences"]
    all_mould_ids = st.session_state["all_mould_ids"]
    moulds_list = st.session_state["moulds_list"]
    makespan_scaled = st.session_state["makespan_scaled"]
    ms_baseline = st.session_state["ms_baseline"]
    elapsed = st.session_state["elapsed"]
    time_by_mould = st.session_state["time_by_mould"]

    makespan_h = makespan_scaled / SCALE
    baseline_h = ms_baseline / SCALE
    improvement = ((baseline_h - makespan_h) / baseline_h * 100) if baseline_h > 0 else 0

    # Results KPIs
    st.markdown("## 📊 Optimisation Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-improvement">
            <h2>{makespan_h:.1f}h</h2><p>Optimised Makespan</p></div>""",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <h2>{baseline_h:.1f}h</h2><p>Baseline Makespan</p></div>""",
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-improvement">
            <h2>{improvement:.1f}%</h2><p>Improvement</p></div>""",
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <h2>{elapsed:.1f}s</h2><p>Solve Time</p></div>""",
                    unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Schedule", "🔲 Mould Layouts", "📋 Cycle Details", "📈 Analysis"
    ])

    # tab 1: schedule
    with tab1:
        st.plotly_chart(
            create_gantt_chart(schedule_df, all_mould_ids,
                               "Optimised Production Schedule"),
            use_container_width=True,
        )

        st.markdown("### Mould Timeline")
        st.dataframe(
            schedule_df.sort_values(["start_h", "mould_id"]).reset_index(drop=True),
            use_container_width=True,
        )

        # Download CSV
        csv_buffer = schedule_df.to_csv(index=False)
        st.download_button("📥 Download Schedule CSV", csv_buffer,
                           "schedule.csv", "text/csv")

    # tab 2: mould layouts
    with tab2:
        st.markdown("### Panel Placement Visualisation")
        moulds_dict = {m["mould_id"]: m for m in moulds_list}

        sel_mould = st.selectbox("Select mould",
                                 [mid for mid in all_mould_ids
                                  if sequences.get(mid)])
        if sel_mould:
            seq = sequences.get(sel_mould, [])
            step_options = [f"Step {i+1} — {len(c['panels'])} panels"
                           for i, c in enumerate(seq)]
            sel_step_str = st.selectbox("Select step", step_options)
            sel_step_idx = step_options.index(sel_step_str)
            cyc = seq[sel_step_idx]
            mould = moulds_dict[sel_mould]
            placement = find_placement(cyc["panels"], mould)

            col_layout, col_details = st.columns([3, 2])
            with col_layout:
                fig = draw_mould_layout_plotly(
                    placement, mould,
                    f"Step {sel_step_idx+1} — {sel_mould} "
                    f"({mould['mould_type']})")
                st.plotly_chart(fig, use_container_width=True)

            with col_details:
                st.markdown(f"**Cycle {cyc['cycle_id']}** | "
                            f"Recipe: `{cyc['recipe']}`")
                for slot in sorted((placement or {}).keys()):
                    info = placement[slot]
                    p = info["panel"]
                    st.markdown(
                        f"- **{slot}**: `{p['panel_id']}` — "
                        f"a={info['a_len']}mm, b={info['b_len']}mm "
                        f"(ceil={p['has_ceiling']}, "
                        f"recess={p['has_recess']}, "
                        f"open={p['num_openings']})"
                    )

    # tab 3: cycle details
    with tab3:
        st.markdown("### All Cycles")
        cycle_rows = []
        for c in sorted(cycles, key=lambda c: (c["mould_id"], c["cycle_id"])):
            cycle_rows.append({
                "Cycle": c["cycle_id"],
                "Mould": c["mould_id"],
                "Panels": len(c["panels"]),
                "Panel IDs": ", ".join(p["panel_id"] for p in c["panels"]),
                "Recipe (b_len, ceil, recess, open)": str(c["recipe"]),
            })
        st.dataframe(pd.DataFrame(cycle_rows), use_container_width=True)

        # Per-mould summary
        st.markdown("### Per-Mould Summary")
        mould_summary = []
        for mid in all_mould_ids:
            mid_cycles = [c for c in cycles if c["mould_id"] == mid]
            mid_panels = sum(len(c["panels"]) for c in mid_cycles)
            t = time_by_mould.get(mid, 0) / SCALE
            util = (t / makespan_h * 100) if makespan_h > 0 else 0
            mould_summary.append({
                "Mould": mid,
                "Cycles": len(mid_cycles),
                "Panels": mid_panels,
                "Time (h)": round(t, 1),
                "Utilisation": f"{util:.0f}%",
            })
        st.dataframe(pd.DataFrame(mould_summary), use_container_width=True)

    # tab 4: analysis
    with tab4:
        st.markdown("### Mould Utilisation")
        util_data = []
        for mid in all_mould_ids:
            t = time_by_mould.get(mid, 0) / SCALE
            util = (t / makespan_h * 100) if makespan_h > 0 else 0
            util_data.append({"Mould": mid, "Utilisation (%)": util, "Time (h)": t})
        util_df = pd.DataFrame(util_data)

        fig_util = px.bar(util_df, x="Mould", y="Utilisation (%)",
                          color="Utilisation (%)",
                          color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
                          range_color=[0, 100])
        fig_util.add_hline(y=80, line_dash="dot", line_color="green",
                           annotation_text="80% target")
        fig_util.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_util, use_container_width=True)

        # Comparison
        st.markdown("### Baseline vs Optimised")
        comp_data = []
        for mid in all_mould_ids:
            # Baseline time
            bl_seq = [c for c in cycles if c["mould_id"] == mid]
            bt, prev = 0, None
            for c in bl_seq:
                bt += setup_time_scaled(prev, c["recipe"]) + CAST_T
                prev = c["recipe"]
            # Optimised time
            ot = time_by_mould.get(mid, 0)
            comp_data.append({
                "Mould": mid,
                "Baseline (h)": bt / SCALE,
                "Optimised (h)": ot / SCALE,
            })
        comp_df = pd.DataFrame(comp_data)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=comp_df["Mould"], y=comp_df["Baseline (h)"],
                                  name="Baseline", marker_color="lightgray"))
        fig_comp.add_trace(go.Bar(x=comp_df["Mould"], y=comp_df["Optimised (h)"],
                                  name="CP-SAT Optimised", marker_color="#2d5986"))
        fig_comp.update_layout(barmode="group", height=350,
                               yaxis_title="Time (hours)")
        st.plotly_chart(fig_comp, use_container_width=True)



# Footer

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8rem;'>"
    "Intelligent Mould Scheduling for Precast Panel Production — "
    "Final Year Project 2025/26</div>",
    unsafe_allow_html=True,
)
