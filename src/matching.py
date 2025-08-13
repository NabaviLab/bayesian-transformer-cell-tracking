from __future__ import annotations
import numpy as np
import networkx as nx
from .belief import belief_propagation_probabilistic

def higher_order_graph_matching_with_divisions(
    G,
    emb_t, emb_tp1,            # (n1,d'), (n2,d')
    mu_1, logvar_1, mu_2, logvar_2,
    sec1_df, sec2_df,          # original per-frame DataFrames (for centroid/area)
    centroid_x_col: str, centroid_y_col: str, area_col: str | None,
    base_percentile=20,
    bp_max_iter=10,
    bp_damping=0.9,
    motion_threshold=200.0,
    area_variation=1.5
):
    C = belief_propagation_probabilistic(
        G, emb_t, emb_tp1, mu_1, logvar_1, mu_2, logvar_2,
        max_iter=bp_max_iter, damping=bp_damping
    )

    n1, n2 = C.shape
    division_threshold = np.percentile(C, base_percentile)

    matches = []
    dummy1, dummy2 = [], []
    potential = {}

    # cache geometric attrs
    x1 = sec1_df[centroid_x_col].values
    y1 = sec1_df[centroid_y_col].values
    x2 = sec2_df[centroid_x_col].values
    y2 = sec2_df[centroid_y_col].values
    has_area = (area_col in sec1_df.columns and area_col in sec2_df.columns) if area_col else False
    if has_area:
        a1 = sec1_df[area_col].values
        a2 = sec2_df[area_col].values

    for i in range(n1):
        candidates = np.where(C[i, :] < division_threshold)[0]
        valid = []
        for j in candidates:
            dx = float(x1[i] - x2[j]); dy = float(y1[i] - y2[j])
            dist = (dx**2 + dy**2) ** 0.5
            area_ok = True
            if has_area:
                area_ok = abs(a1[i] - a2[j]) / (a1[i] + 1e-8) <= area_variation
            if dist <= motion_threshold and area_ok:
                valid.append((j, C[i, j]))
        if not valid:
            dummy1.append(f"1_{i}")
        elif len(valid) == 1:
            j, _ = valid[0]
            matches.append((f"1_{i}", f"2_{j}"))
        else:
            valid.sort(key=lambda x: x[1])
            potential[i] = [j for j, _ in valid[:2]]

    # greedy division resolution allowing up to 2 children
    Gd = nx.Graph()
    for i in potential.keys():
        Gd.add_node(f"1_{i}")
    for js in potential.values():
        for j in js:
            Gd.add_node(f"2_{j}")
    for i, js in potential.items():
        for j in js:
            Gd.add_edge(f"1_{i}", f"2_{j}", weight=C[i, j])

    edges = sorted(Gd.edges(data=True), key=lambda e: e[2]['weight'])
    assigned_parent, assigned_child = {}, {}
    for u, v, d in edges:
        parent, child = (u, v) if u.startswith("1_") else (v, u)
        if child in assigned_child:
            continue
        if parent in assigned_parent and not isinstance(assigned_parent[parent], list):
            assigned_parent[parent] = [assigned_parent[parent]]
        if parent in assigned_parent and len(assigned_parent[parent]) < 2:
            assigned_parent[parent].append(child)
        elif parent not in assigned_parent:
            assigned_parent[parent] = child
        assigned_child[child] = parent

    for p, ch in assigned_parent.items():
        matches.append((p, ch) if isinstance(ch, list) else (p, ch))

    matched_children = set(assigned_child.keys())
    for j in range(n2):
        if f"2_{j}" not in matched_children:
            dummy2.append(f"2_{j}")

    return matches, C, dummy1, dummy2
