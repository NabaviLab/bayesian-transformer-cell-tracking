from __future__ import annotations
import numpy as np

def _all_within_thresh(coords, dist_thresh: float) -> bool:
    for a in range(3):
        for b in range(a+1, 3):
            if np.linalg.norm(coords[a] - coords[b]) > dist_thresh:
                return False
    return True

def _triplet_geometry(coords):
    # simple triangle side lengths
    s1 = np.linalg.norm(coords[0] - coords[1])
    s2 = np.linalg.norm(coords[1] - coords[2])
    s3 = np.linalg.norm(coords[0] - coords[2])
    return [s1, s2, s3]

def _triplet_cost(triA, triB) -> float:
    A = sorted(triA); B = sorted(triB)
    return float(np.sum(np.abs(np.array(A) - np.array(B))))

def construct_higher_order_graph(
    emb_t,      # np.ndarray (n1, d') : embeddings for frame t (use mu_e or z)
    emb_tp1,    # np.ndarray (n2, d') : embeddings for frame t+1
    max_single_neighbors=8,
    max_triplet_neighbors=10,
    triplet_dist_thresh=50.0
):
    """
    Build third-order graph using *embedding space* distances (paper’s first-order),
    and triplet geometry in embedding space as well.
    """
    n1, n2 = emb_t.shape[0], emb_tp1.shape[0]
    G = {
        'nodes_t':   [f"t_{i}"   for i in range(n1)],
        'nodes_t+1': [f"t+1_{j}" for j in range(n2)],
        'edges_single': [],
        'edges_triplet': [],
    }

    # (1) first-order edges using embedding distances
    for i in range(n1):
        d = np.linalg.norm(emb_tp1 - emb_t[i], axis=1)
        nn_idx = np.argsort(d)[:max_single_neighbors]
        for j in nn_idx:
            G['edges_single'].append({'i': f"t_{i}", 'j': f"t+1_{j}", 'cost': float(d[j])})

    # (2) triplets in t
    trip_t = []
    for i1 in range(n1):
        for i2 in range(i1+1, n1):
            for i3 in range(i2+1, n1):
                coords = [emb_t[i1], emb_t[i2], emb_t[i3]]
                if _all_within_thresh(coords, triplet_dist_thresh):
                    trip_t.append((i1, i2, i3, _triplet_geometry(coords)))

    # (3) triplets in t+1
    trip_tp1 = []
    for j1 in range(n2):
        for j2 in range(j1+1, n2):
            for j3 in range(j2+1, n2):
                coords = [emb_tp1[j1], emb_tp1[j2], emb_tp1[j3]]
                if _all_within_thresh(coords, triplet_dist_thresh):
                    trip_tp1.append((j1, j2, j3, _triplet_geometry(coords)))

    # (4) link triplets
    for (i1, i2, i3, triA) in trip_t:
        cand = []
        for (j1, j2, j3, triB) in trip_tp1:
            cand.append((j1, j2, j3, _triplet_cost(triA, triB)))
        cand.sort(key=lambda x: x[3])
        for (j1, j2, j3, c) in cand[:max_triplet_neighbors]:
            G['edges_triplet'].append({
                'i_i2_i3': (f"t_{i1}", f"t_{i2}", f"t_{i3}"),
                'j1_j2_j3': (f"t+1_{j1}", f"t+1_{j2}", f"t+1_{j3}"),
                'cost': float(c)
            })
    return G
