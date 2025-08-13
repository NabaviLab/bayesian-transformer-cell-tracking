from __future__ import annotations
import numpy as np
import torch

def belief_propagation_probabilistic(
    G,
    emb_t, emb_tp1,          # shapes (n1,d'), (n2,d') (only used for sizes)
    mu_1, logvar_1,          # torch (n1, d')  for frame t
    mu_2, logvar_2,          # torch (n2, d')  for frame t+1
    max_iter=5,
    damping=0.7
):
    """
    Message M(i->j) updated with triplet edges and uncertainty weighting.
    """
    single_edges = [(e['i'], e['j']) for e in G['edges_single']]
    M = { (e['i'], e['j']): 0.0 for e in G['edges_single'] }
    cost_single = { (e['i'], e['j']): e['cost'] for e in G['edges_single'] }

    def parse_idx(s):
        return int(s.split('_')[1])

    # prepare uncertainty weights (exp(-0.5*(logvar_i + logvar_j)) averaged across dims)
    # convert to CPU numpy for speed
    lv1 = logvar_1.detach().cpu().numpy()
    lv2 = logvar_2.detach().cpu().numpy()

    triplet_edges = G['edges_triplet']
    for _ in range(max_iter):
        newM = {}
        for (is_, js_) in single_edges:
            old = M[(is_, js_)]
            i = parse_idx(is_)
            j = parse_idx(js_)

            sum_trip = 0.0
            for ed in triplet_edges:
                ti = ed['i_i2_i3']; tj = ed['j1_j2_j3']
                if is_ in ti and js_ in tj:
                    sum_trip += -float(ed['cost'])

            unc = float(np.exp(-0.5 * (lv1[i] + lv2[j])).mean())
            c_ij = float(cost_single[(is_, js_)])
            combined = -c_ij * unc + sum_trip
            newM[(is_, js_)] = (1 - damping) * combined + damping * old
        M = newM

    n1, n2 = emb_t.shape[0], emb_tp1.shape[0]
    C = np.full((n1, n2), 9_999.0, dtype=np.float32)
    for (is_, js_) in single_edges:
        i, j = parse_idx(is_), parse_idx(js_)
        C[i, j] = -M[(is_, js_)]

    # normalize to [0,1]
    cmin, cmax = C.min(), C.max()
    C = (C - cmin) / (cmax - cmin + 1e-8)
    return C
