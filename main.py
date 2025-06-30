import numpy as np
import os
import math
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.patches import ConnectionPatch
import pandas as pd
import torch.distributions as dist
import time
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import mpld3
import plotly.graph_objects as go
import csv
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, DataLoader  
from torch.nn.utils import clip_grad_norm_       



file_path = r'/home/mok23003/cell_tracking/public_data/Flou_N2DL_Hela/Flou_N2DL_Hela_enriched_features.csv'
multisectionmeasurements_df = pd.read_csv(file_path)
multisectionmeasurements_df.dropna(axis=1, how='all', inplace=True)
constant_cols = []
for col in multisectionmeasurements_df.columns:
    if len(multisectionmeasurements_df[col].dropna().unique()) <= 1:
        constant_cols.append(col)
if constant_cols:
    multisectionmeasurements_df.drop(columns=constant_cols, inplace=True)

lineage_info = {}
features_to_use = [col for col in multisectionmeasurements_df.columns if col not in ['FrameID','Label']]
image_column_name = 'FrameID'

def generate_real_data_with_features(df, first_frame, second_frame, features_to_use):
    sec04_data = df[df[image_column_name] == int(first_frame)]
    sec05_data = df[df[image_column_name] == int(second_frame)]
    sec04_cells = sec04_data[features_to_use]
    sec05_cells = sec05_data[features_to_use]
    sec04_names = [f"{cell_id}_{first_frame}" for cell_id in sec04_data['Label'].tolist()]
    sec05_names = [f"{cell_id}_{second_frame}" for cell_id in sec05_data['Label'].tolist()]
    return sec04_cells, sec05_cells, sec04_names, sec05_names

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.ones(out_features, in_features) * -3.0)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -3.0)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_logvar, -5.0)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_logvar, -5.0)

    def forward(self, x, sample=True):
        if sample:
            w_std = torch.exp(0.5 * self.weight_logvar)
            eps_w = torch.randn_like(w_std)
            weight = self.weight_mu + w_std * eps_w
            b_std = torch.exp(0.5 * self.bias_logvar)
            eps_b = torch.randn_like(b_std)
            bias = self.bias_mu + b_std * eps_b
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class BayesianMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2, prior_mu=0.0, prior_sigma=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = BayesianLinear(embed_dim, embed_dim, prior_mu, prior_sigma)
        self.k_proj = BayesianLinear(embed_dim, embed_dim, prior_mu, prior_sigma)
        self.v_proj = BayesianLinear(embed_dim, embed_dim, prior_mu, prior_sigma)
        self.out_proj = BayesianLinear(embed_dim, embed_dim, prior_mu, prior_sigma)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, sample=True):
        bsz, seq_len, _ = x.size()
        Q = self.q_proj(x, sample=sample)
        K = self.k_proj(x, sample=sample)
        V = self.v_proj(x, sample=sample)
        Q = Q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn_output, sample=sample)

class BayesianTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2, ff_hidden_dim=256,
                 prior_mu=0.0, prior_sigma=0.1, use_layernorm=True, dropout=0.1):
        super().__init__()
        self.attn = BayesianMultiheadSelfAttention(embed_dim, num_heads, prior_mu, prior_sigma)
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if use_layernorm:
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)
        self.ff1 = BayesianLinear(embed_dim, ff_hidden_dim, prior_mu, prior_sigma)
        self.ff2 = BayesianLinear(ff_hidden_dim, embed_dim, prior_mu, prior_sigma)
        self.act = nn.ReLU()

    def forward(self, x, sample=True):
        a = self.attn(x, sample=sample)
        a = self.dropout(a)
        if self.use_layernorm:
            x = self.ln1(x + a)
        else:
            x = x + a
        f = self.ff1(x, sample=sample)
        f = self.act(f)
        f = self.ff2(f, sample=sample)
        f = self.dropout(f)
        if self.use_layernorm:
            x = self.ln2(x + f)
        else:
            x = x + f
        return x

def prepare_triplets(
    frame_pairs, 
    df, 
    features_to_use, 
    normalize_features=True
):
    """
    Gathers all (x1, x2_pos, x2_neg) from the given frame pairs into big tensors.
    frame_pairs: list of (frame1, frame2)
    df: your DataFrame (multisectionmeasurements_df)
    features_to_use: which columns to feed into the BNN
    normalize_features: bool to apply global mean/std normalization
    Returns:
        X1_t, X2_pos_t, X2_neg_t (torch.FloatTensor) or (None, None, None) if no data
    """
    X1_list = []
    X2_pos_list = []
    X2_neg_list = []

    # Optionally compute global mean/std for normalization
    if normalize_features:
        all_data = df[features_to_use].values
        mean_ = all_data.mean(axis=0)
        std_ = all_data.std(axis=0) + 1e-8
    else:
        mean_ = 0
        std_ = 1

    for (frame1, frame2) in frame_pairs:
        sec1, sec2, _, _ = generate_real_data_with_features(
            df, frame1, frame2, features_to_use
        )
        if len(sec1) == 0 or len(sec2) == 0:
            continue

        x1_full_np = sec1.values
        x2_full_np = sec2.values

        # Normalize if needed
        x1_full_np = (x1_full_np - mean_) / std_
        x2_full_np = (x2_full_np - mean_) / std_

        # Distances based on 'Centroid_X', 'Centroid_Y'
        centroids1 = sec1[['Centroid_X', 'Centroid_Y']].values
        centroids2 = sec2[['Centroid_X', 'Centroid_Y']].values
        dist_matrix = cdist(centroids1, centroids2)

        # Positive pairs: closest in frame2 for each cell in frame1
        positive_indices = np.argmin(dist_matrix, axis=1)

        # Negative pairs: random but distinct from the positive
        N = x1_full_np.shape[0]
        negative_indices = np.random.randint(0, x2_full_np.shape[0], size=N)
        for i in range(N):
            while negative_indices[i] == positive_indices[i]:
                negative_indices[i] = np.random.randint(0, x2_full_np.shape[0])

        x1_np = x1_full_np
        x2_pos_np = x2_full_np[positive_indices]
        x2_neg_np = x2_full_np[negative_indices]

        X1_list.append(x1_np)
        X2_pos_list.append(x2_pos_np)
        X2_neg_list.append(x2_neg_np)

    if len(X1_list) == 0:
        print("[WARNING] No valid triplets found. Check your data/frames.")
        return None, None, None

    # Concatenate everything
    X1_cat = np.concatenate(X1_list, axis=0)
    X2_pos_cat = np.concatenate(X2_pos_list, axis=0)
    X2_neg_cat = np.concatenate(X2_neg_list, axis=0)

    # Convert to torch
    X1_t = torch.from_numpy(X1_cat).float()
    X2_pos_t = torch.from_numpy(X2_pos_cat).float()
    X2_neg_t = torch.from_numpy(X2_neg_cat).float()

    return X1_t, X2_pos_t, X2_neg_t

class BayesianTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, ff_hidden_dim=256,
                 num_layers=2, prior_mu=0.0, prior_sigma=0.1,
                 dropout=0.1, use_layernorm=True):
        super().__init__()
        self.layers = nn.ModuleList([
            BayesianTransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                use_layernorm=use_layernorm,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, sample=True):
        for layer in self.layers:
            x = layer(x, sample=sample)
        return x

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def train_bnn(
    bnn,
    frame_pairs,
    df,
    features_to_use,
    num_epochs=100,
    lr=1e-3,
    margin=0.2,
    weight_decay=1e-5,
    batch_size=128,
    early_stopping_patience=10,
    reduce_lr_patience=5,
    device='cuda'
):
    """
    An advanced training function that:
     - Gathers triplets from all frame pairs (prepare_triplets)
     - Builds a Dataset/DataLoader for mini-batching
     - Uses a margin-based contrastive loss
     - Includes optional LR scheduling & early stopping
    """

    # 1) Prepare all triplets
    X1_t, X2_pos_t, X2_neg_t = prepare_triplets(frame_pairs, df, features_to_use, normalize_features=True)
    if X1_t is None:
        print("[ERROR] No triplets to train on. Exiting early.")
        return bnn

    # 2) Create Dataset + DataLoader
    dataset = TripletDataset(X1_t, X2_pos_t, X2_neg_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3) Device handling
    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    bnn.to(device)

    # 4) Optimizer & LR scheduler
    optimizer = torch.optim.Adam(bnn.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=reduce_lr_patience, verbose=True
    )

    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(num_epochs):
        bnn.train()
        total_loss = 0.0
        num_batches = 0

        for (x1, x2_pos, x2_neg) in loader:
            x1 = x1.to(device)
            x2_pos = x2_pos.to(device)
            x2_neg = x2_neg.to(device)

            # If BNN expects (B, seq_len, N_features), unsqueeze for seq_len=1
            x1 = x1.unsqueeze(1)
            x2_pos = x2_pos.unsqueeze(1)
            x2_neg = x2_neg.unsqueeze(1)

            optimizer.zero_grad()

            # Forward pass (sample=True for Bayesian)
            mu1, logvar1, _ = bnn(x1, sample=True)
            mu2_pos, logvar2_pos, _ = bnn(x2_pos, sample=True)
            mu2_neg, logvar2_neg, _ = bnn(x2_neg, sample=True)

            # Reparameterize
            z1 = reparameterize(mu1, logvar1)
            z2_pos = reparameterize(mu2_pos, logvar2_pos)
            z2_neg = reparameterize(mu2_neg, logvar2_neg)

            # Compute margin-based contrastive loss
            positive_loss = (z1 - z2_pos).pow(2).mean(dim=1)
            negative_loss = (z1 - z2_neg).pow(2).mean(dim=1)
            contrastive_loss = torch.clamp(positive_loss - negative_loss + margin, min=0.0).mean()

            # Optional: add KL or other terms if your BNN supports it
            # total_loss_batch = contrastive_loss + kl_beta * bnn.kl_loss()  (example)
            total_loss_batch = contrastive_loss

            total_loss_batch.backward()
            clip_grad_norm_(bnn.parameters(), 1.0)
            optimizer.step()

            total_loss += total_loss_batch.item()
            num_batches += 1

        # Average loss for this epoch
        avg_loss = total_loss / (num_batches + 1e-8)
        scheduler.step(avg_loss)

        # Early stopping logic
        if avg_loss < (best_loss - 1e-6):
            best_loss = avg_loss
            no_improvement = 0
        else:
            no_improvement += 1

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | no_improvement: {no_improvement}")

        if no_improvement >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    return bnn


class TripletDataset(Dataset):
    """
    Holds (x1, x2_pos, x2_neg) triplets. Each is shape (N_features).
    """
    def __init__(self, X1, X2_pos, X2_neg):
        super().__init__()
        self.X1 = X1
        self.X2_pos = X2_pos
        self.X2_neg = X2_neg

    def __len__(self):
        # Required: number of triplets
        return self.X1.shape[0]

    def __getitem__(self, idx):
        # Returns a single sample of (x1, x2_pos, x2_neg) for index idx
        return self.X1[idx], self.X2_pos[idx], self.X2_neg[idx]


class BayesianTransformerForCellTracking(nn.Module):
    def __init__(
        self,
        input_dim=10,
        embed_dim=64,
        num_heads=2,
        ff_hidden_dim=256,
        num_layers=2,
        output_dim=2,
        prior_mu=0.0,
        prior_sigma=0.1,
        dropout=0.1,
        use_layernorm=True
    ):
        super().__init__()
        self.feature_attention = BayesianLinear(input_dim, input_dim, prior_mu, prior_sigma)
        self.input_proj = BayesianLinear(input_dim, embed_dim, prior_mu, prior_sigma)
        self.encoder = BayesianTransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            dropout=dropout,
            use_layernorm=use_layernorm
        )
        self.out_mu = BayesianLinear(embed_dim, output_dim, prior_mu, prior_sigma)
        self.out_logvar = BayesianLinear(embed_dim, output_dim, prior_mu, prior_sigma)
        self.alpha = 0.9
        self.register_buffer('prev_attention', torch.zeros(input_dim))

    def forward(self, x, sample=True, frame_idx=None):
        attn_scores = self.feature_attention(x, sample=sample)
        attn_scores = F.softmax(attn_scores, dim=-1)
        if self.training:
            mean_attn = attn_scores.mean(dim=0)
            smoothed_attn = self.alpha * self.prev_attention + (1 - self.alpha) * mean_attn.squeeze(0)
            self.prev_attention = smoothed_attn.detach()
        else:
            smoothed_attn = attn_scores.mean(dim=0).squeeze(0)
        if frame_idx is not None and frame_idx % 2 == 0:
            plt.figure(figsize=(8,4))
            plt.bar(range(x.shape[-1]), smoothed_attn.detach().cpu().numpy())
            plt.title(f"Feature Attention at frame_idx={frame_idx}")
            plt.savefig(f"/home/mok23003/cell_tracking/public_data/results/Fluo-N2DL-HeLa/debuggin_X/feature_attention_{frame_idx}.png", bbox_inches='tight')
            plt.close()
        x = x * smoothed_attn.unsqueeze(0)
        embed = self.input_proj(x, sample=sample)
        enc_out = self.encoder(embed, sample=sample)
        mu = self.out_mu(enc_out, sample=sample)
        logvar = self.out_logvar(enc_out, sample=sample)
        return mu, logvar, smoothed_attn

def construct_higher_order_graph(
    section_1,  # shape (n1, d')
    section_2,  # shape (n2, d')
    max_single_neighbors=8,
    max_triplet_neighbors=10,
    triplet_dist_thresh=50.0
):
    """
    Constructs a third-order graph (triplet-based) for matching frames t -> t+1.
    'edges_single' link each cell i in frame t to candidate matches j in frame t+1.
    'edges_triplet' encode triple-wise geometry by connecting (i1,i2,i3) in t 
    to (j1,j2,j3) in t+1 if their triple geometry is similar enough.
    """
    n1 = section_1.shape[0]
    n2 = section_2.shape[0]
    G = {
        'nodes_t': [f"t_{i}" for i in range(n1)],
        'nodes_t+1': [f"t+1_{j}" for j in range(n2)],
        'edges_single': [],
        'edges_triplet': []
    }

    # (1) First-Order Edges similar to your original bipartite
    for i in range(n1):
        dist_vec = np.linalg.norm(section_2 - section_1[i], axis=1)
        nn_sorted = np.argsort(dist_vec)[:max_single_neighbors]
        for j in nn_sorted:
            cost_ij = dist_vec[j]
            G['edges_single'].append({
                'i': f"t_{i}",
                'j': f"t+1_{j}",
                'cost': cost_ij
            })

    # (2) Gather all triplets (i1,i2,i3) in frame t with each pairwise distance < triplet_dist_thresh
    triplets_t = []
    for i1 in range(n1):
        for i2 in range(i1+1, n1):
            for i3 in range(i2+1, n1):
                coords = [section_1[i1], section_1[i2], section_1[i3]]
                if _all_within_thresh(coords, triplet_dist_thresh):
                    tri_stat = _triplet_geometry(coords)
                    triplets_t.append((i1, i2, i3, tri_stat))

    # Similarly for frame t+1
    triplets_tplus1 = []
    for j1 in range(n2):
        for j2 in range(j1+1, n2):
            for j3 in range(j2+1, n2):
                coords = [section_2[j1], section_2[j2], section_2[j3]]
                if _all_within_thresh(coords, triplet_dist_thresh):
                    tri_stat = _triplet_geometry(coords)
                    triplets_tplus1.append((j1, j2, j3, tri_stat))

    # (3) For each triplet in t, link to up to max_triplet_neighbors triplets in t+1
    for (i1, i2, i3, tri_stat_t) in triplets_t:
        # Compare with all triplets in t+1
        # measure difference in geometry to produce a cost
        cand_list = []
        for (j1, j2, j3, tri_stat_tp1) in triplets_tplus1:
            cost_val = _triplet_cost(tri_stat_t, tri_stat_tp1)
            cand_list.append((j1, j2, j3, cost_val))
        cand_list.sort(key=lambda x: x[3])
        cand_list = cand_list[:max_triplet_neighbors]

        for (j1, j2, j3, cost_val) in cand_list:
            G['edges_triplet'].append({
                'i_i2_i3': (f"t_{i1}", f"t_{i2}", f"t_{i3}"),
                'j1_j2_j3': (f"t+1_{j1}", f"t+1_{j2}", f"t+1_{j3}"),
                'cost': cost_val
            })

    return G

def _all_within_thresh(coords, dist_thresh):
    """
    Check if all pairwise distances among these three coords are < dist_thresh.
    coords is a list of length 3, each shape (d',).
    """
    import numpy as np
    for a in range(3):
        for b in range(a+1,3):
            d = np.linalg.norm(coords[a] - coords[b])
            if d > dist_thresh:
                return False
    return True

def _triplet_geometry(coords):
    """
    Extract a simple measure of the triangle formed by these three points (e.g., side lengths).
    coords is [p1, p2, p3], each shape (d',).
    """
    import numpy as np
    side1 = np.linalg.norm(coords[0] - coords[1])
    side2 = np.linalg.norm(coords[1] - coords[2])
    side3 = np.linalg.norm(coords[0] - coords[2])
    return [side1, side2, side3]

def _triplet_cost(triA, triB):
    """
    Compare two triplets' geometry. e.g. sum of absolute differences in side lengths.
    triA, triB are lists of side lengths.
    """
    import numpy as np
    triA = sorted(triA)
    triB = sorted(triB)
    return float(np.sum(np.abs(np.array(triA) - np.array(triB))))  # L1 difference


def belief_propagation_probabilistic(
    G,
    section_1,       # shape (n1, d')
    section_2,       # shape (n2, d')
    mu_1, logvar_1,  # shape (n1, embed_dim) for each cell
    mu_2, logvar_2,  # shape (n2, embed_dim) for each cell
    max_iter=5,
    damping=0.7
):
    """
    Belief-propagation approach with triplet edges.
    We store a message M(i->j) for each single-edge (i->j).
    The cost from triplet edges modifies these messages.
    """
    import numpy as np
    import torch

    # identify single edges (i->j)
    single_edges = [(e['i'], e['j']) for e in G['edges_single']]
    M = {}
    cost_single = {}
    for e in G['edges_single']:
        i_str, j_str, c = e['i'], e['j'], e['cost']
        M[(i_str, j_str)] = 0.0
        cost_single[(i_str, j_str)] = c

    # triplet edges
    triplet_edges = G['edges_triplet']  # each has 'cost' referencing i1_i2_i3 -> j1_j2_j3

    def parse_index(node_str):
        # e.g. node_str = "t_3" => parse out 3
        # or "t+1_5" => parse out 5
        return int(node_str.split('_')[1])

    for iteration in range(max_iter):
        newM = {}
        for (i_str, j_str) in single_edges:
            old_val = M[(i_str, j_str)]
            i_id = parse_index(i_str)
            j_id = parse_index(j_str)

            # sum up triplet-based influences
            sum_triplet_msgs = 0.0
            for ed in triplet_edges:
                (ti1, ti2, ti3) = ed['i_i2_i3']
                (tj1, tj2, tj3) = ed['j1_j2_j3']
                if i_str in (ti1, ti2, ti3) and j_str in (tj1, tj2, tj3):
                    # smaller cost => stronger match
                    sum_triplet_msgs += - ed['cost']

            # incorporate Bayesian uncertainty
            unc = torch.exp(-0.5 * (logvar_1[i_id] + logvar_2[j_id])).mean().item()

            # single-edge cost
            c_ij = cost_single[(i_str, j_str)]
            combined_msg = -c_ij * unc + sum_triplet_msgs

            new_val = (1 - damping)*combined_msg + damping*old_val
            newM[(i_str, j_str)] = new_val
        M = newM

    # build cost matrix from final messages
    n1 = section_1.shape[0]
    n2 = section_2.shape[0]
    cost_matrix = np.full((n1, n2), 9999.0, dtype=np.float32)
    for (i_str, j_str) in single_edges:
        i_id = parse_index(i_str)
        j_id = parse_index(j_str)
        cost_matrix[i_id, j_id] = -M[(i_str, j_str)]

    # min–max normalization
    cmin = cost_matrix.min()
    cmax = cost_matrix.max()
    denom = cmax - cmin if cmax > cmin else 1e-8
    cost_matrix = (cost_matrix - cmin)/denom
    return cost_matrix


def higher_order_graph_matching_with_divisions(
    G,
    section_1,
    section_2,
    mu_1, logvar_1,
    mu_2, logvar_2,
    image_width=1024,
    image_height=1024,
    base_percentile=20  # Lowered from 20 for stricter threshold
):
    cost_matrix = belief_propagation_probabilistic(
        G, section_1, section_2, mu_1, logvar_1, mu_2, logvar_2, max_iter=10, damping=0.9
    )
    n1, n2 = len(section_1), len(section_2)
    cell_density = n1 / (image_width * image_height + 1e-8)
    final_percentile = base_percentile + 10 if cell_density > 0.1 else base_percentile
    division_threshold = np.percentile(cost_matrix, final_percentile)
    print(f"\n[DEBUG] higher_order_graph_matching_with_divisions:")
    print(f" - cost_matrix shape = {cost_matrix.shape}, n1={n1}, n2={n2}")
    print(f" - division_threshold = {division_threshold:.4f} (percentile {final_percentile})")
    matches = []
    dummy_cells_1 = []
    dummy_cells_2 = []
    potential_matches = {}
    motion_threshold = 200 
    area_variation = 1.5    
    for i in range(n1):
        candidate_js = np.where(cost_matrix[i, :] < division_threshold)[0]
        valid_js = []
        for j in candidate_js:
            dx = section_1[i, 0] - section_2[j, 0]
            dy = section_1[i, 1] - section_2[j, 1]
            dist_ij = np.sqrt(dx**2 + dy**2)
            area1 = section_1[i, 2]  # Assuming area is 3rd column
            area2 = section_2[j, 2]
            ratio_area = abs(area1 - area2) / (area1 + 1e-8)
            if dist_ij <= motion_threshold and ratio_area <= area_variation:
                valid_js.append((j, cost_matrix[i, j]))
        if not valid_js:
            dummy_cells_1.append(f"1_{i}")
        elif len(valid_js) == 1:
            j, _ = valid_js[0]
            matches.append((f"1_{i}", f"2_{j}"))
        else:
            # Limit to top 2 candidates by cost
            valid_js.sort(key=lambda x: x[1])  # Sort by cost
            potential_matches[i] = [j for j, _ in valid_js[:2]]  # Max 2 children
    division_graph = nx.Graph()
    for i in potential_matches.keys():
        division_graph.add_node(f"1_{i}")
    for indices in potential_matches.values():
        for j in indices:
            division_graph.add_node(f"2_{j}")
    for i, js in potential_matches.items():
        for j in js:
            division_graph.add_edge(f"1_{i}", f"2_{j}", weight=cost_matrix[i, j])
    edges = sorted(division_graph.edges(data=True), key=lambda x: x[2]['weight'])
    assigned_parents = {}
    assigned_children = {}
    for u, v, data in edges:
        if u.startswith("1_") and v.startswith("2_"):
            parent, child = u, v
        else:
            parent, child = v, u
        if child in assigned_children:
            continue
        if parent in assigned_parents and not isinstance(assigned_parents[parent], list):
            assigned_parents[parent] = [assigned_parents[parent]]
        if parent in assigned_parents and len(assigned_parents[parent]) < 2:
            assigned_parents[parent].append(child)
        elif parent not in assigned_parents:
            assigned_parents[parent] = child
        assigned_children[child] = parent
    for parent, children in assigned_parents.items():
        matches.append((parent, children) if isinstance(children, list) else (parent, children))
    matched_children = set(assigned_children.keys())
    for j in range(n2):
        node_j = f"2_{j}"
        if node_j not in matched_children:
            dummy_cells_2.append(node_j)
    print(f"[DEBUG] Final matches: {matches}")
    print(f"[DEBUG] dummy_cells_1: {dummy_cells_1}")
    print(f"[DEBUG] dummy_cells_2: {dummy_cells_2}")
    return matches, cost_matrix, dummy_cells_1, dummy_cells_2
def visualize_matching(section_1, section_2, matches, sec04_names, sec05_names, sec04_cells, sec05_cells, frame_idx):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    marker_size = 50
    text_size = 12
    axes[0].scatter(sec04_cells['Centroid_X'], sec04_cells['Centroid_Y'], color='blue', s=marker_size)
    for i, name in enumerate(sec04_names):
        cell_id = name.split('_')[0]
        axes[0].text(sec04_cells.iloc[i]['Centroid_X'], sec04_cells.iloc[i]['Centroid_Y'], cell_id, fontsize=text_size)
    axes[0].set_title(f"Frame {frame_idx}")
    axes[0].set_aspect('equal')
    axes[1].scatter(sec05_cells['Centroid_X'], sec05_cells['Centroid_Y'], color='green', s=marker_size)
    for i, name in enumerate(sec05_names):
        cell_id = name.split('_')[0]
        axes[1].text(sec05_cells.iloc[i]['Centroid_X'], sec05_cells.iloc[i]['Centroid_Y'], cell_id, fontsize=text_size)
    axes[1].set_title(f"Frame {frame_idx + 1}")
    axes[1].set_aspect('equal')
    for match in matches:
        parent_cell = match[0]
        child_cells = match[1] if isinstance(match[1], list) else [match[1]]
        parent_idx = int(parent_cell.split('_')[1])
        coord1 = (sec04_cells.iloc[parent_idx]['Centroid_X'], sec04_cells.iloc[parent_idx]['Centroid_Y'])
        axes[0].plot([coord1[0]], [coord1[1]], 'ro')
        for child_cell in child_cells:
            child_idx = int(child_cell.split('_')[1])
            coord2 = (sec05_cells.iloc[child_idx]['Centroid_X'], sec05_cells.iloc[child_idx]['Centroid_Y'])
            axes[1].plot([coord2[0]], [coord2[1]], 'ro')
            con = ConnectionPatch(xyA=coord1, xyB=coord2, coordsA="data", coordsB="data",
                                  axesA=axes[0], axesB=axes[1], color="red", alpha=0.6, linestyle="--")
            axes[1].add_artist(con)
    out_png = f'/home/mok23003/cell_tracking/public_data/results/Fluo-N2DL-HeLa/debuggin_X/X_matched_frame_{frame_idx}_to_{frame_idx + 1}.png'
    plt.savefig(out_png, dpi=600, format='png')
    plt.close(fig)
    print(f"Matching visualization saved as: {out_png}")

def visualize_dynamic_matching_with_plotly(section_1, section_2, matches, sec04_names, sec05_names, sec04_cells, sec05_cells, frame_idx):
    x1, y1 = sec04_cells['Centroid_X'].values, sec04_cells['Centroid_Y'].values
    x2, y2 = sec05_cells['Centroid_X'].values, sec05_cells['Centroid_Y'].values
    name_to_index_sec04 = {name: idx for idx, name in enumerate(sec04_names)}
    name_to_index_sec05 = {name: idx for idx, name in enumerate(sec05_names)}
    matched_indices = []
    for match in matches:
        parent_cell = match[0]
        child_cells = match[1] if isinstance(match[1], list) else [match[1]]
        idx1 = int(parent_cell.split('_')[1])
        for child_cell in child_cells:
            idx2 = int(child_cell.split('_')[1])
            matched_indices.append((idx1, idx2))
    num_points_sec1 = len(x1)
    num_points_sec2 = len(x2)
    matched_indices_dict = {str(i): j for i,j in matched_indices}
    matched_indices_reverse_dict = {str(j): i for i,j in matched_indices}
    unmatched_indices_sec1 = set(range(num_points_sec1)) - set(i for i,_ in matched_indices)
    unmatched_indices_sec2 = set(range(num_points_sec2)) - set(j for _,j in matched_indices)
    cell_ids_sec1 = [n.split('_')[0] for n in sec04_names]
    cell_ids_sec2 = [n.split('_')[0] for n in sec05_names]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Frame {frame_idx}', f'Frame {frame_idx + 1}'), horizontal_spacing=0.1)
    matched_idx1 = [i for i,_ in matched_indices]
    mc1 = np.column_stack((x1[matched_idx1], y1[matched_idx1]))
    matched_sec1_names = [cell_ids_sec1[i] for i in matched_idx1]
    fig.add_trace(go.Scatter(x=mc1[:, 0], y=mc1[:, 1], mode='markers', marker=dict(color='blue', size=8),
                             text=matched_sec1_names, name=f'Matched Cells Frame {frame_idx}', hoverinfo='text', customdata=matched_idx1), row=1, col=1)
    unmatched_idx1 = list(unmatched_indices_sec1)
    uc1 = np.column_stack((x1[unmatched_idx1], y1[unmatched_idx1]))
    unmatched_sec1_names = [cell_ids_sec1[i] for i in unmatched_idx1]
    fig.add_trace(go.Scatter(x=uc1[:, 0], y=uc1[:, 1], mode='markers', marker=dict(color='red', size=8),
                             text=unmatched_sec1_names, name=f'Unmatched Cells Frame {frame_idx}', hoverinfo='text', customdata=unmatched_idx1), row=1, col=1)
    matched_idx2 = [j for _,j in matched_indices]
    mc2 = np.column_stack((x2[matched_idx2], y2[matched_idx2]))
    matched_sec2_names = [cell_ids_sec2[j] for j in matched_idx2]
    fig.add_trace(go.Scatter(x=mc2[:, 0], y=mc2[:, 1], mode='markers', marker=dict(color='green', size=8),
                             text=matched_sec2_names, name=f'Matched Cells Frame {frame_idx + 1}', hoverinfo='text', customdata=matched_idx2), row=1, col=2)
    unmatched_idx2 = list(unmatched_indices_sec2)
    uc2 = np.column_stack((x2[unmatched_idx2], y2[unmatched_idx2]))
    unmatched_sec2_names = [cell_ids_sec2[j] for j in unmatched_idx2]
    fig.add_trace(go.Scatter(x=uc2[:, 0], y=uc2[:, 1], mode='markers', marker=dict(color='red', size=8),
                             text=unmatched_sec2_names, name=f'Unmatched Cells Frame {frame_idx + 1}', hoverinfo='text', customdata=unmatched_idx2), row=1, col=2)
    fig.update_xaxes(title_text='X Coordinate', row=1, col=1)
    fig.update_xaxes(title_text='X Coordinate', row=1, col=2)
    fig.update_yaxes(title_text='Y Coordinate', row=1, col=1)
    fig.update_yaxes(title_text='Y Coordinate', row=1, col=2)
    fig.update_layout(title=f"Interactive Matching Visualization for Frame {frame_idx} to {frame_idx + 1}",
                      hovermode='closest', showlegend=True, margin=dict(t=100))
    import json
    js_mi = matched_indices_dict
    js_mi_r = matched_indices_reverse_dict
    js_cmd_sec1_m = matched_idx1
    js_cmd_sec2_m = matched_idx2
    js_code = f"""
    var matchedIndicesDict = {json.dumps(js_mi)};
    var matchedIndicesReverseDict = {json.dumps(js_mi_r)};
    var cellIDsFrame1 = {json.dumps(cell_ids_sec1)};
    var cellIDsFrame2 = {json.dumps(cell_ids_sec2)};
    var js_customdata_sec1_matched = {json.dumps(js_cmd_sec1_m)};
    var js_customdata_sec2_matched = {json.dumps(js_cmd_sec2_m)};
    var myPlot = document.getElementById('plot');
    function dataToPaper(x, axis) {{
        var axisObject = myPlot._fullLayout[axis];
        var range = axisObject.range;
        var domain = axisObject.domain;
        var d = (x - range[0]) / (range[1] - range[0]);
        return domain[0] + d * (domain[1] - domain[0]);
    }}
    myPlot.on('plotly_hover', function(data){{
        var point = data.points[0];
        var curveNumber = point.curveNumber;
        var x0 = point.x; var y0 = point.y;
        Plotly.relayout(myPlot, {{ 'shapes': [], 'annotations': [] }});
        var shapes = [];
        var annotations = [];
        if(curveNumber === 0) {{
            var idx1 = point.customdata;
            var idx2 = matchedIndicesDict[idx1];
            if(idx2 !== undefined) {{
                idx2 = parseInt(idx2);
                var indexInData2 = js_customdata_sec2_matched.indexOf(idx2);
                var data2 = myPlot.data[2];
                var x1 = data2.x[indexInData2];
                var y1 = data2.y[indexInData2];
                var x0_paper = dataToPaper(x0, 'xaxis');
                var y0_paper = dataToPaper(y0, 'yaxis');
                var x1_paper = dataToPaper(x1, 'xaxis2');
                var y1_paper = dataToPaper(y1, 'yaxis2');
                shapes.push({{
                    type: 'line', x0: x0_paper, y0: y0_paper, x1: x1_paper, y1: y1_paper,
                    xref: 'paper', yref: 'paper',
                    line: {{ color: 'red', width: 2, dash: 'dot' }}
                }});
                var c1 = cellIDsFrame1[idx1];
                var c2 = cellIDsFrame2[idx2];
                annotations.push({{
                    x: 0.5, y: 1.15, xref: 'paper', yref: 'paper', xanchor: 'center', yanchor: 'bottom',
                    text: 'Frame {frame_idx} Cell ID: ' + c1 + ' ↔ Frame {frame_idx+1} Cell ID: ' + c2,
                    showarrow: false, font: {{ size: 14 }}, align: 'center', bordercolor: 'black',
                    borderwidth: 1, bgcolor: 'white', opacity: 0.8
                }});
            }}
        }} else if(curveNumber === 2) {{
            var idx2 = point.customdata;
            var idx1 = matchedIndicesReverseDict[idx2];
            if(idx1 !== undefined) {{
                idx1 = parseInt(idx1);
                var indexInData1 = js_customdata_sec1_matched.indexOf(idx1);
                var data1 = myPlot.data[0];
                var x1 = data1.x[indexInData1];
                var y1 = data1.y[indexInData1];
                var x0_paper = dataToPaper(x1, 'xaxis');
                var y0_paper = dataToPaper(y1, 'yaxis');
                var x1_paper = dataToPaper(point.x, 'xaxis2');
                var y1_paper = dataToPaper(point.y, 'yaxis2');
                shapes.push({{
                    type: 'line', x0: x0_paper, y0: y0_paper, x1: x1_paper, y1: y1_paper,
                    xref: 'paper', yref: 'paper',
                    line: {{ color: 'red', width: 2, dash: 'dot' }}
                }});
                var c1 = cellIDsFrame1[idx1];
                var c2 = cellIDsFrame2[idx2];
                annotations.push({{
                    x: 0.5, y: 1.15, xref: 'paper', yref: 'paper', xanchor: 'center', yanchor: 'bottom',
                    text: 'Frame {frame_idx} Cell ID: ' + c1 + ' ↔ Frame {frame_idx+1} Cell ID: ' + c2,
                    showarrow: false, font: {{ size: 14 }}, align: 'center', bordercolor: 'black',
                    borderwidth: 1, bgcolor: 'white', opacity: 0.8
                }});
            }}
        }}
        Plotly.relayout(myPlot, {{ 'shapes': shapes, 'annotations': annotations }});
    }});
    myPlot.on('plotly_unhover', function(data){{
        Plotly.relayout(myPlot, {{ 'shapes': [], 'annotations': [] }});
    }});
    """
    out_html = f"/home/mok23003/cell_tracking/public_data/results/Fluo-N2DL-HeLa/debuggin_X/X_interactive_frame_{frame_idx}_to_{frame_idx + 1}.html"
    fig.write_html(out_html, include_plotlyjs='cdn', full_html=True, post_script=js_code, config={'responsive': True}, div_id='plot')
    print(f"Interactive visualization saved to: {out_html}")

def update_tracking_with_divisions(
    tracking,
    matches,
    frame_idx,
    num_frames,
    sec04_names,
    sec05_names,
    lineage_info,
    sec04_cells,
    sec05_cells
):
    import numpy as np
    name_to_index_sec04 = {name: idx for idx, name in enumerate(sec04_names)}
    name_to_index_sec05 = {name: idx for idx, name in enumerate(sec05_names)}
    matched_cells_in_frame_idx = set()
    matched_cells_in_frame_idx_plus1 = set()
    for match in matches:
        parent_cell = match[0]
        child_cells = match[1] if isinstance(match[1], list) else [match[1]]
        parent_idx = int(parent_cell.split('_')[1])
        parent_cell_id = int(sec04_names[parent_idx].split('_')[0])
        parent_cx = sec04_cells.iloc[parent_idx]['Centroid_X']
        parent_cy = sec04_cells.iloc[parent_idx]['Centroid_Y']
        parent_centroid = np.array([parent_cx, parent_cy], dtype=np.float32)
        child_cell_ids = []
        child_centroids = []
        for child in child_cells:
            child_idx = int(child.split('_')[1])
            child_cell_id = int(sec05_names[child_idx].split('_')[0])
            child_cell_ids.append(child_cell_id)
            ccx = sec05_cells.iloc[child_idx]['Centroid_X']
            ccy = sec05_cells.iloc[child_idx]['Centroid_Y']
            child_centroids.append(np.array([ccx, ccy], dtype=np.float32))
        matched_cells_in_frame_idx.add(parent_cell_id)
        matched_cells_in_frame_idx_plus1.update(child_cell_ids)
        if len(child_cells) == 1:
            child_cell_id = child_cell_ids[0]
            child_centroid = child_centroids[0]
            track_id = None
            for t_id, arr in tracking.items():
                if isinstance(arr[frame_idx], dict) and arr[frame_idx]['cell_id'] == parent_cell_id:
                    track_id = t_id
                    break
            if track_id is None:
                track_id = len(tracking)
                tracking[track_id] = [None] * num_frames
                lineage_info[track_id] = {
                    'start': frame_idx,
                    'end': frame_idx,
                    'parent': 0
                }
            tracking[track_id][frame_idx] = {'cell_id': parent_cell_id, 'centroid': parent_centroid}
            tracking[track_id][frame_idx + 1] = {'cell_id': child_cell_id, 'centroid': child_centroid}
            lineage_info[track_id]['end'] = frame_idx + 1
        else:
            track_id = None
            for t_id, arr in tracking.items():
                if isinstance(arr[frame_idx], dict) and arr[frame_idx]['cell_id'] == parent_cell_id:
                    track_id = t_id
                    break
            if track_id is None:
                track_id = len(tracking)
                tracking[track_id] = [None] * num_frames
                tracking[track_id][frame_idx] = {'cell_id': parent_cell_id, 'centroid': parent_centroid}
                lineage_info[track_id] = {
                    'start': frame_idx,
                    'end': frame_idx,
                    'parent': 0
                }
            for f in range(frame_idx + 1, num_frames):
                tracking[track_id][f] = None
            lineage_info[track_id]['end'] = frame_idx
            for (c_id, c_centroid) in zip(child_cell_ids, child_centroids):
                new_t_id = len(tracking)
                tracking[new_t_id] = [None] * num_frames
                tracking[new_t_id][frame_idx + 1] = {'cell_id': c_id, 'centroid': c_centroid}
                lineage_info[new_t_id] = {
                    'start': frame_idx + 1,
                    'end': frame_idx + 1,
                    'parent': track_id
                }
    all_cells_in_frame_idx = set(int(name.split('_')[0]) for name in sec04_names)
    unmatched_cells_in_frame_idx = all_cells_in_frame_idx - matched_cells_in_frame_idx
    all_cells_in_frame_idx_plus1 = set(int(name.split('_')[0]) for name in sec05_names)
    unmatched_cells_in_frame_idx_plus1 = all_cells_in_frame_idx_plus1 - matched_cells_in_frame_idx_plus1
    for cell_id in unmatched_cells_in_frame_idx_plus1:
        t_id = len(tracking)
        tracking[t_id] = [None] * num_frames
        tracking[t_id][frame_idx + 1] = {'cell_id': cell_id, 'centroid': np.array([0,0], dtype=np.float32)}
        lineage_info[t_id] = {
            'start': frame_idx + 1,
            'end': frame_idx + 1,
            'parent': 0
        }
    return tracking

def generate_output_csv_with_divisions(tracking, num_frames):
    data = []
    for track_id, track in tracking.items():
        row = [track_id] + track[:num_frames]
        data.append(row)
    columns = ["Tracking_ID"] + [f"Frame_{i}" for i in range(num_frames)]
    tracking_df = pd.DataFrame(data, columns=columns)
    out_csv = f'/home/mok23003/cell_tracking/public_data/results/Fluo-N2DL-HeLa/debuggin_X/matching_restuls_X.csv'
    tracking_df.to_csv(out_csv, index=False)
    print(f'Tracking results saved to {out_csv}')

def export_lineage_to_mantrack(lineage_info, output_path):
    with open(output_path, 'w') as f:
        for track_id in sorted(lineage_info.keys()):
            L = track_id + 1
            B = lineage_info[track_id]['start']
            E = lineage_info[track_id]['end']
            parent_track_id = lineage_info[track_id]['parent']
            if parent_track_id == 0:
                P = 0
            else:
                P = parent_track_id + 1
            line = f"{L} {B} {E} {P}\n"
            f.write(line)
    print(f"man_track.txt file saved at: {output_path}")

def main(start_frame, end_frame):
    start_time = time.time()
    num_frames = len(multisectionmeasurements_df[image_column_name].unique())
    bnn = BayesianTransformerForCellTracking(
        input_dim=len(features_to_use),
        embed_dim=64,
        num_heads=2,
        ff_hidden_dim=256,
        num_layers=2,
        output_dim=2,
        prior_mu=0.0,
        prior_sigma=0.1,
        dropout=0.1,
        use_layernorm=True
    )
    frame_pairs = [(i, i+1) for i in range(start_frame, end_frame)]
    train_bnn(
    bnn=bnn,
    frame_pairs=frame_pairs,
    df=multisectionmeasurements_df,
    features_to_use=features_to_use,
    num_epochs=100
    )
    tracking = {}
    for frame_idx in range(start_frame, end_frame):
        section_1, section_2, sec04_names, sec05_names = generate_real_data_with_features(
            multisectionmeasurements_df, frame_idx, frame_idx + 1, features_to_use
        )
        print(f"\n[DEBUG] Frame {frame_idx} → {frame_idx+1}")
        print(f"  section_1 shape = {section_1.shape}, section_2 shape = {section_2.shape}")
        print(f"  sec04_names[:5] = {sec04_names[:5]}, sec05_names[:5] = {sec05_names[:5]}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        section_1_tensor = torch.tensor(section_1.values, dtype=torch.float32).unsqueeze(1).to(device)
        section_2_tensor = torch.tensor(section_2.values, dtype=torch.float32).unsqueeze(1).to(device)

        mu_1, logvar_1, _ = bnn(section_1_tensor, sample=False, frame_idx=frame_idx)
        mu_2, logvar_2, _ = bnn(section_2_tensor, sample=False, frame_idx=frame_idx)

        G = construct_higher_order_graph(section_1.values, section_2.values)
        matches, cost_matrix, dummy_cells_1, dummy_cells_2 = higher_order_graph_matching_with_divisions(
            G,
            section_1.values,
            section_2.values,
            mu_1,
            logvar_1,
            mu_2,
            logvar_2
        )
        division_threshold = np.percentile(cost_matrix.flatten(), 15)
        print(f"""
        Frame {frame_idx}→{frame_idx+1}:
        Matched: {len(matches)} pairs
        Unmatched: {len(dummy_cells_1)} (prev frame), {len(dummy_cells_2)} (next frame)
        Avg. Match Cost: {cost_matrix[np.where(cost_matrix < division_threshold)].mean():.3f}
        ± {cost_matrix[np.where(cost_matrix < division_threshold)].std():.3f}
        """)
        sec04_cells = multisectionmeasurements_df[multisectionmeasurements_df[image_column_name] == frame_idx]
        sec05_cells = multisectionmeasurements_df[multisectionmeasurements_df[image_column_name] == (frame_idx + 1)]
        tracking = update_tracking_with_divisions(
            tracking,
            matches,
            frame_idx,
            num_frames,
            sec04_names,
            sec05_names,
            lineage_info,
            sec04_cells,
            sec05_cells
        )
        visualize_matching(
            section_1.values,
            section_2.values,
            matches,
            sec04_names,
            sec05_names,
            sec04_cells,
            sec05_cells,
            frame_idx
        )
        visualize_dynamic_matching_with_plotly(
            section_1.values,
            section_2.values,
            matches,
            sec04_names,
            sec05_names,
            sec04_cells,
            sec05_cells,
            frame_idx
        )
    export_lineage_to_mantrack(lineage_info, "/home/mok23003/cell_tracking/public_data/results/Fluo-N2DL-HeLa/debuggin_X/X_man_track.txt")
    generate_output_csv_with_divisions(tracking, num_frames)
    print(f"Total execution time: {time.time() - start_time} seconds")

if __name__ == "__main__":
    start_frame = 0
    end_frame = 90
    main(start_frame, end_frame)