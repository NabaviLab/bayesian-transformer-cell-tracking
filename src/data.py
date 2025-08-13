from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist

def load_feature_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # drop empty/constant columns
    df = df.dropna(axis=1, how='all')
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)
    return df

def select_feature_columns(df: pd.DataFrame, frame_col: str, label_col: str) -> list[str]:
    return [c for c in df.columns if c not in [frame_col, label_col]]

def frame_slice(df: pd.DataFrame, frame_col: str, frame_idx: int) -> pd.DataFrame:
    return df[df[frame_col] == int(frame_idx)].copy()

def generate_real_data_with_features(df: pd.DataFrame,
                                     frame_col: str,
                                     label_col: str,
                                     first_frame: int,
                                     second_frame: int,
                                     features_to_use: list[str]):
    f1_df = frame_slice(df, frame_col, first_frame)
    f2_df = frame_slice(df, frame_col, second_frame)
    X1 = f1_df[features_to_use]
    X2 = f2_df[features_to_use]
    names1 = [f"{cid}_{first_frame}" for cid in f1_df[label_col].tolist()]
    names2 = [f"{cid}_{second_frame}" for cid in f2_df[label_col].tolist()]
    return X1, X2, names1, names2, f1_df, f2_df

def prepare_triplets(frame_pairs: list[tuple[int,int]],
                     df: pd.DataFrame,
                     features_to_use: list[str],
                     frame_col: str,
                     centroid_x_col: str,
                     centroid_y_col: str,
                     normalize_features: bool = True):
    X1_list, X2p_list, X2n_list = [], [], []

    if normalize_features:
        arr = df[features_to_use].values
        mean_ = arr.mean(axis=0)
        std_ = arr.std(axis=0) + 1e-8
    else:
        mean_ = 0.0
        std_ = 1.0

    for (f1, f2) in frame_pairs:
        sec1 = frame_slice(df, frame_col, f1)
        sec2 = frame_slice(df, frame_col, f2)
        if len(sec1) == 0 or len(sec2) == 0:
            continue

        x1_full = sec1[features_to_use].values
        x2_full = sec2[features_to_use].values
        x1_full = (x1_full - mean_) / std_
        x2_full = (x2_full - mean_) / std_

        c1 = sec1[[centroid_x_col, centroid_y_col]].values
        c2 = sec2[[centroid_x_col, centroid_y_col]].values
        dmat = cdist(c1, c2)
        pos_idx = np.argmin(dmat, axis=1)

        N = x1_full.shape[0]
        neg_idx = np.random.randint(0, x2_full.shape[0], size=N)
        for i in range(N):
            while neg_idx[i] == pos_idx[i]:
                neg_idx[i] = np.random.randint(0, x2_full.shape[0])

        X1_list.append(x1_full)
        X2p_list.append(x2_full[pos_idx])
        X2n_list.append(x2_full[neg_idx])

    if not X1_list:
        return None, None, None

    def to_tensor(lst):
        return torch.from_numpy(np.concatenate(lst, axis=0)).float()

    return to_tensor(X1_list), to_tensor(X2p_list), to_tensor(X2n_list)

class TripletDataset(Dataset):
    def __init__(self, X1: torch.Tensor, X2p: torch.Tensor, X2n: torch.Tensor):
        self.X1, self.X2p, self.X2n = X1, X2p, X2n

    def __len__(self): return self.X1.shape[0]

    def __getitem__(self, idx):
        return self.X1[idx], self.X2p[idx], self.X2n[idx]
