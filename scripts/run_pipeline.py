from __future__ import annotations
import os
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.utils import ensure_dir, set_seed
from src.data import load_feature_table, select_feature_columns, generate_real_data_with_features
from src.models.model import BayesianTransformerForCellTracking
from src.training import train_bnn
from src.graph import construct_higher_order_graph
from src.matching import higher_order_graph_matching_with_divisions
from src.tracking import update_tracking_with_divisions, export_lineage_to_mantrack, generate_output_csv_with_divisions
from src.visualize import visualize_matching, visualize_dynamic_matching_with_plotly

def main(cfg_path: str = "configs/default.yaml"):
    with open(cfg_path, "r") as f:
        C = yaml.safe_load(f)

    set_seed(C.get("seed", 123))

    out_dir = ensure_dir(C["out_dir"])
    figs_dir = ensure_dir(out_dir / "figures")
    results_dir = ensure_dir(out_dir / "results")

    # --- Data ---
    df = load_feature_table(C["csv_path"])
    features_to_use = select_feature_columns(df, C["frame_col"], C["label_col"])

    # ensure centroids exist
    assert C["centroid_x_col"] in df.columns and C["centroid_y_col"] in df.columns, \
        "Centroid columns missing in CSV."

    input_dim = len(features_to_use) if C["input_dim"] is None else C["input_dim"]

    # --- Model ---
    model = BayesianTransformerForCellTracking(
        input_dim=input_dim,
        embed_dim=C["embed_dim"],
        num_heads=C["num_heads"],
        ff_hidden_dim=C["ff_hidden_dim"],
        num_layers=C["num_layers"],
        output_dim=C["output_dim"],
        prior_mu=C["prior_mu"],
        prior_sigma=C["prior_sigma"],
        dropout=C["dropout"],
        use_layernorm=C["use_layernorm"],
        save_feature_attention_every=C["save_feature_attention_every"],
        out_fig_dir=str(figs_dir / "feature_attention")
    )

    # --- Train ---
    frame_pairs = [(i, i + 1) for i in range(C["start_frame"], C["end_frame"])]
    model = train_bnn(
        model=model,
        frame_pairs=frame_pairs,
        df=df,
        features_to_use=features_to_use,
        frame_col=C["frame_col"],
        centroid_x_col=C["centroid_x_col"],
        centroid_y_col=C["centroid_y_col"],
        num_epochs=C["num_epochs"],
        lr=C["lr"],
        margin=C["margin"],
        weight_decay=C["weight_decay"],
        batch_size=C["batch_size"],
        early_stopping_patience=C["early_stopping_patience"],
        reduce_lr_patience=C["reduce_lr_patience"],
        device=C["device"]
    )

    lineage_info = {}
    tracking = {}

    device = "cuda" if (C["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    model.to(device)
    model.eval()

    unique_frames = sorted(df[C["frame_col"]].unique())
    num_frames = len(unique_frames)

    for f in tqdm(range(C["start_frame"], C["end_frame"]), desc="Linking"):
        X1, X2, names1, names2, sec1_df, sec2_df = generate_real_data_with_features(
            df, C["frame_col"], C["label_col"], f, f+1, features_to_use
        )

        # Tensors (B, 1, F)
        t1 = torch.tensor(X1.values, dtype=torch.float32).unsqueeze(1).to(device)
        t2 = torch.tensor(X2.values, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            mu1, logv1, _ = model(t1, sample=False, frame_idx=f)
            mu2, logv2, _ = model(t2, sample=False, frame_idx=f)

        # embeddings (use mu_e): shapes (n, d')
        emb1 = mu1.squeeze(1).detach().cpu().numpy()
        emb2 = mu2.squeeze(1).detach().cpu().numpy()

        # HO graph on embeddings
        G = construct_higher_order_graph(
            emb1, emb2,
            max_single_neighbors=C["max_single_neighbors"],
            max_triplet_neighbors=C["max_triplet_neighbors"],
            triplet_dist_thresh=C["triplet_dist_thresh"]
        )

        # BP + divisions
        matches, Cmat, d1, d2 = higher_order_graph_matching_with_divisions(
            G, emb1, emb2,
            mu1.squeeze(1), logv1.squeeze(1),  # (n1,d')
            mu2.squeeze(1), logv2.squeeze(1),  # (n2,d')
            sec1_df, sec2_df,
            C["centroid_x_col"], C["centroid_y_col"], C.get("area_col", None),
            base_percentile=C["base_percentile"],
            bp_max_iter=C["bp_max_iter"],
            bp_damping=C["bp_damping"],
            motion_threshold=C["motion_threshold"],
            area_variation=C["area_variation"]
        )

        # update tracks/lineage
        tracking = update_tracking_with_divisions(
            tracking, matches, f, num_frames,
            names1, names2, sec1_df, sec2_df,
            C["centroid_x_col"], C["centroid_y_col"], lineage_info
        )

        # save figs
        png_path = figs_dir / f"matched_frame_{f}_to_{f+1}.png"
        visualize_matching(
            sec1_df, sec2_df, matches, names1, names2,
            C["centroid_x_col"], C["centroid_y_col"],
            str(png_path),
            f"Frame {f}", f"Frame {f+1}"
        )
        html_path = figs_dir / f"interactive_frame_{f}_to_{f+1}.html"
        visualize_dynamic_matching_with_plotly(
            sec1_df, sec2_df, matches, names1, names2,
            C["centroid_x_col"], C["centroid_y_col"],
            str(html_path),
            f"Frame {f}", f"Frame {f+1}"
        )

    export_lineage_to_mantrack(lineage_info, str(results_dir / "man_track.txt"))
    generate_output_csv_with_divisions(tracking, num_frames, str(results_dir / "matching_results.csv"))
    print("Done.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()
    main(args.config)
