from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from .data import prepare_triplets, TripletDataset

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def train_bnn(model,
              frame_pairs,
              df,
              features_to_use,
              frame_col,
              centroid_x_col,
              centroid_y_col,
              num_epochs=100,
              lr=1e-3,
              margin=0.2,
              weight_decay=1e-5,
              batch_size=128,
              early_stopping_patience=10,
              reduce_lr_patience=5,
              device='cuda'):
    X1, X2p, X2n = prepare_triplets(frame_pairs, df, features_to_use,
                                    frame_col, centroid_x_col, centroid_y_col,
                                    normalize_features=True)
    if X1 is None:
        print("[ERROR] No triplets to train on.")
        return model

    loader = DataLoader(TripletDataset(X1, X2p, X2n), batch_size=batch_size, shuffle=True)

    device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                       patience=reduce_lr_patience, verbose=True)

    best, patience = float('inf'), 0
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        nb = 0
        for x1, x2p, x2n in loader:
            x1, x2p, x2n = x1.to(device), x2p.to(device), x2n.to(device)
            # add sequence dimension L=1
            x1 = x1.unsqueeze(1); x2p = x2p.unsqueeze(1); x2n = x2n.unsqueeze(1)

            opt.zero_grad()
            mu1, lv1, _ = model(x1, sample=True)
            mu2p, lv2p, _ = model(x2p, sample=True)
            mu2n, lv2n, _ = model(x2n, sample=True)

            z1   = reparameterize(mu1,  lv1)   # (B,1,d')
            z2p  = reparameterize(mu2p, lv2p)
            z2n  = reparameterize(mu2n, lv2n)

            # FIX: compute squared L2 over all non-batch dims -> shape (B,)
            pos = (z1 - z2p).pow(2).sum(dim=(1,2))
            neg = (z1 - z2n).pow(2).sum(dim=(1,2))
            loss = torch.clamp(pos - neg + margin, min=0.0).mean()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item(); nb += 1

        avg = total / max(nb, 1)
        sched.step(avg)
        improved = avg < best - 1e-6
        best = min(best, avg)
        patience = 0 if improved else patience + 1
        print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg:.4f}; no_improve={patience}")
        if patience >= early_stopping_patience:
            print("Early stopping.")
            break
    return model
