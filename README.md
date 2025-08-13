# Bayesian Transformers + Higher-Order Graph Matching for Cell Tracking

This repository provides a complete, reproducible implementation of:
- **Uncertainty-aware embeddings** via a **Bayesian Transformer**; and  
- **Higher-order graph matching** with **belief propagation** for robust cell linkage and division handling.

The code mirrors the camera-ready submission and figure pipeline: feature extraction → Bayesian transformer embedding (μ_e, logσ_e²) → third-order matching with messages and lineage export. See the paper for details. :contentReference[oaicite:2]{index=2}

## Quick start

### 1) Install
```bash
git clone https://github.com/NabaviLab/bayesian-transformer-cell-tracking.git
cd bayes-track
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
