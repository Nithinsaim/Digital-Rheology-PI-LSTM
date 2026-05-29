# 🧪 Digital Rheology of Polymers Using Physics-Informed LSTM

> **Manuscript in Preparation (MDPI)** | Amrita Vishwa Vidyapeetham, Coimbatore

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![R2](https://img.shields.io/badge/R²-0.9819-brightgreen?style=for-the-badge)]()
[![Physics](https://img.shields.io/badge/Physics--Informed-LSTM-orange?style=for-the-badge)]()

---

## 📄 Abstract

Viscoelastic materials subjected to **Large Amplitude Oscillatory Shear (LAOS)** exhibit strong nonlinear stress–strain responses, making data-driven prediction challenging. We develop a hybrid **Physics-Informed LSTM (PI-LSTM)** model that integrates a two-layer recurrent neural architecture with a **Burgers viscoelastic constitutive law**.

Training is performed using an **Augmented Lagrangian (ADMM-like)** framework that enforces physical residuals through adaptive multipliers. The model achieves **R² = 0.9819**, accurately captures hysteresis loops and peak stresses, and consistently outperforms baseline purely data-driven models.

---

## 🏗️ System Architecture

```
Input: Strain ε(t), Strain Rate ε̇(t), Time t
          │
          ▼
  ┌─────────────────────────────────┐
  │   2-Layer Unidirectional LSTM   │
  │   (128 hidden units per layer)  │
  └─────────────────────────────────┘
          │  hidden state h_t ∈ R^128
          ▼
  LayerNorm → FC1(128→128, SiLU, dropout=0.2) → FC2(128→1)
          │
          ▼  σ_NL (learned nonlinear stress)
          +
  σ_Linear = softplus(E₂) · ε   (physics-based linear term)
          │
          ▼
  σ_pred = σ_NL + σ_Linear
          │
          ▼
  Physics Residual: r(t) = dσ_NL/dt + σ_NL/λ_eff − G_eff · dε/dt
          │
          ▼
  Augmented Lagrangian Loss:
  L_AL = L_data + Σ M_i·r_i + (ρ/2)·Σ r_i²
          │
          ▼
  AdamW Optimizer + Gradient Clipping + LR Scheduling
```

---

## 📁 Repository Structure

```
Digital-Rheology-PI-LSTM/
├── README.md
├── requirements.txt
├── src/
│   ├── pi_lstm_model.py         # PI-LSTM architecture (Burgers + LSTM)
│   ├── burgers_model.py         # Burgers constitutive equation
│   ├── augmented_lagrangian.py  # AL optimization framework
│   ├── train.py                 # Training loop (70 epochs, early stopping ep.67)
│   ├── evaluate.py              # R², RMSE, REC curve, stress distribution
│   └── dataset.py               # LAOS dataset loader (Choi & Rogers)
├── notebooks/
│   └── PI_LSTM_Rheology.ipynb
├── results/
│   ├── validation_loss.png      # Convergence over 70 epochs
│   ├── stress_strain_lissajous.png
│   ├── residual_distribution.png
│   ├── rec_curve.png
│   └── metrics.json             # R²=0.9819, RMSE
├── images/
│   └── pi_lstm_architecture.png
└── LICENSE
```

---

## ⚙️ Methods

### Dataset
- **Source:** Choi & Rogers — thixotropic fumed silica suspension (R972, Evonik)
- **Medium:** Paraffin oil + low-MW polybutene (69:31 wt%)
- **Concentration:** 2.9 vol% fumed silica
- **Rheometer:** ARES-G2 (TA Instruments), cone-plate geometry (40mm, 2°), 20°C
- **Protocol:** LAOS (Large Amplitude Oscillatory Shear)

### Neural Architecture

| Component | Specification |
|-----------|--------------|
| Input sequence length | 64 (stride 32) |
| LSTM layers | 2 unidirectional |
| LSTM hidden units | 128 per layer |
| FC1 | 128→128, SiLU activation, dropout=0.2 |
| FC2 | 128→1 (σ_NL output) |
| Physics term | σ_Linear = softplus(E₂)·ε |

### Burgers Constitutive Model

```
σ + (E₁/η₁ + E₂/η₂ + E₂/η₁)·σ̇ + (E₁E₂/η₁η₂)·σ̈ = η₁ε̇ + (E₂η₁/η₁η₂)·ε̈
```

Parameters E₁, E₂, η₁, η₂ optimized to match experimental LAOS response.

### Loss Function

```
L_AL = L_data + Σ M_i·r_i + (ρ/2)·Σ r_i²

where:
  L_data = (1/N) Σ w_i(σ̂_i - σ_i)²,   w_i = 1/(1 + σ_i²)
  r(t)   = dσ_NL/dt + σ_NL/λ_eff − G_eff·dε/dt
  M_i updated: M_{k+1} = M_k + ρ·r_i
```

### Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Epochs | 70 (early stopping @ epoch 67) |
| Gradient clipping | ✅ |
| LR scheduling | ✅ |

---

## 📊 Results

### Performance Metrics

| Metric | PI-LSTM (Proposed) | Baseline LSTM |
|--------|--------------------|---------------|
| **R²** | **0.9819** | 0.91 |
| RMSE | Lower | Higher |
| Hysteresis loop capture | ✅ Accurate | ❌ Partial |
| Peak stress reproduction | ✅ | ❌ |
| Physics constraint satisfied | ✅ | ❌ |

### Key Findings
- Validation loss converges stably — early stopping triggered at epoch **67**
- Residual distribution tightly centered near zero — physics constraints respected
- **R² = 0.9819** on max-fidelity LAOS test set
- REC curve shows >90% of predictions within 0.5 Pa absolute error tolerance

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python src/train.py --epochs 70 --hidden 128 --lr 1e-3
python src/evaluate.py --model_path models/pi_lstm.pth
```

---

## 📦 requirements.txt

```
torch==2.0.1
numpy==1.24.3
scipy==1.11.1
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
tqdm==4.65.0
```

---

## 👥 Authors

| Name | Affiliation |
|------|------------|
| Deepak Skandh | Amrita Vishwa Vidyapeetham |
| **Nithin S** | **Amrita Vishwa Vidyapeetham** |
| Akhillesh Varathan | Amrita Vishwa Vidyapeetham |
| Kavin | Amrita Vishwa Vidyapeetham |
| Neelesh Ashok (Supervisor) | neelesh@cb.amrita.edu |

---

## 📚 Citation

```bibtex
@article{skandh2026pi_lstm,
  title   = {Digital Rheology of Polymers Using Physics-Informed LSTM Approach},
  author  = {Skandh, Deepak and Nithin and Varathan, Akhillesh and Kavin and Ashok, Neelesh},
  journal = {Journal Not Specified (MDPI)},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## 📜 Key References
1. Raissi et al. Physics-Informed Neural Networks. *J. Comput. Phys.*, 2019.
2. Hochreiter & Schmidhuber. Long Short-Term Memory. *Neural Comput.*, 1997.
3. Choi & Rogers. Thixotropic fumed silica LAOS dataset, 2025.
4. Boyd et al. ADMM Distributed Optimization. *Found. Trends ML*, 2011.

---

<div align="center">
📍 Amrita Vishwa Vidyapeetham, Coimbatore, Tamil Nadu &nbsp;|&nbsp; Manuscript In Preparation (MDPI)
</div>
