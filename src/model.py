"""
model.py
Physics-Informed LSTM (PI-Hybrid) for Polymer Rheology

Architecture:
  Input  : [ε(t), ε̇(t), t] sequences of length 64
  LSTM   : 2-layer, 128 hidden units → LayerNorm → FC(128,SiLU) → σ_NL
  Physics: σ_linear = softplus(E₂) * ε
  Output : σ_pred = σ_NL + σ_linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PIHybridLSTM(nn.Module):
    """
    Physics-Informed Hybrid LSTM model for LAOS stress prediction.

    Combines:
      - Data-driven LSTM encoder for nonlinear stress σ_NL
      - Physics-based linear term σ_linear = softplus(E₂)·ε
      - Residual operator for Burgers constitutive constraint
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        # ── LSTM encoder ──────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

        # ── Nonlinear stress head ─────────────────────────────────────────────
        self.fc1    = nn.Linear(hidden_size, hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.fc2    = nn.Linear(hidden_size, 1)

        # ── Learnable Burgers elastic parameter E₂ ────────────────────────────
        self.log_E2 = nn.Parameter(torch.tensor(0.0))   # softplus ensures > 0

    def forward(self, x):
        """
        Args:
            x : (B, T, 3) — [ε, ε̇, t] sequences

        Returns:
            sigma_pred : (B, T) predicted stress
            sigma_nl   : (B, T) nonlinear component (for residual computation)
        """
        # LSTM sequence encoding
        h, _         = self.lstm(x)                     # (B, T, 128)
        h            = self.layer_norm(h)

        # Nonlinear stress prediction
        feat         = F.silu(self.fc1(h))              # SiLU activation
        feat         = self.drop(feat)
        sigma_nl     = self.fc2(feat).squeeze(-1)       # (B, T)

        # Physics-based linear stress: σ_linear = softplus(E₂) · ε
        E2           = F.softplus(self.log_E2)
        epsilon      = x[..., 0]                        # strain channel
        sigma_linear = E2 * epsilon                     # (B, T)

        sigma_pred   = sigma_nl + sigma_linear

        return sigma_pred, sigma_nl

    def compute_residual(self, sigma_nl: torch.Tensor,
                         epsilon: torch.Tensor,
                         t: torch.Tensor,
                         lambda_eff: float = 1.0,
                         G_eff: float = 1.0) -> torch.Tensor:
        """
        Burgers-type physics residual (central finite difference):
        r(t) = dσ_NL/dt + σ_NL/λ_eff - G_eff * dε/dt

        Args:
            sigma_nl   : (B, T) nonlinear stress predictions
            epsilon    : (B, T) strain values
            t          : (B, T) time values
            lambda_eff : effective relaxation time (learnable future work)
            G_eff      : effective modulus

        Returns:
            residual : (B, T-2) interior residuals (central diff excludes endpoints)
        """
        # Central finite differences (numerically stable, clamped dt)
        dt = (t[:, 2:] - t[:, :-2]).clamp(min=1e-6)

        d_sigma = (sigma_nl[:, 2:] - sigma_nl[:, :-2]) / dt
        d_eps   = (epsilon[:, 2:]  - epsilon[:, :-2])  / dt

        sigma_mid = sigma_nl[:, 1:-1]

        residual = d_sigma + sigma_mid / lambda_eff - G_eff * d_eps
        return residual


if __name__ == '__main__':
    model  = PIHybridLSTM(input_size=3, hidden_size=128, num_layers=2)
    dummy  = torch.randn(8, 64, 3)     # batch=8, seq=64, features=3
    s_pred, s_nl = model(dummy)
    print(f'sigma_pred shape : {s_pred.shape}')    # (8, 64)
    print(f'sigma_nl   shape : {s_nl.shape}')      # (8, 64)

    eps  = dummy[..., 0]
    t    = dummy[..., 2]
    res  = model.compute_residual(s_nl, eps, t)
    print(f'residual   shape : {res.shape}')       # (8, 62)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters : {total_params:,}')
