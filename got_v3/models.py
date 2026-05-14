from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class ScalarTokenEmbedding(nn.Module):
    """Embeds a sequence of scalar coefficients into transformer token vectors."""

    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.scalar_proj = nn.Linear(1, d_model)
        self.pos_emb     = nn.Embedding(n_tokens, d_model)
        self.type_emb    = nn.Embedding(2, d_model)   # 0 = a-coeff, 1 = b-coeff

    def forward(self, an: torch.Tensor, bn: torch.Tensor) -> torch.Tensor:
        """
        an: (B, A)
        bn: (B, Bn)
        returns: (B, A+Bn, d_model)
        """
        x = torch.cat([an, bn], dim=1)        # (B, T)
        B, T = x.shape
        A    = an.shape[1]

        tok = self.scalar_proj(x.unsqueeze(-1))   # (B, T, d)

        pos_ids  = torch.arange(T, device=x.device)
        pos      = self.pos_emb(pos_ids).unsqueeze(0)   # (1, T, d)

        type_ids = torch.cat([
            torch.zeros(A,     device=x.device, dtype=torch.long),
            torch.ones(T - A,  device=x.device, dtype=torch.long),
        ])
        typ = self.type_emb(type_ids).unsqueeze(0)      # (1, T, d)

        return tok + pos + typ


class GeometricBottleneck(nn.Module):
    """Projects CLS hidden state h → geometric latent z ∈ ℝ^k."""

    def __init__(self, d_model: int, k: int):
        super().__init__()
        self.to_z = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, k),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.to_z(h)


class GOTv3(nn.Module):
    """
    Geometry-first Geometric Operator Autoencoder.

    Architecture
    ────────────
    Encoder:  [an, bn] tokens  →  Transformer  →  h ∈ ℝ^d_model
    Bottleneck:               h  →  z ∈ ℝ^k   (geometric latent)
    Decoder:                  z  →  [an, bn] reconstruction
    Heads on z:
        component_head   →  K mixture logits
    Heads on (h ‖ z):
        delta_head       →  log10|s·v − ζ(3)| prediction
        conv_head        →  convergence rate
        plateau_head     →  plateau risk (BCE)
    Operator field (z ‖ Δcoeff) → Δz  (linearises operator action)
    """

    def __init__(
        self,
        n_a:          int,
        n_b:          int,
        k:            int   = 3,
        d_model:      int   = 128,
        n_heads:      int   = 8,
        n_layers:     int   = 4,
        dropout:      float = 0.1,
        n_components: int   = 20,
    ):
        super().__init__()
        self.n_a      = n_a
        self.n_b      = n_b
        self.n_tokens = n_a + n_b
        self.k        = k
        self.d_model  = d_model

        self.token_emb = ScalarTokenEmbedding(self.n_tokens, d_model)
        self.cls       = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= 4 * d_model,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                enable_nested_tensor=False)
        self.bottleneck = GeometricBottleneck(d_model, k)

        # Decoder: z → reconstructed coefficients
        self.decoder = nn.Sequential(
            nn.Linear(k, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, self.n_tokens),
        )

        # Geometric head: component classification from z only
        self.component_head = nn.Sequential(
            nn.LayerNorm(k),
            nn.Linear(k, 64), nn.GELU(),
            nn.Linear(64, n_components),
        )

        # Arithmetic heads: use full context (h ‖ z)
        hz = d_model + k
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hz),
            nn.Linear(hz, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.conv_head = nn.Sequential(
            nn.LayerNorm(hz),
            nn.Linear(hz, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.plateau_head = nn.Sequential(
            nn.LayerNorm(hz),
            nn.Linear(hz, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Operator field: (z, Δcoeff) → predicted Δz
        self.op_field = nn.Sequential(
            nn.Linear(k + self.n_tokens, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, k),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def freeze_encoder(self):
        """Freeze token_emb + CLS token + encoder + bottleneck for Phase 2."""
        for p in self.token_emb.parameters():
            p.requires_grad_(False)
        self.cls.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.bottleneck.parameters():
            p.requires_grad_(False)
        n_frozen    = sum(1 for p in self.parameters() if not p.requires_grad)
        n_trainable = sum(1 for p in self.parameters() if p.requires_grad)
        print(f"[model] encoder frozen — {n_frozen} frozen, {n_trainable} trainable params")

    # ------------------------------------------------------------------
    def encode(self, an: torch.Tensor, bn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x   = self.token_emb(an, bn)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.encoder(x)
        h   = x[:, 0]
        z   = self.bottleneck(h)
        return h, z

    def forward(self, an: torch.Tensor, bn: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, z = self.encode(an, bn)
        hz   = torch.cat([h, z], dim=1)
        return {
            "h":                h,
            "z":                z,
            "coeff_hat":        self.decoder(z),
            "component_logits": self.component_head(z),
            "delta_pred":       self.delta_head(hz).squeeze(-1),
            "conv_pred":        self.conv_head(hz).squeeze(-1),
            "plateau_logit":    self.plateau_head(hz).squeeze(-1),
        }

    def predict_op_vector(self, z: torch.Tensor, delta_coeff: torch.Tensor) -> torch.Tensor:
        """Predict Δz = F_op(z, Δcoeff)."""
        return self.op_field(torch.cat([z, delta_coeff], dim=1))
