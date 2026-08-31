from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from malr.vendor.patchtst import Flatten_Head, TSTiEncoder


class PatchTSTCore(nn.Module):
    """Official PatchTST encoder/head with an explicit patch-latent entry."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.2,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        norm: str = "BatchNorm",
        res_attention: bool = True,
    ) -> None:
        super().__init__()
        if c_in <= 0:
            raise ValueError("c_in must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if stride <= 0 or patch_len <= 0:
            raise ValueError("patch_len and stride must be positive")
        if not 0.0 <= fc_dropout < 1.0:
            raise ValueError("fc_dropout must be in [0, 1)")
        self.c_in = c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        # Kept in the experiment profile for parity with the official scripts.
        # The official supervised flatten head does not consume fc_dropout.
        self.fc_dropout = float(fc_dropout)
        self.patch_num = 1 + (seq_len + stride - patch_len) // stride
        self.backbone = TSTiEncoder(
            c_in=c_in,
            patch_num=self.patch_num,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act="gelu",
            res_attention=res_attention,
            pe="zeros",
            learn_pe=True,
        )
        self.head = Flatten_Head(
            individual=False,
            n_vars=c_in,
            nf=d_model * self.patch_num,
            target_window=pred_len,
            head_dropout=head_dropout,
        )

    @property
    def W_P(self) -> nn.Linear:
        return self.backbone.W_P

    @property
    def W_pos(self) -> nn.Parameter:
        return self.backbone.W_pos

    @property
    def encoder(self) -> nn.Module:
        return self.backbone.encoder

    def _validate_patches(self, patches: Tensor) -> None:
        expected = (self.c_in, self.patch_num, self.patch_len)
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
            raise ValueError(
                "clean patches must have shape "
                f"[batch, {self.c_in}, {self.patch_num}, {self.patch_len}]"
            )

    def project_clean_patches(self, patches: Tensor) -> Tensor:
        self._validate_patches(patches)
        return self.W_P(patches)

    def encode_latents(self, latents: Tensor) -> Tensor:
        if latents.ndim != 4:
            raise ValueError("latents must have shape [batch, variables, patches, d_model]")
        batch, variables, patch_num, d_model = latents.shape
        if variables != self.c_in:
            raise ValueError(f"expected {self.c_in} variables, got {variables}")
        if patch_num != self.patch_num:
            raise ValueError(f"expected {self.patch_num} patches, got {patch_num}")
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")

        encoded_input = latents.reshape(batch * variables, patch_num, d_model)
        encoded_input = self.backbone.dropout(encoded_input + self.W_pos)
        encoded = self.encoder(encoded_input)
        encoded = encoded.reshape(batch, variables, patch_num, d_model)
        return encoded.permute(0, 1, 3, 2).contiguous()

    def forecast_from_latents(self, latents: Tensor) -> Tensor:
        return self.head(self.encode_latents(latents))

    def forward_clean(self, patches: Tensor) -> Tensor:
        return self.forecast_from_latents(self.project_clean_patches(patches))


def load_clean_initialization(
    model: PatchTSTCore,
    clean_state_dict: Mapping[str, Tensor],
) -> None:
    incompatible = model.load_state_dict(clean_state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "clean checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
