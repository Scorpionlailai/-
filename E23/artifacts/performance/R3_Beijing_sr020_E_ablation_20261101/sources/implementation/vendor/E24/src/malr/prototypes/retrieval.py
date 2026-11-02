from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from malr.data.masks import generate_variable_missing_mask


class MaskAwareRetrievalEncoder(nn.Module):
    """Encode a fixed-length masked patch in the clean–mask retrieval space."""

    def __init__(
        self,
        patch_len: int,
        retrieval_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or retrieval_dim <= 0:
            raise ValueError("patch_len and retrieval_dim must be positive")
        hidden = retrieval_dim if hidden_dim is None else int(hidden_dim)
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        self.patch_len = int(patch_len)
        self.retrieval_dim = int(retrieval_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * patch_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, retrieval_dim),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[-1] != self.patch_len:
            raise ValueError("values must have shape [batch, patch_len]")
        if mask.shape != values.shape:
            raise ValueError("mask must match values")
        masked_values = values * mask.to(dtype=values.dtype)
        encoded = self.network(torch.cat((masked_values, mask.to(values.dtype)), dim=-1))
        return F.normalize(encoded, dim=-1, eps=1e-8)


class RetrievalEncoderEMA:
    """Keep a trainable retrieval encoder and two non-trainable views.

    The EMA view follows the trainable encoder after each optimizer step.  The
    frozen view is copied from EMA only when ``begin_epoch`` is called, so all
    retrieval queries in one epoch use an identical parameter snapshot.
    """

    def __init__(
        self,
        encoder: MaskAwareRetrievalEncoder,
        decay: float = 0.99,
    ) -> None:
        if not isinstance(encoder, MaskAwareRetrievalEncoder):
            raise TypeError("encoder must be a MaskAwareRetrievalEncoder")
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        self.encoder = encoder
        self.ema_encoder = copy.deepcopy(encoder).eval()
        self.frozen_encoder = copy.deepcopy(encoder).eval()
        self.decay = float(decay)
        self._frozen_epoch = 0
        self._freeze(self.ema_encoder)
        self._freeze(self.frozen_encoder)

    @staticmethod
    def _freeze(encoder: nn.Module) -> None:
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

    @property
    def frozen_epoch(self) -> int:
        return self._frozen_epoch

    def begin_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch != self._frozen_epoch + 1:
            raise ValueError("retrieval EMA epochs must begin consecutively")
        self.frozen_encoder.load_state_dict(self.ema_encoder.state_dict(), strict=True)
        self._freeze(self.frozen_encoder)
        self._frozen_epoch = epoch

    @torch.no_grad()
    def update_ema(self) -> None:
        source_parameters = dict(self.encoder.named_parameters())
        ema_parameters = dict(self.ema_encoder.named_parameters())
        if source_parameters.keys() != ema_parameters.keys():
            raise ValueError("retrieval encoder EMA parameter names do not match")
        for name, source in source_parameters.items():
            target = ema_parameters[name]
            target.mul_(self.decay).add_(source.detach(), alpha=1.0 - self.decay)

        source_buffers = dict(self.encoder.named_buffers())
        ema_buffers = dict(self.ema_encoder.named_buffers())
        if source_buffers.keys() != ema_buffers.keys():
            raise ValueError("retrieval encoder EMA buffer names do not match")
        for name, source in source_buffers.items():
            target = ema_buffers[name]
            if target.is_floating_point() or target.is_complex():
                target.mul_(self.decay).add_(source.detach(), alpha=1.0 - self.decay)
            else:
                target.copy_(source)
        self._freeze(self.ema_encoder)

    @torch.no_grad()
    def encode_frozen(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.frozen_encoder(values, mask)

    @torch.no_grad()
    def encode_ema(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.ema_encoder(values, mask)

    @staticmethod
    def _state_mapping(state: object) -> dict[str, Tensor]:
        if not isinstance(state, (tuple, list)):
            raise ValueError("retrieval EMA state must be a sequence")
        mapping: dict[str, Tensor] = {}
        for item in state:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("retrieval EMA state item is invalid")
            name, value = item
            if not isinstance(name, str):
                raise ValueError("retrieval EMA state name must be a string")
            array = np.asarray(value, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError("retrieval EMA state must be finite")
            mapping[name] = torch.from_numpy(np.array(array, copy=True))
        return mapping

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "patch_len": self.encoder.patch_len,
            "retrieval_dim": self.encoder.retrieval_dim,
            "decay": self.decay,
            "frozen_epoch": self._frozen_epoch,
            "ema_state": retrieval_state_tuple(self.ema_encoder),
            "frozen_state": retrieval_state_tuple(self.frozen_encoder),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or state.get("format_version") != 1:
            raise ValueError("unsupported retrieval EMA state version")
        if state.get("patch_len") != self.encoder.patch_len:
            raise ValueError("retrieval EMA patch geometry does not match encoder")
        if state.get("retrieval_dim") != self.encoder.retrieval_dim:
            raise ValueError("retrieval EMA dimension does not match encoder")
        try:
            decay = float(state["decay"])
            frozen_epoch = int(state["frozen_epoch"])
            ema_state = self._state_mapping(state["ema_state"])
            frozen_state = self._state_mapping(state["frozen_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("retrieval EMA state is incomplete") from exc
        if decay != self.decay:
            raise ValueError("retrieval EMA decay does not match manager")
        if frozen_epoch < 0:
            raise ValueError("retrieval EMA frozen epoch must be non-negative")
        try:
            self.ema_encoder.load_state_dict(ema_state, strict=True)
            self.frozen_encoder.load_state_dict(frozen_state, strict=True)
        except RuntimeError as exc:
            raise ValueError("retrieval EMA state does not match encoder") from exc
        self._freeze(self.ema_encoder)
        self._freeze(self.frozen_encoder)
        self._frozen_epoch = frozen_epoch


def normalized_clean_keys(projection: nn.Module, clean_patches: Tensor) -> Tensor:
    """Create detached unit clean keys from the frozen clean patch projection."""

    if clean_patches.ndim != 2:
        raise ValueError("clean_patches must have shape [batch, patch_len]")
    with torch.no_grad():
        projected = projection(clean_patches)
        return F.normalize(projected, dim=-1, eps=1e-8).detach()


def masked_pair_loss(queries: Tensor, clean_keys: Tensor) -> Tensor:
    if queries.shape != clean_keys.shape or queries.ndim != 2:
        raise ValueError("queries and clean_keys must have matching [batch, dim] shape")
    if queries.shape[0] == 0:
        return queries.sum() * 0.0
    return (1.0 - F.cosine_similarity(queries, clean_keys.detach(), dim=-1)).mean()


def masked_contrastive_loss(
    queries: Tensor,
    clean_keys: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    if queries.shape != clean_keys.shape or queries.ndim != 2:
        raise ValueError("queries and clean_keys must have matching [batch, dim] shape")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if queries.shape[0] == 0:
        return queries.sum() * 0.0
    logits = queries @ clean_keys.detach().transpose(0, 1) / temperature
    labels = torch.arange(queries.shape[0], device=queries.device)
    return F.cross_entropy(logits, labels)


def build_clean_patch_table(
    standardized_values: np.ndarray,
    *,
    train_end: int,
    patch_len: int,
    scan_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(standardized_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < train_end or train_end <= 0:
        raise ValueError("standardized_values must contain a valid training interval")
    if patch_len <= 0 or scan_stride <= 0:
        raise ValueError("patch_len and scan_stride must be positive")
    starts = np.arange(0, train_end - patch_len + 1, scan_stride, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("training interval is shorter than patch_len")
    patches = np.stack(
        [
            np.stack(
                [values[start : start + patch_len, variable] for start in starts],
                axis=0,
            )
            for variable in range(values.shape[1])
        ],
        axis=0,
    )
    return np.ascontiguousarray(patches, dtype=np.float32), starts


@dataclass(frozen=True)
class RetrievalTrainingResult:
    encoder: MaskAwareRetrievalEncoder
    clean_keys: np.ndarray
    mean_loss: float
    epochs: int


def retrieval_state_tuple(
    encoder: MaskAwareRetrievalEncoder,
) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple(
        (name, value.detach().cpu().numpy().astype(np.float32, copy=True))
        for name, value in encoder.state_dict().items()
    )


def load_mask_aware_encoder(
    state: tuple[tuple[str, np.ndarray], ...],
    *,
    patch_len: int,
    retrieval_dim: int,
) -> MaskAwareRetrievalEncoder:
    if not state:
        raise ValueError("grouped bank does not contain a retrieval encoder")
    state_mapping = {
        name: torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
        for name, value in state
    }
    first_weight = state_mapping.get("network.0.weight")
    if first_weight is None or first_weight.ndim != 2 or first_weight.shape[1] != 2 * patch_len:
        raise ValueError("retrieval encoder state has incompatible patch geometry")
    encoder = MaskAwareRetrievalEncoder(
        patch_len=patch_len,
        retrieval_dim=retrieval_dim,
        hidden_dim=int(first_weight.shape[0]),
    )
    encoder.load_state_dict(state_mapping, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def train_mask_aware_encoder(
    clean_projection: nn.Module,
    clean_patches: np.ndarray,
    *,
    n_vars: int,
    start_rate: float,
    block_len: int,
    seed: int,
    epochs: int = 20,
    hidden_multiplier: int = 2,
    retrieval_dim: int | None = None,
    batch_size: int = 1024,
    temperature: float = 1.0,
    device: str | torch.device = "cpu",
) -> RetrievalTrainingResult:
    patches = np.asarray(clean_patches, dtype=np.float32)
    if patches.ndim != 3 or patches.shape[0] != n_vars:
        raise ValueError("clean_patches must have shape [n_vars, starts, patch_len]")
    if epochs <= 0 or hidden_multiplier <= 0 or batch_size <= 0:
        raise ValueError("retrieval training sizes must be positive")
    if retrieval_dim is None:
        try:
            retrieval_dim = int(clean_projection.out_features)
        except AttributeError as exc:
            raise ValueError(
                "retrieval_dim is required for projections without out_features"
            ) from exc
    if retrieval_dim <= 0:
        raise ValueError("retrieval_dim must be positive")
    selected_device = torch.device(device)
    clean_projection = clean_projection.to(selected_device).eval()
    for parameter in clean_projection.parameters():
        parameter.requires_grad_(False)
    encoder = MaskAwareRetrievalEncoder(
        patch_len=patches.shape[-1],
        retrieval_dim=retrieval_dim,
        hidden_dim=hidden_multiplier * retrieval_dim,
    ).to(selected_device)
    clean_tensor = torch.from_numpy(
        patches.reshape(-1, patches.shape[-1])
    ).to(selected_device)
    with torch.no_grad():
        clean_keys_tensor = normalized_clean_keys(clean_projection, clean_tensor)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    starts_count = patches.shape[1]
    train_length = starts_count + patches.shape[-1] - 1
    losses: list[float] = []
    for epoch in range(epochs):
        timeline_mask, _ = generate_variable_missing_mask(
            train_length,
            n_vars,
            start_rate,
            block_len,
            seed + epoch,
            dataset="train",
            split="train",
            support_start=0,
        )
        mask_table = np.stack(
            [
                np.stack(
                    [
                        timeline_mask[start : start + patches.shape[-1], variable]
                        for start in range(starts_count)
                    ],
                    axis=0,
                )
                for variable in range(n_vars)
            ],
            axis=0,
        )
        value_tensor = torch.from_numpy(
            (patches * mask_table.astype(np.float32)).reshape(-1, patches.shape[-1])
        ).to(selected_device)
        mask_tensor = torch.from_numpy(
            mask_table.reshape(-1, patches.shape[-1])
        ).to(selected_device)
        order = torch.randperm(value_tensor.shape[0], device=selected_device)
        encoder.train()
        for offset in range(0, order.numel(), batch_size):
            indices = order[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            query = encoder(
                value_tensor.index_select(0, indices),
                mask_tensor.index_select(0, indices),
            )
            target = clean_keys_tensor.index_select(0, indices)
            loss = masked_pair_loss(query, target) + 0.5 * masked_contrastive_loss(
                query, target, temperature=temperature
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return RetrievalTrainingResult(
        encoder=encoder,
        clean_keys=clean_keys_tensor.detach().cpu().numpy().reshape(
            n_vars, starts_count, retrieval_dim
        ),
        mean_loss=float(np.mean(losses)) if losses else 0.0,
        epochs=epochs,
    )
