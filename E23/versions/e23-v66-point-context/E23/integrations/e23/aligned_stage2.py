"""Common downstream protocol matching the cached author T1 P4 forecast stage."""
import random
import numpy as np
import torch


def aligned_identity(enabled: bool) -> dict:
    if type(enabled) is not bool:
        raise ValueError("common Stage2 flag must be boolean")
    return dict(downstream_protocol="t1_mse_v1", forecast_padding="replicate_last_value_v1",
                stage2_rng_reset_before_training=True) if enabled else {}


def reset_training_rng(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
