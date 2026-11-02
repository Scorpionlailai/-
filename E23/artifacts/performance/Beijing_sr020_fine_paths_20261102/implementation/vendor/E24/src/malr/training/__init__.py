from .checkpoints import load_checkpoint, save_checkpoint
from .online_checkpoint import load_online_checkpoint, save_online_checkpoint
from .unified_objective import UnifiedStage1Loss, residual_injection_weight, unified_stage1_loss
from .losses import (
    Stage1Loss,
    forecast_mse,
    imputation_mae,
    imputation_mse,
    latent_recovery_mse,
)

__all__ = [
    "Stage1Loss",
    "forecast_mse",
    "imputation_mse",
    "imputation_mae",
    "latent_recovery_mse",
    "load_checkpoint",
    "save_checkpoint",
    "load_online_checkpoint",
    "save_online_checkpoint",
    "UnifiedStage1Loss",
    "residual_injection_weight",
    "unified_stage1_loss",
    "OnlineTrainer",
    "OnlineTrainingResult",
]


def __getattr__(name: str):
    if name in {"OnlineTrainer", "OnlineTrainingResult"}:
        from .online_trainer import OnlineTrainer, OnlineTrainingResult

        return {"OnlineTrainer": OnlineTrainer, "OnlineTrainingResult": OnlineTrainingResult}[name]
    raise AttributeError(name)
