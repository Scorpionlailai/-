"""Stage2 repair update policies; the predictor and objective stay unchanged."""
import torch


def repair_update_identity(mode: str) -> dict:
    if mode == "joint":
        return {}
    if mode == "frozen_v1":
        return dict(repair_update_policy=mode, repair_learning_rate_scale=0.0,
                    repair_training_mode="eval", repair_parameters_trainable=False)
    if mode == "slow_lr_0p1_v1":
        return dict(repair_update_policy=mode, repair_learning_rate_scale=0.1,
                    repair_training_mode="train", repair_parameters_trainable=True)
    raise ValueError(f"unsupported repair update policy: {mode}")


def build_stage2_optimizer(model, *, mode: str, learning_rate: float,
                           weight_decay: float):
    repair_update_identity(mode)
    if mode == "frozen_v1":
        model.repair.requires_grad_(False)
        model.repair.eval()
    if mode == "slow_lr_0p1_v1":
        repair_ids = {id(p) for p in model.repair.parameters()}
        repair = [p for p in model.repair.parameters() if p.requires_grad]
        other = [p for p in model.parameters() if p.requires_grad and id(p) not in repair_ids]
        return torch.optim.Adam(
            [{"params": repair, "lr": learning_rate * 0.1},
             {"params": other, "lr": learning_rate}],
            lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                            lr=learning_rate, weight_decay=weight_decay)


def stage2_training_mode(model, mode: str):
    repair_update_identity(mode)
    model.train()
    if mode == "frozen_v1":
        model.repair.eval()
