"""Coordinates for numeric repair evidence; loss and head scales stay separate."""
import torch

POLICIES = ("global_v1", "robust_local_v1")
FLOOR = .1

def repair_scale(std):
    return torch.sqrt(std.square()+FLOOR**2)

def numeric_features(values, mean, std, policy, *, extra_axis=False):
    if policy not in POLICIES:
        raise ValueError("unsupported repair coordinates")
    tail=(None,)*(values.ndim-std.ndim)
    scale=std[(...,)+tail]
    if policy == "global_v1":
        return values*scale+mean[(...,)+tail]
    return values*scale/repair_scale(std)[(...,)+tail]
