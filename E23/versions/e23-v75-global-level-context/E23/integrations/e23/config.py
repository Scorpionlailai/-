"""E23-only settings and catalog-checked P4 inputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
import os

import yaml

from integrations.forecast_compare.protocol import P4Protocol, load_p4_protocol
from malr.migration import MigrationCatalog


@dataclass(frozen=True)
class E23Settings:
    config_sha256: str
    d_model: int
    historical_topk: int
    historical_threshold: float
    variable_topk: int
    variable_offsets: tuple[int, ...]
    temporal_local_radius: int
    temporal_boundaries: tuple[int, ...]
    residual_scale: float
    retrieval_loss_weight: float
    specialist_exposure: float
    batch_size: int
    seed: int
    stage1_epochs: int
    stage2_epochs: int
    stage1_lr: float
    stage2_lr: float
    weight_decay: float
    historical_top_p: float | None = None
    variable_top_p: float | None = None
    historical_candidate_cap: int = 8
    variable_selection: str = "global_topk"
    variable_aggregation: str = "scalar_v1"
    historical_retrieval: str = "one_group"
    temporal_selection: str = "banded_v1"
    post_fusion_refiner: str = "none"
    multihead_router: str = "none"
    stage1_objective: str = "window_smooth_l1"
    recovery_schedule: str = "parallel_v1"
    recovery_rounds: int = 1
    recovery_layers: int = 1
    expert_groups: int = 1
    activation_checkpointing: bool = False
    code_freeze_sha256: str | None = None
    point_refiner_span: int = 0
    repair_stride: int = 8
    point_feature_width: int = 0
    repair_patch_length: int = 16
    historical_query_length: int = 0
    point_refiner_arch: str = "legacy"
    front_context: str = "none"
    point_router: str = "none"
    overlap_fusion: str = "none"
    stage1_structure: str = "none"
    point_candidate_readout: str = "none"
    variable_calibration: str = "none"
    front_variable_context: str = "none"
    repair_coordinates: str = "global_v1"
    point_variable_readout: str = "none"
    historical_key_policy: str = "legacy_v1"
    recent_mean_projection: str = "none"
    temporal_evidence_readout: str = "none"
    repair_input_scaling: str = "legacy_v1"
    point_context_readout: str = "none"
    stage1_gap_structure: str = "none"
    front_level_context: str = "none"

    @property
    def variable_candidate_limit(self) -> int:
        count = self.variable_topk if self.variable_top_p is None else 6
        return count * len(self.variable_offsets)

    @property
    def temporal_candidate_limit(self) -> int:
        return 11 if self.temporal_selection == "visible_attention_v1" else 2 + 2 * len(self.temporal_boundaries)


_MODEL = frozenset({
    "d_model", "historical_topk", "historical_threshold", "variable_topk",
    "variable_offsets", "temporal_local_radius", "temporal_boundaries", "residual_scale",
})
_MODEL_TOPP = _MODEL | {"historical_top_p", "variable_top_p", "historical_candidate_cap"}
_MODEL_WINDOW = _MODEL | {"variable_selection"}
_MODEL_CHANNEL_WINDOW = _MODEL_WINDOW | {"variable_aggregation"}
_MODEL_H = _MODEL | {"historical_retrieval"}
_MODEL_H_WINDOW = _MODEL_WINDOW | {"historical_retrieval"}
_MODEL_T_WINDOW = _MODEL_WINDOW | {"temporal_selection"}
_MODEL_POST_WINDOW = _MODEL_WINDOW | {"post_fusion_refiner"}
_MODEL_HEAD_WINDOW = _MODEL_WINDOW | {"multihead_router"}
_MODEL_ITERATIVE_WINDOW = _MODEL_WINDOW | {"recovery_schedule", "recovery_rounds"}
_MODEL_FEATURE_WINDOW = _MODEL_ITERATIVE_WINDOW | {"recovery_layers"}
_MODEL_POINT_WINDOW = _MODEL_ITERATIVE_WINDOW | {"point_refiner_span"}
_MODEL_FRONT_WINDOW = _MODEL_POINT_WINDOW | {"front_context"}
_MODEL_EVIDENCE_WINDOW = _MODEL_FRONT_WINDOW | {"point_router", "overlap_fusion"}
_MODEL_CANDIDATE_WINDOW = _MODEL_EVIDENCE_WINDOW | {"point_candidate_readout"}
_MODEL_CALIBRATION_WINDOW = _MODEL_CANDIDATE_WINDOW | {"variable_calibration"}
_MODEL_SIGNATURE_WINDOW = _MODEL_CALIBRATION_WINDOW | {"front_variable_context"}
_MODEL_COORDINATES_WINDOW = _MODEL_SIGNATURE_WINDOW | {"repair_coordinates"}
_MODEL_VARIABLE_READOUT_WINDOW = _MODEL_COORDINATES_WINDOW | {"point_variable_readout"}
_MODEL_KEY_POLICY_WINDOW = _MODEL_VARIABLE_READOUT_WINDOW | {"historical_key_policy"}
_MODEL_RECENT_MEAN_WINDOW = _MODEL_KEY_POLICY_WINDOW | {"recent_mean_projection"}
_MODEL_TEMPORAL_EVIDENCE_WINDOW = _MODEL_RECENT_MEAN_WINDOW | {"temporal_evidence_readout"}
_MODEL_CORE_SCALE_WINDOW = _MODEL_TEMPORAL_EVIDENCE_WINDOW | {"repair_input_scaling"}
_MODEL_POINT_CONTEXT_WINDOW = _MODEL_CORE_SCALE_WINDOW | {"point_context_readout"}
_MODEL_LEVEL_WINDOW = _MODEL_POINT_CONTEXT_WINDOW | {"front_level_context"}
_MODEL_NONOVERLAP_WINDOW = _MODEL_POINT_WINDOW | {"repair_stride", "point_feature_width"}
_MODEL_HYBRID_WINDOW = _MODEL_NONOVERLAP_WINDOW | {"repair_patch_length", "historical_query_length", "point_refiner_arch"}
_MODEL_STACKED_WINDOW = _MODEL_ITERATIVE_WINDOW | {"recovery_layers", "expert_groups", "activation_checkpointing"}
_TRAINING = frozenset({
    "retrieval_loss_weight", "specialist_exposure", "batch_size", "seed",
    "stage1_epochs", "stage2_epochs", "stage1_lr", "stage2_lr", "weight_decay",
})


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in {int, float} or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def load_e23_settings(path: Path, *, protocol: P4Protocol) -> E23Settings:
    """Parse settings without granting this file authority over the P4 protocol."""
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    document = yaml.safe_load(raw)
    if not isinstance(document, dict) or set(document) != {"model", "training"}:
        raise ValueError("E23 settings contain unknown data, mask or target override")
    model, training = document["model"], document["training"]
    if not isinstance(model, dict) or set(model) not in (
        _MODEL, _MODEL_TOPP, _MODEL_WINDOW, _MODEL_CHANNEL_WINDOW, _MODEL_H, _MODEL_H_WINDOW,
        _MODEL_T_WINDOW,
        _MODEL_POST_WINDOW,
        _MODEL_HEAD_WINDOW,
        _MODEL_ITERATIVE_WINDOW, _MODEL_STACKED_WINDOW, _MODEL_FEATURE_WINDOW, _MODEL_POINT_WINDOW,
        _MODEL_NONOVERLAP_WINDOW, _MODEL_HYBRID_WINDOW, _MODEL_FRONT_WINDOW, _MODEL_EVIDENCE_WINDOW,
        _MODEL_CANDIDATE_WINDOW,
        _MODEL_CALIBRATION_WINDOW,
        _MODEL_SIGNATURE_WINDOW, _MODEL_COORDINATES_WINDOW, _MODEL_VARIABLE_READOUT_WINDOW, _MODEL_KEY_POLICY_WINDOW, _MODEL_RECENT_MEAN_WINDOW, _MODEL_TEMPORAL_EVIDENCE_WINDOW, _MODEL_CORE_SCALE_WINDOW, _MODEL_POINT_CONTEXT_WINDOW, _MODEL_LEVEL_WINDOW,
    ):
        raise ValueError("unknown or missing E23 model setting")
    if not isinstance(training, dict) or set(training) not in (
        _TRAINING, _TRAINING | {"stage1_objective"}, _TRAINING | {"stage1_objective", "stage1_structure"},
        _TRAINING | {"stage1_objective", "stage1_structure", "stage1_gap_structure"},
    ):
        raise ValueError("unknown or missing E23 training setting")
    settings = E23Settings(
        config_sha256=sha256(raw).hexdigest(),
        d_model=_integer(model["d_model"], "d_model"),
        historical_topk=_integer(model["historical_topk"], "historical_topk"),
        historical_threshold=_number(model["historical_threshold"], "historical_threshold"),
        variable_topk=_integer(model["variable_topk"], "variable_topk"),
        variable_offsets=tuple(_integer(x, "variable_offsets") for x in model["variable_offsets"]),
        temporal_local_radius=_integer(model["temporal_local_radius"], "temporal_local_radius"),
        temporal_boundaries=tuple(_integer(x, "temporal_boundaries") for x in model["temporal_boundaries"]),
        residual_scale=_number(model["residual_scale"], "residual_scale"),
        retrieval_loss_weight=_number(training["retrieval_loss_weight"], "retrieval_loss_weight"),
        specialist_exposure=_number(training["specialist_exposure"], "specialist_exposure"),
        batch_size=_integer(training["batch_size"], "batch_size"),
        seed=_integer(training["seed"], "seed"),
        stage1_epochs=_integer(training["stage1_epochs"], "stage1_epochs"),
        stage2_epochs=_integer(training["stage2_epochs"], "stage2_epochs"),
        stage1_lr=_number(training["stage1_lr"], "stage1_lr"),
        stage2_lr=_number(training["stage2_lr"], "stage2_lr"),
        weight_decay=_number(training["weight_decay"], "weight_decay"),
        historical_top_p=(_number(model["historical_top_p"], "historical_top_p")
                          if "historical_top_p" in model else None),
        variable_top_p=(_number(model["variable_top_p"], "variable_top_p")
                        if "variable_top_p" in model else None),
        historical_candidate_cap=(_integer(model["historical_candidate_cap"], "historical_candidate_cap")
                                  if "historical_candidate_cap" in model else 8),
        variable_selection=model.get("variable_selection", "global_topk"),
        variable_aggregation=model.get("variable_aggregation", "scalar_v1"),
        historical_retrieval=model.get("historical_retrieval", "one_group"),
        temporal_selection=model.get("temporal_selection", "banded_v1"),
        post_fusion_refiner=model.get("post_fusion_refiner", "none"),
        multihead_router=model.get("multihead_router", "none"),
        stage1_objective=training.get("stage1_objective", "window_smooth_l1"),
        recovery_schedule=model.get("recovery_schedule", "parallel_v1"),
        recovery_rounds=_integer(model.get("recovery_rounds", 1), "recovery_rounds"),
        recovery_layers=_integer(model.get("recovery_layers", 1), "recovery_layers"),
        expert_groups=_integer(model.get("expert_groups", 1), "expert_groups"),
        activation_checkpointing=model.get("activation_checkpointing", False),
        point_refiner_span=_integer(model.get("point_refiner_span", 0), "point_refiner_span"),
        repair_stride=_integer(model.get("repair_stride", 8), "repair_stride"),
        point_feature_width=_integer(model.get("point_feature_width", 0), "point_feature_width"),
        repair_patch_length=_integer(model.get("repair_patch_length", 16), "repair_patch_length"),
        historical_query_length=_integer(model.get("historical_query_length", 0), "historical_query_length"),
        point_refiner_arch=model.get("point_refiner_arch", "legacy"),
        front_context=model.get("front_context", "none"),
        point_router=model.get("point_router", "none"),
        overlap_fusion=model.get("overlap_fusion", "none"),
        stage1_structure=training.get("stage1_structure", "none"),
        stage1_gap_structure=training.get("stage1_gap_structure", "none"),
        point_candidate_readout=model.get("point_candidate_readout", "none"),
        variable_calibration=model.get("variable_calibration", "none"),
        front_variable_context=model.get("front_variable_context", "none"),
        repair_coordinates=model.get("repair_coordinates", "global_v1"),
        point_variable_readout=model.get("point_variable_readout", "none"),
        historical_key_policy=model.get("historical_key_policy", "legacy_v1"),
        recent_mean_projection=model.get("recent_mean_projection", "none"),
        temporal_evidence_readout=model.get("temporal_evidence_readout", "none"),
        repair_input_scaling=model.get("repair_input_scaling", "legacy_v1"),
        point_context_readout=model.get("point_context_readout", "none"),
        front_level_context=model.get("front_level_context", "none"),
        code_freeze_sha256=(sha256(Path(os.environ["MALR_E23_CODE_FREEZE"]).read_bytes()).hexdigest()
                            if os.environ.get("MALR_E23_CODE_FREEZE") else None),
    )
    if settings.stage1_gap_structure not in ("none", "long_gap_moments_v1"):
        raise ValueError("unsupported Stage1 gap structure")
    if settings.stage1_gap_structure != "none" and settings.stage1_objective != "global_mse_v1":
        raise ValueError("long-gap supervision requires global Stage1 MSE")
    if settings.recovery_schedule not in ("parallel_v1", "iterative_tv_h_v1", "stacked_parallel_tv_h_v1", "feature_tv_h_v1"):
        raise ValueError("unsupported E23 recovery schedule")
    if settings.front_context not in ("none", "window_cnn_v1"):
        raise ValueError("unsupported front context")
    if settings.point_router not in ("none", "pointwise_v1") or settings.overlap_fusion not in ("none", "confidence_v1"):
        raise ValueError("unsupported point evidence policy")
    if settings.repair_input_scaling not in ("legacy_v1", "all_floor_v1"):
        raise ValueError("unsupported repair input scaling")
    if settings.repair_input_scaling != "legacy_v1" and (settings.front_context != "window_cnn_v1" or settings.repair_coordinates != "robust_local_v1" or settings.point_candidate_readout != "eight_head_v1"):
        raise ValueError("full repair scale requires original CNN/readout and robust coordinates")
    if settings.temporal_evidence_readout not in ("none", "visible_cross_attention_v1"):
        raise ValueError("unsupported temporal evidence readout")
    if settings.temporal_evidence_readout != "none" and (settings.front_context != "window_cnn_v1" or settings.point_candidate_readout != "eight_head_v1" or settings.repair_coordinates != "robust_local_v1"):
        raise ValueError("temporal evidence requires original CNN/readout and robust coordinates")
    if settings.recent_mean_projection not in ("none", "mean_v1"):
        raise ValueError("unsupported recent mean projection")
    if settings.recent_mean_projection != "none" and (settings.point_candidate_readout != "eight_head_v1" or settings.front_context != "window_cnn_v1" or settings.stage1_structure != "recent_mean_v2"):
        raise ValueError("recent mean projection requires candidate/window and mean supervision")
    if settings.stage1_structure == "recent_mean_v2" and settings.recent_mean_projection == "none":
        raise ValueError("recent mean supervision requires mean projector")
    if settings.stage1_structure not in ("none", "recent_state_v1", "recent_mean_v2"):
        raise ValueError("unsupported Stage1 structure loss")
    if settings.point_candidate_readout not in ("none", "eight_head_v1"):
        raise ValueError("unsupported point candidate readout")
    if settings.point_candidate_readout != "none" and settings.front_context != "window_cnn_v1":
        raise ValueError("point candidate readout requires window context")
    if settings.variable_calibration not in ("none", "identity_scale_v1"):
        raise ValueError("unsupported variable calibration")
    if settings.historical_key_policy not in ("legacy_v1", "global_diverse_v1"):
        raise ValueError("unsupported historical key policy")
    if settings.historical_key_policy != "legacy_v1" and (settings.historical_retrieval != "one_group" or settings.recovery_schedule != "iterative_tv_h_v1" or settings.front_context != "window_cnn_v1"):
        raise ValueError("global diverse keys require original contextual history")
    if settings.point_variable_readout not in ("none", "variable_conditioned_v1"):
        raise ValueError("unsupported point variable readout")
    if settings.point_variable_readout != "none" and (settings.point_candidate_readout != "eight_head_v1" or settings.point_refiner_span != 32 or settings.front_context != "window_cnn_v1"):
        raise ValueError("variable readout requires original candidate/tail32 model")
    if settings.repair_coordinates not in ("global_v1", "robust_local_v1"):
        raise ValueError("unsupported repair coordinates")
    if settings.repair_coordinates != "global_v1" and (settings.front_context != "window_cnn_v1" or settings.variable_calibration != "none"):
        raise ValueError("local repair coordinates require original window CNN without V calibration")
    if settings.point_context_readout not in ("none", "direct_point_v1", "segment_attention_v1", "segment_attention_pre_tail_v2"):
        raise ValueError("unsupported point context readout")
    if settings.point_context_readout != "none" and (settings.front_context != "window_cnn_v1" or settings.repair_coordinates != "robust_local_v1"):
        raise ValueError("point context requires robust visible window CNN")
    if settings.front_level_context not in ("none", "global_level_v1"):
        raise ValueError("unsupported front level context")
    if settings.front_level_context != "none" and (settings.front_context != "window_cnn_v1" or settings.repair_coordinates != "robust_local_v1" or settings.front_variable_context != "signature_plain_pos_v3"):
        raise ValueError("global level context requires robust local positional signature CNN")
    if settings.front_variable_context not in ("none", "signature_attention_v1", "signature_plain_v2", "signature_plain_pos_v3"):
        raise ValueError("unsupported front variable context")
    if settings.front_variable_context != "none" and settings.front_context != "window_cnn_v1":
        raise ValueError("front variable context requires window CNN")
    if settings.variable_calibration != "none" and (settings.variable_selection != "window_context_topk_v1"
            or settings.recovery_schedule != "iterative_tv_h_v1" or settings.front_context != "window_cnn_v1"):
        raise ValueError("variable calibration requires original front/contextual V")
    if (settings.point_router != "none" or settings.overlap_fusion != "none" or settings.stage1_structure != "none") and settings.front_context != "window_cnn_v1":
        raise ValueError("point series requires v46 front context")
    if settings.front_context != "none" and (
        settings.d_model != 64 or settings.repair_patch_length != 16 or settings.repair_stride != 8
        or settings.point_refiner_span != 32 or settings.point_refiner_arch != "legacy"
        or settings.recovery_schedule != "iterative_tv_h_v1" or settings.recovery_rounds != 1):
        raise ValueError("front context requires original v41-slow repair geometry")
    hybrid = settings.point_refiner_arch in ("hybrid_v1", "cnn_v1", "none_v1")
    expected_span = 0 if settings.point_refiner_arch == "none_v1" else 32
    if hybrid:
        if (settings.repair_patch_length not in (4,8,16) or settings.repair_stride != settings.repair_patch_length
                or settings.historical_query_length != 16 or settings.point_feature_width != 0
                or settings.point_refiner_span != expected_span or settings.recovery_schedule != "iterative_tv_h_v1"):
            raise ValueError("invalid hybrid fine geometry")
    elif settings.point_refiner_arch != "legacy" or settings.repair_patch_length != 16 or settings.historical_query_length:
        raise ValueError("unknown fine repair configuration")
    if (not hybrid and settings.repair_stride not in (8,16)) or settings.point_feature_width not in (0,32):
        raise ValueError("unsupported nonoverlap repair geometry")
    if settings.repair_stride == 16 and (settings.recovery_schedule != "iterative_tv_h_v1"
            or settings.recovery_rounds != 1 or settings.point_refiner_span != expected_span):
        raise ValueError("nonoverlap trial requires single-round tail32 recovery")
    if settings.point_feature_width and settings.repair_stride != 16:
        raise ValueError("point feature expansion requires nonoverlap repair")
    if settings.point_refiner_span not in (0, 32, 96):
        raise ValueError("unsupported point refiner span")
    if settings.point_refiner_span and (settings.recovery_schedule != "iterative_tv_h_v1"
                                        or settings.recovery_rounds != 1):
        raise ValueError("point refiner requires original v31-B recovery")
    stacked = settings.recovery_schedule == "stacked_parallel_tv_h_v1"
    feature_tv = settings.recovery_schedule == "feature_tv_h_v1"
    if type(settings.activation_checkpointing) is not bool:
        raise ValueError("activation_checkpointing must be boolean")
    if stacked and (settings.recovery_layers not in (1,4) or settings.expert_groups not in (1,8)
                    or settings.recovery_rounds != 1):
        raise ValueError("unsupported stacked layer/group geometry")
    if not stacked and not feature_tv and (settings.recovery_layers != 1 or settings.expert_groups != 1 or settings.activation_checkpointing):
        raise ValueError("stacked settings require stacked schedule")
    if feature_tv and (settings.recovery_layers not in (1,2) or settings.recovery_rounds != 1 or settings.expert_groups != 1 or settings.activation_checkpointing):
        raise ValueError("unsupported feature TV/H geometry")
    if settings.recovery_rounds not in (1, 3):
        raise ValueError("unsupported E23 recovery rounds")
    if settings.recovery_schedule == "parallel_v1" and settings.recovery_rounds != 1:
        raise ValueError("parallel recovery requires one round")
    if settings.recovery_schedule in ("iterative_tv_h_v1", "stacked_parallel_tv_h_v1", "feature_tv_h_v1") and (
        settings.d_model != 64 or settings.stage1_objective != "global_mse_v1"
        or settings.variable_selection != "window_context_topk_v1"
        or settings.variable_aggregation != "scalar_v1"
        or settings.historical_retrieval != "one_group" or settings.historical_top_p is not None
        or settings.variable_top_p is not None or settings.temporal_selection != "banded_v1"
        or settings.post_fusion_refiner != "none" or settings.multihead_router != "none"
    ):
        raise ValueError("iterative recovery requires the v28 width-64 global Stage 1 design")
    if settings.stage1_objective not in ("window_smooth_l1", "global_mse_v1"):
        raise ValueError("unsupported E23 Stage 1 objective")
    if settings.stage1_objective == "global_mse_v1" and (
        settings.d_model != 64 or settings.variable_selection != "window_context_topk_v1"
        or settings.variable_aggregation != "scalar_v1"
        or settings.historical_retrieval != "one_group" or settings.historical_top_p is not None
        or settings.temporal_selection != "banded_v1" or settings.post_fusion_refiner != "none"
        or settings.multihead_router != "none"
    ):
        raise ValueError("E23 global Stage 1 objective requires the v12 architecture")
    if (
        settings.d_model not in (16, 64, 128) or settings.historical_topk != 4
        or settings.historical_threshold != 0.7 or settings.variable_topk != 4
        or settings.variable_offsets != ((-16//settings.repair_stride, 0, 16//settings.repair_stride) if hybrid else (-1,0,1))
        or settings.temporal_local_radius != (16//settings.repair_stride if hybrid else (1 if settings.repair_stride == 16 else 2))
        or settings.temporal_boundaries != (tuple(v//settings.repair_stride for v in (16,32,64,128)) if hybrid else ((1,2,4,8) if settings.repair_stride == 16 else (2,4,8,16)))
        or settings.residual_scale != 0.1
        or settings.retrieval_loss_weight != 0.1
        or settings.specialist_exposure != 0.2
        or settings.batch_size != 128 or settings.seed != protocol.training_seed
        or settings.stage1_epochs not in (100, 200) or settings.stage2_epochs != 100
        or settings.stage1_lr != 0.001 or settings.stage2_lr != 0.0001
        or settings.weight_decay != 0.0
    ):
        raise ValueError("E23 settings differ from the locked first-run design")
    if settings.stage1_epochs == 200 and (settings.recovery_schedule != "iterative_tv_h_v1"
            or settings.recovery_rounds != 1 or settings.point_refiner_span not in (0, 32)):
        raise ValueError("increased Stage1 budget requires the original v31-B/tail32 design")
    if settings.historical_top_p is not None and (
        settings.historical_top_p != 0.9 or settings.variable_top_p != 0.9
        or settings.historical_candidate_cap != 8
    ):
        raise ValueError("E23 top-p differs from the locked trial design")
    if settings.variable_selection not in ("global_topk", "window_context_topk_v1",
                                            "window_context_topk_v2", "window_affine_v1"):
        raise ValueError("unsupported E23 variable selection")
    if settings.variable_aggregation not in (
        "scalar_v1", "channelwise_v1", "channelwise_bounded_v1",
    ):
        raise ValueError("unsupported E23 variable aggregation")
    if settings.variable_aggregation != "scalar_v1" and (
        settings.d_model != 64 or settings.variable_selection != "window_context_topk_v1"
        or settings.historical_retrieval != "one_group" or settings.historical_top_p is not None
        or settings.temporal_selection != "banded_v1" or settings.post_fusion_refiner != "none"
        or settings.multihead_router != "none"
    ):
        raise ValueError("E23 channelwise V requires the v12 width-64 design")
    if settings.variable_selection != "global_topk" and (
        settings.d_model not in (64, 128) or settings.historical_top_p is not None
        or (settings.d_model == 128 and settings.variable_selection != "window_context_topk_v1")
    ):
        raise ValueError("E23 contextual V requires supported width and top-k")
    if settings.multihead_router not in (
        "none", "eight_candidate_heads_v1", "eight_full_width_gated_v1",
    ):
        raise ValueError("unsupported E23 multihead router")
    if settings.multihead_router == "eight_candidate_heads_v1" and (
        settings.d_model not in (64, 128) or settings.variable_selection != "window_context_topk_v1"
        or settings.historical_retrieval != "one_group" or settings.historical_top_p is not None
        or settings.temporal_selection != "banded_v1" or settings.post_fusion_refiner != "none"
    ):
        raise ValueError("E23 multihead router requires width 64 or 128 and v12 candidate policies")
    if settings.multihead_router == "eight_full_width_gated_v1" and (
        settings.d_model != 64 or settings.variable_selection != "window_context_topk_v1"
        or settings.historical_retrieval != "one_group" or settings.historical_top_p is not None
        or settings.temporal_selection != "banded_v1" or settings.post_fusion_refiner != "none"
    ):
        raise ValueError("E23 full-width head gate requires width 64 and v12 candidate policies")
    if settings.historical_retrieval not in ("one_group", "two_group_visible_rerank_v1"):
        raise ValueError("unsupported E23 historical retrieval")
    if settings.historical_retrieval != "one_group" and (
        settings.d_model != 64 or settings.historical_top_p is not None
    ):
        raise ValueError("E23 two-group H requires width 64 top-k")
    if settings.temporal_selection not in ("banded_v1", "visible_attention_v1"):
        raise ValueError("unsupported E23 temporal selection")
    if settings.temporal_selection == "visible_attention_v1" and (
        settings.d_model != 64 or settings.variable_selection != "window_context_topk_v1"
        or settings.historical_retrieval != "one_group" or settings.historical_top_p is not None
    ):
        raise ValueError("E23 temporal attention requires the v12 width-64 design")
    if settings.post_fusion_refiner not in ("none", "evidence_factorized_v1"):
        raise ValueError("unsupported E23 post-fusion refiner")
    if settings.post_fusion_refiner == "evidence_factorized_v1" and (
        settings.d_model != 64 or settings.variable_selection != "window_context_topk_v1"
        or settings.temporal_selection != "banded_v1"
        or settings.historical_retrieval != "one_group" or settings.historical_top_p is not None
    ):
        raise ValueError("E23 post-fusion refiner requires the v12 width-64 design")
    if (
        protocol.config.data.seq_len != 96 or protocol.config.data.pred_len != 96
        or protocol.config.patch.length != 16 or protocol.config.patch.stride != 8
        or protocol.config.data.datasets["ETTh1"].n_vars != 7
    ):
        raise ValueError("E23 settings require the locked ETTh1 P4 protocol")
    return settings


def load_e23_p4_protocol(
    *, catalog: MigrationCatalog, protocol_config: Path, mask_artifacts: Path,
    training_seed: int | None = None,
) -> P4Protocol:
    """Load fixed E22 masks from the migration catalog, never E23-local results."""
    root = catalog.paper_root
    config = Path(protocol_config).resolve()
    masks = Path(mask_artifacts).resolve()
    if not config.is_relative_to(root / "E23"):
        raise ValueError("E23 P4 protocol config must come from E23")
    if not masks.is_relative_to(root / "E22/artifacts"):
        raise ValueError("E23 P4 fixed masks must come from E22")
    protocol = load_p4_protocol(config, masks, catalog=catalog)
    if training_seed is not None:
        from integrations.forecast_compare.protocol import with_training_seed
        protocol = with_training_seed(protocol, training_seed)
    return protocol


__all__ = ["E23Settings", "load_e23_settings", "load_e23_p4_protocol"]
