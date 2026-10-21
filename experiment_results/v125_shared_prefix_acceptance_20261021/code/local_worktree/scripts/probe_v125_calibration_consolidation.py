"""Frozen full-validation branch audit; no training or test-set access.

Two separate identities: sealed v125 and the locked seed2023 VC1 warm candidate.
Core probes use one neural pass and no LSV/V postprocessing. Deployed VC1 is
reported separately, never silently treated as the dual-core output.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch
from torch import nn

from integrations.e23.calibration_ensemble import build_calibration_repair
from integrations.e23.direct_variable_transport import materialize_transport_recipe
from integrations.e23.global_only_adaptation import RepairBandMetrics
from integrations.e23.level_shape_pilot_data import iter_pilot_batches
from integrations.e23.no_recent_specialization import remove_recent_specialization
from integrations.e23.train import _new_memory
from malr.evaluation.report import file_sha256
from scripts.run_v125_H_soft_memory_heads import forward as deployed_forward
from scripts.run_v125_level_shape_pilot import forward as core_forward
from scripts.v125_original_pilot_setup import (
    load_original_pilot, build_original_pilot, PARENT, MANIFEST, SELECTION)

PERF = Path("/home/xyw/paper/E23/artifacts/performance")
CANDIDATE = PERF / "v125_no_recent_uniformCNN_validation_candidate_20261020/selected-mae.pt"
RECIPE = PERF / "v125_VC1_mode_supported_variable_transport_repair_candidate_20261020/direct_variable_transport_recipe.json"


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def gap_lengths(hidden):
    """Lengths of contiguous hidden runs, clipped by window edges."""
    t = torch.arange(hidden.shape[1], device=hidden.device)[None, :, None]
    left = torch.where(~hidden, t, -1).cummax(1).values
    right = torch.where(~hidden, t, hidden.shape[1]).flip(1).cummin(1).values.flip(1)
    return torch.where(hidden, right - left - 1, 0)


class GapMetrics:
    def __init__(self, device):
        self.total = torch.zeros(4, 4, dtype=torch.float64, device=device)

    def update(self, error, hidden):
        length = gap_lengths(hidden)
        for i, (lo, hi) in enumerate(((1, 8), (9, 16), (17, 32), (33, 96))):
            active = hidden & (length >= lo) & (length <= hi)
            e = error[active].double()
            self.total[i] += torch.stack((e.square().sum(), e.abs().sum(),
                torch.as_tensor(e.numel(), device=e.device),
                active.flatten(1).any(1).sum()))

    def finalize(self):
        return {name: dict(mse=float(v[0]/v[2]) if v[2] else None,
                           mae=float(v[1]/v[2]) if v[2] else None,
                           count=int(v[2]), windows=int(v[3]))
                for name, v in zip(("1-8", "9-16", "17-32", "33-96"), self.total.cpu())}


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build(bundle, origin, variant, device):
    if origin == "sealed_v125":
        p = bundle["payload"]
        model = build_calibration_repair(bundle["protocol"], bundle["settings"],
            bundle["parent"]["calibration_ensemble_spec"], device=device)
        model.load_state_dict(p["model_state"], strict=True)
        model.historical.retrieval.load_state_dict(p["retrieval_state"])
        memory = _new_memory(bundle["protocol"])
        memory.load_state_dict(p["memory_state"])
        model.historical.refresh_snapshot(memory.snapshot())
    else:
        model = build_original_pilot(bundle, device)
        remove_recent_specialization(model, point_scope="full96")
        model.load_state_dict(torch.load(CANDIDATE, map_location=device,
                                        weights_only=False)["model_state"], strict=True)
        model.H_transport_runtime = "legacy"
        model.H_transport_pairs = materialize_transport_recipe(json.loads(RECIPE.read_text()), device)
    if variant == "coarse":
        model.recent_mean_projector = None
    elif variant in ("single64", "single256"):
        cal = model.recent_mean_projector
        index = 0 if variant == "single64" else 1
        cal.members = nn.ModuleList([cal.members[index]])
        cal.member_widths = (64 if index == 0 else 256,)
        cal.weights = torch.ones(1, dtype=cal.weights.dtype, device=device)
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def profile(model, batch, call, device):
    # Timing excludes data loading/transfer/metric calculation, includes dispatch.
    for _ in range(5):
        call(model, batch, 0)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(20):
        t = perf_counter()
        call(model, batch, 0)
        torch.cuda.synchronize(device)
        timings.append(perf_counter() - t)
    peaks = dict(allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30,
                 reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30)
    # A separate instrumented pass attributes GPU time. Nested rows overlap.
    events, handles, starts = {}, [], {}
    selected = {"": model}
    selected.update(dict(model.named_children()))
    cal = model.recent_mean_projector
    if cal is not None:
        selected.update({f"calibration_width_{w}": m
                         for w, m in zip(cal.member_widths, cal.members)})
    for name, module in selected.items():
        if module is None:
            continue
        def before(m, args, key=name):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            starts.setdefault(key, []).append(event)
        def after(m, args, output, key=name):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events.setdefault(key, []).append((starts[key].pop(), end))
        handles.extend([module.register_forward_pre_hook(before),
                        module.register_forward_hook(after)])
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    call(model, batch, 0)
    end.record()
    torch.cuda.synchronize(device)
    for h in handles:
        h.remove()
    attributed = {name or "whole_neural_network": dict(
        calls=len(pairs), gpu_ms=sum(a.elapsed_time(b) for a, b in pairs))
        for name, pairs in events.items()}
    root_ms = attributed["whole_neural_network"]["gpu_ms"]
    median = float(np.median(timings))
    return dict(batch_size=len(batch.refs), warmups=5, repetitions=20,
        median_batch_seconds=median, p10_batch_seconds=float(np.quantile(timings, .1)),
        p90_batch_seconds=float(np.quantile(timings, .9)),
        windows_per_second=len(batch.refs)/median,
        ms_per_window=median*1000/len(batch.refs), peak_memory=peaks,
        component_gpu_profile=attributed,
        whole_pipeline_gpu_ms=begin.elapsed_time(end),
        outside_neural_forward_gpu_ms=begin.elapsed_time(end)-root_ms,
        component_times_are_nested_not_additive=True)


@torch.no_grad()
def main():
    start = perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed_probe.py")
    def status(**kw):
        write(out/"status.json", dict(pid=os.getpid(), elapsed_seconds=perf_counter()-start, **kw))
    status(status="loading")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    bundle = load_original_pilot()
    protocol = bundle["protocol"]
    indices = dict(validation=np.arange(len(protocol.refs["validation"])))
    assert len(indices["validation"]) == 5175
    np.save(out/"validation_indices.npy", indices["validation"])
    write(out/"identity.json", dict(dataset="Weather", scene="sr015_bl8", seq_len=96,
        manifest_sha256=file_sha256(MANIFEST),
        sealed_selection=str(SELECTION), sealed_selection_sha256=file_sha256(SELECTION),
        sealed_checkpoint_sha256=bundle["parent"]["checkpoint_sha256"],
        AT0_parent=str(PARENT), AT0_sha256=file_sha256(PARENT),
        VC1_warm_candidate=str(CANDIDATE), VC1_warm_sha256=file_sha256(CANDIDATE),
        V_recipe_sha256=file_sha256(RECIPE),
        probe_sha256=file_sha256(Path(__file__)), training_updates=0, test_read=False,
        subset_selection=False, validation_windows=5175,
        interpretation="Frozen output intervention only, NOT independently trained single calibrators",
        coarse_cut="Input to original dual calibrator; includes original point CNN (32 sealed /96 warm)",
        core_policy="One neural pass. No LSV or fixed numerical variable transport in all four core arms",
        deployed_policy="Separate VC1_warm deployed reference retains LSV3 neural passes and fixed V",
        single_branch_policy="Retain entire selected original member, including its level path when present; weight1; preserve atom and no-source/known policies",
        gap_policy="Contiguous valid hidden runs within96, all96 scored. Weather has no native missing target; this script is not a PAMAP2 data adapter",
        parameters_policy="Resident named neural parameters; detached retrieval bank/key storage not counted as trainable parameters",
        GPU=str(device), precision="FP32", batch=128))
    results = {}
    for origin in ("sealed_v125", "VC1_warm_seed2023"):
        for variant in ("coarse", "single64", "single256", "dual", "deployed"):
            if origin == "sealed_v125" and variant == "deployed":
                continue
            key = origin + "/" + variant
            status(status="building", candidate=key)
            model = build(bundle, origin, variant, device)
            fingerprint = state_sha(model)
            call = deployed_forward if variant == "deployed" else core_forward
            batch = next(iter_pilot_batches(protocol, indices, split="validation",
                         epoch=None, batch_size=128)).to(device)
            params = sum(p.numel() for p in model.parameters())
            components = {n: sum(p.numel() for p in m.parameters())
                          for n, m in model.named_children()}
            resources = profile(model, batch, call, device)
            bands, gaps = RepairBandMetrics(device), GapMetrics(device)
            torch.cuda.reset_peak_memory_stats(device)
            validation_start = perf_counter()
            forward_seconds = 0.
            mask_hash = hashlib.sha256()
            windows = 0
            for cpu in iter_pilot_batches(protocol, indices, split="validation", epoch=None, batch_size=128):
                mask_hash.update(cpu.observed_mask.numpy().tobytes())
                batch = cpu.to(device)
                tick = perf_counter()
                explicit = call(model, batch, 0).explicit
                torch.cuda.synchronize(device)
                forward_seconds += perf_counter()-tick
                assert torch.isfinite(explicit.normalized_sequence).all()
                assert torch.equal(explicit.normalized_sequence[explicit.sequence_observed],
                                   batch.normalized_observed[explicit.sequence_observed])
                bands.update(explicit.normalized_sequence, batch.normalized_target,
                    explicit.sequence_observed, explicit.sequence_validity, batch.query_std)
                hidden = explicit.sequence_validity & ~explicit.sequence_observed
                gaps.update((explicit.normalized_sequence-batch.normalized_target)*batch.query_std[:,None], hidden)
                windows += len(batch.refs)
                if windows % 1024 == 0:
                    status(status="validating", candidate=key, windows=windows)
            metrics, stratified = bands.finalize(), gaps.finalize()
            assert windows == 5175
            assert sum(v["count"] for v in stratified.values()) == metrics["full96"]["count"]
            assert state_sha(model) == fingerprint
            results[key] = dict(parameters=params, parameters_by_component=components,
                metrics=metrics, gap_metrics=stratified, inference=resources,
                full_validation_wall_seconds=perf_counter()-validation_start,
                full_validation_forward_seconds=forward_seconds,
                full_validation_peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30,
                full_validation_peak_reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30,
                mask_sha256=mask_hash.hexdigest(), weights_sha256=fingerprint, weights_unchanged=True,
                incremental_training_seconds=0., ancestor_training_cost_included=False)
            write(out/"results.json", results)
            print(json.dumps(dict(candidate=key, parameters=params, full96=metrics["full96"],
                                  inference_ms_per_window=resources["ms_per_window"])), flush=True)
            del model, batch, explicit, bands, gaps
            gc.collect()
            torch.cuda.empty_cache()
    assert len({r["mask_sha256"] for r in results.values()}) == 1
    # Original archived runs allowed cuDNN TF32. Replay that identity separately;
    # all nine comparative probes above deliberately share TF32=False.
    torch.backends.cudnn.allow_tf32 = True
    original = build(bundle, "sealed_v125", "dual", device)
    source_metrics = RepairBandMetrics(device)
    for cpu in iter_pilot_batches(protocol, indices, split="validation", epoch=None, batch_size=128):
        batch = cpu.to(device)
        explicit = core_forward(original, batch, 0).explicit
        source_metrics.update(explicit.normalized_sequence, batch.normalized_target,
            explicit.sequence_observed, explicit.sequence_validity, batch.query_std)
    archived_replay = source_metrics.finalize()
    for metric in ("mse", "mae"):
        assert archived_replay["full96"][metric] == bundle["parent"]["validation_repair_metrics"]["global_"+metric]
    torch.backends.cudnn.allow_tf32 = False
    locked = json.loads((CANDIDATE.parent/"selection.json").read_text())["primary"]["validation"]
    for metric in ("mse", "mae"):
        assert abs(results["VC1_warm_seed2023/deployed"]["metrics"]["full96"][metric] -
                   locked["full96"][metric]) < 1e-10
    write(out/"report.json", dict(results=results, elapsed_seconds=perf_counter()-start,
        archived_cudnn_TF32_true_replay=archived_replay,
        locked_source_replay_verified=True, interpretation="Read-only validation intervention; no candidate training or promotion"))
    status(status="completed")


if __name__ == "__main__":
    main()
