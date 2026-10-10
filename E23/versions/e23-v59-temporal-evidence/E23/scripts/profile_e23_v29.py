"""Throwaway CUDA preflight; no optimizer updates or selected checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import (build_forecast_model, _new_memory, _stage_observed,
    _publish_epoch, _starts, _retrieval_loss, stage1_objective, stage2_objective)
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.author_provenance import write_new_record
from malr.migration import MigrationCatalog


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-one", type=Path, required=True)
    parser.add_argument("--config-three", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this preflight measures CUDA execution")
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(ROOT.parent / "E22/migration/paper_catalog.jsonl", project="E23"),
        protocol_config=ROOT / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=ROOT.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context",
    )
    props = torch.cuda.get_device_properties(device)
    result = {"purpose": "throwaway preflight, no optimizer updates", "batch_size": 128,
              "precision": "float32", "warmup_steps": 4, "measured_steps": 10,
              "device": str(device), "gpu": props.name, "gpu_uuid": str(props.uuid), "models": []}
    for config in (args.config_one, args.config_three):
        settings = load_e23_settings(config, protocol=protocol)
        torch.manual_seed(settings.seed)
        model = build_forecast_model(protocol, settings, device=device)
        repair = model.repair
        repair.historical.retrieval.begin_epoch(1)
        memory = _new_memory(protocol)
        batches = iter(iter_p4_batches(protocol, "train", batch_size=128, epoch=0))
        known = set()
        for _ in range(4):
            batch = next(batches).to(device)
            starts = _starts(batch, stride=repair.stride, patches=repair.patch_count, device=device)
            _stage_observed(memory, repair, batch, starts, known_identities=known)
        _publish_epoch(memory, repair, 1)
        batch = next(batches).to(device)
        starts = _starts(batch, stride=repair.stride, patches=repair.patch_count, device=device)
        row = {"config": str(config.resolve()), "rounds": settings.recovery_rounds,
               "repair_parameters": sum(p.numel() for p in repair.parameters() if p.requires_grad),
               "temporary_bank_members": memory.snapshot().member_count, "stages": {}}
        for stage in ("repair", "forecast"):
            model.train()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            query_calls, source_calls = [], []
            original_query = repair.historical.index.query
            def counted(*arguments, **options):
                query_calls.append(1)
                return original_query(*arguments, **options)
            repair.historical.index.query = counted
            handle = repair.temporal.source_encoder.register_forward_hook(lambda *_: source_calls.append(1))
            forward_seconds = backward_seconds = 0.
            try:
                for index in range(14):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(device)
                    started = perf_counter()
                    if stage == "repair":
                        out = repair(batch.normalized_observed, batch.observed_mask,
                            batch.query_mean, batch.query_std, starts, epoch=1, explore_variable=True,
                            specialist_exposure=.2, specialist_seed=settings.seed)
                        loss, _ = stage1_objective(out.explicit.normalized_sequence,
                            batch.normalized_target, batch.observed_mask, _retrieval_loss(repair, batch),
                            retrieval_weight=.1, loss_mode="global_mse_v1", query_std=batch.query_std)
                    else:
                        prediction, out = model(batch.normalized_observed, batch.observed_mask,
                            batch.query_mean, batch.query_std, starts)
                        loss = stage2_objective(prediction, batch.forecast_target, loss_mode="mse_plus_0p2_mae")
                    torch.cuda.synchronize(device)
                    forwarded = perf_counter()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite preflight loss")
                    loss.backward()
                    torch.cuda.synchronize(device)
                    finished = perf_counter()
                    if index >= 4:
                        forward_seconds += forwarded - started
                        backward_seconds += finished - forwarded
                    if index == 3:
                        torch.cuda.reset_peak_memory_stats(device)
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("nonfinite preflight gradient")
            finally:
                repair.historical.index.query = original_query
                handle.remove()
            if len(query_calls) != 14 or len(source_calls) != 14:
                raise RuntimeError("cache preflight expected one H query/T encoding per forward")
            row["stages"][stage] = {
                "forward_ms": forward_seconds * 100, "backward_ms": backward_seconds * 100,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                "h_queries": len(query_calls), "t_source_encodings": len(source_calls),
                "expert_availability": out.router.available[..., 1:].float().mean((0, 1, 2)).tolist(),
            }
        result["models"].append(row)
        del model, repair, loss, out, batch, memory, original_query
        gc.collect()
        torch.cuda.empty_cache()
    write_new_record(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
