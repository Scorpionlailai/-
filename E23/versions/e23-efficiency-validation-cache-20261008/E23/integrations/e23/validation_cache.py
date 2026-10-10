"""Reuse explicit repair inputs only for a fixed, frozen-repair validation set."""
from dataclasses import dataclass
from time import perf_counter

import torch

from integrations.forecast_compare.protocol import iter_p4_batches
from .forecast_sensitive import parameter_hash


@dataclass
class ForecastValidationCache:
    model: object
    protocol: object
    settings_sha256: str
    repair_sha256: str
    padding: str
    batches: list
    metadata: dict

    @classmethod
    @torch.no_grad()
    def build(cls, model, protocol, settings, *, device, starts_fn):
        if model.forecast_input != "explicit_v1":
            raise ValueError("validation cache requires explicit predictor inputs")
        if any(p.requires_grad for p in model.repair.parameters()):
            raise ValueError("validation cache requires the entire repairer frozen")
        model.eval()
        repair_hash = parameter_hash(model.repair)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        rng_cpu = torch.get_rng_state().clone()
        rng_cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        cached, tensor_bytes, windows = [], 0, 0
        for cpu_batch in iter_p4_batches(protocol, "validation", batch_size=settings.batch_size):
            batch = cpu_batch.to(device)
            starts = starts_fn(batch, stride=model.repair.stride,
                               patches=model.repair.patch_count, device=device)
            output = model.repair(batch.normalized_observed, batch.observed_mask,
                                  batch.query_mean, batch.query_std, starts)
            values, observed, valid = model.explicit_predictor_inputs(output.explicit)
            tensors = tuple(t.detach().cpu().clone() for t in (
                values, observed, valid, batch.query_mean, batch.query_std,
                batch.forecast_target,
            ))
            cached.append(tensors)
            windows += values.shape[0]
            tensor_bytes += sum(t.numel() * t.element_size() for t in tensors)
        if not cached:
            raise ValueError("fixed validation contains no batches")
        if repair_hash != parameter_hash(model.repair):
            raise RuntimeError("building cache changed the frozen repairer")
        if not torch.equal(rng_cpu, torch.get_rng_state()):
            raise RuntimeError("cache preparation consumed training CPU randomness")
        if rng_cuda is not None and not torch.equal(rng_cuda, torch.cuda.get_rng_state(device)):
            raise RuntimeError("cache preparation consumed training CUDA randomness")
        metadata = dict(schema="fixed_validation_explicit_inputs_v1", split="validation",
                        storage="cpu", windows=windows, batches=len(cached),
                        tensor_bytes=tensor_bytes, preparation_seconds=perf_counter()-started,
                        repair_sha256=repair_hash, validation_mask_sha256=protocol.mask_sha256["validation"],
                        source_csv_sha256=protocol.source_sha256, randomness_unchanged=True,
                        training_masks_cached=False, test_inputs_cached=False)
        return cls(model, protocol, settings.config_sha256, repair_hash,
                   model.forecast_padding, cached, metadata)

    @torch.no_grad()
    def evaluate(self, model, protocol, settings, *, device):
        if model is not self.model or protocol is not self.protocol:
            raise ValueError("validation cache belongs to a different model or protocol")
        if settings.config_sha256 != self.settings_sha256 or model.forecast_padding != self.padding:
            raise ValueError("validation cache configuration changed")
        if any(p.requires_grad for p in model.repair.parameters()):
            raise ValueError("validation cache requires the entire repairer frozen")
        if parameter_hash(model.repair) != self.repair_sha256:
            raise ValueError("validation cache repair weights/buffers changed")
        model.eval()
        squared = absolute = 0.0
        count = 0
        for index, tensors in enumerate(self.batches, 1):
            values, observed, valid, mean, std, target = (t.to(device) for t in tensors)
            prediction = model.predictor(values, observed, mean, std, patch_validity=valid)
            error = prediction - target
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            count += error.numel()
            if index == 1 or index % 5 == 0 or index == len(self.batches):
                print(f"stage=forecast validation batch={index}/{len(self.batches)} cache=fixed", flush=True)
        return {"mse": squared/count, "mae": absolute/count}
