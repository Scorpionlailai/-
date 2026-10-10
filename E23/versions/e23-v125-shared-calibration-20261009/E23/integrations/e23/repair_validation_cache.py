"""Cache only fixed validation inputs to the trainable mean/dense repair suffix."""
from dataclasses import fields, replace
from time import perf_counter
from types import SimpleNamespace

import torch

from integrations.forecast_compare.protocol import iter_p4_batches
from .train import _starts


def explicit_to(explicit, device):
    return replace(explicit, **{field.name: getattr(explicit, field.name).detach().to(device)
                               for field in fields(explicit)
                               if isinstance(getattr(explicit, field.name), torch.Tensor)})


class FixedRepairValidationCache:
    def __init__(self, model, protocol, settings, frozen_hash):
        if model.recent_mean_projector is None:
            raise ValueError('This cache requires the mean/dense suffix')
        if any(parameter.requires_grad for name, parameter in model.named_parameters()
               if not name.startswith('recent_mean_projector.')):
            raise ValueError('Validation cache requires a completely frozen expert prefix')
        self.model, self.protocol, self.settings = model, protocol, settings
        self.frozen_hash = frozen_hash
        self.expected_hash = frozen_hash(model)
        self.rows = []
        self.build_seconds = 0.

    @staticmethod
    def accumulate(totals, sequence, target, mask, std):
        hidden = ~mask
        error = sequence-target
        global_error = error*std[:, None, :]
        small = hidden & (std[:, None, :] < .01)
        for key, value in [
            ('window_squared', error.square()[hidden].sum()),
            ('global_squared', global_error.square()[hidden].sum()),
            ('global_absolute', global_error.abs()[hidden].sum()),
            ('small_window_squared', error.square()[small].sum()),
            ('small_global_squared', global_error.square()[small].sum()),
            ('small_count', small.sum()), ('count', hidden.sum()),
        ]:
            totals[key] = totals.get(key, 0)+value.item()

    @staticmethod
    def metrics(t):
        if t['count'] <= 0:
            raise ValueError('No validation hidden coordinates')
        return dict(window_mse=t['window_squared']/t['count'],
                    global_mse=t['global_squared']/t['count'],
                    global_mae=t['global_absolute']/t['count'],
                    hidden_coordinates=int(t['count']),
                    small_std_hidden_fraction=t['small_count']/t['count'],
                    small_std_window_squared_share=t['small_window_squared']/t['window_squared'] if t['window_squared'] else 0.,
                    small_std_global_squared_share=t['small_global_squared']/t['global_squared'] if t['global_squared'] else 0.)

    @torch.no_grad()
    def build(self, *, device):
        if self.rows:
            raise RuntimeError('Validation cache already built')
        self.model.eval()
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(device).clone()
        started = perf_counter()
        captured = []

        def capture(module, args):
            explicit, router, points, mean, std = args
            captured.append((explicit_to(explicit, 'cpu'), router.latents.detach().cpu(),
                             points.detach().cpu(), mean.detach().cpu(), std.detach().cpu()))

        handle = self.model.recent_mean_projector.register_forward_pre_hook(capture)
        totals = {}
        try:
            for cpu in iter_p4_batches(self.protocol, 'validation', batch_size=self.settings.batch_size):
                batch = cpu.to(device)
                starts = _starts(batch, stride=self.model.stride, patches=self.model.patch_count, device=device)
                output = self.model(batch.normalized_observed, batch.observed_mask,
                                    batch.query_mean, batch.query_std, starts)
                if len(captured) != 1:
                    raise RuntimeError('Expected exactly one suffix invocation per validation batch')
                self.rows.append((*captured.pop(), batch.normalized_target.detach().cpu(),
                                  batch.observed_mask.detach().cpu()))
                self.accumulate(totals, output.explicit.normalized_sequence,
                                batch.normalized_target, batch.observed_mask, batch.query_std)
        finally:
            handle.remove()
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
        assert self.expected_hash == self.frozen_hash(self.model)
        self.build_seconds = perf_counter()-started
        return self.metrics(totals)

    @torch.no_grad()
    def evaluate(self, *, device):
        if not self.rows or self.expected_hash != self.frozen_hash(self.model):
            raise RuntimeError('Frozen expert prefix changed or cache missing')
        self.model.eval()
        totals = {}
        for explicit, latents, points, mean, std, target, mask in self.rows:
            std = std.to(device)
            result, _ = self.model.recent_mean_projector(
                explicit_to(explicit, device), SimpleNamespace(latents=latents.to(device)),
                points.to(device), mean.to(device), std)
            self.accumulate(totals, result.normalized_sequence, target.to(device), mask.to(device), std)
        return self.metrics(totals)

    def identity(self):
        tensor_bytes = 0
        for explicit, *rest in self.rows:
            tensors = [getattr(explicit, field.name) for field in fields(explicit)] + rest
            tensor_bytes += sum(value.numel()*value.element_size() for value in tensors
                                if isinstance(value, torch.Tensor))
        return dict(enabled=True, split='fixed_validation_only',
                    batch_count=len(self.rows), windows=sum(row[-1].shape[0] for row in self.rows),
                    tensor_bytes=tensor_bytes, storage='cpu', build_seconds=self.build_seconds,
                    frozen_expert_hash=self.expected_hash, training_masks_cached=False,
                    build_included_in_training_time=True)
