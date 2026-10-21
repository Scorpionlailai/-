"""Reuse a deterministic frozen repair prefix across independent calibrators.

No network parameters, optimizer, loss, branch weights or inference rule change.
Captured inputs live for one training batch only. Caller owns separate branch
optimizer and RNG states; a stochastic prefix is explicitly unsupported.
"""
import torch


def capture_frozen_calibration_inputs(model, run_prefix):
    cal = model.recent_mean_projector
    if cal is None or not hasattr(cal, 'members') or len(cal.members) != 2:
        raise ValueError('Expected original dual calibration')
    calibration_parameters = {id(p) for p in cal.parameters()}
    if any(p.requires_grad for p in model.parameters() if id(p) not in calibration_parameters):
        raise ValueError('Shared prefix must be fully frozen')
    calibration_modules = {id(m) for m in cal.modules()}
    if any(m.training for m in model.modules() if id(m) not in calibration_modules):
        raise ValueError('Shared frozen prefix must be in eval mode')
    devices = {p.device for p in model.parameters() if p.device.type == 'cuda'}
    if len(devices) > 1:
        raise ValueError('Expected one device for shared calibration')
    device = next(iter(devices), None)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device is not None else None
    captured = []
    had_local = 'forward' in cal.__dict__
    previous = cal.__dict__.get('forward')

    def capture(*args):
        if len(args) != 5:
            raise ValueError('Unexpected calibration input interface')
        captured.append(args)
        return args[0], torch.zeros_like(args[3])

    cal.forward = capture
    try:
        with torch.no_grad():
            run_prefix()
    finally:
        if had_local:
            cal.forward = previous
        else:
            del cal.forward
    cpu_unchanged = torch.equal(cpu_rng, torch.get_rng_state())
    cuda_unchanged = device is None or torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    if not cpu_unchanged or not cuda_unchanged:
        torch.set_rng_state(cpu_rng)
        if device is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        raise ValueError('Frozen prefix consumed RNG; independent branch streams would change')
    if len(captured) != 1:
        raise ValueError('Expected exactly one frozen prefix forward per batch')
    args = captured[0]
    if args[0].normalized_sequence.requires_grad:
        raise ValueError('Captured coarse recovery must be frozen')
    return args
