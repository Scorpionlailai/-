"""Bounded execution equivalence/cost audit, not architecture fitting.

Sealed Weather v125 weights, unchanged recent structure/loss. Two independently
optimized calibrators: original sequential schedule versus one frozen prefix per
shared batch with independent dropout RNG streams. Fixed bounded updates;
no test access and no checkpoint selection. PAMAP2 claims require its assets.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from integrations.e23.shared_frozen_calibration import capture_frozen_calibration_inputs
from integrations.e23.level_shape_pilot_data import iter_pilot_batches
from integrations.e23.global_only_adaptation import RepairBandMetrics
from scripts.probe_v125_calibration_consolidation import load_original_pilot, build, core_forward, state_sha, profile
from scripts.run_v125_H_soft_memory_heads import frozen_hash


def branch_loss(explicit, batch):
    hidden = explicit.sequence_validity & ~explicit.sequence_observed
    error = (explicit.normalized_sequence-batch.normalized_target)*batch.query_std[:, None]
    main = error[hidden].abs().mean()
    return main, main+.25*error[:, -32:][hidden[:, -32:]].abs().mean()


def full_branch(model, batch, index):
    cal = model.recent_mean_projector
    model.recent_mean_projector = cal.members[index]
    try:
        return core_forward(model, batch, 0).explicit
    finally:
        model.recent_mean_projector = cal


def stream_call(state, device, function):
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    try:
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state(state['cuda'], device)
        result = function()
        state['cpu'] = torch.get_rng_state().clone()
        state['cuda'] = torch.cuda.get_rng_state(device).clone()
        return result
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda, device)


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


@torch.no_grad()
def validate(model, protocol, device):
    model.eval()
    acc = RepairBandMetrics(device)
    windows = 0
    for cpu in iter_pilot_batches(protocol, dict(validation=np.arange(5175)), split='validation', epoch=None, batch_size=128):
        b = cpu.to(device)
        e = core_forward(model, b, 0).explicit
        assert torch.equal(e.normalized_sequence[e.sequence_observed], b.normalized_observed[e.sequence_observed])
        acc.update(e.normalized_sequence, b.normalized_target, e.sequence_observed, e.sequence_validity, b.query_std)
        windows += len(b.refs)
    assert windows == 5175
    return acc.finalize()


def main():
    start = perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--train-windows', type=int, choices=[1024, 4096], default=1024)
    ap.add_argument('--epochs', type=int, choices=[1, 4], default=1)
    ap.add_argument('--shared-first', action='store_true')
    args = ap.parse_args()
    out = args.output
    out.mkdir(exist_ok=False)
    def status(**kw):
        write(out/'status.json', dict(pid=os.getpid(), elapsed_seconds=perf_counter()-start, **kw))
    status(status='loading')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    bundle = load_original_pilot()
    protocol = bundle['protocol']
    indices = dict(train=np.linspace(0, len(protocol.refs['train'])-1, args.train_windows, dtype=np.int64))
    batches = [batch for epoch in range(args.epochs) for batch in
        iter_pilot_batches(protocol, indices, split='train', epoch=epoch, batch_size=128)]
    updates = args.train_windows//128*args.epochs
    assert len(batches) == updates
    identity = dict(dataset='Weather', scene='sr015_bl8', parent='sealed_original_v125',
        parent_checkpoint_sha256=bundle['parent']['checkpoint_sha256'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        updates_per_branch=updates, branches=2, selected_checkpoints=False,
        train_windows=args.train_windows, epochs=args.epochs, mask_epochs=list(range(args.epochs)),
        execution_order='shared_first' if args.shared_first else 'reference_first',
        source_weights_reused=True, optimizer='Adam', lr=.001, weight_decay=0.,
        objective='Original full96MAE +.25recent32MAE, unchanged',
        prefix='all frozen/eval, including original recentCNN; independent full original cal members train',
        schedule=dict(reference='branch64 all batches then branch256 all batches', shared='batchwise one deterministic prefix then independent64/256 updates'),
        branch_rng='Separate cpu/cuda streams each seeded2023; no global branch stream mixing',
        label='Computational equivalence/cost audit. Not formal model training or efficacy selection',
        PAMAP2_run=False, test_read=False, Stage2=False,
        prospective_gate=dict(training_seconds_reduction_min=.15, memory_increase_max=.05,
                              validation_mae_relative_max=.01, validation_mse_relative_max=.02))
    write(out/'identity.json', identity)
    results = {}
    for arm in (['shared', 'reference'] if args.shared_first else ['reference', 'shared']):
        arm_start = perf_counter()
        status(status='building', arm=arm)
        model = build(bundle, 'sealed_v125', 'dual', device)
        cal = model.recent_mean_projector
        for member in cal.members:
            member.train().requires_grad_(True)
        frozen = frozen_hash(model)
        optimizers = [torch.optim.Adam(m.parameters(), lr=.001, weight_decay=0.) for m in cal.members]
        streams = [dict(cpu=torch.Generator().manual_seed(2023).get_state(),
                        cuda=torch.Generator(device=device).manual_seed(2023).get_state()) for _ in range(2)]
        # Count actual backbone calls independently from the schedule label.
        counter = [0]
        handle = model.front_context.register_forward_hook(lambda *unused: counter.__setitem__(0, counter[0]+1))
        torch.cuda.reset_peak_memory_stats(device)
        training_seconds = 0.
        histories = [[], []]
        def update(index, batch, shared_args=None):
            member = cal.members[index]
            optimizer = optimizers[index]
            optimizer.zero_grad(set_to_none=True)
            explicit = member(*shared_args)[0] if shared_args is not None else full_branch(model, batch, index)
            main, loss = branch_loss(explicit, batch)
            assert torch.isfinite(loss)
            if not histories[index]:
                gradients = torch.autograd.grad(main, tuple(member.parameters()), retain_graph=True, allow_unused=True)
                norm = sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)**.5
                assert np.isfinite(norm) and norm > 0
                write(out/f'{arm}_member{index}_main_gradient.json', dict(main_only_gradient_norm=norm, independent_of_recent_aux=True))
            loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            optimizer.step()
            histories[index].append(dict(main=float(main.detach()), total=float(loss.detach())))
            optimizer.zero_grad(set_to_none=True)
        if arm == 'reference':
            for index in (0, 1):
                for step, cpu in enumerate(batches):
                    batch = cpu.to(device)
                    torch.cuda.synchronize(device)
                    tick = perf_counter()
                    stream_call(streams[index], device, lambda: update(index, batch))
                    torch.cuda.synchronize(device)
                    training_seconds += perf_counter()-tick
                    status(status='matching_updates', arm=arm, branch=index, updates=step+1)
        else:
            for step, cpu in enumerate(batches):
                batch = cpu.to(device)
                torch.cuda.synchronize(device)
                tick = perf_counter()
                context = capture_frozen_calibration_inputs(model, lambda: core_forward(model, batch, 0))
                for index in (0, 1):
                    stream_call(streams[index], device, lambda: update(index, batch, context))
                torch.cuda.synchronize(device)
                training_seconds += perf_counter()-tick
                status(status='matching_updates', arm=arm, updates_per_branch=step+1)
        handle.remove()
        assert counter[0] == (2*updates if arm == 'reference' else updates)
        assert frozen_hash(model) == frozen
        peak = dict(allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30,
                    reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30)
        torch.save(dict(model_state=model.state_dict(), optimizer_states=[o.state_dict() for o in optimizers],
                        streams=streams, identity=identity, arm=arm), out/(arm+f'_after{updates}.pt'))
        fingerprint = state_sha(model)
        tick = perf_counter()
        validation = validate(model, protocol, device)
        validation_seconds = perf_counter()-tick
        sample = batches[0].to(device)
        inference = profile(model, sample, core_forward, device)
        results[arm] = dict(training_seconds=training_seconds, actual_prefix_calls=counter[0],
            training_peak_memory=peak, validation=validation, validation_seconds=validation_seconds,
            inference=inference, weight_state_sha256=fingerprint, histories=histories,
            total_parameters=sum(p.numel() for p in model.parameters()),
            frozen_state_bank_keys_unchanged=True, arm_workflow_seconds=perf_counter()-arm_start)
        write(out/(arm+'_result.json'), results[arm])
        del model, cal, optimizers, batch, sample
        if arm == 'shared':
            del context
        gc.collect()
        torch.cuda.empty_cache()
    ref, shared = results['reference'], results['shared']
    saved = [torch.load(out/(a+f'_after{updates}.pt'), map_location='cpu', weights_only=False) for a in ['reference', 'shared']]
    exact = ref['weight_state_sha256'] == shared['weight_state_sha256']
    max_delta = max(float((saved[0]['model_state'][k]-saved[1]['model_state'][k]).abs().max())
                    for k in saved[0]['model_state'] if saved[0]['model_state'][k].is_floating_point())
    assert exact, f'Matching execution changed weights, max delta {max_delta}'
    assert ref['histories'] == shared['histories'] and ref['validation'] == shared['validation']
    for i in (0, 1):
        for key in ['cpu', 'cuda']:
            assert torch.equal(saved[0]['streams'][i][key], saved[1]['streams'][i][key])
    report = dict(identity=identity, results=results, exact_states=True, max_weight_delta=max_delta,
        exact_losses=True, exact_branch_RNG=True, exact_full5175_validation=True,
        training_seconds_reduction=1-shared['training_seconds']/ref['training_seconds'],
        workflow_seconds=perf_counter()-start,
        limitation='Bounded update accounting includes independent main-gradient preflight and early optimizer allocation. Not a full100epoch time or a new model efficacy result.')
    write(out/'report.json', report)
    status(status='completed')
    print(json.dumps({k:v for k,v in report.items() if k not in ['results','identity']}), flush=True)


if __name__ == '__main__':
    main()
