"""Registered finite Stage1 matching: one64 calibrator versus original dual.

Both share the locked VC1 warm prefix and one-pass inference. No LSV/fixedV,
predictor, test set, new module or new loss recipe. Not full-budget training.
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
from torch.nn import functional as F
from integrations.e23.aligned_stage2 import reset_training_rng
from integrations.e23.global_only_adaptation import RepairBandMetrics
from integrations.e23.level_shape_pilot_data import iter_pilot_batches
from scripts.probe_v125_calibration_consolidation import (
    load_original_pilot, build, core_forward, state_sha, GapMetrics, profile,
    CANDIDATE, MANIFEST, file_sha256)
from scripts.run_v125_H_soft_memory_heads import frozen_hash


def objective(model, batch):
    explicit = core_forward(model, batch, 0).explicit
    hidden = explicit.sequence_validity & ~explicit.sequence_observed
    error = (explicit.normalized_sequence-batch.normalized_target)*batch.query_std[:, None]
    main = error[hidden].abs().mean()
    cal = model.recent_mean_projector
    weights = cal.weights.to(error)
    proposals = sum(w*m.global_dense_refiner.atom_proposal for w, m in zip(weights, cal.members))
    logits = sum(w*m.global_dense_refiner.atom_logits for w, m in zip(weights, cal.members))
    selected = cal.atom_semantics["selected"].to(error.device)[None, None]
    atom = cal.atom_semantics["atoms"].to(error)[None, None]
    truth = batch.normalized_target*batch.query_std[:, None]+batch.query_mean[:, None]
    with torch.no_grad():
        atom_error = (atom-truth).abs()
        proposal_error = (proposals.detach()-truth).abs()
        action = (atom_error <= proposal_error).to(logits)
        cost = (atom_error-proposal_error).abs()
    aux = (F.binary_cross_entropy_with_logits(logits, action, reduction="none")*cost)[hidden & selected].mean()
    return explicit, main, main+.1*aux


def write(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


@torch.no_grad()
def evaluate(model, protocol, indices, device):
    bands, gaps = RepairBandMetrics(device), GapMetrics(device)
    windows = 0
    for cpu in iter_pilot_batches(protocol, indices, split="validation", epoch=None, batch_size=128):
        batch = cpu.to(device)
        explicit = core_forward(model, batch, 0).explicit
        assert torch.isfinite(explicit.normalized_sequence).all()
        assert torch.equal(explicit.normalized_sequence[explicit.sequence_observed],
                           batch.normalized_observed[explicit.sequence_observed])
        bands.update(explicit.normalized_sequence, batch.normalized_target,
                     explicit.sequence_observed, explicit.sequence_validity, batch.query_std)
        hidden = explicit.sequence_validity & ~explicit.sequence_observed
        gaps.update((explicit.normalized_sequence-batch.normalized_target)*batch.query_std[:, None], hidden)
        windows += len(batch.refs)
    assert windows == 5175
    return dict(**bands.finalize(), gaps=gaps.finalize())


def main():
    started = perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    out = args.output
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out/"executed_runner.py")
    def status(**kw):
        write(out/"status.json", dict(pid=os.getpid(), elapsed_seconds=perf_counter()-started, **kw))
    status(status="loading")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    bundle = load_original_pilot()
    protocol = bundle["protocol"]
    assert protocol.training_seed == 2023
    indices = dict(train=np.linspace(0, len(protocol.refs["train"])-1, 4096, dtype=np.int64),
                   validation=np.arange(len(protocol.refs["validation"])))
    for split, ix in indices.items():
        np.save(out/(split+"_indices.npy"), ix)
    identity = dict(
        label="VC1暖启动结构收敛 / single64 matched Stage1",
        parent=str(CANDIDATE), parent_sha256=file_sha256(CANDIDATE),
        manifest_sha256=file_sha256(MANIFEST),
        runner_sha256=file_sha256(Path(__file__)),
        dataset="Weather", scene="sr015_bl8", seq_len=96, pred_len=96,
        arms={"D": "one-pass dual64/256 weight0.5/0.5", "S": "one-pass single64"},
        parent_warm_seed=2023, training_seed=2023, train_windows=4096,
        validation_windows=5175, epochs=20, batch_size=128, updates_per_arm=640,
        mask_epochs=[7301, 7320], optimizer="Adam", lr=1e-5, weight_decay=0.,
        clip_norm=1., precision="FP32", TF32=False, mode="all eval, calibrator gradients enabled",
        objective="full96 MAE +0.1 original full96 atom-risk BCE, no recent weighting",
        primary="full96 complete validation MAE", secondary="full96 complete validation MSE",
        epoch0_included=True,
        train_scope="Only retained original global_dense_refiner; whole common CNN/HVT/prefix/bank/key frozen",
        inference="One main forward, both no LSV/fixedV. Same frozen probe core semantics",
        history="Inherited weights previously trained with recent emphasis; not from-scratch bias-free training",
        test_read=False, Stage2=False, official_PAMAP2_ablation_modified=False,
        accuracy_guards_relative={"mae": .01, "mse": .02},
        resource_goals={"parameter_reduction": .4, "latency_reduction": .2},
        cost_scope="This entire warm run incremental only; ancestor costs required",
    )
    write(out/"identity.json", identity)
    results, initial_common = {}, None
    for arm, variant in (("D", "dual"), ("S", "single64")):
        status(status="building", arm=arm)
        arm_start = perf_counter()
        model = build(bundle, "VC1_warm_seed2023", variant, device)
        for member in model.recent_mean_projector.members:
            member.global_dense_refiner.requires_grad_(True)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        params = [p for p in model.parameters() if p.requires_grad]
        assert all(n.startswith("recent_mean_projector.members.") and ".global_dense_refiner." in n for n in names)
        common = state_sha(model.recent_mean_projector.members[0])
        if initial_common is None:
            initial_common = common
        assert common == initial_common
        frozen = frozen_hash(model)
        write(out/arm/"model_identity.json", dict(parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in params), trainable_names=names,
            common64_initial_sha256=common, frozen_state_bank_key_sha256=frozen))
        reset_training_rng(2023)
        optimizer = torch.optim.Adam(params, lr=1e-5, weight_decay=0.)
        torch.cuda.reset_peak_memory_stats(device)
        train_seconds = validation_seconds = save_seconds = 0.
        selected, history = {}, []
        digest = hashlib.sha256()
        for epoch in range(21):
            if epoch:
                assert not any(m.training for m in model.modules())
                for step, cpu in enumerate(iter_pilot_batches(protocol, indices, split="train",
                                            epoch=7300+epoch, batch_size=128)):
                    digest.update(cpu.observed_mask.numpy().tobytes())
                    digest.update(np.asarray([r.target_start for r in cpu.refs], dtype=np.int64).tobytes())
                    batch = cpu.to(device)
                    tick = perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    explicit, main_loss, loss = objective(model, batch)
                    assert torch.isfinite(main_loss) and torch.isfinite(loss)
                    if epoch == 1 and step == 0:
                        gradients = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
                        norms = {str(i): sum(float(g.detach().double().square().sum())
                            for n, g in zip(names, gradients) if g is not None and
                            n.startswith(f"recent_mean_projector.members.{i}."))**.5
                            for i in range(len(model.recent_mean_projector.members))}
                        assert all(np.isfinite(v) and v > 0 for v in norms.values())
                        assert all(g is None or torch.isfinite(g).all() for g in gradients)
                        write(out/arm/"main_loss_gradient_preflight.json",
                              dict(main_MAE_independently_checked=True, member_gradient_norms=norms,
                                   auxiliary_excluded=True, full96_no_recent_weight=True))
                        del gradients
                    loss.backward()
                    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
                    torch.nn.utils.clip_grad_norm_(params, 1.)
                    optimizer.step()
                    torch.cuda.synchronize(device)
                    train_seconds += perf_counter()-tick
                    if step % 8 == 0:
                        status(status="training", arm=arm, epoch=epoch, updates=(epoch-1)*32+step+1)
                assert step == 31
            assert frozen_hash(model) == frozen
            tick = perf_counter()
            metrics = evaluate(model, protocol, indices, device)
            validation_seconds += perf_counter()-tick
            row = dict(epoch=epoch, updates=32*epoch, metrics=metrics)
            history.append(row)
            def save(path):
                nonlocal save_seconds
                tick = perf_counter()
                torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    identity=identity, arm=arm, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device), numpy_rng=np.random.get_state(),
                    frozen_state_bank_key_sha256=frozen, **row), path)
                save_seconds += perf_counter()-tick
            if epoch == 0:
                save(out/arm/"epoch0.pt")
            for metric in ("mae", "mse"):
                if metric not in selected or metrics["full96"][metric] < selected[metric]["metrics"]["full96"][metric]:
                    selected[metric] = row
                    save(out/arm/("selected-"+metric+".pt"))
            write(out/arm/"history.json", history)
            status(status="epoch_completed", arm=arm, **row)
            print(json.dumps(dict(arm=arm, epoch=epoch, full96=metrics["full96"])), flush=True)
        save(out/arm/"terminal.pt")
        for metric in ("mae", "mse"):
            path = out/arm/("selected-"+metric+".pt")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            tick = perf_counter()
            replay = evaluate(model, protocol, indices, device)
            validation_seconds += perf_counter()-tick
            assert replay == selected[metric]["metrics"]
            selected[metric] = dict(selected[metric], checkpoint=str(path), sha256=file_sha256(path))
        peaks = dict(allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30,
                     reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30)
        model.load_state_dict(torch.load(out/arm/"selected-mae.pt", map_location=device,
                                        weights_only=False)["model_state"], strict=True)
        assert frozen_hash(model) == frozen
        batch = next(iter_pilot_batches(protocol, indices, split="validation", epoch=None, batch_size=128)).to(device)
        inference = profile(model, batch, core_forward, device)
        results[arm] = dict(epoch0=history[0], primary=selected["mae"], secondary=selected["mse"],
            training_seconds=train_seconds, validation_seconds=validation_seconds, save_seconds=save_seconds,
            arm_seconds=perf_counter()-arm_start, training_peak_memory=peaks, inference=inference,
            mask_order_sha256=digest.hexdigest(), common64_initial_sha256=common, updates=640,
            frozen_parameters_bank_keys_unchanged=True)
        write(out/arm/"result.json", results[arm])
        del model, optimizer, params, batch, explicit, loss, main_loss, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    assert results["D"]["mask_order_sha256"] == results["S"]["mask_order_sha256"]
    write(out/"report.json", dict(identity=identity, results=results,
        workflow_seconds=perf_counter()-started, no_automatic_promotion=True))
    status(status="completed")


if __name__ == "__main__":
    main()
