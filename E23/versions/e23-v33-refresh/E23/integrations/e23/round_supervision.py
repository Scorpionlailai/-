"""Direct Stage1 supervision of each explicit repair state, using one decoder."""
from malr.data.patching import patchify_with_state
from malr.models.explicit_repair import build_explicit_repair

ROUND_WEIGHTS = (0.2, 0.3, 0.5)


def decode_repair_rounds(model, output, batch):
    if output.iterations is None or len(output.iterations) != 3:
        raise ValueError("round supervision requires three repair states")
    patches = patchify_with_state(batch.normalized_observed, batch.observed_mask,
                                 model.patch_length, model.stride)
    original = patches.values.masked_fill(~patches.effective_mask, 0)
    results = [build_explicit_repair(model.decoder(item.latents), original,
        patches.effective_mask, patches.validity, batch.query_mean, batch.query_std,
        window_len=model.window_length, patch_len=model.patch_length, stride=model.stride,
    ).normalized_sequence for item in output.iterations[:-1]]
    return (*results, output.explicit.normalized_sequence)


def supervised_round_objective(model, output, batch, retrieval_loss, *, retrieval_weight):
    from .train import stage1_objective
    losses = [stage1_objective(sequence, batch.normalized_target, batch.observed_mask,
        retrieval_loss * 0, retrieval_weight=0, loss_mode="global_mse_v1",
        query_std=batch.query_std)[1]["repair"]
        for sequence in decode_repair_rounds(model, output, batch)]
    repair = sum(weight * loss for weight, loss in zip(ROUND_WEIGHTS, losses))
    weighted = retrieval_weight * retrieval_loss
    return repair + weighted, {"repair": repair, "retrieval": retrieval_loss,
        "retrieval_weighted": weighted, **{f"round_{i+1}": loss for i, loss in enumerate(losses)}}
