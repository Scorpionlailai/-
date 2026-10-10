import numpy as np
import torch

from malr.models.e23.contracts import ExpertInputs
from malr.models.e23.historical import HistoricalExpert
from malr.prototypes.online_bank import OnlinePrototypeMemory
from malr.prototypes.tensor_index import TensorRetrieval
from malr.prototypes.retrieval import MaskAwareRetrievalEncoder


def _inputs(*, observed=True):
    values = torch.tensor([[[[1., 999., 3., 999.]]]])
    mask = torch.tensor([[[[observed, False, observed, False]]]])
    return ExpertInputs(
        base_latents=torch.ones(1, 1, 1, 8), patch_values=values,
        patch_mask=mask, patch_validity=torch.ones_like(mask),
        rho=mask.float().mean(-1), patch_starts=torch.tensor([[8]]),
        query_mean=torch.zeros(1, 1), query_std=torch.ones(1, 1),
        damaged=torch.ones(1, 1, 1, dtype=torch.bool),
    )


def _memory():
    memory = OnlinePrototypeMemory(
        patch_len=4, patch_stride=2, train_end=32, variable_count=1,
    )
    memory.stage_patch(0, 0, np.array([1., 2., 3., 4.]),
                       np.array([1., 0., 0., 0., 0., 0., 0., 0.]), observed=True)
    memory.stage_patch(0, 12, np.array([100., 100., 100., 100.]),
                       np.array([1., 0., 0., 0., 0., 0., 0., 0.]), observed=True)
    return memory.publish_epoch(1)


def test_h_empty_bank_and_all_unobserved_query_are_unavailable():
    expert = HistoricalExpert(patch_length=4, d_model=8, retrieval_dim=8, variable_count=1,
                              similarity_threshold=-1.0)
    assert isinstance(expert.key_encoder, MaskAwareRetrievalEncoder)
    assert expert.key_encoder.network[0].in_features == 8
    empty = expert(_inputs())
    assert not empty.available.any()
    assert torch.equal(empty.message, torch.zeros_like(empty.message))
    expert.refresh_snapshot(_memory())
    no_obs = expert(_inputs(observed=False))
    assert not no_obs.available.any()
    assert no_obs.provenance["candidate_count"].max() == 0


def test_h_uses_visible_error_only_and_excludes_future_member():
    expert = HistoricalExpert(patch_length=4, d_model=8, retrieval_dim=8, variable_count=1,
                              similarity_threshold=-1.0)
    expert.refresh_snapshot(_memory())
    first = expert(_inputs())
    assert first.available.all()
    assert first.provenance["candidate_count"].max() <= 4
    assert first.provenance["candidate_starts"][first.provenance["candidate_valid"]].tolist() == [0]
    torch.testing.assert_close(first.provenance["visible_error"][0, 0, 0, 0], torch.tensor(0.))
    changed_values = _inputs().patch_values.clone()
    changed_values[..., 1] = -123456
    changed_values[..., 3] = 555555
    changed = ExpertInputs(**{**vars(_inputs()), "patch_values": changed_values})
    second = expert(changed)
    torch.testing.assert_close(first.message, second.message)
    torch.testing.assert_close(first.quality, second.quality)
    assert torch.isfinite(first.message).all()


def test_h_raw_to_query_normalization_and_message_gradient():
    expert = HistoricalExpert(patch_length=4, d_model=8, retrieval_dim=8, variable_count=1,
                              similarity_threshold=-1.0)
    expert.refresh_snapshot(_memory())
    inputs = _inputs()
    inputs = ExpertInputs(**{**vars(inputs), "query_mean": torch.ones(1, 1),
                             "query_std": torch.full((1, 1), 2.)})
    output = expert(inputs)
    assert output.available.all()
    assert torch.isfinite(output.quality).all()
    output.message.square().mean().backward()
    assert expert.message_projector[0].weight.grad is not None
    assert expert.message_projector[0].weight.grad.abs().sum() > 0


def test_h_top_p_uses_eight_static_slots_and_keeps_causal_members_only():
    expert = HistoricalExpert(patch_length=4, d_model=8, retrieval_dim=8, variable_count=1,
                              topk=4, top_p=0.9, candidate_cap=8,
                              similarity_threshold=-1.0)
    expert.refresh_snapshot(_memory())
    result = expert(_inputs())
    provenance = result.provenance
    assert provenance["candidate_valid"].shape == (1, 1, 1, 8)
    assert provenance["candidate_count"].item() == 1
    assert provenance["candidate_starts"][provenance["candidate_valid"]].tolist() == [0]
    assert provenance["weights"][~provenance["candidate_valid"]].sum() == 0


def test_h_two_group_visible_rerank_keeps_four_and_ignores_hidden_values():
    expert = HistoricalExpert(patch_length=4, d_model=8, retrieval_dim=8, variable_count=1,
                              retrieval_strategy="two_group_visible_rerank_v1")
    expert.refresh_snapshot(_memory())
    raw = torch.tensor([10., 1., 1., 1., 1., 20., 30., 40.])[:, None].expand(8, 4)
    candidates = TensorRetrieval(
        raw[None, None, None],
        torch.tensor([.99, .9, .9, .9, .9, .85, .75, .69])[None, None, None],
        torch.arange(8)[None, None, None],
        torch.ones(1, 1, 1, 8, dtype=torch.long),
        torch.ones(1, 1, 1, 8, dtype=torch.bool),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])[None, None, None],
    )
    expert.index.query_two_groups = lambda *_args, **_kwargs: candidates
    first = expert(_inputs())
    provenance = first.provenance
    assert provenance["candidate_valid"].shape == (1, 1, 1, 4)
    assert provenance["candidate_starts"][0, 0, 0].tolist() == [1, 2, 3, 4]
    assert provenance["candidate_count"].item() == 4
    assert provenance["weights"].sum().item() > 0.99
    changed_values = _inputs().patch_values.clone()
    changed_values[..., 1::2] = 1e6
    changed = expert(ExpertInputs(**{**vars(_inputs()), "patch_values": changed_values}))
    torch.testing.assert_close(first.message, changed.message)
    torch.testing.assert_close(first.quality, changed.quality)
    expert.index.query_two_groups = lambda *_args, **_kwargs: TensorRetrieval(
        candidates.raw_patches, candidates.key_similarity, candidates.starts,
        candidates.group_support, torch.zeros_like(candidates.valid), candidates.group_ids,
    )
    empty = expert(_inputs())
    assert not empty.available.any()
    assert empty.provenance["candidate_count"].item() == 0
    assert empty.provenance["weights"].sum() == 0
