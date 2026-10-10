from dataclasses import replace
import torch
from tests.unit.test_e23_iterative import _parts
from malr.models.e23.iterative import IterativeTVHController


def test_refresh_requeries_each_round_and_keeps_original_evidence(monkeypatch):
    _, inputs, parts = _parts()
    controller = IterativeTVHController(8, 3, refresh=True)
    decoder = torch.nn.Linear(8, 6)
    h_calls, v_calls = [], []
    hm = parts['historical'].prepare_candidates
    vm = parts['variable'].prepare_sources
    def h_capture(arg, **kw):
        h_calls.append((arg.patch_mask.clone(), kw.get('query_estimates')))
        return hm(arg, **kw)
    def v_capture(arg, **kw):
        v_calls.append(kw.get('refresh_context', False))
        return vm(arg, **kw)
    monkeypatch.setattr(parts['historical'], 'prepare_candidates', h_capture)
    monkeypatch.setattr(parts['variable'], 'prepare_sources', v_capture)
    out = controller(inputs, **parts, decoder=decoder)
    assert len(h_calls) == 3 and v_calls == [False, True, True]
    assert h_calls[0][1] is None and h_calls[1][1] is not None
    assert all(torch.equal(mask, inputs.patch_mask) for mask, _ in h_calls)
    assert all(i.update_gate is not None and ((i.update_gate > 0) & (i.update_gate < 1)).all()
               for i in out.iterations[1:])
    assert out.iterations[0].update_gate is None
    assert out.router.latents.grad_fn is not None


def test_closed_update_gate_recovers_first_round():
    _, inputs, parts = _parts()
    controller = IterativeTVHController(8, 3, refresh=True)
    with torch.no_grad():
        controller.update_gate[-1].bias.fill_(-100)
    out = controller(inputs, **parts, decoder=torch.nn.Linear(8, 6))
    torch.testing.assert_close(out.router.latents, out.iterations[0].latents, rtol=0, atol=0)


def test_query_estimates_do_not_change_observed_scores_or_allow_noncausal_members():
    _, inputs, parts = _parts()
    h = parts['historical']
    h.prepare_candidates(inputs, query_estimates=torch.zeros_like(inputs.patch_values))
    altered = replace(inputs, patch_values=torch.where(inputs.patch_mask, inputs.patch_values,
                                                     torch.full_like(inputs.patch_values, 9999)))
    a = h.prepare_candidates(inputs, query_estimates=torch.zeros_like(inputs.patch_values))
    b = h.prepare_candidates(altered, query_estimates=torch.zeros_like(inputs.patch_values))
    torch.testing.assert_close(a.base_logits, b.base_logits, rtol=0, atol=0)
    assert torch.equal(a.valid, b.valid)
    starts = a.provenance['candidate_starts']
    query_starts = inputs.patch_starts
    if query_starts.ndim == 2:
        query_starts = query_starts[:, None].expand_as(inputs.damaged)
    assert ((starts + 6 <= query_starts[..., None]) | ~a.valid).all()


def test_refresh_specialist_targets_never_read_other_experts():
    _, inputs, parts = _parts()
    parts['specialist_exposure'] = 1.
    out = IterativeTVHController(8, 3, refresh=True)(inputs, **parts, decoder=torch.nn.Linear(8, 6))
    assert out.specialist_selected.any()
    first = out.iterations[0].allowed
    for item in out.iterations:
        assert torch.equal(item.allowed[out.specialist_selected], first[out.specialist_selected])
        assert not item.preview_gates[..., 0][~item.allowed[..., 2]].any()
        assert not item.preview_gates[..., 1][~item.allowed[..., 1]].any()
