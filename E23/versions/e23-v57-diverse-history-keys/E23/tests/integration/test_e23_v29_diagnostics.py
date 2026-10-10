import torch
from tests.integration.test_e23_iterative_model import _model
from tests.integration.test_e23_model import _inputs


def test_round_decode_matches_final_repair_and_preserves_valid_observations():
    from scripts.diagnose_e23_v29_rounds import decode_rounds
    torch.manual_seed(34)
    model = _model().eval()
    values, mask, valid, starts, mean, std = _inputs()
    with torch.no_grad():
        _, output = model(values * mask, mask, mean, std, starts, valid_mask=valid)
        sequences = decode_rounds(model.repair, output, values * mask, mask, mean, std, valid_mask=valid)
    assert len(sequences) == 3
    torch.testing.assert_close(sequences[-1], output.explicit.normalized_sequence, rtol=0, atol=0)
    for sequence in sequences:
        assert torch.equal(sequence[mask & valid], values[mask & valid])
        assert torch.isfinite(sequence).all()
