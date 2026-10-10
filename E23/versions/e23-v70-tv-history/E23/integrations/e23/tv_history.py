"""Stage1-only alignment to the same frozen encoder used by the history bank."""
import torch
from torch.nn import functional as F


def tv_query_alignment_loss(historical, provenance, clean_raw, clean_validity):
    keys = provenance['tv_query_keys']
    eligible = provenance['tv_query_eligible'].bool()
    if clean_raw.shape != clean_validity.shape or keys.shape[:-1] != clean_raw.shape[:-1]:
        raise ValueError('TV query alignment geometry mismatch')
    if not eligible.any():
        return keys.sum() * 0.
    target = historical.retrieval.encode_frozen(clean_raw[eligible], clean_validity[eligible])
    return (1. - F.cosine_similarity(keys[eligible], target.detach(), dim=-1)).mean()
