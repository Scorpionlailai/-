"""Historical expert compatibility wrapper; state keys retain their original paths."""
from torch import Tensor
from .contracts import ExpertEvidence, ExpertInputs
from .historical_candidates import HistoricalCandidateCache, HistoricalCandidateProvider
from .historical_messages import initialize_historical_messages, historical_from_cache


class HistoricalExpert(HistoricalCandidateProvider):
    def __init__(self, patch_length: int, d_model: int, retrieval_dim: int, variable_count: int,
                 *, topk: int = 4, similarity_threshold: float = .7,
                 top_p: float | None = None, candidate_cap: int = 8,
                 retrieval_strategy: str = "one_group", contextual: bool = False,
                 tv_query: str = "none"):
        super().__init__(patch_length, d_model, retrieval_dim, variable_count,
                         topk=topk, similarity_threshold=similarity_threshold,
                         top_p=top_p, candidate_cap=candidate_cap,
                         retrieval_strategy=retrieval_strategy, tv_query=tv_query)
        initialize_historical_messages(self, patch_length, d_model, contextual)

    from_cache = historical_from_cache

    def forward(self, inputs: ExpertInputs) -> ExpertEvidence:
        return self.from_cache(inputs, self.prepare_candidates(inputs))


__all__ = ["HistoricalExpert", "HistoricalCandidateCache"]
