"""Parameter ownership of one complete, independently initialized expert group."""
from torch import nn
from .historical_messages import HistoricalMessageEncoder
from .variable import SparseVariableExpert
from .temporal import MultiScaleTemporalExpert
from .router import ExpertRouter


class TVHGroup(nn.Module):
    def __init__(self, variable_count: int, patch_length: int, d_model: int, *,
                 variable_topk: int = 4, variable_offsets: tuple[int, ...] = (-1,0,1),
                 temporal_local_radius: int = 2, temporal_boundaries: tuple[int, ...] = (2,4,8,16)):
        super().__init__()
        self.historical = HistoricalMessageEncoder(patch_length, d_model, contextual=True)
        self.variable = SparseVariableExpert(variable_count,patch_length,d_model,
            topk=variable_topk,offsets=variable_offsets,selection='window_context_topk_v1',contextual=True)
        self.temporal = MultiScaleTemporalExpert(patch_length,d_model,local_radius=temporal_local_radius,
            boundaries=temporal_boundaries,contextual=True)
        self.router = ExpertRouter(d_model)
        self.condition_embedding = nn.Embedding(3,d_model)
        nn.init.zeros_(self.condition_embedding.weight)
