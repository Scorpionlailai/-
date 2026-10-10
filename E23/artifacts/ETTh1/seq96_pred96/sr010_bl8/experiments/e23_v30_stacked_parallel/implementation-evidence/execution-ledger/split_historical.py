from pathlib import Path
p = Path('src/malr/models/e23/historical.py')
s = p.read_text()
start = s.index('class HistoricalExpert')
provider = s[:start] + s[start:s.index('    def from_cache(')]
provider = provider.replace('class HistoricalExpert(nn.Module):', 'class HistoricalCandidateProvider(nn.Module):')
provider = provider.replace('        contextual: bool = False,\n', '')
a = provider.index('        self.message_projector =')
b = provider.index('        self.index:', a)
initialization = provider[a:b]
provider = provider[:a] + provider[b:]
provider = provider.replace('from math import sqrt\n', '')
provider += '\n\n__all__ = ["HistoricalCandidateProvider", "HistoricalCandidateCache"]\n'
Path('src/malr/models/e23/historical_candidates.py').write_text(provider)
method = s[s.index('    def from_cache('):s.index('    def forward(')]
method = method.replace('        if cache.normalized_raw.shape != (batch, variables, patches, self.topk, self.patch_length):', '        count = getattr(self, "topk", cache.normalized_raw.shape[-2])\n        if cache.normalized_raw.shape != (batch, variables, patches, count, self.patch_length):')
method = method.replace('return self._empty(base)', 'return empty_historical(base, self.patch_length, self.d_model, count)')
method = method.replace('self.topk, self.d_model)', 'count, self.d_model)')
# Shared unregistered functions preserve the old expert parameter paths.
method = method.replace('    def from_cache(\n        self,', 'def historical_from_cache(\n    self,', 1)
lines = method.splitlines()
# Dedent original method body while retaining its multiline signature.
method = '\n'.join(line[4:] if line.startswith('    ') else line for line in lines)
method = method.replace('def historical_from_cache(\nself,', 'def historical_from_cache(\n    self,') + '\n'
empty = s[s.index('    def _empty('):s.index('    def _empty_cache(')]
empty = empty.replace('    def _empty(self, base: Tensor, *, candidate_count: Tensor | None = None)', 'def empty_historical(base: Tensor, patch_length: int, d_model: int, topk: int, *, candidate_count: Tensor | None = None)')
empty = '\n'.join(line[4:] if line.startswith('    ') else line for line in empty.splitlines())
empty = empty.replace('self.topk', 'topk').replace('self.patch_length', 'patch_length').replace('self.d_model', 'd_model') + '\n'
init = 'def initialize_historical_messages(self, patch_length: int, d_model: int, contextual: bool):\n' + '\n'.join(line[4:] for line in initialization.splitlines()) + '\n'
message = '''"""Independent historical messages over shared immutable candidate values."""\nfrom __future__ import annotations\nfrom math import sqrt\nimport torch\nfrom torch import Tensor, nn\nfrom .contracts import CandidateEvidence, ExpertEvidence, ExpertInputs\nfrom .historical_candidates import HistoricalCandidateCache\n\n'''
message += init + '\n' + empty + '\n' + method + '''\nclass HistoricalMessageEncoder(nn.Module):\n    def __init__(self, patch_length: int, d_model: int, *, contextual: bool):\n        super().__init__()\n        if min(patch_length, d_model) < 1:\n            raise ValueError("historical message geometry must be positive")\n        self.patch_length = patch_length\n        self.d_model = d_model\n        initialize_historical_messages(self, patch_length, d_model, contextual)\n\n    from_cache = historical_from_cache\n'''
Path('src/malr/models/e23/historical_messages.py').write_text(message)
p.write_text('''"""Historical expert compatibility wrapper; state keys retain their original paths."""\nfrom torch import Tensor\nfrom .contracts import ExpertEvidence, ExpertInputs\nfrom .historical_candidates import HistoricalCandidateCache, HistoricalCandidateProvider\nfrom .historical_messages import initialize_historical_messages, historical_from_cache\n\n\nclass HistoricalExpert(HistoricalCandidateProvider):\n    def __init__(self, patch_length: int, d_model: int, retrieval_dim: int, variable_count: int,\n                 *, topk: int = 4, similarity_threshold: float = .7,\n                 top_p: float | None = None, candidate_cap: int = 8,\n                 retrieval_strategy: str = "one_group", contextual: bool = False):\n        super().__init__(patch_length, d_model, retrieval_dim, variable_count,\n                         topk=topk, similarity_threshold=similarity_threshold,\n                         top_p=top_p, candidate_cap=candidate_cap,\n                         retrieval_strategy=retrieval_strategy)\n        initialize_historical_messages(self, patch_length, d_model, contextual)\n\n    from_cache = historical_from_cache\n\n    def forward(self, inputs: ExpertInputs) -> ExpertEvidence:\n        return self.from_cache(inputs, self.prepare_candidates(inputs))\n\n\n__all__ = ["HistoricalExpert", "HistoricalCandidateCache"]\n''')
