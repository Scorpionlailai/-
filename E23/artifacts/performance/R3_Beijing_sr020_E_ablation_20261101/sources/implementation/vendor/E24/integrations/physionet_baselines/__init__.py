"""Author repair baselines on the shared ETTh1/Weather block-mask protocol."""
import sys
from pathlib import Path
from types import ModuleType

# On the author's Python 3.8 environment, load only the unchanged PatchTST
# submodules: the normal package initializer eagerly loads unrelated E24
# retrieval models requiring Python 3.10. This changes no model source.
if sys.version_info < (3, 10):
    package = ModuleType('malr.models')
    package.__path__ = [str(Path(__file__).resolve().parents[2] / 'src/malr/models')]
    sys.modules['malr.models'] = package
