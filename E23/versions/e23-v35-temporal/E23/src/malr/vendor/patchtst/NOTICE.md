# PatchTST Vendor Notice

- Upstream: https://github.com/yuqinie98/PatchTST
- Pinned commit: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`
- License: Apache License 2.0; see `LICENSE` in this directory.
- Imported files:
  - `PatchTST_supervised/layers/PatchTST_layers.py`
  - `PatchTST_supervised/layers/PatchTST_backbone.py`
  - `PatchTST_supervised/layers/RevIN.py`
- Local modifications: attribution headers and package-relative imports. The local `malr.models.patchtst.PatchTSTCore` exposes the upstream projection, positional encoding, channel-independent Transformer encoder, and flatten head as separate call boundaries without changing their equations.

