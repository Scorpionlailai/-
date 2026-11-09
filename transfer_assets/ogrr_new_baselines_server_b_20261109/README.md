# OGRR new-baseline handoff — server B

Read `SERVER_B_PROMPT.md` first. This handoff is for **BeijingAirQuality and Exchange-Rate only**: 2 datasets × 4 scenes × SAITS/ModernTCN/TimeMixer++ = **24 cases**. Each case independently trains the original model, freezes it, trains OGRR, and evaluates both on the same 20 test masks. The original server owns ETTh1, Weather, PAMAP2 and PhysioNet2012 (48 cases); do not duplicate them.

## Assets
- `OGRR_server_b_20261109.tar.gz`: complete handoff, original data, shared masks, sealed NN implementations and dependencies. No pretrained backbone weights are required: these three new baselines train from scratch.
- `asset_index.json`: archive SHA256 and file digests.
- `BUNDLE_MANIFEST.json`: per-file SHA256, sizes and runtime versions.
- `profiles.json`: fixed author/default settings and explicitly recorded adaptations.
- `server_assignment.json`: nonoverlapping machine assignments.

## Setup
1. Download this directory **at the supplied commit**, rather than mixing files from another branch.
2. Verify archive SHA256 against `asset_index.json`.
3. Extract into a new staging directory first: `tar -xzf OGRR_server_b_20261109.tar.gz -C /path/to/staging`. Archive paths are relative to `/home/xyw` (`paper/...` and `r3_release_20261030/...`).
4. Verify each staged file against `BUNDLE_MANIFEST.json`. Inspect existing destinations before installation; retain differing existing files rather than overwriting them. If the server uses another home directory, document minimal path adapters without changing NN computations or mask generation.
5. Follow `SERVER_B_PROMPT.md`: run tests, mask alignment, per-case smoke, then two `partition_worker.py --server server_b` workers. **Do not run the all-dataset queue.**

The bundle has been read back and all manifest-listed file digests verified on the source server. Test scores must not guide seed or architecture changes. Save and return raw/refined paired metrics, stage costs, memory, checkpoints, configuration and source hashes.
