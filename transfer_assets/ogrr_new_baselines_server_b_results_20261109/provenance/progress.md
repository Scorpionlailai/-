# Server B execution ledger

Source commit: f46822e78a4cb96b931ba8f54e233e1778ccf15c
Scope: BeijingAirQuality / Exchange-Rate only, 24 cases, original + frozen OGRR.

- Download/archive/976 staged file SHA256 verified; installed 975 new files, skipped 1 identical; no conflicts.
- All implementation_sha256.json entries verified at installed paths.
- Both GPUs idle; independent matching environment setup pending.
- Environment download recovery: stalled micromamba connection interrupted; same official Python package downloaded independently, original MD5 dcc0756cfcab6dd08cbdbda60a0bd390 verified; partial file retained.
- Runtime transport: official PyTorch downloads consistently slow; use SJTU mirror transport for the identical wheel, still requiring original official index SHA256 d6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3 before installation.
- Independent exact core runtime ready: Python3.10.21, torch2.7.1+cu128, NumPy2.2.6; pip check clean.
- Model tests: 5 passed; runtime data preflight: 8 scenes / 168 masks exact. All 976 installed bundle files reverified unchanged.
- Started server_b partition workers, slots0/1, one child per GPU; see worker_launch.json.
- Initial pipeline verification: both first Beijing cases passed smoke/export/freeze checks and entered full coarse training; two workers and dataset archive monitor alive; implementation hashes unchanged; no failed cases. See launch_verification.json for timestamped epochs.
