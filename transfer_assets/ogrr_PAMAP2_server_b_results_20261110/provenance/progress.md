# PAMAP2 Server B execution ledger

Source addon commit: b97f79f4a65df143042c97dee49e9ed0d1ab0aab
Frozen training implementation: f46822e78a4cb96b931ba8f54e233e1778ccf15c
Scope: PAMAP2 only, 12 new cases. Old 24 complete cases retained.

- Archive and 493 file hashes verified; 488 installed new, 5 identical reused; no differing destinations.
- Original assignment and global summaries/statuses backed up under backups/. Installed supplied new assignment; server_a/server_b each36, disjoint union72.
- Each slot has6 new PAMAP2 cases and12 existing complete cases to skip. No old partition workers active; both GPUs idle.
- Prior24 selected/latest/results/complete SHA256 and mtime snapshot saved before installation.
- CPU preflight passed for all4 scenes: 40 vars, train10958/validation1306/test3349, 84 original masks byte/array exact; native-valid scoring and record boundaries checked.
- Launched two server_b workers with explicit --gpu0/1; each skips12 previous complete cases and executes6 PAMAP2 cases. See worker_launch.json.
- Initial launch verified: first SAITS/ModernTCN smoke/freeze/export checks passed; both entered full coarse training. Prior24 hashes+mtimes and all sealed implementation hashes rechecked unchanged; no failures.
- PAMAP2-only background monitor writes progress_pamap2.json and will validate/archive all12 cases using the prior result delivery format. No GitHub result upload started for this addon.
