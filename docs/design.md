# CVPR bilingual draft v0.1

User approved an initial manuscript uploaded to their existing GitHub repository, then iterative revision. English uses the unmodified official CVPR2027 review style; Chinese is a readable section-by-section companion, with the same method, formulas, evidence and caveats. Model is original complete R3/H_GLOBAL WITH learned Confidence, q+r residual head and backend visible-neighbor correction. No new training.

One manuscript-only Git branch in Scorpionlailai/-; no change to default branch or experiment code. English main.tex/main_en.tex, Chinese main_zh.tex; shared math, BibTeX and generated numerical tables; previews and an Overleaf ZIP are produced locally. Overleaf credentials/project not yet supplied, so this task pushes GitHub only.

Scope: Abstract, Introduction, Related Work, Method, Experiments, Model Analysis, Limitations, Conclusion and required LLM Usage Statement. Use audited six-dataset R3 formal metrics, new paired R3/E6 analyses and component ablations, with explicit bank separation. PAMAP2 formal table uses20261014 masks whereas mechanism comparison uses20261018; do not merge their numeric identities. Main full-baseline comparison remains explicitly pending an aligned export, not filled from old v125.

Every numerical table is regenerated from checked CSV snapshots; source URLs, hashes, caveats and a readiness ledger accompany the manuscript. References are checked against primary pages/BibTeX. The architecture artwork supplied in chat has no local source; use a labeled placeholder until final vector artwork is available. Do not generate new mechanism plots.

The existing GitHub repository is public; the manuscript branch will also be public under the user's upload request. Local original vendor/template source is preserved but not uploaded wholesale. Only the official style/bibliography style files needed to compile are copied, with attribution.
