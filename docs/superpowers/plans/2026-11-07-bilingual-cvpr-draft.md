# Bilingual CVPR initial draft Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. No implementer subagents; request a final read-only review.

**Goal:** Create, compile, verify and push a reviewable English/Chinese R3 manuscript to a manuscript-only branch of the existing GitHub repository.
**Architecture:** Two language entrypoints and corresponding sections, shared equations/BibTeX/data-driven tables. Preserve source/experiment identity and expose incomplete evidence as TODOs.
**Tech Stack:** LaTeX, official CVPR2027 style, ctex, Tectonic, Python CSV tooling, Git/GitHub CLI.
**Spec:** docs/design.md

## Global constraints
- Original complete R3 including neighbor; no new model training.
- English official review style; Chinese companion may have different pagination.
- No fabricated baselines, references, claims of global SOTA or statistical significance.
- PAMAP2 formal20261014 versus analysis20261018 banks remain separate.
- Public manuscript-only branch; never overwrite existing branches or include weights/raw data/credentials.

## Review focus
- Formula matches source: observed preservation, train-global H coordinates, target-conditioned messages, q+r, signed tanh reference gate.
- 20 test masks versus3 training seeds; component PAMAP2 single-seed versus overall3-seed.
- Every CSV cell traceable to frozen snapshot and explicit scope.
- Both PDFs compile without undefined citations, references or overfull text.
- English/Chinese claims and equations align; incomplete baseline comparison remains visible.

### Task1: Evidence and table pipeline
- [ ] Copy finite source snapshots with hashes and protocol identity; verify formal archive audits.
- [ ] Test that missing/duplicate records or mixed bank identities cannot silently generate complete tables.
- [ ] Implement table generation and verify against actual paired summaries.

### Task2: Manuscript and compilation
- [ ] Fetch primary reference metadata, copy unmodified official style, write both corresponding versions and shared equations/tables.
- [ ] Compile both PDFs, inspect rendered pages/text and address diagnostics.
- [ ] Write Overleaf/build instructions and a concrete readiness ledger.

### Task3: Review and upload
- [ ] Read-only scientific/provenance review; resolve material issues.
- [ ] Run numerical/LaTeX checks and generate source ZIP/previews.
- [ ] Create a manuscript-only branch, commit, push to existing remote and verify remote SHA.
