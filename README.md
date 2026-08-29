# VNU-SecAlign / Vi-InjectEval

Cross-lingual (Vietnamese) generalization study of Meta's **SecAlign++** prompt-injection
defense, plus **Vi-InjectEval** — an independent, public Vietnamese prompt-injection benchmark.

See `paper/manuscript/` for the current draft and `docs/research_notes/` for the underlying
mechanism analysis and attack taxonomy this project builds on.

## Research questions

- **RQ1** — Does an English-only SecAlign++ security policy generalize zero-shot to Vietnamese?
- **RQ2** — If not, how much does adding Vietnamese preference data help, and at what utility cost?
- **RQ3** — Under a 1-GPU budget, which hyperparameters (Optuna, anchored near Meta's published
  values) get closest to Meta-SecAlign-8B's ASR?
- **RQ4** — Do 10 new attack vectors represent uncovered attack classes, and does fine-tuning
  reduce ASR on a held-out set disjoint from data generation?

## Repository layout

```
configs/            Training & eval configs (hyperparameter anchors, Optuna search space)
external/           Third-party code as git submodules (e.g. facebookresearch/Meta_SecAlign)
data/
  preference/        Preference pairs (self-generated chosen=task-completion / rejected=injection-compliant)
  attack_vectors/     10 novel attack vectors; splits/train_80 and splits/heldout_20 are kept
                       physically separate — heldout_20 must never be touched by data generation
  benchmarks/          AlpacaFarm, SEP, CyberSecEval2, InjecAgent, Vi-InjectEval
src/vi_secalign/     Installable package: data_gen, training, evaluation, models
checkpoints/         Downloaded/trained model weights (gitignored — see .gitignore)
results/             Per-phase outputs, one folder per RQ, plus ablations/
notebooks/           Exploratory notebooks (descriptively named, not "Untitled*")
docs/research_notes/ Internal analysis docs (SecAlign++ mechanism breakdown, attack taxonomy source)
paper/               manuscript/, related_work/, figures/, admin/
poster/, slide/      Conference/defense material
tools/hf_upload/     Utilities for publishing checkpoints/datasets to Hugging Face
archive/             Pre-reorg legacy artifacts, kept for reference — see caveats below
```

## Phase → output mapping

| Phase | What | Where results land |
|---|---|---|
| 0 | Download Meta-SecAlign-8B, sanity-check eval pipeline vs. published numbers | `results/phase0_sanity_check/` |
| 1 | Zero-shot eval on Vi-InjectEval (answers RQ1) | `results/phase1_zero_shot/` |
| 1.5 | Vietnamese preference data + training — **only if Phase 1 shows a real gap** (answers RQ2) | `results/phase1_5_vi_defense/` |
| — | Optuna hyperparameter search (answers RQ3) | `results/optuna_trials/` |
| 2 | 10 novel attack vectors, train/held-out split, judge audit (answers RQ4) | `results/phase2_heldout/` |

## Important methodology constraints (see project memory for full rationale)

- Any self-generated preference pair must follow the original SecAlign++ structure: **chosen =
  task completion, rejected = injected-instruction compliance** — never a refusal template.
- **Never reuse `archive/legacy_vihos_vihsd_data/`** (ViHOS/ViHSD hate-speech data) as a stand-in
  for Vietnamese prompt-injection data — wrong task, kept only for historical reference.
- `data/attack_vectors/splits/heldout_20/` must stay untouched by every data-generation step.
  Final Phase-2 ASR is reported from held-out data only.
- ASR for encoding/obfuscation attack vectors (Encoding Chain, Unicode Homoglyph, Base64,
  Leetspeak) needs manual judge audit before reporting — witness-word matching can false-positive
  on garbled model output.

## `archive/` — what's in it and why it's not deleted

Contents moved here during the 2026-08 reorganization of the original (pre-reorg) project layout.
Kept for provenance, not meant to be built on directly:

- `legacy_checkpoints/` — old LoRA adapters trained with hyperparameters now considered too far
  from Meta's published recipe (e.g. lr 5e-6, LoRA r=16); gitignored, local only.
- `legacy_data/` — old EN preference datasets, superseded by using the public Meta-SecAlign-8B
  checkpoint directly instead of retraining an EN baseline from scratch.
- `legacy_vihos_vihsd_data/` — explicitly excluded from the current methodology (see above).
- `legacy_notebooks/`, `legacy_results/`, `legacy_duplicate_results/`, `legacy_misc/` — as named.

## External dependencies

`external/meta_secalign` is a git submodule pinned to
`facebookresearch/Meta_SecAlign@2031502` (the commit this project was built against). After
cloning:

```
git submodule update --init --recursive
```
