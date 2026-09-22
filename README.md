# VNU-SecAlign / Vi-InjectEval

Cross-lingual (Vietnamese) generalization study of Meta's **SecAlign++** prompt-injection
defense, plus **Vi-InjectEval** — an independent, public Vietnamese prompt-injection benchmark.

See `paper/manuscript/` for the current draft and `docs/research_notes/` for the underlying
mechanism analysis and attack taxonomy this project builds on.

## Naming conventions

| Term | Refers to |
|---|---|
| **SecAlign (original, CCS'25)** | Chen et al., ACM CCS 2025 — chosen/rejected from existing dataset labels, LoRA on q/v_proj only, no 70B model |
| **SecAlign++ / Meta-SecAlign (arXiv 2507.02735)** | Chen et al., FAIR Meta — self-generated responses, randomized injection position, LoRA also on MLP, 8B + 70B checkpoints |
| **VNU-SecAlign v1** | The prior project iteration (formerly under `19-mar/`, now in `archive/`) — not reused as-is, reference only |
| **VNU-SecAlign v2** | This project |

The two Meta papers differ in several important ways (see `docs/research_notes/` and project memory) — always cite the specific one when quoting a number or hyperparameter; "Meta's original research" is only used when the distinction genuinely doesn't matter.

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
  benchmarks/          AlpacaFarm, SEP, CyberSecEval2 (prompt-injection subtask only — see below),
                       InjecAgent, Vi-InjectEval
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

- Any self-generated preference pair must follow the SecAlign++ structure: **chosen =
  task completion, rejected = injected-instruction compliance** — never a refusal template.
- **Never reuse the following as Vietnamese/prompt-injection preference data** — all excluded for
  concrete, verified reasons, not just caution:
  - `archive/legacy_vihos_vihsd_data/` (ViHOS/ViHSD) — wrong task (hate-speech, not prompt
    injection); `chosen` is a single fixed refusal sentence repeated across 100% of entries.
  - `archive/legacy_data/pku_en_preferences.json` — `chosen` values are genuine, diverse
    task-completions (not a refusal template), but the dataset is general harmlessness/safety
    (PKU-SafeRLHF), not prompt injection — it never trains the user/input trust distinction that
    is the actual SecAlign mechanism.
  - `archive/legacy_data/{combined_safety_dpo.json, dpo_dataset_clean.json,
    dpo_dataset_final.json, hallucination_preferences.json}` — these are concatenations that mix
    the (off-target but "clean") PKU data with refusal-template Vietnamese/V2-attack data; treat
    them as fully excluded, not as "already filtered."
  - `data/attack_vectors/_legacy_v2_reference/{dataset_v2_attacks.json, dataset_v2_with_rejected.json}`
    — **not usable as preference data at all**: `chosen` is the identical fixed refusal sentence
    "Tôi không thể trả lời câu hỏi này." across 100% of the 2,000 entries in both files. Only the
    `prompt`/`attack_type`/`original_prompt` fields are useful as a reference for layout.
- `data/attack_vectors/splits/heldout_20/` must stay untouched by every data-generation step.
  Final Phase-2 ASR is reported from held-out data only.
- ASR for the 4 encoding/obfuscation attack vectors (Encoding Chain, Unicode Homoglyph, Structured
  Data-Field Injection, Leetspeak) must be reported as **three numbers, not one**: raw ASR
  conflates whether the model can decode the obfuscated payload at all (a capability effect,
  independent of the defense) with whether it then respects the trust boundary. Report
  `decode_accuracy_rate` (isolated via a neutral, non-adversarial probe), `raw_asr`, and
  `conditional_asr` (ASR restricted to samples that decoded successfully — the number that
  actually reflects the defense). See `evaluation/attack_vectors_eval.py --decode_check` and
  project memory for the full rationale (including the real-world precedent: larger models can be
  *more* susceptible to cipher-style jailbreaks precisely because they decode better).
- Two cheap eval-time ablations are required to check the defense is learning the real
  user/input trust mechanism rather than a positional shortcut: (1) re-run with
  `instruction_hierarchy=False` (flattens `user`+`input` into one message — ASR should jump back
  toward baseline if the mechanism is real) and (2) re-run with the `user`/`input` message order
  swapped. Results go in `results/ablations/mechanism_validity/`.

## `archive/` — what's in it and why it's not deleted

Contents moved here during the 2026-08 reorganization of the original (pre-reorg) project layout.
Kept for provenance, not meant to be built on directly:

- `legacy_checkpoints/` — old LoRA adapters trained with hyperparameters now considered too far
  from Meta's published recipe (e.g. lr 5e-6, LoRA r=16); gitignored, local only.
- `legacy_data/` — old EN/mixed preference datasets, superseded by using the public
  Meta-SecAlign-8B checkpoint directly instead of retraining an EN baseline from scratch; also
  excluded from any new preference-data pipeline (see "Important methodology constraints" above
  for exactly why each file is excluded).
- `legacy_vihos_vihsd_data/` — explicitly excluded from the current methodology (see above).
- `legacy_notebooks/`, `legacy_results/`, `legacy_duplicate_results/`, `legacy_misc/` — as named.

## Known limitations & deliberate scope decisions

- **10 novel attack vectors, validated at 8B only.** Neither Meta paper confirms that a
  vulnerability found on an 8B-scale defended model behaves the same way at 70B — the papers only
  show 8B/70B ASR convergence on the standard benchmarks (AlpacaFarm/SEP/InjecAgent/MMLU), not on
  agentic benchmarks (AgentDojo/WASP) at 8B, and not on this project's own attack vectors at all.
  A stratified spot-check (e.g. 50–100 held-out samples per vector against the public
  `facebook/Meta-SecAlign-70B` checkpoint, via brief cloud inference rather than local hosting) is
  the intended way to get evidence here; if that spot-check isn't run, state explicitly that
  cross-scale generalization of the new attack vectors is untested — don't conflate it with the
  paper's own (narrower) 8B/70B convergence claim.
- **GRPO / online RL is deliberately out of scope**, not an oversight: it conflicts with the
  1-GPU budget (needs multi-completion sampling per step, several× the compute of offline DPO),
  and its reward would depend on `judge_injection_following()`, which is already known to be
  unreliable for the encoding/obfuscation attack classes — online RL is more prone to reward
  hacking than offline DPO, so using it before that judge issue is fixed would compound a known
  weakness rather than fix anything. None of RQ1–4 require an online-RL method.
- **Realistic venue/quartile expectations**: the original SecAlign paper is a top-tier venue
  publication (ACM CCS). This project, as scoped, is a controlled replication + extension rather
  than a new method — realistically a Q2–Q3 journal or a good NLP/security workshop paper, unless
  Phase 1's cross-lingual result is surprising, the mechanism-validity ablations above show clear
  depth, or the DPO+RPO+cDPO ablation shows a clear ASR improvement with flat MMLU. State this
  target honestly rather than overclaiming.

## External dependencies

`external/meta_secalign` is a git submodule pinned to
`facebookresearch/Meta_SecAlign@2031502` (the commit this project was built against). After
cloning:

```
git submodule update --init --recursive
```

Training in this project uses **TRL's `DPOTrainer`/`DPOConfig`** (`src/vi_secalign/training/`),
not the submodule's own `secalign_plus_plus.py` (which shells out to torchtune's
`lora_dpo_distributed` recipe). This is a deliberate deviation: torchtune's DPO loss exposes no
`rpo_alpha`/`label_smoothing` parameters (confirmed by grepping the whole submodule), so RPO and
cDPO — a real part of this project's contribution — are only reachable via TRL. The submodule is
still used directly for data generation (`utils.py`) and evaluation (`test.py`,
`test_lm_eval.py`, `test_injecagent.py`, `run_tests.py`), just not for training.
