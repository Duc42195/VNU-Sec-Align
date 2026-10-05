# Working process
Gotchas and working flows learned the hard way (setup, environment, review habits). Append only.

Format:
## YYYY-MM-DD — short title
What happens, what to do instead.

## 2026-10-05 — Re-verify a prior session's technical claims against primary source before spending pod money on them
A plan written in an earlier session (torchtune T9b plan, `radiant-sprouting-firefly.md`) claimed
`checkpointer.adapter_checkpoint` could load Meta's published LoRA adapter as a training init,
"verified via git clone/gh api". When actually writing the code (this session), re-checking the
same source turned up two real blockers that claim missed: (1) the recipe only honors that field
when `resume_from_checkpoint=True`, which also demands a recipe-state file Meta's adapter doesn't
have; (2) even past that, the field loads the checkpoint with zero key-name/permutation conversion
(torchtune ships `tune_to_peft_adapter_weights` for export, explicitly has no reverse function) —
so a PEFT-format adapter's weights would land under the wrong keys entirely. Do this: treat "I
verified this last session" as a claim to recheck, not a fact to build on, whenever the next step
would spend real money (GPU rental) or produce a result that's expensive to redo -- a 5-minute
re-fetch of the pinned source is cheap insurance against discovering a wrong assumption mid-run.

## 2026-10-05 — Verify a pinned library version's internals via raw GitHub, no clone/install needed
To check torchtune==0.6.0 behavior (recipe logic, checkpointer internals, dataset schema) before
committing to a design, fetched the exact tagged files directly:
`curl -s https://raw.githubusercontent.com/pytorch/torchtune/v0.6.0/<path>` — no local pip install,
no git clone, no GPU, no HF token. This is enough to settle "does this field actually do what the
docstring/plan claims" questions straight from source, cheaply, before writing any training code or
touching a pod. Prefer this over trusting a library's docstring example or a remembered summary
when a wrong assumption would be expensive to discover later (training code, hyperparameter
choices, anything gating a paid run).

## 2026-10-05 — A vendored/patched third-party file: diff against the real upstream to know exactly what changed
`external/meta_secalign/helpers/_preference.py` looked at a glance like it might be a stale/custom
copy of torchtune's dataset loader. Rather than guessing its intent, fetched the real
`torchtune/datasets/_preference.py` at the matching pinned version and ran a plain `diff` against
it -- confirmed in one step that this is a deliberate Meta patch (prompt/chosen/rejected as plain
strings, not chat-message lists) and that our own data file's schema already matches it exactly, no
conversion script needed. Do this whenever a vendored file's purpose/currency is unclear and a
same-version upstream copy is fetchable -- a diff answers "what's actually different" in one shot,
cheaper than reading both in full or asking.

## 2026-10-05 — `colab exec -f <file>` only ever transmits that one file; bundle, don't reference
Confirmed again (same root cause as the T10 "colab upload gives 500" lesson, `notebooks/T10_MANUAL.md`
step 4): `colab exec -f path/to/script.py` sends only that script's own content to the remote
session -- it cannot pull in a second local file by path, even a small one, because the remote
session has no access to this machine's filesystem at all. When a remote script needs another
file's content (a patch to apply, a data file), embed that content literally inside the one script
passed to `-f` (e.g. `tools/pod_setup/apply_torchtune_preference_patch.py` embeds the full patched
file as a string constant) rather than writing a script that tries to read a sibling path.

## 2026-10-05 — `colab exec` CLI cannot pass extra args to the `-f` script; use `--env` or bundle
`colab exec -f script.py -- --config x=y` fails with typer's "Got unexpected extra
argument(s)" -- the only flags are `-s/--session`, `-f/--file`, `--env KEY=VALUE`, etc.
When a remote script needs override values or sys.argv from the recipe CLI, do one of:
(a) generate a bundle file locally (header that sets `sys.argv` + stripped-`__main__`
recipe source + explicit `sys.exit(recipe_main())`; see
`tools/pod_setup/make_t9b_recipe_bundle.py`) and `colab exec -f` that bundle; (b) pass values
via `--env` and have the remote script read `os.environ`. `--env` also works for the inline
`echo '...' | colab exec` form. Do this: never put `--` or recipe-style `key=value` overrides
after the `-f` argument.

## Standup checks
Read by `/plan-check` when it is installed. Add 3 to 6 checks that fit this project as bullet lines starting with `- ` below this paragraph. One check per line: a read-only command to run or a file to read, and what counts as a failure. `/plan-check` runs them and prints only the ones that fail.
Examples (not active, copy the ones you want): the gate passes on the default branch; a task is `done` but its `review` is still `pending`; a task is in progress while a task in its `depends` is not done; the tracker and `plan.csv` disagree (only with an external tracker); every number in the report cites a run-id that is a chosen result in `decisions-log.md` (research projects).
