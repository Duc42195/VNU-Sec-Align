# AGENTS.md — VNU-SecAlign / Vi-InjectEval

The single rulebook for every person and every AI tool in this repo. Other AI config files only point here.

## Profile
- What: Cross-lingual (Vietnamese) generalization study of Meta's SecAlign++ prompt-injection defense, plus Vi-InjectEval, an independent public Vietnamese prompt-injection benchmark
- Goal: Publish Vi-InjectEval and the cross-lingual SecAlign++ paper/thesis, answering RQ1-4 (zero-shot generalization, VN preference data, 1-GPU hyperparameter search, 10 new attack vectors)
- Stack: Python (torch, transformers, trl, peft, vllm) - ML research
- Research project: yes
- Task id prefix: `T` · Git host: GitHub · Default branch: `main` · External tracker: none
- Gate command: ``
- Python: `python3` (only `/plan-check` needs it)

## Where things live (read the relevant one before you act)
| Need | Where |
|---|---|
| **Hard project rules — frozen dirs, Status discipline, commit format, sensitive data** | **`.agents/CLAUDE.md` — read this in full before touching anything; it overrides the generic rules below wherever they'd conflict** |
| What is planned, how far, DoD | `plan.csv` (source of truth for tasks — Vietnamese columns: Giai đoạn, Sprint/Tuần, TaskID, Task, Ưu tiên, Ước lượng, Dependency, Deliverable, DoD (check), Status, Ghi chú). `.agents/tools/plan.sh` assumes the generic template's columns (status/owner/review/mr) and does **not** match this file — don't use it here. |
| Full decision history (Context/Decision/Rejected alternatives/Consequences, #1–#35+) | `.agents/record.md` — the project's real decision log; append-only, same discipline as `.agents/wiki/decisions-log.md` below but pre-dates this scaffold. Keep using `record.md` as primary; `decisions-log.md` is available for any decision outside its scope. |
| Machine-logged action history | `.agents/action-history.md` |
| Lessons, gotchas, open questions (new, scaffold-provided) | `.agents/wiki/` |
| Who is who | `.agents/roles.md` |

## Working rules
1. Work from `plan.csv`. Read `.agents/CLAUDE.md` §2 before touching the `Status` column.
2. Commit directly to `main` (no feature branches, no PR/MR — this project doesn't use that flow): `[TaskID] <wip|done|blocked>: <desc>`, e.g. `[T10] done: VN_ASR held-out eval for T9 checkpoint`. See `.agents/CLAUDE.md` §3.
3. **Never set `Status = Done`** in `plan.csv` (agent-only cap: `In progress`, via commit). `Done` is set only by the user, or by a DoD-check hook once one exists — see `.agents/CLAUDE.md` §2. A `/done <task-id>` style close should commit and append a results note to the task's `Ghi chú` cell, never flip `Status` itself.
4. No CI gate (research project, no test suite) — correctness is checked via real pod runs, `record.md` decisions, and manual verification, not an automated command.
5. Git (`record.md` + commits) and `plan.csv`'s `Ghi chú`/DoD are the truth. If `plan.csv`'s `Status` disagrees with reality, flag it to the user — don't self-correct it to `Done`.
6. No secrets in the repo. Tokens/passwords live in environment variables, `.agents/state/` (git-ignored), or the user's own memory — never hardcoded or committed.
7. When you edit `plan.csv`, keep one task per row and the header exactly as it is; this project's own governance in `.agents/CLAUDE.md` takes precedence over the generic plan.csv conventions this scaffold otherwise assumes (owner/review/mr/depends columns do not exist here).

## Knowledge capture (hard rule)
In the same session it happens, append (never delete, never rewrite history):
- a decision → `.agents/wiki/decisions-log.md`
- an error you hit and fixed → `.agents/wiki/learnings.md` as symptom · root cause · fix · lesson
- an open question opened, changed or closed → `.agents/wiki/open-questions.md`
- a process gotcha → `.agents/wiki/working-process.md`
To reverse an earlier call, append a superseding entry that points to the old one.

## Decision integrity
- One topic, one current decision. Before recording one, read `decisions-log.md` for the same `Topic:`. If it is already decided, add a new entry with `Supersedes: <date — title>`; never edit or delete the old entry.
- Every decision has a `Topic:` line, and topics must fit together: if a new decision conflicts with another topic, resolve it before recording it.
- If `Research project` is `yes`: the chosen result of an experiment is a decision with a `Result:` line (run-id, config, metrics, artifact path); a better run supersedes it. Every table, number and figure in a report, paper or wiki page carries its run-id and must equal the chosen result's. If it does not, regenerate it from that run.

## Reading the team record
- Who does what: `.agents/roles.md` (currently one maintainer, owns everything).
- How far: `plan.csv`'s `Status` and `Ghi chú` columns, plus `.agents/record.md`'s latest Decision entries (highest-numbered = most recent state).
- Is it right: `DoD (check)` column in `plan.csv` (mostly empty today — no automated gate yet).
- Who blocks whom: `Dependency` column in `plan.csv`.
