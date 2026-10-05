---
description: Wrap up a task - commit directly to main, append a results note to plan.csv (never set Status=Done)
argument-hint: <task-id>
allowed-tools: Bash, Read, Edit
---
Wrap up task `$ARGUMENTS`. Stop at the first step that fails and report it.

1. Read `plan.csv` and confirm a row with id `$ARGUMENTS` exists. If not, stop.
2. Read `.agents/CLAUDE.md` in full if you haven't this session — it has the hard rules (frozen dirs, commit format, Status discipline) that override anything generic below.
3. No automated gate exists for this project — verify correctness manually (the work actually ran / produced the claimed output), not via a test command.
4. Commit directly to `main` (no branch, no PR): `[$ARGUMENTS] done: <short description>`, imperative, lowercase description.
5. Append a one-line results note to the task's `Ghi chú` cell in `plan.csv` (what ran, key numbers, artifact path). **Do not touch the `Status` cell** — leave it for the user to flip to `Done` (see `.agents/CLAUDE.md` §2). You may push `Status` to `In progress` if it was still empty/`Not started`.
6. If this closes out a methodological decision or experiment result, add an entry to `.agents/record.md` (Context/Decision/Rejected alternatives/Consequences, or a `Result:` line with run-id/config/metrics/artifact path) rather than `.agents/wiki/decisions-log.md`, to keep the project's one existing decision history intact.
7. Report: task id, commit hash, what changed in `plan.csv`'s `Ghi chú`, and the next task per `Dependency`/plan order that's now unblocked.
