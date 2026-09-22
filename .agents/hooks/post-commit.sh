#!/usr/bin/env bash
# .agents/hooks/post-commit.sh
#
# Nguồn sự thật cho logic nằm trong .agents/CLAUDE.md muc 2-3. File này CHỈ được git track khi
# đặt ở .agents/hooks/ (khac voi .git/hooks/, thu muc do khong duoc git theo doi) - xem
# .agents/hooks/install.sh de biet cach kich hoat that.
#
# Hanh vi (dung 6a-6f trong yeu cau goc):
#   a. Lay commit message moi nhat, parse theo pattern "[TaskID] <wip|done|blocked>: <mo ta>".
#      Khong khop pattern -> thoat im lang (exit 0, khong in gi).
#   b. Mo plan.csv o root, tim dong co TaskID khop. Khong tim thay -> in canh bao, khong sua gi.
#   c. wip -> neu Status dang "Not started", doi thanh "In progress".
#   d. done -> CHI doi thanh "Done" neu cot "DoD (check)" cua dong do khong rong VA lenh trong do
#      chay tra ve exit code 0. Nguoc lai: giu/dat "In progress", in canh bao ro ly do.
#   e. blocked -> doi Status thanh "Blocked".
#   f. Append 1 dong vao .agents/action-history.md: timestamp, short hash, TaskID, status moi.
#
# CSV duoc parse/ghi bang Python (nhung mo dau la file .sh nhu yeu cau) vi cac truong trong
# plan.csv co the chua dau phay ben trong ngoac kep (vd cot "Task", "Ghi chu") - parse bang
# bash thuan (cut/awk theo dau phay) se sai/hong du lieu.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

python3 - "$REPO_ROOT" <<'PYEOF'
import csv
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(sys.argv[1])
plan_path = repo_root / "plan.csv"
history_path = repo_root / ".agents" / "action-history.md"

# --- a. Lay va parse commit message moi nhat ---
commit_subject = subprocess.run(
    ["git", "-C", str(repo_root), "log", "-1", "--pretty=%s"],
    capture_output=True, text=True, check=True,
).stdout.strip()

match = re.match(r"^\[([^\]]+)\]\s*(wip|done|blocked)\s*:\s*(.*)$", commit_subject)
if not match:
    sys.exit(0)  # khong khop pattern -> thoat im lang

task_id, action, _description = match.group(1), match.group(2), match.group(3)

# --- b. Mo plan.csv, tim dong co TaskID khop ---
if not plan_path.exists():
    print(f"[post-commit] CANH BAO: khong tim thay {plan_path}, khong sua gi.")
    sys.exit(0)

with plan_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))

if not rows:
    print(f"[post-commit] CANH BAO: {plan_path} rong, khong sua gi.")
    sys.exit(0)

header = rows[0]
try:
    task_id_col = header.index("TaskID")
    status_col = header.index("Status")
    dod_col = header.index("DoD (check)")
except ValueError as e:
    print(f"[post-commit] CANH BAO: plan.csv thieu cot bat buoc ({e}), khong sua gi.")
    sys.exit(0)

target_row_idx = None
for i, row in enumerate(rows[1:], start=1):
    if len(row) > task_id_col and row[task_id_col] == task_id:
        target_row_idx = i
        break

if target_row_idx is None:
    print(f"[post-commit] CANH BAO: khong tim thay TaskID '{task_id}' trong plan.csv, khong sua gi.")
    sys.exit(0)

row = rows[target_row_idx]
# dam bao row du dai bang header (phong truong hop cell rong cuoi dong bi cat bot khi doc)
while len(row) < len(header):
    row.append("")

current_status = row[status_col]
new_status = current_status  # mac dinh: khong doi neu khong khop dieu kien nao ben duoi

# --- c. wip ---
if action == "wip":
    if current_status == "Not started":
        new_status = "In progress"

# --- d. done ---
elif action == "done":
    dod_cmd = row[dod_col].strip()
    if not dod_cmd:
        new_status = "In progress"
        print(f"[post-commit] CANH BAO: TaskID {task_id} commit 'done' nhung cot 'DoD (check)' dang RONG "
              f"-> giu Status = 'In progress', KHONG set Done.")
    else:
        result = subprocess.run(dod_cmd, shell=True, cwd=repo_root)
        if result.returncode == 0:
            new_status = "Done"
        else:
            new_status = "In progress"
            print(f"[post-commit] CANH BAO: TaskID {task_id} commit 'done' nhung lenh DoD "
                  f"({dod_cmd!r}) that bai (exit code {result.returncode}) "
                  f"-> giu Status = 'In progress', KHONG set Done.")

# --- e. blocked ---
elif action == "blocked":
    new_status = "Blocked"

row[status_col] = new_status
rows[target_row_idx] = row

with plan_path.open("w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

# --- f. Append vao action-history.md ---
short_hash = subprocess.run(
    ["git", "-C", str(repo_root), "log", "-1", "--pretty=%h"],
    capture_output=True, text=True, check=True,
).stdout.strip()
timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

with history_path.open("a", encoding="utf-8") as f:
    f.write(f"| {timestamp} | {short_hash} | {task_id} | {new_status} |\n")

print(f"[post-commit] {task_id}: {current_status!r} -> {new_status!r} (action={action}, commit={short_hash})")
PYEOF
