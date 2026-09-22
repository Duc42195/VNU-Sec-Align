#!/usr/bin/env bash
# .agents/hooks/install.sh
#
# .git/hooks/ khong duoc git theo doi (khong nam trong lich su commit) - moi lan clone repo
# nay ve may khac phai chay lai script nay 1 lan de kich hoat hook that su o .git/hooks/post-commit.
#
# Chay: bash .agents/hooks/install.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC="$REPO_ROOT/.agents/hooks/post-commit.sh"
DEST="$REPO_ROOT/.git/hooks/post-commit"

if [ ! -f "$SRC" ]; then
    echo "Khong tim thay $SRC - dang co dung repo/thu muc khong?" >&2
    exit 1
fi

cp "$SRC" "$DEST"
chmod +x "$DEST"

echo "Da cai dat hook: $SRC -> $DEST (da chmod +x)"
