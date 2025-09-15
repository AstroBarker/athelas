#!/bin/sh

# Get the root of the Git repo
REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/.git/hooks"
REPO_HOOKS_DIR="$REPO_ROOT/scripts/hooks"

HOOK_NAME="pre-commit"
TARGET_HOOK="$REPO_HOOKS_DIR/$HOOK_NAME"
DEST_HOOK="$HOOKS_DIR/$HOOK_NAME"

echo "[install-hooks] Installing $HOOK_NAME hook..."

# Ensure target hook exists
if [ ! -f "$TARGET_HOOK" ]; then
    echo "[install-hooks] ERROR: $TARGET_HOOK not found!"
    exit 1
fi

# Only chmod if not already executable
[ -x "$TARGET_HOOK" ] || chmod +x "$TARGET_HOOK"

# Create symlink in .git/hooks
ln -sf "$TARGET_HOOK" "$DEST_HOOK"

echo "[install-hooks] Hook symlinked to $DEST_HOOK"
