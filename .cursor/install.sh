#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the astrogwb checkout.
#
# Installs the uv package manager and the `just` task runner (both to
# ~/.local/bin), makes them available on PATH for future shells, then syncs the
# full development environment exactly as AGENTS.md and CI describe:
#   uv sync --extra notebook --group dev
#
# Kept idempotent: re-running skips an already-installed uv/just and lets uv
# reconcile the venv against uv.lock without rewriting the lockfile.
set -euo pipefail

LOCAL_BIN="$HOME/.local/bin"
export PATH="$LOCAL_BIN:$PATH"

# Ensure ~/.local/bin is on PATH for every future non-login shell the agent opens.
BASHRC="$HOME/.bashrc"
PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if [ -f "$BASHRC" ] && ! grep -qF "$PATH_LINE" "$BASHRC"; then
    printf '\n# Added by astrogwb Cloud Agent install\n%s\n' "$PATH_LINE" >>"$BASHRC"
fi

# 1. uv: the project's package manager (pins Python >=3.12 and reads uv.lock).
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 2. just: every CI check is a `just` recipe; keep it available for the agent.
if ! command -v just >/dev/null 2>&1; then
    uv tool install rust-just
fi

# 3. The full development environment: notebook extra + dev group, per AGENTS.md.
uv sync --extra notebook --group dev

echo "astrogwb dev environment ready: $(uv --version), $(just --version)"
