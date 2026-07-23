#!/bin/bash
# =============================================================================
# Isotrieve Agent Launch Script
# =============================================================================
# This script:
#   1. Starts the MLX model server in the background
#   2. Waits for it to be ready
#   3. Launches hermes-agent in the worktree with the iteration task
#   4. Cleans up on exit
#
# Usage:
#   ./start-agent.sh
#
# Prerequisites:
#   - mlx-lm installed (pip install mlx-lm --break-system-packages)
#   - hermes-agent installed (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash)
#   - isotrieve dependencies installed in .agent-venv
# =============================================================================

set -euo pipefail

WORKTREE="/Users/kpatel/Desktop/agent-communication-agent-worktree"
MAIN_REPO="/Users/kpatel/Desktop/agent-communication"
VENV_PYTHON="$MAIN_REPO/.agent-venv/bin/python3.13"
HERMES_BIN="$HOME/.local/bin/hermes"
MODEL="mlx-community/Hermes-4-14B-4bit"
PORT=8080
MAX_WAIT=120  # seconds to wait for model server

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}->${NC} $1"; }
log_ok()    { echo -e "${GREEN}✓${NC} $1"; }
log_warn()  { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }

cleanup() {
    log_info "Cleaning up..."
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log_info "Stopping model server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---- Preflight checks ----

if [ ! -d "$WORKTREE" ]; then
    log_error "Worktree not found at $WORKTREE"
    log_info "Create it with: cd $MAIN_REPO && git worktree add $WORKTREE -b agent/iteration-branch"
    exit 1
fi

if [ ! -x "$HERMES_BIN" ]; then
    log_error "hermes command not found at $HERMES_BIN"
    log_info "Install with: curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash"
    exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
    log_error "Python venv not found at $VENV_PYTHON"
    log_info "Create with: python3.13 -m venv $MAIN_REPO/.agent-venv && $MAIN_REPO/.agent-venv/bin/pip install -e '$MAIN_REPO/isotrieve-python[dev]'"
    exit 1
fi

# ---- Start model server ----

log_info "Starting MLX model server on port $PORT..."
mlx_lm.server --model "$MODEL" --port "$PORT" --host 127.0.0.1 &
SERVER_PID=$!
log_ok "Model server started (PID $SERVER_PID)"

# Wait for server to be ready
log_info "Waiting for model server to be ready..."
elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        log_ok "Model server ready (${elapsed}s)"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    log_error "Model server did not become ready within ${MAX_WAIT}s"
    exit 1
fi

# ---- Launch hermes-agent ----

log_info "Launching hermes-agent in worktree..."
log_info "Worktree: $WORKTREE"
log_info "Model: $MODEL @ localhost:$PORT"
log_info "Max iterations: 20"
echo ""

cd "$WORKTREE"
"$HERMES_BIN" --accept-hooks
