#!/usr/bin/env bash
# setup_collector.sh — build the COLLECTION environment from public sources.
#
# The trainer venvs (sft/opd/rlvr per-project) are separate and tiny; this
# script builds the other half of the two-environment reality: the venv that
# runs `gsj-run --driver uniagent` (the collector process), plus the shared
# on-disk assets every project's config.yaml points at:
#
#   collector-venv/           python venv: uni-agent stack + the library wheel
#   vendor/uni-agent/         uni-agent source @ the pinned sha (+ verl submodule)
#   assets/templates/         the .pi render templates (from the library repo @ v0.4.0)
#   assets/pages/             the MCP page corpus (release tarball, sha256-verified)
#   assets/gsj-mcp/           the MCP shim, extracted FROM the sandbox image
#                             (the H-28 workaround — see FINDINGS.md)
#   + the Qwen3-0.6B tokenizer snapshot (HF cache; path printed at the end)
#
# Sources are public URLs only. Run on the GPU host, from the repo root.
# Idempotent: every step is skipped or no-ops when already done.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"

# ---- the published identities (library docs/publishing.md) ------------------
LIB_REPO="https://github.com/MHGanainy/gsj-envloader"
LIB_TAG="v0.4.0"
RAW="https://raw.githubusercontent.com/MHGanainy/gsj-envloader/${LIB_TAG}"
WHEEL_URL="${LIB_REPO}/releases/download/${LIB_TAG}/gsj_envloader-0.4.0-py3-none-any.whl"
PAGES_URL="${LIB_REPO}/releases/download/${LIB_TAG}/gsj-pages-20260204.tar.gz"
PAGES_SHA256="8c394701f81c29a6ddc8e0100337cb8e36206da18f0bb4ca441394467efa0f83"
IMAGE="ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-mcp1.5.0-2"
# uni-agent: GitHub source is the ONLY channel (no tags/releases; PyPI
# 'uni-agent' is an unrelated name-squat). Sha = env.provenance.uniagent_sha.
UNIAGENT_REPO="https://github.com/verl-project/uni-agent.git"
UNIAGENT_SHA="73b0f41efa88b311fd69129c6f835c012e925e73"
# codec tokenizer source (driver.snapshot_path): the library's dev model pin
# (devharness/vllm/model-0.6b.env @ v0.4.0 — id + revision, resolved locally)
MODEL_ID="Qwen/Qwen3-0.6B"
MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"

log() { echo "[setup] $*"; }

# ---- 1. the venv ------------------------------------------------------------
if [ ! -d collector-venv ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --seed --python 3.12 collector-venv
  elif [ -x "$HOME/.local/bin/uv" ]; then
    "$HOME/.local/bin/uv" venv --seed --python 3.12 collector-venv
  else
    python3 -m venv collector-venv
  fi
fi
PIP="$ROOT/collector-venv/bin/pip"
PY="$ROOT/collector-venv/bin/python"

# ---- 2. uni-agent source @ the pin (+ verl submodule) -----------------------
if [ ! -d vendor/uni-agent/.git ]; then
  mkdir -p vendor
  git clone "$UNIAGENT_REPO" vendor/uni-agent
fi
git -C vendor/uni-agent checkout --quiet "$UNIAGENT_SHA"
git -C vendor/uni-agent submodule update --init --depth 1 verl

# ---- 3. install: uni-agent (editable), verl (--no-deps), frozen pins, wheel -
# uni-agent's pyproject declares NO dependencies; the working dependency set
# is the library's frozen devharness/uniagent/requirements.txt — fetched from
# the public library repo at the tag (the file is the CP-05 environment
# freeze; there is no separately published lockfile artifact).
mkdir -p assets
if [ ! -f assets/uniagent-requirements.txt ]; then
  curl -fsSL "${RAW}/devharness/uniagent/requirements.txt" -o assets/uniagent-requirements.txt
fi
"$PIP" install --disable-pip-version-check -q -e vendor/uni-agent
"$PIP" install --disable-pip-version-check -q --no-deps -e vendor/uni-agent/verl
"$PIP" install --disable-pip-version-check -q -r assets/uniagent-requirements.txt
"$PIP" install --disable-pip-version-check -q "gsj-envloader @ ${WHEEL_URL}"
log "collector-venv: $("$PY" -c 'import gsj.envloader as g; print("gsj-envloader", g.__version__)')"

# ---- 4. the .pi render templates (library repo @ the tag) -------------------
mkdir -p assets/templates
for f in settings.json.tmpl mcp.json.tmpl models.json.tmpl; do
  [ -f "assets/templates/$f" ] || curl -fsSL "${RAW}/devharness/pi/templates/$f" -o "assets/templates/$f"
done
log "templates: $(ls assets/templates)"

# ---- 5. the page corpus (release tarball, verified) -------------------------
if [ ! -d assets/pages ]; then
  curl -fsSL "$PAGES_URL" -o assets/pages.tar.gz
  echo "${PAGES_SHA256}  assets/pages.tar.gz" | sha256sum -c -
  tar -xzf assets/pages.tar.gz -C assets/
  rm assets/pages.tar.gz
fi
log "pages: $(find assets/pages -name 'page_*.md' | wc -l | tr -d ' ') files"

# ---- 6. the MCP shim, extracted from the image (H-28 workaround) ------------
# The library bind-mounts mcp_launch.server's parent dir over /opt/gsj-mcp
# unconditionally, and the config cannot name the in-image path (HOLES H-28).
# Workaround: extract the image's own copy and mount it back — the mount then
# re-mounts the image's own bytes.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "image $IMAGE not in the local daemon; trying anonymous pull"
  docker pull "$IMAGE" || {
    log "ERROR: cannot pull. On egress-locked hosts, ship it from any host"
    log "that can: docker save $IMAGE | ssh <this-host> docker load"
    exit 1
  }
fi
if [ ! -f assets/gsj-mcp/server.py ]; then
  mkdir -p assets/gsj-mcp
  cid=$(docker create "$IMAGE")
  docker cp "$cid":/opt/gsj-mcp/server.py    assets/gsj-mcp/server.py
  docker cp "$cid":/opt/gsj-mcp/decisions.py assets/gsj-mcp/decisions.py
  docker rm "$cid" >/dev/null
fi
log "mcp shim: $(ls assets/gsj-mcp)"

# ---- 7. the codec tokenizer snapshot ----------------------------------------
SNAPSHOT=$("$PY" - "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
PY
)
log "codec snapshot: $SNAPSHOT"

echo
log "DONE. Now check every */config.yaml:"
log "  - absolute paths must match THIS checkout root: $ROOT"
log "  - driver.snapshot_path must be: $SNAPSHOT"
log "seed a project with:"
log "  collector-venv/bin/gsj-run --config <project>/config.yaml --driver uniagent"
log "(regenerate: wait_all — Ctrl-C it once the round completes; see the project READMEs)"
