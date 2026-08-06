#!/usr/bin/env bash
# setup_collector.sh — build the COLLECTION environment from public sources.
#
# The trainer venvs (sft/opd/rlvr per-project) are separate and tiny; this
# script builds the other half of the two-environment reality: the venv that
# runs `gsj-collect` / `gsj-run --driver uniagent` (the collector process),
# plus the shared on-disk assets every project's config.yaml points at.
#
# v0.5.0 (the library's CP-28 closure) DELETED four of this script's CP-27
# workarounds:
#   - render templates: no longer fetched — they ship as package data inside
#     the wheel (configs simply omit `templates_dir`)
#   - MCP shim: no longer extracted from the image — configs name the baked
#     copy directly (`mcp_launch.server: /opt/gsj-mcp/server.py` +
#     `in_image: true`)
#   - gate pins: now a published release asset (was: a committed byte-copy
#     of the library source tree)
#   - frozen collector deps: installed by the library's own published
#     install_collector_env.sh (was: an inline re-derivation of the
#     sglang --no-deps nuance)
# and the codec snapshot download is gone too: configs carry the
# host-portable driver `{model_id, revision}` pin pair and the driver
# resolves it (HF cache, download on miss) at first run.
#
# What this leaves on disk:
#   collector-venv/           python venv: uni-agent stack + the library wheel
#   vendor/uni-agent/         uni-agent source @ the pinned sha (+ verl submodule)
#   assets/pages/             the MCP page corpus (release tarball, sha256-verified)
#   assets/pins.dev.json      the dev gate pins (release asset, sha256-verified)
#                             — dev-environment DATA: these validate the
#                             published dev fixtures/codec; your own
#                             environment regenerates its own pins (gsj-pin)
#
# Sources are public URLs only. Run on the GPU host, from the repo root.
# Idempotent: every step is skipped or no-ops when already done.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"

# ---- the published identities (library docs/publishing.md) ------------------
LIB_REPO="https://github.com/MHGanainy/gsj-envloader"
LIB_TAG="v0.5.0"
RAW="https://raw.githubusercontent.com/MHGanainy/gsj-envloader/${LIB_TAG}"
WHEEL_URL="${LIB_REPO}/releases/download/${LIB_TAG}/gsj_envloader-0.5.0-py3-none-any.whl"
PAGES_URL="${LIB_REPO}/releases/download/${LIB_TAG}/gsj-pages-20260204.tar.gz"
PAGES_SHA256="8c394701f81c29a6ddc8e0100337cb8e36206da18f0bb4ca441394467efa0f83"
PINS_URL="${LIB_REPO}/releases/download/${LIB_TAG}/gsj-pins-0.5.0.json"
PINS_SHA256="766ca30ea8ca15bcca249f167c8f42d23f2cb97eb3c33adf8bc97f8c4fa15d44"
IMAGE="ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-mcp1.5.0-2"
# uni-agent: GitHub source is the ONLY channel (no tags/releases; PyPI
# 'uni-agent' is an unrelated name-squat). Sha = env.provenance.uniagent_sha.
UNIAGENT_REPO="https://github.com/verl-project/uni-agent.git"
UNIAGENT_SHA="73b0f41efa88b311fd69129c6f835c012e925e73"

log() { echo "[setup] $*"; }

# fetch() downloads via tmp+mv so an interrupted transfer can never leave a
# truncated file that a rerun's existence guard would then accept.
fetch() {
  local url="$1" dest="$2"
  [ -f "$dest" ] && return 0
  curl -fsSL "$url" -o "$dest.tmp"
  mv "$dest.tmp" "$dest"
}

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

# ---- 3. install: uni-agent (editable), verl (--no-deps), frozen set, wheel --
# The frozen dependency set installs via the library's PUBLISHED installer
# (the F-01 recipe as code — everything-but-sglang via -r, then the sglang
# pin --no-deps; library publishing.md §4). Both files fetched at the tag.
mkdir -p assets
fetch "${RAW}/devharness/uniagent/requirements.txt" assets/uniagent-requirements.txt
fetch "${RAW}/devharness/uniagent/install_collector_env.sh" assets/install_collector_env.sh
"$PIP" install --disable-pip-version-check -q -e vendor/uni-agent
"$PIP" install --disable-pip-version-check -q --no-deps -e vendor/uni-agent/verl
bash assets/install_collector_env.sh "$PIP" assets/uniagent-requirements.txt
"$PIP" install --disable-pip-version-check -q "gsj-envloader @ ${WHEEL_URL}"
# probe as a standalone command: inside log "$(...)" a failure would be
# masked (set -e does not fail on substitutions in an argument position)
VERSION_PROBE=$("$PY" -c 'import gsj.envloader as g; print("gsj-envloader", g.__version__)')
log "collector-venv: $VERSION_PROBE"

# ---- 4. the page corpus (release tarball, verified) -------------------------
if [ ! -d assets/pages ]; then
  curl -fsSL "$PAGES_URL" -o assets/pages.tar.gz
  echo "${PAGES_SHA256}  assets/pages.tar.gz" | sha256sum -c -
  tar -xzf assets/pages.tar.gz -C assets/
  rm assets/pages.tar.gz
fi
log "pages: $(find assets/pages -name 'page_*.md' | wc -l | tr -d ' ') files"

# ---- 5. the gate pins (release asset, verified) -----------------------------
# Published dev-environment DATA (library publishing.md §3): validates the
# published fixtures/templates/codec exactly. NOT a portable trust root —
# your own artifacts get your own pins (gsj-pin).
if [ ! -f assets/pins.dev.json ]; then
  fetch "$PINS_URL" assets/pins.dev.json.download
  echo "${PINS_SHA256}  assets/pins.dev.json.download" | sha256sum -c -
  mv assets/pins.dev.json.download assets/pins.dev.json
fi
log "pins: assets/pins.dev.json ($(wc -c < assets/pins.dev.json | tr -d ' ') bytes)"

# ---- 6. the sandbox image (preflight convenience — the library's own
#          docker preflight refuses to auto-pull) -----------------------------
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "image $IMAGE not in the local daemon; trying anonymous pull"
  docker pull "$IMAGE" || {
    log "ERROR: cannot pull. On egress-locked hosts, ship it from any host"
    log "that can: docker save $IMAGE | ssh <this-host> docker load"
    exit 1
  }
fi
log "image: $IMAGE present"

echo
log "DONE. Check every */config.yaml: absolute paths must match THIS"
log "checkout root: $ROOT   (nothing else varies per host — templates ship"
log "in the wheel, the MCP shim is the image's own, the codec snapshot"
log "resolves from the driver {model_id, revision} pin pair at first run)"
log "seed a project with:"
log "  collector-venv/bin/gsj-collect --config <project>/config.yaml"
log "(bounded: exits at the episode target or round-complete; Ctrl-C drains)"
