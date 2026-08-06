# pins/ — the gate fingerprints, carried as repo DATA

`pins.dev.json` is a **byte-identical copy** of the library repo's
`devharness/pins/pins.dev.json` at tag `v0.4.0`
(sha256 `766ca30ea8ca15bcca249f167c8f42d23f2cb97eb3c33adf8bc97f8c4fa15d44`).
Every `config.yaml` in this repo points `collector.pins_path` at this copy.

Why a copy and not a fetch: the pins are *generated* data (`gsj-pin` over
captured episode artifacts — see the library README §6). An external
consumer cannot regenerate them without the capture harness, so the only
way to get them is to take the library's generated file as-is. There is no
published release asset carrying them (registered as a finding, see
`../FINDINGS.md`), hence the committed copy with this provenance note.

What they gate (library README §6): G1 skill-card, G2 system prompt
(docker mode collapses to a container singleton — host-path independent),
G3 tool roster, G4 tokenizer/template, G6 no-thinking tail, G7 settings.
They are valid for exactly this stack: the
`ghcr.io/mhganainy/gsj-pi-harness:pi0.83.0-mcp1.5.0-2` sandbox image,
Qwen3-0.6B/4B family tokenizer, and the v0.4.0 templates fetched by
`../setup_collector.sh`. Change any of those and episodes will quarantine
with gate failures until you re-pin (which requires the library harness).
