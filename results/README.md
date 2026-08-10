# Curated results

`runs/` is scratch and is not committed. It is whatever the last invocation
produced. This directory is the published artifact: the specific outputs being
claimed in the technical note, copied here deliberately.

The acceptance criteria require the final artifact to carry raw
predictions/logits, configs, an environment lock, provenance, logs, the actual
compute cost, and a machine-readable metrics table. Copy these from the tier
that is actually being reported:

```bash
TIER=full        # the tier whose numbers appear in the write-up
mkdir -p results/$TIER
cp -r runs/$TIER/results        results/$TIER/
cp -r runs/$TIER/provenance     results/$TIER/
cp -r runs/$TIER/figures        results/$TIER/
cp    runs/$TIER/ledger_inference.json results/$TIER/
cp -r runs/$TIER/logits         results/$TIER/     # ~150 MB at full scale
cp    configs/$TIER.yaml        results/$TIER/config.yaml
cp    env/requirements.lock.txt results/$TIER/
cp    docs/frozen_spec.json     results/$TIER/
```

Before publishing, confirm:

- [ ] `spec_hash` in `provenance/run_manifest.json` matches `docs/frozen_spec.json`
- [ ] `provenance/split_audit.json` shows `"overlap": 0`
- [ ] `ledger_inference.json` is inside the tier's gate, and the tiers actually
      run sum to <= 6 GPU-hours and <= $1.62
- [ ] `results/calibration.json` does **not** report `"degenerate": true`
      (if it does, say so in the note rather than quietly reporting the numbers)
- [ ] every `[FILL]` in `docs/technical_note.md` is resolved
- [ ] `docs/novelty_search_log.md` is completed and dated, and the claim wording
      matches its verdict

The logit cache is the expensive artifact to reproduce and the cheap one to
store: every metric, ablation and figure in the repository can be regenerated
from it on a CPU in minutes, with no GPU and no spend.
