# Historical essentiality sources

This directory preserves source provenance, not model weights or a claim of
fresh model inference. `archive/` contains byte-for-byte historical scripts and
summary files recovered on 5 September 2026. Those scripts retain obsolete host
paths and known defects and must not be used as the current reproduction entry.

Use `python scripts/reproduce_essentiality.py --help` for the CPU archived-score
recomputation. The required original records and result shards are enumerated,
sized and SHA256-hashed in each run's `report.json`. No historical files are
modified. See `docs/ESSENTIALITY_REPRODUCTION.md` for protocols and limitations.
