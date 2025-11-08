minimal, deterministic, and maps 1:1 to the modules in 04_engg_spec.md and contracts.

```text
arc-solver/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ .python-version                 # e.g. 3.11.x
├─ requirements.txt                # numpy, scipy, scikit-image (optional), ortools pinned
├─ .env.example                    # OMP/OPENBLAS/MKL/NUMEXPR=1, PYTHONHASHSEED=0
│
├─ docs/
│  ├─ anchors/
│  │  ├─ 00_math_spec.md
│  │  ├─ 01_addendum.md
│  │  ├─ 02_addendum.md
│  │  ├─ 03_annex.md
│  │  ├─ 04_engg_spec.md
│  │  └─ 05_contracts.md
│  ├─ DEV_NOTES.md                 # optional: WO checklist, acceptance criteria per module
│  └─ RUNBOOK.md                   # how to run harness on all 1000 tasks
│
├─ data/
│  ├─ arc/                         # ARC-AGI jsons (sorted load); path configured in src/arcsolver/config.py
│  └─ outputs/                     # final predictions.json + receipts per run (byte-stable)
│
├─ receipts/                       # always-on receipts per task/stage (JSON)
│  └─ <task_id>/
│     ├─ wo01.json
│     ├─ wo02.json
│     └─ ... (up to final)
│
├─ scripts/
│  ├─ run_harness.sh               # runs harness over all tasks with stage gates
│  ├─ run_one.sh                   # run a single task id for quick debug
│  └─ verify_idempotence.sh        # runs Φ twice, byte-compares outputs
│
├─ src/
│  └─ arcsolver/
│     ├─ __init__.py
│     ├─ config.py                 # env guards; threads=1; dtype constants; SCALE=1_000_000
│     ├─ types.py                  # dataclasses: Grid, Canvas, Component, Signatures, FlowIDs
│     ├─ io.py                     # load/save ARC json; sorted keys; byte-exact ints
│     ├─ bins.py                   # build_bins(); predicates; hashes
│     ├─ embed.py                  # embed_to_canvas(); center/topleft predicate (content != 0)
│     ├─ color_align.py            # signatures; Hungarian adapter (int64 costs; lex tiebreak)
│     ├─ mask.py                   # forward meet closure; color-agnostic lift; build A_{p,c}
│     ├─ objects.py                # 4-connectivity components; bbox/centroid; relations
│     ├─ scores.py                 # Π-safe ŝ; FREE predicate gate; symmetry transport; cost int64
│     ├─ flows.py                  # unified pixel-level MCF via OR-Tools; KKT/conservation checks
│     ├─ assign.py                 # decode from pixel flows; bit-meter (ceil(log2 |orbit|))
│     ├─ relax.py                  # laminar greedy relaxation; IIS builder
│     ├─ compose.py                # stage links and fixed-point equalities (exact rows)
│     ├─ ledger.py                 # ΔN, E_min reporting; uncertainty packs selection
│     ├─ receipts.py               # write per-stage receipts (hashes, symmetries, flow stats)
│     ├─ harness.py                # CLI: runs “up to WO#”; validates stage invariants on all tasks
│     └─ utils/
│        ├─ hungarian_adapter.py   # deterministic wrapper over scipy’s LSA with lex offsets
│        ├─ flow_adapter.py        # deterministic OR-Tools builder; node/arc ordering fixed
│        └─ label_adapter.py       # 4-conn labeling (scikit-image or scipy ndimage), −1→0 map
│
└─ tests/
   ├─ unit/
   │  ├─ test_bins.py              # hashes stable
   │  ├─ test_embed.py             # center vs topleft; byte-equal re-embed
   │  ├─ test_color_align.py       # adversarial ties → deterministic perm
   │  ├─ test_meet.py              # order-free closure (F hash invariant)
   │  ├─ test_flows.py             # tiny synthetic graph → exact optimum; KKT≥0
   │  ├─ test_assign.py            # one-of-10; multiway tie bits
   │  └─ test_free_predicate.py    # ŝ∘U==ŝ & constraints invariant for verified FREE maps
   └─ golden/
      ├─ test_idempotence.py       # Φ∘Φ=Φ on a small battery
      └─ test_subset_arc.py        # deterministic outputs & receipts on a curated subset
```

### quick notes

* **single source of truth for paths**: `src/arcsolver/config.py` should point to `data/arc/` and `receipts/`, no hardcoded paths in modules.
* **determinism knobs**: set env vars in `scripts/run_harness.sh` and enforce in `config.py` at import time.
* **always-on receipts**: everything critical writes under `receipts/<task_id>/woXX.json`, even in early WOs.
* **no parallelism**: keep OR-Tools and numpy single-thread (guarded in `config.py`).
* **anchors are canonical**: modules mirror `04_engg_spec.md` sections; contracts in `05_contracts.md` apply to all.
