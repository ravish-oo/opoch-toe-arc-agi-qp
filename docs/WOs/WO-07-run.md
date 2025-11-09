WO-7 must not try to “re-parse receipts” to reconstruct arrays. Receipts are for **auditing**; the **source of truth** for the flow is the **upstream artifacts** (A_mask, costs, quotas, bins, cell caps, equalizer verdicts, faces) produced by WO-1…WO-6. Here’s how to keep development fast **and** faithful without rebuilding the whole world each time.

---

## What’s wrong

* **Receipts ≠ artifacts.** Our WO-4/5/6 receipts intentionally store *hashes, shapes, and stats*, not the full arrays. Trying to run WO-7 off these will force you to either guess the arrays or add ad-hoc formats.
* **WO-5 is the heavy stage.** Recomputing it 1,120× every edit is slow.

---

## Correct plan (from WO-05 onward)

### 1) Keep the pipeline, add a **materialized artifact cache**

* Do **not** change receipts.
* Add a `.cache/woNN/` directory per stage, ignored by Git.
* Each task write a single compressed NPZ with all **arrays** that the next stage needs, plus a tiny `manifest.json`:

```
.cache/
  wo04/
    00d62c1b.<SHA256-of-inputs-and-code>.npz
      - A_mask.npy (bool [N,C])
      - F.npz (optional)
      - bin_ids.npy (int64 [N])
      - manifest.json { "wo":4, "H":..., "W":..., "code_hash": "...", "input_hash": "..." }
  wo05/
    00d62c1b.<SHA>.npz
      - equalizer_edges.npz  # e.g. dict (s,c)->(E×2 int32)
      - cell_caps.npy (int32 [H,W])
      - faces_R.npy (int32 [H,C]) optional
      - faces_S.npy (int32 [W,C]) optional
      - ...
```

* The **stage runner** (`stages_woNN.py`) first tries to `load_cache()`. If `exists && manifest.hashes match`, use it; else compute from source (`STAGES[1..NN-1]`) and then `save_cache()`.

This is still option (1) (recompute from source) but **incremental**. You pay WO-5 once per task per code-hash; subsequent WO-7 runs over all 1,120 tasks reuse cached NPZs.

### 2) Define **cache keys** deterministically

* `input_hash` = SHA256 of the raw task JSON (canonical JSON dump).
* `code_hash` = short SHA of the Git tree for that WO’s module set (e.g., `stages_wo05.py`, `eqs.py`, `mask.py`, `color_align.py`, etc.).
* `free_maps_hash` = SHA of `ctx.embedding["free_maps_verified"]` where relevant (WO-6→WO-7).
* Include `H_out,W_out,C` in manifest for sanity.

On load, if **any** hash mismatches → invalidate and rebuild.

### 3) Keep WO-7 pure & tiny

WO-7 must:

* `load_cache(wo4)`, `load_cache(wo5)`, `load_cache(wo6)` → you now have `A_mask`, `bin_ids`, `cell_caps`, `costs`, `equalizer rows`/`faces`.
* Build the OR-Tools graph, solve, run **primal + cost-equality + (optional) reduced-cost** checks, and write `wo07.json` + `flow.npz` (selected x[p,c] if you want to materialize).

No reading of receipts for arrays; only to **verify hashes**.

---

## Dev flow without burning hours

### A) “Shard” mode for tight loops

Add `--tasks=shard://<preset>` in the harness to run a representative subset (10–50 tasks) that stresses WO-7:

* coverage on: small/large H×W, both faces on/off, with/without equalizers, with/without free maps, cells with tight caps, etc. (We can check tags from earlier receipts to pick a balanced sample.)

```
bash scripts/run_harness.sh --upto-wo 7 --tasks shard://wo7_fast --strict --progress
```

### B) “Full” mode for gating

Nightly or before merging a big change:

```
bash scripts/run_harness.sh --upto-wo 7 --tasks all --strict --progress
```

Because of the cache, WO-5/WO-6 artifacts are reused; you’re only rebuilding the graph and solving. If WO-5 is still the bottleneck, you can prime `.cache/wo05/` once by running `--upto-wo 5` overnight.

---

## Practical sizes (storing arrays is OK)

Typical max ARC: (H,W≤30), (N≤900), (C=10)

* `A_mask`: 900×10 bool ~ 9 KB (compressed ~1–3 KB)
* `costs`: 900×10 int64 ~ 72 KB (compressed ~20–40 KB depending)
* equalizer edges: worst case a few KB per cell; many tasks empty
* faces: (H·C + W·C) ints, tens of KB max

Even at 100 KB per task × 1,120 ≈ **112 MB** raw; NPZ compresses 2–5×. This is fine for a local cache. Don’t put it in Git; use `.gitignore`.

---

## Actionable patch list

1. **Add** `src/arcsolver/cache.py` with `load_cache(wo, task_id, code_hash, input_hash) → dict[str, np.ndarray] | None` and `save_cache(...)`.
2. **Update** `stages_wo04/wo05/wo06` to **write NPZ** artifacts + manifest; keep existing JSON receipts unchanged.
3. **Update** `stages_wo07` to **only** consume caches (compute upstream on cache miss via `STAGES[...]`, not by reading receipts).
4. **Harness**: add `--tasks` selector (`all | list:file | shard://wo7_fast | regex:`…).
5. **Progress**: keep `flow_feasible_ok`, `one_of_10_ok`, `kkt_ok|cost_equal_ok`, `idempotence_ok`, `faces_ok`. Count **only accepted instances** (as you did for WO-6) and surface cache-hit rates.

---

## Why this is still 100% on-spec

* The min-cost flow stage still takes **integer** costs & capacities (03_annex A.2), enforces Π-safe constraints (00 §8; 01 §5–§7), and checks **primal/KKT** (MIT OCW note you cited).
* FREE maps from WO-6 were already enforced via score projection; flows remain single-commodity integer per OR-Tools.
* Receipts remain *small, auditable*; caches are just build artifacts, not part of the audit trail.

---

### TL;DR

* Don’t try to reconstruct arrays from receipts; add a **materialized NPZ cache per WO**.
* For quick iteration, run a **curated shard**; prime caches for the full suite.
* WO-7 stays tiny: load NPZ, build OR-Tools graph, solve, KKT/feasibility checks, write receipt.
* This gives you fast dev cycles and preserves the spec’s determinism and auditability.
