# WO-A: Per-test concrete pack builder + per-canvas WO-4/5

**Purpose (what changes):**

* Make **packs concrete per `(task,test,canvas)`**, not task-level.
* Refactor **WO-4/5** into **pure per-canvas callables** and write **canvas-keyed caches**.
* Enumerate packs with **faces modes** (rows/cols/none) for each concrete canvas (size law per test).

**Anchors to read:** @docs/anchors/00_math_spec.md §1 (size laws), §2 (embed), §4 (equalizers on allowed), **§8 (faces)**, @docs/anchors/01_addendum.md §12 (packs), @docs/anchors/03_annex.md A.1–A.3 (determinism).

## Scope & libraries

* **NumPy** (`array_equal`, `argsort(kind="stable")`, `unique(return_counts=True)`)
* **SciPy csgraph** (deterministic `breadth_first_tree` for equalizer edges if rebuilding)
* **json/hashlib/pathlib** for canonical receipts and cache paths

## Deliverables

1. **WO-4 per-canvas** function:

   ```
   .cache/wo04/<task>.<test>.<canvas_id>.npz
   ⇒ {"A_mask": (N,C) bool, "bin_ids": (N,), "meta": {...}}
   ```
2. **WO-5 per-canvas** function:

   ```
   .cache/wo05/<task>.<test>.<canvas_id>.npz
   ⇒ {"equalizer_edges": {(s,c): E×2}, "faces_R": (H,C)|None, "faces_S": (W,C)|None, "meta": {...}}
   ```

   • Equalizers restricted to **allowed** set; **skip singletons**.
   • Faces = **meet across aligned trainings on this concrete canvas** (00 §8).
3. **WO-9A′** per-test pack builder:

   ```
   .cache/wo09/<task>.<test>.packs.json
   ⇒ {"packs":[{"pack_id","canvas_id","size":{"law","H","W"},"faces_candidates":{"R","S"}, "free_maps":[...]}], "size_laws_complete": true, "hash": "..."}
   ```

   • Concretize **constant/linear** laws from `.cache/wo01/<task>.size_laws.json` per test.
   • Enumerate faces modes per canvas.
   • Include **only verified FREE** maps from `.cache/wo06/<task>.free_maps.json`.

## Receipts (first-class)

* `wo09a_prime.json` with canonical JSON `hash`. No arrays in receipts; arrays live in NPZ.
* Include `size_laws_complete: true` (not interim).

## Acceptance (reviewer, must be 100%)

* `packs_exist_ok`: `packs.json` exists for **every** `(task,test)` in the shard.
* `packs_deterministic_ok`: re-run 5 random tasks; `wo09a_prime.json["hash"]` stable.
* `faces_mode_ok`: if `faces_R/S` present in NPZ → at least one pack has `faces=rows|cols`; if absent → all packs have `faces=none`.
* `equalizers_allowed_ok`: for every `(s,c)` in `equalizer_edges`, all endpoints are in `A_mask==True` and **|S|≥2** (sample 5 tasks).
* `cache_namespace_ok`: NPZ files exist at `<task>.<test>.<canvas_id>.npz` for **both** WO-4 and WO-5.

**Shard run:**

```
bash scripts/run_harness.sh --upto-wo 9 --strict --subset shards/per_canvas_50.txt
```

If any metric <100% ⇒ implementation gap in WO-A (not dataset).

---

## Guardrails (avoid debugging hell)

* **Feature flag per WO**: land WO-A first; freeze; then WO-B.
* **Receipts hash**: each stage emits a top-level `hash`; reviewer compares hashes between runs (Annex A.1–A.3 determinism).
* **No arrays in receipts**: arrays are only in `.npz`; receipts carry **hashes and proofs**.
* **Zero-supply gate**: in WO-7 precheck & WO-10; prevents “OPTIMAL zero-flow” regressions.

---

## Why this won’t mint bits / won’t break idempotence

* All packs come from **proved** atoms (size laws §1, faces §8, verified FREE §B/G).
* Each pack solved **once** (no branching search); TU + integers ⇒ unique extreme solution; Φ selection is a pure lex-min (§12).
* Re-run ⇒ same caches, same receipts, same `Ŷ` (Φ ∘ Φ = Φ).

---
