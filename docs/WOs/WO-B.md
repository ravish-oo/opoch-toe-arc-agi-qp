# WO-B: Per-canvas costs + pack solve + WO-10 selection (remove DEFERRED)

**Purpose (what changes):**

* Refactor **WO-6** into **per-canvas callable** and write **canvas-keyed costs**.
* Update **WO-9B** to read per-canvas caches and **solve each pack once**.
* Update **WO-10** to **select lex-min across solved packs**; remove “DEFERRED linear law”; keep zero-supply gate.

**Anchors to read:** @docs/anchors/00_math_spec.md §1/§8/§16, @docs/anchors/01_addendum.md §10/§11/§12/§F, @docs/anchors/03_annex.md A.1–A.3.

## Scope & libraries

* **NumPy**, **json/hashlib/pathlib**
* **OR-Tools** only via existing WO-7 runner

## Deliverables

1. **WO-6 per-canvas** function:

   ```
   .cache/wo06/<task>.<test>.<canvas_id>.npz
   ⇒ {"costs": (N,C) int64, "pi_safe_ok": true, "free_invariance_ok": true, "meta": {...}}
   ```

   • Use **verified FREE** maps to project ŝ (§B); integerize costs (Annex A.2).

2. **WO-9B** (adapted):

   * For each pack in `.cache/wo09/<task>.<test>.packs.json`:
     • Load **WO-4/5/6 NPZ** by `canvas_id`.
     • Call **WO-7**; apply **laminar relax** (mask-before-quota fix).
     • If `INFEASIBLE`, compute **IIS (hard tier only)** with 1-minimality replay.
   * Write `wo09b.json` with `packs_tried` (no “DEFERRED”), proof fields, IIS, and `hash`.

3. **WO-10** (adapted selection):

   * **Reject zero-supply** packs (gate stays).
   * Select **lex-min** across packs: `(optimal_cost, delta_bits, (H,W), pack_id)` (§12/§16).
   * Decode (WO-8), evaluate (if ground truth), write `wo10.json` & `eval.json`.

## Receipts (first-class)

* `wo09b.json`: lists **every** `pack_id` with `status ∈ {OPTIMAL, INFEASIBLE}`, proofs (budget, primal, capacity, cost equality, kkt), and IIS if infeasible; canonical `hash`.
* `wo10.json`: `pack_viability`, `final` proof flags, IIS (if UNSAT), canonical `hash`. No arrays inside.

## Acceptance (reviewer, must be 100%)

* `wo06_canvas_ok`: for each pack, a costs NPZ exists and `pi_safe_ok & free_invariance_ok` are `true`.
* `wo09b_solved_ok`: **no** pack has `status:"DEFERRED"`; each is OPTIMAL or INFEASIBLE.
* `laminar_respects_tiers_ok`: faces drops precede quotas; **never** drop mask/equalizer/cell (sample 10 packs).
* `proofs_ok`: for chosen pack, `budget_preservation_ok & primal_balance_ok & capacity_ok & cost_equal_ok (& kkt_ok)` are `true`.
* `iis_present_ok`: for infeasible finals, IIS exists and each `row.feasible_when_removed == true`.
* `wo10_selection_ok`: chosen pack is lex-min by `(cost, Δbits, (H,W), pack_id)` across **all** solved packs; zero-supply packs are blocked.
* `determinism_ok`: re-run 5 tasks; `wo09b.json["hash"]` and `wo10.json["hash"]` stable.

**Shard run:**

```
bash scripts/run_harness.sh --upto-wo 11 --strict --subset shards/per_canvas_50.txt --evaluate
```

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
