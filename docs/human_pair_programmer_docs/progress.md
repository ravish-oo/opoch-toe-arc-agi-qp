Short answer:

* **Yes** — for each WO, the reviewer can just replace the “metrics block” with the WO-specific lines (e.g., for WO-2 use only `embed_idempotent_ok` and `embed_metamorphic_ok`), and paste the same template.
* **No** — you don’t need a brand-new harness patch every time. Do one small, generic harness patch now (below). After that, each WO only **adds** its own metric lines; no structural change.

---

## One-time harness patch (generic, supports all WOs)

Add this once; you won’t patch again per WO.

**`src/arcsolver/harness.py`**

```python
# Register WO-specific metric keys here once (extend-only)
WO_METRICS = {
    1: ["bins_sum_ok", "bins_hash_stable", "center_all_ok"],
    2: ["embed_idempotent_ok", "embed_metamorphic_ok"],
    3: ["hungarian_bijection_ok", "signature_tie_break_ok"],
    4: ["closure_order_independent_ok", "avg_admits_before", "avg_admits_after"],
    5: ["bin_constancy_proved"],
    6: ["free_cost_invariance_ok"],
    7: ["flow_feasible_ok", "kkt_ok", "one_of_10_ok"],
    8: ["idempotence_ok", "bits_sum"],
    9: ["laminar_confluence_ok", "iis_count"],
}

def init_progress(wo:int)->dict:
    keys = WO_METRICS.get(wo, [])
    return {"wo": wo, "tasks_total": 0, "tasks_ok": 0,
            "metrics": {k: {"ok": 0, "total": 0, "sum": 0} for k in keys}}

def acc_bool(progress:dict, key:str, ok:bool):
    if key not in progress["metrics"]: return
    m = progress["metrics"][key]; m["total"] += 1; m["ok"] += int(bool(ok))

def acc_sum(progress:dict, key:str, val:int):
    if key not in progress["metrics"]: return
    progress["metrics"][key]["sum"] += int(val)

# At start of run (for --upto-wo N):
progress = init_progress(N)

# In per-task loop you simply call acc_bool/acc_sum for the keys your WO computes.
# Example for WO-2:
# acc_bool(progress, "embed_idempotent_ok", embed_idempotent_ok)
# acc_bool(progress, "embed_metamorphic_ok", embed_metamorphic_ok)

# At end of run, write progress JSON (already implemented in receipts.write_run_progress).
```

That’s it. From now on, each WO just **calls `acc_bool/acc_sum` for its own keys**. No more harness plumbing.

---

## Reviewer template: exactly what to swap per WO

Use this unchanged template; only replace the **metrics block** based on the WO.

**Subject:** ARC Solver — WO-01 Progress (Π/GLUE/FY/TU receipts)

Run on all 1000 tasks:

```bash
bash scripts/run_harness.sh --upto-wo <work order number> --strict
```

What to verify:

1. Receipts are present for every task and **byte-identical** on two runs.
2. Open `progress/progress_wo{N}.json` and confirm the metrics below are green.

**Metrics to report (replace this block per WO):**

* **WO-1 (Bins & Predicates)**

  * `bins_sum_ok` = 100%
  * `bins_hash_stable` = 100%
  * `center_all_ok` = 100%

**How to flag issues**

* Any red metric = **implementation gap** for this WO; fix code.
* Only at WO-9 is “UNSAT” legitimate, and only if an **IIS** is emitted (minimal infeasible subsystem; standard LP notion). ([SAS Help Center][4])

**Why these metrics are enough (no labels needed)**

* Metamorphic relations test invariances when no oracle exists. ([Wikipedia][1])
* Min-cost-flow optimality via reduced costs/KKT is the textbook certificate of FY-tightness. ([Homes][2])
* Total unimodularity guarantees integrality for network LPs (sanity for later steps). ([Wikipedia][5])
* IIS is the standard proof object for infeasibility. ([Gurobi Help Center][3])

Please paste back the two numbers `tasks_total`, `tasks_ok`, and the metric table from `progress_wo{N}.json`, plus any failing task IDs if present.

---

* **WO-2 (Embedding)**

  * `embed_idempotent_ok` = 100%  (re-embed round-trip is byte-equal)
  * `embed_metamorphic_ok` = 100% (FREE roll/permutation doesn’t change the predicate result; metamorphic relation) ([Wikipedia][1])

* **WO-3 (Color alignment)**

  * `hungarian_bijection_ok` = 100%
  * `signature_tie_break_ok` = 100%

* **WO-4/5 (Mask closure & Equalizers)**

  * `closure_order_independent_ok` = 100%
  * `avg_admits_after ≤ avg_admits_before` (nonincreasing)

* **WO-6 (Scores + FREE predicate)**

  * `free_cost_invariance_ok` = 100% (ŝ∘U = ŝ and constraints invariant for verified FREE U)

* **WO-7 (Unified flow)**

  * `flow_feasible_ok` = 100%
  * `kkt_ok` = 100% (reduced-cost complementary-slackness optimality for min-cost flow) ([Homes][2])
  * `one_of_10_ok` = 100%

* **WO-8 (Decode & Bit-meter)**

  * `idempotence_ok` = 100% (Φ∘Φ = Φ)
  * `bits_sum` (report the value; trend should not explode)

* **WO-9 (Relaxation & IIS)**

  * `laminar_confluence_ok` = 100%
  * `iis_count` reported with an IIS for each infeasible case (minimal infeasible subset) ([Gurobi Help Center][3])
