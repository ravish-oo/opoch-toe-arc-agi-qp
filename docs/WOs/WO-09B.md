# WO-9B — Laminar Greedy Relax + IIS

**Goal.** For each **pack** from WO-9A, enforce the spec’s **laminar precedence** (hard tiers never relaxed; faces before quotas; free symmetries last), call WO-7 as the feasibility oracle, and either:

* select the first **feasible** pack (lex-min among feasible per 01 §12), or
* if **infeasible** for all packs after legal relaxations, emit an **IIS** (Irreducible Infeasible Subsystem) certificate (01 §11).

**Anchors to read before coding**

* @docs/anchors/00_math_spec.md : §8 (faces optional; budgets & mask), §15 (no hidden search)
* @docs/anchors/01_addendum.md : §10 (laminar precedence; minimal, confluent greedy), §11 (IIS), §12 (packs)
* @docs/anchors/03_annex.md : A.1–A.3 (byte-exact integers; lex order; determinism)

---

## Libraries (mature, well-documented; no algorithm invention)

* **OR-Tools** (already used in WO-7): `SimpleMinCostFlow` for re-solve; we use WO-7’s stage runner, not the API directly here. Reference docs for feasibility, supplies, objective equality, and (if exposed) reduced costs/potentials. ([Google for Developers][1])
* **NumPy**: integer arrays; `np.array_equal` for byte-exact equality; stable sorts when needed. ([NumPy][2])
* **Python stdlib**: `json.dumps(sort_keys=True, separators=(",",":"))` for canonical receipt JSON (deterministic). ([Python documentation][3])

**Theory facts we rely on (no re-implementation):**

* **Laminar family ⇒ matroid greedy is optimal & confluent** (we rely on the standard laminar matroid greedy theorem to justify that our greedy dropping by tier is minimal & order-independent). ([MIT Mathematics][4])
* **IIS definition** (minimal infeasible subset): infeasible, and removing any single constraint makes it feasible. We **format** an IIS from WO-7’s failing tier rows; we do not compute an LP IIS from OR-Tools (MCF doesn’t expose IIS). ([SAS Help Center][5])

---

## Inputs (from cache; never receipts)

* **Packs** (WO-9A cache JSON): size law, faces_mode, free_maps_verified, quick flags.
* **WO-4 cache**: `A_mask:(N,C)`, `bin_ids:(N,)`.
* **WO-5 cache**: `quotas:(S,C)`, `faces_R:(H,C)`/`faces_S:(W,C)` (optional), canonical equalizer rows (optional).
* **WO-6 cache**: `costs:(N,C)`, `free_maps_verified` (exact).
* **WO-7 stage runner**: callable to **(re)solve** a pack with its relaxations; WO-7 produces feasibility/optimality receipts (primal, cost equality, etc.) and the flow.

---

## Output

* **Selected pack**, with the list of **drops** (faces, quotas) taken, and WO-7 feasibility summary.
* If still infeasible: **IIS** with tier and identifiers of the minimal conflicting set.
* Receipt `wo09b.json` (see §Receipts).

---

## Algorithm (exact, deterministic)

For each pack in **lex order** from WO-9A:

### Step 0 — Build the “hard rows” and “relaxable rows”

* **Never relax** (hard): **mask**, **equalizers**, **cell/canvas fixed rows** (gravity/harmonic if present).
* **Relaxable (laminar)** (01 §10):

  1. **Faces** (shared rows, top relax tier)
  2. **Quotas** (bin budgets)
  3. **Free symmetries** (lowest; but WO-6 already enforced cost invariance, so there is nothing to “drop”—we simply don’t apply a projector if it’s not FREE)

→ This exactly matches the laminar family the anchors require; the **matroid-greedy** fact is what gives us minimal, confluent drops. ([MIT Mathematics][4])

### Step 1 — Try WO-7 as-is (no drops)

* Call the WO-7 stage runner with the pack’s size law, faces_mode, quotas, and costs.
* If **OPTIMAL** with `primal_balance_ok & capacity_ok & cost_equal_ok & one_of_10_ok` ⇒ **select** this pack and stop search.
* If **INFEASIBLE** or any primal check fails, get the WO-7 “reason”:

  * If the reason cites **faces** only (e.g., row sums unsupported; faces conflict) ⇒ go to Step 2a.
  * Else if the reason is **quota capacity** (e.g., `q[s,c] > |allowed(B_s,c)|`) ⇒ go to Step 2b.
  * Else if reason cites a **hard tier** (mask/equalizer/cell rows) ⇒ go to Step 3 (IIS).

> **Note:** MCF status/receipts should be parsed; we don’t rely on any OR-Tools exception. See OR-Tools docs for min-cost flow feasibility semantics; we use WO-7’s receipts (sum of supplies equals flow, objective equality, etc.) to confirm. ([Google for Developers][6])

### Step 2a — Drop **faces** (top relax tier), re-solve

* If `faces_mode != "none"`: set `faces_mode="none"` and call WO-7 again. Record drop: `{"tier":"faces","mode": "rows_as_supply"|"cols_as_supply"}`.
* If **OPTIMAL** now ⇒ select and stop. If still **INFEASIBLE**, continue to Step 2b.

### Step 2b — Reduce **quotas** minimally (laminar greedy), re-solve

For any ((s,c)) with `q[s,c] > |allowed(B_s,c)|`, set
`drop = q[s,c] - |allowed(B_s,c)|` and **reduce** `q[s,c] := q[s,c] - drop`. Record the drops as `{"tier":"quota","s":s,"c":c,"drop":drop}`.

* Call WO-7 again. If **OPTIMAL** ⇒ select and stop.
* If still **INFEASIBLE**, **one** more iteration: for the remaining quota conflicts, reduce until all `q[s,c] ≤ |allowed|` (or until WO-7 becomes feasible). Record drops.
* If still **INFEASIBLE** after the second quota pass ⇒ Step 3.

> This greedy bound-tightening is the laminar repair prescribed by 01 §10; it is minimal because faces have higher precedence and were already dropped, and we only reduce the **excess** (`q − |allowed|`), nothing more.

### Step 3 — IIS (irreducible infeasible subsystem)

At this point, infeasibility is due to **hard tiers** only (mask, equalizer, cell fixed rows). Build a small IIS object:

* **IIS structure:**

  ```json
  {
    "present": true,
    "tier": "mask|equalizer|cell",
    "rows": [
      {"type":"mask","p":123,"c":4},                    // A[p,c]==0 but required
      {"type":"equalizer","bin":8,"color":1,"edge":[p,q]},
      {"type":"cell_cap","r":3,"j":5}
    ]
  }
  ```
* It should be **minimal**: remove any one row from `rows` and WO-7 should become feasible (we can verify by temporarily dropping each row and re-solving; stop at the first subset that passes). This matches the IIS definition we cite. ([SAS Help Center][5])

Stop trying packs after the first IIS (we return the IIS with the best pack ID lexicographically to keep determinism).

---

## Receipts (first-class)

Write `receipts/<task>/wo09b.json`:

```json
{
  "stage":"wo09b",
  "packs_tried":[
    {
      "pack_id":"size=12x12|faces=rows_as_supply|free=[roll(2,0)]",
      "drops":[
        {"tier":"faces","mode":"rows_as_supply"},
        {"tier":"quota","s":8,"c":1,"drop":2}
      ],
      "result":{
        "status":"OPTIMAL",
        "primal_balance_ok":true,
        "capacity_ok":true,
        "cost_equal_ok":true,
        "one_of_10_ok":true,
        "selected":true
      }
    }
  ],
  "selected_pack_id":"size=12x12|faces=none|free=[roll(2,0)]",
  "iis": null,
  "hash":"<sha256 of canonical JSON>"
}
```

**Determinism:** compute `hash` as sha256 of the canonical JSON (`sort_keys=True`, `separators=(",",":")`) and store it (audit). ([Python documentation][3])

---

## Stage runner & harness (no “god function”)

**Code layout**

* `src/arcsolver/relax.py`

  * `try_pack(pack) -> RunResult` (wraps WO-7)
  * `drop_faces(pack) -> pack'`
  * `reduce_quotas_minimally(pack, A_mask, bin_ids, quotas) -> pack'`
  * `build_iis(pack, wo7_receipt) -> IIS`
* `src/arcsolver/stages_wo09b.py`

  * Loads packs from WO-9A cache, runs the algorithm above, writes `wo09b.json`, updates context (selected pack & drops).

**Pipeline registration**

```python
from .pipeline import STAGES
STAGES[10] = run_wo09b
```

(We advance the stage index to keep WO-9A at 9 and WO-9B at 10 in `STAGES`, or keep 9B at 10 and WO-10 at 11 depending on your numbering.)

---

## Reviewer – ultra-short acceptance (WO-9B)

Run:

```bash
bash scripts/run_harness.sh --upto-wo 10 --strict --progress
```

**Must be 100% (over accepted instances):**

* `pack_choose_ok` — if any pack is feasible, `selected_pack_id` exists and WO-7 receipts for that pack show `OPTIMAL ∧ primal_balance_ok ∧ capacity_ok ∧ cost_equal_ok ∧ one_of_10_ok`.
* `laminar_respects_tiers_ok` — **never** drop mask/equalizers/cell; **faces** must be dropped **before** quotas; quota reductions are minimal (only excess).
* `determinism_ok` — re-run 3–5 tasks; `wo09b.json["hash"]` unchanged.

**If still infeasible** (no feasible pack):

* `iis_present_ok` — IIS exists, and each row is necessary (dropping any one row makes the model feasible upon re-solve), matching the IIS definition. ([SAS Help Center][5])

If any of the above <100% → **WO-9B implementation gap** (bad tiering, non-deterministic choice, or IIS not minimal). Attach `wo09b.json`, the relevant `.cache/wo09a/*.packs.json`, and the WO-7 receipts used.

---

## Anti-optimization note (CPU)

* WO-9B calls WO-7 at most a **handful** of times per pack (as-is → drop faces → reduce quotas 1–2 passes). Packs are few (often 1). With caches, this is **CPU-trivial**.

---

[1]: https://developers.google.com/optimization/reference/graph/min_cost_flow/SimpleMinCostFlow?utm_source=chatgpt.com "SimpleMinCostFlow | OR-Tools"
[2]: https://numpy.org/devdocs/reference/generated/numpy.array_equal.html?utm_source=chatgpt.com "numpy.array_equal — NumPy v2.4.dev0 Manual"
[3]: https://docs.python.org/3/library/json.html?utm_source=chatgpt.com "JSON encoder and decoder — Python 3.14.0 documentation"
[4]: https://math.mit.edu/~goemans/18453S17/matroid-notes.pdf?utm_source=chatgpt.com "5. Matroid optimization 5.1 Definition of a ..."
[5]: https://documentation.sas.com/doc/en/casmopt/8.3/casmopt_optlp_details19.htm?utm_source=chatgpt.com "Irreducible Infeasible Set"
[6]: https://developers.google.com/optimization/reference/graph/min_cost_flow?utm_source=chatgpt.com "min_cost_flow | OR-Tools"

## **minimal IIS** (Irreducible Infeasible Subsystem) Clarification
Great question. You want a **minimal IIS** (Irreducible Infeasible Subsystem) but you’re not getting one from the MCF solver. Two things are true and both matter:

1. **OR-Tools `SimpleMinCostFlow` does not expose any IIS facility.** Its API gives you flows, costs, and (sometimes) node “potentials,” but nothing to compute IIS for an infeasible model. You must treat WO-7 as a **feasibility oracle** and build the IIS yourself on top of it. ([Google for Developers][1])

2. An IIS is formally a **subset of constraints that is infeasible, and removing any single member makes it feasible** (1-minimal). That’s the target you need to reach. ([support.gurobi.com][2])

Below is a practical, spec-tight way to do this **without LP APIs**, using standard, well-documented methods (QuickXplain or delta debugging) and your existing WO-7 runner as the feasibility checker.

---

## What you build the IIS over (hard tier only)

Per our spec and laminar precedence, IIS rows must come from **never-relax** tiers:

* **Mask rows**: disallowed ((p,c)) pairs (i.e., arcs you suppressed because (A_{p,c}=0)).
* **Equalizer rows**: each equalizer edge ((p,q,c)) tying two pixels in a test-allowed bin ((|S_{s,c}|\ge 2)).
* **Cell/canvas fixed rows** (if any): e.g., cell capacity “hard” rows when they are not part of shared relaxables.

You already identify these in WO-7 receipts when a pack fails. Collect them into a **candidate set** (C) of row IDs.

> Faces and quotas are **relaxable** and must **not** enter the IIS (they are dropped earlier by the laminar greedy). The remaining infeasibility is precisely the “hard” system you certify.

---

## The oracle we’ll use

Define a function:

```
FEASIBLE(R_removed): bool
    → run WO-7 once with the same pack, but with rows in R_removed not enforced
    → return True iff solver status is OPTIMAL and primal/capacity/cost checks pass
```

WO-7 is fast, so you can call it multiple times per task.

---

## Two standard, documented ways to compute a **minimal** conflict set

### A) QuickXplain (preferred; fewer oracle calls)

QuickXplain is a **divide & conquer** algorithm to compute a minimal conflict (a minimal unsatisfiable subset) using only a feasibility checker. It is widely used in constraint programming, is non-intrusive, and has clean minimality guarantees. In short:

* If the whole set (C) is feasible, there is **no** conflict (shouldn’t happen here).
* Otherwise, recursively split (C) into two halves (C_1,C_2) and use the oracle to prune halves that don’t contain conflict. The algorithm returns a **minimal** conflict set (K\subseteq C).

See Junker (AAAI 2004) and later expositions (Rodler 2020/2022) for the exact recursion and proofs. ([aaai.org][3])

**What you implement:** a 60–100 LOC recursive function `quickxplain(C, oracle)` that:

* takes a **deterministically ordered** list of row IDs (sort lex by tier/type/id so your result is stable),
* calls `FEASIBLE()` on subsets,
* returns a **1-minimal** infeasible subset.

### B) ddmin (delta debugging) (simpler to code)

`ddmin` iteratively reduces a failing set by removing chunks and keeping any removal that **still** fails; it guarantees **1-minimality** (remove any single element and failure disappears), though it may take more oracle calls than QuickXplain. It is well-documented and used to isolate minimal failure-inducing inputs. ([cs.purdue.edu][4])

**What you implement:** a 40–80 LOC `ddmin(C, oracle)`:

* Start with chunk size `k=2`; split (C) into `k` contiguous chunks (deterministic order).
* For each chunk (D), test `FEASIBLE(C\D)`. If still infeasible, set `C := C\D` and restart with `k = 2`.
* If no chunk works, increase `k = min(len(C), 2k)` and repeat until `k > len(C)`.
* The result is **1-minimal**.

Either A or B is acceptable. If you want **fewer solves**, prefer QuickXplain; if you want **simplest code**, start with `ddmin`.

---

## How this satisfies the IIS definition

When your function returns (K\subseteq C):

* By construction, **WO-7 infeasible** on (K) (because we only keep subsets that still fail), **and**
* For **every** row (r∈K), `FEASIBLE(K \ {r}) == True` (ddmin enforces this 1-minimality; QuickXplain guarantees minimality by recursion).

That matches the formal IIS definition used by commercial solvers like Gurobi/CPLEX (subset infeasible; remove any one row → feasible). ([support.gurobi.com][2])

---

## Determinism & receipts

* **Fix a total order** on row IDs: e.g., `("mask", p, c) < ("equalizer", s, c, p, q) < ("cell", r, j)` and sort candidates **once**. Both QuickXplain and ddmin will then produce a **deterministic** IIS for a given pack.
* In `wo09b.json`, include:

  ```json
  "iis": {
    "present": true,
    "rows": [
      {"type":"mask","p":123,"c":4},
      {"type":"equalizer","bin":8,"color":1,"p":55,"q":61}
    ],
    "minimality_checks": [
      {"row_idx":0, "feasible_when_removed": true},
      {"row_idx":1, "feasible_when_removed": true}
    ]
  }
  ```

  That “feasible_when_removed” list is the **byte-exact proof of minimality** your reviewer can replay.

---

## Practical tips (to keep it small and fast)

* Start (C) from the **WO-7 fail report** (only rows implicated). If you don’t have a precise implicant set, start from all hard rows in scope for the pack; ddmin will shrink it.
* Cache `FEASIBLE()` results by a canonical key of rows removed (e.g., a frozenset of row IDs → True/False) to avoid duplicate solves.
* If multiple test inputs exist for a task, run IIS per test input independently (each has its own mask/rows).

---

## Why this is spec-conformant

* **We are not “computing IIS” inside the solver** (no LP toolbox). We use **WO-7 as an oracle** and standard conflict-set algorithms (QuickXplain / ddmin) to obtain a **1-minimal** hard-tier subset — exactly what §11 wants (a minimal conflict subsystem), and consistent with the formal IIS definition used in LP/MIP literature and tool docs. ([support.gurobi.com][2])
* This respects **laminar precedence** (§10): faces/quotas have already been relaxed; only mask/equalizers/cell rows remain. The IIS you output is about those **never-relaxed** truths.

If you want, I can drop a ~70-line `ddmin()` and ~90-line `quickxplain()` skeleton wired to a `feasible(rows_removed)` callback so you can paste and run it against your current WO-7 runner.

[1]: https://developers.google.com/optimization/reference/graph/min_cost_flow/SimpleMinCostFlow?utm_source=chatgpt.com "SimpleMinCostFlow | OR-Tools"
[2]: https://support.gurobi.com/hc/en-us/articles/15656630439441-How-do-I-use-compute-IIS-to-find-a-subset-of-constraints-that-are-causing-model-infeasibility?utm_source=chatgpt.com "How do I use 'compute IIS' to find a subset of constraints ..."
[3]: https://aaai.org/Papers/AAAI/2004/AAAI04-027.pdf?utm_source=chatgpt.com "QUICKXPLAIN: Preferred Explanations and Relaxations for ..."
[4]: https://www.cs.purdue.edu/homes/xyzhang/fall07/Papers/delta-debugging.pdf?utm_source=chatgpt.com "Simplifying and Isolating Failure-Inducing Input"
