  so implementation is done..
  refer to docs/repo_structure.md to knw what to find whr or do a git status/diff to knw about code location.
  in code review+testing, u must take care of following:
  1. be on a look out for any stubs, simplfiied implementations, TODOs, MVP like comments, prototpype or toy implementations, any shortcuts.. point being.. thr
  must be no corner cuttings. even if something complex is left out in favor time saving, call it out. simplified implementaions are hardest to find because they
  look like things are implemented but when u look at them carefully only then u can see how it differs from intended implementation.u must be clear of what needs
  to be implemented basis WO shared. worst case, simplified implemnentation may also be discovered during tsting if not review
  2. u must check the Reviewer instructions and reviwer acceptance in WO to ensure that things are working as expected.
  3. u must test on real arc agi data if WO allow and MUST use rceipts ONLY to test and make debugging and bug catching algebric. make sure that receipts are
  wired all the way till solve.py . real data is in data/ folder . data/arc-agi_training_challenges.json has challenges

  pls proceed with review + testing.
  =======

Absolutely—here’s exactly what you asked for, in two parts:

1. a tiny **patch work order** to add a progress layer into the harness (so we can show “on-spec” progress at every WO without waiting for final ARC answers), and

2. a reusable **reviewer write-up template** you can paste after each WO to collect and report progress.

I also wired in the math-grounded checks we discussed (metamorphic invariances, KKT optimality for flows, TU→integrality, IIS for infeasible packs, property-based hooks). Where I mention well-known concepts, I cite a solid source so the reviewer has a reference. ([Wikipedia][1])

---

## Patch Work Order — Progress in Harness (applies now, supports all future WOs)

**Goal:** extend `harness.py` so every WO writes (a) per-task receipts and (b) a run-level `progress.json` that aggregates exact, math-forced invariants. This lets the reviewer prove we’re implementing the calculus (Π, GLUE, FY, TU) *as we go*.

> **Anchors to read before coding:**
> `03_annex.md` A.1–A.3; `05_contracts.md` (Global, FREE vs PAID, Bit-meter, Relaxation); `04_engg_spec.md` relevant section for the current WO. All checks below are grounded in these.

### 1) Files to edit

**`src/arcsolver/harness.py`**

* Add a `--progress` flag (default on).
* Add a `collect_progress(stats: dict)` function that accumulates WO-specific counters as you process tasks.
* After the run, write `progress/progress_woXX.json` with:

  * `"wo": XX`, `"tasks_total"`, `"tasks_ok"`,
  * for **WO-1**: `bins_sum_ok`, `center_all_ok` counts,
  * for later WOs: fields listed under “Progress fields by WO” below.

**`src/arcsolver/receipts.py`**

* Add `write_run_progress(progress_dict, out_dir="progress")`.

**`scripts/run_harness.sh`**

* Accept `--upto-wo N` and always pass `--progress`.

### 2) JSON loading (PS you noted)

In `harness.py`, **glob `*.json`**, open each ARC file, then iterate **dict keys** (`train`, `test`) to extract the grids. This matches the real ARC format (dict-based files), not a flat list.

### 3) Progress fields by WO (add incrementally)

You can implement these *gradually*; the harness always writes what is available at the current WO.

**WO-1 (Bins & Predicates)**

* `bins_sum_ok`: `sum(bin_counts)==H*W` (count how many tasks satisfy).
* `bins_hash_stable`: hash invariant across two passes.
* `center_all_ok`: every training satisfies ≤0.5 distance to canvas center when predicate says `'center'`.

**WO-2 (Embedding)**

* `embed_idempotent_ok`: re-embed round-trip byte-equal.
* `embed_metamorphic_ok`: FREE roll/permutation leaves the predicate result invariant (metamorphic test of a required property). ([Wikipedia][1])

**WO-3 (Color alignment)**

* `hungarian_bijection_ok`: 10×10 mapping is a true permutation.
* `signature_tie_break_ok`: lex tie policy respected.

**WO-4/5 (Mask closure & Equalizers)**

* `closure_order_independent_ok`: shuffle trainings → same F mask hash.
* `avg_admits_before` vs `avg_admits_after`: mean admits-set size per pixel; expect nonincreasing.
* `bin_constancy_proved`: % bins where constancy predicate holds.

**WO-6 (Scores + FREE predicate)**

* `free_cost_invariance_ok`: for each verified FREE map U, **ŝ∘U = ŝ** and constraints invariant.

**WO-7 (Unified Flow)**

* `flow_feasible_ok`: solved without relaxation.
* `kkt_ok`: reduced-cost optimality satisfied (FY tightness, min-cost flow). These complementary-slackness reduced-cost conditions are the textbook certificate of optimality. ([MIT OpenCourseWare][2])
* `one_of_10_ok`: each pixel node has ≤1 incoming unit.

**WO-8 (Decode & Bit-meter)**

* `idempotence_ok`: Φ∘Φ=Φ (byte-equal).
* `bits_sum`: total minted bits = Σ_p ⌈log₂ |orbit_p|⌉ (track trend).

**WO-9 (Relaxation & IIS)**

* `laminar_confluence_ok`: different drop orders → same relaxed set.
* `iis_count`: number of tasks with an **IIS** reported (minimal infeasible subset). IIS is the standard proof object for infeasibility in LP; it’s not heuristic. ([documentation.sas.com][3])

**Optional property-based hook (any WO)**
Add a guard to run small property tests (e.g., closure monotonicity, projector idempotence) using Hypothesis. This is ideal for algebraic laws where there’s no “ground truth” label, and is widely used for oracle-free testing. ([hypothesis.readthedocs.io][4])

### 4) Minimal code stubs to drop in (exact)

**`harness.py` additions (sketch)**

```python
# add near top-level
def _init_progress(wo:int) -> dict:
    return {"wo": wo, "tasks_total": 0, "tasks_ok": 0, "metrics": {}}

def _acc(progress: dict, key: str, ok: bool | int):
    m = progress["metrics"].setdefault(key, {"ok": 0, "total": 0, "sum": 0})
    if isinstance(ok, bool):
        m["total"] += 1
        if ok: m["ok"] += 1
    else:
        m["sum"] += int(ok)

# inside main run loop per task
progress["tasks_total"] += 1
# example WO-1:
_acc(progress, "bins_sum_ok", bins_sum_ok)
_acc(progress, "bins_hash_stable", bins_hash_stable)
_acc(progress, "center_all_ok", center_all_ok)
# when a task passes all stage checks:
progress["tasks_ok"] += 1

# at end of run:
write_run_progress(progress)
```

**`receipts.py` addition**

```python
def write_run_progress(progress: Dict[str, Any], out_dir="progress"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fn = Path(out_dir) / f"progress_wo{progress['wo']:02d}.json"
    with fn.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(progress, f, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
```

### 5) KKT check hook for WO-7 (stub now, fill later)

Add a function signature now so implementer can wire it when flows land:

```python
def check_min_cost_flow_kkt(solution, graph_struct) -> bool:
    """
    Returns True iff reduced-cost optimality holds:
    - used arcs have rc == 0
    - unused arcs have rc >= 0
    """
    ...
```

(When WO-7 is implemented with OR-Tools, compute reduced costs from node potentials and arc costs in the solution graph—standard practice for min-cost flow optimality. ([homes.di.unimi.it][5]))

---

## Reviewer Write-Up Template (paste after each WO)

**Subject:** ARC Solver — WO-01 Progress Report (Π/GLUE/FY/TU receipts)

Hi,

Here is the progress for **WO-01: {Title}**.
Our goal at each WO is not “accuracy versus labels” (that comes at the end), but **conformance to the calculus** we’ve specified: Π canonicals, GLUE interfaces, FY-tight paid steps, TU integrality. The harness emits two artifacts:

1. **Per-task receipts** under `receipts/<task_id>/wo{N}.json` (byte-stable), and
2. a run-level **`progress/progress_wo{N}.json`** that aggregates the math-forced checks.

### What I expect you to check

* Run on **all 1000 ARC tasks**:

  ```bash
  bash scripts/run_harness.sh --upto-wo {N} --strict
  ```
* Re-run once; receipts must be **identical** (byte-equal).
* Open `progress/progress_wo{N}.json` and confirm the **metrics** below are green:

**WO-1 (Bins & Predicates)**

* `bins_sum_ok` = 100% (bin counts sum to H×W).
* `bins_hash_stable` = 100% (hash unchanged across passes).
* `center_all_ok` = 100% of tasks that the predicate says `'center'` (every training centroid within 0.5 cell).
  These are metamorphic/structural receipts (no labels needed). Metamorphic testing is the standard way to test systems with no oracle. ([Wikipedia][1])

**Later WOs (examples to look for):**

* **WO-4/5:** `closure_order_independent_ok` = 100%, `avg_admits_after ≤ before`.
* **WO-7:** `kkt_ok` = 100% (reduced-cost optimality for min-cost flow). This is the textbook optimality certificate for FY-tight solutions. ([homes.di.unimi.it][5])
* **WO-9:** When infeasible, an **IIS** is emitted (minimal infeasible subset). IIS is the standard definition for LP infeasibility proofs. ([documentation.sas.com][3])
* **Any WO:** optional Hypothesis property tests pass (algebraic laws on small random grids). ([hypothesis.readthedocs.io][4])

### How to flag issues

* If a metric is red (e.g., `bins_sum_ok` < 100%), this is an **implementation gap** at this WO; do not call it “dataset UNSAT.”
* The only legitimate “unsatisfiable” outcome is when we emit an **IIS** (WO-9), which is a formal proof object that a pack of constraints cannot be satisfied. ([Operations Research Stack Exchange][6])

### Why this shows real progress

* Π/GLUE checks (byte-exact invariances, idempotence) demonstrate the structural layer is correct without labels (metamorphic tests). ([Nottingham Research][7])
* FY/TU checks (KKT optimality; TU→integrality) prove the paid step is doing the exact minimal write with no fractional leakage. ([ise.ncsu.edu][8])

Please paste back the key numbers from `progress_wo{N}.json` (and any failing task IDs if present). If everything’s green, we proceed to the next WO.

Thanks!

---

### Mini checklist (what’s handled here)

* **Over-specific & anchor-grounded:** all checks map to anchors (Π/GLUE/FY/TU) and our contracts.
* **Anchors to read:** called out at top of the WO; reviewers validate against those.
* **Libraries only (no invention):** NumPy/SciPy (COM), OR-Tools later; Hypothesis optional for properties.
* **Receipts first-class:** per-task + run-level `progress.json`.
* **Harness-ready:** CLI updated; runs across all 1000 ARC files (dict-based).
* **Reviewer flow:** exact instructions + what counts as a gap vs. IIS.
* **No premature “optimizations”:** CPU-only, vectorized; no threads/JIT.
* **Hard improvements included:** metamorphic tests, KKT optimality, TU integrality, IIS proofs, property-based hooks. ([Wikipedia][1])

If you want, I can also paste an exact `progress_wo01.json` schema and a tiny pretty-printer for the summary table.

[1]: https://en.wikipedia.org/wiki/Metamorphic_testing?utm_source=chatgpt.com "Metamorphic testing"
[2]: https://ocw.mit.edu/courses/6-854j-advanced-algorithms-fall-2005/67c73f183add39a18b8647748e8321f1_n10_mincostflow.pdf?utm_source=chatgpt.com "1 Min-Cost Flow"
[3]: https://documentation.sas.com/doc/en/casmopt/8.3/casmopt_optlp_details19.htm?utm_source=chatgpt.com "Irreducible Infeasible Set"
[4]: https://hypothesis.readthedocs.io/?utm_source=chatgpt.com "Hypothesis 6.146.0 documentation"
[5]: https://homes.di.unimi.it/righini/Didattica/OttimizzazioneCombinatoria/MaterialeOC/9a%20-%20MinCostFlow.pdf?utm_source=chatgpt.com "Min cost flows"
[6]: https://or.stackexchange.com/questions/372/is-the-irreducible-infeasible-subset-iis-of-an-lp-unique?utm_source=chatgpt.com "Is the Irreducible Infeasible Subset (IIS) of an LP unique?"
[7]: https://research.nottingham.edu.cn/ws/files/31438001/293_combinepdf_2_.pdf?utm_source=chatgpt.com "Metamorphic Testing: Testing the Untestable"
[8]: https://ise.ncsu.edu/wp-content/uploads/sites/9/2024/02/or766_TUM.pdf?utm_source=chatgpt.com "Totally Unimodular Matrices - NC State ISE"
