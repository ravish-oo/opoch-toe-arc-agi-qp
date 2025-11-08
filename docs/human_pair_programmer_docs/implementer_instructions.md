# Understand what we are doing
pls understand what we are doing. read @docs/anchors/maths/01_math_spec.md @docs/anchors/maths/03_math_spec_patch2.md 
@docs/anchors/maths/02_math_spec_patch1.md
  @docs/anchors/maths/04_math_spec_addendum.md @docs/anchors/engineering/computing_spec.md

# Understand ur role
Implement exactly the WO interface; no stubs, no TODOs, no extra helpers outside the spec. 
Use only the allowed primitives and frozen orders; no randomness, no floats, no heuristics.
On any unprovable case: return silent (A=all, S=0) or the specified FAIL (UNSAT, SIZE_INCOMPATIBLE, FIXED_POINT_NOT_REACHED).
Emit receipts for every public call (hashes, counts, attempts list) and ensure double-run identical hashes.
In code keep pure functions, zero side effects except receipts.


# wo prompt
here is the WO. do refer to @docs/repo_structure.md to knw the folder structure.
  [Pasted text #1 +161 lines]
  ---
  pls read and tell me that u hv understood/confirmed/verified below:
  1. have 100% clarity
  2. WO adheres with ur understanding of our math spec and engg spec and that engineering = math. The program is the proof.
  3. u can see that debugging is reduced to algebra and WO adheres to it 
  4. no room for hit and trials

once u confirm above, we can start coding!

# how to debug
u may hv tried few things but see if u want to try something from here. also when u say u hypothesize, why do i need to to do guess and hope. i mean best part of programming is that u can print output at each step and study  it when out and find exactly when it breaks.. that's what debuggers formalized but old school way us to print the outputs and see. u r trained on code that probably didnt hv these prints for debug but that's how its done. 

so u must not "hypothesize" and fix. hypothesize to investigate, print outputs and settle hypothesis rather than hit and hope. that just wont work. so get back to 0th principle of coding. print and see and fix. simple as that.. 
hope this helps 


# latest
now ur role is that of an implemetner. following is the work order u need to implement.
  refer to @docs/repo_structure.md  to knw repo strucutre.
  data/ folder has training challenges of arc agi 
  do create a venv if u need any installations.
  ---

  ---
  wo above has all the details u need. pls note any code in WO is sketch/illustrative only
  pls understand and see if u hv 100% clarity on what needs to be done.
  once u confirm above, we can start coding!
─────────────────────────────────────────────



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

