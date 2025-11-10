# WO-9A — Packs & Size Law

**Goal.** Enumerate all **deterministic “packs”** the solver may try in WO-9B. A pack fixes:

* the **output size law** (00 §1),
* the **faces mode** (rows_as_supply / cols_as_supply / none) when faces are present (00 §8),
* the set of **verified FREE maps** (from WO-2/WO-3; no Hungarian), and
* a few **quick feasibility flags** (quota vs mask, faces consistency)
  so WO-9B can run **laminar greedy relaxation** or select a feasible pack.

Everything is **Π-safe, byte-exact, and deterministic**; we only use mature stdlib/NumPy functions (no algorithms to invent).

---

## Anchors to read before coding

* @docs/anchors/00_math_spec.md : §1 (size inference), §8 (faces optional), §15 (no hidden search)
* @docs/anchors/01_addendum.md : §10 (laminar precedence), §11 (IIS), §12 (packs)
* @docs/anchors/03_annex.md : A.1–A.3 (byte exact integers, lex order)

---

## Libraries (mature, documented)

* **NumPy** for set/count ops, stable ordering:

  * `np.array_equal` (byte-exact equality) ([numpy.org][1])
  * `np.argsort(..., stable=True)` or `np.sort(..., stable=True)` for **stable** lex ordering (NumPy ≥2.0) ([numpy.org][2])
  * `np.unique(..., return_counts=True)` or `np.unique_counts` for tallies (if needed) ([numpy.org][3])
* **Python stdlib**

  * `json.dumps(..., sort_keys=True, separators=(",",":"))` for canonical JSON (stable receipts) ([Python documentation][4])
  * `hashlib.sha256()` to hash inputs/pack IDs (determinism) ([Python documentation][5])
  * `pathlib`, `dataclasses`, `typing`

*No SciPy, no OR-Tools here; this stage only composes metadata.*

---

## Inputs (from cache; never from receipts)

* **WO-1/WO-2**: `H_out, W_out, law_used` (and any **alternate size laws** if you record them)
* **WO-4**: `A_mask:(N,C) bool`, `bin_ids:(N,) int64`
* **WO-5**: optional faces arrays (`faces_R:(H×C)`, `faces_S:(W×C)`), canonical **equalizer rows** (only needed for 9B; not consumed here)
* **WO-6**: `free_maps_verified` (rolls/perms with `verified_all_trainings:true`)
* **WO-7**: nothing required

All arrays come from `.cache/woNN/…npz`. Receipts are only for audit.

---

## Outputs

* **packs**: a deterministic, lex-sorted list of packs; each pack records

  * `size_law`: `{"law": "constant|linear|content", "H":…, "W":…}`
  * `faces_mode`: `"none" | "rows_as_supply" | "cols_as_supply"`
  * `free_maps`: list of `{"type":"roll"|"perm", …}` already **verified**
  * `quick`: static flags per pack (see §Quick checks)
  * `pack_id`: stable string/sha for cross-WO references
* **Receipt** `wo09a.json` (see §Receipts)

---

## Pack construction (deterministic)

1. **Size law enumeration (00 §1).**

   * Start with the chosen size law from WO-1/WO-2.
   * If you stored more than one valid law (e.g., constant vs content-based), include **each law** that passed WO-1’s proof.
   * Order laws **lexicographically** by `("law", H, W)` using **stable** sort so equal keys preserve input order (NumPy: `stable=True`). ([numpy.org][2])

2. **Faces mode (00 §8).**

   * If **no faces arrays** exist in WO-5 cache → only `faces_mode="none"`.
   * If faces exist:

     * produce packs with `faces_mode="rows_as_supply"` and `faces_mode="cols_as_supply"`, **and** a pack with `faces_mode="none"` (faces are “optional if shared”).
   * Do **not** “detect” faces here; just reflect presence/absence. (§15: no hidden search.)

3. **FREE maps (02 §B/§G).**

   * Include **only** `free_maps_verified` emitted by WO-2/WO-3 (e.g., `{"type":"roll","dy":p_y,"dx":p_x}` or `{"type":"perm","perm":[…]}`).
   * Exclude WO-3 **Hungarian** permutations; those are alignments, not symmetries.

4. **Pack ID (stable).**
   Build a stable textual ID (for receipts and logs), e.g.:
   `"size=HxW|faces=rows|free=[roll(2,0);perm(0,1,…)]"` and also compute `sha256(pack_id.encode())` for quick equality checks (determinism). ([Python documentation][5])

5. **Final pack order.**

   * Sort packs lex by: `(law, H, W, faces_mode, free_maps_lex)` using **stable** sort.
   * `free_maps_lex`: string-join of each map in a canonical form (rolls before perms; numbers in ascending order).

---

## Quick checks (no solving; O(NC); gates for 9B)

For each pack, compute (integers only, byte-exact):

* **Capacity conflict** (00 §8 + 03.A.1):
  For each bin/color, `allowed[s,c] = |{p ∈ B_s : A_mask[p,c]==1}|`.
  If **any** `q[s,c] > allowed[s,c]` → record
  `capacity_conflicts += [{"bin":s,"color":c,"q":q,"allowed":allowed}]`, and set `quick.capacity_ok = False`.

* **Faces consistency** (00 §8):
  If faces present, check per color (c),
  `sum_s q[s,c] == sum_r R[r,c]` and/or `== sum_j S[j,c]`.
  If not, set `quick.faces_conflict = True`.

* **Trivial periods** (optional flag):
  If a roll is `(p_y==H or p_x==W)` (identity), flag `quick.trivial_period=true` (no effect, but helps reviewers).

These **do not** modify packs; they help 9B decide whether to try a pack first or prepare for relaxation.

---

## Receipts (first-class)

Write `receipts/<task_id>/wo09a.json`:

```json
{
  "stage": "wo09a",
  "packs": [
    {
      "pack_id": "size=12x12|faces=rows_as_supply|free=[roll(2,0)]",
      "size_law": {"law":"content","H":12,"W":12},
      "faces_mode": "rows_as_supply",
      "free_maps": [{"type":"roll","dy":2,"dx":0}],
      "quick": {
        "capacity_ok": true,
        "faces_conflict": false,
        "capacity_conflicts": []
      }
    }
  ],
  "packs_count": 3,
  "hash": "<sha256 of canonical JSON>"
}
```

* Serialize with `json.dumps(..., sort_keys=True, separators=(",",":"))` for stable receipts. ([Python documentation][4])
* Include a top-level `hash` (sha256 of the packed JSON) for determinism audits. ([Python documentation][5])

---

## Stage runner & harness

**Code placement**

* `src/arcsolver/packs.py` — pure functions to enumerate packs and compute quick checks.
* `src/arcsolver/stages_wo09a.py` — loads caches (WO-1/2,4,5,6), calls `packs.enumerate_packs(...)`, writes `wo09a.json`, saves `packs.json` (identical content) to `.cache/wo09/<task>.packs.json`.

**Pipeline registration**

```python
from .pipeline import STAGES
STAGES[9] = run_wo09a
```

**Harness flags**

* Add `--upto-wo 9` to execute WO-1…WO-9A.
* Add progress counters:

  * `packs_exist_ok` (packs_count ≥ 1)
  * `packs_deterministic_ok` (wo09a.json hash stable over two runs)
  * `quick_checks_ok` (no internal exceptions; flags computed)

No “god function”: WO-9A never re-implements earlier WOs; it only **loads cache** and writes packs.

---

## Reviewer – ultra-short acceptance (WO-9A)

Run:

```bash
bash scripts/run_harness.sh --upto-wo 9 --strict --progress
```

**Must be 100% (over accepted instances):**

* `packs_exist_ok` — every task has ≥1 pack (at least the chosen WO-1 size & faces=none).
* `packs_deterministic_ok` — re-run → identical `wo09a.json` hash.
* `quick_checks_ok` — capacity/face flags computed without crash; capacity tallies are **byte-exact** integers.

Spot-check 3–5 tasks:

* Verify pack order is lex-stable (size law then faces then free maps).
* If faces present, faces totals equal `sum(q[s,c])` per color; else `faces_mode:"none"`.
* FREE maps list equals WO-6 `free_maps_verified`; zero Hungarian entries.

If anything <100% → **WO-9A implementation gap** (pack enumeration, ordering, or quick-check arithmetic). Attach `wo09a.json` + cache hashes.

---

## Anti-optimization note (CPU)

WO-9A is O(NC) counting and string assembly. No threading/JIT; use integer ops and NumPy reductions only.

---

[1]: https://numpy.org/doc/2.1/reference/generated/numpy.array_equal.html?utm_source=chatgpt.com "numpy.array_equal — NumPy v2.1 Manual"
[2]: https://numpy.org/devdocs/reference/generated/numpy.argsort.html?utm_source=chatgpt.com "numpy.argsort — NumPy v2.4.dev0 Manual"
[3]: https://numpy.org/doc/stable/reference/generated/numpy.unique.html?utm_source=chatgpt.com "numpy.unique — NumPy v2.3 Manual"
[4]: https://docs.python.org/3/library/json.html?utm_source=chatgpt.com "JSON encoder and decoder — Python 3.14.0 documentation"
[5]: https://docs.python.org/3/library/hashlib.html?utm_source=chatgpt.com "hashlib — Secure hashes and message digests"

## Known interim approach
 ⚠️ DOCUMENTED INTERIM APPROACH (Transparent, Not Hidden)

  Single Size Law (stages_wo09a.py:42-56, 114-115)

  Spec expectation (lines 69-71):
  "If you stored more than one valid law (e.g., constant vs content-based), include each law that passed WO-1's proof."

  Implementation:
  # INTERIM: single size law (spec prefers extending WO-1 to enumerate all proven laws)
  if mode in ["topleft", "center"]:
      law_type = "constant"
  else:
      law_type = "content"

  size_laws = [size_law]  # Single law only

  Receipt flag:
  "size_laws_complete": False,
  "note": "interim single size law; WO-1 extension pending"

  Assessment:
  - ✓ Transparent documentation in code and receipt
  - ✓ Flag allows downstream stages to detect interim status
  - ⚠️ Deviates from spec's "enumerate all proven laws"
  - ⚠️ Single-law limitation may miss valid alternative size laws

  Impact: Functional for single-law tasks but incomplete per spec. Not a "hidden" simplification.

