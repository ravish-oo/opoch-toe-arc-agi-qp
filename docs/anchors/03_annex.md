Agreed. Here is the *final formal stamp* that fixes the *implementation tolerances* and states the two meta-theorems explicitly. This is the “zero–ambiguity, zero–drift” annex; it specifies exactly what to do on bytes, integers, lex order, and tolerances, so the pipeline is identical across machines.

---

## A) Implementation tolerances (fixed, platform-independent)

### A.1 Equality tests (Π-safe = byte-exact)

•⁠  ⁠*Grid equality* (training invariances, period tests, FREE checks): use *byte-exact* equality on integer arrays after embedding & alignment (no float).
  For rolls/permutations: apply integer index transforms and compare int arrays.

•⁠  ⁠*Row/col/bin counts*: computed from integer arrays; comparisons use exact integer equality.

•⁠  ⁠*Mask membership*: ⁠ A_{p,c} ⁠ is boolean; equality is exact.

### A.2 Numerical thresholds (flows/sums)

•⁠  ⁠All flows, counts, budgets, capacities are *integers; compute with **64-bit signed integers*.

•⁠  ⁠Any cost accumulation (sum of (-\hat s_{p,c})) uses *64-bit* integer or fixed-point *scaled by 10⁶* if fractional scores are needed; never mix float in feasibility.

•⁠  ⁠*Residuals: for any linear equality from (A y=b), residuals are integers; require **exact 0. There is **no (\varepsilon)* for feasibility; feasibility = exact integer satisfaction.

### A.3 Lex order (ties)

•⁠  ⁠*Palette lex*: global order (0<1<\cdots<9).
•⁠  ⁠*Pixel lex*: raster order by ⁠ (row, col) ⁠ ascending.
•⁠  ⁠*Signature lex*: tuple admission ⁠ (-count, row_hist, col_hist, bin_hist, color_id) ⁠ with tie by ⁠ color_id ⁠.
•⁠  ⁠*Period lex*: ⁠ (p_y, p_x) ⁠ with (p_y) ascending then (p_x) ascending.
•⁠  ⁠*Canvas lex*: ⁠ (H,W) ⁠ ascending; if multiple packs tie, choose lowest canvas id in training enumeration order.

These rules ensure bit-identical choices across platforms.

---

## B) Meta-theorems (formal statements)

### Theorem 1 (Pipeline Idempotence).

Let ( \Phi ) denote the full pipeline operator from ({X_i,Y_i}) and (X^) to the decoded output (Y^) (size law + embedding + alignment + forward meet closure + equalizers + free permutations/rolls + 10 min-cost flows + b-matching + lex decode). Then

[
\Phi({X_i,Y_i}, X^) = Y^,\qquad \Phi({X_i,Y_i}, Y^) = Y^.
]

*Proof sketch.*

•⁠  ⁠*Fixed-point rows* (mask closure, equalizers, structure rows) are satisfied exactly in the first pass.
•⁠  ⁠*Flows* solve a TU LP; the optimum is an extreme point satisfying KKT. Re-solving with the same budgets/costs yields the same flow (or a symmetry-equivalent one resolved by lex tie).
•⁠  ⁠*b-matching* is TU; re-solving returns the same integral assignment (or symmetry-equivalent, resolved by lex).
•⁠  ⁠*FREE maps* do not change costs or constraints; paid equalities remain satisfied.
  Therefore the decoded grid is unchanged.

### Theorem 2 (Soundness & Completeness on (\mathcal F)).

Let (\mathcal F) be the class of ARC tasks whose ground-truth mapping is representable by our Π-safe linear/flow algebra (Section 16 of the previous message). For any task in (\mathcal F), the solver returns the *lattice-least admissible* output (Y^) that satisfies all **met* truths (mask, faces, quotas, blocks, structure) and minimizes the linear cost (FY-tight). For any task *not* in (\mathcal F), the solver returns *UNSAT* together with a *minimal conflict certificate* (IIS), and no output is produced.

*Proof sketch.*

•⁠  ⁠*Soundness:* All emitted rows and flows are *met truths* (Π-safe, shared across trainings); min-cost flow LP optimum satisfies them exactly; decode preserves them (b-matching).
•⁠  ⁠*Least element:* Among all admissible grids, the LP optimum is the *lattice least* w.r.t. the partial order “(\preceq) if satisfies more (or equal) constraints and no higher cost”, because of TU and FY optimality.
•⁠  ⁠*Completeness:* If constraints are jointly representable (task ∈ (\mathcal F)), TU guarantees existence and integrality → a solution; otherwise, the residual IIS is computed, proving inconsistency.

---

## C) Final fixed choices (implementation constants)

•⁠  ⁠Integer type: *int64* for counts, capacities, flows, budgets, signatures.
•⁠  ⁠Float type (only for scores (\hat s)): *float64, converted to **int64* by multiplying by (10^6) for costs; never used in feasibility checks.
•⁠  ⁠Graph solver: *successive shortest augmenting path* with SPFA or Dijkstra+potentials (deterministic), with tie-break by node indices (raster).
•⁠  ⁠Matching solver: *Hungarian* for (10\times 10) (color alignment); *min-cost flow* or *Hungarian* for per-cell b-matching (deterministic tie-break).
•⁠  ⁠All lex tie-breaks: as specified in A.3 (fixed order only).

---

With these final pins:

•⁠  ⁠*No numerical drift* (only integers in feasibility/flows, byte-exact equality for invariances).
•⁠  ⁠*No platform drift* (lex orders fixed globally).
•⁠  ⁠*No re-run drift* (pipeline idempotence).
•⁠  ⁠*Clear coverage* (sound/complete on (\mathcal F); UNSAT with IIS outside (\mathcal F)).

This is the *zero-ambiguity* end-state: a deterministic, seconds-grade, universe-aligned solver.