# 05_contracts.md — Determinism & Fidelity Contracts

This file is a non-negotiable checklist. Every module must satisfy these contracts.
Violating any item is a bug.

## Global

- Python: 3.11.x
- Libraries: numpy==2.1.x, scipy==1.13.x (Hungarian or ndimage), scikit-image==0.23.x (optional for labeling), ortools==9.10.x
- Threads: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1
- Hash seed: PYTHONHASHSEED=0
- Dtypes:
  - Grids: np.int32 (values in {−1,0..9}; −1 only for padding during embedding)
  - Counts / costs / indices: np.int64
  - Scores ŝ: float64 internally, then **cost = round(−ŝ * SCALE)** with SCALE=1_000_000 → int64
  - Never use floats in feasibility/equality.
- Equality: byte-exact on integer arrays; no epsilons.
- Orders: palette lex 0<…<9; pixel lex (row,col) ascending; periods lex (p_y,p_x); canvas lex (H,W);
  signatures lex (−count, row_hist, col_hist, bin_hist, color_id).
- Background is **0**; padding sentinel is **−1**. “Content” means `!= 0` (not `!= −1`).

## FREE vs PAID (decidable)

A map U is **FREE** iff:
1) Cost-invariant: J(Uy) = J(y) for all feasible y (equivalently ŝ∘U = ŝ),
2) Constraint-invariant: A U = A for all emitted linear equalities (mask, equalizers, faces, blocks),
3) U is a verified permutation/roll (period or palette alignment) from trainings.

Otherwise it is **PAID** and must be encoded as linear rows or costs.

## Canonical predicates

- Centering: centroid of non-background (value≠0) within 0.5 cell of canvas center in both axes on **every** training → 'center', else 'topleft'.
- Periods: equality under integer rolls only; byte-exact; pick shared lex-min (p_y,p_x).
- Color alignment: signatures = (−count, row_hist, col_hist, bin_hist, color_id);
  costs for Hungarian are int64; lex tie is encoded via cost offsets or pre-ordering.

## Masks & Equalizers

- Forward meet closure must be **monotone, extensive, idempotent** and order-independent.
- Constancy on bin B_s for color c holds iff all trainings have zero variance on B_s∩{A_{p,c}=1}.
- Equalizers only tie within (B_s × {c}); bins are disjoint; rows commute.

## Scores (ŝ) & Costs

- ŝ must be Π-safe (depend only on bins/mask/verified free transforms); never on raw color ids.
- If a FREE symmetry is verified, either project first or transport/average ŝ so ŝ∘U = ŝ.
- Bound: max |ŝ| ≤ 10^6 so ∑|cost| ≪ 2^63; assert in code.

## Flows (unified pixel-level graph)

- Build one graph that routes bin supplies → (optional rows/cols) → shared cell node → per-color lane → **per-pixel node (cap=1)** → sink.
- Do **not** create arcs for forbidden pairs (A_{p,c}=0).
- All IDs (nodes, arcs) are added in sorted raster/lex order.
- Solver: OR-Tools SimpleMinCostFlow; integer capacities/costs only; deterministic augmentation.
- After solve: check flow conservation at every node; check no forbidden arc used; shared cell caps respected.
- KKT reduced-cost check: all used arcs have zero reduced cost; all unused arcs have ≥0.

## One-of-10 & Decode

- With pixel nodes cap=1 the exclusivity is enforced by flow. If a separate b-matching is used, it must be TU and deterministic.
- Decode Y[p] = the unique color whose arc into pixel p carries flow; if multiple (shouldn’t), apply palette lex tie (and ledger bits).

## Relaxation (laminar)

- Tiers: Cell/FD equalities ⊇ Row/Column faces ⊇ Bin quotas ⊇ Free symmetries.
- Drop only non-shared at lowest tier first; if still infeasible, drop dominated shared rows.
- Greedy removal must be **minimal** and **confluent** (laminar matroid).
- UNSAT must return an **IIS** (irreducible infeasible subsystem) with rank proof.

## Bit-meter

- For each pixel p, compute orbit 𝒪_p of indistinguishable colors under remaining symmetries and constraints; bits at p = ⌈log2 |𝒪_p|⌉.
- Total ΔN = sum over pixels; E_min = k_B T ln2 · ΔN (numeric k_B T only for reporting).

## Idempotence & Determinism

- Φ∘Φ = Φ byte-exact.
- No RNG; no nondeterministic iteration order; all JSON output sorted keys; newline '\n'.
- Cross-platform (Linux/macOS) outputs must be byte-identical given pinned deps and env.

## Receipts (always on)

For each task & stage: write JSON with hashes of bins and F mask, chosen laws, permutations, symmetry groups, flow stats (cost, nodes/arcs, conservation, KKT pass), bits, and any IIS.
