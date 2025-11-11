# Bottom-up WO plan (atomic, ≤300–400 LOC per WO)

1. **WO-0: Core types & guards** ✅ COMPLETED
   dtypes, lex orders, env vars, helpers. Unit tests: dtype & env asserts.

2. **WO-1: Bins & predicates** ✅ COMPLETED
   periphery–parity bins, center predicate, bbox on `!=0`. Tests: fixed hashes.

3. **WO-2: Embedding & Period tests** ✅ COMPLETED
   topleft/center embed; byte-exact period detection. Tests: round-trip.

4. **WO-3: Color signatures + Hungarian adapter**
   build signatures; deterministic alignment. Tests: adversarial ties.

5. **WO-4: Forward meet closure**
   order-free closure + color-agnostic lift. Tests: shuffle trainings, same mask.

6. **WO-5: Equalizers & structure rows**
   spanning-tree equalizers; settle G; harmonic fill (row builders only). Tests: commuting rows; uniqueness.

7. **WO-6: Scores (Π-safe) + FREE predicate**
   symmetry-invariant `ŝ`; FREE gate. Tests: `ŝ∘U==ŝ` and constraint invariance.

8. **WO-7: Unified flow (OR-Tools)**
   build pixel-level graph with shared cell caps; solve; KKT/cons checks. Tests: synthetic quotas/faces/masks; idempotence.

9. **WO-8: Decode + Bit-meter**
   decode from incoming pixel flows; orbit-size bits; ledger totals. Tests: multiway ties produce `ceil(log2 m)`.

10. **WO-9: Size inference + packs + relax**
    size laws, packs, laminar greedy relaxation; IIS. Tests: UNSAT certificates.
    # WO-9A — Packs & Size Law
    # WO-9B — Laminar Greedy Relax + IIS

11. **WO-10: End-to-end Φ + receipts**
    glue all; receipts JSON; golden tests on a fixed ARC subset.

Each WO is small, self-contained, and testable in isolation. Claude Code won’t need to invent algorithms; it just wires adapters and calls into libraries with strict pre/post checks.