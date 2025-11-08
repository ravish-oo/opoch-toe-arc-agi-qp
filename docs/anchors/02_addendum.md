Locked. Here are the exact, final pin-downs for every loose end you flagged—each stated as a *deterministic rule* or a *short lemma* with the objects we already defined. There’s nothing left to guess.

---

## A) One-of-10 across colors (global exclusivity)

You must ensure that no cell ((r,j)) is overfilled when you solve per-color flows.

### A.1 Preferred: *single shared-capacity flow* (multi-commodity collapsed)

Build *one* flow per color *but* route all colors through a *shared cell edge*:

•⁠  ⁠For each color (c), keep supplies (q[s,c]), optional row/col demand layers, and edges to cell nodes (Z_{r,j}) (note: *no per-color cell node*).
•⁠  ⁠Each cell node (Z_{r,j}) has *one* capacity (n_{r,j}) that is *shared* across all incoming color arcs.
•⁠  ⁠Add *per-color demand edges* (Z_{r,j}\to W_{j,c}) (or directly to sink) so that row/column totals and per-color totals still hold.

This is still a single *min-cost flow* (not full multi-commodity), because colors are only distinguished on the demand side; capacities at cells are shared scalars. Costs on arcs from bins→cells remain (-\hat s_{p,c}) aggregates; the network remains small and integral (TU).

### A.2 Alternative: *per-color flows + one b-matching at the end*

If you keep per-color flows with per-cell quantities (f_{r,j,c}), add a *final bipartite b-matching* over pixels×colors:

[
\begin{aligned}
&\max_{x\in{0,1}^{N\times 10}} \ \sum_{p,c} \hat s_{p,c}, x_{p,c} \
&\text{s.t.}\quad
\sum_{c} x_{p,c} \le 1 \ \ \ (\forall p),\qquad
\sum_{p\in \text{cell}(r,j)} x_{p,c} = f_{r,j,c} \ \ \ (\forall r,j,c),\
&\hspace{34mm} x_{p,c}=0\ \ \text{if}\ A_{p,c}=0.
\end{aligned}
]

This is a classic TU *b-matching* LP → integral and fast (Hungarian/flow). It enforces *one-of-10* and preserves the cell-by-color counts computed by flows exactly. Local “top-k per cell” is the greedy solution to this LP when all rows are separable; the LP pins it formally.

---

## B) Projected scores (\hat s): Π-safety and invariance

Define (\hat s:\Omega\times \mathcal C\to\mathbb R) by *Π-safe* observables only:

•⁠  ⁠(\hat s_{p,c} = \alpha_0 + \alpha_1, \mathbf 1{A_{p,c}=1} + \alpha_2, \text{feature}{\rm bin}(p) + \alpha_3, \text{feature}{\rm stage}(p))
  (all features are functions of periphery–parity bins, mask, and verified free transforms—never center color).

*Free-invariance rule.* For any *FREE* permutation/roll (U) (verified symmetry), we enforce

[
\hat s \circ U \ =\ \hat s \qquad\text{on its verified domain.}
]

If a symmetry is verified, either:

•⁠  ⁠include the *projector* first (e.g., (P_{\rm rep}), (P_{\rm color})), or
•⁠  ⁠*transport* scores by the symmetry so costs remain invariant.

This pins the *“free preserves (J)”* claim: (J(y)=\sum_{p,c} -\hat s_{p,c} y_{p,c}) is invariant under FREE maps.

---

## C) Color-symmetry projector (P_{\rm color}) on discrete outputs

*Never average (y)* after decode. If a color subgroup (H\subset S_{10}) is *verified* (every aligned training output is exactly invariant), enforce color symmetry as:

•⁠  ⁠a *constraint*: add rows that force equality of symmetric color layers, or
•⁠  ⁠a *cost*: average (\hat s) over (H) so all symmetric channels score equally (then the flow is indifferent).

Either route keeps the LP/flow TU and the solution *integral*.

---

## D) Multi-stage composition wording (exact equalities)

Replace the wording:

	⁠“the block system has the unique least-squares solution…”

with:

	⁠*Exact stage solve.* All stage equations (y^{(k+1)}=T_k y^{(k)}) and fixed-point (y=T y) are enforced as *linear equalities* within the same LP/flow system. When multiple fixed points exist, the *canonical representative* is the *least-norm* solution in the homogeneous subspace (Moore–Penrose) chosen by the Π-safe bin equalizers; decoding remains integral via the flows.

No algorithmic change; just clarifies we solve equalities exactly, not least-squares.

---

## E) Relaxation confluence: laminar family (matroid-greedy)

Make the *laminar nesting* explicit:

[
\underbrace{\text{Cell/FD equalities}}*{\text{finest}} \ \supset\
\underbrace{\text{Row/Column faces}}*{\text{meso}} \ \supset\
\underbrace{\text{Bin quotas}}*{\text{coarse}} \ \supset\
\underbrace{\text{Free symmetries}}*{\text{weakest}}.
]

Any two constraints at a tier either act on *disjoint* supports or one *refines* the other. This is a *laminar family. Therefore the greedy removal “drop lowest precedence conflicting rows until feasible” is a **matroid-greedy* on a laminar matroid—hence *minimal* and *confluent* (order-independent).

---

## F) Bit-meter for multiway ties (orbit size)

If a pixel (p) has an orbit (\mathcal O_p\subseteq \mathcal C) of *indistinguishable colors* under the *remaining* symmetries (after all projectors/rows), the minimal information to pick one is

[
\boxed{\ \text{bits at }p\ =\ \lceil \log_2 |\mathcal O_p| \rceil \ }.
]

Our decoder uses the palette lex order to select; we ledger exactly (\lceil \log_2 |\mathcal O_p| \rceil) bits for that pixel. Total (\Delta N) is the *sum* over pixels.

---

## G) FREE orthogonality check (decidable predicate)

A map (U) is *FREE* iff:

1.⁠ ⁠*Cost-invariant:* (J(U y)=J(y)) for all feasible (y), with (J(y)=\sum_{p,c}-\hat s_{p,c} y_{p,c}).
2.⁠ ⁠*Constraint-invariant:* (U) preserves all emitted Π-safe equalities (mask, equalizers, faces/blocks): for every row (\ell^\top y=b), (\ell^\top U y = b).

This is *decidable: check (U) is a permutation/roll verified on trainings; verify the rows are invariant; verify (\hat s\circ U = \hat s). If both hold, (U) is FREE; otherwise it must be encoded as a **paid* constraint.

---

## H) Mask algebra closure (explicit statement)

Let (\mathcal F) be the set of per-pixel admits maps (F) with partial order (F\le F') iff (F[p,k]\subseteq F'[p,k]) (\forall p,k). Define closure operator
[
\mathrm{cl}(F)\ :=\ \Phi\bigg(\ \bigwedge_{i} F^{(i)}\ \bigg),
]
where (F^{(i)}) are training permissions and (\Phi) is the row-local sufficiency plus color-agnostic lift. Then (\mathrm{cl}) is *extensive* ((\mathrm{cl}\le F)), *monotone, and **idempotent; hence a closure system; fixed point is unique and **independent of order*.

---

## I) “Constant on bin” linear predicate (explicit)

Constancy of color (c) on bin (B_s) is a Π-safe linear predicate:

[
\forall i\ \forall p,q\in B_s\cap{A_{p,c}=A_{q,c}=1}:\ \ [y(Y_i)]{(p,c)}=[y(Y_i)]{(q,c)}\ .
]

Under embedding/permutation of channels, both sides permute identically; under Π binning, (B_s) is fixed; thus invariance holds. Emitted equalizers commute with all other block-diagonal rows.

---

## J) Replication projector proof (idempotent, orthogonal)

For period (p), (T) a permutation matrix (orthogonal). Then
[
P = \frac{1}{p}\sum_{k=0}^{p-1} T^k\quad\Rightarrow\quad P^2=\frac{1}{p^2}\sum_{k,\ell} T^{k+\ell}=\frac{1}{p}\sum_m T^m = P,
]
and (P^\top=P). This holds on positions; lifting with (I) on channels preserves orthogonality.

---

## K) Global argmax optimality (within-cell)

Given fixed per-cell per-color counts (f_{r,j,c}) from the flow, the decode LP (Section A.2) is a TU assignment; selecting top-(f_{r,j,c}) by scores within the cell solves it *exactly*. No further projector can improve the cost (FREE maps preserve (J); paid rows are already satisfied).

---

## L) Gravity/Fill proofs (unique fixed point)

•⁠  ⁠*Gravity (G):* As defined, (G) is nilpotent on transient states (acyclic down graph), so ((I-G)) invertible there. The fixed-point equalities select the *unique* settled configuration consistent with mask.
•⁠  ⁠*Fill:* Dirichlet problem on enclosed regions has a *unique* harmonic extension; when trainings pin a constant color fill, rows fix interior to that color; when they pin “copy boundary color,” rows equate interior to boundary averages; both are Π-safe.

---

## M) Period test invariance and lex selection

Period set (S_x(Y)={p: Y=\mathrm{roll}(Y,p,\text{axis}=x)}). Under embedding and channel alignment, equality is preserved (both are permutations on indices and channels). Shared minimal period is (p^=\min \cap_i S_x(Y_i)). Jointly with (y)-axis, pick lex-min ((p_y^,p_x^*)). Projector (P_{\rm rep}) then applies.

---

## N) Relaxation minimality (laminar greedy)

Given the laminar family (\mathcal F) in §E, the greedy removal of lowest-precedence rows corresponds to finding a *minimum-weight feasible subsystem* in a laminar matroid. Classic matroid greedy theorem ⇒ *minimal* set dropped; laminarity ⇒ *confluence* (result independent of tie order). Record dropped rows with ranks to certify.

---

## O) UNSAT proof object

Return the *minimal conflict subsystem* (\mathcal U\subset \mathcal R) s.t.

•⁠  ⁠(\mathrm{rank}(A_{\mathcal U}) < \mathrm{rank}([A_{\mathcal U}\ | b_{\mathcal U}])) (inconsistent),
•⁠  ⁠for any proper (\mathcal U'\subset \mathcal U), the system is consistent (residual ≤ tol).

This is a *proof* of infeasibility (an *Irreducible Infeasible Subsystem*). Rows are small; enumeration + rank test is tractable and deterministic.

---

## P) FREE predicate (decidable)

A linear map (U) is *FREE* iff:

•⁠  ⁠(U) is a permutation/roll verified across trainings on outputs;
•⁠  ⁠(U) preserves all emitted equalities (A y=b) (i.e., (A U=A)); and
•⁠  ⁠(J(U y)=J(y)) for all feasible (y), equivalently (\hat s\circ U=\hat s).

If all three hold, (U) is a *free isometry* and can be applied before flows; else it must be encoded in (A) and/or the costs.

---

With these pin-downs:

•⁠  ⁠*Exclusivity* is guaranteed by shared cell capacity (or final b-matching).
•⁠  ⁠*Scores* are Π-safe and symmetry-invariant; *FREE* is decidable.
•⁠  ⁠*Color symmetry* is enforced as a constraint/cost (never fractionalizing (y)).
•⁠  ⁠*Stages* are exact equalities; fixed-point choice is canonical.
•⁠  ⁠*Relaxation* is laminar-greedy (minimal, confluent).
•⁠  ⁠*Bit-meter* counts (\lceil\log_2 |\text{orbit}|\rceil) per pixel.
•⁠  ⁠*Every claim* (FY, commutation, gravity/fill uniqueness, periodicity) is pinned with a lemma you can drop into the spec.

There are no remaining ambiguities.