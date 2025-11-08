now let's expand this wo. so now tell me what kinds of "mature, well-documented Python libraries for every primitive we need" can be reused for

**WO-2: Embedding & Period tests**
   topleft/center embed; byte-exact period detection. Tests: round-trip.

now here are the things u must take care of:
1. now thr is nothing called underspecificy in development. instead  we shud be ovespecific. but anything that u r specifying must be STRICTLY grounded in our anchor docs we created.
2. do tell in WO which anchor docs they shud refer to before proceeding if any..
3. Plus we must be explicit about wht packges' functions and libs to use here so that claude doenst reinvent the wheel.  we dont want to implement any algo. 
4. then we must hv receipts as first class citizens part of WO with clear understanding as to how they capture and ensure that our implementation matches math spec 1:1. 
5. make sure u include runner/harness changes that are needed to accomodate this wo and how can reviewer use it to test on real arc agi tests 
6. a clear instruction to reviewer that tells them how to use runner for all 1000 arc agi tasks, how to use receipts to knw that math implementation is right.. and how to identtify a legit implementation gap or unsatisifable task if any in WO. this instruction MUST ensure that math and implementation match 100% 
7. we do not get trapped in optimization fixes that enforces simplified implemtnatinns unless we are really looking at huge compute times (ps we will do on CPU)
8. Incorproate applicable "hard improvements", "Adapters that keep fidelity", "Library choices" that we disucssed in the WO to ensure we dont miss anything

so can u create an expanded WO accommodating all of what i said above with a small checklist at the end of WO to show how u took care of each of these?
============


















now let's expand this wo. so now tell me what kinds of "mature, well-documented Python libraries for every primitive we need" can be reused for
## WO-3G — FREE Verifier: **Component-wise transport (per-component pose, types only)**

**Scope (types only)**
Per pair, detect **connected components** in (T_X) and prove that each component is carried to a corresponding set of components in (T_Y) via a **single global pose** (composition of a D4 and a translation).

1. Label components in (T_X) and (T_Y) with **4-connectivity** using `skimage.measure.label` (documented; `connectivity=1`) ([scikit-image.org][4]).
2. For each labeled component in (T_X), extract its type mask and bounding box; apply each D4 pose and integer translations (`np.rot90`, `np.roll`) and verify exact equality to **some** union of labeled components in (T_Y) (types only) ([NumPy][1]).
3. Accept if a **single pose** explains all components (same D4 and same translation per component relative to its bbox origin).

**Libs & calls**
`skimage.measure.label` (4-conn) ([scikit-image.org][4]), `numpy.rot90` ([NumPy][1]), `numpy.roll` ([NumPy][2]), `numpy.array_equal`.

**IO**
`verify_component_transport(T_X, T_Y) -> Optional(("component_transport", (d4, dr, dc)))`

**Receipts**

* Component counts in (T_X, T_Y); per component: bbox, applied pose, boolean match.
* `components_covered = 100%` and `single_pose=True`.

**Pass**
No false positives; returns one terminal if every (X) component is explained by the same pose.

**Runner**
WO-4 consumes this terminal at **lower priority** than `identity` if you keep the frozen order or **insert** after D4/translate to reduce “identity” dominance (you can append to the order without regressing v0).

now here are the things u must take care of:
1. now thr is nothing called underspecificy in dev. instead  we shud be ovespecific. but anything that u r specifying must be STRICTLY grounded in our anchor docs we created.
2. do tell in WO which anchor docs they shud refer to before proceeding if any..
3. Plus we must be explicit about wht packges' functions and libs to use here so that claude doenst reinvent the wheel.  we dont want to implement any algo. 
4. then we must hv receipts as first class citizens part of WO with clear understanding as to how they capture and ensure that our implementation matches math spec 1:1.  pls ensure that the receipts flow from this WO-03xx wo to WO-04's file and eventually to solve.py to keep our testing and debugging pipeline tight and intact 
5. make sure u include runner changes that are needed to accomodate this wo and how can reviewer use it to test on real arc agi tests 
6. a clear instruction to reviewer that tells them how to use runner for all 1000 arc agi tasks, how to use receipts to knw that math implementation is right.. and how to identtify a legit implementation gap or unsatisifable task if any in WO. this instruction MUST ensure that math and implementation match 100% 
7. we do not get trapped in optimization fixes that enforces simplified implemtnatinns unless we are really looking at huge compute times (ps we will do on CPU)
8. Ensure that u add explicit wiring instruction to WO-04 so that we can test immediately if it increases some matches

so can u create an expanded WO accommodating all of what i said above with a small checklist at the end of WO to show how u took care of each of these?

=======

now let's expand this wo. so now tell me what kinds of "mature, well-documented Python libraries for every primitive we need" can be reused for
## WO-8 — v0 Runner (Batch + Audit)

**Goal:** End-to-end solve on all tasks we can prove FREE for; write predictions and receipts.

* **Scope:** `solve_batch` that: Π on Y₀ → K; FREE proof → transport → fill; write outputs JSON and receipts JSONL; `audit <id>` CLI to dump receipts for one task.
* **Libs:** `numpy`, `typer/argparse`, `hashlib`
* **IO:** `solve(challenges.json) -> predictions.json + receipts.jsonl`
* **Receipts:** corpus summary (counts by terminal, FREE_PROVEN vs UNPROVEN), frozen simplicity order echo.
* **Pass criteria:** **600+ tasks** attemptable now; outputs + receipts produced deterministically; no crashes on full corpus.

now here are the things u must take care of:
1. now thr is nothing called underspecificy in dev. instead  we shud be ovespecific. but anything that u r specifying must be STRICTLY grounded in our anchor docs we created.
2. do tell in WO which anchor docs they shud refer to before proceeding if any..
3. Plus we must be explicit about wht packges' functions and libs to use here so that claude doenst reinvent the wheel.  we dont want to implement any algo. 
4. then we must hv receipts as first class citizens part of WO with clear understanding as to how they capture and ensure that our implementation matches math spec 1:1.
5. make sure u include runner changes that are needed to accomodate this wo and how can reviewer use it to test on real arc agi tests 
6. a clear instruction to reviewer that tells them how to use runner for all 1000 arc agi tasks, how to use receipts to knw that math implementation is right.. and how to identtify a legit implementation gap or unsatisifable task if any in WO. this instruction MUST ensure that math and implementation match 100% 
7. we do not get trapped in optimization fixes that enforces simplified implemtnatinns unless we are really looking at huge compute times (ps we will do on CPU)

so can u create an expanded WO accommodating all of what i said above with a small checklist at the end of WO to show how u took care of each of these?