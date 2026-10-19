# Diagnostics thresholds, convection scaling and pressure residual

date: 2026-10-19
plan: diagnostics-transport follow-up + 0.6.0 t3/t4 prototype
depends on: diagnostics-transport-worklog.md (transport roundoff), 1.0.0 convection scaling

## Review of previous task

The transport roundoff fix reserved a 1e-6 outflow margin and used nonnegative-weight donor updates; sharpness stayed 3.25× (26→8 cells) and conservation <1e-6. The new HUD showed exact minima and CFL, and the queue is bounded to one frame. Four GPU snapshots (blobs at N=512/K=60, 4 and 6 fixed substeps; convection at N=512/K=60, 6 fixed and 8 auto) were supplied:

- blobs: drift ok, min dye ≈−3e-28, maxM 0.61→0.43, neg 42k→47k, maxDiv 3.79→0.21, maxVel 160→95, CFL 1.34→0.52
- convection: maxM 1.01→1.04, over 99k→224k, maxDiv 23→10, p ±34k→+179k, maxVel 185→442, CFL 1.03→1.84, E drift 3e-6 warn

 Findings:
 - Tiny dye negatives (−3e-29 to −3e-28) are 19 orders below the −1e-9 meaningful threshold; they are f32/FCT roundoff, not a monotone failure, but were counted as cntNeg 42k and triggered the red NEGATIVE warning on every frame.
 - over=1.001 is correct for liquid, but at N=512 38–85 % of cells sit at 1.001–1.01 (maxM only 1.01–1.04): mild widespread overpack, not the severe 30.7 case of the previous session. The HUD only warned on maxM>2, so it hid the mild case.
 - maxDiv 10–23 and p ±179k at N=512 convection show the projection not converged at K=60. CFL >1 with 8 auto substeps confirms the limiter is active. Velocities 400+ cells/s suggest beta linear in N is too aggressive.
 - The prior maxM=30.7 compression is now 0.4–1.04, but the underlying pressure/beta issues remain.

## Changes

### Diagnostics (CPU + WGSL)

- `src/core/stats.js` totalsFromFields: cntNeg now counts only values < −1e-9 (dye and mass/E share the threshold; dye sees −3e-28 as 0). WGSL `statsReduceA` matches the CPU threshold. min values are still exact for the snapshot.
- `src/app/hud.js` healthLine: epsilon-filtered NEGATIVE, tiered compression (mild `over>0 && maxM≤1.05`, moderate `1.05<maxM≤1.5`, severe `>1.5`), `recommendedK = ceil(N*0.18+20)` suggestion when maxDiv>1, CFL wording “try more substeps”, and residual display when present. Perf line unchanged.
- `src/gpu/shaders.js` conduction/stats: negative test aligned to −1e-9.

### Thermal scaling

- `src/core/config.js` prepareScenario: convection beta now `beta*min(4, sqrt(N/64))` instead of `beta*N/64`. At N=512 beta goes 0.016→0.0056 (2.8× vs 8×), turnover still faster than fixed beta but velocities drop from ~440 to the expected 150–200 range in the next browser pass. heatBand and T view range unchanged. Added `recommendedIters(N)` helper for the HUD and for 0.6.0 measurements.

### Pressure residual

- `src/core/solver.js` poissonRhs now stores `divBefore = max|b|` (RHS before solve); step stores `divAfter = maxDiv` after applyGrad and `residual = divAfter/divBefore`. stats() exposes `diag {divBefore,divAfter,residual}`. GPU path still reports maxDiv; diag is CPU-only until a GPU readback is added (0.6.0 t1). Lets the user see “div 23 → 0.2” and whether more K helps, instead of guessing from maxM.

### Multigrid prototype (CPU)

- New `src/core/multigrid.js`: 2-level V-cycle (2 pre, coarse = K, 2 post), harmonic-mean face weights, residual `r = b − A p`, averaged restriction, injected prolongation. Created lazily in solver when `cfg.useMultigrid` (default false, off for all tests). Measures:
  - N=64 K=20 plain 0.070 residual vs MG 0.078 (MG slightly worse, same maxM)
  - N=64 K=60 plain 0.007 vs MG 0.043
  - N=128 K=60 plain 0.046 vs MG 0.101
  Naive injection/averaging with variable density does not beat plain RB at equal iteration budget; coefficient-aware restriction and better prolongation are needed. Kept disabled, documented as the 0.6.0 t4 baseline.

### File wiring

- `index.html` loads `multigrid.js` before `solver.js`.
- `src/core/config.js` adds `useMultigrid:false` and exports `recommendedIters`.
- `src/gpu/solver.js`, `src/app/main.js` untouched except for HUD wording.

## Validation

- `node experiments/hud.js` — PASS (minima, rate labels, configuration context, CFL/negative/compression warnings).
- `node experiments/sharpness.js` — PASS (26→8 cells, 3.25×, 0 violations).
- `node experiments/transport-roundoff.js` — PASS (f32 overshoot reproduced, 100k bounded, WGSL structure ok).
- `node experiments/convection.js` — 12 checks PASS (onset, buoyancy, closure, T range; heatBand=16 at N=512 still correct after beta change).
- `node experiments/gpu-mock.js` — 25 checks PASS (pool swap keeps pipelines, buoy/stats plumbing, param indices incl. new slots).
- `node experiments/ui.js` — 22 checks PASS; `controls.js` 23 PASS; `performance.js` PASS; `gpu-mock` buff/tex counts ok.
- `node experiments/conformance.js` — 44 of 50 checks in 120 s (wall integrity, still, stirred, dam, substep policy, buoy spawn/model all PASS; emitter/draw were starved by timeout, not failed). Isolated draw stroke with paint+stir at N=64 survived with cntNeg 0, maxM 2.86 and drift <1e-6, matching the worklog claim.
- `experiments/logs/2026-10-19-gpu-parity.log` unchanged — aggregate E/T <2e-6 still holds; pointwise drift separate.
- Multigrid isolated test above — MG disabled by default, so no regression when off; when on, residual worse as recorded.

## Next step

- Browser pass at N=512 convection with the new beta and HUD: record sim/s, maxDiv, p range, maxVel, CFL and residual suggestion at K=60 and K=recommendedIters (107 at N=512). Confirm velocities ~150–200 instead of 400 and over count drops.
- Implement GPU timestamp queries (0.6.0 t1) and a CPU `gpu-bench.html`/`cpu-bench.js` baseline at 4 fixed substeps vs 8 auto, with ns histogram and readback latency columns.
- Replace naive MG restriction with harmonic-mean coefficient restriction and bilinear prolongation, then re-measure residual vs RB time; if it stalls on variable density, keep RB and raise K per the recommended curve.
- Consider counting overpack severity as a histogram (1.001/1.01/1.1) rather than a single count, to avoid the 99k/224k mild-vs-severe confusion in the next HUD iteration.
