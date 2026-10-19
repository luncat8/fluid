# CFL capping, beta rescaling and pressure guidance

date: 2026-10-19
plan: diagnostics-transport follow-up #2 + 0.6.0 t4 iteration 2
depends on: diagnostics-residual-worklog.md (epsilon, sqrt beta, residual, MG prototype)

## Review of previous task

The residual prototype added `divBefore/divAfter` and a 2-level V-cycle with averaged restriction and injected prolongation; measured plain 0.007 vs MG 0.043 at N=64/K=60 and 0.046 vs 0.101 at N=128/K=60 (naive prolongation worse). HUD unified epsilon 1e-9 and tiered compression, but kept its own `ceil(N*0.18+20)` while `config.recommendedIters` used `28+N*0.15`.

Three new browser snapshots at N=512/K=60 were supplied:

- blobs ns=4/8 dt=0.00416 CFL 1.29 maxDiv 2.49 p ±3.3k maxVel 155
- convection ns=6/8 dt=0.00277 CFL 1.33 maxDiv 80.5 p -4.6k..47k maxVel 240 over 171k maxM 1.055
- convection ns=7/8 dt=0.00238 CFL 1.41 maxDiv 40.5 p -8.6k..72k maxVel 296 over 166k maxM 1.033

Findings:
- Auto max 8 is capping every fast frame (ns==8) leaving CFL >1.3 with limiter active. Fixed Steps slider also capped at 8, so user could not raise it.
- Beta sqrt scaling 2.8x at N=512 still explodes to 240-300 cells/s (vs 400 before), keeping CFL >1 even if cap were higher.
- maxDiv 40-80 at K=60 confirms Gauss-Seidel needs O(N) iters; 60 is half the required 113 at N=512.
- recommendedIters duplication hides the authoritative value; HUD suggestion 113 vs config 105 at N=512.
- 65% overpack mild (1.001-1.05) is not severe compression but tracks divergence not being removed.

## Changes

### Config / HUD

- `src/core/config.js` `substepMax: 8 -> 16`. Comment notes capping at 8 left CFL 1.29/1.33 at 6-7/8. Cost is 2x worst-case dispatches (130*16=2080 vs 1040 per frame at K=60) but only when velocity spikes; typical convection now 6-10 substeps instead of 8 capped.
- `index.html` Fixed steps slider `max 8 -> 16` to match config.
- `src/core/config.js` `prepareScenario` beta: `beta*sqrt(N/64)` cap 4 -> `beta*(1+0.18*log2(N/64))` cap 1.8. At N=512 1.54x (0.00308) vs 2.8x (0.0056), 45% lower, peak velocity ~170 vs 240-296, CFL ~0.9 at 6 substeps. At N=64 factor 1.0 preserves all convection.js baselines (still 4.4 cells/s at 2s). Log-linear justified: v ~ sqrt(g beta ΔT N), so keeping CFL flat wants beta ~1/N, milder log keeps turnover visible without explosion.
- `recommendedIters(N)` unified to `ceil(N*0.18+20)` capped 20..200. N=512 ->113 (was 105), N=64 ->32, N=1024 ->200. HUD `recommendedK()` now calls `Fluid.core.config.recommendedIters` instead of duplicating formula.
- `src/app/hud.js` healthLine: capping annotation ` (capped at 16)` when `cfl>1 && ns>=substepMax`. Helps distinguish "need more substeps" vs "need higher max".
- `src/app/controls.js` already used `cfg.substepMax` for slider max, now 16.

### Multigrid CPU prototype v2

- `src/core/multigrid.js`: kept arithmetic mean coarse mass and full-weighting restriction, replaced injected prolongation with bilinear interpolation (4 coarse -> 1 fine weighted by xf,yf). Mass dye channels also copied for completeness. Previous injection gave plain 0.007 vs MG 0.043 at N64/K60; new bilinear gives 0.007 vs 0.043 still  (N128 0.046 vs 0.101) — not yet winning but no longer catastrophically worse at K20 (19.9->15.9). Documented as baseline for 0.6.0 t4; variable coefficient on coarse still stalls without harmonic face-weight restriction. Kept disabled (`useMultigrid:false`).

### Tests

- `experiments/gpu-mock.js` expectation updated: capped substeps 8->16 (`on ns 16/16 dt 0.00104`). Validates plumbing still passes with larger max.

## Validation

- `node experiments/convection.js` PASS all 8 (stratified view 293..343, heatBand 16 at N512, rolls 4.4 cells/s at 2s, derive 350, conduction drift 2.3e-8, heat ledger 3e-7, buoyancy sign, stability 10x).
- `node experiments/hud.js` PASS (minima, context, rates, CFL/compression, float warning).
- `node experiments/gpu-mock.js` PASS 25 (pool, buoy cadence, binding counts, resize reuses pipelines, capped 16).
- `node experiments/transport-roundoff.js` PASS (f32 overshoot 1.000000119, 100k bounded, WGSL structure).
- `node experiments/sharpness.js` PASS 26->8 cells 3.25x, 0 violations, drift 6.3e-9.
- `node experiments/conformance.js` 45/50 in 120s (still/stirred/dam/substep/buoy/pressure coloring/wall/emitter all PASS; emitter/draw starved by time, not failed; fixed clamp now 100->16 as expected). Full run needs ~180s, same as previous 44/50 baseline.
- `experiments/logs/2026-10-19-gpu-parity.log` unchanged (<2e-6 aggregate E/T).

CPU spot: N512 convection with new beta predicts maxVel ~170 (was 240-296), CFL 0.9 at ns6 (was 1.33 capped), maxDiv ~2-3 at K113 vs 40-80 at K60; blobs CFL 0.8 at ns4.

## Next step

- Browser pass at N=512: compare (a) ns 6/8 fixed K60 beta-old vs new (CFL, maxVel, maxDiv, p range), (b) auto 16 capped vs uncapped, (c) K=60 vs K=113 at same ns. Expect CFL <1.0, maxVel ~170, maxDiv <3, p range halved, over 171k->~40k.
- Implement GPU timestamp queries (0.6.0 t1) to separate encode vs queue; add `gpu-bench.html` with pinned substep policy and ns histogram.
- If MG still stalls, try harmonic-weighted restriction (face w averaging) or keep RB and auto-raise K to recommendedIters on resize (hysteresis 2s). Do not enable MG until residual <0.5*plain at equal wall time.
