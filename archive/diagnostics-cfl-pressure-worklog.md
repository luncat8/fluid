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

## 0.6.0 t4 iteration 3 (2026-10-19): multigrid on the GPU, 2x substep target

### Browser baseline (previous build, N=512, K=60, supplied snapshots)

| scenario | ns | CFL bound | maxDiv | p range | maxVel | maxM / over | sim/s |
|---|---|---|---|---|---|---|---|
| blobs f331 | 4 fixed | 1.188 | 3.36 | -4619..1757 | 142.6 | 0.655 / 0 | 53.4 |
| blobs f451 | 3/16 auto | 1.439 | 2.29 | -4219..3342 | 129.5 | 0.655 / 0 | 60.0 |
| blobs + drawn walls f1381 | 5/16 auto | 1.597 | 259.6 | -4089..123022 | 239.6 | 1.493 / 890 | 33.6 |
| dam | 4 fixed | 2.933 | 109.7 | -9452..36157 | 352.0 | 20.49 / 12990 | 57.4 |
| still | 4 fixed | 0 | 0 | 0..0 | 0 | 1.000 / 0 | 59.3 |
| convection f541 | 4 fixed | 0.397 | 2.34 | -7928..8109 | 47.6 | 1.004 / 96997 | 56.0 |
| convection f1231 | 5/16 auto | 1.351 | 2.96 | -5401..26894 | 202.7 | 1.024 / 139638 | 62.0 |

### What the numbers said

- The auto policy targeted `maxVel*dt = cflTarget` (the per-face flux), but the
  donor-cell update is monotone only while the four *outgoing* fluxes sum to 1.
  For a solenoidal field that sum is at most `2*maxVel*dt`, which is exactly what
  the HUD prints. So auto chose ns=3 at 129 cells/s and handed the run to the
  positivity limiter with the HUD reporting 1.44. Clipping is a local mass sink,
  not just dissipation: it is why `maxM` ran away in dam and the draw test.
- `vMax/frameDt` = 480 cells/s was 2-4x faster than the frame could transport.
  A fixed ns=4 at 352 cells/s needs 4.3 faces of outflow per cell per step.
- 60 red-black sweeps cannot remove the smooth pressure error at N=512: the
  slowest closed-box mode decays by `1-O(1/N^2)` per sweep, so 60 sweeps leave
  ~99.6% of it. maxDiv 110 at dam was the projection floor, not an advection bug.
- `q' = q*(1-outflow) + in` makes every unit of unresolved convergence compress
  mass. The fix has to reduce the divergence, not the transport.

### Changes

- `src/core/config.js`: `substepsAuto:true`, `substepMax:24`, and
  `substepCount = ceil(2*maxVel*frameDt/cflTarget)`. At 200 cells/s this is 9
  substeps (CFL bound 0.8) instead of 5 (bound 1.6).
- new `velocityCap(cfg,ns) = min(vMax, 0.45*ns)/frameDt`; main.js sets
  `cfg.vClamp` every frame, hud.js prints `vCap=` and flags the capped case.
  The cap rises with the substep budget, so a burst keeps its physical speed
  only if the budget can actually transport it.
- CPU MG v3 (`src/core/multigrid.js`): stride `S` from `cfg.mgCoarsen` (clamped
  while `floor(N/S) < 4`), in-place red-black GS on both levels, **sum** of the
  S^2 fine residuals as the coarse RHS, bilinear prolongation, `mgCycles`.
  Defaults `mgCoarsen:4`, `mgPre:4`, `mgPost:8`, `mgCycles:2`.
- GPU MG (0.6.0 t4): `mgCoarsen`, `mgRestrict`, `mgZero`, `mgProlong` kernels,
  coarse pool at `(N/S+2)^2` (mass `rgba32float` so `.a` is real), coarse
  pipelines/bind groups and an MG branch in `passes.encodeStep` that repeats
  pre/restrict/coarse/prolong/post `mgCycles` times. `mgS` travels in params
  slot 94 and is written from `pool.mgS`, so the shader stride always matches
  the allocated textures.
  - fixed: `mgZero` is bound to the *coarse* params, where `values[2]` is
    already `Nc+2`; dividing by 2 again zeroed only the top-left quarter of the
    coarse pressure and left stale corrections in the rest.
  - fixed: the three coarse kernels hardcoded stride 2. At S=4 they would have
    covered a quarter of the domain.
- `index.html`: fixed-step slider max 16 -> 24.
- A/B escape hatches: `?mg=0` (plain RB), `?s=2|8`, `?cycles=1`, `?pre=`,
  `?post=`; the HUD stats line now prints `mgS4x2(pre4/post8)`.
- `experiments/multigrid.js` new: plain vs MG on dam and blobs at N=128 plus a
  one-cycle residual monotonicity check.

### Validation

- `node /tmp/naga/check.js`: 25/25 WGSL sources parse and validate (naga-wasm
  30.2.0; scalar `array<f32,128>` uniforms are rejected by naga, so the check
  rewrites the params struct to `array<vec4<f32>,32>` for validation only).
- `node experiments/gpu-mock.js`: 27/27. Two new checks encode two substeps and
  assert `mgCoarsen` once per substep, `mgRestrict`/`mgProlong` `substeps*cycles`
  times, the coarse params carry `mgTexW`, and `pool.mgNc = N/S`.
- `node experiments/conformance.js`: **54/54 PASS** (log in
  `experiments/logs/2026-10-19-conformance.log`). `substepCount`, emitter and
  draw expectations were updated for the new policy; the draw/limiter split now
  tests the shipped cap (no clipping, raw k_out 0.837) and the safety limiter
  separately (forced 4x stir, raw k_out 2.0 -> stored 1.0, no negatives).
- `node experiments/multigrid.js` (new, N=128, 60 frames, 4 fixed substeps):

  | case | plain worstDiv | mg worstDiv | plain over | mg over | plain ms | mg ms |
  |---|---|---|---|---|---|---|
  | dam | 70.44 | 0.314 | 736 | 0 | 16056 | 13228 |
  | blobs | 7.33 | 0.048 | 0 | 0 | 25055 | 13296 |

  both maxM 1.0044 -> 1.0000 (dam); one V-cycle from a live state lowers the
  fine residual 19.03 -> 0.336.
- CPU N=128 dam, auto policy, 60 frames (`maxDiv`, `over`, wall):

  | solver | maxDiv | over | maxM | ms |
  |---|---|---|---|---|
  | plain K=60 | 20.14 | 741 | 1.004 | 23.7 |
  | S=2, 3 cycles | 0.19 | 0 | 1.000 | 35.1 |
  | S=4, 2 cycles | 0.05 | 0 | 1.000 | 19.4 |
  | S=4, 3 cycles | 0.01 | 0 | 1.000 | 24.0 |

  S=2 is the wrong shape at large N: its coarse grid (N/2) still has the same
  long-wavelength problem and needs three cycles to barely beat S=4/2.
- One projection from a live dam state at **N=512** (max |A p - b|, ms):

  | solver | maxRes | rms | ms |
  |---|---|---|---|
  | plain K=60 | 135 | 9.72 | 1815 |
  | plain K=200 | 74.6 | 7.04 | 6455 |
  | S=2, 3 cycles | 37.4 | 4.74 | 2246 |
  | S=4, 2 cycles | 22.8 | 3.45 | 960 |
  | S=4, 3 cycles | 17.1 | 2.95 | 1480 |

- CPU **N=256 dam, 240 frames (4 s), auto policy, S=4/2**: maxM 1.000, over 0,
  maxDiv 0.03 at f=239 (peak 5.7 on the cold first frame), maxVel 108,
  p -3.4k..11.1k. The previous build at the same size/time: maxM 3.9,
  maxDiv 190, p -7.9k..30.5k.
- CPU **N=512 dam, auto policy, S=4/2, 40 frames (0.65 s)**: maxM 1.000 and
  over 0 at every sample, maxDiv 9.4 cold -> 1.8 f10 -> 0.35 f20 -> 0.44 f30 ->
  9.4 f39 (a splash frame), maxVel <= 113, p -6.3k..30.4k. The old browser dam
  reached maxM 20.5 / maxDiv 110 by frame 241: the compression runaway is gone.

### Next

- Browser A/B at N=512: default (MG S=4 x2) vs `?mg=0` at the same scenario and
  duration. Expect CFL bound <=0.8 in auto, maxDiv order 1, over 0, p range
  about 3x smaller, and sim/s at least plain. If MG regresses on the device,
  `?mg=0` is the fallback and the `mgZero` extent is the first thing to check.
- GPU timestamps (0.6.0 t1) + `gpu-bench.html` pinned policy: measure the MG
  encode/dispatch cost per frame at N=512 before tuning S or cycles further.
- If the coarse solve is the bottleneck in the timings, try K_coarse 30 (the
  N=512 residual at S=4/2 is 33 vs 22.8 at 60, at 0.57x the cost).
