# convection demo — flat temperature map (worklog)

report: at N=512, frame ~2521, the temperature map showed no change, with all
conservation drifts green.

## diagnosis

three stacked causes, the first a real physics bug:

1. buoyancy sign inverted (bug). `cellAx/cellAy` (cpu) and `cellAccelX/Y`
   (wgsl) computed the thermal term as `-g * beta * (T - TRef)`. the density
   term in the same formula proves the convention: heavier-than-mean
   accelerates toward -y (down), so a hot density *deficit* must accelerate
   toward +y. with the minus sign hot fluid sank into the bottom wall and
   cold fluid rose into the top wall — the heated box was stably pinned
   and could never convect. the old sign check in convection.js measured
   +0.004 mean rise, i.e. it passed on noise. flipped both terms in both
   solvers; the check now measures +3.18 over the hot layer.
2. nothing to see in the default scenarios: blobs/dam/still start at uniform
   T_amb, so a flat temperature map there is correct, not a bug.
3. the convection scenario could not start on demo timescales: uniform
   start, heater only in the bottom 2 cells, conduction at 0.2 spreads
   ~1 cell in 10 s, and view mode 1 spanned 273..373 K. measured on the cpu
   oracle: after 10 s only the bottom rows were off ambient, max|v| ~7.

## fix

- sign flip above (cpu solver + wgsl forces).
- convection scenario starts unstably stratified, `convBotT=343` K at the
  bottom to `convTopT=293` K (= T_amb, the cold-plate target) at the top,
  plus the existing sinusoidal roll seed (`buildScenario`).
- `config.prepareScenario`, called by both solvers on reset, is the single
  place that sets per-scenario thermal setup: heater/cold-plate band scales
  as max(2, N/32) (16 cells at 512), temperature view range matches the
  stratification (293..343 instead of 273..373), beta scales as
  beta * max(1, N/64) (velocities are cells/s, so a fixed beta makes a 512
  box turn over 8x slower on screen than a 64 box).
- heater thermostat: new param slot 44 `heatTarget` (= convBotT in the
  convection scenario, 0 = uncapped elsewhere). the fixed-power band
  otherwise ran away to a measured 467 K in 20 s, because conduction away
  from it is ~100x slower than the input. cpu and wgsl heatSources both
  honour it; only energy actually added is ledger-counted, as before.
- selecting convection in the scenario dropdown or with `n` switches to
  the temperature view (through the view menu's single writer).
- regression tests in experiments/convection.js (`testScenarioPresentation`):
  the shipped demo at N=512 starts stratified over most of its view range,
  the band scales with N, rolls are underway within 2 s, and the hot-layer
  rise check uses a TRef+35 threshold (with the stratified start a low
  threshold averages the sinking return flow into the "hot" mean).

## validation

- `node experiments/convection.js` — ALL PASSED
  (experiments/logs/2026-10-19-convection-demo.log): onset 1.17 s,
  peak |v| 36.9 at N=64 (was ~5.6 with the inverted sign), hot-layer mean
  rise +3.18, T capped at 343, m/E drift 9.5e-6 / 1.0e-8 over 5000 steps.
- spot check at N=256 with the scaled scenario beta/band: after 5 s
  max|v| 94 cells/s and mean |T - initial profile| 11.2 K — the field is
  visibly overturning, not creeping.
- conformance, sharpness, buoy, gpu-mock, ui, input, controls — all pass.
  gpu-mock lints the wgsl param indices, so slot 44 is checked there;
  numeric gpu parity for the flipped forces term remains a browser/dawn
  run (same one-line formula in both implementations).

## next step suggestion

run `experiments/gpu-parity.mjs` on a real WebGPU backend and eyeball the
convection scenario at 512 in the browser; if parity holds, consider a
temperature legend for view mode 1 using `Fluid.gpu.view.tempToRgb`.
