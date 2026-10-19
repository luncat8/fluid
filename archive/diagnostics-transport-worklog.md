# Throughput, control semantics and transport roundoff

## Review and diagnosis

Reviewed the convection work and current GPU dispatch chain. FCT is wired to
`encodeStep`: enabled adds limiterPrepare/limiterApply; disabled uses donor mass.
It only changes RGB tracers. Gravity remains a uniform parameter in the forces
kernel, and the buoy control owns a read-only dam marker, not thermal buoyancy.
Neither toggle removes the dominant pressure solve (2*K dispatches/substep).
At K=60, FCT adds two of 133 step dispatches; five versus six substeps adds 20%
more solver work. Actual browser GPU timing is required to explain a hardware
performance cliff; dispatch counts alone do not establish its cause.

The old FPS EMA averaged reciprocals of rAF intervals, counting callbacks that
submitted no simulation. Main could also submit indefinitely ahead of the GPU.
The supplied large compression/divergence cannot be declared healthy just because
global conserved totals are close. The original snapshots omit exact minima, so
they do not establish the magnitude or source of the negative values.

## Changes

- Windowed callback, simulation and render rates, simulated-time/wall-time speed,
  CPU encoding time, and last queue completion latency. Queue latency explicitly
  includes scheduling/driver overhead, not a pure GPU timestamp.
- One application frame in flight using onSubmittedWorkDone. Preserve fractional
  accumulator time for high-refresh pacing, bound backlog to two ticks, and avoid
  resubmitting a clean paused display. Queue failures stop the loop with an error.
- Exact conserved minima and scenario/toggle settings in copied snapshots; a
  conservative CFL bound from last sampled face speed, negative-value and severe
  compression warnings. Stats frame remains visible so stale samples are apparent.
- Clarify Dye FCT / Auto CFL / Dam float labels and help. Disable the float control
  outside dam while preserving its preference for when dam is selected.
- Reproduced f32 outflow normalization overshoot: four independently rounded
  normalized face values re-sum to 1.0000001192092896. Reserve 1e-6 volume margin
  in the shared face limiter and evaluate donor updates as remaining nonnegative
  weight plus incoming contributions in CPU and WGSL. Never clamp conserved
  state. This fixes a concrete roundoff hazard, not the whole projection or
  compression problem, and does not prove the source of the reported negatives.
- Keep the scheme specification and README aligned with those semantics.

## Validation

- `node experiments/conformance.js`: all pass, including draw/stir positivity,
  source accounting and flux limiting. Full output in the diagnostics log.
- `node experiments/convection.js`: all pass, including 10x stability sweep.
- `node experiments/sharpness.js`: pass; 240 steps, width 26 donor vs 8 FCT
  (3.25x), zero local-bound violations; FCT vs donor total-dye drift 6.31e-9.
- `node experiments/transport-roundoff.js`: old f32 overshoot reproduced;
  100000 margin-limited samples have nonnegative donor weight. Production CPU
  transport positivity/conservation with FCT on/off; WGSL structural check only.
- `node experiments/performance.js`: windowed rate/latency/error tests and real
  main-loop harness at 144 Hz (~60 sim/s), held queue, pause/redraw/single-step.
- `node experiments/hud.js`: exact minima, rate labels, configuration context,
  CFL/negative/compression warnings and float applicability.
- controls (23), ui (22), input (28), gpu-mock: all checks pass. Buoy experiment
  completed successfully. `git diff --check` clean.
- Actual GPU parity attempted with external optional webgpu@0.6.2. Blocked:
  libvulkan.so.1 is missing and no adapter is available. No numeric GPU or browser
  performance claim is made for this change.

## Next step

On the user's browser GPU, reset and compare identical N=512/K=60 runs at five
and six substeps using sim/s, speed, queue latency, exact minima and stats age;
compare FCT on/off from separate resets in color view. Then measure pressure
residual versus substep count and prototype a multigrid pressure solver if
projection is confirmed as the bottleneck. Investigate compression separately:
more accurate timing and donor positivity do not restore incompressibility.
