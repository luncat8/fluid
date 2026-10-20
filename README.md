# Fluid

A conservative 2D WebGPU fluid engine for interactive games. One substance, three
conserved color tracers, conserved internal energy; aggregate phases later. The default
grid is 512×512; 1024×1024 runs about four times as many cells per pass.
Currently implemented: full velocity solver (forces, semi-Lagrangian advection,
variable-density red-black Gauss-Seidel projection), donor-cell mass/energy transport,
Zalesak FCT for RGB dye tracers, ledger-accounted dye and wall painting, conservation
meters, eight view modes, a diagnostic floating body that reads the field without touching
it (dam scenario), an optional auto-CFL substep policy, thermal conduction with
ledger-accounted heater/cold-plate sources and Boussinesq buoyancy (convection
scenario: an unstably stratified box, hot at the bottom and cold at the top,
sustained by a thermostat-capped heater band and a cold plate; selecting it
switches to the temperature view, whose range matches the stratification), and a CPU reference solver with a
conformance suite. GPU thermal pipelines mirror the CPU ones; browser parity/performance
measurements remain pending. The CPU reference additionally carries the 0.5.0 phase core
(caloric EOS with latent plateaus, freeze/melt masks, expansion source S, phase-aware
conduction with an enthalpy budget); its WGSL port is pending.

## Run

- Directly: open `index.html` from disk (`file://`) in a WebGPU browser.
- Or: `python3 run_index.py` (options: `port`, `--host`, `--no-open`, `--page`) and open
  `http://127.0.0.1:8000/index.html`. The server sends COOP/COEP headers for cross-origin
  isolation and finer browser timers; for the GPU diagnostic page run
  `python3 run_browser_tests.py`.

Keys: `0-7` view modes · `space` pause · `.` step · `r` reset · `n` scenario
(blobs / dam / still / convection) · `b` paint dye/wall/erase · `c` dye color · `f` toggle dye FCT
· drag to stir and paint · `g` gravity · `u` auto-substep policy (off = the fixed step
count from the panel, 4 by default) · `y` buoy · `+/-` pressure iterations · `[ ]` grid
size · `s` dump stats · `h` help.

The view modes also switch from the radio row under the canvas: click a mode, or hover
one on a mouse/trackpad. Touch and pen stay click-only, including on hybrid devices.
Shortcuts leave focused form controls alone; click the canvas to return keyboard control to the simulation.

Dye FCT sharpens RGB tracer fronts, not mass, temperature, or velocity. Compare
separate reset runs in color view: enabling it cannot recover edges already smeared
by donor-cell transport. Gravity changes forces, not pass count; a uniform-density,
uniform-temperature closed box has no density-driven motion. The **Dam float**
control is a read-only floating marker in the dam scenario, not thermal buoyancy.

Performance readouts distinguish simulation frames/s (target 60), renders/s, and
browser rAF callbacks/s. Speed is simulated seconds per wall second; CPU encode ms
measures submission work; queue ms is last submission-to-completion latency, including
driver/scheduling overhead, **not** a GPU timestamp. One application frame is allowed
in flight; overload slows simulated time rather than building an unbounded GPU queue.
A clean paused view submits no redraws. Changing substeps from 5 to 6 adds 20% more
solver work, but display scheduling can make observed rates change nonlinearly.

The HUD shows drift of all five conserved invariants against the source ledger
(green < 1e-6, yellow < 1e-3, red above), min-value watchdogs, negative/overpack
cell counts, max velocity and divergence. It also flags a conservative 2D CFL
bound (`2 * max face speed * dt`, based on the last stats sample), negative values,
and severe compression. Small global drift does not imply a stable local field.
Copied stats include exact minima and scenario/toggle settings. The donor limiter
reserves a 1e-6 outflow margin and transport uses nonnegative weights to avoid
f32 cancellation in almost-emptied cells; no conserved state is clipped.

## Validation

Run the portable CPU, UI-contract and recording-GPU-mock suites together with
`node experiments/run-all.js`. This is the full profile; `--quick` runs the fast
numerical/UI/plumbing checks without the longer numerical histories (~7 s here versus
about 10.5 min for the full run). The runner continues
after a suite fails, reports per-suite/total durations, and exits nonzero if any
suite fails. To keep a single, timestamped transcript under `experiments/logs/`,
run `python3 run_tests.py`; `python3 run_tests.py --copy` also copies the complete
transcript to the system clipboard when a clipboard utility is installed (the saved
log remains available otherwise; `--quick` is supported here too). The optional
real-device Node parity experiment is not in the default suite because it needs
the external `webgpu` package and a working backend; opt in with
`python3 run_tests.py --gpu --copy`.

For actual browser/WebGPU diagnostics, run
`python3 run_browser_tests.py` (options: `port`, `--host`, `--no-open`, `--query`;
it serves `experiments/browser-tests.html` on the same COOP/COEP server as
`run_index.py`). It executes the
selected scenarios (including a repeatable paint/stir stress case) at a fixed
grid/frame count, resets between scenarios, and provides **Copy report** and
**Save report** (save into `experiments/logs/` as `<date>-GPU-browser-tests.log`).
Query parameters prefill comparable runs, for example
`--query "n=512&frames=600&scenarios=blobs,draw"` and the same query plus `&mg=0` for
multigrid versus red-black; `&auto=0&ns=4` selects the same fixed four-substep
policy in both.
The report records `crossOriginIsolated` and an observed
`performance.now()` quantum; the Python server's COOP/COEP headers enable the
finer timer where the browser supports it. These browser measurements are a
GPU diagnostic capture, not a substitute for the CPU acceptance suite, and
queue wait includes driver/scheduling overhead rather than a GPU timestamp.

`node experiments/conformance.js` runs the CPU reference (the numeric oracle for
the scheme in `0.0.0-plan.md` section 5) through: stats-reduction error, FCT dye
sharpness/bounds, still box, stirred closed box, dam break, wall integrity, emitter
accounting, and a draw stroke that guards the per-cell flux outflow limiter. Run
`node experiments/eos.js` for the caloric EOS contract (plateau boundaries, T(E)
monotonicity, rho* continuity, S signs and clamps) and `node experiments/phase.js`
for the phase transitions (melt/boil/condense time series, latent ledger, freeze
mask), `node experiments/sharpness.js` for the isolated upwind-versus-FCT front measurement,
`node experiments/convection.js` for the thermal plan (derive_T, clamped conduction
conservation/monotonicity, source ledger closure, convection onset and buoyancy sign,
10x stability sweep), `node experiments/buoy.js` for the floating body (float line versus dye level and versus
the sample refresh interval), `node experiments/gpu-mock.js` for the GPU plumbing (a
recording mock device drives real app frames and lints the WGSL param indices), and
`node experiments/ui.js` for the view-mode row contract (a small DOM stub; it checks the
single writer, click, keyboard, hover gating and radio state). Run
`node experiments/input.js` for canvas pointer ownership, cancellation and keyboard
focus handling, and `node experiments/controls.js` for slider drag ownership and
configuration synchronization. `node experiments/performance.js` checks windowed
rates and drives the real app loop through high-refresh pacing, queue backpressure,
pause and single-step. `node experiments/transport-roundoff.js` reproduces f32
unit-outflow overshoot and checks the positive-weight transport (CPU numeric and
WGSL structure, not GPU execution). `node experiments/hud.js` checks snapshot
context, minima, rates and warnings. The mock runs no shader,
so it is plumbing only, not a numeric oracle.
Keep conformance output in `experiments/logs/<date>-conformance.log`.

For numeric CPU/GPU parity, the optional `node experiments/gpu-parity.mjs` runner
executes the actual compute shaders and compares mass, thermal fields, velocity,
pressure, flux, divergence source, and flags at selected steps. It uses the optional
Node `webgpu` package and requires a working hardware or software WebGPU backend (a
null backend is not a numeric oracle). Install it outside the repo and run, for example:

```sh
npm install --prefix /tmp/fluid-webgpu webgpu@0.6.2
FLUID_WEBGPU_MODULE=/tmp/fluid-webgpu/node_modules/webgpu/index.js \
  FLUID_WEBGPU_BACKEND=vulkan node experiments/gpu-parity.mjs
```

The measured 500-step heated-box results are recorded in
`experiments/logs/2026-10-19-gpu-parity.log`. Per the clarified t8 criterion, aggregate
total-E and mean-T are required below 2e-6; this run meets it. Pointwise E/T drift is
reported separately in the log. Browser/hardware GPU performance is still pending.

The app has no build-time or runtime dependencies. All WGSL is inline as JS
template strings; `file://` works. The optional numeric parity experiment is the one
exception and keeps its Node dependency outside the repository.

Start with `0.0.0-plan.md` for the architecture and the scheme spec. Development
records are in `archive/*-worklog.md`.
