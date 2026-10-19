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
measurements remain pending.

## Run

- Directly: open `index.html` from disk (`file://`) in a WebGPU browser.
- Or: `python3 run_index.py` (options: `port`, `--host`, `--no-open`) and open
  `http://127.0.0.1:8000/index.html`.

Keys: `0-7` view modes · `space` pause · `.` step · `r` reset · `n` scenario
(blobs / dam / still / convection) · `b` paint dye/wall/erase · `c` dye color · `f` toggle dye FCT
· drag to stir and paint · `g` gravity · `u` auto-substep policy (off = the fixed step
count from the panel, 4 by default) · `y` buoy · `+/-` pressure iterations · `[ ]` grid
size · `s` dump stats · `h` help.

The view modes also switch from the radio row under the canvas: click a mode, or hover
one on a mouse/trackpad. Touch and pen stay click-only, including on hybrid devices.
Shortcuts leave focused form controls alone; click the canvas to return keyboard control to the simulation.

The HUD shows drift of all five conserved invariants against the source ledger
(green < 1e-6, yellow < 1e-3, red above), min-value watchdogs, negative/overpack
cell counts, max velocity and divergence.

## Validation

`node experiments/conformance.js` runs the CPU reference (the numeric oracle for
the scheme in `0.0.0-plan.md` section 5) through: stats-reduction error, FCT dye
sharpness/bounds, still box, stirred closed box, dam break, wall integrity, emitter
accounting, and a draw stroke that guards the per-cell flux outflow limiter. Run
`node experiments/sharpness.js` for the isolated upwind-versus-FCT front measurement,
`node experiments/convection.js` for the thermal plan (derive_T, clamped conduction
conservation/monotonicity, source ledger closure, convection onset and buoyancy sign,
10x stability sweep), `node experiments/buoy.js` for the floating body (float line versus dye level and versus
the sample refresh interval), `node experiments/gpu-mock.js` for the GPU plumbing (a
recording mock device drives real app frames and lints the WGSL param indices), and
`node experiments/ui.js` for the view-mode row contract (a small DOM stub; it checks the
single writer, click, keyboard, hover gating and radio state). Run
`node experiments/input.js` for canvas pointer ownership, cancellation and keyboard
focus handling, and `node experiments/controls.js` for slider drag ownership and
configuration synchronization. The mock runs no shader,
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
