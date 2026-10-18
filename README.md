# Fluid

A conservative 2D WebGPU fluid engine for interactive games. One substance, three
conserved color tracers, conserved internal energy; aggregate phases later. The default
grid is 512×512; 1024×1024 runs about four times as many cells per pass.
Currently implemented: full velocity solver (forces, semi-Lagrangian advection,
variable-density red-black Gauss-Seidel projection), donor-cell mass/energy transport,
Zalesak FCT for RGB dye tracers, ledger-accounted dye and wall painting, conservation
meters, six view modes, and a CPU reference solver with a conformance suite. GPU FCT
pipelines are implemented; browser parity/performance measurements remain pending.

## Run

- Directly: open `index.html` from disk (`file://`) in a WebGPU browser.
- Or: `python3 run_index.py` (options: `port`, `--host`, `--no-open`) and open
  `http://127.0.0.1:8000/index.html`.

Keys: `0-5` view modes · `space` pause · `.` step · `r` reset · `n` scenario
(blobs / dam / still) · `b` paint dye/wall/erase · `c` dye color · `f` toggle dye FCT
· drag to stir and paint · `g` gravity · `+/-` pressure iterations · `[ ]` grid size
· `s` dump stats · `h` help.

The HUD shows drift of all five conserved invariants against the source ledger
(green < 1e-6, yellow < 1e-3, red above), min-value watchdogs, negative/overpack
cell counts, max velocity and divergence.

## Validation

`node experiments/conformance.js` runs the CPU reference (the numeric oracle for
the scheme in `0.0.0-plan.md` section 5) through: stats-reduction error, FCT dye
sharpness/bounds, still box, stirred closed box, dam break, wall integrity, emitter
accounting, and a draw stroke that guards the per-cell flux outflow limiter. Run
`node experiments/sharpness.js` for the isolated upwind-versus-FCT front measurement.
Keep conformance output in `experiments/logs/<date>-conformance.log`.

No build, no dependencies, no modules, no network. All WGSL is inline as JS
template strings; `file://` works.

Start with `0.0.0-plan.md` for the architecture and the scheme spec. Development
records are in `archive/*-worklog.md`.
