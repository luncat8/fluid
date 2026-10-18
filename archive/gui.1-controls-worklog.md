# gui.1 — HTML controls panel (in addition to keyboard shortcuts)

date: 2026-10-18
status: implemented
scope: all controls that were keyboard-only now have HTML equivalents

## changes

- new `src/app/controls.js`: creates HTML controls from the DOM; syncs bidirectionally
  with `cfg` so keyboard shortcuts and HTML controls always agree
- `index.html`: added `controls-panel` section with view selector, scenario selector,
  pause/step/reset buttons, FCT/gravity checkboxes, paint mode buttons, dye color
  buttons, pressure iters slider, grid size slider
- `src/app/style.css`: control panel styling (dark theme, monospace, compact rows)
- `src/app/input.js`: `paintColor` moved from local `state` to `cfg.paintColor` so
  both controls.js and input.js share one source of truth; keyboard view range
  extended to 0-7 (matching viewModeNames in config.js)
- `src/app/main.js`: instantiates controls, calls controls.update in the frame loop
- `src/app/hud.js`: help text updated to "0-7 view"

## controls provided

| control | HTML | keyboard |
|---------|------|----------|
| view mode | dropdown (0-7) | 0-7 |
| scenario | dropdown | n |
| pause/resume | button | space |
| step once | button | . |
| reset | button | r |
| FCT toggle | checkbox | f |
| gravity toggle | checkbox | g |
| paint mode | 3 buttons (dye/wall/erase) | b |
| dye color | 4 buttons (R/G/B/Y) | c |
| pressure iters | range slider 1-200 | +/- |
| grid size N | range slider 64-1024 | [ ] |

## validation

- all JS files pass `node --check`
- conformance test: 22/22 PASS (stats, FCT, still box, stirred box)
- controls panel renders in the HUD column; dropdowns are populated from
  config.js at creation time; all controls sync both directions with 200ms throttle

## performance notes (1024x1024 fluctuation)

no atomics in use. the frame time variance at 1024 is from variable substeps (1-8)
and the K=60 pressure solver (960 dispatches at 8 substeps). statsReduceB is
single-threaded over 16641 partials but only fires every 30 frames. recorded in
findings-pitfalls-skills.md.

## next

- add per-frame substep count to the HUD so the user can see when substeps spike
- consider reducing default K or adding adaptive pressure convergence check
- view modes 6 (thermal) and 7 (phase) need shader implementation (0.4.0 / 0.5.0)
