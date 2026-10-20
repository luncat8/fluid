# Unified test runner and browser diagnostics

## Changes

- Added `experiments/run-all.js`: sequential full and quick profiles, per-suite
  monotonic timings, continued execution after a failure, and a nonzero final
  exit code if any suite fails. Real WebGPU parity is explicit opt-in because
  it needs the external Node WebGPU backend.
- Added `run_tests.py`: UTF-8-safe output tee to a timestamped log in
  `experiments/logs/`, environment header, and optional system-clipboard copy.
- Added `experiments/browser-tests.html`: an in-browser real-WebGPU diagnostic
  capture using the app solver, fixed frame counts, per-scenario resets, final
  and periodic stats, Copy/Save report controls, and a repeatable scripted
  paint/stir case. It supports auto or fixed substeps and the `?mg=0` red-black
  A/B path. The capture labels queue wait as scheduling-inclusive, not a GPU
  timestamp; divergence and compression remain reported measurements rather
  than invented pass/fail thresholds.
- Extended `run_index.py --page` to serve the browser harness without changing
  the existing `index.html` default. COOP/COEP headers are kept on all responses
  so the capture can report cross-origin isolation and the browser's timer
  quantum.
- Corrected a stale controls test: after auto-CFL became the default, the test
  checked that the fixed-step slider affected `substepCount` without switching
  auto mode off. It now exercises the checkbox and fixed branch explicitly.

## Validation

- `node experiments/run-all.js --quick`: all 9 suites pass (~6.5 s).
- `python3 run_tests.py --quick`: all 9 pass; combined capture saved as
  `experiments/logs/2026-10-19-23-53-36-test-suite.log`.
- `experiments/browser-tests.js`: fake-device UI contract passes for the five
  scenarios, scripted paint, in-flight lock, complete report and clipboard.
- Browser page inline script passes `node --check`; a one-shot HTTP request
  verified that the test page is served with COOP/COEP headers.
- The first full-profile run found only the stale fixed-step expectation; the
  other 11 suites (including conformance, convection, buoy and multigrid) passed.
  After correcting the controls test, the full 13-suite profile passed in 629.50 s;
  transcript: `experiments/logs/2026-10-19-test-suite-full.log`.
- No real WebGPU adapter is available in this environment, so the browser page's
  actual shader execution and performance capture remain for a user's device.

## Follow-up

The supplied manual captures show conservation drift below about 1e-7 and no
negative cells. The two non-draw blobs samples also have no overpack and low
maxDiv (0.032 and 0.010). The two draw samples still show local compression
(maxM 1.235/1.481, over 554/833) and high maxDiv (3.84/6.51), even though their
reported CFL bounds are below 1. The repeatable browser `draw/stir` case is now
available to compare against identical `?mg=0` runs; for example, use
`?n=512&frames=600&scenarios=blobs,draw` and the same URL plus `&mg=0`. Run the
same grid, frame count, selected scenarios and substep policy for both captures
before drawing a solver conclusion; these different manual runs are not a
controlled A/B test.
