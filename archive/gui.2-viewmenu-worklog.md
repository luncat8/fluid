# gui.2 — view-mode row under the canvas

date: 2026-10-19
status: implemented, validated headless (node); browser visual and pointer pass pending
source: user contract "View-mode menu (click + pointer hover)" + "move it to row below canvas"
plan: archive/gui.2-viewmenu-plan.md

## goal

one radio group for the eight view modes, sitting under the canvas instead of in the
diagnostics panel. it must be switchable by click, by the 0-7 keys, and by hovering a
mode on a device with a fine pointer. one writer owns the mode, so the three input
paths and the render path cannot disagree.

## review of the previous ui work (gui.1 / 1.0.0)

`controls.js` owned a `<select id="ctrl-view">` and wrote `cfg.viewMode` from its
`change` handler; `input.js` wrote `cfg.viewMode` directly for keys 0-7. that is two
writers and a DOM control that the render path could not trust. the select also lived
in the HUD column, away from the canvas it controls. both are replaced here.

## changes

- new `src/app/viewmenu.js`: builds one `<label><input type="radio" name="view-mode"
  value="i">name</label>` per entry of `Fluid.core.config.viewModeNames` into
  `#view-menu`. `setMode(id)` is the single writer: it returns on the active id, rejects
  an unknown id, writes `cfg.viewMode`, marks `cfg._dirty` and checks the matching radio
  (the keyboard path needs that last step: a programmatic check fires no `change`).
- `index.html`: the canvas is wrapped in `.canvas-column` with `#view-menu` directly
  below it; the View select is gone from the controls panel; `viewmenu.js` loads after
  `gpu/view.js` and before `input.js`; the help overlay names click and hover.
- `src/app/input.js`: `createInput(canvas, cfg, solver, setMode)`; keys 0-7 call
  `setMode(parseInt(k,10))` instead of assigning cfg.
- `src/app/main.js`: creates the menu after the solver, then passes `viewMenu.setMode`
  to input. the render path still reads `cfg.viewMode` through `writeParams`, never the
  checked radio.
- `src/app/controls.js`: the view select population and its 5 Hz cfg -> dom sync are
  removed; the panel keeps scenario, transport, toggles, paint, sliders.
- `src/app/style.css`: the row is a compact segmented control: native radios are
  invisible (1 px, opacity 0) but still focusable, the label carries the checked and
  focus-visible styles. hover styling is behind the same `(hover:hover) and
  (pointer:fine)` query as the behavior.
- new `experiments/ui.js`: a dependency-free DOM stub drives the real `viewmenu.js`
  and the keydown route in `input.js`; 19 checks.

## decisions

- markup is generated from `viewModeNames` rather than duplicated as eight static
  labels: one list of names, one list of values, and the initial checked radio follows
  `cfg.viewMode`.
- the delegated `change` listener checks `tagName === "INPUT"`, `type === "radio"` and
  `name === "view-mode"`, so a future checkbox inside the container cannot switch views.
- the hover listener is `pointerover`, not `pointermove`: it fires on entering an
  element, which is exactly the switching event, and far less often. the handler
  allocates nothing: the parent walk uses the event target, `label.control` avoids a
  selector string, and the early return in `setMode` makes repeated visits free.
- the hover gate reads `mediaQuery.matches` per event instead of deciding once at
  setup, so a hybrid laptop that switches from touch to mouse starts hovering without a
  reload. reading the property allocates nothing.
- `pointerover` fires for the radio as well as the label; the parent walk stops at the
  first `LABEL` and then returns, so one enter is one mode change, not two.
- no hover cache is kept: the contract says the input is checked and `setMode` called
  on each label enter, and an early return on the active id is already the fast path.
  a cache would also suppress a re-assert after a keyboard change while the pointer
  stays over the old label.
- the radios are visually hidden, not `display:none`: the group keeps native radio
  keyboard behavior (tab to the checked one, arrows to move) and the label gets the
  focus ring via `:has(input:focus-visible)`.

## validation

`node experiments/ui.js` — 19 checks, ALL PASS (`experiments/logs/2026-10-19-ui.log`):
eight labels, one radio each, group name/values, exactly one checked at boot, label
text; a bubbled `change` sets the mode and dirties the view; one checked after click;
`setMode` on the active id does not dirty; unknown ids are ignored; key 5 reaches the
same writer and checks the radio; a touch/stylus `pointerover` (media query false)
does not switch or uncheck; a fine-pointer `pointerover` switches and checks exactly
one radio; hovering the label text works; the container itself does nothing; a
`change` from a foreign input is ignored.

`node experiments/gpu-mock.js` — ALL PASS (25 checks): the solver, app frame loop and
resize path are unchanged; the new `main.js` create order does not affect it.

`node experiments/conformance.js` — ALL PASS (50 checks, 2m45s): the core scheme is
unchanged, re-run to confirm the app wiring edit changed nothing numeric.

Every js file under `src/` and `experiments/` passes `node --check`.

No browser exists in this sandbox: the visual pass (one-row fit at 1100 px, focus ring,
hover feel, touch-only gating on a real device) remains open.

## review of the next plans (0.6.0, 0.7.0)

- 0.6.0 baseline: every row must now pin the step policy (off at 4 fixed steps, and on
  with an ns histogram). frame cost scales with ns, so an unpinned baseline is not
  comparable. added readback accounting (stats 128 B / 30 frames, buoy 24 B / 4 frames,
  skipped samples, mapAsync latency) and a separate display timing, plus a 4.6 section
  for readback and display variants. frame pacing is reported as p50/p95/1 % low.
- 0.6.0 projection: the plan listed red-black gauss-seidel as a candidate, but the
  current gpu and cpu solvers already implement it, and plain jacobi is rejected in
  0.2.0. t3 is now a convergence/timing trace of the existing red-black baseline; the
  only real candidate is multigrid.
- 0.7.0 sample(): the whole-field cpu mirror was stale by stats_every and 1 MB per
  readback at N=512. replaced with the probe-table contract the buoy already proves:
  up to 32 fixed points, one kernel, one small readback on a probeEvery cadence, with
  the existing generation guard.
- 0.7.0 view mode: setViewMode is declared the same single writer as this row, and
  bindKeys routes through it, so an external caller cannot diverge from click/hover/keys.
- 0.7.0 save/load: the header carries the step policy (substepsAuto, substepsFixed,
  substepMax, frameDt); ns changes the trajectory, so a restored field with a different
  policy is refused or reported, not replayed silently. update() reuses
  config.substepCount rather than re-deriving the policy.

## next

- browser pass: confirm the row fits under the canvas at the default width and wraps
  without clipping below ~700 px; check the focus ring when tabbing into the group;
  confirm a touch device stays click-only and a mouse switches on enter.
- 0.7.0 t8: `sim.setViewMode` must delegate to this writer (the plan now says so), so
  an external caller, the keys and the row share one path.
- if another hover-driven control appears, reuse the parent-walk pattern rather than
  adding a second matchMedia/gate implementation.
