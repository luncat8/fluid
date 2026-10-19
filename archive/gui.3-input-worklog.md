# gui.3 — input ownership

date: 2026-10-19
plan: gui.3-plan.md
status: implemented, validated headless; browser pointer/keyboard pass pending

## review

The view row had native radio semantics, but the global space handler prevented
native activation and toggled pause instead. The same global handler consumed keys
on other controls and repeated resets/resizes on held keys. Its hover media query
also allowed touch/pen pointerover on a laptop with an attached mouse.

Canvas capture was not tracked by pointer ID: a second finger could move or end the
first stroke. Losing capture or window focus could leave mouseStrength active.
Pointerleave ended a captured drag unnecessarily, and default touch scrolling could
cancel drawing. Slider ownership handled local cancellation but not window cancel,
window blur or lost capture. The previous ad-hoc slider checks were not in the repo.

## changes

- input.js: respect native controls/editable content, composition, modifier chords
  and handled events; ignore repeats except pressure adjustment. Skip no-op grid
  resizes. Canvas press focuses the canvas, accepts only the primary button, and
  owns one captured pointer ID. End stirring on up/cancel/lost capture/blur, not leave.
  Keep the last dab on normal release; discard it on cancellation or blur. Clear held
  keys on blur. Reuse paint storage and write UV coordinates without temporary arrays.
- viewmenu.js: gate hover on mouse pointerType and buttons=0 in addition to media.
- controls.js: unlock sliders on lost capture and window cancellation/focus loss.
- style.css: touch-action:none on canvas only; the rest of the page stays scrollable.
- README/help: document focused-control shortcut ownership. New persistent input and
  controls harnesses; extend the view-row harness with hybrid-pointer/drag checks.

## validation

`experiments/logs/2026-10-19-input-ui.log` records:
- node experiments/input.js — 28 checks pass.
- node experiments/controls.js — 17 checks pass.
- node experiments/ui.js — 22 checks pass.
- node experiments/gpu-mock.js — 25 checks pass (no shader execution).
- node --check on all source and experiment JS — pass.

No solver numerics changed; the CPU conformance run was not repeated. DOM stubs
exercise real handlers, not browser event ordering, native default actions or CSS.

## next

Browser pass: tab through controls and use space/arrows without pausing the sim;
click the canvas to restore shortcuts; drag outside it and back; interrupt strokes
and grid drags with focus loss. Check touch input and a hybrid mouse/touch device.
Then add a real WebGPU numeric parity harness: shader execution remains the largest
validation gap before thermal sources/conduction (0.4.0).
