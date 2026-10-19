# gui.2 — view-mode row under the canvas

status: implemented 2026-10-19; browser visual and pointer pass pending
depends on: gui.1 (control panel), 1.0.0 (eight view modes)

## 1. goal

one radio group for the view modes, directly below the canvas. click, the 0-7 keys and
hovering a mode on a fine pointer all change the same mode through one writer, so the
render loop never has to ask the DOM what the current mode is.

## 2. markup

`#view-menu` is the container (`role="radiogroup"`, `aria-label="View mode"`). it holds
one `<label>` per mode; each label wraps an
`<input type="radio" name="view-mode" value="<id>">` and the mode name as text. exactly
one radio is checked at a time and it follows `cfg.viewMode`.

the labels are generated from `Fluid.core.config.viewModeNames` at create time, so the
mode list exists once. the container is a sibling of the canvas inside
`.canvas-column`, not part of the diagnostics panel.

## 3. behavior

	setMode(id)   the single writer. returns immediately when id is already active;
	              rejects an unknown id; otherwise writes cfg.viewMode (the source of
	              truth), sets cfg._dirty, and checks the matching radio. the last step
	              is for the keyboard path: assigning checked does not fire change.

	click         delegated `change` on the container. act only when the target is an
	              INPUT, type radio, name "view-mode"; call setMode(target.value).

	keyboard      input.js routes keys 0-7 to setMode. no direct cfg.viewMode writes
	              remain in the app.

	hover         delegated `pointerover` on the container. walk the event target up
	              until the container or a LABEL; stop if neither. inside the label take
	              `label.control` (the wrapped input), check it, call setMode(value).

	gate          hover only when matchMedia("(hover: hover) and (pointer: fine)")
	              matches. read `.matches` per event so a hybrid device follows its
	              primary pointer; keep one MediaQueryList, never call matchMedia in the
	              handler. touch and pen-only sessions stay click-only.

	render        writeParams reads cfg.viewMode. the frame draws when cfg._dirty or a
	              sim step ran. no render path reads a checked radio or a select.

## 4. hot path

the pointerover handler allocates nothing: no closures, object literals, arrays or
selector strings per event. the parent walk uses node references, the input comes from
`label.control`, and `setMode` returns before writing when the id is active, so
re-entering the same label is free. `pointerover` is used instead of `pointermove`
because entering an element is the switching event and it fires far less often.

## 5. style and placement

the row sits under the canvas, inside the canvas column, and is a compact segmented
control. native radios are visually hidden with a 1 px, opacity 0 box rather than
`display: none`, so the group keeps tab/arrow behavior; the label shows checked and
`:has(input:focus-visible)` states. hover styling is behind the same fine-pointer media
query as the behavior.

## 6. validation

- `node experiments/ui.js`: DOM stub, 19 checks covering markup, single writer, click,
  keyboard, hover gating, hover switching and radio exclusivity.
- `node experiments/gpu-mock.js`: unchanged app frame plumbing, ALL PASS.
- every js file passes `node --check`.
- browser pass pending: one-row fit, focus ring, hover feel, touch-only gating.

## 7. non-goals

no continuous pointermove switching, no animated transition, no persistence of the
chosen mode. a future engine api exposes `sim.setViewMode(mode)` and delegates to this
writer (0.7.0 t8), keeping keyboard, hover, click and external calls on one path.
