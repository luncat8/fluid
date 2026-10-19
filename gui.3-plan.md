# gui.3 — input ownership

## goal

native controls keep their keyboard behavior, a canvas stroke has one pointer owner,
and interrupted gestures cannot leave a force or a slider lock behind. retain classic
scripts, file:// support, and allocation-free paint-request flushing.

## contract

- Global shortcuts ignore native interactive controls, editable content, composition,
  browser modifier chords and events already handled elsewhere. Discrete commands
  ignore key repeat; pressure adjustment may repeat. Canvas focus restores shortcuts.
- Only a primary-button canvas press starts a stroke. Capture the pointer and ignore
  other pointer IDs until it ends. Captured movement outside the canvas continues with
  coordinates clamped to the domain; touch-action prevents scrolling during a stroke.
- Release, cancel, lost capture and window blur clear stirring and capture. Normal
  release retains the last queued dab; cancellation and blur discard it. Blur clears
  held-key state. Keep one reusable request object for painting.
- View-mode hover requires a fine hover-capable environment AND a mouse event with no
  pressed buttons. Touch/pen remain click-only even on a hybrid device.
- Slider ownership ends on release, cancel, lost capture or focus loss, including
  window-level cancellation and blur. Input previews grid size; change alone commits
  its rebuild. Config synchronization must not overwrite a held thumb.

## validation

Dependency-free Node contracts: experiments/input.js, experiments/controls.js and
experiments/ui.js. GPU mock remains a plumbing check, not shader validation. Follow
with browser checks for native space/arrow behavior, pointer capture outside the
canvas, touch scrolling, hybrid pointers and focus loss during slider drags.
