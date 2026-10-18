# Fluid

A planned conservative 2D WebGPU fluid engine. Currently implemented: page skeleton
and capability probe only; the canvas does not yet simulate fluid.

## Probe first

1. Open `experiments/probe.html` directly from disk in the intended browser (`file://`).
2. Wait for the result and choose **Save probe log**.
3. Put the downloaded log in `experiments/logs/` with the browser/machine identified.
   Review the log before sharing: it includes browser and GPU information.
4. If WebGPU is unavailable on `file://`, run `python3 -m http.server 8000` from
   this directory and open `http://localhost:8000/experiments/probe.html`.
   Save that result separately; HTTP/HTTPS success does not verify `file://`.

No build, dependencies, modules, network assets or local fetches are used by the page.
The probe validates device creation and the five required texture formats, not shader
execution or simulation performance. Do not continue the GPU harness until the target
browser result establishes the runtime path.

## Validation

`node experiments/probe-test.js` tests probe control flow with mocked WebGPU objects.
It is **not** evidence of browser/GPU support.

Start with `0.0.0-plan.md` for architecture; `0.5.0-plan.md` covers phase changes and
thermal solid growth. Development records are in `archive/*-worklog.md`.
