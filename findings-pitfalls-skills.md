# findings, pitfalls, skills

notes for llm agents working in this repo. add a line when something cost you time or when
you find a good way to do something. keep entries short, factual and dated when the fact
might change. the model itself lives in 0.0.0-plan.md section 5, not here.

legend: [verified] measured or read in a spec. [to verify] reasoned, needs a probe.

## webgpu facts that shape the architecture

- [verified] core webgpu allows read_write access on storage textures only for r32float,
	r32sint, r32uint. every other storage format is read or write within one shader.
	we therefore never write in place: every stencil pass is ping pong (read texture A with
	textureLoad, write texture B with a write-only storage binding). this needs no optional
	feature and no extension. the wgsl extension readonly_and_readwrite_storage_textures and
	the texture-formats-tier2 feature widen this, but we do not request them (0.0.0 d11).
- [verified] core storage texture formats: r32{uint,sint,float}, rg32{uint,sint,float},
	rgba32{uint,sint,float}, rgba16{uint,sint,float}, rgba8{unorm,snorm,uint,sint}.
	our field set uses rg32float, rgba32float, r32float, rgba8uint, rgba16float — all core.
- [verified] float32 textures are not filterable without the float32-filterable feature.
	always textureLoad them (integer coordinates, no sampler). when a filtered upscale is
	wanted, convert into an rgba16float display texture in a compute pass: rgba16float is
	filterable in core. that is why the render pass never samples a sim field directly.
- [verified] default limits: maxComputeInvocationsPerWorkgroup 256 (128 in compatibility
	mode), maxComputeWorkgroupStorageSize 16384 bytes, maxComputeWorkgroupsPerDimension
	65535, maxUniformBufferBindingSize 65536, maxStorageTexturesPerShaderStage 4.
	skill: use 8x8 = 64 invocations. it fits compatibility mode too and stencil kernels do
	not need more. count storage texture bindings per stage: our heaviest pass (to_display)
	stays at or below 4.
- [verified] no float atomics in core wgsl (atomicAdd is i32/u32 only). all reductions are
	hierarchical: workgroup shared memory -> one partial per workgroup -> a second tiny pass.
	bonus: order independent, so results are deterministic, which save/load and tests need.
- [verified] dispatches inside one compute pass are ordered and the implementation inserts
	the barriers, so a pass can read what the previous dispatch wrote. there is no
	synchronization *within* a dispatch, which is the only reason ping pong exists.
- [verified] copyTextureToBuffer needs bytesPerRow to be a multiple of 256; writeTexture
	does not. pad the readback path or read garbage at row ends.
- [verified] out of bounds textureLoad returns zeros instead of erroring. index bugs in the
	ghost ring therefore look like "mass silently leaks at the border". clamp indices
	explicitly and assert on the cpu reference.
- [to verify] webgpu requires a secure context. browsers disagree about whether file:// is
	one: chrome documentation lists file: as an insecure protocol, other sources state
	file:// qualifies. probe it first (experiments/probe.html, task 0.1.0 t2) and record the
	result here. fallback if it fails: python3 -m http.server in the repo root.
- [verified] fetch() of a local file fails on file:// (opaque origin), while <script src>
	and <img src> work. that is why all wgsl is inline as js template strings and the page
	uses classic script tags with relative paths (0.0.0 d12).
- [verified] device.lost and device.onuncapturederror are silent unless wired. wire both to
	an on screen panel on day one; webgpu validation errors are otherwise invisible.

## conservation skills (the point of this project)

- skill: compute a face flux exactly once per step into its own texture, then let both
	neighbouring cells read that one stored float and apply it with opposite signs. the global
	sum then cannot drift, because the two operations use the identical value. recomputing the
	flux inside the advection kernel (even from the same velocity texture) breaks this.
- skill: advect conserved *densities* (mass, dye mass, internal energy), never primitives
	(temperature, concentration, velocity). advecting T and recomputing E loses energy exactly
	where two cells with different heat capacity mix.
- skill: pairwise exchanges (conduction, viscosity) must be ping pong too, and both cells
	must derive the exchanged amount from the same pre-update values. in place means the
	second writer sees the first writer's result and conservation dies silently.
- skill: clamp the exchange to 0.5 * C_harmonic * delta. that single clamp makes explicit
	diffusion unconditionally stable and monotone for any coefficient, so "more conductive"
	degrades to "rate limited" instead of exploding.
- skill: never clip a negative density to zero — that deletes mass. use a flux limiter
	(zalesak fct) which redistributes instead of clipping, and keep a min value watchdog in
	the stats reduction so a limiter bug is visible in the same frame.
- skill: keep a ledger. expected = initial + injected - removed - boundary_loss, and show
	drift_rel in the hud colored by magnitude. a conserved quantity that is not measured is
	not conserved, it just looks conserved today.
- skill: emitter amounts must be summed from the exact same numbers uploaded to the gpu (a
	weight table), not from an analytic cpu formula. a 1 ulp difference per step shows up as
	drift after an hour.
- pitfall: float16 for any conserved field. forbidden (0.0.0 section 6). if memory is tight,
	lower the grid, not the precision.

## numerics pitfalls

- pitfall: arithmetic mean face density in a variable density projection produces spurious
	currents at large density ratios. use the harmonic mean, and floor rho at rho_vac so
	1/rho cannot explode next to vacuum.
- pitfall: gravity applied after the projection adds energy every step. apply forces before.
- pitfall: semi-lagrangian advection of a conserved quantity. unconditionally stable and
	quietly non-conservative. it is fine for velocity only, and momentum is explicitly not a
	tracked invariant in this project.
- pitfall: lax-wendroff / fct high order fluxes are only valid for |flux| <= 1 per face per
	substep. enforce cfl <= 0.8 with substeps and count violating faces as a watchdog.
- pitfall: latent heat plateaus must be evaluated from the final E (a piecewise function),
	never integrated incrementally, or a large dt skips the plateau and the budget breaks.
- pitfall: T_ref (buoyancy reference) and T_amb (initial/boundary temperature) must be the
	same constant, otherwise a uniform field starts moving by itself.
- pitfall: buoyancy or capacity on vacuum cells (m ~ 0) creates fast invisible jets and
	invented energy. guard with m >= m_eps, do not clamp m to eps.

## agent workflow in this repo

- plans are artifacts, not logs (AGENTS.md). when a plan is implemented, move it to archive/
	with its worklog and update 0.0.0 section 12 roadmap, but keep the plan text readable by
	someone who has never seen the chat.
- the scheme spec is 0.0.0-plan.md section 5. cpu reference and wgsl both implement it. if
	they disagree, fix section 5 first, then both. do not patch one side to match the other.
- every acceptance criterion in a plan is a number. if a plan's acceptance cannot be
	measured, the plan is not finished being written.
- node experiments need no dependencies and no npm install. they run with
	`node experiments/<name>.js` and append to experiments/logs/<date>-<name>.log.
- files are isomorphic: iife attaching to the global Fluid namespace, then
	`if (typeof module !== "undefined" && module.exports) module.exports = ...`
	so the browser gets script tags and node gets require, from one file.
- no allocations in the frame loop: bind groups, pipelines, textures, uniform staging and
	scratch typed arrays are created at setup or on resize. the hud text rebuild is the one
	acknowledged exception and is throttled to 5 hz.
- patch scripts must read before they write. `open(p,'w').write(open(p).read().replace(...))`
	evaluates the write-open first, which truncates the file, so the read returns an empty string and
	the source is gone: happened on 2026-10-18 to src/gpu/passes.js and src/gpu/fields.js and was
	recovered with `git checkout --` plus a re-patch. assert each anchor exists before writing, and
	prefer a line-indexed splice over a multi-line literal match when you have not confirmed the
	exact tab depth of the anchor.
- note: .gitignore.txt in the repo root is not an active gitignore file (git reads
	.gitignore). if node_modules ever appears, rename it first. [to verify with the owner]

## 2026-10-18 probe and solid-growth review

- [verified] Node probe control-flow tests passed (mocks only, no browser capability).
  Probe page, test and logs were retired 2026-10-18: capability facts below are the
  record; the code was noise in the repo.
- [verified 2026-10-18] User target-browser file:// probe passed on Nvidia Ampere
  (core features only, all five formats ok, limits maxTextureDimension2D 8192,
  maxStorageTexturesPerShaderStage 4). file:// is a secure context in that browser;
  runtime path is core WebGPU with no optional features.
- skill: createTexture can report validation asynchronously. Use pushErrorScope and await
  popErrorScope for each tested format; a returned texture handle alone is not a pass.
- pitfall: latent heat is already included in conserved enthalpy E. Adding m*L_f when
  the freeze mask flips counts it twice. Growth removes energy through conduction.
- [to verify] phase conduction needs enthalpy-aware pair bounds and a multi-face budget

## 2026-10-18 numerics findings (0.2.0 / 0.3.0 implementation)

- [verified] plain jacobi cannot solve the closed-box (all-neumann) pressure system:
  the checkerboard mode is an exact eigenvalue -1 of the jacobi iteration on any
  bipartite 5-point stencil with dropped boundary faces. it oscillates forever
  (residual stalls at its amplitude: measured 25.77 of |b|~400 after 600 iterations,
  any K). red-black gauss-seidel kills it in one sweep (omega=1; over-relaxation
  diverges on the singular system: measured div-after-grad 400/800 at omega 1.7/1.9
  vs 0.38 at omega 1.0, 40 sweeps). default pressureIters=60.
- [verified] the 0.0.0 section 5.6 gravity form a = g*(rho_ref - rho)/rho_ref is
  unstable with an inexact projection: the uniform part re-enters through the walls
  every step, the residual divergence slowly evacuates/overpacks corner mass, and
  buoyancy amplifies it (still box exploded to 1e5 max|vel| at any K). subtracting
  the domain-mean density instead (a = g*(rho_mean - rho)/rho_ref) implements the
  plan's own "only differences matter" claim exactly: uniform boxes are then exactly
  still (measured max|vel| = 0 after 500 frames) and the feedback is gone.
- [verified] donor-cell transport with f32 storage has a conservation floor: the
  per-cell store rounds to f32, so the global total is a random walk at ~eps32 scale
  (measured 3.7e-8 relative over 1200 stirred frames, 8.5e-9 over a dam break).
  "exactly 0.0" is not achievable with f32 fields; acceptance thresholds are 1e-7.
- [verified] the +-1 per-face flux clamp is NOT a safety net: donor cell needs the
  per-cell total outflow k_out <= 1. a diverging cell can empty through up to 4
  faces at once (k_out up to 4) and the q' = q*(1-k)+in update amplifies
  sign-alternating mass every step (measured 2026-10-18: dye draw + drag stir at
  N=256, stale substep counts -> minM -8, maxM 2.3e38, totals NaN in ~15 frames).
  fix: compute_flux scales the faces a cell owns as outflow so they sum to <= 1
  (each face leaves exactly one of its two cells, so the shared stored flux stays
  consistent and conservation is exact). this makes transport monotone at any
  cfl; substepping (cfl 0.8, now from maxVel and live drag speed) keeps fluxes
  below the limiter so mass still follows the velocity field. the earlier note
  here ("clamp stays as the safety net only") was wrong and is superseded.
- [verified] a 20:1 density ratio (liquid 1.0 in 0.05 ambient) is numerically stiff:
  near-vacuum cells amplify residual pressure error by 1/rho (20x) and the interface
  churns. scenario ambient is 0.3 (~3:1) until 0.5.0 gives gas its own rho* and S.
- [verified] liquid needs the expansion source S from day one, not only gas in 0.5.0:
  without it nothing resists compaction, the projection residual overpacks the pile
  and gravity runs it away. minimal form used now: S = clamp((m - rho_liq)/(m*tau),
  0, S_max) for m > rho_liq only. dam break at 64^2, K=60, substepped: maxDiv 0.5,
  zero overpack, no negatives.
- [verified] the false hud [warn] on m/E drift was the level-2 sequential f32 sum of
  the stats reduction, not field loss: sequential error +0.035 on m=5055 and +25 on
  E=6.2e6 (the exact numbers seen in the hud), kahan compensation drops it to 1e-8
  relative. all reductions that must match an f64 ledger sum use kahan.
- skill: gpu bind groups for every (mass-parity, vel-parity) combination are
  prebuilt at setup/resize (~30 groups); the frame loop only selects them. press
  parity is invariant: one red-black sweep goes press[0] -> press[1] -> press[0].
- skill: interactive painting uploads a small patch texture and runs an additive
  kernel; the ledger sums the same f32 patch values the gpu adds, so injection
  accounting is exact by construction (measured 2.8e-8 relative over 50 patches).
- [verified] no bgra swizzle is needed anywhere: webgpu fragment output is logical
  rgba and the bgra8unorm canvas target maps it; the 0.3.0 plan note was over-cautious.
- pitfall: file:// clamps performance timers, so fps/frame-ms numbers from a local
  page are coarse. use run_index.py (http server) when timing matters; timing a
  submit call measures encode, not gpu execution, so the hud shows rAF fps only.
  A prior N=256 report was ~180 fps on the reporter GPU; it is not comparable to the
  user's later N=1024 report (~11 fps, K=60, up to 8 substeps). The default is now
  512 per the user's request. No current-browser benchmark was run in this turn.
  By code count, K=60 means 120 full-grid red/black projection dispatches per
  substep, or 960 per simulated frame at 8 substeps; this is a dispatch estimate,
  not a GPU timing. The pressure solve is the obvious next performance target.
  The earlier isolated-pair clamp claim does not by itself establish monotonicity
  of a simultaneous four-neighbour update or latent-plateau treatment.
- [verified 2026-10-18] Y inversion: physics Y is up (0 bottom, N top), storage Y is
  down (0 top). with the old renderVert uv (0,1),(2,1),(0,-1) NDC bottom sampled
  texture bottom (high physics at screen bottom) -> inverted draw, inverted gravity,
  and inverted drag. fixed renderVert to (0,0),(2,0),(0,2) so NDC bottom -> uv 0 ->
  texture top (low physics at screen bottom), NDC top -> uv 1 -> texture bottom
  (high at top). painting and gravity now match screen; no change to input mapping
  (v=1-(y/h) already maps screen top to high physics).
- [superseded design note, 2026-10-18] An earlier FCT draft proposed composing the
  anti-diffusive scalar correction with the k_out <= 1 volume-flux budget. That is
  dimensionally invalid: dye anti-flux is a tracer-density flux, not a volume flux.
  The shipped design limits only RGB dye with Zalesak cell bounds; mass and energy
  remain donor-cell. Do not add the dye correction to F or k_out.

## 2026-10-18 FCT and performance follow-up

- [verified CPU] Dye-only Zalesak FCT uses `d(F)=0.5*abs(F)*(1-abs(F))*(qR-qL)`
  and two-sided receiver/sender ratios from the 3x3 bounds of the low-order result.
  A 1e-5 ratio safety margin avoids f32 boundary undershoots; no post-hoc clipping.
- [verified CPU] `node experiments/sharpness.js`: N=256, 240 steps, F=0.5 gives
  upwind front width 26 cells and FCT width 8 (3.25x sharper), FCT front range
  0..0.8, zero local-bound violations, relative dye-sum difference 6.307e-9, and
  matching fluid mass totals. A deterministic random 2D field also has zero bound
  violations, min dye 1.409e-7, and FCT correction drift 3.127e-10.
- [verified CPU] Full conformance passes with FCT enabled, including 10,000 closed
  transport steps: worst five-invariant drift=5.96e-7 (mass), below 1e-6; 1200-frame
  stirred drift worst_rel=4.66e-7; no negative cells; wall/emitter/draw/dam tests pass.
  Extra f32 write for corrected dyes raises the measured drift floor, so FCT's
  acceptance ceiling is 1e-6 relative (not exact-zero or 1e-7).
- [verified validation only] WGSL parses; compute/render pipelines, bind groups, and
  an eight-substep FCT command buffer (K=0) validate and submit through Dawn's null
  backend. One isolated FCT-off step also submits. A separate repeated mixed-submit
  probe aborted natively (`std::system_error: Invalid argument`), so treat this as
  pipeline/encoding validation only, not numerical parity. The null backend does not
  execute on the user's browser GPU.
- FCT GPU storage adds 40 bytes/cell (third mass slot + limiterA/B): about +10.6 MB
  at N=512 and +42.1 MB at N=1024, before stats/staging. New total texture/staging
  budget is ~42 MB at N=512 / ~168 MB at N=1024.
- [not measured] Browser CPU/GPU parity and frame-time overhead at 512/1024 remain
  pending. The current code has ~960 full-grid pressure dispatches per simulated
  frame at K=60 and 8 substeps; focus future timing on pressure iteration count,
  substep behavior, and a multigrid path rather than claiming FCT or the default
  change improved FPS without measurement.

## 2026-10-18 performance analysis (user report: 1024x1024 fluctuation)

- [verified by code review] No atomics used anywhere. Stats reduction: workgroup
  shared memory → partials buffer → single-thread Kahan final reduce.
  All dispatches within one compute pass are implicitly ordered.
- The 1024x1024 frame time fluctuation is caused by:
  1. Variable substeps (1-8) depending on maxVel: 8 substeps = 8× all passes
  2. Pressure solver: K=60 → 120 red/black dispatches per substep → up to 960
     per frame at 8 substeps. Each dispatch touches 1026×1026 texture.
  3. FCT adds 2 extra full-grid dispatches per substep (limiterPrepare + limiterApply)
  4. statsReduceB is single-threaded: at N=1024, 129×129=16641 workgroups, each
     writing 32 floats = ~2MB of partials to loop over sequentially. Runs every
     statsEvery=30 frames.
- The dominant cost at 1024 is the pressure iteration count × substeps. At K=60
  and 8 substeps that's 960 full-grid dispatches. Reducing K or the substep cap
  is the most effective lever.
- The statsReduceB single-thread bottleneck at 1024 is notable but only fires
  every 30 frames, so it causes a spike, not steady fluctuation.

## 2026-10-18 diagnostic body (1.0.0) and the optional substep policy

- skill: a read-only diagnostic body costs one small kernel, not a solver change. one
	workgroup (64 lanes, workgroup reduction, no atomics) sums the cell block under the
	rectangle into a 24-byte storage buffer; the frame loop copies it into the existing
	MAP_READ pattern, and the integration is a dozen lines of cpu at the frame rate. the
	conserved fields, the ledger and the drift meters are untouched, and the conformance
	suite pins that: the field after 400 frames is bit-identical with and without the body.
- skill: sampling every N frames needs a held force plus body inertia, nothing else. the
	sample is one frame plus N frames stale; measured (experiments/buoy.js, dam at 64^2,
	600 frames) the rest float line moves by a few percent of submersion between N=1 and
	N=8 (62.6%/64.1%/63.8%/67.8%) and the body starts to lag at N=16 and N=32 (end
	|vy| 7.9 and 10.5 cells/s instead of 0.1). default N=4 sits at the flat end of the curve.
- verified: the float line is `meanBlueInFootprint = buoyRho`, so it is set by the dye level
	and the hull density only, both tunables. measured at N=64: blue x0.25/0.5/1 -> submerged
	95%/82%/64%; hull rho 0.1/0.25/0.4 -> 32%/64%/83%. with more dye per cell than buoyRho the
	lift saturates and the body can ride the smeared dye and spray instead of the surface
	(blue x2 ends at 85% and cy=19.6, above the water). that is the documented approximation;
	buoyRho is the knob, not a bug to chase.
- [verified] the substep policy is a *quality* knob, not a stability one. with substepsAuto off
	(one step per frame, dt=1/60, K=60, dam at 64^2, 400 frames) the run stays finite, keeps
	zero negative cells and maxM 1.070, versus 0.998 with the auto policy; max|div u| is 1.70
	against 0.51. monotone transport comes from the per-cell outflow limiter in compute_flux
	(see the 0.2.0/0.3.0 notes), so ns=1 is a legitimate measurement configuration.
- pitfall: `new Float32Array(typedArray, byteOffset, length)` ignores the offset and length and
	copies the *whole* source (the typed-array branch of the constructor takes one argument).
	only an ArrayBuffer or a SharedArrayBuffer source honours them. a readback helper written as
	`new Float32Array(range, 0, n)` therefore reads fine against a real device (getMappedRange
	returns an ArrayBuffer) and throws or overflows against a mock that hands back a Float32Array.
	return the ArrayBuffer from a mock getMappedRange so this class of bug shows up in node.
- skill: for gpu plumbing without a browser, a recording mock device is enough to catch the
	bug classes that actually bite: missing pool buffers, wrong usage flags, bind groups with
	the wrong entry count, a dispatch that fires on the wrong frame, a copy with a bad size, and
	params that never reach the uniform buffer. see experiments/gpu-mock.js; it also lints that
	every `params.values[i]` index in every wgsl source is a declared slot in
	Fluid.core.config.paramsOff, which is otherwise invisible until a shader silently reads zero.
	it executes no shader, so it is not a numeric oracle - conformance.js and the cpu reference
	stay that.

## 2026-10-19 control panel and resize

- pitfall: a cfg -> dom sync loop that rewrites `input.value` fights a pointer drag.
  a range input commits on `change` (release), so cfg still holds the old value while
  the thumb moves; the next sync snaps the thumb back and the release can commit the
  old value, so the change appears to do nothing. track pointerdown/up/cancel/blur and
  skip the sync for the held element. [verified in a DOM-stub test, not a browser]
- skill: live readout on `input`, expensive work on `change`. a resize reallocates
  every texture (~20) and rebuilds ~30 bind groups; doing that per input event stalls
  the page under a moving pointer.
- skill: compute pipelines are size independent when dispatch sizes come from the
  uniform block, so a grid resize can keep them and rebuild only the pool and the
  bind groups. recompiling 18 shader modules plus the render pipeline was the visible
  hitch on every Grid N change.
- pitfall: a factory that returns `{pool,...}` as an object literal freezes that
  handle. after resize, `solver.pool` pointed at the destroyed pool. assign the new
  pool/flagsMirror onto the returned object inside resize.
- pitfall: assigning `canvas.width`/`height` clears the drawing surface right away, so
  a resize that waits for the next rAF flashes a blank frame. repaint (`renderOnly`)
  in the same task. [reasoned from the html canvas spec, not observed here]
- note: one substep at 60 Hz is not a "safe default off", it is a dt the scheme cannot
  resolve at 512^2 (cfl ratio 20+). the outflow limiter keeps it monotone, but the
  transport then follows the limiter, which reads as smeared/sticky liquid. the fixed
  floor default is 4 (dt 1/240).

## 2026-10-19 view-mode row

- skill: one `change` listener on the radio container plus one `pointerover` listener
	implements click, keyboard and hover. walk `event.target` up to the nearest `LABEL`
	and take `label.control`; that finds the wrapped input without building a selector
	string or an array per event. [verified in experiments/ui.js]
- pitfall: `pointerover` fires for the radio inside the label as well as the label and
	the text. stop at the first `LABEL` and return, and make the handler idempotent
	(`checked = true` plus a `setMode` that returns on the active id), or one enter
	becomes two updates. [verified in experiments/ui.js]
- pitfall: a programmatic `input.checked = true` does not fire `change`. a keyboard
	shortcut that only writes cfg leaves the radio stale, so the single writer has to
	sync the radio too (or the render loop has to, once per change).
- skill: gate hover switching by reading `matchMedia('(hover: hover) and (pointer:
	fine)').matches` at event time, not once at setup: a hybrid touch laptop changes its
	primary pointer. keep the MediaQueryList, never call `matchMedia` per event.
- skill: hide radios with `opacity: 0` and a 1 px box, not `display: none`, to keep
	native radio keyboard behavior (tab to the checked one, arrows move the group); draw
	the focus ring on the label with `:has(input:focus-visible)`.
- skill: a ~100 line DOM stub is enough to test this contract in node. it needs radio
	group semantics (checking one unchecks the group), bubbled events, `label.control`,
	and a `window` alias. see experiments/ui.js; no jsdom dependency.

## 2026-10-19 thermal (0.4.0)

- skill: carry cumulative source sums in unused field channels (therm.b/a).
  donor transport is conservative for any scalar, so the field totals of the
  accumulators are the exact source ledger after any amount of advection —
  the GPU gets exact heater accounting from its normal stats reduction, no
  atomics or extra readback. the JS ledger is snapshot-corrected by the
  readback delta, like the paint ledger. [verified in experiments/convection.js
  on cpu; gpu side is plumbing-checked only]
- pitfall: therm.g (T) must be derived, never transported. advecting T as a
  scalar and also transporting E double-counts the thermodynamics: after one
  step T and E disagree and buoyancy/conduction read a stale field. derive
  after transport and after sources; the extra ping-pong is cheap.
- note: conduction by the plan formula (coef*dT*dt of energy against capacity
  m*c) has diffusion time L^2*C/coef — ~5000 s across 32 cells at coef=0.2.
  it is a local smoother at game scales; advection does the heat transport.
  test relaxation with a large coefficient and test the clamp separately.
- note: beta=4e-4 with g=300 gives ~0.1 cells/s^2 per 10 K — convection took
  ~9 s to reach 2 cells/s. beta=2e-3 onsets in ~0.4 s and peaks ~8 cells/s.
  beta only bites where T actually varies, so raising the default was safe
  for every pre-thermal scenario (all uniform T_amb).
- pitfall: f32 field-store rounding under sustained boundary jets is biased,
  not a random walk: mass drift grew ~linearly at 2e-9/step over 5000 steps
  (1.1e-5) while the ledger-accounted E stayed at 3e-7. do not chase this as
  a conservation bug without first running the conduction-off control (4e-8).

## 2026-10-19 pressure solve / multigrid (0.6.0 t4)

- skill: the donor-cell CFL bound is the *sum* of a cell's outgoing face
  fluxes, which for a divergence-free field is at most 2*maxVel*dt. Target
  `2*maxVel*dt <= cflTarget` for substeps, not `maxVel*dt`: the old per-face
  target picked half the substeps and left the positivity limiter clipping at
  every peak, which is a local mass sink (dam N=256 maxM 3.9).
- skill: two-level multigrid wins when the coarse level can solve the modes it
  owns. At N=512 a coarse N/2 grid is still too fine for 60 GS sweeps and S=2
  needs 3 cycles to barely beat S=4/2; S=4 (Nc=128) with 2 cycles had 5.9x
  lower residual than plain K=60 at 0.53x the wall time and 300x lower maxDiv
  end-to-end at N=128. Scale the coarse stride with N, not down to the finest
  possible level.
- pitfall: `mgZero` is bound to the *coarse* params, where `values[2]` is
  already `Nc+2`. Dividing by the stride a second time zeroes only the
  top-left quarter and leaves stale corrections elsewhere. Any kernel that
  shares the params layout with a different grid must document which size it
  sees.
- pitfall: a fixed coarsening stride compiled into the shader (2x2 loops,
  `(x-0.5)*0.5`) silently drops 3/4 of the domain when `mgCoarsen` changes.
  Pass S in a params slot and use it in the loop bounds, the coarse extent and
  the prolongation mapping; allocate from the same value.
- skill: the velocity cap has to follow the substep budget, otherwise a high
  fixed cap lets the scene move faster than the transported cells
  (`velocityCap = min(vMax, 0.45*ns)/frameDt`). With auto substeps the cap only
  grows after the count is chosen, so it bounds growth per frame instead of
  fighting the sim.

## 2026-10-20 phase core (0.5.0 t1-t3) and the browser-GPU runner

- pitfall: the caloric curve classifies by E/m only, so a cold thin cell
  (scenario ambient m=0.3 at T_amb) is a *liquid-branch* state with rho* =
  rho_liq. Wiring the plan's two-sided S straight to rho* collapses every
  ambient toward rho_liq in one step. S must be two-sided only where vapor is
  present (phi_g > 0) and one-sided anti-overpack for phi_g = 0 — the 0.2.0
  minimal form. Thin cold cells are the air tier of the density hierarchy and
  carry no expansion target. Documented in 0.5.0 section 3.1/3.2.
- pitfall: therm.b/a are the donor-transported heat-source accumulators the
  stats reduction reads back (0.4.0). The 0.5.0 plan wanted them for phi_l/phi_g;
  derived state must not be transported. sfield (S, phi_l, phi_g, -) is the
  scratch home: written by phase_eos, read one step behind by poisson_rhs.
- skill: scenario energy must come from the *inverse* caloric curve
  (energyFromT) once the branch layout exists. Writing E = m*c*T after a curve
  change silently re-temperatures every scenario (the convection stratification
  would start at 356 K, not 293 K).
- skill: a pairwise conduction clamp proves nothing for four neighbours with
  unequal capacities (c_eff differs across phases) and says nothing about
  latent plateaus. Cap each face additionally by 0.25 of the *enthalpy distance
  to thermal equilibrium* (eqDist): both endpoints compute the same cap from
  pre-update state, the four-face sum stays inside the temperature hull, and no
  face traverses a latent interval. Measured: the budget is exactly the pairwise
  clamp for equal capacities (no behavior change in the old tests) and binds for
  unequal ones.
- skill: serve the browser diagnostics page with run_browser_tests.py (COOP/
  COEP); a file:// or plain-server capture reports quantum=0.1 ms and
  crossOriginIsolated=false, which makes queue/wait timings far coarser than the
  page thinks it is measuring.

## 2026-10-20 Ampere MG vs RB A/B capture and 0.5.0 WGSL parity

- [verified on Nvidia Ampere] `2026-10-20-02_20-GPU-browser-tests.log` (MG default)
  vs `2026-10-20-02_21-GPU-browser-tests,mg=0.log` (Red-Black `?mg=0`) at `N=512`,
  `600` frames: `crossOriginIsolated=true` improved `performance.now()` quantum
  20x (`0.100000 ms -> 0.005000 ms`). `blobs` `maxM = 0.899917` (`over = 0`) is the
  exact initial `0.3 + 0.6 * fall` peak; `draw` `maxM = 1.2165` (`over = 213..215`)
  comes from additive `paintDye` (`+0.35` onto `0.8665`), which relaxes back to
  `over = 0, maxM ~ 0.54` at `tauPhase = 0.05 s` after painting finishes. MG cuts
  final `maxDiv` 177x on `blobs` (`2.922 -> 0.0165`) and 108x on `draw`
  (`2.765 -> 0.0256`); at equal `ns=8` on `draw`, MG runs `24.54 ms` (`40.3 fps`,
  `295` dispatches/step) vs RB `23.34 ms` (`42.4 fps`, `120` dispatches/step).
- pitfall: unsymmetric `S = (m - rho*) / (m * tau_phase)` diverges toward `-inf`
  when a thin headspace cell (`m << rho_liq`) heats into the boiling plateau
  (`phi_g = 0+`, where `rho*` starts at `rho_liq`), saturating `-S_max` and driving
  a spurious implosion while heating. Using `max(m, rho*) * tau_phase` for
  `phi_g > 0` keeps `|S| < 1/tau_phase = 20 < S_max = 30` symmetrically on both
  expansion and condensation (`sSat = 0` across 5,000 steps of `cycle`).
- pitfall: coarse-grid multigrid blocking on all-Neumann Poisson solves must mark a
  coarse cell blocked only when `openCount == 0` (averaging mass over open fine
  cells) and mirror `pressC[hc]` across blocked coarse neighbours during bilinear
  prolongation. Using `anyBlocked` (`||`) plus Dirichlet `0` across blocked coarse
  cells injects spurious boundary gradients and blows up to `NaN` in 18 steps on
  enclosed frozen cavities; open-count + Neumann mirror gives 224x / 154x residual
  speedup and zero `NaN` on curved frozen vaults (`arch`).
- skill: core WebGPU caps `maxStorageTexturesPerShaderStage = 4`. Binding inputs as
  unfiltered textures (`textureLoad`) leaves all 4 write storage slots free for
  `phaseEos` (`thermDst`, `flagsDst`, `sfieldDst`, `accumDst`). Running the
  post-transport chain twice (`phaseA -> freezeTmp -> conduction -> heatSources ->
  phaseB -> freezeMain`) leaves live state in canonical textures, preserves
  `state.t` and `state.v` ping-pong parity, and makes `encodePrimePhase` strictly
  idempotent (`f(f(x)) = f(x)`) for `reset()`, `paintDye()`, and `paintWall()`.
- pitfall: any Node suite that exercises `Fluid.core.solver` with `useMultigrid: true`
  must `require("src/core/multigrid.js")`, or the CPU solver silently falls back to
  plain Red-Black GS while the GPU runs two-level MG (the pre-existing root cause of
  `dam` parity failure in `gpu-parity.mjs`, and why `testCondense` previously saw
  `maxVel = 0.5057` from unconverged `RB K=10` instead of the converged Poisson
  value `0.4724` / `0.4700`).
- pitfall: `prepareScenario(cfg)` must reset non-scenario overrides (`thermalSources`,
  `TCeil`, `gravityY`, `heatPower`, `coolTop`) only when `cfg._prevScenario !==
  cfg.scenario`, or explicit overrides passed to `solver.create(cfg)` on a fresh
  config object get wiped on initial reset.

## 2026-10-20 Ampere hardware sign-off of the phase scenarios (0.5.0 a10)

- pitfall: `fMom=0` in a browser capture is most likely correct, not a dead counter.
  `frozenMomentum` charges only on the freeze *transition* (`frozen && prev frozen
  flag == 0`), sampling pre-zeroing cell velocity, and the current scenarios freeze
  quasi-statically: every transition happens at init (velocity exactly 0) and the
  solid inventory only shrinks afterwards (verified on the CPU oracle, rain/cycle,
  10 s sim; at N=512 rain ends with 42436 solid cells vs 42496 initialized cold
  rows). A capture can show 42k solid cells and still legitimately read `fMom=0`.
  Check sol-vs-initialized-rows arithmetic before suspecting the stats path; note
  the GPU fMom path has never carried a nonzero value end-to-end — `gpu-parity.mjs`
  compares phaseCounts + latentChargedE only, and latE sharing the accum texture +
  reduction is currently the only live proof of that plumbing.
- skill: per-substep cost `queue_wait / ns` is ~2.1-2.6 ms at 512^2 (MG, K=60) on
  Ampere across all five phase scenarios, so frame rate is driven by the
  auto-substep count (ns=12-15 in boiling scenarios), not by CPU encode
  (<= 0.29 ms) or per-pass overhead. For performance work use `jet` as the watch
  case (33.3 fps, closest to the a10 bar) and quote cost per substep, not per
  frame, when comparing captures with different CFL histories.

## 2026-10-20 solver-work studies (0.6.0 t3) and timestamp profiling (t1)

- pitfall: `diag.divAfter` / sampled `maxDiv` cannot judge projection quality in
  boiling scenarios: the projection targets `div u = S`, so post-projection
  divergence floors at `max|S|` (~19/s in jet) and every solver setting from
  RB K20 to the full default reads the same. Use the true Poisson residual
  `mg.residualNorm()/divBefore` on the oracle, or an S=0 scenario (dam), to
  compare solver variants (experiments/mg-work.js vs mg-work2.js).
- skill: count GPU solver work two ways before arguing about it: fine-sweep
  equivalents (cells) and dispatch count. The shipping default (2 cycles,
  pre4/post8, K=60 coarse) is 36.5 feq but 295 dispatches per projection, 240
  of them tiny coarse red/black dispatches (129^2) — dispatch-bound, not
  cell-bound. Coarse K is therefore the first knob: at 2 cycles K60->20 keeps
  dam div at 0.29 vs 0.17 (round 2) while cutting 160 dispatches; at 1 cycle
  K is free (fine-level error dominates). The second V-cycle is worth ~20x
  residual on dam but indistinguishable on jet/cycle (trueRes 3.3e-2/4.0e-2 for
  cur, c2K20 and c1K20 alike) — cycle count is a quality knob for liquid-only
  flow, a pure cost in boiling flow.
- pitfall: under-smoothed multigrid is *unstable*, not just inaccurate:
  1 cycle with pre2/post4 blows up to NaN in `cycle` (step 40 at N=128,
  dt=1/240); NaN fields read as maxM=0/div=0 because NaN loses every max
  reduction, so check finiteness of totals, not the divergence line. The
  coarse correction injects a component that post-smoothing must damp at
  coefficient jumps: pre2/post8, pre3/post6 and 2 cycles are all stable.
  Keep pre+post >= 10 sweeps around the correction at 1 cycle.
- skill: timestampQueries only exist at pass boundaries, so per-group GPU
  timing means splitting the frame's single compute pass at group boundaries.
  Keep the split encode *dispatch-identical* to the fast path (same segment
  functions) and verify mechanically: diff the ordered setPipeline/setBindGroup/
  dispatch sequence against git; profiled fps is then only mildly pessimistic
  and group proportions are trustworthy.
