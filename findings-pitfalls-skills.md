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

## 2026-10-20 projection defaults adopted (0.6.0 t11): coarse K 20, cycles stay 2

- skill: `divAfter/divBefore` (the a2 ratio) and `mg.residualNorm()/divBefore`
  (the 4.1 "true residual") are not interchangeable. They are roughly related
  by dt, but each is a max over cells and the maxima need not coincide: on dam
  at N=512, rel=5.5e-3 while dt*true=2.3e-4 (24x). Use `true` for solver
  quality, `rel` for the a2 bar, and quote the dt with either.
- skill: one sweep count cannot serve both solver paths. A fine sweep covers
  16x the cells of a coarse sweep (S=4), so `pressureIters` is now the mg=0
  red-black count (60) and `mgCoarseIters` the coarse count per V-cycle (20).
  `Fluid.core.config.solverIters(cfg)` / `setSolverIters(cfg, v)` are the only
  places that choose, and every UI/query/report path goes through them.
- pitfall: an auto-substep policy rewrites a solver A/B even at a fixed URL.
  Three N=512 runs of the same scenario set observed ns 6/7/15/12, 6/7/12/12 and
  6/11/10/12 (blobs/dam/cycle/jet), i.e. three different dam/cycle dts. Only
  blobs and jet kept dt across all three. For a solve comparison use
  `&auto=0&ns=4`, and always print ns next to the ms.
- pitfall: comparing two solver settings by warming each up on its own setting
  measures where the chaotic flow ended up, not the solve. Coarse K=60 vs 20 on
  dam reads +69 % divergence at N=128 (180+90 medians, 4.56e-4 vs 7.71e-4) and
  +193 % at N=256 (120+60, 6.40e-4 vs 1.88e-3), while one step from a *shared*
  warm state reads +16 % at N=128 and +8 % at N=512. Compare solver variants
  with experiments/mg-samestate.js (one step from a shared state); use
  mg-defaults.js only for "what does a long run under this setting look like".
  The same effect moved the 64^2 conformance dam bar from maxDiv 0.5 to 1.38
  under the new default: a different splash, not a weaker solve.
- pitfall: sampled peak `maxDiv` in a 600-frame hardware capture is a
  trajectory too. The same N=512 blobs scene read 1.43 / 3.16 / 4.67 in the
  three runs (K60c2 / K20c2 / K20c1) with no quality information in the
  difference. The hardware is for time; the oracle is for quality.
- skill: coarse K 60 -> 20 costs +16 % peak divergence on dam and blobs at the
  same state (+8 % at N=512), leaves the true Poisson residual unchanged
  (+0.01 % / +0.07 %), is free in boiling flow, and buys 25-48 % frame rate at
  N=512 (press 2.5/1.7/1.6/1.7 -> 1.8/1.2/1.1/1.2 ms/substep). Over a long run
  the divergence ratio grows more (+69 % at N=128, +193 % at N=256) because the
  trajectories separate. K=40 is the conservative rung: +4-6 % same-state and
  rel below the a2 1e-3 bar at N=64/128/256 (K=20 crosses it at N=256,
  1.9e-3); it has no hardware row yet, so it stays a `&k=40` fallback.
- skill: 1 cycle is residual-neutral in boiling flow (jet +6.6 %, cycle -0.3 %
  of the true residual at the same state) but 21-22x worse peak divergence on
  dam/blobs at N=128 (rel 7.7e-3 / 4.3e-3, above the 1e-3 bar), and the damage
  is phase- and grid-dependent (+23 % at N=512 during the early splash). The
  multi-step rows show the same split (dam rel 9.0e-3 vs 4.6e-4 at N=128; jet
  2.83e-1 unchanged, i.e. the S floor). If it is ever adopted it must be
  conditional on the scenario, never a global default.

## 2026-10-20 fixed-policy solver baseline and the residual channel (0.6.0 t2)

- skill: price a knob's floor before tuning it. the four pinned N=512 rows fit
  `press = 0.026 + cycles*(0.466 + 0.0071*K)` ms/substep (K40 predicted within 1 %),
  i.e. one coarse sweep is 7.1 us while the rest of a V-cycle is 466 us and does not
  depend on K at all. that kills the K ladder: 60 -> 20 removes 31 % of press and
  20 -> 0 would remove only 23 % more, while dropping one cycle removes 49 %. a knob
  that multiplies one additive term cannot beat the term it does not touch; a third
  level shrinks both, so it is the next candidate.
- pitfall: never carry a delta over from a row with a different step policy. the t11
  note that the a2-conservative K=40 fallback "costs ~12 fps" came from auto-policy
  rows; at pinned ns=4 the same switch costs 0.28 ms/substep = 17 % of the fps
  (104 -> 87) and 18-19 % in the auto policy. an unpinned price tag turns a
  measurement into a preference.
- pitfall: the hardware report's sampled `maxDiv` cannot rank a projection. across the
  pinned rows it does rise monotonically with fewer sweeps (blobs 1.06/1.51/2.81/6.27
  for K60/K40/K20/K20-1cyc), but it is flat in `cycle` (31.7/31.2/30.3, the S floor),
  it *falls* in `jet` (25.2 -> 22.3), and it jumps 4x for one cycle in `cycle` where
  the oracle says the residual is unchanged. direction only, never magnitude.
- skill: give the hardware its own quality number instead of arguing from a blind one.
  `max|b - L p|` and `max|b|` are stats slots 32/33, printed as
  `poisson: res= rhs= true=`. that `true` is the residual of the system just solved,
  captured before transport. slots 34/35 are the post-step reading (`drift=`). a
  capture dated before this split stored the drift in 32/33, so its `true` peak in
  boiling flow is not a solver ranking. note the units: `true` is ~1e-2 (residual
  over rhs) and is NOT the a2 `rel` (~1e-3, divAfter over divBefore); `rel` floors
  at max|S| in boiling flow, and `drift` can exceed 1 without the solve having failed.
- skill: one definition per operator, even across a language boundary. the poisson
  stencil now exists twice in total (module-scope `faceWeight`/`computeResidual` in
  src/core/multigrid.js, and `POISSON_HELPERS` in src/gpu/shaders.js interpolated into
  both kernels that need it) instead of once per caller. two tricks made that possible:
  a shared wgsl fragment has to be interpolated *after* the binding block (it
  references those bindings), and the five pressure reads stay in the caller because a
  storage texture load takes no level argument while a sampled one does; passing
  `out=null` to computeResidual keeps the diagnostic from needing a 1 MB scratch field.
  folding the max into the loop that already visits every cell turned `residualNorm()`
  into a return value, and re-running the oracle (bit-identical rows) is what proved
  the refactor free.
- skill: generate capture tables from the log files, and let the grouping be the
  guard. experiments/gpu-log-table.js pools rows by (scenario, dt, ns, N) and prints
  "no peer: not comparable" for a lone row, so the three auto rows that differ only by
  policy (dam ns 7/8/7, cycle ns 12/9/15) refuse to be compared instead of quietly
  becoming a fake speedup.
- pitfall: the browser report's header `gravity=300` is the pre-scenario cfg value. the
  `cycle` scenario scales gravity with the box (600 at N=1024), so compare scenario rows
  across N, never headers.

## 2026-10-21 boiling residual spike is coefficient drift

- pitfall: a frame-end `max|b - L p| / max|b|` is not the solve. `b` and `p` belong to
  the projection; mass and flags after that are whatever transport and phase just wrote.
  One emptied neighbor (weight ~1/rhoVac = 1000) or one cell that melts next to a stale
  p=0 turns a tight solve into a residual of 1e5. Measured: uniform-density field, drop
  one cell to rhoVac, residual x499; melt a blocked cell inside p=5e4, residual 0 -> 2e5;
  dam N=64 step 10, end/solve=556 while solve/rhs=1.4e-4; dam N=128, max drift/solve=767
  and drift/rhs=0.36. Cycle N=128 reproduces the hardware event at the same sim time:
  step 1170 (t=4.875 s), post-step residual 1.2e4 times the solve and 668 times the
  solve rhs, against the device's poissonTrue=26 at frame 301 (t=5.0 s, absRes=5.0e4,
  rhs=1.9e3; that 26 is post-step residual over the post-step rhs). It grows with K (68/49/26/15 for
  K60/K40/K20/1-cycle) because pressure amplitude grows, and the auto-policy rows
  (smaller dt, less mass moved per step) peak at 0.18-0.39. Do not add sweeps to chase it.
  Device confirmation, same preset after the split (`2026-10-21-07_19`): drift at frame
  301 is 26.11078936008509 and driftRes is 49994.73828125, bit-identical to the old
  absRes; end-of-run drift equals the old true (0.009715138446710316) and the solve
  is 0.00877. The sampled true peak is the frame-1 cold start (0.141), not the boil.
- skill: capture the residual in the pressure pass, before transport, and keep the
  post-step reading under another name. GPU slots 32/33 are the solve (`true`,
  residual over the solve rhs). Slots 34/35 are the drift, and the printed ratio is
  residual over the post-step rhs — the formula older logs called `true`. CPU
  `diag.poissonRes` is stored at the end of `project()`; `residualNorm()` after
  `step()` is the drift residual. The two reductions need their own bind groups:
  `layout: "auto"` pipelines do not share a bind group even when the entries match.
- pitfall: older oracle rows (`mg-work*`, `mg-samestate`, `mg-defaults`) call
  `residualNorm()` after `step()`, so their `true` column includes this drift. On a
  same-state single step it is usually small next to rhs; a multi-step peak in boiling
  flow is not a solver ranking. it also hides a real regression: dam 1-cycle vs 2-cycle
  is drift 1.05x and solve 18x (`2026-10-21-mg-cycle-gate.log`).
- pitfall: a same-state step cannot clear a solver that changes the warm start.
  cycle at 1 cycle is 1.00x on one step inherited from a 2-cycle warmup, and 27x on
  the solve (37x peakDivA) once it has been 1-cycle for its own 180 steps. jet stays
  1.00x on that trajectory. do not promote "residual-neutral in cycle" from the old row.
- skill: jet ships 1 V-cycle. the fixed-ns pair (`2026-10-21-08_02`) has settled
  true 0.0100 vs 0.0115, maxDiv 23.7 vs 27.9, maxVel unchanged, 158 vs 105 fps.
  the frame-1 true peak is 1.84x and is the cold start, not the ranking. the gate
  is `productCycleGate` in prepareScenario; `&cycles=` pins an A/B. oracle scripts
  that do not set the flag keep the cycle count they named. do not extend it.
- skill: a capture can be in the tree and still be "missing" from the notes. the
  auto-policy jet row was quoted from a previous session's reading because the
  pasted copy of the capture had been dropped by a rebase, while the raw capture
  (`2026-10-21-20_46-GPU-browser-tests.log`) was sitting in `experiments/logs/`
  with every conserved quantity bit-identical to the pasted one. before writing
  "the row does not exist", diff the numbers against the captures that are there:
  identical totals, pressure span and phase counts mean the same run, and only
  the fps line differs.

## 2026-10-21 mac-cormack velocity advection (0.6.0 t5)

- pitfall: a "retained energy improves by >= 25 %" bar is unreachable at a short window,
  and it looks like a scheme failure when it is arithmetic. mac/sl can never exceed
  1/keep_sl, so a window where semi-lagrangian has only lost 16 % has a ceiling of 1.193x.
  measured: 1.149x at 1 s (ceiling 1.193x), 1.553x at 4 s (ceiling 1.777x). any ratio bar
  has to name the vortex age; the plan and the oracle now print the ceiling next to the
  ratio so a "below the bar" row cannot be misread.
- pitfall: mac-cormack without its limiter creates energy at the wall, not in the interior.
  the backward trace from the face next to a blocked face reads that face's 0, so the
  correction extrapolates past the boundary value: 7.7 % per step on a constant field,
  compounding. without a projection, peak speed goes 80 -> 105 -> 270 -> 400 -> 441 over
  4 s, retained KE 1.31, vClamp tripped 256 times. with the limiter
  the peak speed never leaves 80.00 and vClamp stays 0. the projection hides this (it
  removes the overshoot every step, so limited and raw agree to six digits), which is
  exactly why the limiter must not be dropped because "the product path has a projection".
- pitfall: a vortex run without the pressure projection is not a vortex decay measurement.
  the momentum equation becomes inviscid Burgers: the vortex steepens instead of decaying,
  peak vorticity rises 3.6-23x, maxDiv reaches 5e3-2e5 and the retained KE ranks the
  schemes by how much shock dissipation they add. keep it only as an attribution row and
  label it (`experiments/vortex-decay.js --no-proj` prints no decision block).
- skill: time the slice, not just the step, when the step is dominated by something else.
  the cpu step reads +22 % for mac-cormack and looks like it fails a 15 % bar, but the
  advection slice is +178 % and is only 9 % of that step; the rest is projection. on the
  device the vel group is 0.033 ms of a 1.448 ms substep sum, so neither number is the
  gate. two `process.hrtime.bigint()` calls around the advection call, accumulated, is
  enough — no profiler plumbing needed in an oracle.
- skill: prove a refactor free by re-running an existing oracle against its retained log
  instead of writing a new test. extracting `sampleU`/`sampleV`/`traceU`/`traceV` out of
  `advectVel` reproduced `2026-10-21-residual-spike.log` digit for digit (1.000e+3 ->
  4.995e+5, dam worst end/solve 5.56e+2 at step 10), which is stronger evidence than a
  unit test of the sampler and costs one command.
- skill: a fixed-point check separates bias from sharpening in two lines. on a constant
  field a correct mac-cormack is bit-identical to semi-lagrangian in the interior
  (correction = (src - back)/2 = 0), and any difference at the wall is the boundary
  stencil, not the scheme. assert both: interior diff === 0, wall overshoot <= 0 limited
  and > 0 raw.
- pitfall: an orphan experiment file reads as a second opinion when it is a dead draft.
  `experiments/residual-spikes.js` was referenced by nothing (`run-all.js`, the plans and
  the worklogs all name `residual-spike.js`) and could not have produced a row: its last
  line was a leaked heredoc terminator, so it exited 1 after doing all the work. before
  keeping a measurement script, grep the plans and `run-all.js` for its name, and run it
  once — a script nobody runs is where that kind of breakage hides.

## 2026-10-21 vorticity confinement (0.6.0 t6)

- pitfall: a wall face is not a resolved shear. the curl of a velocity field that
  is zero on a wall reads a huge vorticity there, and the confinement force
  (F = eps*h*(N x omega)) reads |omega|'s gradient, so a masked wall turns the
  force into a boundary push. the fix that keeps the physics is a free-slip
  mirror, so the wall contributes no vorticity and |omega| stays smooth to
  the boundary (the current form — centred velocities that drop blocked
  faces, blocked neighbours mirroring the cell — is in the 2026-10-22
  section below).
- pitfall: a force that concentrates vorticity trades shape for energy. the
  numbers first recorded here ranked the force on the vorticity it amplified;
  that guidance is superseded by the 2026-10-22 section below (rank on the
  profile error, errL2, and read keep ratios above 1 as distortion).
- pitfall: an energy source needs a different watchdog than a scheme. the
  advection rows are held to "no new extremum" (the limiter's job); a
  confinement row is supposed to push the peak vorticity up, so its watchdog is
  the absolute velocity clamp (`diag.velClamp`), which has to stay 0, plus the
  residual divergence, which is where the force turns into noise (eps=32 gives
  35x the shipping row).
- skill: a force is a row, not a scheme. it drops into the existing oracle as
  one more row over the same stirred vortex, with the strength as the only new
  knob, and the shipping path stays untouched when the knob is 0 — the scratch
  is allocated on first use, so the default does not even pay for the memory.
- skill: the wall convention is the part of a force that a port has to
  reproduce exactly. the free-slip mirror is one line in the cpu solver and the
  whole meaning of the row depends on it; write it into the port's task, not
  only into the code.

## 2026-10-22 the curl that was not a curl (0.6.0 t5/t6 correction, t16)

- pitfall: on this MAC layout (u at the west face, v at the south face of each
  cell), `(vT-vB)-(uR-uL)` is dv/dy-du/dx, not the curl. it is exactly 0 for
  solid-body rotation and for pure shear, and on the gaussian vortex it peaks
  on the diagonal at 4.55 where the curl peaks in the core at 12.34 (analytic
  2*A/sigma = 12.365). it shipped in both the confinement force and the oracle
  that ranked it, so the two agreed on a quantity that was never vorticity:
  t6's "eps=8 keeps 3.44x the peak vorticity" and t5's "1.517x retained peak
  vorticity" are superseded (corrected: mac 1.058x at 4 s; confinement ranked
  on errL2, optimum eps=2). the KE/divergence/peak-speed/clamp rows of those
  logs were computed correctly and stand, digit for digit.
- skill: pin a derivative to analytic fixed points where no scheme is
  involved: solid-body rotation (curl 2*Omega), pure shear u=k*y (curl -k),
  uniform flow (0), the gaussian core value (2*A/sigma). four asserts in
  `experiments/vortex-decay.js` (`curlFixedPoints`) now fail the run if the
  stencil regresses. any field the oracle measures should have one analytic
  value it is held to.
- skill: one definition, not two agreeing copies. `measure()` now calls the
  solver's `vorticityAt` instead of re-deriving the curl in the experiment.
  the curl is a diagnostic, not part of the advection scheme under test, so
  sharing it removes the drift channel that let the bug validate itself; the
  analytic asserts are what keep the shared definition honest.
- the centred MAC curl is the average of the cell's four corner curls:
  dv/dx - du/dy from neighbouring cell-centred velocities. cell-centred
  velocity drops a blocked face instead of averaging in its stored 0, and a
  blocked neighbour mirrors the cell itself (free slip) — the same convention
  as the |omega| gradient. this is the form a WGSL port (t17) must reproduce.
- pitfall: keep-ratios above 1 are distortion, not retention. with the true
  curl every confinement eps drives peak vorticity past its initial value
  (keep 1.28 at eps=0.5 to 4.32 at eps=8), so ranking on "peak vorticity kept"
  puts the most distorted row first. the vortex is a steady solution, so
  errL2 against the initial field is the honest rank row; it is also what
  separates mac (0.0855) from every confinement-on-sl row (best 0.161).
- skill: a WGSL kernel can be numerically checked without a GPU. the npm
  `wgsl_reflect` package (installed outside the repo, like gpu-parity's
  `webgpu`) parses WGSL properly and interprets compute dispatches on the
  CPU: `experiments/wgsl-exec.mjs` runs the real mac-cormack shader pair and
  compares to the oracle (<= 2.6e-4 on fields where the limiter fires). that
  is the check to run before asking for a device capture of a new kernel —
  the recording mock proves plumbing only and brace-counts the WGSL.
- pitfall: parallel edits to the same file can silently drop one another.
  three simultaneous edits to `src/core/solver.js` reported success and two
  were lost; the test run caught it (a missing export). edit one file at a
  time and re-read the diff before trusting a batch.

## 2026-10-23 Ampere MacCormack rows, t17 rejection on MacCormack, and t8 pass fusion

- finding: WGSL MacCormack (`t16`, `mac-fixed-k20`, `2026-10-23-04_53-GPU-browser-tests.log`,
  Ampere, `N=512`, 600f, `K=20`, pinned `ns=4`) adds `+16..19 us/substep` to the
  `vel` group (`0.034 -> 0.050` blobs, `0.035 -> 0.054` dam, `0.025/0.031 -> 0.048`
  cycle, `0.029 -> 0.047 ms` jet) and moves overall fps by `-0.6 %` to `+0.6 %`
  (`104.5 / 105.0 / 106.1 / 158.6 fps`), clearing the `<= 15 %` equal-`ns` bar by
  more than 10x with exact conservation (`[ok]` on all five channels) and equal or
  lower settled `true` Poisson residual (`3.89e-5 / 1.68e-4 / 7.19e-3 / 2.97e-3`).
  the `rgba32float` trace-helper scratch idea is closed: `18 us` is ~1 % of a
  2-cycle substep, so doubling scratch bandwidth to save four velocity taps in
  pass 2 has no headroom.
- pitfall: an equal-`ns` A/B proves kernel cost, not auto-CFL frame rate, when a
  higher-order advection scheme preserves larger peak speeds. at pinned `ns=4`,
  MacCormack keeps `dam` end `maxVel` at `108.8` vs `67.0` on semi-Lagrangian
  (and on the CPU auto policy at `N=64`, raises mean `ns` `+19.6 %` on `blobs`
  and `+9.8 %` on `dam`). because `dam` (`67.9 fps`, `ns=7`) and `cycle` (`71.6 fps`,
  `ns=12`, `CFL=0.7819` in `2026-10-23-04_54`) sit within one substep step-up of
  the 60 fps gate, `cfg.advectMac` stays `0` until the 4-scenario auto-policy
  capture (`python3 run_gpu_mac_k20.py`) is recorded.
- pitfall: every preset that needs a hardware capture should have a dedicated
  one-click `run_gpu_*.py` wrapper. `run_gpu_capture.py` with no arguments runs
  `k20` (`mac=0`), which is why `04_54` re-captured the `mac=0` auto baseline
  (`71.8 / 67.9 / 71.6 / 76.8 fps`) instead of `mac=1` auto.
- pitfall: `browser-tests.html` does not report a continuous per-frame GPU trace.
  It accumulates the unique timestamp-query results read during stats polls
  (typically ~20 samples across a 600-frame run), while `frames/s` measures all
  600 frames. `04_54` blobs has `step=2.400 ms` and `press/step=75.3 %` in that
  sparse aggregate at `71.8 fps`; the log has no per-sample series, so do not
  assign the timing to frames 541-600 or infer a late-run P-state change from it.
  Use `frames/s` for full-run throughput and the profile line for sampled group
  proportions only.
- finding: vorticity confinement (`t17`) is rejected when stacked on MacCormack
  (`2026-10-23-vortex-decay-mac-conf.log`, `N=128`, 4 s, shipping projection).
  confinement compensates first-order numerical viscosity by injecting core spin;
  once MacCormack cuts `errL2` from `0.2737` to `0.0855`, `eps=1, 2` make `errL2`
  worse (`0.1096 / 0.2312`), inflate `maxDiv` `6.7x / 20x`, push peak speed past
  the initial `80` (`84.19 / 95.54`), and trip `macLimiter` `44x / 86x`. even
  `eps=0.5` overshoots initial peak speed (`80.63` at 3 s) with `5.7x` `maxDiv`,
  and `eps=0.25` spins core vorticity to `1.81x` with `3.3x` `maxDiv` for `-0.023`
  `errL2`. rejecting the WGSL port avoids two extra velocity passes and a scratch
  texture.
- skill: when two `override`-specialized compute pipelines share prebuilt bind
  groups (like `projRed`/`projBlack` sharing `projLayout`, or `t8`'s `phaseEos`/
  `phaseSources` sharing `groups.phaseB`), declare an explicit
  `device.createBindGroupLayout(...)`. under `layout: "auto"`, WebGPU treats each
  pipeline's derived bind group layout as distinct even when compiled from the
  same shader module, which passes a recording mock and fails real-device
  validation.
- pitfall: `wgsl_reflect`'s `WgslExec` CPU interpreter silently drops `.r/.g/.b/.a`
  component writes on `var q: vec4<f32>` (`q.r = ...` leaves `q` unchanged), while
  supporting `.r/.g/.b/.a` reads and `vec4<f32>(...)` value construction. writing
  `q = vec4<f32>(nextE, nextT, ...)` is cleaner WGSL on hardware and lets
  `experiments/wgsl-exec.mjs` verify bit-exactness (`maxDiff === 0`) of fused vs
  unfused WGSL passes before any device run.

- pitfall: a +3% fps change is only 2.91% frame-time saving. For a >3%
  frame-time adoption gate use `1 - baselineFps / candidateFps`. Match MG cycle
  count and MacCormack setting as well as ns; gpu-log-table's dt/ns grouping
  alone does not guarantee comparable work. Repeated paired captures matter
  when observed savings are comparable to historical baseline variation.
- finding: the follow-up Ampere t8 captures (`2026-10-23-06_38*`, `06_40`,
  `06_41`, `06_42`, `06_43`) keep `cfg.fusePasses = 0`. Two fixed runs per
  setting (ns=4) average 1.37/2.02/1.16/2.51 % frame-time savings for
  blobs/dam/cycle/jet; no scenario clears 3 %. The single auto pair gives
  0.93/1.32/3.79/2.78 %; only cycle clears 3 %, once. All matched on/off
  state and diagnostic lines are identical; the auto jet E drift warning
  (`1.216e-6`) is identical too. See `archive/0.6.0-mac-captures-t8-worklog.md`.
- pitfall: these browser captures report fps across all 600 measured frames and
  do not discard a warm-up window. Their GPU timestamp line aggregates sparse
  samples read at stats polls (20/21 samples here), not a last-60-frame window;
  the blobs averages vary sharply across runs, but the logs do not expose the
  individual samples needed to localize that variation. The fixed fusion runs
  were grouped by setting, not interleaved. Do not label them warmed, interleaved
  A/B evidence; if chasing the one auto cycle win, add an explicit warm-up/discard
  period and alternate several runs. This cycle-specific repeat is deferred in
  `deferred/0.6.0-t8-cycle-fusion-validation-plan.md`; continue with the next
  planned 0.6.0 optimization instead of adding the warm-up harness now.

## 2026-10-24 engine api, probe table, and the interpreter's thread cap

- finding: a game-facing probe query does not need a field mirror or a second
  readback path. 32 fixed slots live in one storage buffer; four 8-thread
  workgroups (one per eight slots) read `mass/therm/vel/press` at the rounded cell
  and write eight raw numbers back. `T` and `phase` are derived on the cpu with the same
  `phaseInto` the oracle uses, so the caloric EOS has exactly one definition and
  the kernel never duplicates it. Cost is one dispatch + one 2112-byte copy on
  `cfg.probeEvery` frames, sharing the single-in-flight rule with stats/buoy.
- pitfall: readback tables must be cpu-owned per slot. Compacting the table on
  remove changes ids a game is holding; keep stable slots with an active flag and
  let the kernel return early for inactive ones.
- pitfall: `wgsl_reflect`'s `WgslExec` CPU interpreter silently executes only a
  prefix of a large workgroup. `@workgroup_size(32)` with `dispatchWorkgroups(1)`
  wrote 3 threads; 16 wrote 1, 64 wrote 6, while 8 wrote all 8. Kernels that must
  be executable in `experiments/wgsl-exec.mjs` should use small workgroups and
  more of them: the probe kernel is `@workgroup_size(8)` with
  `dispatchWorkgroups(MAX/8)`, and the test compares every measured slot against
  `probes.sampleCpu` with zero difference. Do not "fix" a failing exec row by
  lowering the tolerance before checking how many invocations actually ran.
- finding: one pacing gate for the whole app. `Fluid.core.pacing` (moved out of
  `src/app/`) now lives under the engine: `update()` accrues, takes at most one
  fixed step, refuses to submit while `busy`, and marks the resize repaint as a
  submission too. The lab page keeps only DOM work. A second loop in the page
  would have re-derived the substep policy - the exact duplication that kept
  "off means one step" alive in a test once already.
- pitfall: `setGrid` destroys and rebuilds the pool, so every buffer starts
  zeroed - including the probe table. Re-upload the cpu-owned table from `reset()`
  or probes silently stop after a resize while the api still reports them live.
  A resize test that only checks `pool.N` will not see this; read the new buffer.
- pitfall: testing a demo page against a stub engine proves the page calls *some*
  API, not the real one. Boot `examples/demo.js` on `Fluid.engine` over the
  recording mock and fake only `Fluid.gpu.device.init`; that is what caught the
  missing facade export and the stale probe table above.
- pitfall: two writers for `viewMode` (a DOM row and an API setter) disagree.
  `bindViewMode(fn)` lets the page's control stay the single writer; the engine
  setter delegates instead of assigning. Without a bound writer the engine owns
  `cfg.viewMode`. `experiments/engine.js` pins both branches.
- finding: input listeners must be removable. `attachInput`/`detachInput` and
  `destroy()` are only honest if every `addEventListener` went through one list;
  the harness counts window+canvas listeners before and after and requires equality.
- pitfall: a 20-cycle create/destroy loop is the only cheap leak test available
  without a GPU. The shared recording mock (`experiments/mock-device.js`) tracks
  live buffers/textures, so "pool destroyed" is an assertion rather than a comment.

## 2026-10-24 preallocated emitter slots, exact GPU ledger closure, and t8b input review

- skill: state-dependent GPU emitters (`drain` clamping at vacuum `m -> 0`,
  `heat`/`cold`/`freeze` clamping at target enthalpy floors/ceilings) cannot be
  pre-summed on the CPU like additive `paintDye` patches because the CPU does not
  have the live substep `mass`/`therm` textures. Close the conservation ledger on
  the GPU without per-substep CPU stalls by splitting channels:
  1. Sensible/latent thermal emitters (`heat`, `cold`, `freeze`) accumulate
     `eNew - eOld` / `eOld - eNew` directly into per-cell `therm.b` / `therm.a`
     (Sterbenz-exact in `f32` when `0.5 * vOld <= vNew <= 2 * vOld`), which
     `statsReduceA`/`statsReduceB` already Kahan-sum into `heatInE`/`heatOutE`.
  2. Mass/tracer emitters (`dye`, `drain`) run a single-workgroup `emitterTally`
     pass (`@workgroup_size(64)`) over only active `dye`/`drain` bounding boxes
     (with first-emitter bounding-box ownership so overlapping discs evaluate each
     cell once) and Kahan-accumulate the exact `vNew - vOld` / `vOld - vNew` into
     an 80-byte `pool.emitterLedger` (`[0..9]` sums + `[10..19]` compensations).
  3. `forces` reads `(emitterLedger[0] - emitterLedger[5]) * invN * invN + rhoBase`
     every substep so hydrostatic `rhoMean` stays live during multi-substep mass
     injection/drain without waiting for a `statsEvery` readback.
  4. `statsReduceB` copies `emitterLedger[0..9]` into `stats[36..45]`. At copy
     launch, `gpu/solver.js` snapshots `pendingEmitNet = emitNetKnown`, and on
     `mapAsync` completion adds `(emitNetGpu - pendingEmitNet)` to the snapshot's
     `expected` totals while advancing live `expected` by `(emitNetGpu - emitNetKnown)`.
     Interleaved CPU `paintDye` patches between readback launch and completion
     remain exact and are never double-counted or overwritten.
- finding: dispatch gating keeps idle emitters completely free on the GPU. Header
  counters `[momentumCount, scalarCount, massDrainCount]` in `src/core/emitters.js`
  gate `forces` (loop bound 0 when `momentumCount == 0`), skip `applyEmitters`
  entirely when `scalarCount == 0`, and skip `emitterTally` when `massDrainCount == 0`
  (thermal-only `heat`/`cold`/`freeze` rigs add 1 dispatch/substep, not 2).
- pitfall: in Boussinesq convection (`gravityY = 300, beta = 0.002`), liquid cooled
  from `293 K` to `TMelt = 273 K` gains `-12 cells/s^2` downward acceleration and
  sinks out of a suspended `freeze` emitter disc in `< 0.7 s`. Crossing the fusion
  plateau (`cLiq * 20 + Lf = 84 + 334 = 418` energy/cell) requires either enough
  power density (`value >= ~900`) to extract `Lf` within the parcel's residence
  time or placing the sink against a boundary/interface. `cold` floors enthalpy
  strictly above `E2 = cSol * TMelt + Lf` (`TMelt + 2` by default) so even high
  power never crosses into slush/ice, whereas `freeze` floors enthalpy below
  `E1 = cSol * TMelt` (`TMelt - 25` by default) so `phaseEos` drives `phi_l -> 0`.
- pitfall: `wgsl_reflect` (`WgslParser` / `WgslExec`) rejects two valid WGSL
  constructs in compute kernels: vector swizzle assignments (`q.r = ...`) and
  `continue`/`break` inside `for (...)` headers (use structured `if/else` or
  `loop { if (...) { break; } }`). In addition, `WgslExec` does not coroutine-yield
  at `workgroupBarrier()` and leaves `@builtin(local_invocation_index)` bogus on 1D
  `@workgroup_size(64)` dispatches, so shared-memory tree reductions are parse-checked
  with `WgslReflect` while cell/slot kernels (`applyEmitters`, `forces`,
  `statsReduceB`, `probeSample`) are executed cell-for-cell against the CPU oracle.
- pitfall: `0.7.0 t8b` review caught four subtle keyboard-binding edge cases in
  `src/app/input.js` and `src/api/engine.js`: `e.target` on `window`/`document`
  lacks `.closest()`, `blur` on a custom `keyTarget !== window` did not clear
  `state.keys`, `r`/`n`/`[`/`]` bypassed `engine.reset()`/`engine.setGrid()`
  (skipping canvas resize and pacing reset), and `sim.bindKeys()` with an omitted
  argument returned `null` instead of defaulting to `window`. All four are fixed
  and pinned in `experiments/input.js` and `experiments/engine.js`.

## 2026-10-24 moving collider scalar-flux consistency: eliminating diagonal stripes and Poisson blowup

- pitfall: when moving a rigid collider (`flags.r & 2u`), `colliderVel` prescribes
  rigid-body velocity `(vx, vy)` on faces bordering the collider, and the pressure
  Poisson solve diverts fluid out through open faces so `div u = 0`. Zeroing scalar
  transport flux in `computeFlux` (`rawFx`/`rawFy`, `isBlocked3`) across `(flags.r |
  flags.b) > 0u` starved the cell ahead of an advancing collider of inflow while open
  faces drained 100% of its mass in one step (`rho -> rhoVac = 0.001`). The `1/rho`
  Poisson face weight spiked to `1000`, creating an `8000:1` condition jump and
  diagonal checkerboard stripes (`poissonRes > 1700`). Fix: in `computeFlux` (`rawFx`/
  `rawFy` and WGSL `isBlocked3`), zero flux only on hard-blocked cells (static walls
  `(f.r & 1u) != 0u || f.r == 255u` and frozen solids `f.b > 0u`), allowing active
  moving collider faces (`f.r & 2u`) to transport scalar mass/energy consistently with
  `div u = 0` (`poissonRes < 63`, exact `1e-6` global conservation, and stationary
  colliders unchanged since `vel = 0`).

## 2026-10-25 0.7.0 t6 save/load, t9 phase-change events, and t12 lifecycle suite

- finding (`0.7.0 t6` binary snapshot): `src/core/save.js` packs a versioned
  little-endian `ArrayBuffer` (`256 B` header, 52 physics/policy `CFG_KEYS` as `f64`,
  96-element CPU state as `f64`, 24-key conservation ledger as `f64`, 3 tables
  `probes`/`emitters`+`emitterLedger`/`colliders`+`buoy`, and 9 tight field blocks
  `mass`, `therm`, `vel`, `press`, `flags`, `flagsMirror`, `sfield`, `accum`,
  `mgPress`). On GPU, `serialize()` records 8 `copyTextureToBuffer` commands (with
  256-byte row alignment) plus `copyBufferToBuffer` of `pool.emitterLedger` into a
  single transient `MAP_READ | COPY_DST` staging buffer in one submit and destroys
  the staging buffer inside `finally {}`.
- pitfall: storing the CPU state block (`pSpanSmooth`, `time`, `rhoMean`, `emitInjKnown`)
  in a `Float32Array` truncated `pSpanSmooth` from `f64` to `f32` on `serialize()`,
  causing a 1-ULP difference in `cfg.PMin`/`cfg.PMax` after the next stats readback
  (`pSpanSmooth * 0.85 + span * 0.15`). Keeping the 96-element CPU state block in
  `Float64Array` preserves exact IEEE-754 `f64` state across mid-run snapshots so
  100-step replays are byte-for-byte identical (`0` byte diffs across all `349,344 B`
  at `N = 64`). Also snapshot `cfgState(cfg)` synchronously before `await mapAsync`
  so slider edits while readback is in flight cannot mislabel the captured fields.
- finding (`0.7.0 t9` `onPhaseChange`): `phaseChangeRow` (`@workgroup_size(8)`, one
  thread per interior row `row = 0..N-1`) and `phaseChangeCompact` (`@workgroup_size(1)`)
  compare `flags.g` against `pool.phasePrev` (`255u` = unseeded/static wall, `0..5u` =
  phase), update `phasePrev` in place, sum the 6x6 phase-transition matrix, and compact
  up to 64 cell records `(x, y, from, to, phiL, phiG, S, T)` into `560` floats
  (`2,240 B`) with zero atomics and zero `workgroupBarrier` calls (100% executable in
  `WgslExec` with `0` diff against `Fluid.core.phaseEvents.scanCpu`). Gated off when no
  listener is registered (`0` dispatches, `0` copies), and delivered outside submit in
  `pollReadback()` only when `batch.total > 0`.


## 2026-10-25 page script graphs and attributing allocations to a file

- pitfall (page wiring): an HTML runner's `<script src>` list is a dependency list, and
  nothing else validates it. When `src/app/loop.js` became `Fluid.engine.update()` and
  `src/core/pacing.js`, `experiments/api-test.html` kept the dead tag and lost the new
  module; the page loaded without an error until `Fluid.engine.create` reached
  `Fluid.core.pacing.create`, so the only browser-side lifecycle/leak suite was dead and
  every headless suite stayed green. Node suites `require()` the modules in their own order,
  so they cannot see the page's graph. Check it headlessly: resolve every `src`, then require
  that each `Fluid.<ns>.<name>` used by a loaded script is *defined* by another loaded script
  (`definers` scanned over `src/**/*.js`), and load the scripts in document order inside a
  `vm` sandbox to catch load-time throws. Do not check "definer before user": classic scripts
  defer their bodies, so `solver.js` may reference `Fluid.core.emitters` from a function
  before `emitters.js` runs. `experiments/pages.js` is 17 checks over 4 runners in 0.05 s.
- skill (allocation audit without a profiler): wrap the array constructors, count while a
  flag is set, and attribute each hit through `new Error().stack` by taking the first frame
  outside the harness file. That separates `src/` from the mock device without instrumenting
  either. Two positive controls matter more than the number itself: the audited window must
  advance the simulation (or a pacing gate silently skips the frames and "0 allocations" is
  vacuous), and the harness allocations must still be counted (`1260` here), or a broken
  wrapper passes. Scope the flag to the synchronous call: async readback continuations run
  after the `await`, so they are excluded automatically. `subarray`/`slice` and object
  literals never run a constructor and are outside this technique.
- finding: 200 engine frames of `update`/`render`/`stats`/`sample` plus 100 CPU reference
  steps allocate `0` typed arrays or ArrayBuffers in `src/`. The remaining allocations in the
  frame's async tail are one view per `mapAsync` (`new Float32Array(getMappedRange())`),
  which is a mapped-range lifetime question rather than a hot-path one.

## 2026-10-25 one readback buffer, and what getMappedRange really hands back

- pitfall (WebGPU mapping): `GPUBuffer.getMappedRange()` returns an `ArrayBuffer`, repeated
  calls within one map return the same one per (offset,size), and `unmap()` **detaches** it.
  A typed-array view over it is therefore valid only until the next `unmap()`; caching
  `new Float32Array(getMappedRange())` across frames does not save the allocation, it keeps a
  detached view that throws on first read. The per-map floor is one driver `ArrayBuffer` plus
  one view. Saving the allocation means not reading the bytes at all, so the only real lever
  is the **number of maps**. Consolidating the stats, buoy, probe and phase-event readbacks
  into one buffer with a 256-byte-aligned slot each cut a step+poll cycle from 3 maps and 3
  views to 1 and 1. Copy offsets must stay 4-byte aligned (`copyBufferToBuffer`) and map
  offsets 8-byte aligned (`mapAsync`); 256-byte slots satisfy both with room to spare.
- pitfall (a mock kinder than the spec is a test that cannot fail): the recording mock device
  returned its backing `ArrayBuffer` from `getMappedRange()` and never detached it, so a
  cross-frame cached view would have passed every headless suite and broken on the first real
  driver. Make a readback mock hand out a snapshot per `mapAsync`, one `ArrayBuffer` per
  (offset,size), and detach them all on `unmap` (`structuredClone(ab,{transfer:[ab]})` is the
  portable detach in node). Then an illegal view fails in the same place a driver would.
- skill (counting an async path): the `a6` probe counts only a synchronous window, which by
  construction excludes the `mapAsync` continuation where the view is built. The same probe
  works across awaits if the flag is simply left on for the whole step+poll cycle; on the
  mock, 40 cycles then report `40 maps, 40x Float32Array @ src/gpu/solver.js`. Keep the
  positive control (`maps` must be ≥ cycles) so a poll that stops mapping cannot pass as
  "0 allocations".
- finding (one in-flight flag is enough): storage buffers keep their newest values, so
  skipping a copy while a map is open costs one frame of staleness, never a sample. That is
  why four per-channel pending/in-flight pairs could collapse into one `readMask` bitmask and
  one `readInFlight` flag without weakening any staleness guarantee - but capture the
  expected totals at *encode* time, next to the copy, or stats can be paired with another
  frame's ledger.
