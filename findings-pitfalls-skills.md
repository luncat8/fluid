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
