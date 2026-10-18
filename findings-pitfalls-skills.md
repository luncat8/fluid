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
