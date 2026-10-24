"use strict";
// headless harness for the gpu side: a recording mock device, enough to drive
// Fluid.gpu.solver through real app frames. it validates plumbing (pipelines, bind
// group bindings, params slots, dispatch cadence, async readback wiring) and lints the
// wgsl sources; it does NOT execute shaders, so it is not a numeric oracle - the cpu
// reference plus conformance.js is that.
// run: node experiments/gpu-mock.js

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/solver.js"));
require(path.join(root, "src/core/probes.js"));
require(path.join(root, "src/core/emitters.js"));
require(path.join(root, "src/core/buoy.js"));
require(path.join(root, "src/gpu/shaders.js"));
require(path.join(root, "src/gpu/fields.js"));
require(path.join(root, "src/gpu/passes.js"));
require(path.join(root, "src/gpu/solver.js"));

let failures = 0;
function check(name, ok, detail) {
	console.log((ok ? "PASS " : "FAIL ") + name + (detail ? "  " + detail : ""));
	if (!ok) failures++;
}

// ---- mock device: shared recording mock, never executes shaders ----
const mock = require("./mock-device.js");
// what the mock "kernel" leaves in the sample buffer: half a footprint of blue,
// a rightward flow. the body must read this back as sub=0.5 and neutral lift.
const BUOY_PATTERN = new Float32Array([8, 32, 32, 32, 0, 0.5]);
const PATTERNS = { buoy: BUOY_PATTERN };

// one app frame, the shared substep policy the app loop calls
function appFrame(cfg, solver, maxVel) {
	const frameDt = cfg.frameDt;
	const ns = Fluid.core.config.substepCount(cfg, maxVel);
	cfg.vClamp = Fluid.core.config.velocityCap(cfg, ns);
	const st = solver.getStats();
	solver.frame(frameDt / ns, ns, st.frame % cfg.statsEvery === 0);
	return ns;
}

// ---- 1. plumbing: pipelines, buffers, dispatch cadence, readback ----
async function testPlumbing() {
	const cfg = Object.assign({}, Fluid.core.config.defaults);
	cfg.N = 64; cfg.texW = 66; cfg.texH = 66; cfg.scenario = "dam";
	cfg._paused = false;
	const m = mock.create(PATTERNS);
	const sink = m.sink;
	const device = m.device;
	const canvas = { width: 64, height: 64 };
	const solver = Fluid.gpu.solver.create(device, canvas, mock.mockContext(), "bgra8unorm", cfg);
	const pool = solver.pool;
	check("gpu pool has the buoy sample buffer and its readback twin",
		pool.buoy && pool.buoyRead && pool.buoy.size === 128 && pool.buoyRead.usage === (1 | 8),
		"buoy=" + pool.buoy.size + "B usage=" + pool.buoy.usage + " read usage=" + pool.buoyRead.usage);
	check("the buoySample compute pipeline is compiled with the params struct prefix",
		sink.modules.some(m => m.label === "buoySample-mod" && /struct Params/.test(m.code) && /@workgroup_size\(64\)/.test(m.code)));
	const projectCode = sink.modules.find(m => m.label === "project-mod").code;
	check("WGSL pressure kernel retains opposite-color values between red/black sweeps",
		/var p = src;/.test(projectCode) && /\(u32\(x \+ y\) & 1u\) == COLOR/.test(projectCode),
		"color gate and source-copy initialization are both present");

	const buoyEvery = cfg.buoySampleEvery;
	const frames = 4 * buoyEvery;
	let sampleFrames = 0;
	for (let f = 0; f < frames; f++) {
		const before = sink.copies.length;
		appFrame(cfg, solver, 40);
		if (sink.copies.slice(before).some(c => c[0] === "buoy")) sampleFrames++;
		await solver.pollReadback();
	}
	check("the buoy field sample is requested every cfg.buoySampleEvery frames only",
		sampleFrames === 4, "sample copies=" + sampleFrames + " over " + frames + " frames, expected 4");
	check("the body marched every frame and ingested one readback per request",
		solver.buoy.state.frames === frames && solver.buoy.state.samples === 4,
		"marches=" + solver.buoy.state.frames + " samples=" + solver.buoy.state.samples);
	check("the readback reaches the body: half a footprint of blue reads as sub=0.5, neutral lift",
		Math.abs(solver.buoy.state.sub - 0.5) < 1e-6 && Math.abs(solver.buoy.state.blue - 0.25) < 1e-6 &&
		Math.abs(solver.buoy.state.ux - 1) < 1e-6 && solver.buoy.state.aY === 0,
		"sub=" + solver.buoy.state.sub.toFixed(3) + " blue=" + solver.buoy.state.blue.toFixed(3) +
		" ux=" + solver.buoy.state.ux.toFixed(3) + " aY=" + solver.buoy.state.aY);
	check("every dispatch set a bind group, and the sample pass binds four resources",
		sink.bindings.length === m.counts.dispatches && sink.bindings.includes(4),
		"dispatches=" + m.counts.dispatches + " bindGroup calls=" + sink.bindings.length);
	const buoyCopies = sink.copies.filter(c => c[0] === "buoy");
	check("the buoy sample buffer is copied back 24 bytes (6 floats) at a time",
		buoyCopies.length > 0 && buoyCopies.every(c => c[1] === "buoyRead" && c[2] === 24),
		"copies=" + buoyCopies.length + " sizes=" + [...new Set(buoyCopies.map(c => c[2]))].join(","));
	check("the field sample runs after the sim steps and before the display pass",
		(() => {
			const at = sink.pipeOrder.lastIndexOf("buoySample");
			return at > 0 && sink.pipeOrder[at + 1] === "toDisplay";
		})(), "order=" + sink.pipeOrder.slice(sink.pipeOrder.lastIndexOf("buoySample") - 2,
			sink.pipeOrder.lastIndexOf("buoySample") + 2).join(">"));
	const uni = new Float32Array(pool.params._buf);
	check("the uniform buffer holds the live rect and substep count, not just cfg",
		uni[5] === cfg.substeps && uni[88] === cfg.buoyX0 && uni[90] === cfg.buoyW &&
		uni[93] === 1 && Math.abs(uni[92] - cfg.buoySub) < 1e-6,
		"substeps=" + uni[5] + " rect=[" + [uni[88], uni[89], uni[90], uni[91]].join(",") + "] show=" + uni[93]);
	check("the force kernel receives the live mean density in params slot 27",
		uni[27] === Math.fround(cfg.rhoMean) && cfg.rhoMean > 0,
		"uniform=" + uni[27] + " cfg=" + cfg.rhoMean);
	check("paused redraw pushes params too, so toggles show without a sim step",
		(() => {
			const n = sink.writes.filter(w => w === "params").length;
			solver.renderOnly();
			return sink.writes.filter(w => w === "params").length === n + 1;
		})());

	// grid resize: pipelines are size independent, so only the pool and the bind
	// groups are rebuilt and the exported handles follow the new pool. recompiling
	// the wgsl set here was the hitch the grid slider hit.
	const modulesBefore = sink.modules.length;
	solver.resize(128);
	const orderBefore = sink.pipeOrder.length;
	appFrame(cfg, solver, 40);
	check("resize reuses the compiled pipelines and swaps the pool",
		sink.modules.length === modulesBefore && solver.pool !== pool && solver.pool.N === 128 &&
		solver.flagsMirror.length === 130 * 130 * 4 && canvas.width === 128 && sink.pipeOrder.length > orderBefore,
		"modules +" + (sink.modules.length - modulesBefore) + " N=" + solver.pool.N +
		" flags=" + solver.flagsMirror.length + " canvas=" + canvas.width);

	// multigrid cadence (0.6.0 t4): the coarse level must actually be encoded.
	// one mgCoarsen per substep (mass is fixed during the solve), and the
	// pre/restrict/coarse/prolong/post block repeated cfg.mgCycles times.
	// this is the only runtime proof the mg branch is wired, since the mock
	// cannot execute the shaders.
	{
		sink.pipeOrder.length = 0;
		solver.frame(cfg.frameDt / 2, 2, false);
		const order = sink.pipeOrder;
		const count = (name) => order.filter(p => p === name).length;
		check("multigrid: coarse passes are encoded and ordered in the step",
			count("mgCoarsen") === 2 && count("mgRestrict") === 2 * cfg.mgCycles &&
			count("mgProlong") === 2 * cfg.mgCycles &&
			order.indexOf("mgCoarsen") < order.indexOf("mgRestrict") &&
			order.indexOf("mgRestrict") < order.lastIndexOf("mgProlong"),
			"cycles=" + cfg.mgCycles + " coarsen=" + count("mgCoarsen") +
			" restrict=" + count("mgRestrict") + " prolong=" + count("mgProlong"));
		check("multigrid: the coarse solve uses its own bound params and press textures",
			solver.pool.mgS === cfg.mgCoarsen && solver.pool.mgNc === Math.floor(solver.pool.N / cfg.mgCoarsen) &&
			new Float32Array(solver.pool.mgParams._buf)[2] === solver.pool.mgTexW,
			"S=" + solver.pool.mgS + " Nc=" + solver.pool.mgNc + " texW=" + solver.pool.mgTexW);
		check("solve residual is not captured on a non-stats frame",
			count("poissonReduceA") === 0 && count("poissonReduceB") === 0,
			"reduceA=" + count("poissonReduceA"));
		sink.pipeOrder.length = 0;
		solver.frame(cfg.frameDt / 2, 2, true);
		const order2 = sink.pipeOrder;
		const count2 = (name) => order2.filter(p => p === name).length;
		const gradAt = order2.lastIndexOf("applyGrad");
		const redAt = order2.lastIndexOf("poissonReduceA");
		const massAt = order2.lastIndexOf("advectMass");
		check("solve residual is reduced once, after the last projection and before transport",
			count2("poissonReduceA") === 1 && count2("poissonReduceB") === 1 &&
			gradAt >= 0 && gradAt < redAt && redAt < massAt,
			"grad=" + gradAt + " reduce=" + redAt + " mass=" + massAt +
			" counts=" + count2("poissonReduceA") + "/" + count2("poissonReduceB"));
	}

	// mac-cormack cadence (0.6.0 t16): one extra semi-lagrangian dispatch into
	// the scratch plus the correction dispatch, per substep, between forces
	// and the poisson rhs. with the flag off neither may appear.
	{
		cfg.advectMac = 1;
		sink.pipeOrder.length = 0;
		solver.frame(cfg.frameDt / 2, 2, false);
		const order = sink.pipeOrder;
		const count = (name) => order.filter(p => p === name).length;
		check("mac-cormack: forward and correction passes are encoded in the vel segment",
			count("advectVelMac") === 2 && count("advectVel") === 2 &&
			order.indexOf("forces") < order.indexOf("advectVelMac") &&
			order.indexOf("advectVelMac") < order.indexOf("rhs"),
			"mac=" + count("advectVelMac") + " sl=" + count("advectVel"));
		cfg.advectMac = 0;
		sink.pipeOrder.length = 0;
		solver.frame(cfg.frameDt / 2, 2, false);
		check("mac-cormack: off by default, the shipping path is one advection dispatch",
			sink.pipeOrder.filter(p => p === "advectVelMac").length === 0 &&
			sink.pipeOrder.filter(p => p === "advectVel").length === 2,
			"mac=" + sink.pipeOrder.filter(p => p === "advectVelMac").length);
	}
	{
		const t0 = solver.state.t;
		solver.frame(cfg.frameDt / 3, 3, false);
		const tUnfused3 = solver.state.t;
		sink.pipeOrder.length = 0;
		cfg.fusePasses = 1;
		solver.frame(cfg.frameDt / 3, 3, false);
		const cnt = name => sink.pipeOrder.filter(p => p === name).length;
		const tFused3 = solver.state.t;
		check("t8 pass fusion: advectScalar and phaseSources replace 2 dispatches/substep and track therm parity",
			cnt("advectScalar") === 3 && cnt("advectMass") === 0 && cnt("advectTherm") === 0 &&
			cnt("heatSources") === 0 && cnt("phaseEos") === 3 && cnt("phaseSources") === 3 &&
			tUnfused3 === 1 - t0 && tFused3 === tUnfused3,
			"scalar=" + cnt("advectScalar") + " phaseSrc=" + cnt("phaseSources") +
			" t:" + t0 + "->" + tUnfused3 + "(unf3)->" + tFused3 + "(fus3)");
		cfg.fusePasses = 0;
	}

	// the body must go away with the scenario
	cfg.scenario = "blobs";
	solver.reset();
	check("no body outside the dam scenario, and the overlay switch follows",
		!solver.buoy.state.active && cfg.buoyShow === 0, "show=" + cfg.buoyShow);
	solver.destroy();

	// substep policy: the frame count follows the toggle, dt follows the count
	const m2 = mock.create(PATTERNS);
	const cfg2 = Object.assign({}, cfg, {N: 64, texW: 66, texH: 66, scenario: "dam", substepsAuto: false});
	const solver2 = Fluid.gpu.solver.create(m2.device, canvas, mock.mockContext(), "bgra8unorm", cfg2);
	const nsOff = appFrame(cfg2, solver2, 1e6);
	const cfg3 = Object.assign({}, cfg2, {substepsAuto: true});
	const solver3 = Fluid.gpu.solver.create(m2.device, canvas, mock.mockContext(), "bgra8unorm", cfg3);
	const nsOn = appFrame(cfg3, solver3, 1e6);
	check("app frame loop: fixed steps when the policy is off, capped substeps when on",
		nsOff === cfg2.substepsFixed && cfg2.substeps === nsOff && nsOn === cfg3.substepMax &&
		cfg3.dt === cfg3.frameDt / nsOn,
		"off ns=" + nsOff + "/" + cfg2.substepsFixed + " dt=" + cfg2.dt.toFixed(5) +
		" / on ns=" + nsOn + "/" + cfg3.substepMax + " dt=" + cfg3.dt.toFixed(5));
	solver2.destroy(); solver3.destroy();
	// every pool allocation is destroyed with the solver, and destroy is idempotent
	check("no buffer or texture leaks across the solvers (pools destroyed on destroy)",
		m.counts.buffersAlive === 0 && m.counts.texturesAlive === 0 &&
		m2.counts.buffersAlive === 0 && m2.counts.texturesAlive === 0,
		"alive buffers=" + (m.counts.buffersAlive + m2.counts.buffersAlive) +
		" textures=" + (m.counts.texturesAlive + m2.counts.texturesAlive) +
		" created=" + m.counts.buffers + "/" + m.counts.textures);
}

// ---- 2. params: cfg -> paramsArray -> the indices the wgsl reads ----
function testParams() {
	const cfg = Object.assign({}, Fluid.core.config.defaults);
	cfg.N = 512; cfg.texW = 514; cfg.texH = 514; cfg.scenario = "dam";
	Fluid.core.buoy.create(cfg);           // spawn writes the runtime keys
	const off = Fluid.core.config.paramsOff;
	const a = Fluid.core.config.makeParamsArray();
	cfg.buoyX0 = 151; cfg.buoyY0 = 373; cfg.buoyW = 16; cfg.buoyH = 64; cfg.buoySub = 0.5; cfg.buoyShow = 1;
	Fluid.core.config.writeParams(a, cfg);
	check("the buoy rectangle reaches the uniform block at the slots the wgsl reads",
		a[88] === 151 && a[89] === 373 && a[90] === 16 && a[91] === 64 && a[92] === 0.5 && a[93] === 1,
		"[" + a.slice(88, 94).join(",") + "]");
	check("the mac-cormack limiter reaches the uniform block (an unwritten slot would read as limiter off)",
		a[off.macLimiter] === cfg.macLimiter && a[off.macLimiter] === 1,
		"slot " + off.macLimiter + "=" + a[off.macLimiter]);
	check("the uniform block holds every declared slot: 128 floats, 512 bytes",
		a.length === 128 && Math.max(...Object.values(off)) < a.length);
	check("the 16x64 reference rectangle is exact on the reference grid",
		(() => {
			const s = Fluid.core.buoy.scaledSize(cfg);
			return s.w === 16 && s.h === 64;
		})(), JSON.stringify(Fluid.core.buoy.scaledSize(cfg)));
}

// ---- 3. wgsl lint: balanced source, declared param slots, bindings match the groups ----
function testWgsl() {
	const sh = Fluid.gpu.shaders;
	const names = Object.keys(sh).filter(k => k !== "paramsStruct");
	let balanced = true, declared = true, missing = [];
	for (const n of names) {
		const code = n === "renderVert" || n === "renderFrag" ? sh[n] : sh.paramsStruct + sh[n];
		let depth = { brace: 0, paren: 0, brack: 0 };
		for (const ch of code) {
			if (ch === "{") depth.brace++; else if (ch === "}") depth.brace--;
			else if (ch === "(") depth.paren++; else if (ch === ")") depth.paren--;
			else if (ch === "[") depth.brack++; else if (ch === "]") depth.brack--;
			if (depth.brace < 0 || depth.paren < 0 || depth.brack < 0) balanced = false;
		}
		if (depth.brace || depth.paren || depth.brack) balanced = false;
		if (n === "paramsStruct") continue;
		for (const m of sh[n].matchAll(/params\.values\[(\d+)\]/g)) {
			const idx = +m[1];
			if (!Object.values(Fluid.core.config.paramsOff).includes(idx)) { declared = false; missing.push(n + ":" + idx); }
		}
	}
	check("wgsl: " + names.length + " sources have balanced braces, parens and brackets", balanced);
	check("wgsl: every params.values[i] index is a declared slot in config.paramsOff", declared,
		missing.length ? "undeclared: " + missing.join(" ") : "all indices resolve");
	const buoy = sh.buoySample;
	check("wgsl: buoy_sample reads the mass texture, the velocity texture and one storage buffer",
		/binding\(0\) var<uniform> params/.test(buoy) && /binding\(1\) var massTex: texture_storage_2d<rgba32float,read>/.test(buoy) &&
		/binding\(2\) var velTex: texture_2d<f32>/.test(buoy) && /binding\(3\) var<storage,read_write> buoySample: array<f32>/.test(buoy));
	check("wgsl: buoy_sample is a single workgroup with a workgroup reduction",
		/@workgroup_size\(64\)/.test(buoy) && /workgroupBarrier\(\)/.test(buoy) && /dispatchWorkgroups\(1\)/.test(
			Fluid.gpu.passes.encodeBuoySample.toString()));
	check("wgsl: the display overlay reads the body rectangle through the same slots",
		/values\[88\]/.test(sh.toDisplay) && /values\[93\]/.test(sh.toDisplay) && /buoyOverlay\(/.test(sh.toDisplay));
	check("wgsl: solve residual writes slots 32/33, post-step drift writes 34/35",
		/stats\[32u\] = maxRes/.test(sh.poissonReduceB) && /stats\[33u\] = maxRhs/.test(sh.poissonReduceB) &&
		/stats\[34u\] = maxRes/.test(sh.statsReduceB) && !/stats\[32u\] = maxRes/.test(sh.statsReduceB));
}

// ---- 4. cpu sampler: the twin the kernel mirrors, on a known field ----
function testSampler() {
	const N = 16, texW = N + 2, texH = N + 2;
	const cfg = Object.assign({}, Fluid.core.config.defaults, {
		N, texW, texH, scenario: "dam", buoyRefW: 8, buoyRefH: 8, buoyU: 0.5, buoyV: 0.5
	});
	const buoy = Fluid.core.buoy.create(cfg);
	const mass = new Float32Array(texW * texH * 4);
	const vel = new Float32Array(texW * texH * 2);
	// blue only in the lower half of the body footprint, u = 2 everywhere
	const st = buoy.state;
	for (let y = 1; y < texH - 1; y++) {
		for (let x = 1; x < texW - 1; x++) {
			const i = (x + y * texW) * 4;
			mass[i + 3] = 1;
			if (x >= st.x0 && x < st.x0 + st.w && y >= st.y0 && y < st.y0 + st.h / 2) { mass[i + 2] = 0.5; }
			vel[(x + y * texW) * 2] = 2;
		}
	}
	const raw = new Float32Array(Fluid.core.buoy.SAMPLE_LEN);
	buoy.sampleCpu(mass, vel, texW, texH, raw);
	buoy.ingest(raw);
	const cells = st.w * st.h;
	check("cpu sampler: sums the rectangle the gpu kernel walks, half of it blue",
		raw[0] === 0.5 * cells / 2 && raw[2] === cells && raw[3] === 2 * cells,
		"blue=" + raw[0] + " count=" + raw[2] + " uSum=" + raw[3]);
	check("cpu sampler: half a footprint of blue gives half submersion and neutral lift",
		Math.abs(st.sub - 0.5) < 1e-6 && Math.abs(st.blue - 0.25) < 1e-6 && Math.abs(st.aY) < 1e-9,
		"sub=" + st.sub.toFixed(3) + " blue=" + st.blue.toFixed(3) + " aY=" + st.aY);
	check("cpu sampler: the body follows the flow horizontally by drag, not by teleport",
		st.ux === 2, "mean u=" + st.ux);
}

// ---- 4b. probe table: layout, cadence, readback, cpu oracle ----
// measured values the mock "kernel" leaves in slots 0 and 1: slot 0 is liquid at
// ~323 K with a rightward/upward flow, slot 1 is a cold vacuum cell.
function probePattern() {
	const p = new Float32Array(Fluid.core.probes.LEN);
	const write = (id, m, dr, dg, db, E, u, v, pr) => {
		const b = Fluid.core.probes.slotBase(id);
		p[b + 2] = m; p[b + 3] = dr; p[b + 4] = dg; p[b + 5] = db;
		p[b + 6] = E; p[b + 7] = u; p[b + 8] = v; p[b + 9] = pr;
	};
	write(0, 1, 0.1, 0.2, 0.3, 2.0 * 273 + 334 + 4.2 * 50, 3, -2, 12);
	write(1, 0.02, 0, 0, 0.01, 1, 0.5, 0.5, 0);
	return p;
}
function testProbeTable() {
	const P = Fluid.core.probes;
	const cfg = Object.assign({}, Fluid.core.config.defaults, {N: 16, texW: 18, texH: 18});
	const table = P.create();
	const out = P.makeSample();
	const ids = [];
	for (let i = 0; i < P.MAX; i++) ids.push(P.add(table, cfg, i % 16, 2));
	check("probe table: 32 stable slots, the 33rd add is refused",
		ids.every((id, i) => id === i) && P.add(table, cfg, 1, 1) === -1 && table.live === P.MAX,
		"ids=" + ids[0] + ".." + ids[ids.length - 1] + " live=" + table.live);
	check("probe table: upload carries the live count and the active flag the kernel reads",
		table.upload[0] === P.MAX && table.upload[P.slotBase(3) + P.P_ACTIVE] === 1 &&
		table.upload[P.slotBase(3) + P.P_X] === 3,
		"count=" + table.upload[0] + " x=" + table.upload[P.slotBase(3)]);
	check("probe table: clamped to the interior cell range on add and move",
		(() => {
			P.remove(table, 5);
			const id = P.add(table, cfg, -5, 1e6);
			const clamped = P.move(table, cfg, id, 1e6, -1) && table.slots[id].x === 16 && table.slots[id].y === 1;
			P.remove(table, id);
			return clamped && id === 5;
		})(), "removed slot is reused, coordinates clamp to [1,N]");
	// a fresh sample is refused until the first readback lands
	P.remove(table, 0);
	const id0 = P.add(table, cfg, 4, 4);
	check("probe sample: no value before the first readback, stale value drops on invalidate",
		P.sample(table, cfg, 10, id0, out) === false && out.valid === false,
		"valid=" + out.valid);
	const raw = new Float32Array(P.LEN);
	raw.set(probePattern());
	P.ingest(table, raw, 7);
	const ok = P.sample(table, cfg, 11, id0, out);
	check("probe sample: readback values, age and the caloric EOS derivation reach the caller",
		ok && out.valid && out.m === 1 && Math.abs(out.dr - 0.1) < 1e-6 && out.E === 2.0 * 273 + 334 + 4.2 * 50 &&
		out.u === 3 && out.v === -2 && out.p === 12 && out.age === 4 &&
		out.phase === Fluid.core.solver.PHASE_LIQUID && Math.abs(out.T - 323) < 1e-4,
		"m=" + out.m + " T=" + out.T.toFixed(2) + " phase=" + out.phase + " age=" + out.age);
	P.invalidate(table);
	check("probe sample: a reset or resize invalidates the completed sample",
		P.sample(table, cfg, 12, id0, out) === false && out.valid === false);

	// cpu oracle: same clamps and the same derivation on plain fields
	const P2 = P.create();
	const id = P.add(P2, cfg, 9, 12);
	const mass = new Float32Array(18 * 18 * 4);
	const therm = new Float32Array(18 * 18 * 4);
	const vel = new Float32Array(18 * 18 * 2);
	const press = new Float32Array(18 * 18);
	const i4 = (9 + 12 * 18) * 4, i2 = (9 + 12 * 18) * 2;
	mass[i4] = 0.2; mass[i4 + 1] = 0.4; mass[i4 + 2] = 0.6; mass[i4 + 3] = 1;
	therm[i4] = 2.0 * 273 + 334 + 4.2 * 20;
	vel[i2] = 7; vel[i2 + 1] = -3; press[9 + 12 * 18] = 44;
	check("probe cpu oracle: reads the probe cell, not the texel origin",
		P.sampleCpu(P2, cfg, mass, therm, vel, press, 18, 18, id, out) &&
		out.m === 1 && Math.abs(out.dr - 0.2) < 1e-6 && Math.abs(out.db - 0.6) < 1e-6 &&
		out.u === 7 && out.v === -3 && out.p === 44,
		"m=" + out.m + " db=" + out.db + " u=" + out.u + " p=" + out.p);
	const edge = P.add(P2, cfg, 16, 16);
	const e4 = (16 + 16 * 18) * 4;
	mass[e4 + 3] = 1;
	therm[e4] = 2.0 * 273 + 334 + 4.2 * 20;
	check("probe cpu oracle: the same interior clamp as the kernel, T/phase from the shared EOS",
		P.sampleCpu(P2, cfg, mass, therm, vel, press, 18, 18, edge, out) &&
		out.x === 16 && out.phase === Fluid.core.solver.PHASE_LIQUID,
		"x=" + out.x + " T=" + out.T.toFixed(2));
}

async function testProbePlumbing() {
	const N = 32;
	const cfg = Object.assign({}, Fluid.core.config.defaults, {
		N, texW: N + 2, texH: N + 2, scenario: "blobs", probeEvery: 4
	});
	const m = mock.create({probes: probePattern()});
	const canvas = {width: N, height: N};
	const solver = Fluid.gpu.solver.create(m.device, canvas, mock.mockContext(), "bgra8unorm", cfg);
	check("probe pool: one storage table and its map-read twin, sized by core/probes",
		solver.pool.probes.size === Fluid.core.probes.LEN * 4 &&
		solver.pool.probesRead.size === Fluid.core.probes.LEN * 4,
		"bytes=" + solver.pool.probes.size);
	const out = Fluid.core.probes.makeSample();
	const id0 = solver.addProbe(10, 20);
	const id1 = solver.addProbe(30, 40);
	const upload = new Float32Array(solver.pool.probes._buf);
	check("probe upload: add writes count, position and active flag before any dispatch",
		id0 === 0 && id1 === 1 && upload[0] === 2 &&
		upload[Fluid.core.probes.slotBase(0) + Fluid.core.probes.P_X] === 10 &&
		upload[Fluid.core.probes.slotBase(0) + Fluid.core.probes.P_Y] === 20 &&
		upload[Fluid.core.probes.slotBase(1) + Fluid.core.probes.P_ACTIVE] === 1,
		"count=" + upload[0]);
	check("probe sample: refused before the first readback", solver.sampleProbe(0, out) === false);

	let copies = 0;
	for (let f = 0; f < 2 * cfg.probeEvery; f++) {
		const before = m.sink.copies.length;
		solver.frame(cfg.frameDt / 4, 4, false);
		if (m.sink.copies.slice(before).some(c => c[0] === "probes")) copies++;
		await solver.pollReadback();
	}
	check("probe readback: one copy every cfg.probeEvery frames, 2112 bytes, never per substep",
		copies === 2 && m.sink.copies.filter(c => c[0] === "probes").every(c => c[2] === Fluid.core.probes.LEN * 4),
		"copies=" + copies + "/" + (2 * cfg.probeEvery) + " frames");
	const got = solver.sampleProbe(0, out);
	check("probe readback reaches the table: values, age, and the derived phase",
		got && out.valid && out.m === 1 && Math.abs(out.dr - 0.1) < 1e-6 && out.u === 3 && out.p === 12 &&
		out.phase === Fluid.core.solver.PHASE_LIQUID && out.age <= cfg.probeEvery,
		"m=" + out.m + " T=" + out.T.toFixed(2) + " phase=" + out.phase + " age=" + out.age);
	solver.moveProbe(0, 32, 31);
	check("probe move: rewrites only that slot, count unchanged",
		upload[0] === 2 && upload[Fluid.core.probes.slotBase(0) + Fluid.core.probes.P_X] === 32 &&
		upload[Fluid.core.probes.slotBase(0) + Fluid.core.probes.P_Y] === 31 &&
		upload[Fluid.core.probes.slotBase(1) + Fluid.core.probes.P_ACTIVE] === 1,
		"x=" + upload[Fluid.core.probes.slotBase(0)]);
	solver.removeProbe(1);
	check("probe remove: clears the active flag, drops the count, the id stops sampling",
		upload[0] === 1 && upload[Fluid.core.probes.slotBase(1) + Fluid.core.probes.P_ACTIVE] === 0 &&
		solver.sampleProbe(1, out) === false,
		"count=" + upload[0]);
	solver.removeProbe(0);
	check("probe cadence stops with an empty table (no dispatch, no readback)",
		(() => {
			const before = m.sink.copies.length;
			const order = m.sink.pipeOrder.length;
			solver.frame(cfg.frameDt / 4, 4, false);
			return m.sink.copies.length === before &&
				m.sink.pipeOrder.slice(order).indexOf("probeSample") < 0;
		})(), "live=" + solver.probes.live);
	const wgsl = Fluid.gpu.shaders.probeSample;
	check("wgsl: probe_sample covers the 32 slots in 4 groups of 8, reading every queried field",
		/@workgroup_size\(8\)/.test(wgsl) && /idx >= 32u/.test(wgsl) &&
		/binding\(1\) var massTex: texture_2d<f32>/.test(wgsl) &&
		/binding\(2\) var thermTex: texture_2d<f32>/.test(wgsl) &&
		/binding\(5\) var<storage,read_write> probes: array<f32>/.test(wgsl) &&
		/dispatchWorkgroups\(Fluid.core.probes.MAX\/8\)/.test(Fluid.gpu.passes.encodeProbeSample.toString()),
		"workgroup=8 x4, six bindings, one dispatch");
	solver.destroy();
	check("probe pool is released with the solver", m.counts.buffersAlive === 0 && m.counts.texturesAlive === 0);
}

async function testPhasePlumbing() {
	require("../src/gpu/view.js");
	require("../src/app/hud.js");
	require("../src/app/viewmenu.js");
	require("../src/app/input.js");
	require("../src/app/controls.js");
	const m = mock.create(PATTERNS);
	const sink = m.sink;
	const device = m.device;
	const N = 64;
	const cfg = Object.assign({}, Fluid.core.config.defaults, {
		N, texW: N + 2, texH: N + 2, scenario: "ice", buoyEnabled: 0
	});
	const canvas = { width: N, height: N };
	const solver = Fluid.gpu.solver.create(device, canvas, mock.mockContext(), "bgra8unorm", cfg);
	check("phase pool: 40-float partial stride, stats-len buffer, rgba32float sfield, rg32float accum",
		solver.pool.statsPart.size === solver.pool.numWorkgroups * 40 * 4 &&
		solver.pool.stats.size === Fluid.core.stats.STATS_LEN * 4 &&
		solver.pool.sfield.format === "rgba32float" &&
		solver.pool.accum.format === "rg32float",
		"statsPart=" + solver.pool.statsPart.size + "B");
	check("phase pipelines: phaseEos and freezeVel compiled and dispatched during reset priming",
		sink.modules.some(m => m.label === "phaseEos-mod") &&
		sink.modules.some(m => m.label === "freezeVel-mod") &&
		sink.pipeOrder.includes("phaseEos") && sink.pipeOrder.includes("freezeVel") &&
		cfg.chargeLatent === 1);
	const st = solver.getStats();
	check("reset seeds CPU phase census matching scenario cells (ice + liquid = 64^2)",
		st.phaseCounts[1] > 0 && st.phaseCounts[3] > 0 &&
		st.phaseCounts.reduce((a, b) => a + b, 0) === N * N,
		"counts=" + st.phaseCounts.join("/"));
	// Minimal DOM stub for HUD + input 'o' overlay shortcut
	const listeners = {};
	const origWin = globalThis.window, origDoc = globalThis.document;
	globalThis.document = {
		getElementById: () => ({ textContent: "", addEventListener() {} })
	};
	globalThis.window = {
		addEventListener(type, fn) { (listeners[type] || (listeners[type] = [])).push(fn); }
	};
	const hud = Fluid.app.hud.create(solver, cfg, null);
	const snap = hud.buildSnapshot();
	check("HUD snapshot includes 6-phase census, latent/momentum ledgers, and sSat counter",
		/phases vac=0 sol=\d+ melt=0 liq=\d+ boil=0 gas=0 \| latentE=0 fMom=0 sSat=0/.test(snap),
		snap.split("\n")[2]);
	const mockCanvas = {
		addEventListener() {}, setPointerCapture() {}, hasPointerCapture: () => false,
		releasePointerCapture() {}, focus() {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: 64, height: 64 })
	};
	Fluid.app.input.create(mockCanvas, cfg, solver, () => {});
	const beforeOverlay = cfg.overlay;
	for (const fn of listeners.keydown || []) fn({ key: "o", target: null, preventDefault() {} });
	check("keyboard shortcut 'o' toggles solid overlay flag and marks view dirty",
		cfg.overlay === (beforeOverlay > 0 ? 0 : 1) && cfg._dirty === true,
		"overlay " + beforeOverlay + " -> " + cfg.overlay);
	globalThis.window = origWin;
	globalThis.document = origDoc;
	solver.destroy();
}

async function testEmitterPlumbing() {
	const S = Fluid.core.stats;
	const E = Fluid.core.emitters;
	const statsSnap = new Float32Array(S.STATS_LEN);
	const m = mock.create({ stats: statsSnap, probes: new Float32Array(Fluid.core.probes.LEN) });
	const sink = m.sink;
	const N = 64;
	const cfg = Object.assign({}, Fluid.core.config.defaults, {
		N, texW: N + 2, texH: N + 2, scenario: "still", buoyEnabled: 0
	});
	const canvas = { width: N, height: N };
	const solver = Fluid.gpu.solver.create(m.device, canvas, mock.mockContext(), "bgra8unorm", cfg);
	check("emitter pool: storage table + 80B Kahan ledger buffer allocated",
		solver.pool.emitters.size === E.LEN * 4 &&
		solver.pool.emitterLedger.size === S.EMIT_LEDGER_LEN * 4,
		"emitters=" + solver.pool.emitters.size + "B ledger=" + solver.pool.emitterLedger.size + "B");

	// 0 emitters: neither applyEmitters nor emitterTally dispatches
	let p0 = sink.pipeOrder.length;
	solver.frame(cfg.frameDt / 2, 2, false);
	let slice = sink.pipeOrder.slice(p0);
	check("emitter dispatch gating: idle table adds zero dispatches",
		!slice.includes("applyEmitters") && !slice.includes("emitterTally"));

	// momentum-only emitter: folded into forces, zero extra dispatches
	const momId = solver.addEmitter({ kind: "momentum", x: 32, y: 32, radius: 6, vx: 120, vy: 40 });
	p0 = sink.pipeOrder.length;
	solver.frame(cfg.frameDt / 2, 2, false);
	slice = sink.pipeOrder.slice(p0);
	check("emitter dispatch gating: momentum-only emitter runs inside forces with zero extra passes",
		momId === 0 && !slice.includes("applyEmitters") && !slice.includes("emitterTally"));

	// thermal-only emitter (freeze): runs applyEmitters, skips emitterTally (therm.a accounts heatOut)
	const fzId = solver.addEmitter({ kind: "freeze", x: 20, y: 20, radius: 5, value: 90 });
	p0 = sink.pipeOrder.length;
	solver.frame(cfg.frameDt / 2, 2, false);
	slice = sink.pipeOrder.slice(p0);
	check("emitter dispatch gating: freeze/heat/cold runs applyEmitters without emitterTally",
		fzId === 1 && slice.filter(n => n === "applyEmitters").length === 2 && !slice.includes("emitterTally"));

	// dye + drain emitters: runs both emitterTally and applyEmitters, and profiled vs unprofiled leave identical parities
	const dyeId = solver.addEmitter({ kind: "dye", x: 16, y: 16, radius: 5, value: 0.4, dye: [1, 0.2, 0.1] });
	const drnId = solver.addEmitter({ kind: "drain", x: 48, y: 48, radius: 5, value: 0.3 });
	p0 = sink.pipeOrder.length;
	solver.frame(cfg.frameDt / 2, 2, false);
	slice = sink.pipeOrder.slice(p0);
	check("emitter dispatch gating: dye/drain runs emitterTally (1 wg) before applyEmitters each substep",
		dyeId === 2 && drnId === 3 &&
		slice.filter(n => n === "emitterTally").length === 2 &&
		slice.filter(n => n === "applyEmitters").length === 2 &&
		slice.indexOf("emitterTally") < slice.indexOf("applyEmitters"));

	// async readback pairing: simulate GPU emitterLedger + heatIn/Out arriving while paintDye interleaves
	const initM = solver.getStats().ledger.initialM;
	const initDr = solver.getStats().ledger.initialDr;
	const initE = solver.getStats().ledger.initialE;
	statsSnap[0] = initM + 12.5 - 3.25;
	statsSnap[1] = initDr + 6.0 - 1.5;
	statsSnap[4] = initE + 400.0 - 90.0 + 50.0 - 20.0;
	statsSnap[17] = 50.0;
	statsSnap[18] = 20.0;
	statsSnap[S.IDX_EMIT_INJ_M] = 12.5;
	statsSnap[S.IDX_EMIT_INJ_DR] = 6.0;
	statsSnap[S.IDX_EMIT_INJ_E] = 400.0;
	statsSnap[S.IDX_EMIT_REM_M] = 3.25;
	statsSnap[S.IDX_EMIT_REM_DR] = 1.5;
	statsSnap[S.IDX_EMIT_REM_E] = 90.0;
	solver.frame(cfg.frameDt, 1, true);
	// interleave a brush paint AFTER stats copy was queued, BEFORE mapAsync resolves
	solver.paintDye(0.5, 0.5, [0, 1, 0]);
	const paintM = solver.getStats().ledger.injectedM;
	await solver.pollReadback();
	const st = solver.getStats();
	check("emitter readback pairing: expected totals pair with GPU emitterLedger without losing interleaved paintDye",
		Math.abs(st.expected.m - statsSnap[0]) < 1e-5 &&
		Math.abs(st.expected.dr - statsSnap[1]) < 1e-5 &&
		Math.abs(st.expected.E - statsSnap[4]) < 1e-3 &&
		Math.abs(st.ledger.injectedM - (paintM + 12.5)) < 1e-5 &&
		Math.abs(st.ledger.removedM - 3.25) < 1e-5 &&
		Math.abs(cfg.rhoBase - (initM + paintM) / (N * N)) < 1e-6,
		"expM=" + st.expected.m.toFixed(3) + " ledgerInjM=" + st.ledger.injectedM.toFixed(3));

	// resize clamps coordinates to new grid and re-uploads live slots
	solver.moveEmitter(dyeId, 60, 62);
	solver.resize(128);
	const uploaded = new Float32Array(solver.pool.emitters._buf);
	check("emitter resize: live slots survive pool rebuild and re-upload to new buffer",
		solver.emitterCount() === 4 && uploaded[0] === 4 &&
		uploaded[E.slotBase(dyeId) + E.E_X] === 60,
		"live=" + uploaded[0]);
	solver.removeEmitter(momId);
	solver.removeEmitter(fzId);
	solver.removeEmitter(dyeId);
	solver.removeEmitter(drnId);
	solver.destroy();
	check("emitter pool buffers destroyed cleanly with zero leaks",
		m.counts.buffersAlive === 0 && m.counts.texturesAlive === 0);
}

(async function main() {
	testParams();
	testWgsl();
	testSampler();
	testProbeTable();
	await testPlumbing();
	await testProbePlumbing();
	await testPhasePlumbing();
	await testEmitterPlumbing();
	console.log(failures === 0 ? "ALL GPU-MOCK CHECKS PASSED" : failures + " FAILURES");
	process.exitCode = failures === 0 ? 0 : 1;
})();
