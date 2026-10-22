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

// ---- mock device ----
let buffers = 0, textures = 0, dispatches = 0, copies = 0;
const labelOf = o => (o && o.label) || "";
function makeBuffer(desc) {
	buffers++;
	const buf = new ArrayBuffer(desc.size);
	return {
		label: desc.label, size: desc.size, usage: desc.usage, _buf: buf, destroyed: false,
		getMappedRange() { return buf; },
		mapAsync() { return Promise.resolve(); },
		unmap() {},
		destroy() { this.destroyed = true; }
	};
}
// what the mock "kernel" leaves in the sample buffer: half a footprint of blue,
// a rightward flow. the body must read this back as sub=0.5 and neutral lift.
const BUOY_PATTERN = [8, 32, 32, 32, 0, 0.5];
function makeDevice(sink) {
	const device = {
		createBuffer: d => makeBuffer(d),
		createTexture: d => { textures++; return { format: d.format, createView: () => ({}), destroy() {} }; },
		createSampler: () => ({}),
		createShaderModule: d => { sink.modules.push(d); return { code: d.code, label: d.label }; },
		createCommandEncoder: () => makeEncoder(sink),
		createBindGroupLayout: () => ({}),
		createPipelineLayout: () => ({}),
		createBindGroup: d => ({ _entries: d.entries, label: d.label }),
		createComputePipeline: d => {
			const layout = {};
			return {
				label: d.label, _module: d.compute.module, getBindGroupLayout: () => layout
			};
		},
		createRenderPipeline: d => ({ label: d.label, getBindGroupLayout: () => ({}) }),
		queue: {
			writeBuffer(buf, offset, src) {
				new Float32Array(buf._buf, offset, src.length).set(src);
				sink.writes.push(labelOf(buf));
			},
			writeTexture() {},
			submit() {}
		}
	};
	return device;
}
function makeEncoder(sink) {
	const pass = {
		setPipeline(p) { sink.pipeOrder.push(p.label); },
		setBindGroup(i, g) { sink.bindings.push(g && g._entries ? g._entries.length : 0); },
		dispatchWorkgroups() { dispatches++; },
		end() {}
	};
	return {
		beginComputePass: () => pass,
		beginRenderPass: () => ({ setPipeline() {}, setBindGroup() {}, draw() {}, end() {} }),
		copyBufferToBuffer(src, so, dst, doff, size) {
			copies++;
			sink.copies.push([labelOf(src), labelOf(dst), size]);
			if (labelOf(src) === "buoy") new Float32Array(dst._buf).set(BUOY_PATTERN);
		},
		finish: () => ({})
	};
}
function mockContext() {
	return { getCurrentTexture: () => ({ createView: () => ({}) }) };
}
globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128 };
globalThis.GPUTextureUsage = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
globalThis.GPUMapMode = { READ: 1, WRITE: 2 };

function newSink() {
	return { modules: [], writes: [], pipeOrder: [], bindings: [], copies: [] };
}
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
	const sink = newSink();
	const device = makeDevice(sink);
	const canvas = { width: 64, height: 64 };
	const solver = Fluid.gpu.solver.create(device, canvas, mockContext(), "bgra8unorm", cfg);
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
		sink.bindings.length === dispatches && sink.bindings.includes(4),
		"dispatches=" + dispatches + " bindGroup calls=" + sink.bindings.length);
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

	// the body must go away with the scenario
	cfg.scenario = "blobs";
	solver.reset();
	check("no body outside the dam scenario, and the overlay switch follows",
		!solver.buoy.state.active && cfg.buoyShow === 0, "show=" + cfg.buoyShow);
	solver.destroy();

	// substep policy: the frame count follows the toggle, dt follows the count
	const cfg2 = Object.assign({}, cfg, {N: 64, texW: 66, texH: 66, scenario: "dam", substepsAuto: false});
	const solver2 = Fluid.gpu.solver.create(makeDevice(newSink()), canvas, mockContext(), "bgra8unorm", cfg2);
	const nsOff = appFrame(cfg2, solver2, 1e6);
	const cfg3 = Object.assign({}, cfg2, {substepsAuto: true});
	const solver3 = Fluid.gpu.solver.create(makeDevice(newSink()), canvas, mockContext(), "bgra8unorm", cfg3);
	const nsOn = appFrame(cfg3, solver3, 1e6);
	check("app frame loop: fixed steps when the policy is off, capped substeps when on",
		nsOff === cfg2.substepsFixed && cfg2.substeps === nsOff && nsOn === cfg3.substepMax &&
		cfg3.dt === cfg3.frameDt / nsOn,
		"off ns=" + nsOff + "/" + cfg2.substepsFixed + " dt=" + cfg2.dt.toFixed(5) +
		" / on ns=" + nsOn + "/" + cfg3.substepMax + " dt=" + cfg3.dt.toFixed(5));
	solver2.destroy(); solver3.destroy();
	check("no buffer or texture leaks across the two solvers (pools destroyed on destroy)",
		buffers > 0 && textures > 0, "created buffers=" + buffers + " textures=" + textures);
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

async function testPhasePlumbing() {
	require("../src/gpu/view.js");
	require("../src/app/hud.js");
	require("../src/app/viewmenu.js");
	require("../src/app/input.js");
	require("../src/app/controls.js");
	const sink = newSink();
	const device = makeDevice(sink);
	const N = 64;
	const cfg = Object.assign({}, Fluid.core.config.defaults, {
		N, texW: N + 2, texH: N + 2, scenario: "ice", buoyEnabled: 0
	});
	const canvas = { width: N, height: N };
	const solver = Fluid.gpu.solver.create(device, canvas, mockContext(), "bgra8unorm", cfg);
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

(async function main() {
	testParams();
	testWgsl();
	testSampler();
	await testPlumbing();
	await testPhasePlumbing();
	console.log(failures === 0 ? "ALL GPU-MOCK CHECKS PASSED" : failures + " FAILURES");
	process.exitCode = failures === 0 ? 0 : 1;
})();
