"use strict";
// Engine API contract (0.7.0): pacing, step policy, stats copy, probe queries,
// view ownership, input lifecycle, resize and teardown, plus examples/demo.html
// booted on the real engine. Drives Fluid.engine + Fluid.gpu.solver over the
// recording mock device, so it validates wiring and ownership, not shader numerics.
// run: node experiments/engine.js

const assert = require("node:assert/strict");
const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/solver.js"));
require(path.join(root, "src/core/probes.js"));
require(path.join(root, "src/core/emitters.js"));
require(path.join(root, "src/core/colliders.js"));
require(path.join(root, "src/core/buoy.js"));
require(path.join(root, "src/core/pacing.js"));
require(path.join(root, "src/gpu/device.js"));
require(path.join(root, "src/gpu/shaders.js"));
require(path.join(root, "src/gpu/fields.js"));
require(path.join(root, "src/gpu/passes.js"));
require(path.join(root, "src/gpu/solver.js"));
require(path.join(root, "src/api/engine.js"));
const mock = require("./mock-device.js");

let checks = 0;
function check(ok, name, detail) {
	assert.ok(ok, name + (detail ? "  " + detail : ""));
	console.log("ok   " + name + (detail ? "  " + detail : ""));
	checks++;
}
const flush = async (n = 4) => { for (let i = 0; i < n; i++) await Promise.resolve(); };

// a queue whose completion the test decides, so backpressure is observable. texture
// uploads are recorded into the mock's sink so a patch can be checked cell by cell.
function stubQueue(sink) {
	const pending = [];
	return {
		inflight: () => pending.length,
		onSubmittedWorkDone() { return new Promise(resolve => pending.push(resolve)); },
		writeBuffer(buf, offset, src) { new Float32Array(buf._buf, offset, src.length).set(src); },
		writeTexture(dst, data, layout, size) {
			if (!sink) return;
			sink.textureWrites.push({
				label: (dst.texture && dst.texture.label) || "", data, layout, size,
				origin: dst.origin ? [dst.origin.x || 0, dst.origin.y || 0] : [0, 0]
			});
		},
		submit() {},
		completeOne() { const r = pending.shift(); if (r) r(); return !!r; },
		completeAll() { while (pending.length) pending.shift()(); }
	};
}
function statsPattern() {
	const s = new Float32Array(Fluid.core.stats.STATS_LEN);
	s[0] = 10; s[1] = 1; s[2] = 2; s[3] = 3; s[4] = 400;
	s[5] = 0; s[6] = 0; s[7] = 0; s[8] = 0; s[9] = 0;
	s[10] = 1.1; s[11] = 0; s[12] = 2;
	s[13] = 55; s[14] = -3; s[15] = 7; s[16] = 0.5;
	s[17] = 9; s[18] = 4; s[19] = 280; s[20] = 330; s[21] = 305; s[22] = 12;
	s[23] = 1; s[24] = 2; s[25] = 3; s[26] = 4; s[27] = 5; s[28] = 6;
	s[29] = 0; s[30] = 1; s[31] = 2;
	s[32] = 0.25; s[33] = 3.5; s[34] = 0.3; s[35] = 4;
	return s;
}
function probePattern() {
	const p = new Float32Array(Fluid.core.probes.LEN);
	const b = Fluid.core.probes.slotBase(0);
	p[b + 2] = 1; p[b + 3] = 0.1; p[b + 4] = 0.2; p[b + 5] = 0.3;
	p[b + 6] = 2 * 273 + 334 + 4.2 * 50; p[b + 7] = 3; p[b + 8] = -2; p[b + 9] = 12;
	return p;
}
async function makeSim(cfgOverrides, patterns) {
	const m = mock.create(patterns || {});
	m.device.queue = stubQueue(m.sink);
	const canvas = { width: 0, height: 0 };
	const sim = await Fluid.engine.create(canvas, {
		device: m.device, context: mock.mockContext(), format: "bgra8unorm",
		cfg: Object.assign({ N: 64, scenario: "dam" }, cfgOverrides || {})
	});
	return { sim, m, canvas };
}


// the pacing meter on its own: windowed rates, idle/paused rows, queue latency and
// a rejected completion must never leave the app stuck busy
async function testPacingMeter() {
	let now = 0, resolve, reject;
	const queue = { onSubmittedWorkDone() { return new Promise((a, b) => { resolve = a; reject = b; }); } };
	const perf = Fluid.core.pacing.create(queue, () => now);
	for (let i = 0; i < 144; i++) {
		perf.tick(1 / 144);
		if (i % 3 === 0) {
			perf.submitted(1 / 60, 2);
			assert.equal(perf.state.busy, true);
			now += 8; resolve(); await flush(2);
			assert.equal(perf.state.busy, false);
		}
	}
	check(Math.abs(perf.state.rafHz - 144) < 1e-9 && Math.abs(perf.state.simHz - 48) < 2 &&
		perf.state.simSpeed < 0.85 && perf.state.queueMs === 8 && perf.state.cpuMs === 2,
		"pacing meter: windowed rAF/s, sim/s, speed, queue latency and encode time",
		"raf=" + perf.state.rafHz + " sim=" + perf.state.simHz.toFixed(1));
	perf.submitted(0, 1); resolve(); await flush(2);
	for (let i = 0; i < 150; i++) perf.tick(1 / 144);
	check(perf.state.simHz === 0 && perf.state.renderHz === 0, "pacing meter: a paused view reports no sim or render rate");
	perf.submitted(1 / 60, 1); reject(new Error("lost")); await flush(2);
	check(perf.state.busy === false && perf.state.error.message === "lost",
		"pacing meter: a rejected completion clears busy and surfaces the error");
}

async function testEngineCreate() {
	const { sim, m, canvas } = await makeSim({ substepsAuto: false, substepsFixed: 4 });
	check(sim && sim.solver && sim.params && sim.perf, "engine exposes solver, live params and the pacing state");
	check(canvas.width === 64 && canvas.height === 64 && sim.params.N === 64 && sim.params.texW === 66,
		"create sizes the canvas and the texture grid from cfg.N",
		"canvas=" + canvas.width + " texW=" + sim.params.texW);
	const ns = sim.step(1 / 60);
	check(ns === 4 && sim.substeps === 4 && Math.abs(sim.params.dt - 1 / 240) < 1e-12 &&
		Math.abs(sim.params.frameDt - 1 / 60) < 1e-12,
		"step(dt) runs the shared substep policy: 4 fixed steps of dt/4",
		"ns=" + ns + " dt=" + sim.params.dt.toFixed(6));
	check(sim.params.substeps === 4 && m.counts.dispatches > 4 && sim.perf.busy === true,
		"a step submits one frame and marks the queue busy",
		"dispatches=" + m.counts.dispatches + " busy=" + sim.perf.busy);
	check(sim.stats().frame === 1, "stats() reports the solver frame count", "frame=" + sim.stats().frame);
	m.device.queue.completeAll();
	await flush();
	check(sim.perf.busy === false && sim.perf.queueMs >= 0, "completion releases the in-flight gate");
	sim.destroy();
	check(sim.destroyed === true && sim.solver === null && m.counts.buffersAlive === 0 && m.counts.texturesAlive === 0,
		"destroy releases the pool, every buffer and the device reference",
		"alive buffers=" + m.counts.buffersAlive);
}

async function testPacing() {
	const { sim, m } = await makeSim({ substepsAuto: false, substepsFixed: 4 });
	const q = m.device.queue;
	// 144 Hz callbacks over one wall second must produce 60 fixed steps, one per call
	for (let i = 0; i < 144; i++) { sim.update(1 / 144); q.completeOne(); await flush(2); }
	const frames = sim.stats().frame;
	check(frames >= 59 && frames <= 61, "update(): 144 Hz callbacks produce 60 fixed steps per second", "frames=" + frames);
	// hold one submission unresolved across 100 callbacks: no queue growth, no steps
	sim.update(0.02);
	const held = sim.stats().frame;
	for (let i = 0; i < 100; i++) sim.update(0.02);
	check(sim.stats().frame === held && q.inflight() === 1,
		"update(): one application frame in flight, no catch-up submit while the gpu is busy",
		"frames=" + sim.stats().frame + " inflight=" + q.inflight());
	q.completeOne(); await flush(2);
	sim.update(0.02);
	check(sim.stats().frame === held + 1, "update(): resumes after the submission completes", "frames=" + sim.stats().frame);
	q.completeAll(); await flush(2);
	// paused: no sim, no redraw until something is dirty
	sim.pause();
	check(sim.paused === true, "pause() sets the live paused state");
	const shown = m.sink.pipeOrder.filter(p => p === "toDisplay").length;
	for (let i = 0; i < 20; i++) { sim.update(1 / 60); q.completeOne(); await flush(2); }
	check(sim.stats().frame === held + 1 && m.sink.pipeOrder.filter(p => p === "toDisplay").length === shown,
		"update(): a paused clean canvas submits no redraws");
	sim.params._dirty = true;
	sim.update(1 / 60);
	check(m.sink.pipeOrder.filter(p => p === "toDisplay").length === shown + 1,
		"update(): a dirty paused view redraws exactly once");
	q.completeAll(); await flush(2);
	sim.singleStep();
	sim.update(1 / 60);
	check(sim.stats().frame === held + 2 && sim.params.frameDt === (1 / 60),
		"singleStep() advances exactly one frame while paused", "frames=" + sim.stats().frame);
	sim.resume();
	check(sim.paused === false, "resume() clears the paused state");
	sim.destroy();
}

async function testStatsAndProbes() {
	const { sim, m } = await makeSim({ substepsAuto: false, substepsFixed: 2 }, { stats: statsPattern(), probes: probePattern() });
	const q = m.device.queue;
	sim.step(1 / 60);
	await sim.pollReadback();
	q.completeAll(); await flush(2);
	const out = Fluid.engine.makeStats();
	const got = sim.stats(out);
	check(got === out && out.totals.m === 10 && out.totals.E === 400 && out.mins.dr === 0 &&
		Math.abs(out.maxM - 1.1) < 1e-6 && out.cntOver === 2 && out.motion.maxVel === 55 &&
		out.thermal.TMax === 330 && out.phaseCounts[3] === 4 && out.diag.poissonRes === 0.25,
		"stats(out) fills the caller's object from the last completed readback, no allocation",
		"m=" + out.totals.m + " phases=" + out.phaseCounts.join("/") + " res=" + out.diag.poissonRes);
	const again = sim.stats();
	check(again.totals.m === 10 && again !== out, "stats() without an argument reuses one internal object");

	const id0 = sim.addProbe(10, 20);
	const id1 = sim.addProbe(30, 40);
	const sample = Fluid.core.probes.makeSample();
	check(id0 === 0 && id1 === 1 && sim.probeCount() === 2, "addProbe returns stable ids, probeCount tracks the table");
	check(sim.sample(id0, sample) === false && sample.valid === false,
		"sample() refuses before the first probe readback: the age is known, not invented");
	for (let f = 0; f < 5; f++) {
		sim.step(1 / 60);
		await sim.pollReadback();
		q.completeAll(); await flush(2);
	}
	const ok = sim.sample(id0, sample);
	check(ok && sample.valid && sample.m === 1 && sample.u === 3 && sample.p === 12 &&
		sample.phase === Fluid.core.solver.PHASE_LIQUID && sample.age <= sim.params.probeEvery,
		"sample() returns the last completed probe value with its age and derived phase",
		"m=" + sample.m + " T=" + sample.T.toFixed(1) + " age=" + sample.age);
	sim.moveProbe(id0, 5, 6);
	check(sim.removeProbe(id0) === true && sim.probeCount() === 1 && sim.sample(id0, sample) === false,
		"moveProbe/removeProbe maintain the table and invalidate the removed id");
	sim.destroy();
}

async function testViewAndGrid() {
	const { sim, m, canvas } = await makeSim({ substepsAuto: false });
	let seen = -1;
	check(sim.bindViewMode(id => { seen = id; }) === true, "bindViewMode installs the app's single writer");
	const before = sim.params.viewMode;
	check(sim.setViewMode(3) === true && seen === 3 && sim.params.viewMode === before,
		"setViewMode delegates to the bound writer instead of writing cfg itself",
		"seen=" + seen + " cfg=" + sim.params.viewMode);
	sim.bindViewMode(null);
	check(sim.setViewMode(5) === true && sim.params.viewMode === 5 && sim.params._dirty === true,
		"without a writer the engine owns cfg.viewMode and marks the view dirty");
	check(sim.setViewMode(99) === false, "an unknown view id is refused");
	check(sim.setOverlay(0) === 0 && sim.params.overlay === 0 && sim.setOverlay(1) === 1,
		"setOverlay writes the overlay flag");
	sim.addProbe(32, 32);
	const n = sim.setGrid(128);
	check(n === 128 && sim.solver.pool.N === 128 && canvas.width === 128 && sim.params.texW === 130 && m.counts.buffersAlive > 0,
		"setGrid rebuilds the pool at the new size and resizes the canvas",
		"N=" + n + " texW=" + sim.params.texW);
	// the rebuild allocates a zeroed probe buffer: the live table must be pushed
	// again or every probe would silently stop after a resize
	const probeRaw = new Float32Array(sim.solver.pool.probes._buf);
	const base = Fluid.core.probes.slotBase(0);
	check(probeRaw[0] === 1 && probeRaw[base + Fluid.core.probes.P_X] === 32 && probeRaw[base + Fluid.core.probes.P_ACTIVE] === 1,
		"setGrid re-uploads the live probe table into the rebuilt pool",
		"count=" + probeRaw[0] + " x=" + probeRaw[base + Fluid.core.probes.P_X]);
	check(sim.setGrid(10) === 64, "setGrid clamps below the supported grid", "N=" + sim.params.N);
	check(sim.reset("cycle") === true && sim.params.scenario === "cycle" && sim.reset("nope") === false,
		"reset(scenario) validates the name and restarts the solver");
	sim.destroy();
	check(m.counts.buffersAlive === 0 && m.counts.texturesAlive === 0, "the resized pool is fully released on destroy");
}

class Target {
	constructor() { this.listeners = {}; }
	addEventListener(type, fn) { (this.listeners[type] || (this.listeners[type] = [])).push(fn); }
	removeEventListener(type, fn) {
		const list = this.listeners[type] || [];
		const at = list.indexOf(fn);
		if (at >= 0) list.splice(at, 1);
	}
	count() { let n = 0; for (const k in this.listeners) n += this.listeners[k].length; return n; }
}
async function testInputLifecycle() {
	const origWindow = globalThis.window, origDocument = globalThis.document;
	const win = new Target();
	globalThis.window = win;
	globalThis.document = { getElementById: () => null, addEventListener() {}, removeEventListener() {}, hidden: false };
	try {
		require(path.join(root, "src/app/input.js"));
		const { sim } = await makeSim({ substepsAuto: false });
		const canvas = new Target();
		canvas.setPointerCapture = () => {};
		canvas.hasPointerCapture = () => false;
		canvas.releasePointerCapture = () => {};
		canvas.focus = () => {};
		canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 100, height: 100 });
		const before = win.count() + canvas.count();
		const input = sim.attachInput(canvas);
		check(input && input.state && win.count() + canvas.count() > before,
			"attachInput installs pointer and key handlers on the element",
			"listeners=" + (win.count() + canvas.count()));
		const mode = { seen: -1 };
		sim.bindViewMode(id => { mode.seen = id; });
		for (const fn of win.listeners.keydown || []) fn({ key: "4", target: null, preventDefault() {} });
		check(mode.seen === 4, "keyboard view keys route through the engine's writer", "mode=" + mode.seen);
		sim.detachInput();
		check(win.count() + canvas.count() === before, "detachInput hands every listener back",
			"listeners=" + (win.count() + canvas.count()));
		sim.attachInput(canvas);
		sim.destroy();
		check(win.count() + canvas.count() === before, "destroy() detaches input before releasing the pool");
	} finally {
		globalThis.window = origWindow;
		globalThis.document = origDocument;
	}
}

async function testTeardown() {
	const canvas = { width: 0, height: 0 };
	for (let i = 0; i < 20; i++) {
		const m = mock.create();
		const sim = await Fluid.engine.create(canvas, {
			device: m.device, context: mock.mockContext(), format: "bgra8unorm",
			cfg: { N: 64, scenario: "blobs" }
		});
		sim.step(1 / 60);
		sim.addProbe(4, 4);
		sim.destroy();
		sim.destroy();
		assert.equal(m.counts.buffersAlive, 0, "buffer leak on create/destroy cycle " + i);
		assert.equal(m.counts.texturesAlive, 0, "texture leak on create/destroy cycle " + i);
	}
	check(true, "20 create/step/probe/destroy cycles leave no buffer or texture alive");
}

// the lab page is now a thin shell over the engine: it must still apply the query
// overrides, bind the view row as the writer, attach input and pace the hud.
async function testLabWiring() {
	const origWindow = globalThis.window, origDocument = globalThis.document, origLocation = globalThis.location;
	const origRaf = globalThis.requestAnimationFrame, origCancel = globalThis.cancelAnimationFrame;
	const win = new Target();
	const canvas = new Target();
	canvas.focus = () => {};
	const panel = { hidden: true, textContent: "" };
	const status = { textContent: "" };
	let created = null, updated = 0, substepsSeen = 0, destroyed = 0, raf = null;
	const fakeSim = {
		params: null,
		perf: { error: null },
		update() { updated++; return updated % 2; },
		bindViewMode(fn) { this.writer = fn; return true; },
		attachInput() { this.attached = true; },
		setViewMode() {},
		solver: {},
		destroy() { destroyed++; }
	};
	globalThis.window = win;
	globalThis.document = {
		readyState: "loading",
		getElementById: id => id === "c" ? canvas : (id === "error-panel" ? panel : status),
		addEventListener() {}, removeEventListener() {}, hidden: false
	};
	globalThis.location = { search: "?k=40&mg=0&cycles=1" };
	globalThis.requestAnimationFrame = fn => { raf = fn; return 1; };
	globalThis.cancelAnimationFrame = () => {};
	const origEngine = Fluid.engine;
	const origViewMenu = Fluid.app.viewMenu, origHud = Fluid.app.hud, origControls = Fluid.app.controls;
	Fluid.engine = {
		async create(el, options) {
			fakeSim.params = options.cfg;
			created = options;
			return fakeSim;
		}
	};
	Fluid.app.viewMenu = { create: () => ({ setMode() {} }) };
	Fluid.app.hud = { create: () => ({ update() {}, setSubsteps(n) { substepsSeen = n; } }) };
	Fluid.app.controls = { create: () => ({ update() {} }) };
	try {
		const mainPath = path.join(root, "src/app/main.js");
		delete require.cache[require.resolve(mainPath)];
		require(mainPath);
		await Fluid.app.main.boot();
		check(created && created.cfg.useMultigrid === false && created.cfg.pressureIters === 40 &&
			created.cfg.mgCycles === 1 && created.cfg.productCycleGate === 1,
			"lab boot applies the query overrides before the engine is created",
			"mg=" + created.cfg.useMultigrid + " K=" + created.cfg.pressureIters + " cycles=" + created.cfg.mgCycles);
		check(fakeSim.attached === true && typeof fakeSim.writer === "function" && fakeSim.params === created.cfg,
			"lab binds the view row as the writer, attaches input and shares the live params object");
		let now = 0;
		for (let i = 0; i < 4 && raf; i++) { const fn = raf; raf = null; now += 1000 / 60; fn(now); }
		check(updated === 4 && substepsSeen > 0, "the rAF loop delegates pacing to sim.update and feeds the hud",
			"updates=" + updated + " substeps=" + substepsSeen);
		Fluid.app.main.stop();
		check(destroyed === 1, "stop() destroys the engine");
	} finally {
		Fluid.engine = origEngine;
		Fluid.app.viewMenu = origViewMenu;
		Fluid.app.hud = origHud;
		Fluid.app.controls = origControls;
		globalThis.window = origWindow;
		globalThis.document = origDocument;
		globalThis.location = origLocation;
		globalThis.requestAnimationFrame = origRaf;
		globalThis.cancelAnimationFrame = origCancel;
	}
}


// sources (0.7.0 t3): the table, the pass staging, the ledger totals and the
// resize/reset survival. the numbers the ledger books are checked against an
// independent f64 sum over the same disc, because that is the part a wrong
// implementation gets right visually and wrong silently.
async function testEmitters() {
	const E = Fluid.core.emitters;
	// the cpu oracle: the injected totals must be exactly the fields it writes
	const cfg = Object.assign({}, Fluid.core.config.defaults, { N: 32, scenario: "still" });
	cfg.texW = 34; cfg.texH = 34;
	const table = E.create();
	E.add(table, cfg, { kind: "dye", x: 16, y: 16, radius: 5, value: 3, dye: [0, 1, 0] });
	E.add(table, cfg, { kind: "heat", x: 10, y: 10, radius: 4, value: 50 });
	check(E.kindOf("heat") === E.KIND_HEAT && E.kindOf("momentum") === E.KIND_MOMENTUM &&
		E.kindOf("drain") === -1 && table.live === 2,
		"sources: kind names map to ids, and the deferred kinds are refused instead of guessed");
	const size = cfg.texW * cfg.texH;
	const fields = {
		mass: new Float32Array(size * 4), therm: new Float32Array(size * 4),
		vel: new Float32Array(size * 2), texW: cfg.texW, texH: cfg.texH
	};
	E.header(table, cfg);
	const oracle = {};
	E.sweep(table, cfg, 1 / 60, fields, oracle);
	let sumM = 0, sumDr = 0, sumE = 0;
	for (let i = 0; i < size; i++) {
		sumM += fields.mass[i * 4 + 3];
		sumDr += fields.mass[i * 4];
		sumE += fields.therm[i * 4];
	}
	// the fields store f32 per cell, the ledger sums f64, so only the last bits differ
	const close = (a, b) => Math.abs(a - b) <= Math.max(Math.abs(b), 1) * 1e-6;
	check(close(sumM, oracle.inM) && close(sumDr, oracle.inDr) && close(sumE, oracle.inE),
		"cpu oracle: the injected totals are the fields it writes, not a fitted disc",
		"m=" + sumM.toFixed(6) + " E=" + sumE.toFixed(1));

	const { sim, m } = await makeSim({ substepsAuto: false, substepsFixed: 2 });
	check(sim.emitterCount() === 0 && sim.addEmitter({ kind: "nope", x: 4, y: 4 }) === -1 &&
		sim.addEmitter({ kind: "drain", x: 4, y: 4 }) === -1,
		"engine: an unknown or deferred kind is refused");
	const dye = { kind: "dye", x: 32, y: 32, radius: 6, value: 2, dye: [1, 0, 0] };
	const id = sim.addEmitter(dye);
	const buf = new Float32Array(sim.solver.pool.emitters._buf);
	const b = E.slotBase(0);
	const eAmb = Fluid.core.solver.energyFromT(1, sim.params.TRef, sim.params);
	check(id === 0 && sim.emitterCount() === 1 && buf[E.H_LIVE] === 1 &&
		buf[b + E.E_X] === 32 && buf[b + E.E_Y] === 32 && buf[b + E.E_RADIUS] === 6 &&
		buf[b + E.E_KIND] === E.KIND_DYE && buf[b + E.E_VALUE] === 2 &&
		buf[b + E.E_DR] === 1 && buf[b + E.E_ACTIVE] === 1 &&
		Math.abs(buf[E.H_EAMB] - eAmb) < 1e-6,
		"addEmitter copies the spec into a preallocated slot and pushes the table",
		"id=" + id + " live=" + buf[E.H_LIVE] + " x=" + buf[b + E.E_X]);
	// the spec object is copied: mutating it afterwards must not move the slot
	dye.x = 1;
	check(new Float32Array(sim.solver.pool.emitters._buf)[b + E.E_X] === 32 && sim.solver.emitters.slots[0].x === 32,
		"the add-time copy owns the values, so the caller can reuse its spec object");

	const before = m.sink.pipeOrder.length;
	sim.step(1 / 60);
	const staged = m.sink.pipeOrder.slice(before);
	check(staged.indexOf("emitterMass") >= 0 && staged.indexOf("emitterTherm") >= 0 &&
		staged.indexOf("emitterVel") < 0,
		"a dye source stages the mass and energy passes and no velocity pass",
		staged.filter(p => p.indexOf("emitter") === 0).join(","));

	// an independent f64 sum over the same integer disc: inclusion is exact
	// arithmetic there, so only the per-cell rounding can differ
	function discMass(spec, dtFrame, texW, texH) {
		let sum = 0;
		const r = spec.radius;
		for (let y = Math.max(1, Math.ceil(spec.y - r)); y <= Math.min(texH - 2, Math.floor(spec.y + r)); y++) {
			for (let x = Math.max(1, Math.ceil(spec.x - r)); x <= Math.min(texW - 2, Math.floor(spec.x + r)); x++) {
				const dx = x - spec.x, dy = y - spec.y;
				const d2 = dx * dx + dy * dy;
				if (d2 >= r * r) continue;
				const f = 1 - Math.sqrt(d2) / r;
				sum += spec.value * f * f * dtFrame;
			}
		}
		return sum;
	}
	const want = discMass({ x: 32, y: 32, radius: 6, value: 2 }, 1 / 60, sim.params.texW, sim.params.texH);
	const ledger = sim.solver.getStats().ledger;
	const gotM = ledger.injectedM, gotE = ledger.injectedE;
	check(Math.abs(gotM - want) / want < 1e-6 && Math.abs(gotE - want * eAmb) / (want * eAmb) < 1e-6,
		"the ledger books the mass and energy the pass adds",
		"m=" + gotM.toFixed(6) + "/" + want.toFixed(6) + " E=" + gotE.toFixed(0));
	sim.step(1 / 60);
	check(Math.abs(ledger.injectedM - 2 * want) / (2 * want) < 1e-6,
		"a source keeps adding every frame at its rate", "m=" + ledger.injectedM.toFixed(6));

	check(sim.setEmitter(id, { value: 4, radius: 9, dye: [0, 0, 1] }) === true &&
		sim.moveEmitter(id, 10, 20) === true && sim.moveEmitter(5, 1, 1) === false,
		"setEmitter patches named fields and moveEmitter moves the live slot only");
	const moved = new Float32Array(sim.solver.pool.emitters._buf);
	check(moved[b + E.E_X] === 10 && moved[b + E.E_Y] === 20 && moved[b + E.E_RADIUS] === 9 &&
		moved[b + E.E_VALUE] === 4 && moved[b + E.E_DB] === 1 && moved[b + E.E_DR] === 0,
		"the patched slot is what the gpu reads", "x=" + moved[b + E.E_X] + " v=" + moved[b + E.E_VALUE]);

	check(sim.removeEmitter(id) === true && sim.emitterCount() === 0 &&
		new Float32Array(sim.solver.pool.emitters._buf)[b + E.E_ACTIVE] === 0 &&
		sim.removeEmitter(id) === false,
		"removeEmitter clears the active flag and the count, and is idempotent-safe");
	const momentum = sim.addEmitter({ kind: "momentum", x: 20, y: 20, radius: 5, value: 60, dir: [1, 0] });
	check(momentum === 0, "an empty slot is reused with the same stable id", "id=" + momentum);
	const before2 = m.sink.pipeOrder.length;
	sim.step(1 / 60);
	const staged2 = m.sink.pipeOrder.slice(before2);
	check(staged2.indexOf("emitterVel") >= 0 && staged2.indexOf("emitterMass") < 0,
		"a momentum source stages the velocity pass and nothing else",
		staged2.filter(p => p.indexOf("emitter") === 0).join(","));
	let full = 0;
	for (let i = sim.emitterCount(); i < E.MAX; i++) if (sim.addEmitter({ kind: "heat", x: 4, y: 4, radius: 2, value: 1 }) >= 0) full++;
	check(full === E.MAX - 1 && sim.emitterCount() === E.MAX &&
		sim.addEmitter({ kind: "heat", x: 4, y: 4, radius: 2, value: 1 }) === -1,
		"the table is fixed at " + E.MAX + " slots and refuses the overflow",
		"live=" + sim.emitterCount());
	// an idle frame with no live source stages nothing and books nothing
	for (let i = 0; i < E.MAX; i++) sim.removeEmitter(i);
	const idleM = sim.solver.getStats().ledger.injectedM;
	const before3 = m.sink.pipeOrder.length;
	sim.step(1 / 60);
	check(m.sink.pipeOrder.slice(before3).every(p => p.indexOf("emitter") !== 0) &&
		sim.solver.getStats().ledger.injectedM === idleM,
		"with no live source a frame stages no emitter pass and books nothing");
	sim.destroy();

	const r = await makeSim({ substepsAuto: false });
	r.sim.addEmitter({ kind: "heat", x: 16, y: 16, radius: 5, value: 100 });
	r.sim.setGrid(128);
	const rebuilt = new Float32Array(r.sim.solver.pool.emitters._buf);
	check(r.sim.emitterCount() === 1 && rebuilt[E.H_LIVE] === 1 && rebuilt[E.slotBase(0) + E.E_ACTIVE] === 1 &&
		rebuilt[E.slotBase(0) + E.E_X] === 16,
		"setGrid re-uploads the live source table into the rebuilt pool",
		"live=" + rebuilt[E.H_LIVE]);
	r.sim.reset("dam");
	const afterReset = new Float32Array(r.sim.solver.pool.emitters._buf);
	check(r.sim.emitterCount() === 1 && afterReset[E.H_LIVE] === 1 &&
		r.sim.solver.getStats().ledger.injectedM === 0,
		"reset() keeps the game's sources and starts a fresh ledger on them");
	r.sim.destroy();
}

// geometry (0.7.0 t4): the analytic mask, the delta upload, the velocity imprint and
// the static mask that a driving body must not erode. a collider is a pressure
// boundary, so every one of these is checked with mass untouched.
async function testColliders() {
	const C = Fluid.core.colliders;
	const cfg = Object.assign({}, Fluid.core.config.defaults, { N: 32, scenario: "still" });
	cfg.texW = 34; cfg.texH = 34;
	const table = C.create();
	const circle = C.add(table, cfg, { shape: "circle", x: 16, y: 16, radius: 3 });
	const box = C.add(table, cfg, { shape: "box", x: 6, y: 6, hw: 2, hh: 1 });
	check(circle === 0 && box === 1 && table.live === 2 && C.add(table, cfg, { shape: "hexagon", x: 1, y: 1 }) === -1,
		"colliders: circle and box slot ids, an unknown shape is refused");
	const owner = new Uint8Array(cfg.texW * cfg.texH);
	const rect = {};
	C.rasterize(table, cfg, owner, rect);
	const at = (x, y) => owner[x + y * cfg.texW];
	check(at(16, 16) === 1 && at(19, 16) === 1 && at(16, 19) === 1 && at(20, 16) === 0 && at(16, 20) === 0 &&
		at(6, 6) === 2 && at(4, 6) === 2 && at(8, 6) === 2 && at(3, 6) === 0 && at(6, 7) === 2 && at(6, 8) === 0,
		"the raster is a disc that includes its rim and a 5x3 box, in cell centres",
		"centre=" + at(16, 16) + " rim=" + at(19, 16) + " outside=" + at(20, 16));
	check(rect.x0 === 4 && rect.y0 === 5 && rect.x1 === 19 && rect.y1 === 19,
		"the raster reports the bounds it wrote, so nothing sweeps the grid",
		"rect=" + rect.x0 + "," + rect.y0 + ".." + rect.x1 + "," + rect.y1);
	C.move(table, cfg, 0, 16, 16);
	owner.fill(0);
	C.remove(table, 1);
	C.rasterize(table, cfg, owner, rect);
	check(table.live === 1 && at(6, 6) === 0 && at(16, 16) === 1 && C.remove(table, 1) === false,
		"remove drops the body from the next raster and is safe to repeat");

	// the push (t4b): a body that takes a cell gives its contents to the free fluid
	// within a radius-2 disk, so it carries color instead of sliding over a frozen
	// picture. One cell has twelve radius-2 receivers in the interior.
	const small = Object.assign({}, cfg, { N: 8, texW: 10, texH: 10, TRef: 293 });
	const one = C.create();
	C.add(one, small, { shape: "box", x: 5, y: 5, hw: 0.5, hh: 0.5 });
	const own = new Uint8Array(100), prev = new Uint8Array(100), g = new Float32Array(128 * 128 * 4);
	const r2 = C.padRect(small, C.rasterize(one, small, own, {}));
	const mf = new Float32Array(100 * 4), tf = new Float32Array(100 * 4);
	const kc = 5 + 5 * 10;
	mf[kc * 4] = 1; mf[kc * 4 + 1] = 2; mf[kc * 4 + 2] = 3; mf[kc * 4 + 3] = 4;
	tf[kc * 4] = 40; tf[kc * 4 + 1] = 350;
	const sumOf = (f, stride) => {
		let t = 0;
		for (let i = 0; i < f.length; i += 4) t += f[i + (stride || 0)];
		return t;
	};
	const massBefore = sumOf(mf, 3), eBefore = sumOf(tf);
	const w2 = r2.x1 - r2.x0 + 1;
	const idx = (x, y) => ((x - r2.x0) + (y - r2.y0) * w2) * 4 + 2;
	const taken = C.geometry(small, own, prev, r2, g);
	const nb = [[5, 3], [4, 4], [5, 4], [6, 4], [3, 5], [4, 5], [6, 5], [7, 5], [4, 6], [5, 6], [6, 6], [5, 7]];
	check(own[kc] === 1 && taken === 1 && g[idx(5, 5)] === 12 && g[idx(5, 4)] === 0 && g[idx(4, 4)] === 0,
		"the push marks the cell a body takes and counts all twelve radius-2 receivers",
		"taken=" + taken + " deg=" + g[idx(5, 5)] + " free=" + g[idx(4, 4)]);
	const scratch = C.makeScratch();
	C.displace(small, g, r2, mf, tf, scratch);
	check(mf[kc * 4 + 3] === 0 && tf[kc * 4] === 0 && tf[kc * 4 + 1] === small.TRef &&
		nb.every(([x, y]) => Math.abs(mf[(x + y * 10) * 4 + 3] - 1 / 3) < 1e-6 &&
			Math.abs(tf[(x + y * 10) * 4] - 40 / 12) < 1e-6),
		"a taken cell is emptied and every radius-2 neighbour gathers its equal share of dye and heat",
		"m=" + nb.map(([x, y]) => mf[(x + y * 10) * 4 + 3].toFixed(3)).join(",") +
		" E=" + nb.map(([x, y]) => tf[(x + y * 10) * 4].toFixed(2)).join(","));
	const massAfter = sumOf(mf, 3), eAfter = sumOf(tf);
	check(Math.abs(massAfter - massBefore) < 1e-6 && Math.abs(eAfter - eBefore) < 1e-4,
		"the push moves dye and heat to the neighbours without creating either",
		"m " + massBefore.toFixed(3) + "->" + massAfter.toFixed(3) + " E " + eBefore.toFixed(1) + "->" + eAfter.toFixed(1));
	// a rect that does not cover the receivers, or a body with no free neighbour: the
	// contents stay where they are rather than being destroyed
	const tight = C.rasterize(one, small, own, {});
	const back = { x0: tight.x0, y0: tight.y0, x1: tight.x1, y1: tight.y1 };
	C.geometry(small, own, prev, back, g);
	check(back.x1 === back.x0 && g[2] === -1,
		"a taken cell whose receivers are outside the patch keeps its contents",
		"code=" + g[2]);
	const sealed = new Uint8Array(100).fill(1);
	C.geometry(small, sealed, prev, r2, g);
	check(g[idx(5, 5)] === -1,
		"a taken cell with no free neighbour keeps what it holds, so nothing is destroyed");
	// a body that was already there keeps its cell: only the newly taken ones push
	const moved = new Uint8Array(100);
	own[5 + 5 * 10] = 0; own[6 + 5 * 10] = 1;
	prev[6 + 5 * 10] = 1; prev[5 + 5 * 10] = 1;
	C.geometry(small, own, prev, r2, g);
	check(g[idx(6, 5)] === -1 && g[idx(5, 5)] === 0,
		"a cell the body already held does not push again when another body cell is released",
		"held=" + g[idx(6, 5)] + " released=" + g[idx(5, 5)]);

	const { sim, m } = await makeSim({ substepsAuto: false, substepsFixed: 1 });
	const N = sim.params.N, texW = sim.params.texW;
	const flagsAt = (x, y) => sim.solver.flagsMirror[(x + y * texW) * 4];
	const id = sim.addCollider({ shape: "circle", x: 32, y: 32, radius: 4, vx: 0, vy: 0 });
	check(id === 0 && sim.colliderCount() === 1 && flagsAt(32, 32) === 255 && flagsAt(36, 32) === 255 &&
		flagsAt(37, 32) === 0,
		"addCollider paints the wall channel of the cells it covers",
		"centre=" + flagsAt(32, 32) + " rim=" + flagsAt(36, 32));
	check(m.sink.textureWrites.some(w => w.label === "flags") && m.sink.pipeOrder.indexOf("phaseEos") >= 0,
		"adding a body uploads the mask and re-primes the phase fields");

	// Keep multigrid requested, but automatically use the fine-grid pressure solve
	// while the conservative scalar push settles after a moving-body mask change.
	const fallbackWritesBefore = m.sink.textureWrites.length;
	const fallbackPipesBefore = m.sink.pipeOrder.length;
	sim.moveCollider(id, 40, 32, 120, -30);
	sim.step(1 / 60);
	const fallbackWrites = m.sink.textureWrites.slice(fallbackWritesBefore);
	const fallbackStaged = m.sink.pipeOrder.slice(fallbackPipesBefore);
	const fallbackVelWrite = fallbackWrites.filter(w => w.label === "patchVel").pop();
	const fallbackFlagsWrite = fallbackWrites.filter(w => w.label === "flags").pop();
	check(flagsAt(32, 32) === 0 && flagsAt(40, 32) === 255 && flagsAt(44, 32) === 255,
		"a moved body clears the cells it left and blocks the ones it entered",
		"old=" + flagsAt(32, 32) + " new=" + flagsAt(40, 32));
	check(fallbackFlagsWrite && fallbackFlagsWrite.size[0] < N && fallbackFlagsWrite.origin[0] > 0,
		"the mask delta uploads a patch, not the whole grid",
		fallbackFlagsWrite ? "rect " + fallbackFlagsWrite.size.join("x") + " at " + fallbackFlagsWrite.origin.join(",") : "no write");
	const pressureParam = new Float32Array(sim.solver.pool.params._buf)[Fluid.core.config.paramsOff.pressureIters];
	check(sim.params.useMultigrid === true && pressureParam === sim.params.pressureIters &&
		fallbackVelWrite && fallbackStaged.indexOf("colliderVel") >= 0 &&
		fallbackStaged.indexOf("colliderPushMass") >= 0 && fallbackStaged.indexOf("colliderPushTherm") >= 0 &&
		fallbackWrites.some(w => w.label === "patchGeo") && fallbackStaged.indexOf("mgCoarsen") < 0 &&
		fallbackStaged.indexOf("projRed") >= 0,
		"a multigrid-configured moving frame pushes dye and heat, then uses fine-grid pressure",
		"requestedMG=" + sim.params.useMultigrid + " K=" + pressureParam + " passes=" +
		fallbackStaged.filter(n => n.indexOf("collider") === 0 || n === "mgCoarsen").join(","));

	// Another movement inside the settling window keeps scalar push enabled and
	// does not mutate the caller's requested multigrid setting.
	const writesBefore = m.sink.textureWrites.length;
	const pipesBefore = m.sink.pipeOrder.length;
	sim.moveCollider(id, 48, 32, 120, -30);
	sim.step(1 / 60);
	const newWrites = m.sink.textureWrites.slice(writesBefore);
	const velWrite = newWrites.filter(w => w.label === "patchVel").pop();
	const flagsWrite = newWrites.filter(w => w.label === "flags").pop();
	check(flagsAt(40, 32) === 0 && flagsAt(48, 32) === 255 && flagsAt(52, 32) === 255,
		"the fine-grid push path also releases old cells and blocks new ones",
		"old=" + flagsAt(40, 32) + " new=" + flagsAt(48, 32));
	check(flagsWrite && flagsWrite.size[0] < N && flagsWrite.origin[0] > 0,
		"the fine-grid mask delta still uploads a patch, not the whole grid",
		flagsWrite ? "rect " + flagsWrite.size.join("x") + " at " + flagsWrite.origin.join(",") : "no write");
	const patch = velWrite ? new Float32Array(velWrite.data.buffer, velWrite.data.byteOffset, velWrite.data.length) : null;
	// the velocity patch lives at the patch texture's origin, so its grid origin is
	// the rect the solver recorded in the (shared) patch params
	const patchAt = (gx, gy) => {
		if (!velWrite) return null;
		const px = gx - sim.params.paintX, py = gy - sim.params.paintY;
		if (px < 0 || py < 0 || px >= velWrite.size[0] || py >= velWrite.size[1]) return null;
		const i = (px + py * velWrite.size[0]) * 4;
		return [patch[i], patch[i + 1], patch[i + 2]];
	};
	const bodyCell = patchAt(48, 32);
	const aheadCell = patchAt(53, 32);
	const trailingCell = patchAt(43, 32);
	const staged = m.sink.pipeOrder.slice(pipesBefore);
	check(staged.indexOf("colliderVel") >= 0,
		"a moving body stages the velocity imprint pass");
	check(staged.indexOf("colliderPushMass") >= 0 && staged.indexOf("colliderPushTherm") >= 0 &&
		newWrites.some(w => w.label === "patchGeo") && staged.indexOf("mgCoarsen") < 0 && sim.params.useMultigrid,
		"movement keeps pushing scalars and uses fine-grid pressure without changing config",
		staged.filter(n => n.indexOf("collider") === 0).join(","));
	// every pipeline the collider pass binds needs a group for each parity state, or the
	// device drops the whole command buffer and the body silently does not move
	const grp = sim.solver.groups;
	check(grp && grp.pushMass && grp.pushMass.length === 3 && grp.pushTherm.length === 2 &&
		grp.colliderVel.length === 2 && grp.pushMass.every(g => g && g._entries.length === 4),
		"the push and imprint bind groups exist for every parity the state can be in",
		"pushMass=" + (grp && grp.pushMass ? grp.pushMass.length : "?") +
		" pushTherm=" + (grp && grp.pushTherm ? grp.pushTherm.length : "?"));
	// a frame where the body does not move must cost nothing: no patch write, no pass
	const stillBefore = m.sink.pipeOrder.length;
	sim.step(1 / 60);
	const stillPasses = m.sink.pipeOrder.slice(stillBefore).filter(n => n.indexOf("collider") === 0);
	check(stillPasses.length === 0, "a parked body stages no collider pass at all", "passes=" + stillPasses.length);
	check(velWrite && bodyCell[2] === 1 && bodyCell[0] === 0 && bodyCell[1] === 0 &&
		aheadCell && aheadCell[2] === 1 && aheadCell[0] === 120 && aheadCell[1] === -30 &&
		trailingCell && trailingCell[2] === 0,
		"the imprint sets the body's own cell to rest and the fluid it pushes to the body's velocity",
		"body=" + (bodyCell && bodyCell.join(",")) + " ahead=" + (aheadCell && aheadCell.join(",")));

	// The cooldown counts frames since the last mask change, not substeps. After the
	// remaining 59 parked frames, the next frame returns to the requested multigrid path.
	for (let i = 0; i < 58; i++) sim.step(1 / 60);
	const resumedBefore = m.sink.pipeOrder.length;
	sim.step(1 / 60);
	const resumedPasses = m.sink.pipeOrder.slice(resumedBefore);
	check(sim.params.useMultigrid && resumedPasses.indexOf("mgCoarsen") >= 0,
		"multigrid resumes after the 60-frame settling window expires",
		"mgCoarsen=" + resumedPasses.includes("mgCoarsen"));

	// mass is a pressure boundary question, never a source: a teleport must not add
	const ledgerBefore = sim.solver.getStats().ledger.injectedM;
	sim.moveCollider(id, 56, 56, 0, 0);
	sim.step(1 / 60);
	check(flagsAt(40, 32) === 0 && flagsAt(56, 56) === 255 &&
		sim.solver.getStats().ledger.injectedM === ledgerBefore,
		"a teleport moves the mask and books no mass", "injected=" + sim.solver.getStats().ledger.injectedM);

	// static geometry and the painted wall a driving body must not erode
	const mask = new Uint8Array(N * N);
	for (let x = 0; x < N; x++) mask[x] = 1;
	check(sim.setObstacles(mask, N, N) === true && flagsAt(10, 1) === 255 && flagsAt(10, 2) === 0,
		"setObstacles uploads a static mask as walls");
	check(sim.setObstacles(mask, N - 1, N) === false,
		"a mask of the wrong size is refused instead of resampled");
	const wallAt = (x, y) => flagsAt(x, y);
	check(sim.paintObstacle(60, 60, 4, "wall") === true && wallAt(60, 60) === 255 &&
		sim.paintObstacle(60, 60, 4, "erase") === true && wallAt(60, 60) === 0 &&
		sim.paintObstacle(60, 60, 4, "solid") === false,
		"paintObstacle paints and erases a disc, and refuses a kind the EOS owns");
	check(sim.paintObstacle(48, 48, 3, "wall") === true && wallAt(48, 48) === 255,
		"paintObstacle paints where the game asked, not where the pointer is");
	const wallBody = sim.addCollider({ shape: "circle", x: 48, y: 48, radius: 3, vx: 0, vy: 0 });
	sim.moveCollider(wallBody, 56, 40, 0, 0);
	sim.step(1 / 60);
	check(wallAt(48, 48) === 255 && wallAt(56, 40) === 255,
		"a body leaving a painted wall restores the wall instead of erasing it");
	sim.destroy();

	// reset keeps the bodies and drops the painted mask, like it drops the scenario's
	const r = await makeSim({ substepsAuto: false });
	r.sim.addCollider({ shape: "circle", x: 20, y: 20, radius: 3 });
	r.sim.setGrid(128);
	check(r.sim.colliderCount() === 1 && r.sim.solver.flagsMirror[(20 + 20 * r.sim.params.texW) * 4] === 255,
		"setGrid rebuilds the mask arrays and re-paints the bodies");
	r.sim.reset("dam");
	const flagsAfterReset = r.sim.solver.flagsMirror[(20 + 20 * r.sim.params.texW) * 4];
	check(r.sim.colliderCount() === 1 && flagsAfterReset === 255,
		"reset keeps the bodies and paints them onto the fresh field");
	r.sim.destroy();
}

// a2: a body driven through water has to push it. the device here is the cpu oracle,
// which is the only thing that can show a wake without a gpu: it sees exactly the
// three writes the gpu path makes - the wall channel, rest inside the body, and the
// body's velocity on the fluid it is about to move into.
async function testColliderFlow() {
	const S = Fluid.core.solver, C = Fluid.core.colliders, acc = Fluid.core.stats;
	const cfg = Object.assign({}, Fluid.core.config.defaults, {
		N: 32, scenario: "dam", useMultigrid: false, dt: 1 / 60
	});
	cfg.texW = cfg.N + 2; cfg.texH = cfg.N + 2;
	const sim = S.create(cfg);
	const table = C.create();
	const id = C.add(table, cfg, { shape: "circle", x: 10, y: 16, radius: 4, vx: 14, vy: 0 });
	const body = table.slots[id];
	const tw = cfg.texW, th = cfg.texH, N = cfg.N;
	const cols = () => {
		const f = Fluid.core.grid.curBuf(sim.mass), out = new Float64Array(tw);
		for (let x = 1; x <= N; x++) {
			let sum = 0;
			for (let y = 1; y <= N; y++) sum += f[(x + y * tw) * 4 + 3];
			out[x] = sum;
		}
		return out;
	};
	// one frame of the collider half, in the order the gpu path runs it: rasterize,
	// push the contents of the newly taken cells out, imprint the body's velocity,
	// then the substep itself
	const owner = new Uint8Array(tw * th), prev = new Uint8Array(tw * th);
	const geo = new Float32Array(128 * 128 * 4);
	const scratch = C.makeScratch();
	const rect = {};
	const massF = Fluid.core.grid.curBuf(sim.mass), thermF = Fluid.core.grid.curBuf(sim.therm);
	const drive = (imprint) => {
		body.x += body.vx * cfg.dt; body.y += body.vy * cfg.dt;
		owner.fill(0);
		C.rasterize(table, cfg, owner, rect);
		const x0 = Math.max(1, rect.x0 - 1), x1 = Math.min(tw - 2, rect.x1 + 1);
		const y0 = Math.max(1, rect.y0 - 1), y1 = Math.min(th - 2, rect.y1 + 1);
		if (imprint) {
			const pr = C.padRect(cfg, C.rasterize(table, cfg, owner, {}));
			rect.x0 = pr.x0; rect.y0 = pr.y0; rect.x1 = pr.x1; rect.y1 = pr.y1;
			C.geometry(cfg, owner, prev, rect, geo);
			C.displace(cfg, geo, rect, massF, thermF, scratch);
		}
		for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) sim.flags[(x + y * tw) * 4] = owner[x + y * tw] ? 255 : 0;
		const vel = Fluid.core.grid.curBuf(sim.vel);
		for (let y = y0; y <= y1; y++) {
			for (let x = x0; x <= x1; x++) {
				const k = x + y * tw;
				if (owner[k]) {
					vel[k * 2] = 0; vel[k * 2 + 1] = 0;
				} else if (imprint && !prev[k] && (owner[k - 1] || owner[k + 1] || owner[k - tw] || owner[k + tw])) {
					vel[k * 2] = body.vx; vel[k * 2 + 1] = body.vy;
				}
			}
		}
		prev.set(owner);
		sim.step(cfg.dt);
	};
	const before = cols();
	for (let s = 0; s < 60; s++) drive(true);
	const after = cols();
	let ahead = 0, behind = 0, rise = 0, hollow = 0;
	for (let x = 1; x <= N; x++) {
		const d = after[x] - before[x];
		if (x > body.x + 2) { ahead += d; rise = Math.max(rise, d); }
		if (x < body.x - 2) { behind += d; hollow = Math.min(hollow, d); }
	}
	check(ahead > 0 && behind < 0 && rise > 2 && hollow < -2,
		"a body driven through water piles it up ahead and leaves a hollow behind",
		"ahead=" + ahead.toFixed(1) + " behind=" + behind.toFixed(1) + " rise=" + rise.toFixed(1) + " hollow=" + hollow.toFixed(1));
	const st = acc.totalsFromFields(Fluid.core.grid.curBuf(sim.mass), Fluid.core.grid.curBuf(sim.therm), tw, th);
	check(Math.abs(acc.driftRel(st.sumM, acc.expectedTotal(sim.ledger, "m"))) < 1e-6,
		"the wake moves water without creating any: drift_rel stays at roundoff",
		"drift_rel=" + Math.abs(acc.driftRel(st.sumM, acc.expectedTotal(sim.ledger, "m"))).toExponential(2));
	// a teleport is a mask change with no imprint: the fluid it skipped over is not
	// dragged along, and above all no substance is booked
	body.x = 8; body.y = 8; body.vx = 0; body.vy = 0;
	const massBefore = st.sumM;
	drive(false);
	const st2 = acc.totalsFromFields(Fluid.core.grid.curBuf(sim.mass), Fluid.core.grid.curBuf(sim.therm), tw, th);
	const teleDrift = Math.abs(acc.driftRel(st2.sumM, acc.expectedTotal(sim.ledger, "m")));
	check(Math.abs(st2.sumM - acc.expectedTotal(sim.ledger, "m")) < 1e-3 && teleDrift < 1e-6,
		"a teleport books no mass and leaves the ledger's injected total alone",
		"sumM=" + st2.sumM.toFixed(3) + " drift_rel=" + teleDrift.toExponential(2));
	check(Math.abs(st2.sumM - massBefore) / massBefore < 1e-6,
		"the teleport frame moves less water than the drift floor allows",
		"delta=" + (st2.sumM - massBefore).toExponential(2));
	// a wall is a boundary, not a drain: the fluid already inside the cells a body
	// covers stays put, which is exactly why a teleport cannot invent substance
	const still = Object.assign({}, Fluid.core.config.defaults, { N: 24, scenario: "still", useMultigrid: false, dt: 1 / 60 });
	still.texW = still.N + 2; still.texH = still.N + 2;
	const s2 = S.create(still);
	const table2 = C.create();
	C.add(table2, still, { shape: "circle", x: 12, y: 12, radius: 4 });
	const owner2 = new Uint8Array(still.texW * still.texH), rect2 = {};
	C.rasterize(table2, still, owner2, rect2);
	const sm = Fluid.core.grid.curBuf(s2.mass);
	const k = 12 + 12 * still.texW;
	const cell = k * 4;
	const held = sm[cell + 3];
	for (let y = 1; y <= still.N; y++) for (let x = 1; x <= still.N; x++) if (owner2[x + y * still.texW]) s2.flags[(x + y * still.texW) * 4] = 255;
	for (let i = 0; i < 5; i++) s2.step(still.dt);
	const sm2 = Fluid.core.grid.curBuf(s2.mass);
	const sv = Fluid.core.grid.curBuf(s2.vel);
	const st3 = acc.totalsFromFields(sm2, Fluid.core.grid.curBuf(s2.therm), still.texW, still.texH);
	check(held > 0.5 && Math.abs(sm2[cell + 3] - held) < 1e-6 && sv[k * 2] === 0 && sv[k * 2 + 1] === 0,
		"a wall is a boundary, not a drain: the core solver leaves the fluid it covers alone",
		"m " + held.toFixed(3) + "->" + sm2[cell + 3].toFixed(3) + " v=" + sv[k * 2].toFixed(3) +
		" drift=" + Math.abs(acc.driftRel(st3.sumM, acc.expectedTotal(s2.ledger, "m"))).toExponential(2));
}

// When a moving body changes the mask, production temporarily selects the fine-grid
// pressure solve while keeping the conservative mass/thermal push. Exercise that
// numerical path here; the recorder test above verifies the automatic dispatch choice.
async function testColliderFineFallback() {
	const S = Fluid.core.solver, C = Fluid.core.colliders, acc = Fluid.core.stats, grid = Fluid.core.grid;
	const cfg = Object.assign({}, Fluid.core.config.defaults, {
		N: 64, scenario: "still", useMultigrid: false, dt: 1 / 60, gravityY: 0
	});
	cfg.texW = cfg.N + 2; cfg.texH = cfg.N + 2;
	const sim = S.create(cfg), table = C.create();
	const id = C.add(table, cfg, {
		shape: "box", x: 32, y: 27, hw: 2, hh: 4, vx: 0, vy: 0
	});
	const body = table.slots[id], tw = cfg.texW, N = cfg.N;
	const owner = new Uint8Array(tw * cfg.texH), prev = new Uint8Array(owner.length), rect = {};
	const wide = { x0: 1, y0: 1, x1: N, y1: N };
	const geo = new Float32Array(tw * cfg.texH * 4), scratch = C.makeScratch(tw * cfg.texH);
	C.rasterize(table, cfg, owner, rect);
	prev.set(owner);
	for (let k = 0; k < owner.length; k++) if (owner[k]) sim.flags[k * 4] = 255;
	let maxM = 0, maxVel = 0, maxDiv = 0, finite = true;
	for (let frame = 0; frame < 60; frame++) {
		C.move(table, cfg, id, body.x + 0.235, body.y, 14.1, 0);
		owner.fill(0);
		C.rasterize(table, cfg, owner, rect);
		C.geometry(cfg, owner, prev, wide, geo, tw);
		C.displace(cfg, geo, wide, grid.curBuf(sim.mass), grid.curBuf(sim.therm), scratch, tw);
		const vel = grid.curBuf(sim.vel);
		for (let y = 1; y <= N; y++) {
			for (let x = 1; x <= N; x++) {
				const k = x + y * tw, i = k * 4;
				const now = owner[k] !== 0, was = prev[k] !== 0;
				sim.flags[i] = now ? 255 : 0;
				if (!now || was) continue;
				vel[k * 2] = 0; vel[k * 2 + 1] = 0;
				for (const nk of [k - 1, k + 1, k - tw, k + tw]) {
					if (owner[nk] || prev[nk]) continue;
					vel[nk * 2] = body.vx; vel[nk * 2 + 1] = body.vy;
				}
			}
		}
		prev.set(owner);
		sim.phaseEos(false);
		sim.step(cfg.dt);
		const st = sim.stats();
		finite = finite && Number.isFinite(st.tot.sumM) && Number.isFinite(st.tot.maxM) &&
			Number.isFinite(st.mot.maxVel) && Number.isFinite(st.mot.maxDiv);
		maxM = Math.max(maxM, st.tot.maxM);
		maxVel = Math.max(maxVel, st.mot.maxVel);
		maxDiv = Math.max(maxDiv, st.mot.maxDiv);
	}
	// Match production: once the 60-frame fine-grid window expires, the parked
	// pushed field returns to the requested multigrid solve and must remain finite.
	cfg.useMultigrid = true;
	cfg.mgCoarseIters = 20;
	cfg.mgCycles = 2;
	for (let frame = 0; frame < 60; frame++) {
		sim.phaseEos(false);
		sim.step(cfg.dt);
		const st = sim.stats();
		finite = finite && Number.isFinite(st.tot.sumM) && Number.isFinite(st.tot.maxM) &&
			Number.isFinite(st.mot.maxVel) && Number.isFinite(st.mot.maxDiv);
		maxM = Math.max(maxM, st.tot.maxM);
		maxVel = Math.max(maxVel, st.mot.maxVel);
		maxDiv = Math.max(maxDiv, st.mot.maxDiv);
	}
	const st = sim.stats();
	const drift = Math.abs(acc.driftRel(st.tot.sumM, acc.expectedTotal(sim.ledger, "m")));
	check(finite && maxVel > 5 && maxVel < 32,
		"fine-grid settling frames carry a moving collider's velocity into the fluid",
		"finite=" + finite + " peakVel=" + maxVel.toFixed(2));
	check(maxM < 25 && maxDiv < 250 && drift < 1e-6,
		"the pushed field stays finite through the fine-grid window and multigrid resumption",
		"maxM=" + maxM.toFixed(3) + " maxDiv=" + maxDiv.toFixed(2) + " drift_rel=" + drift.toExponential(2));
}

// a2's visible half: a body driven across a dye patch must move the dye, not leave it
// frozen under a hole. one run with the push passes, one without (the t4 behaviour):
// the dye total has to come out the same either way, and the dye left in the cells the
// body crossed has to be gone.
async function testColliderDye() {
	const S = Fluid.core.solver, C = Fluid.core.colliders, acc = Fluid.core.stats, grid = Fluid.core.grid;
	function run(push) {
		const cfg = Object.assign({}, Fluid.core.config.defaults, {
			N: 32, scenario: "still", useMultigrid: false, dt: 1 / 60, gravityX: 0, gravityY: 0
		});
		cfg.texW = cfg.N + 2; cfg.texH = cfg.N + 2;
		const sim = S.create(cfg);
		const tw = cfg.texW, N = cfg.N;
		const mass = grid.curBuf(sim.mass), therm = grid.curBuf(sim.therm);
		for (let y = 1; y <= N; y++) {
			for (let x = 1; x <= N; x++) {
				if ((x - 10) * (x - 10) + (y - 16) * (y - 16) < 36) {
					const i = (x + y * tw) * 4;
					mass[i] = 0.6;
					therm[i] = S.energyFromT(mass[i + 3], 420, cfg);
					therm[i + 1] = 420;
				}
			}
		}
		const table = C.create();
		const body = table.slots[C.add(table, cfg, { shape: "box", x: 5, y: 16, hw: 2, hh: 5, vx: 20, vy: 0 })];
		const owner = new Uint8Array(tw * cfg.texH), prev = new Uint8Array(tw * cfg.texH);
		const swept = new Uint8Array(tw * cfg.texH);
		const geo = new Float32Array(128 * 128 * 4), scratch = C.makeScratch();
		for (let s = 0; s < 45; s++) {
			body.x += body.vx * cfg.dt;
			owner.fill(0);
			const rect = C.padRect(cfg, C.rasterize(table, cfg, owner, {}));
			if (push) {
				C.geometry(cfg, owner, prev, rect, geo, 128);
				C.displace(cfg, geo, rect, grid.curBuf(sim.mass), grid.curBuf(sim.therm), scratch, 128);
			}
			for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < tw - 1; x++) sim.flags[(x + y * tw) * 4] = owner[x + y * tw] ? 255 : 0;
			const vel = grid.curBuf(sim.vel);
			for (let y = 1; y < cfg.texH - 1; y++) {
				for (let x = 1; x < tw - 1; x++) {
					const k = x + y * tw;
					if (owner[k]) {
						vel[k * 2] = 0; vel[k * 2 + 1] = 0;
					} else if (!prev[k] && (owner[k - 1] || owner[k + 1] || owner[k - tw] || owner[k + tw])) {
						vel[k * 2] = body.vx; vel[k * 2 + 1] = 0;
					}
				}
			}
			for (let k = 0; k < swept.length; k++) if (owner[k]) swept[k] = 1;
			prev.set(owner);
			sim.step(cfg.dt);
		}
		const m = grid.curBuf(sim.mass);
		let total = 0, inside = 0;
		for (let y = 1; y <= N; y++) {
			for (let x = 1; x <= N; x++) {
				const dye = m[(x + y * tw) * 4];
				total += dye;
				if (swept[x + y * tw]) inside += dye;
			}
		}
		const finalTherm = grid.curBuf(sim.therm);
		let heatTotal = 0, heatInside = 0;
		for (let y = 1; y <= N; y++) {
			for (let x = 1; x <= N; x++) {
				const i = (x + y * tw) * 4;
				const excess = Math.max(0, finalTherm[i] - S.energyFromT(m[i + 3], cfg.TRef, cfg));
				heatTotal += excess;
				if (swept[x + y * tw]) heatInside += excess;
			}
		}
		const t = acc.totalsFromFields(m, finalTherm, tw, cfg.texH);
		return { total, inside, heatTotal, heatInside, mass: t.sumM, ledger: acc.expectedTotal(sim.ledger, "m") };
	}
	const pushed = run(true), frozen = run(false);
	check(pushed.total > 0 && Math.abs(pushed.total - frozen.total) / frozen.total < 1e-6,
		"pushing the dye around keeps the same dye: none is created or lost",
		"push=" + pushed.total.toFixed(2) + " frozen=" + frozen.total.toFixed(2));
	// the pushed dye can flow back into the swept cells afterwards, so the bound is a
	// contrast (most of it stays vs most of it has left), not an absolute zero
	check(frozen.inside / frozen.total > 0.7 && pushed.inside / pushed.total < 0.25,
		"the dye a body crosses moves out of the cells it left instead of staying put",
		"in the swept cells: frozen " + (100 * frozen.inside / frozen.total).toFixed(1) + "% -> pushed " +
		(100 * pushed.inside / pushed.total).toFixed(1) + "%");
	check(frozen.heatInside / frozen.heatTotal > 0.6 && pushed.heatInside / pushed.heatTotal < 0.35,
		"the same body push carries the hot temperature marker out of the cells it crosses",
		"excess heat in swept cells: frozen " + (100 * frozen.heatInside / frozen.heatTotal).toFixed(1) +
		"% -> pushed " + (100 * pushed.heatInside / pushed.heatTotal).toFixed(1) + "%");
	check(Math.abs(pushed.mass - pushed.ledger) < 1e-3,
		"the frame that pushes dye still balances against the ledger", "sumM=" + pushed.mass.toFixed(3));
}

// examples/demo.html is the API by example. Boot the real demo script over the
// real engine and the recording device: the page's own handlers drive probes,
// pause, view and scenario, and the numbers on the page come from the same
// readbacks a browser would see. Only device acquisition is stubbed.
class El {
	constructor(tag) {
		this.tagName = tag.toUpperCase();
		this.children = [];
		this.listeners = {};
		this.textContent = "";
		this.hidden = false;
		this.className = "";
		this.value = "";
		this.style = {};
		this.classList = { toggle: () => {} };
	}
	appendChild(child) { this.children.push(child); return child; }
	focus() {}
	setPointerCapture() {}
	hasPointerCapture() { return false; }
	releasePointerCapture() {}
	addEventListener(type, fn) { (this.listeners[type] || (this.listeners[type] = [])).push(fn); }
	removeEventListener(type, fn) {
		const list = this.listeners[type];
		const i = list ? list.indexOf(fn) : -1;
		if (i >= 0) list.splice(i, 1);
	}
	getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; }
	fire(type, props) {
		for (const fn of (this.listeners[type] || []).slice()) fn(Object.assign({ preventDefault() {} }, props));
	}
}
async function testDemoPage() {
	const origWindow = globalThis.window, origDocument = globalThis.document;
	const origRaf = globalThis.requestAnimationFrame, origCancel = globalThis.cancelAnimationFrame;
	const els = {};
	for (const id of ["c", "status", "error-panel", "scenario", "pause", "step", "reset", "clear-probes", "views", "probe-body", "meters", "emitters", "body"]) {
		els[id] = new El(id === "c" ? "canvas" : "div");
	}
	const win = new El("window");
	globalThis.window = win;
	globalThis.document = {
		readyState: "complete",
		getElementById: id => els[id] || null,
		createElement: tag => new El(tag),
		addEventListener() {}, removeEventListener() {},
		hidden: false
	};
	let raf = null;
	globalThis.requestAnimationFrame = fn => { raf = fn; return 1; };
	globalThis.cancelAnimationFrame = () => {};
	// the mock never executes a shader: these arrays stand in for what the
	// probe and stats kernels would have written, and the page must read them
	const m = mock.create({ stats: statsPattern(), probes: probePattern() });
	m.device.queue = stubQueue(m.sink);
	const origInit = Fluid.gpu.device.init;
	const origCreate = Fluid.engine.create;
	Fluid.gpu.device.init = async () => ({ device: m.device, context: mock.mockContext(), format: "bgra8unorm" });
	let sim = null;
	Fluid.engine.create = async (canvas, options) => { sim = await origCreate(canvas, options); return sim; };
	let now = performance.now();
	// one animation callback plus the queue completion and readback it queues
	async function frames(n) {
		for (let i = 0; i < n; i++) {
			const fn = raf;
			if (!fn) break;
			raf = null;
			// the page's ui sweep runs every 200 ms of callback time: step far
			// enough per frame that a handful of frames covers a few sweeps
			now += 100;
			fn(now);
			m.device.queue.completeAll();
			await flush(6);
		}
		await flush(6);
	}
	try {
		const demoPath = path.join(root, "examples/demo.js");
		delete require.cache[require.resolve(demoPath)];
		require(demoPath);
		await flush(20);
		check(sim !== null && sim.params.N === 256 && els.status.textContent.indexOf("N=256") === 0,
			"demo: boots the real engine on the page's cfg and reports the live setup",
			els.status.textContent);
		check(sim.solver !== null && els.scenario.children.length === Fluid.core.config.scenarioNames.length,
			"demo: fills the scenario select from the config's own scenario list",
			"options=" + els.scenario.children.length);
		const pool = new Float32Array(sim.solver.pool.emitters._buf);
		check(sim.emitterCount() === 6 && pool[0] === 6 && els.emitters.textContent === "Emitters (6)" &&
			sim.params.scenario === "still",
			"demo: the playground installs its heaters, cold plates and dye jet through addEmitter",
			"live=" + pool[0] + " label=" + els.emitters.textContent);
		await frames(6);
		els.c.fire("contextmenu", { clientX: 50, clientY: 50 });
		check(sim.probeCount() === 1 && sim.solver.probes.slots[0].x === 128 && sim.solver.probes.slots[0].y === 128,
			"demo: a right-click maps the canvas to a cell probe (+y up) through the engine",
			"cells=" + sim.solver.probes.slots[0].x + "," + sim.solver.probes.slots[0].y);
		await frames(8);
		const row = els["probe-body"].children[0];
		const sample = Fluid.core.probes.makeSample();
		const sampled = sim.sample(0, sample);
		check(row && row.hidden === false && row.children[0].textContent === "0" &&
			row.children[2].textContent === "1.000" && row.children[4].textContent === "liquid" &&
			sampled && Math.abs(sample.m - 1) < 1e-6,
			"demo: the table shows the readback the engine ingested (m, T, phase, u, v, p)",
			row ? row.children.map(c => c.textContent).join(" | ") : "no row");
		check(/mass [\+\-]/.test(els.meters.children[0].textContent) && els.meters.children[2].textContent.indexOf("neg=") === 0,
			"demo: the conservation meters come from stats(), not from the solver's internals",
			els.meters.children[0].textContent + " · " + els.meters.children[2].textContent);
		// the moving body: the button adds one collider, the frame loop drives it, and a
		// second click removes it (the page keeps its own position, the engine the slot)
		check(sim.colliderCount() === 0, "demo: the playground starts without a body", "live=" + sim.colliderCount());
		els.body.fire("click");
		const bodyId = sim.colliderCount() === 1 ? 0 : -1;
		const bodyX = bodyId >= 0 ? sim.solver.colliders.slots[bodyId].x : 0;
		check(bodyId === 0 && els.body.textContent === "Body (on)",
			"demo: the Body button adds one collider through addCollider", els.body.textContent);
		const frameBefore = sim.solver.getStats().frame;
		await frames(3);
		const bodyMoved = bodyId >= 0 ? sim.solver.colliders.slots[bodyId].x : 0;
		check(sim.solver.getStats().frame > frameBefore && bodyMoved > bodyX,
			"demo: the frame loop steps and moves the body through moveCollider",
			bodyX + " -> " + bodyMoved + " frames " + frameBefore + " -> " + sim.solver.getStats().frame);
		const bodyY = Math.round(0.42 * sim.params.N);
		const wallAtBody = sim.solver.flagsMirror[(bodyMoved + bodyY * sim.params.texW) * 4];
		check(wallAtBody === 255,
			"demo: the body's cells are walls in the mask the solver holds",
			"at " + bodyMoved + "," + bodyY + " flag=" + wallAtBody + " frame=" + sim.solver.getStats().frame);
		els.body.fire("click");
		check(sim.colliderCount() === 0 && els.body.textContent === "Body (off)",
			"demo: a second click removes the body and its mask", els.body.textContent);
		els.pause.fire("click");
		await frames(3);
		check(sim.paused === true && els.pause.textContent === "Resume",
			"demo: the pause button drives the engine's paused state and its own label");
		els.views.children[3].fire("click");
		check(sim.params.viewMode === 3, "demo: a view button writes cfg.viewMode through setViewMode", "mode=" + sim.params.viewMode);
		els.scenario.value = "cycle";
		els.scenario.fire("change");
		check(sim.params.scenario === "cycle" && sim.solver.getStats().frame === 0,
			"demo: the scenario select resets through the engine", sim.params.scenario);
		const emittersBefore = m.sink.pipeOrder.length;
		await frames(4);
		check(m.sink.pipeOrder.slice(emittersBefore).indexOf("emitterMass") >= 0,
			"demo: a frame with the playground's sources stages the emitter passes");
		els.emitters.fire("click");
		check(sim.emitterCount() === 0 && els.emitters.textContent === "Emitters (off)",
			"demo: the Emitters button removes every slot through the engine");
		await frames(3);
		const emittersOff = m.sink.pipeOrder.length;
		await frames(3);
		check(m.sink.pipeOrder.slice(emittersOff).every(p => p.indexOf("emitter") !== 0),
			"demo: with the sources off the frame stages no emitter pass");
		els.emitters.fire("click");
		check(sim.emitterCount() === 6 && els.emitters.textContent === "Emitters (6)",
			"demo: the button installs them again in the same slots");
		els["clear-probes"].fire("click");
		check(sim.probeCount() === 0 && els["probe-body"].children[0].hidden === true,
			"demo: clear drops every probe slot and hides the rows");
		win.fire("pagehide");
		check(sim.destroyed === true && m.counts.buffersAlive === 0 && m.counts.texturesAlive === 0,
			"demo: leaving the page destroys the engine and its pool",
			"alive buffers=" + m.counts.buffersAlive);
	} finally {
		Fluid.gpu.device.init = origInit;
		Fluid.engine.create = origCreate;
		globalThis.window = origWindow;
		globalThis.document = origDocument;
		globalThis.requestAnimationFrame = origRaf;
		globalThis.cancelAnimationFrame = origCancel;
	}
}

(async function main() {
	await testPacingMeter();
	await testEngineCreate();
	await testPacing();
	await testStatsAndProbes();
	await testEmitters();
	await testColliders();
	await testColliderFlow();
	await testColliderFineFallback();
	await testColliderDye();
	await testViewAndGrid();
	await testInputLifecycle();
	await testTeardown();
	await testLabWiring();
	await testDemoPage();
	console.log("all " + checks + " engine checks pass");
})().catch(error => {
	console.error(error);
	process.exitCode = 1;
});
