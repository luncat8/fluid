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

// a queue whose completion the test decides, so backpressure is observable
function stubQueue() {
	const pending = [];
	return {
		inflight: () => pending.length,
		onSubmittedWorkDone() { return new Promise(resolve => pending.push(resolve)); },
		writeBuffer(buf, offset, src) { new Float32Array(buf._buf, offset, src.length).set(src); },
		writeTexture() {},
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
	m.device.queue = stubQueue();
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
		const keyTarget = new Target();
		const pointerBefore = canvas.count();
		const keys = sim.bindKeys(keyTarget);
		check(keys && (keyTarget.listeners.keydown || []).length === 1 && canvas.count() === pointerBefore,
			"bindKeys installs keyboard handlers on the requested element without pointer listeners");
		keyTarget.listeners.keydown[0]({ key: "2", target: null, preventDefault() {} });
		check(mode.seen === 2, "bindKeys routes its view shortcuts through the same writer", "mode=" + mode.seen);
		sim.detachKeys();
		check(keyTarget.count() === 0 && canvas.count() === pointerBefore,
			"detachKeys removes its handlers and leaves canvas pointer ownership untouched");
		sim.attachInput(canvas);
		const pointerListeners = canvas.count();
		const windowKeys = (win.listeners.keydown || []).length;
		sim.bindKeys(keyTarget);
		check(canvas.count() === pointerListeners && (keyTarget.listeners.keydown || []).length === 1 &&
			(win.listeners.keydown || []).length === windowKeys - 1,
			"bindKeys replaces attachInput's key owner while preserving its pointer owner");
		sim.detachKeys();
		check(canvas.count() === pointerListeners && (canvas.listeners.pointerdown || []).length === 1,
			"detachKeys leaves an attached pointer stroke handler active");
		sim.destroy();
		check(win.count() + canvas.count() === before && keyTarget.count() === 0,
			"destroy() detaches all input owners before releasing the pool");
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
	for (const id of ["c", "status", "error-panel", "scenario", "pause", "step", "reset", "clear-probes", "views", "probe-body", "meters"]) {
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
	m.device.queue = stubQueue();
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
