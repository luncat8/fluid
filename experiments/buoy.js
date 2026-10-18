"use strict";
// buoy measurement (draft.txt: "buoy stays partially submerged in blue, the amount of
// blue inside it corresponds to the buoy volume"). drives src/core/buoy.js against the
// cpu reference solver with the cadence the app uses: the body is marched every frame,
// the field sample is refreshed every cfg.buoySampleEvery frames and ingested one
// frame later, which is the gpu readback latency.
//
// one run per grid size drives many buoy variants: the dye is a passive tracer, the
// flow does not feel it, so scaling the sampled blue is exactly what a bluer or a
// dimmer scenario would give the body, and the sampling interval is a body-side knob.
// run: node experiments/buoy.js | tee experiments/logs/<date>-buoy.log

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const buoyMod = require(path.join(root, "src/core/buoy.js"));
const stats = Fluid.core.stats;
const cfg0 = Fluid.core.config.defaults;

function makeCfg(over) {
	const c = Object.assign({}, cfg0, over);
	c.texW = c.N + 2; c.texH = c.N + 2;
	c.substepsAuto = true;
	return c;
}
// one fluid run, one body per variant; returns per-variant metrics
function measure(cfgOver, specs) {
	const cfg = makeCfg(cfgOver);
	const sim = solverMod.create(cfg);
	const runs = specs.map(sp => {
		const vcfg = Object.assign({}, cfg, sp.cfg);
		const buoy = buoyMod.create(vcfg);
		return {
			name: sp.name, blue: sp.blue === undefined ? 1 : sp.blue, cfg: vcfg, buoy,
			pending: new Float32Array(buoyMod.SAMPLE_LEN), ready: false,
			maxStep: 0, prevY: buoy.state.cy, prevSub: 0, settle: 0, subBand: [1, 0], trace: []
		};
	});
	const frames = cfgOver.frames;
	for (let f = 0; f < frames; f++) {
		const mot = stats.motionTotals(sim.vel.cur === 0 ? sim.vel.a : sim.vel.b,
			sim.press.cur === 0 ? sim.press.a : sim.press.b, cfg.texW, cfg.texH, cfg.dt);
		let ns = Math.ceil(mot.maxVel * cfg.frameDt / cfg.cflTarget);
		ns = ns < 1 ? 1 : (ns > cfg.substepMax ? cfg.substepMax : ns);
		cfg.dt = cfg.frameDt / ns;
		for (let s = 0; s < ns; s++) sim.step(cfg.dt);
		const mass = sim.mass.cur === 0 ? sim.mass.a : sim.mass.b;
		const vel = sim.vel.cur === 0 ? sim.vel.a : sim.vel.b;
		for (const r of runs) update(r, mass, vel, f, frames);
	}
	return { cfg, sim, runs };
}
function update(r, mass, vel, f, frames) {
	const b = r.buoy;
	if (r.ready) { b.ingest(r.pending); r.ready = false; }
	b.march(r.cfg.frameDt);
	if (b.state.active && f % Math.max(1, r.cfg.buoySampleEvery) === 0) {
		b.sampleCpu(mass, vel, r.cfg.texW, r.cfg.texH, r.pending);
		if (r.blue !== 1) { r.pending[0] *= r.blue; r.pending[5] *= r.blue; }
		r.ready = true;
	}
	const d = Math.abs(b.state.cy - r.prevY); r.prevY = b.state.cy;
	if (d > r.maxStep) r.maxStep = d;
	if (f > 60) {
		r.subBand[0] = Math.min(r.subBand[0], b.state.sub);
		r.subBand[1] = Math.max(r.subBand[1], b.state.sub);
	}
	// "settled" = the last frame where the float line still moved by over 1% of the rect
	if (Math.abs(b.state.sub - r.prevSub) > 0.01) r.settle = f;
	r.prevSub = b.state.sub;
	if (f % 50 === 0 || f === frames - 1) {
		r.trace.push("    f=" + f + " cy=" + b.state.cy.toFixed(2) + " vy=" + b.state.vy.toFixed(2) +
			" sub=" + (100 * b.state.sub).toFixed(1) + "% blue/cell=" + b.state.blue.toFixed(3));
	}
}
function line(r) {
	const s = r.buoy.state;
	return "  " + (r.name + "                    ").slice(0, 22) + " sub=" + (100 * s.sub).toFixed(1) +
		"% [" + (100 * r.subBand[0]).toFixed(0) + ".." + (100 * r.subBand[1]).toFixed(0) + "]" +
		" cy=" + s.cy.toFixed(1) + " vy=" + s.vy.toFixed(2) + " maxStep=" + r.maxStep.toFixed(3) +
		" settled@f=" + r.settle + " samples=" + s.samples;
}

const specs = [
	{name: "sample every 1", cfg: {buoySampleEvery: 1}},
	{name: "sample every 2", cfg: {buoySampleEvery: 2}},
	{name: "sample every 4 (default)", cfg: {buoySampleEvery: 4}},
	{name: "sample every 8", cfg: {buoySampleEvery: 8}},
	{name: "sample every 16", cfg: {buoySampleEvery: 16}},
	{name: "sample every 32", cfg: {buoySampleEvery: 32}},
	{name: "blue x0.25 ([r,g,64])", blue: 0.25},
	{name: "blue x0.5 ([r,g,128])", blue: 0.5},
	{name: "blue x1 ([r,g,255])", blue: 1},
	{name: "blue x2", blue: 2},
	{name: "rho 0.1 (lighter hull)", cfg: {buoyRho: 0.1}},
	{name: "rho 0.4 (heavier hull)", cfg: {buoyRho: 0.4}},
	{name: "drag tau 0.04 (loose)", cfg: {buoyDrag: 0.04}},
	{name: "drag tau 0.5 (stiff)", cfg: {buoyDrag: 0.5}}
];
console.log("=== dam break, N=64, 600 frames, buoy scaled to " +
	JSON.stringify(buoyMod.scaledSize(makeCfg({N: 64, scenario: "dam"}))) + " ===");
const small = measure({N: 64, scenario: "dam", frames: 600, pressureIters: 30}, specs);
for (const r of small.runs) console.log(line(r));
console.log("--- float trace, default variant (sub = mean blue in the rect / local peak) ---");
for (const t of small.runs[2].trace) console.log(t);

console.log("=== same on N=128 (rect 4x16), 400 frames ===");
const big = measure({N: 128, scenario: "dam", frames: 400, pressureIters: 30},
	[specs[2], specs[6], specs[7], specs[8], specs[9]]);
for (const r of big.runs) console.log(line(r));
console.log("--- float trace, default variant ---");
for (const t of big.runs[0].trace) console.log(t);
