"use strict";
// 0.6.0 t3/t11 measurement: projection residual vs solver work on the CPU oracle.
// run: node experiments/mg-work.js | tee experiments/logs/<date>-mg-work.log
//
// Each row runs a developed violent scenario with one solver setting and reports
// the median relative residual diag.pressureResidual (divAfter/divBefore) left by
// each projection, the absolute divergence, whole-step wall time, and the work in
// GPU terms: fine-grid sweep equivalents and dispatch count per projection, using
// the encodeStep structure (coarse level = 1/S^2 cells; restrict/coarsen read the
// fine grid once per coarse cell, so they count as ~1 fine equivalent each).
// The question: how much of the current default (2 V-cycles, pre4/post8, K=60
// coarse sweeps) is buying residual, and what is the cheapest setting that holds
// the same divergence on the scenarios that dominate hardware frame time.

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/multigrid.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const stats = Fluid.core.stats;
const cfg0 = Fluid.core.config.defaults;

const DT = 1 / 240;
const CONFIGS = [
	{ name: "cur c2 pre4 post8 K60", over: { useMultigrid: true, mgCycles: 2, mgPre: 4, mgPost: 8, pressureIters: 60 } },
	{ name: "c1 pre4 post8 K60", over: { useMultigrid: true, mgCycles: 1, mgPre: 4, mgPost: 8, pressureIters: 60 } },
	{ name: "c1 pre2 post4 K60", over: { useMultigrid: true, mgCycles: 1, mgPre: 2, mgPost: 4, pressureIters: 60 } },
	{ name: "c1 pre4 post8 K20", over: { useMultigrid: true, mgCycles: 1, mgPre: 4, mgPost: 8, pressureIters: 20 } },
	{ name: "c1 pre2 post4 K20", over: { useMultigrid: true, mgCycles: 1, mgPre: 2, mgPost: 4, pressureIters: 20 } },
	{ name: "RB K60", over: { useMultigrid: false, pressureIters: 60 } },
	{ name: "RB K20", over: { useMultigrid: false, pressureIters: 20 } }
];

function gpuWork(c) {
	// {feq: fine sweep equivalents, disp: dispatches} per projection, encodeStep shape
	if (!c.useMultigrid) return { feq: c.pressureIters, disp: 2 * c.pressureIters };
	const S2 = (c.mgCoarsen || 4) * (c.mgCoarsen || 4);
	const cyc = c.mgCycles, K = c.pressureIters;
	const feq = cyc * (c.mgPre + c.mgPost) + cyc * K / S2 + cyc * 2 + 1;
	const disp = 1 + cyc * (2 * (c.mgPre + c.mgPost) + 2 + 2 * K + 1);
	return { feq, disp };
}
function median(a) {
	const s = a.slice().sort((x, y) => x - y);
	return s[Math.floor(s.length / 2)];
}
function p95(a) {
	const s = a.slice().sort((x, y) => x - y);
	return s[Math.min(s.length - 1, Math.floor(s.length * 0.95))];
}
function runRow(scenario, N, warmup, measure, item) {
	const cfg = Object.assign({}, cfg0, item.over, { N, scenario });
	cfg.texW = N + 2; cfg.texH = N + 2;
	Fluid.core.config.prepareScenario(cfg);
	const sim = solverMod.create(cfg);
	for (let s = 0; s < warmup; s++) sim.step(DT);
	const res = [], div = [];
	const t0 = Date.now();
	for (let s = 0; s < measure; s++) {
		sim.step(DT);
		res.push(sim.diag.pressureResidual);
		div.push(sim.diag.divAfter);
	}
	const msStep = (Date.now() - t0) / measure;
	const st = sim.stats();
	const w = gpuWork(cfg);
	const line = [
		"  " + item.name.padEnd(24),
		"relRes med=" + median(res).toExponential(2),
		"div med=" + median(div).toExponential(2),
		"div p95=" + p95(div).toExponential(2),
		"cpu " + msStep.toFixed(1) + " ms/step",
		"work=" + w.feq.toFixed(1) + " feq",
		"disp=" + w.disp,
		"maxM=" + st.tot.maxM.toFixed(4),
		"neg=" + st.tot.cntNeg
	].join(" | ");
	process.stdout.write(line + "\n");
	return { name: item.name, relRes: median(res), divMed: median(div), msStep, feq: w.feq, disp: w.disp, neg: st.tot.cntNeg };
}

console.log("mg-work study · dt=" + DT + " · warm-started pressure, S=4 coarse level");
console.log("feq = fine sweep equivalents per projection (GPU encodeStep shape), disp = dispatches per projection");

for (const scenario of ["jet", "cycle", "dam"]) {
	console.log("== " + scenario + " N=128 (180 warmup + 90 measured steps) ==");
	for (const item of CONFIGS) runRow(scenario, 128, 180, 90, item);
}
console.log("== jet N=256 scaling check (120 warmup + 60 measured steps) ==");
for (const item of [CONFIGS[0], CONFIGS[4], CONFIGS[5]]) runRow("jet", 256, 120, 60, item);
console.log("done");
