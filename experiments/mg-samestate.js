"use strict";
// 0.6.0 t11: controlled projection comparison. Every candidate runs one step
// from the *same* warm state (warmed at the pre-t11 reference: 2 cycles,
// pre4/post8, coarse K=60), so the numbers are solver quality, not where the
// chaotic dam splash happened to be after N steps.
// run: node experiments/mg-samestate.js | tee experiments/logs/<date>-mg-samestate.log
//      node experiments/mg-samestate.js --n512
//
// rel = divAfter/divBefore (the a2 ratio), true = mg.residualNorm()/divBefore
// (the 0.6.0 4.1 judge, operator units). rel is roughly dt*true, but the two
// maxima need not land on the same cell, so do not convert one into the other.
// The multi-step counterpart, where each config warms up on its own and the
// peak metric follows the trajectory, is experiments/mg-defaults.js.

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/multigrid.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const cfg0 = Fluid.core.config.defaults;
const DT = 1 / 240;
const REFERENCE = { mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 60 };

function make(scenario, N) {
	const cfg = Object.assign({}, cfg0, { N, scenario });
	cfg.texW = N + 2; cfg.texH = N + 2;
	Fluid.core.config.prepareScenario(cfg);
	return { cfg, sim: solverMod.create(cfg) };
}
function sameState(scenario, N, warmup, item) {
	// every candidate reaches the identical warm state at the reference setting;
	// only the measured step runs with the candidate's overrides.
	const { cfg, sim } = make(scenario, N);
	Object.assign(cfg, REFERENCE);
	for (let s = 0; s < warmup; s++) sim.step(DT);
	Object.assign(cfg, item.over);
	sim.step(DT);
	const b = Math.max(sim.diag.divBefore, 1e-9);
	const residual = sim.mg ? sim.mg.residualNorm() : sim.diag.divAfter / DT;
	const st = sim.stats();
	return {
		divB: b,
		divA: sim.diag.divAfter,
		rel: sim.diag.pressureResidual,
		true: residual / b,
		maxM: st.tot.maxM,
		over: st.tot.cntOver
	};
}
const CANDIDATES = [
	{ name: "ref", over: {} },
	{ name: "c2k40", over: { mgCycles: 2, mgCoarseIters: 40 } },
	{ name: "c2k20", over: { mgCycles: 2, mgCoarseIters: 20 } },
	{ name: "c1k20", over: { mgCycles: 1, mgCoarseIters: 20 } }
];
function line(name, r) {
	console.log("  " + name.padEnd(6) +
		" divB=" + r.divB.toExponential(3) +
		" divA=" + r.divA.toExponential(3) +
		" rel=" + r.rel.toExponential(3) +
		" true=" + r.true.toExponential(3) +
		" maxM=" + r.maxM.toFixed(4) +
		" over=" + r.over);
}

const n512 = process.argv.includes("--n512");
const jobs = n512
	? [["dam", 512, 90]]
	: [["dam", 128, 180], ["blobs", 128, 180], ["jet", 128, 180], ["cycle", 128, 180]];
console.log("mg-samestate · pinned dt=" + DT + " · one step from a shared warm state" +
	" (reference: 2 cycles pre4/post8 coarse K=60)" + (n512 ? " · N=512 check" : ""));
for (const job of jobs) {
	console.log("== " + job[0] + " N=" + job[1] + " (" + job[2] + " warmup, then one step) ==");
	for (const item of CANDIDATES) line(item.name, sameState(job[0], job[1], job[2], item));
}
console.log("done");
