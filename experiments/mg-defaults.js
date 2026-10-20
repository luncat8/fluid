"use strict";
// 0.6.0 t11: multi-step rows for the projection defaults.
// run: node experiments/mg-defaults.js | tee experiments/logs/<date>-mg-defaults.log
//
// Each row warms up *on its own setting* and then runs the measured window, so
// these are trajectories: the peak divergence follows where the splash happens
// to be and must not be read as a controlled quality comparison. Use
// experiments/mg-samestate.js for that (one step from a shared warm state).
// The pinned dt still keeps the substep auto-policy out of the comparison.
// The judge is the true Poisson residual from 0.6.0 section 4.1:
//   true = mg.residualNorm()/divBefore   (operator units, the 4.1 judge)
//   rel  = divAfter/divBefore            (the a2 divergence ratio); roughly
//          dt*true, but the two maxima need not land on the same cell
// Rows:
//   rb60   mg=0, 60 red-black sweeps          recorded 0.5.0 fallback baseline
//   c2k60  2 cycles pre4/post8, coarse K=60    the pre-t11 shipping default
//   c2k20  2 cycles pre4/post8, coarse K=20    adopted default (t11)
//   c1k20  1 cycle  pre4/post8, coarse K=20    measured, not adopted
// The hardware counterpart is experiments/browser-tests.html with the same
// settings; there the report only has maxDiv (a peak, not the residual).
// --n512 runs the dam row at the target resolution instead (90 + 30 steps, the
// whole time is in the solve, so the warmup is shortened; the policy stays pinned).

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/multigrid.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const cfg0 = Fluid.core.config.defaults;
const DT = 1 / 240;

const CONFIGS = [
	{ name: "rb60", over: { useMultigrid: false, pressureIters: 60 } },
	{ name: "c2k60", over: { useMultigrid: true, mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 60 } },
	{ name: "c2k40", over: { useMultigrid: true, mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 40 } },
	{ name: "c2k20", over: { useMultigrid: true, mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 20 } },
	{ name: "c1k20", over: { useMultigrid: true, mgCycles: 1, mgPre: 4, mgPost: 8, mgCoarseIters: 20 } }
];
function median(a) {
	const s = a.slice().sort((x, y) => x - y);
	return s[Math.floor(s.length / 2)];
}
function row(scenario, N, warmup, measure, item) {
	const cfg = Object.assign({}, cfg0, item.over, { N, scenario });
	cfg.texW = N + 2; cfg.texH = N + 2;
	Fluid.core.config.prepareScenario(cfg);
	const sim = solverMod.create(cfg);
	for (let s = 0; s < warmup; s++) sim.step(DT);
	const rel = [], tru = [], divB = [], divA = [], over = [];
	const t0 = Date.now();
	for (let s = 0; s < measure; s++) {
		sim.step(DT);
		const b = Math.max(sim.diag.divBefore, 1e-9);
		tru.push(sim.mg ? sim.mg.residualNorm() / b : sim.diag.pressureResidual);
		rel.push(sim.diag.pressureResidual);
		divB.push(sim.diag.divBefore);
		divA.push(sim.diag.divAfter);
	}
	const st = sim.stats();
	process.stdout.write("  " + item.name.padEnd(6) +
		" rel=" + median(rel).toExponential(2) +
		" true=" + median(tru).toExponential(2) +
		" divB=" + median(divB).toExponential(2) +
		" divA=" + median(divA).toExponential(2) +
		" maxM=" + st.tot.maxM.toFixed(4) +
		" neg=" + st.tot.cntNeg +
		" over=" + st.tot.cntOver +
		" cpu=" + ((Date.now() - t0) / measure).toFixed(1) + "ms/step\n");
}
const n512 = process.argv.includes("--n512");
const jobs = n512
	? [["dam", 512, 90, 30]]
	: [["dam", 64, 80, 30], ["dam", 128, 180, 90], ["dam", 256, 120, 60],
		["blobs", 128, 180, 90], ["jet", 128, 180, 90], ["cycle", 128, 180, 90]];
const configs = n512 ? CONFIGS.filter(c => c.name !== "rb60") : CONFIGS;
console.log("mg-defaults · pinned dt=" + DT + " · warm-started pressure · medians over the measured steps" +
	(n512 ? " · N=512 check (shorter warmup)" : ""));
for (const job of jobs) {
	console.log("== " + job[0] + " N=" + job[1] + " (" + job[2] + " warmup + " + job[3] + " measured) ==");
	for (const item of configs) row(job[0], job[1], job[2], job[3], item);
}
console.log("done");
