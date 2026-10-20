"use strict";
// 0.6.0 t3/t11 measurement, round 2 (follows mg-work.js).
// run: node experiments/mg-work2.js | tee experiments/logs/<date>-mg-work2.log
//
// Round 1 used divAfter, which in boiling scenarios floors at max|S| and hides
// solver quality. This round reports the true Poisson residual
// mg.residualNorm()/divBefore after each projection, sweeps the coarse
// iteration count K at the current 2-cycle setting on dam (S = 0, the
// discriminating case), and bisects the cycle blowup round 1 found at
// pre2/post4 (fields went NaN: maxM read 0 because NaN loses every max).

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/multigrid.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const cfg0 = Fluid.core.config.defaults;

const DT = 1 / 240;
function makeSim(scenario, N, over) {
	const cfg = Object.assign({}, cfg0, over, { N, scenario });
	cfg.texW = N + 2; cfg.texH = N + 2;
	Fluid.core.config.prepareScenario(cfg);
	return { cfg, sim: solverMod.create(cfg) };
}
function median(a) {
	const s = a.slice().sort((x, y) => x - y);
	return s[Math.floor(s.length / 2)];
}
function gpuWork(c) {
	if (!c.useMultigrid) return { feq: c.pressureIters, disp: 2 * c.pressureIters };
	const S2 = (c.mgCoarsen || 4) * (c.mgCoarsen || 4);
	const cyc = c.mgCycles, K = c.mgCoarseIters;
	return {
		feq: cyc * (c.mgPre + c.mgPost) + cyc * K / S2 + cyc * 2 + 1,
		disp: 1 + cyc * (2 * (c.mgPre + c.mgPost) + 2 + 2 * K + 1)
	};
}
function residualRow(scenario, item) {
	const { cfg, sim } = makeSim(scenario, 128, item.over);
	for (let s = 0; s < 180; s++) sim.step(DT);
	const rel = [], div = [];
	const t0 = Date.now();
	for (let s = 0; s < 90; s++) {
		sim.step(DT);
		const b = Math.max(sim.diag.divBefore, 1e-9);
		rel.push(sim.mg.residualNorm() / b);
		div.push(sim.diag.divAfter);
	}
	const w = gpuWork(cfg);
	process.stdout.write([
		"  " + item.name.padEnd(24),
		"trueRes med=" + median(rel).toExponential(2),
		"div med=" + median(div).toExponential(2),
		"cpu " + ((Date.now() - t0) / 90).toFixed(1) + " ms/step",
		"work=" + w.feq.toFixed(1) + " feq",
		"disp=" + w.disp
	].join(" | ") + "\n");
}
function stabilityRow(item, steps) {
	const { sim } = makeSim("cycle", 128, item.over);
	let blowStep = -1;
	for (let s = 0; s < steps; s++) {
		sim.step(DT);
		if (s % 10 === 9) {
			const st = sim.stats();
			if (!Number.isFinite(st.tot.sumM) || !Number.isFinite(st.tot.sumE) || !(st.mot.maxVel < 1e5)) { blowStep = s + 1; break; }
		}
	}
	process.stdout.write("  " + item.name.padEnd(24) +
		(blowStep > 0 ? "BLOWUP at step " + blowStep : "stable over " + steps + " steps") + "\n");
}

const CUR = { name: "cur c2 pre4 post8 K60", over: { useMultigrid: true, mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 60 } };
const C2K20 = { name: "c2 pre4 post8 K20", over: { useMultigrid: true, mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 20 } };
const C2K12 = { name: "c2 pre4 post8 K12", over: { useMultigrid: true, mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 12 } };
const C2K8 = { name: "c2 pre4 post8 K8", over: { useMultigrid: true, mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 8 } };
const C1K20 = { name: "c1 pre4 post8 K20", over: { useMultigrid: true, mgCycles: 1, mgPre: 4, mgPost: 8, mgCoarseIters: 20 } };
const PRE3 = { name: "c1 pre3 post6 K20", over: { useMultigrid: true, mgCycles: 1, mgPre: 3, mgPost: 6, mgCoarseIters: 20 } };

console.log("mg-work2 · true Poisson residual (residualNorm/divBefore) + K sweep + stability bisect · dt=" + DT);
console.log("== K sweep at c2 pre4 post8, dam N=128 ==");
for (const item of [CUR, C2K20, C2K12, C2K8]) residualRow("dam", item);
console.log("== true residual on boiling scenarios, N=128 ==");
for (const scenario of ["jet", "cycle"]) {
	console.log("-- " + scenario + " --");
	for (const item of [CUR, C2K20, C1K20]) residualRow(scenario, item);
}
console.log("== cycle stability bisect, 480 steps (2 s), N=128 ==");
stabilityRow({ name: "c1 pre2 post4 K20", over: { useMultigrid: true, mgCycles: 1, mgPre: 2, mgPost: 4, mgCoarseIters: 20 } }, 480);
stabilityRow({ name: "c1 pre2 post8 K20", over: { useMultigrid: true, mgCycles: 1, mgPre: 2, mgPost: 8, mgCoarseIters: 20 } }, 480);
stabilityRow(PRE3, 480);
stabilityRow({ name: "c2 pre2 post4 K20", over: { useMultigrid: true, mgCycles: 2, mgPre: 2, mgPost: 4, mgCoarseIters: 20 } }, 480);
stabilityRow(C1K20, 480);
console.log("done");
