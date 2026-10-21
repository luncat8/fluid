"use strict";
// 0.6.0 re-trial of 1 vs 2 V-cycles on the residual that counts.
// solve = diag.poissonRes/divBefore, captured in project() before transport.
// drift = residualNorm()/divBefore after step(), the old post-step column.
// rel   = divAfter/divBefore.
// Hardware already rejects 1 cycle as a default: dam ends overpacked, cycle
// maxDiv goes 30 -> 122. This asks whether jet's solve is actually neutral,
// which that old column could not say. It does not adopt anything.
// run: node experiments/mg-cycle-gate.js | tee experiments/logs/<date>-mg-cycle-gate.log

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
const C2 = { mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 20 };
const C1 = { mgCycles: 1, mgPre: 4, mgPost: 8, mgCoarseIters: 20 };

function make(scenario, N, over) {
	const cfg = Object.assign({}, cfg0, over, { N, scenario });
	cfg.texW = N + 2;
	cfg.texH = N + 2;
	Fluid.core.config.prepareScenario(cfg);
	return { cfg, sim: solverMod.create(cfg) };
}
function read(sim) {
	const b = Math.max(sim.diag.divBefore, 1e-30);
	const driftRes = sim.mg ? sim.mg.residualNorm() : 0;
	return {
		divB: b,
		divA: sim.diag.divAfter,
		rel: sim.diag.pressureResidual,
		solve: (sim.diag.poissonRes || 0) / b,
		drift: driftRes / b
	};
}
function sameState(scenario, N, warmup) {
	const out = {};
	for (const item of [{ name: "c2k20", over: C2 }, { name: "c1k20", over: C1 }]) {
		const { cfg, sim } = make(scenario, N, {});
		Object.assign(cfg, REFERENCE);
		for (let s = 0; s < warmup; s++) sim.step(DT);
		Object.assign(cfg, item.over);
		sim.step(DT);
		out[item.name] = read(sim);
	}
	return out;
}
function trajectory(scenario, N, warmup, measure, over) {
	const { sim } = make(scenario, N, over);
	for (let s = 0; s < warmup; s++) sim.step(DT);
	let peakSolve = 0, peakRel = 0, peakDivA = 0, peakDrift = 0;
	const t0 = Date.now();
	for (let s = 0; s < measure; s++) {
		sim.step(DT);
		const r = read(sim);
		if (r.solve > peakSolve) peakSolve = r.solve;
		if (r.rel > peakRel) peakRel = r.rel;
		if (r.divA > peakDivA) peakDivA = r.divA;
		if (r.drift > peakDrift) peakDrift = r.drift;
	}
	const st = sim.stats();
	return {
		peakSolve, peakRel, peakDivA, peakDrift,
		maxM: st.tot.maxM, over: st.tot.cntOver,
		ms: (Date.now() - t0) / measure
	};
}
function exp(v) {
	return v.toExponential(3);
}
function ratio(a, b) {
	return b > 0 ? (a / b).toFixed(2) + "x" : "n/a";
}
function printSame(scenario, N, warmup, rows) {
	console.log("== same-state " + scenario + " N=" + N + " (" + warmup + " warmup at K60, then one step) ==");
	for (const name of ["c2k20", "c1k20"]) {
		const r = rows[name];
		console.log("  " + name +
			" solve=" + exp(r.solve) +
			" drift=" + exp(r.drift) +
			" rel=" + exp(r.rel) +
			" divA=" + exp(r.divA));
	}
	console.log("  c1/c2  solve=" + ratio(rows.c1k20.solve, rows.c2k20.solve) +
		" drift=" + ratio(rows.c1k20.drift, rows.c2k20.drift) +
		" rel=" + ratio(rows.c1k20.rel, rows.c2k20.rel) +
		" divA=" + ratio(rows.c1k20.divA, rows.c2k20.divA));
}
function printTraj(scenario, N, warmup, measure, name, r) {
	console.log("  " + name.padEnd(6) +
		" peakSolve=" + exp(r.peakSolve) +
		" peakDrift=" + exp(r.peakDrift) +
		" peakRel=" + exp(r.peakRel) +
		" peakDivA=" + exp(r.peakDivA) +
		" maxM=" + r.maxM.toFixed(4) +
		" over=" + r.over +
		" cpu=" + r.ms.toFixed(1) + "ms");
}

console.log("mg-cycle-gate · dt=" + DT + " · solve is pre-transport, drift is residualNorm after step");
const sameJobs = [["dam", 128, 180], ["blobs", 128, 180], ["jet", 128, 180], ["cycle", 128, 180]];
for (const job of sameJobs) printSame(job[0], job[1], job[2], sameState(job[0], job[1], job[2]));
const trajJobs = [["jet", 128, 90, 90], ["cycle", 128, 90, 90]];
for (const job of trajJobs) {
	console.log("== trajectory " + job[0] + " N=" + job[1] + " (" + job[2] + " own warmup + " + job[3] + " measured) ==");
	const a = trajectory(job[0], job[1], job[2], job[3], C2);
	const b = trajectory(job[0], job[1], job[2], job[3], C1);
	printTraj(job[0], job[1], job[2], job[3], "c2k20", a);
	printTraj(job[0], job[1], job[2], job[3], "c1k20", b);
	console.log("  c1/c2  peakSolve=" + ratio(b.peakSolve, a.peakSolve) +
		" peakDrift=" + ratio(b.peakDrift, a.peakDrift) +
		" peakRel=" + ratio(b.peakRel, a.peakRel) +
		" peakDivA=" + ratio(b.peakDivA, a.peakDivA));
}
console.log("done");
