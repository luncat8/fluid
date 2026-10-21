"use strict";
// The fixed-dt cycle capture (2026-10-21-05_47) reported poissonTrue=26 at frame
// 301. That number was the post-step operator (mass and flags after transport
// and phase) applied to the pressure solved before transport. A face weight that
// moves by O(1) next to a hydrostatic jump reads as a huge residual even when
// the solve itself was tight. diag.poissonRes is the solve; the recompute after
// step() is the drift.
//
// run: node experiments/residual-spike.js
//      node experiments/residual-spike.js --traj 128 400

const assert = require("node:assert/strict");
const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/multigrid.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const mg = Fluid.core.multigrid;
const cfg0 = Fluid.core.config.defaults;
const DT = 1 / 240;

function make(scenario, N) {
	const cfg = Object.assign({}, cfg0, {
		N, scenario, substepsAuto: false, substepsFixed: 4, useMultigrid: true,
		mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 20
	});
	cfg.texW = N + 2;
	cfg.texH = N + 2;
	Fluid.core.config.prepareScenario(cfg);
	return { cfg, sim: solverMod.create(cfg) };
}
function residual(sim, cfg) {
	return mg.computeResidual(
		Fluid.core.grid.curBuf(sim.press), sim.divsrc, Fluid.core.grid.curBuf(sim.mass),
		sim.flags, sim.texW, sim.texH, cfg.rhoVac, null);
}
function fillOpen(mass, texW, texH, rho) {
	for (let y = 1; y < texH - 1; y++) {
		for (let x = 1; x < texW - 1; x++) mass[(x + y * texW) * 4 + 3] = rho;
	}
}
function weightJump() {
	const texW = 10, texH = 10;
	const press = new Float32Array(texW * texH);
	const div = new Float32Array(texW * texH);
	const mass = new Float32Array(texW * texH * 4);
	const flags = new Uint8Array(texW * texH * 4);
	const rhoVac = cfg0.rhoVac;
	fillOpen(mass, texW, texH, 1);
	// a linear field is not a solution at the wall, so compare the same field
	// before and after one interior cell empties. the jump is the weight change.
	for (let y = 1; y < texH - 1; y++) {
		for (let x = 1; x < texW - 1; x++) press[x + y * texW] = 1000 * x;
	}
	const before = mg.computeResidual(press, div, mass, flags, texW, texH, rhoVac, null);
	mass[(4 + 4 * texW) * 4 + 3] = rhoVac;
	const after = mg.computeResidual(press, div, mass, flags, texW, texH, rhoVac, null);
	console.log("  weight jump  before=" + before.toExponential(3) +
		"  after rhoVac neighbor=" + after.toExponential(3) +
		"  ratio=" + (after / Math.max(before, 1e-12)).toExponential(2));
	assert.ok(after > before * 100, "emptying one neighbor must inflate the residual, " + after + " vs " + before);
}
function meltFlip() {
	const texW = 10, texH = 10;
	const press = new Float32Array(texW * texH);
	const div = new Float32Array(texW * texH);
	const mass = new Float32Array(texW * texH * 4);
	const flags = new Uint8Array(texW * texH * 4);
	const rhoVac = cfg0.rhoVac;
	fillOpen(mass, texW, texH, 1);
	const i = 4 + 4 * texW;
	// blocked cells are forced to p=0 and dropped from every neighbour stencil,
	// so a uniform pressure on the open cells is an exact solution until the
	// cell melts and that stale zero couples back in.
	flags[i * 4 + 2] = 1;
	for (let y = 1; y < texH - 1; y++) {
		for (let x = 1; x < texW - 1; x++) {
			if ((x + y * texW) !== i) press[x + y * texW] = 50000;
		}
	}
	const before = mg.computeResidual(press, div, mass, flags, texW, texH, rhoVac, null);
	flags[i * 4 + 2] = 0;
	const after = mg.computeResidual(press, div, mass, flags, texW, texH, rhoVac, null);
	console.log("  melt flip    frozen residual=" + before.toExponential(3) +
		"  after melt=" + after.toExponential(3));
	assert.ok(before < 1, "uniform pressure around a frozen cell is a solution, residual " + before);
	assert.ok(after > 1e4, "melting next to a large pressure must spike, got " + after);
}
function stepSplit(sim, cfg) {
	sim.forces(DT);
	sim.advectVelocity(DT);
	sim.poissonRhs(DT);
	sim.project(Fluid.core.config.solverIters(cfg));
	const solve = residual(sim, cfg);
	const stored = sim.diag.poissonRes;
	const rhs = sim.diag.divBefore;
	sim.applyGrad(DT);
	const mot = Fluid.core.stats.motionTotals(
		Fluid.core.grid.curBuf(sim.vel), Fluid.core.grid.curBuf(sim.press),
		sim.texW, sim.texH, DT);
	sim.diag.divAfter = mot.maxDiv;
	sim.diag.pressureResidual = rhs > 0 ? mot.maxDiv / rhs : 0;
	sim.computeFlux(DT);
	sim.advectScalar();
	sim.phaseEos(true);
	sim.conduction(DT);
	sim.heatSources(DT);
	sim.phaseEos(true);
	cfg.frame++;
	cfg.time += DT;
	return { solve, stored, end: residual(sim, cfg), rhs };
}
function damSplit() {
	const { cfg, sim } = make("dam", 64);
	let worst = null;
	for (let s = 0; s < 40; s++) {
		const row = stepSplit(sim, cfg);
		assert.ok(Math.abs(row.solve - row.stored) <= Math.max(1e-5, 1e-4 * row.solve),
			"project() must store the pre-transport residual, step " + (s + 1));
		const ratio = row.solve > 0 ? row.end / row.solve : 0;
		if (!worst || ratio > worst.ratio) worst = { step: s + 1, ratio, solve: row.solve, end: row.end, rhs: row.rhs };
	}
	const solveRel = worst.rhs > 0 ? worst.solve / worst.rhs : 0;
	console.log("  dam N=64     worst end/solve=" + worst.ratio.toExponential(2) +
		" at step " + worst.step +
		" solve=" + worst.solve.toExponential(3) +
		" end=" + worst.end.toExponential(3) +
		" solve/rhs=" + solveRel.toExponential(3));
	assert.ok(worst.ratio > 8, "post-step residual must pull away from the solve, ratio " + worst.ratio);
	assert.ok(solveRel < 0.05, "the solve itself stays tight when the drift spikes, solve/rhs " + solveRel);
}
function traj(scenario, N, steps) {
	const { cfg, sim } = make(scenario, N);
	let maxEndRel = 0, maxSolveRel = 0, maxRatio = 0, at = 0;
	const t0 = Date.now();
	for (let s = 0; s < steps; s++) {
		const row = stepSplit(sim, cfg);
		const endRel = row.rhs > 1e-9 ? row.end / row.rhs : 0;
		const solveRel = row.rhs > 1e-9 ? row.solve / row.rhs : 0;
		const ratio = row.solve > 1e-9 ? row.end / row.solve : 0;
		if (endRel > maxEndRel) { maxEndRel = endRel; at = s + 1; }
		if (solveRel > maxSolveRel) maxSolveRel = solveRel;
		if (ratio > maxRatio) maxRatio = ratio;
	}
	console.log("  " + scenario.padEnd(6) + " N=" + String(N).padEnd(4) +
		" steps=" + steps +
		" cpu=" + ((Date.now() - t0) / steps).toFixed(1) + "ms" +
		" maxSolve/rhs=" + maxSolveRel.toExponential(2) +
		" maxDrift/rhs=" + maxEndRel.toExponential(2) +
		" at " + at +
		" maxDrift/solve=" + maxRatio.toExponential(2));
}

const args = process.argv.slice(2);
const trajMode = args.includes("--traj");
console.log("residual-spike · dt=" + DT + " · solve residual vs post-step drift");
weightJump();
meltFlip();
damSplit();
if (trajMode) {
	const n = Number(args[args.indexOf("--traj") + 1]) || 64;
	const steps = Number(args[args.indexOf("--traj") + 2]) || 240;
	const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : "";
	for (const scenario of ["cycle", "jet", "dam"]) {
		if (only && scenario !== only) continue;
		traj(scenario, n, steps);
	}
}
console.log("PASS residual spike is post-step coefficient drift, not the solve");
