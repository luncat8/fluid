"use strict";
// 0.5.0 t13 / a7 / a11: Stefan 1D ice-front sqrt(t) scaling, 2D cooling plume
// with solid crust formation, and multigrid stability inside enclosed solid
// cavities and self-supporting solid arches.

const configMod = require("../src/core/config.js");
require("../src/core/grid.js");
const statsMod = require("../src/core/stats.js");
require("../src/core/multigrid.js");
const solverMod = require("../src/core/solver.js");

let failures = 0;
function check(name, cond, detail) {
	console.log((cond ? "PASS" : "FAIL") + " " + name + (detail ? "  " + detail : ""));
	if (!cond) failures++;
}
function cur(pp) { return pp.cur === 0 ? pp.a : pp.b; }

// 1. Classical Neumann/Stefan 1D solidification: semi-infinite cold solid slab
// (y > y0 at Ts < TMelt) meeting liquid just above TMelt (y <= y0) in an
// insulated zero-gravity domain. Exact similarity front position s(t) ~ t^0.5.
function testStefanSqrtT() {
	const N = 64;
	const y0 = 40;
	const cfg = Object.assign({}, configMod.defaults, {
		N,
		texW: N + 2,
		texH: N + 2,
		scenario: "still",
		gravityY: 0,
		thermalSources: 0,
		wallCool: 0,
		kSol: 60,
		kLiq: 10,
		pressureIters: 4,
		useMultigrid: false
	});
	const sim = solverMod.create(cfg);
	const mA = cur(sim.mass), mB = sim.mass.b;
	const tA = cur(sim.therm), tB = sim.therm.b;
	for (let y = 1; y <= N; y++) {
		const T = y > y0 ? 150 : cfg.TMelt + 0.5;
		const E = solverMod.energyFromT(1, T, cfg);
		for (let x = 1; x <= N; x++) {
			const c = x + y * cfg.texW;
			const i4 = c * 4;
			mA[i4] = 0; mA[i4 + 1] = 0; mA[i4 + 2] = 1; mA[i4 + 3] = 1;
			tA[i4] = E; tA[i4 + 1] = T; tA[i4 + 2] = 0; tA[i4 + 3] = 0;
		}
	}
	mB.set(mA);
	tB.set(tA);
	sim.phaseEos(false);
	const init = statsMod.totalsFromFields(cur(sim.mass), cur(sim.therm), cfg.texW, cfg.texH);
	const dt = 1 / 120;
	const samples = [];
	let prevFront = 0;
	let monotone = true;
	const xProbe = 32;
	function frontDepth() {
		// continuous enthalpy-interpolated solid fraction depth into y <= y0:
		// s(t) = sum_{y <= y0} (1 - phi_l(xProbe, y))
		let depth = 0;
		for (let y = 1; y <= y0; y++) {
			const c = xProbe + y * cfg.texW;
			const st = solverMod.phaseOf(cur(sim.mass)[c * 4 + 3], cur(sim.therm)[c * 4], cfg);
			depth += 1 - st.phiL;
		}
		return depth;
	}
	for (let step = 1; step <= 900; step++) {
		sim.step(dt);
		if (step >= 150 && step % 150 === 0) {
			const s = frontDepth();
			if (s < prevFront - 1e-6) monotone = false;
			prevFront = s;
			samples.push({ t: step * dt, s });
		}
	}
	// Ordinary least squares slope of log(s) vs log(t)
	let sx = 0, sy = 0, sxx = 0, sxy = 0;
	for (const p of samples) {
		const lx = Math.log(p.t), ly = Math.log(p.s);
		sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly;
	}
	const nPts = samples.length;
	const slope = (nPts * sxy - sx * sy) / (nPts * sxx - sx * sx);
	const fin = statsMod.totalsFromFields(cur(sim.mass), cur(sim.therm), cfg.texW, cfg.texH);
	const driftM = Math.abs(fin.sumM - init.sumM) / init.sumM;
	const driftE = Math.abs(fin.sumE - init.sumE) / init.sumE;
	check("stefan 1D: ice front grows monotonically into liquid",
		monotone && samples[nPts - 1].s > 4,
		"s(1.25s)=" + samples[0].s.toFixed(2) + " -> s(7.5s)=" + samples[nPts - 1].s.toFixed(2) + " cells");
	check("stefan 1D: front depth log-log slope matches t^0.5 within +-0.08",
		Math.abs(slope - 0.5) <= 0.08,
		"slope=" + slope.toFixed(4) + " target=0.5000");
	check("stefan 1D: mass and energy conserved through solidification (f32 machine precision)",
		driftM < 1e-6 && driftE < 1e-6,
		"drift_m=" + driftM.toExponential(2) + " drift_E=" + driftE.toExponential(2));
}

// 2. 2D cooling plume + top solid crust formation with gravity and Robin ceiling sink
function testCoolingPlumeAndCrust() {
	const N = 64;
	const cfg = Object.assign({}, configMod.defaults, {
		N,
		texW: N + 2,
		texH: N + 2,
		scenario: "still",
		gravityY: 300,
		thermalSources: 1,
		heatPower: 40,
		coolTop: 8.0,
		TCeil: 210,
		heatBand: 4,
		kSol: 8,
		kLiq: 2,
		mgCoarseIters: 8,
		useMultigrid: true
	});
	const sim = solverMod.create(cfg);
	// Fill entire box with warm liquid near TMelt + 8 K so top ceiling freezes a crust
	const mA = cur(sim.mass), mB = sim.mass.b;
	const tA = cur(sim.therm), tB = sim.therm.b;
	for (let y = 1; y <= N; y++) {
		for (let x = 1; x <= N; x++) {
			const c = x + y * cfg.texW;
			const i4 = c * 4;
			const T = cfg.TMelt + 6 + 0.5 * Math.sin(x * 0.3);
			const E = solverMod.energyFromT(1, T, cfg);
			mA[i4] = 0.2; mA[i4 + 1] = 0.3; mA[i4 + 2] = 0.8; mA[i4 + 3] = 1;
			tA[i4] = E; tA[i4 + 1] = T; tA[i4 + 2] = 0; tA[i4 + 3] = 0;
		}
	}
	mB.set(mA);
	tB.set(tA);
	const init = statsMod.totalsFromFields(mA, tA, cfg.texW, cfg.texH);
	sim.ledger.initialM = init.sumM;
	sim.ledger.initialDr = init.sumDr;
	sim.ledger.initialDg = init.sumDg;
	sim.ledger.initialDb = init.sumDb;
	sim.ledger.initialE = init.sumE;
	sim.phaseEos(false);
	for (let step = 0; step < 360; step++) sim.step(1 / 120);
	const st = sim.stats();
	let frozenTop = 0;
	for (let y = N - 5; y <= N; y++) {
		for (let x = 1; x <= N; x++) {
			if (sim.isFrozen(x, y)) frozenTop++;
		}
	}
	const exp = statsMod.writeExpectedTotals({ m: 0, dr: 0, dg: 0, db: 0, E: 0 }, st.ledger);
	const driftM = statsMod.driftRel(st.tot.sumM, exp.m);
	const driftE = statsMod.driftRel(st.tot.sumE, exp.E);
	check("plume + crust: cold ceiling grows a frozen solid crust while convective flow runs underneath",
		frozenTop >= N && st.mot.maxVel > 0.05,
		"frozenTop=" + frozenTop + " maxVel=" + st.mot.maxVel.toFixed(3));
	check("plume + crust: mass and energy ledger conserved with open heat sink/source",
		Math.abs(driftM) < 1e-6 && Math.abs(driftE) < 1e-6 && st.tot.cntNeg === 0,
		"drift_m=" + driftM.toExponential(2) + " drift_E=" + driftE.toExponential(2));
}

// 3. Multigrid inside enclosed frozen cavities + self-supporting solid arch over vacuum
function testSolidArchAndCavityMg() {
	const N = 64;
	const archCfg = Object.assign({}, configMod.defaults, {
		N,
		texW: N + 2,
		texH: N + 2,
		scenario: "arch",
		mgCoarseIters: 8,
		useMultigrid: true
	});
	configMod.prepareScenario(archCfg);
	const arch = solverMod.create(archCfg);
	const initFrozen = arch.stats().phaseCounts[1];
	for (let step = 0; step < 180; step++) arch.step(1 / 120);
	const st = arch.stats();
	// Check frozen arch crown cells overhanging the open chamber below keep u=v=0
	const midX = Math.floor(N * 0.5);
	const midY = 30;
	const vA = cur(arch.vel);
	const vi = (midX + midY * archCfg.texW) * 2;
	check("solid arch: overhangs open chamber under gravity without sagging (u=v=0 inside frozen crown)",
		arch.isFrozen(midX, midY) && vA[vi] === 0 && vA[vi + 1] === 0 && st.phaseCounts[1] > 0.7 * initFrozen,
		"frozen=" + st.phaseCounts[1] + "/" + initFrozen);
	const exp = statsMod.writeExpectedTotals({ m: 0, dr: 0, dg: 0, db: 0, E: 0 }, st.ledger);
	const driftM = statsMod.driftRel(st.tot.sumM, exp.m);
	check("solid arch + MG: finite pressure and conserved mass with non-aligned frozen geometry",
		Number.isFinite(st.mot.pMin) && Number.isFinite(st.mot.pMax) && Math.abs(driftM) < 1e-6,
		"p=[" + st.mot.pMin.toFixed(2) + ".." + st.mot.pMax.toFixed(2) + "] drift_m=" + driftM.toExponential(2));
}

testStefanSqrtT();
testCoolingPlumeAndCrust();
testSolidArchAndCavityMg();

if (failures) {
	console.log("FAILED " + failures + " solid-growth checks");
	process.exitCode = 1;
} else {
	console.log("ALL SOLID-GROWTH TESTS PASSED");
}
