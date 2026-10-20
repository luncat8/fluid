"use strict";
// 0.5.0 phase validation on the cpu oracle: melt a block, boil a hot spot,
// condense a vapor blob. Records phase-histogram / T-plateau / latent-budget
// time series and checks the conservation and freeze-mask contracts (0.5.0
// sections 3.2-3.8, acceptance a1-a3, a5-a6).
// run: node experiments/phase.js | tee experiments/logs/<date>-phase.log

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/multigrid.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const cfg0 = Fluid.core.config.defaults;

let failures = 0;
function check(name, ok, detail) {
	console.log((ok ? "PASS " : "FAIL ") + name + (detail ? "  " + detail : ""));
	if (!ok) failures++;
}
function makeCfg(over) {
	const c = Object.assign({}, cfg0, over);
	c.texW = c.N + 2; c.texH = c.N + 2;
	return c;
}
function cur(pp) { return pp.cur === 0 ? pp.a : pp.b; }
function idx(cfg, x, y) { return (x + y * cfg.texW) * 4; }
function boxScan(cfg, buf, x0, y0, w, h, fn) {
	for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) fn(idx(cfg, x, y));
}
function setBox(sim, cfg, x0, y0, w, h, m, T, EOverride) {
	const ms = cur(sim.mass), ts = cur(sim.therm);
	boxScan(cfg, ms, x0, y0, w, h, i => {
		ms[i] = 0; ms[i + 1] = 0; ms[i + 2] = 0; ms[i + 3] = m;
		ts[i] = EOverride !== undefined ? EOverride : solverMod.energyFromT(m, T, cfg);
		ts[i + 1] = T;
		ts[i + 2] = 0; ts[i + 3] = 0;
	});
}
function boxMean(sim, cfg, x0, y0, w, h, field) {
	const ms = cur(sim.mass), ts = cur(sim.therm), sf = sim.sfield;
	let sum = 0, n = 0;
	boxScan(cfg, ms, x0, y0, w, h, i => {
		sum += field === "m" ? ms[i + 3] : field === "T" ? ts[i + 1] : field === "phiG" ? sf[i + 2] : sf[i + 1];
		n++;
	});
	return sum / n;
}
function totals(sim, cfg) {
	return Fluid.core.stats.totalsFromFields(cur(sim.mass), cur(sim.therm), cfg.texW, cfg.texH);
}

// 1. direct plateau sweep: T flat while phi sweeps, latent ledger matches m*L
function testPlateauSweep() {
	const cfg = makeCfg({ N: 8, scenario: "still" });
	const sim = solverMod.create(cfg);
	const ms = cur(sim.mass), ts = cur(sim.therm);
	const i = idx(cfg, 4, 4);
	const m = 1;
	const E1 = m * cfg.cSol * cfg.TMelt, E2 = E1 + m * cfg.Lf;
	const E3 = E2 + m * cfg.cLiq * (cfg.TBoil - cfg.TMelt), E4 = E3 + m * cfg.Lv;
	ms[i + 3] = m;
	let okFusion = true, okBoil = true;
	ts[i] = E1 - 1; sim.phaseEos(false);
	for (let s = 0; s < 100; s++) {
		ts[i] = E1 - 1 + (E2 - E1 + 2) * s / 99;
		sim.phaseEos(true);
		const o = solverMod.phaseOf(m, ts[i], cfg);
		if (Math.abs(ts[i + 1] - cfg.TMelt) > 1e-3 && o.phase === solverMod.PHASE_MELTING) okFusion = false;
	}
	ts[i] = E3 - 1; sim.phaseEos(false);
	for (let s = 0; s < 100; s++) {
		ts[i] = E3 - 1 + (E4 - E3 + 2) * s / 99;
		sim.phaseEos(true);
		const o = solverMod.phaseOf(m, ts[i], cfg);
		if (Math.abs(ts[i + 1] - cfg.TBoil) > 1e-3 && o.phase === solverMod.PHASE_BOILING) okBoil = false;
	}
	const latent = sim.ledger.latentChargedE;
	check("plateaus: T stays at T_melt / T_boil while phi sweeps", okFusion && okBoil);
	check("plateaus: latent ledger = m*Lf + m*Lv traversed within 0.1%",
		Math.abs(latent - m * (cfg.Lf + cfg.Lv)) / (m * (cfg.Lf + cfg.Lv)) < 1e-3,
		"latent=" + latent.toFixed(2) + " target=" + (m * (cfg.Lf + cfg.Lv)).toFixed(2));
}

// 2. melt a frozen block in warm liquid: frozen at start, melts outside-in,
// T never leaves the plateau before phi_l = 1, latent ~ blockMass*Lf
function testMeltBlock() {
	const cfg = makeCfg({ N: 32, scenario: "still", gravityY: 0, kLiq: 2, kSol: 2 });
	const sim = solverMod.create(cfg);
	const x0 = 13, y0 = 13, w = 6, h = 6;
	setBox(sim, cfg, x0, y0, w, h, 1, 250);
	sim.phaseEos(false);
	const blockMass = w * h;
	const t0 = totals(sim, cfg);
	check("melt: block starts frozen and its faces are blocked",
		sim.isFrozen(x0 + 2, y0 + 2) && sim.uBlocked(x0 + 2, y0 + 2) && sim.vBlocked(x0 + 2, y0 + 2));

	// advective flux must be zero on every face touching the block while a
	// uniform ambient drift keeps the open faces live
	const vs = cur(sim.vel);
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		const i2 = (x + y * cfg.texW) * 2;
		vs[i2] = 3; vs[i2 + 1] = 0;
	}
	sim.computeFlux(1 / 60);
	let fluxBlocked = true, fluxOpen = false;
	for (let y = y0; y < y0 + h; y++) {
		for (let x = x0; x <= x0 + w; x++) {
			if (sim.flux[(x + y * cfg.texW) * 2] !== 0) fluxBlocked = false;
		}
	}
	for (let x = x0; x < x0 + w; x++) {
		for (let y = y0; y <= y0 + h; y++) {
			if (sim.flux[(x + y * cfg.texW) * 2 + 1] !== 0) fluxBlocked = false;
		}
	}
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		if (sim.flux[(x + y * cfg.texW) * 2] !== 0) fluxOpen = true;
	}
	check("melt: zero advective flux on frozen faces while open faces carry drift", fluxBlocked && fluxOpen);
	vs.fill(0);

	const dt = 1 / 60;
	let plateauOK = true, melted = -1, midEdge = -1, midCore = -1;
	const series = [];
	for (let s = 0; s <= 8000; s++) {
		if (s > 0) { sim.conduction(dt); sim.phaseEos(true); }
		const ts = cur(sim.therm);
		// plateau invariant: phi_l < 1 -> T <= T_melt; phi_l > 0 -> T >= T_melt
		for (let y = y0; y < y0 + h; y++) {
			for (let x = x0; x < x0 + w; x++) {
				const i = idx(cfg, x, y);
				const phi = sim.sfield[i + 1];
				if (phi < 1 - 1e-6 && ts[i + 1] > cfg.TMelt + 1e-3) plateauOK = false;
				if (phi > 1e-6 && ts[i + 1] < cfg.TMelt - 1e-3) plateauOK = false;
			}
		}
		const meanPhi = boxMean(sim, cfg, x0, y0, w, h, "phiL");
		if (melted < 0 && meanPhi >= 1) melted = s;
		if (midEdge < 0 && meanPhi >= 0.5) {
			midEdge = sim.sfield[idx(cfg, x0, y0) + 1];
			midCore = sim.sfield[idx(cfg, x0 + 3, y0 + 3) + 1];
		}
		if (s % 1000 === 0) series.push([s, boxMean(sim, cfg, x0, y0, w, h, "T"), meanPhi, sim.ledger.latentChargedE]);
	}
	const t1 = totals(sim, cfg);
	const latent = sim.ledger.latentChargedE;
	check("melt: T stays on the T_melt plateau until each cell is fully melted", plateauOK);
	check("melt: block fully melts (phase histogram loses its solid)", melted >= 0,
		"melted at step " + melted + " of 8000");
	check("melt: melts outside-in (edge phi_l ahead of core at half-melt)", midEdge > midCore,
		"edge=" + midEdge.toFixed(3) + " core=" + midCore.toFixed(3));
	check("melt: latent ledger = block mass * Lf within 1%",
		Math.abs(latent - blockMass * cfg.Lf) / (blockMass * cfg.Lf) < 0.01,
		"latent=" + latent.toFixed(1) + " target=" + (blockMass * cfg.Lf).toFixed(1));
	const driftM = Math.abs(t1.sumM - t0.sumM) / t0.sumM;
	const driftE = Math.abs(t1.sumE - t0.sumE) / t0.sumE;
	check("melt: mass and energy conserved through the transition", driftM < 1e-6 && driftE < 1e-6,
		"drift_m=" + driftM.toExponential(2) + " drift_E=" + driftE.toExponential(2));
	check("melt: block unfrozen and free to move once melted",
		!sim.isFrozen(x0 + 2, y0 + 2) && !sim.uBlocked(x0 + 2, y0 + 2));
	console.log("INFO melt series (step, block mean T, mean phi_l, latent E)");
	for (const row of series) {
		console.log("  " + row[0] + "  T=" + row[1].toFixed(2) + "  phi_l=" + row[2].toFixed(3) + "  latent=" + row[3].toFixed(0));
	}
}

// 3. boil a hot spot: gas branch expands through S, pushes neighbors, conserves
function testBoilExpansion() {
	const cfg = makeCfg({ N: 32, scenario: "still", gravityY: 0, pressureIters: 10 });
	const sim = solverMod.create(cfg);
	const x0 = 13, y0 = 12, w = 6, h = 6;
	setBox(sim, cfg, x0, y0, w, h, 1, 450);
	sim.phaseEos(false);
	const t0 = totals(sim, cfg);
	const m0 = boxMean(sim, cfg, x0, y0, w, h, "m");
	const S0 = sim.sfield[idx(cfg, x0 + 2, y0 + 2)];
	const rhoStar = solverMod.phaseOf(1, cur(sim.therm)[idx(cfg, x0 + 2, y0 + 2)], cfg).rhoStar;
	check("boil: gas-branch spot has S > 0 (expansion source live)", S0 > 0,
		"S=" + S0.toFixed(2) + " rho*=" + rhoStar.toFixed(4));
	let maxVel = 0;
	for (let s = 0; s < 600; s++) {
		sim.step(1 / 60);
		const v = sim.stats().mot.maxVel;
		if (v > maxVel) maxVel = v;
	}
	const t1 = totals(sim, cfg);
	const m1 = boxMean(sim, cfg, x0, y0, w, h, "m");
	const driftM = Math.abs(t1.sumM - t0.sumM) / t0.sumM;
	check("boil: expansion pushes neighbors (maxVel rises)", maxVel > 0.5,
		"maxVel=" + maxVel.toFixed(2));
	check("boil: spot empties toward its gas target", m1 < 0.7 * m0,
		"m " + m0.toFixed(3) + " -> " + m1.toFixed(3) + " (rho*~" + rhoStar.toFixed(4) + ")");
	check("boil: mass exactly conserved while volume changes", driftM < 1e-6,
		"drift_m=" + driftM.toExponential(2));
	const factor = m0 / m1;
	console.log("INFO boil expansion factor " + factor.toFixed(1) + "x (rho_liq/rho_gas at T=450 is "
		+ (cfg.rhoLiq / (cfg.rhoGas0 * cfg.TBoil / 450)).toFixed(1) + "x)");
}

// 4. condense a vapor blob in cold liquid: phi_g falls, S < 0 pulls mass back in.
// high conductivity walks the blob through the plateau in harness time (the
// 0.2 default spreads <1 cell in 10 s; see convection.js testConduction).
function testCondense() {
	const cfg = makeCfg({ N: 32, scenario: "still", gravityY: 0, pressureIters: 10, kLiq: 20, kGas: 2, kSol: 20 });
	const sim = solverMod.create(cfg);
	const x0 = 13, y0 = 12, w = 6, h = 6;
	// mid-boiling blob: phi_g = 0.5 at T_boil, half the liquid density
	const m = 0.5;
	const E3 = m * (cfg.cSol * cfg.TMelt + cfg.Lf + cfg.cLiq * (cfg.TBoil - cfg.TMelt));
	setBox(sim, cfg, x0, y0, w, h, m, cfg.TBoil, E3 + 0.5 * m * cfg.Lv);
	sim.phaseEos(false);
	const t0 = totals(sim, cfg);
	const phi0 = boxMean(sim, cfg, x0, y0, w, h, "phiG");
	let minS = 0, maxVel = 0;
	for (let s = 0; s < 600; s++) {
		sim.step(1 / 60);
		const sf = sim.sfield, ms = cur(sim.mass);
		boxScan(cfg, ms, x0, y0, w, h, i => { if (sf[i] < minS) minS = sf[i]; });
		const v = sim.stats().mot.maxVel;
		if (v > maxVel) maxVel = v;
	}
	const t1 = totals(sim, cfg);
	const phi1 = boxMean(sim, cfg, x0, y0, w, h, "phiG");
	const m1 = boxMean(sim, cfg, x0, y0, w, h, "m");
	const driftM = Math.abs(t1.sumM - t0.sumM) / t0.sumM;
	check("condense: blob loses its vapor fraction to the cold liquid", phi0 > 0.4 && phi1 < 0.05,
		"phi_g " + phi0.toFixed(3) + " -> " + phi1.toFixed(4));
	check("condense: contraction source observed (S < 0)", minS < -0.01,
		"minS=" + minS.toFixed(3));
	// Converged Poisson peak is 0.472 (RB K=300 gives 0.4724, MG gives 0.4700;
	// only unconverged RB K=10 overshot to 0.5057).
	check("condense: convergence moves fluid (maxVel rises)", maxVel > 0.4,
		"maxVel=" + maxVel.toFixed(2));
	check("condense: mass conserved through the contraction", driftM < 1e-6,
		"drift_m=" + driftM.toExponential(2) + " m_end=" + m1.toFixed(3));
}

// 5. freeze flip: velocity zeroed on the mask edge, momentum charged to the ledger
function testFreezeFlip() {
	const cfg = makeCfg({ N: 16, scenario: "still", gravityY: 0 });
	const sim = solverMod.create(cfg);
	const x0 = 6, y0 = 6, w = 4, h = 4;
	const vs = cur(sim.vel);
	const i2 = (x0 + 2 + (y0 + 2) * cfg.texW) * 2;
	vs[i2] = 7; vs[i2 + 1] = -3;
	sim.phaseEos(true);
	check("freeze: liquid keeps its velocity while the mask is off", vs[i2] === 7 && sim.ledger.frozenMomentum === 0);
	setBox(sim, cfg, x0, y0, w, h, 1, 250);
	sim.phaseEos(true);
	check("freeze: mask flip zeroes every face of the frozen cell",
		vs[i2] === 0 && vs[i2 + 1] === 0 && sim.isFrozen(x0 + 2, y0 + 2));
	check("freeze: removed momentum is charged to the ledger", sim.ledger.frozenMomentum > 0,
		"frozenMomentum=" + sim.ledger.frozenMomentum.toFixed(2));
}

// 6. symmetric vapor source bound, view palette twin, and all 5 phase scenarios
function testPhaseScenariosAndViews() {
	const cfg = makeCfg({ N: 32 });
	const viewMod = require("../src/gpu/view.js");
	// Thin air cell heated across TBoil must stay strictly inside (-SMax, +SMax), never saturating
	const thinHot = solverMod.phaseOf(2e-4, solverMod.energyFromT(2e-4, 380, cfg), cfg);
	check("bounded S: thin gas cell crossing TBoil stays strictly inside (-SMax, +SMax)",
		Math.abs(thinHot.S) < cfg.SMax,
		"S=" + thinHot.S.toFixed(3) + " SMax=" + cfg.SMax.toFixed(1));
	// Phase colormap twin produces distinct colors across all 6 phases + static wall
	const seen = new Set();
	for (let p = 0; p < 6; p++) seen.add(viewMod.phaseToRgb(p, false).join(","));
	seen.add(viewMod.phaseToRgb(3, true).join(","));
	check("phaseToRgb: 7 distinct colors across phases 0..5 + static wall", seen.size === 7);
	for (const sc of ["ice", "cycle", "jet", "rain", "arch"]) {
		const scCfg = makeCfg({ N: 32, scenario: sc, pressureIters: 6, useMultigrid: true });
		Fluid.core.config.prepareScenario(scCfg);
		const sim = solverMod.create(scCfg);
		for (let step = 0; step < 30; step++) sim.step(1 / 120);
		const st = sim.stats();
		const exp = Fluid.core.stats.writeExpectedTotals({ m: 0, dr: 0, dg: 0, db: 0, E: 0 }, st.ledger);
		const driftM = Math.abs(Fluid.core.stats.driftRel(st.tot.sumM, exp.m));
		const driftE = Math.abs(Fluid.core.stats.driftRel(st.tot.sumE, exp.E));
		const countSum = st.phaseCounts.reduce((a, b) => a + b, 0);
		check("scenario " + sc + ": finite, non-negative, conserved m/E, complete phase census",
			driftM < 1e-6 && driftE < 1e-6 && st.tot.cntNeg === 0 && countSum === 32 * 32,
			"drift_m=" + driftM.toExponential(2) + " drift_E=" + driftE.toExponential(2) + " counts=" + st.phaseCounts.join("/"));
	}
}

testPlateauSweep();
testMeltBlock();
testBoilExpansion();
testCondense();
testFreezeFlip();
testPhaseScenariosAndViews();

if (failures) {
	console.log("FAILED " + failures + " phase checks");
	process.exitCode = 1;
} else {
	console.log("ALL PHASE TESTS PASSED");
}
