"use strict";
// 0.5.0 caloric eos validation on the cpu oracle (0.0.0 section 5.5):
// plateau boundaries, T(E) monotone with plateaus, rho* continuity across
// phi_g, S sign and clamps, and the piecewise enthalpy distance the conduction
// budget (0.0.0 section 5.7) is built on.
// run: node experiments/eos.js | tee experiments/logs/<date>-eos.log

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const cfg0 = Fluid.core.config.defaults;
const P = solverMod;

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
function bounds(m, cfg) {
	// caloric branch energies at mass m (0.0.0 5.5)
	const E1 = m * cfg.cSol * cfg.TMelt;
	const E2 = E1 + m * cfg.Lf;
	const E3 = E2 + m * cfg.cLiq * (cfg.TBoil - cfg.TMelt);
	const E4 = E3 + m * cfg.Lv;
	return { E1, E2, E3, E4 };
}

// 1. plateau boundaries: phase byte and T at each side of E1..E4
function testBoundaries() {
	const cfg = makeCfg({ N: 8, scenario: "still" });
	const m = 1;
	const { E1, E2, E3, E4 } = bounds(m, cfg);
	const eps = 1e-3;
	const s = E => P.phaseOf(m, E, cfg);
	const rows = [
		["solid", s(E1 - eps), P.PHASE_SOLID, cfg.TMelt],
		["melting lo", s(E1), P.PHASE_MELTING, cfg.TMelt],
		["melting hi", s(E2 - eps), P.PHASE_MELTING, cfg.TMelt],
		["liquid lo", s(E2), P.PHASE_LIQUID, cfg.TMelt],
		["liquid hi", s(E3 - eps), P.PHASE_LIQUID, cfg.TBoil],
		["boiling lo", s(E3), P.PHASE_BOILING, cfg.TBoil],
		["boiling hi", s(E4 - eps), P.PHASE_BOILING, cfg.TBoil],
		["gas", s(E4), P.PHASE_GAS, cfg.TBoil]
	];
	let ok = true, detail = "";
	for (const [name, o, phase, T] of rows) {
		if (o.phase !== phase || Math.abs(o.T - T) > (name === "solid" ? 2 : 1e-3)) {
			ok = false; detail += " " + name + ":phase=" + o.phase + "/T=" + o.T.toFixed(2);
		}
	}
	check("plateau boundaries: phase byte and T on both sides of E1..E4", ok, detail || "8 boundary states");
}

// 2. T(E) monotone non-decreasing across the full curve, flat inside plateaus
function testMonotone() {
	const cfg = makeCfg({ N: 8, scenario: "still" });
	const m = 1;
	const { E1, E2, E4 } = bounds(m, cfg);
	const E5 = E4 + m * cfg.cGas * 200;
	let lastT = -Infinity, ok = true, plateauFlat = true;
	const n = 20000;
	for (let i = 0; i <= n; i++) {
		const E = (E5) * i / n;
		const o = P.phaseOf(m, E, cfg);
		if (o.T < lastT - 1e-4) ok = false;
		lastT = o.T;
		if (E > E1 + 1 && E < E2 - 1 && Math.abs(o.T - cfg.TMelt) > 1e-3) plateauFlat = false;
	}
	check("T(E) monotone non-decreasing over 20k samples", ok);
	check("T(E) flat at T_melt inside the fusion interval", plateauFlat);
}

// 3. energyFromT round trip through the sensible branches
function testRoundTrip() {
	const cfg = makeCfg({ N: 8, scenario: "still" });
	const Ts = [250, 273.5, 293, 340, 373.5, 450];
	let ok = true, detail = "";
	for (const T of Ts) {
		const E = P.energyFromT(1, T, cfg);
		const back = P.phaseOf(1, E, cfg).T;
		if (Math.abs(back - T) > 1e-3) { ok = false; detail += " T=" + T + "->" + back.toFixed(3); }
	}
	check("energyFromT round trips through every sensible branch", ok, detail || "6 temperatures");
}

// 4. rho* continuity and direction across the boiling plateau, and the gas law
function testRhoStar() {
	const cfg = makeCfg({ N: 8, scenario: "still" });
	const m = 1;
	const { E3, E4 } = bounds(m, cfg);
	let last = 0, first = true, ok = true, maxJump = 0;
	const n = 1000;
	for (let i = 0; i <= n; i++) {
		const o = P.phaseOf(m, E3 + (E4 - E3) * i / n, cfg);
		if (!first) {
			if (o.rhoStar > last + 1e-9) ok = false;
			maxJump = Math.max(maxJump, Math.abs(o.rhoStar - last));
		}
		first = false;
		last = o.rhoStar;
	}
	const rhoLiq = P.phaseOf(m, E3, cfg).rhoStar;
	const rhoGasEnd = P.phaseOf(m, E4, cfg).rhoStar;
	const rhoGasLaw = cfg.rhoGas0 * cfg.TBoil / cfg.TBoil;
	check("rho* decreases continuously from rho_liq to rho_gas across phi_g",
		ok && Math.abs(rhoLiq - cfg.rhoLiq) < 1e-6 && Math.abs(rhoGasEnd - rhoGasLaw) < 1e-6,
		"rho_liq=" + rhoLiq.toFixed(4) + " rho_gas=" + rhoGasEnd.toFixed(4) + " maxStep=" + maxJump.toExponential(2));
}

// 5. S sign and clamps: anti-overpack liquid, zero ambient air tier, two-sided vapor
function testSourceS() {
	const cfg = makeCfg({ N: 8, scenario: "still" });
	const liq = P.phaseOf(1.2, P.energyFromT(1.2, 293, cfg), cfg);
	const air = P.phaseOf(0.3, P.energyFromT(0.3, 293, cfg), cfg);
	const { E3, E4 } = bounds(1, cfg);
	// gas at its own target density: m = rho*(T) is equilibrium
	const Tg = 400;
	const mEq = cfg.rhoGas0 * cfg.TBoil / Tg;
	const vapor = P.phaseOf(mEq, mEq * (cfg.cSol * cfg.TMelt + cfg.Lf + cfg.cLiq * (cfg.TBoil - cfg.TMelt) + cfg.Lv + cfg.cGas * (Tg - cfg.TBoil)), cfg);
	const under = P.phaseOf(mEq * 0.5, mEq * 0.5 * (cfg.cSol * cfg.TMelt + cfg.Lf + cfg.cLiq * (cfg.TBoil - cfg.TMelt) + cfg.Lv + cfg.cGas * (Tg - cfg.TBoil)), cfg);
	// SMax below 1/tau so both clamps bind (at the defaults 1/tau caps first)
	const clampCfg = makeCfg({ N: 8, scenario: "still", SMax: 5 });
	const clampHi = P.phaseOf(1e6, P.energyFromT(1e6, 293, clampCfg), clampCfg);
	const clampLo = P.phaseOf(0.004, 0.004 * (clampCfg.cSol * clampCfg.TMelt + clampCfg.Lf + clampCfg.cLiq * (clampCfg.TBoil - clampCfg.TMelt) + clampCfg.Lv + clampCfg.cGas * (Tg - clampCfg.TBoil)), clampCfg);
	check("S: liquid overpack expands toward rho_liq", liq.S > 0 && Math.abs(liq.S - (1.2 - 1) / (1.2 * cfg.tauPhase)) < 1e-4,
		"S=" + liq.S.toFixed(3));
	check("S: thin cold ambient (air tier) carries no source", air.S === 0 && air.phase === P.PHASE_LIQUID,
		"S=" + air.S);
	check("S: vapor at its gas-law target is equilibrium", Math.abs(vapor.S) < 1e-6,
		"rho*=" + vapor.rhoStar.toFixed(5) + " m=" + mEq.toFixed(5) + " phi_g=" + vapor.phiG.toFixed(3));
	check("S: under-dense vapor converges (condensation pull)", under.S < 0,
		"S=" + under.S.toFixed(3) + " rho*=" + under.rhoStar.toFixed(5));
	check("S: clamps at +-SMax", clampHi.S === 5 && clampLo.S === -5,
		"hi=" + clampHi.S + " lo=" + clampLo.S);
}

// 6. eqDist: the piecewise enthalpy distance to thermal equilibrium
function testEqDist() {
	const cfg = makeCfg({ N: 8, scenario: "still" });
	const m = 1;
	const { E1, E2 } = bounds(m, cfg);
	const mid = 0.5 * (E1 + E2);
	const ok = P.eqDist(m, mid, cfg.TMelt, cfg) === 0
		&& Math.abs(P.eqDist(m, E1 - 10, cfg.TMelt, cfg) - 10) < 1e-6
		&& Math.abs(P.eqDist(m, E2 + 10, cfg.TMelt, cfg) - 10) < 1e-6
		&& P.eqDist(m, P.energyFromT(m, 300, cfg), 300, cfg) === 0;
	check("eqDist: latent interval is a set (0 inside), point branches at the exact energy", ok,
		"mid=" + P.eqDist(m, mid, cfg.TMelt, cfg) + " below=" + P.eqDist(m, E1 - 10, cfg.TMelt, cfg).toFixed(1));
}

testBoundaries();
testMonotone();
testRoundTrip();
testRhoStar();
testSourceS();
testEqDist();

if (failures) {
	console.log("FAILED " + failures + " eos checks");
	process.exitCode = 1;
} else {
	console.log("ALL EOS TESTS PASSED");
}
