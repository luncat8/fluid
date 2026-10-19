"use strict";
// 0.4.0 thermal validation on the cpu oracle: derive_T, clamped ping-pong
// conduction, accounted heat sources, and the bottom-heated convection box.
// run: node experiments/convection.js | tee experiments/logs/<date>-convection.log

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const stats = Fluid.core.stats;
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
function cur(sim, pp) { return pp.cur === 0 ? pp.a : pp.b; }
function totals(sim, cfg) { return stats.totalsFromFields(cur(sim, sim.mass), cur(sim, sim.therm), cfg.texW, cfg.texH); }

// 1. derive_T: T is E/(m c) where there is substance, T_amb in vacuum
function testDerive() {
	const cfg = makeCfg({ N: 16, scenario: "still" });
	const sim = solverMod.create(cfg);
	const therm = cur(sim, sim.therm), mass = cur(sim, sim.mass);
	const i = (4 + 4 * cfg.texW) * 4;
	mass[i + 3] = 0.5; therm[i] = 0.5 * cfg.cLiq * 350; therm[i + 1] = 999;
	const j = (8 + 8 * cfg.texW) * 4;
	mass[j + 3] = 0; therm[j] = 0; therm[j + 1] = 999;
	sim.deriveT();
	check("derive_T: T = E/(m c) with substance, T_amb in vacuum, monotone in E",
		Math.abs(therm[i + 1] - 350) < 1e-4 && therm[j + 1] === cfg.TAmb,
		"T_sub=" + therm[i + 1].toFixed(3) + " T_vac=" + therm[j + 1]);
}

// 2. conduction: hot half against cold half, no flow. Pairwise exchange must
// conserve E to the f32 accumulation floor, never overshoot, and relax to the mean.
function testConduction() {
	// the plan exchange is coef*dT*dt of energy against capacity m*c, so the
	// diffusion time over L cells is L^2*C/coef: at the 0.2 default that is
	// ~hours of steps across 32 cells. a high coefficient exercises relaxation;
	// the clamp check below exercises the rate limit separately.
	const cfg = makeCfg({ N: 32, scenario: "still", conductivity: 20 });
	const sim = solverMod.create(cfg);
	const therm = cur(sim, sim.therm), mass = cur(sim, sim.mass);
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		const i = (x + y * cfg.texW) * 4;
		const T = x <= cfg.N / 2 ? 340 : 280;
		therm[i] = mass[i + 3] * cfg.cLiq * T; therm[i + 1] = T;
	}
	const E0 = totals(sim, cfg).sumE;
	let maxT = 340, minT = 280, overshoot = false;
	for (let s = 0; s < 4000; s++) {
		sim.conduction(1 / 60);
		const th = cur(sim, sim.therm);
		let hi = -Infinity, lo = Infinity;
		for (let k = 1; k < th.length; k += 4) { if (th[k] > hi) hi = th[k]; if (th[k] < lo) lo = th[k]; }
		// ghost ring keeps T=0; only interior counts
		hi = -Infinity; lo = Infinity;
		for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
			const T = th[(x + y * cfg.texW) * 4 + 1];
			if (T > hi) hi = T; if (T < lo) lo = T;
		}
		if (hi > maxT + 1e-4 || lo < minT - 1e-4) overshoot = true;
		maxT = hi; minT = lo;
	}
	const E1 = totals(sim, cfg).sumE;
	const drift = Math.abs(E1 - E0) / E0;
	check("conduction: E conserved below 1e-7 over 4000 f32 exchanges (ping-pong)",
		drift < 1e-7, "drift_rel=" + drift.toExponential(2));
	check("conduction: no overshoot, field relaxed toward the 310 mean",
		!overshoot && maxT < 315 && minT > 305,
		"T range after= [" + minT.toFixed(2) + ".." + maxT.toFixed(2) + "]");
	cfg.conductivity = 1e9;
	sim.conduction(1 / 60);
	const th2 = cur(sim, sim.therm);
	let hi2 = -Infinity, lo2 = Infinity;
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		const T = th2[(x + y * cfg.texW) * 4 + 1];
		if (T > hi2) hi2 = T; if (T < lo2) lo2 = T;
	}
	check("conduction: absurd coefficient is clamped, monotone and reported, not explosive",
		sim.thermDiag.clamped > 0 && hi2 <= maxT + 1e-4 && lo2 >= minT - 1e-4,
		"clamped faces=" + sim.thermDiag.clamped + " T range=[" + lo2.toFixed(2) + ".." + hi2.toFixed(2) + "]");
}

// 3. heat sources + ledger: heaters change E only, every joule accounted, and
// (gravity off) stored E + kinetic closes against the ledger.
function testSourcesClosure() {
	const cfg = makeCfg({ N: 32, scenario: "convection", pressureIters: 20, gravityY: 0, gravityX: 0 });
	const sim = solverMod.create(cfg);
	const t0 = sim.stats();
	for (let s = 0; s < 600; s++) sim.step(1 / 240);
	const t1 = sim.stats();
	const expE = stats.expectedTotal(sim.ledger, "E");
	const driftE = Math.abs(stats.driftRel(t1.tot.sumE, expE));
	const driftM = Math.abs(stats.driftRel(t1.tot.sumM, stats.expectedTotal(sim.ledger, "m")));
	const closure = Math.abs((expE - (t1.tot.sumE + t1.thermal.kinetic)) / expE);
	check("heat sources: injected and removed energy both reached the ledger",
		sim.ledger.heatInE > 0 && sim.ledger.heatOutE > 0,
		"in=" + sim.ledger.heatInE.toFixed(1) + " out=" + sim.ledger.heatOutE.toFixed(1));
	check("heat sources: heaters change E only (m drift below 1e-9, E drift below 1e-6)",
		driftM < 1e-9 && driftE < 1e-6,
		"drift_m=" + driftM.toExponential(2) + " drift_E=" + driftE.toExponential(2));
	check("energy budget: stored E + kinetic closes below 1e-3 with gravity off",
		closure < 1e-3, "closure_rel=" + closure.toExponential(2) +
		" kinetic=" + t1.thermal.kinetic.toExponential(2));
	check("heat sources: bottom hotter than top after 600 steps",
		t1.thermal.TMax > cfg.TAmb + 5,
		"T range=[" + t1.thermal.TMin.toFixed(1) + ".." + t1.thermal.TMax.toFixed(1) + "] mean=" + t1.thermal.TMean.toFixed(1));
	return { t0, t1 };
}

// 4. the convection box: hot fluid must rise, cold sink, rolls must appear,
// conserved quantities stay exact with the heater running.
function testConvection() {
	const cfg = makeCfg({ N: 64, scenario: "convection", pressureIters: 20 });
	const sim = solverMod.create(cfg);
	const dt = 1 / 240;
	let onset = -1, peakVel = 0;
	let hotRise = 0, hotCount = 0;
	for (let s = 0; s < 5000; s++) {
		sim.step(dt);
		if (s % 20 !== 19) continue;
		const st = sim.stats();
		if (st.mot.maxVel > peakVel) peakVel = st.mot.maxVel;
		if (onset < 0 && st.mot.maxVel > 2) onset = (s + 1) * dt;
	}
	const st = sim.stats();
	const mass = cur(sim, sim.mass), therm = cur(sim, sim.therm), vel = cur(sim, sim.vel);
	// upward velocity where the fluid is hot: the buoyancy sign check
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		const i = (x + y * cfg.texW) * 4, iv = (x + y * cfg.texW) * 2;
		// only the genuinely hot bottom layer must rise on average: with the
		// stratified start most of the box is above TRef+5, including the
		// sinking return flow, so a low threshold averages the rolls away
		if (therm[i + 1] < cfg.TRef + 35 || mass[i + 3] < cfg.mEps) continue;
		hotRise += 0.5 * (vel[iv + 1] + vel[iv + cfg.texW * 2 + 1]); hotCount++;
	}
	const meanHotV = hotCount ? hotRise / hotCount : 0;
	const driftM = Math.abs(stats.driftRel(st.tot.sumM, stats.expectedTotal(sim.ledger, "m")));
	const driftE = Math.abs(stats.driftRel(st.tot.sumE, stats.expectedTotal(sim.ledger, "E")));
	// stratification: mean T of the bottom quarter against the top quarter
	let botT = 0, topT = 0, botN = 0, topN = 0;
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		const T = therm[(x + y * cfg.texW) * 4 + 1];
		if (y <= cfg.N / 4) { botT += T; botN++; }
		if (y > cfg.N * 3 / 4) { topT += T; topN++; }
	}
	check("convection: motion onsets and reaches roll speeds",
		onset > 0 && peakVel > 2,
		"onset=" + (onset > 0 ? onset.toFixed(2) + "s" : "never") + " peak|v|=" + peakVel.toFixed(2));
	check("convection: hot fluid rises on average (buoyancy sign)",
		meanHotV > 0, "mean v in hot cells=" + meanHotV.toFixed(3) + " over " + hotCount + " cells");
	check("convection: bottom stays warmer than the cold plate",
		botT / botN > topT / topN,
		"bottom=" + (botT / botN).toFixed(1) + " top=" + (topT / topN).toFixed(1) +
		" T range=[" + st.thermal.TMin.toFixed(1) + ".." + st.thermal.TMax.toFixed(1) + "]");
	// m moves only by donor transport, so its drift here is the f32 rounding
	// floor under sustained boundary jets: measured 2.4e-6 at 2400 steps and
	// 1.1e-5 at 5000 (roughly linear, ~2e-9/step; with conduction off the
	// 2400-step run lands at 4e-8, i.e. it is rounding in the velocity
	// realisation, not a thermal leak). E is the thermal invariant of this
	// plan and stays at the tighter floor (3.1e-7 measured).
	check("convection: m and E at the f32 floor with heater and cold plate running",
		driftM < 2e-5 && driftE < 5e-6 && st.tot.cntNeg === 0,
		"drift_m=" + driftM.toExponential(2) + " drift_E=" + driftE.toExponential(2) + " neg=" + st.tot.cntNeg);
	console.log("INFO convection after 5000 steps (20.8s): maxVel=" + st.mot.maxVel.toFixed(2) +
		" heatIn=" + sim.ledger.heatInE.toFixed(0) + " heatOut=" + sim.ledger.heatOutE.toFixed(0) +
		" clampedFaces/step=" + sim.thermDiag.clamped);
}

// 5. stability sweep: 10x defaults must stay finite
function testStability() {
	const cfg = makeCfg({ N: 32, scenario: "convection", pressureIters: 10, beta: cfg0.beta * 10, conductivity: cfg0.conductivity * 10 });
	const sim = solverMod.create(cfg);
	cfg.beta = cfg0.beta * 10; // after create: prepareScenario pins the scenario beta on reset
	for (let s = 0; s < 300; s++) sim.step(1 / 240);
	const st = sim.stats();
	check("stability: finite at 10x beta and 10x conductivity",
		Number.isFinite(st.tot.sumE) && Number.isFinite(st.mot.maxVel) && st.tot.cntNeg === 0,
		"maxVel=" + st.mot.maxVel.toFixed(2) + " Tmax=" + st.thermal.TMax.toFixed(1));
}

// 6. the demo as shipped, at the default 512: the temperature map was flat
// because the box started uniform, the heater band was 2 cells and the view
// spanned 273..373. the scenario must start visibly stratified, with the
// view range matched to it and a band that scales with N.
function testScenarioPresentation(){
	const cfg=makeCfg({N:512,scenario:"convection"});
	const sim=solverMod.create(cfg);
	const st=sim.stats();
	check("presentation: convection starts stratified over most of the view range",
		st.thermal.TMax-st.thermal.TMin>=45&&cfg.TMax-cfg.TMin<=70&&
		cfg.TMin<=st.thermal.TMin+5&&cfg.TMax>=st.thermal.TMax-5,
		"T=["+st.thermal.TMin.toFixed(1)+".."+st.thermal.TMax.toFixed(1)+"] view=["+cfg.TMin+".."+cfg.TMax+"]");
	check("presentation: heater/cold-plate band scales with N",
		cfg.heatBand>=8,"heatBand="+cfg.heatBand+" at N=512");
	// rolls must be underway within the first seconds, not after a minute of
	// conducted warm-up: 2 s at the app substep dt on a mid-size box
	const cfg2=makeCfg({N:128,scenario:"convection",pressureIters:20});
	const sim2=solverMod.create(cfg2);
	for(let s=0;s<480;s++)sim2.step(1/240);
	const st2=sim2.stats();
	check("presentation: rolls underway within 2 s",
		st2.mot.maxVel>2,"maxVel="+st2.mot.maxVel.toFixed(2)+" T=["+st2.thermal.TMin.toFixed(1)+".."+st2.thermal.TMax.toFixed(1)+"]");
}

testScenarioPresentation();
testDerive();
testConduction();
testSourcesClosure();
testConvection();
testStability();
console.log(failures === 0 ? "ALL CONVECTION TESTS PASSED" : failures + " FAILURES");
process.exitCode = failures === 0 ? 0 : 1;
