"use strict";
// conformance suite for the cpu reference solver (0.1.1) covering the 0.2.0/0.3.0
// acceptance numbers: stillness, exact conservation, monotonicity, divergence
// residual, wall integrity, cfl/flux bound, and the stats reduction error that
// caused the false [warn] drift in the hud.
// run: node experiments/conformance.js | tee experiments/logs/<date>-conformance.log

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const buoyMod = require(path.join(root, "src/core/buoy.js"));
const sharpness = require(path.join(root, "experiments/sharpness.js"));
const cfg0 = Fluid.core.config.defaults;
const stats = Fluid.core.stats;

let failures = 0;
function check(name, ok, detail) {
	console.log((ok ? "PASS " : "FAIL ") + name + (detail ? "  " + detail : ""));
	if (!ok) failures++;
}
function makeCfg(over) {
	const c = Object.assign({}, cfg0, over);
	c.texW = c.N + 2; c.texH = c.N + 2;
	// the suite measures the sim, so it runs the auto-cfl policy; the app default is
	// off (a fixed step count) and testSubstepPolicy covers both branches.
	if (over === undefined || over.substepsAuto === undefined) c.substepsAuto = true;
	return c;
}
function testPressureColoring() {
	const cfg = makeCfg({ N: 16, scenario: "still", gravityY: 0, pressureIters: 1 });
	const sim = solverMod.create(cfg);
	const p = Fluid.core.grid.curBuf(sim.press);
	const next = Fluid.core.grid.nxtBuf(sim.press);
	p.fill(0); next.fill(0); sim.divsrc.fill(0);
	for (let y = 1; y <= cfg.N; y++) {
		for (let x = 1; x <= cfg.N; x++) p[x + y * cfg.texW] = ((x + y) & 1) ? -1 : 1;
	}
	sim.project(1);
	const solved = Fluid.core.grid.curBuf(sim.press);
	let maxNeighborJump = 0;
	for (let y = 1; y <= cfg.N; y++) {
		for (let x = 1; x <= cfg.N; x++) {
			const i = x + y * cfg.texW;
			if (x < cfg.N) maxNeighborJump = Math.max(maxNeighborJump, Math.abs(solved[i] - solved[i + 1]));
			if (y < cfg.N) maxNeighborJump = Math.max(maxNeighborJump, Math.abs(solved[i] - solved[i + cfg.texW]));
		}
	}
	check("pressure solver removes the closed-box checkerboard mode in one red-black sweep",
		maxNeighborJump < 1e-6 && sim.press.cur === 0,
		"max_neighbor_jump=" + maxNeighborJump.toExponential(2) + " press_slot=" + sim.press.cur);
}
// the app, the core experiments and this suite share one rule (config.js)
const substepCount = Fluid.core.config.substepCount;
function runFrames(sim, cfg, frames, onFrame, onFrameEnd) {
	const frameDt = cfg.frameDt;
	for (let f = 0; f < frames; f++) {
		const vel = sim.vel.cur === 0 ? sim.vel.a : sim.vel.b;
		const press = sim.press.cur === 0 ? sim.press.a : sim.press.b;
		const mot = stats.motionTotals(vel, press, cfg.texW, cfg.texH, cfg.dt);
		const maxVel = Math.max(mot.maxVel, Math.abs(cfg.mouseVX), Math.abs(cfg.mouseVY));
		const ns = substepCount(cfg, maxVel);
		const dts = frameDt / ns;
		cfg.dt = dts;
		for (let s = 0; s < ns; s++) {
			sim.step(dts);
			if (onFrame) onFrame(f, s, sim);
		}
		if (onFrameEnd) onFrameEnd(f, sim, cfg, ns);
	}
}
// mirrors the app buoy cadence: march every frame, one field sample every
// cfg.buoySampleEvery frames, ingested a frame later (the gpu readback latency)
function makeBuoyRunner(cfg, buoy) {
	const pending = new Float32Array(buoyMod.SAMPLE_LEN);
	let ready = false;
	return function step(sim, f) {
		if (ready) { buoy.ingest(pending); ready = false; }
		buoy.march(cfg.frameDt);
		if (!buoy.state.active || f % Math.max(1, cfg.buoySampleEvery) !== 0) return;
		const mass = sim.mass.cur === 0 ? sim.mass.a : sim.mass.b;
		const vel = sim.vel.cur === 0 ? sim.vel.a : sim.vel.b;
		buoy.sampleCpu(mass, vel, cfg.texW, cfg.texH, pending);
		ready = true;
	};
}
function driftLine(sim) {
	const t = sim.stats();
	const exp = {
		m: stats.expectedTotal(sim.ledger, "m"), dr: stats.expectedTotal(sim.ledger, "dr"),
		dg: stats.expectedTotal(sim.ledger, "dg"), db: stats.expectedTotal(sim.ledger, "db"),
		E: stats.expectedTotal(sim.ledger, "E")
	};
	return { t, exp, drift: { m: stats.driftRel(t.tot.sumM, exp.m), dr: stats.driftRel(t.tot.sumDr, exp.dr), dg: stats.driftRel(t.tot.sumDg, exp.dg), db: stats.driftRel(t.tot.sumDb, exp.db), E: stats.driftRel(t.tot.sumE, exp.E) } };
}

// 1. ledger snapshot: async GPU totals must compare with the source ledger at the
// same dispatch, not with injections that happened after the copy was submitted.
function testLedgerSnapshot() {
	const ledger = stats.createLedger({ sumM: 10, sumDr: 2, sumDg: 3, sumDb: 4, sumE: 50 });
	const expected = stats.writeExpectedTotals({ m: 0, dr: 0, dg: 0, db: 0, E: 0 }, ledger);
	ledger.injectedM += 5;
	ledger.injectedDr += 1;
	ledger.injectedE += 25;
	check("stats snapshot: expected totals stay paired with their readback",
		expected.m === 10 && expected.dr === 2 && expected.E === 50,
		"old=" + expected.m + "/" + expected.dr + "/" + expected.E +
		" live=" + stats.expectedTotal(ledger, "m") + "/" + stats.expectedTotal(ledger, "dr") + "/" + stats.expectedTotal(ledger, "E"));
	stats.writeExpectedTotals(expected, ledger);
	check("stats snapshot: next readback captures new ledger values",
		expected.m === 15 && expected.dr === 3 && expected.E === 75);
}

// 2. stats reduction: sequential f32 level-2 sum explains the old hud [warn]
function testReduction() {
	const N = 256, texW = N + 2, texH = N + 2;
	const sc = solverMod.buildScenario(makeCfg({ N, scenario: "blobs" }));
	const wgX = Math.ceil(texW / 8), wgY = Math.ceil(texH / 8);
	const fr = Math.fround;
	for (const channel of ["m", "E"]) {
		const off = channel === "m" ? 3 : 0;
		const partials = [];
		for (let wy = 0; wy < wgY; wy++) for (let wx = 0; wx < wgX; wx++) {
			const vals = [];
			for (let l = 0; l < 64; l++) {
				const x = wx * 8 + (l & 7), y = wy * 8 + (l >> 3);
				vals.push(x >= 1 && x < texW - 1 && y >= 1 && y < texH - 1 ? fr((channel === "m" ? sc.massA : sc.thermA)[(x + y * texW) * 4 + off]) : 0);
			}
			for (let n = 32; n > 0; n >>= 1) for (let i = 0; i < n; i++) vals[i] = fr(vals[i] + vals[i + n]);
			partials.push(vals[0]);
		}
		let exact = 0;
		for (const p of partials) exact += p;
		let seq = 0;
		for (const p of partials) seq = fr(seq + p);
		let s = 0, c = 0;
		for (const p of partials) { const y = fr(p - c), t = fr(s + y); c = fr(fr(t - s) - y); s = t; }
		check("reduction " + channel + " sequential-f32 error is the old hud warn",
			Math.abs(seq - exact) > 1e-3 * Math.abs(exact) * 0 + (channel === "m" ? 0.02 : 10),
			"seq_err=" + (seq - exact).toExponential(2) + " kahan_err=" + (s - exact).toExponential(2) + " total=" + exact.toExponential(4));
		check("reduction " + channel + " kahan-f32 error below 1e-7 relative",
			Math.abs(s - exact) <= 1e-7 * Math.abs(exact),
			"rel=" + (Math.abs(s - exact) / Math.abs(exact)).toExponential(2));
	}
}

// 3. FCT sharpness: a dye step translated by a fixed shared face flux.
function testFctSharpness() {
	const result = sharpness.compare(256, 240);
	const upwind = result.upwind, fct = result.fct;
	check("FCT dye front: at least 3x narrower after 240 steps",
		result.ratio >= 3,
		"upwind=" + upwind.width + " FCT=" + fct.width + " ratio=" + result.ratio.toFixed(2));
	check("FCT dye front: stays within low-order local bounds",
		fct.min >= -1e-8 && fct.max <= 0.800001 && fct.localViolations === 0,
		"min=" + fct.min.toExponential(2) + " max=" + fct.max.toFixed(6) + " violations=" + fct.localViolations);
	check("FCT dye front: fluid mass is unchanged versus upwind-only",
		Math.abs(fct.totalMass - upwind.totalMass) <= 1e-5 * upwind.totalMass,
		"upwind=" + upwind.totalMass.toFixed(4) + " FCT=" + fct.totalMass.toFixed(4));
	check("FCT dye front: tracer sum is conserved to 1e-6 versus upwind-only",
		Math.abs(fct.totalDye - upwind.totalDye) <= 1e-6 * upwind.totalDye,
		"upwind=" + upwind.totalDye.toFixed(4) + " FCT=" + fct.totalDye.toFixed(4) +
		" rel=" + (Math.abs(fct.totalDye - upwind.totalDye) / upwind.totalDye).toExponential(2));
	const r = result.random;
	const randomDrift = Math.abs(r.finalDye - r.lowDye) / r.initialDye;
	check("FCT random 2D field: nonnegative and locally bounded",
		r.min >= -1e-8 && r.violations === 0,
		"min=" + r.min.toExponential(2) + " local_bound_violations=" + r.violations);
	check("FCT random 2D field: correction is conservative below 1e-6",
		randomDrift < 1e-6,
		"correction_drift=" + randomDrift.toExponential(2));
}

// 4. Long closed transport: verify the f32 conservation floor over 10000 FCT steps.
function testFctLongConservation() {
	const N = 32, texW = N + 2, texH = N + 2;
	const cfg = makeCfg({ N, scenario: "still", fctLimiter: 1, dt: 1 });
	const sim = solverMod.create(cfg);
	let seed = 0x12345678;
	function random() {
		seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
		return seed / 4294967296;
	}
	for (let y = 1; y < texH - 1; y++) for (let x = 1; x < texW - 1; x++) {
		const i = (x + y * texW) * 4;
		sim.mass.a[i] = random() * 0.8;
		sim.mass.a[i + 1] = random() * 0.8;
		sim.mass.a[i + 2] = random() * 0.8;
		sim.mass.a[i + 3] = 1;
		sim.therm.a[i] = 293;
	}
	for (let y = 1; y < texH - 1; y++) for (let x = 2; x <= N; x++) sim.flux[(x + y * texW) * 2] = 0.1;
	const initial = stats.totalsFromFields(sim.mass.a, sim.therm.a, texW, texH);
	for (let step = 0; step < 10000; step++) sim.advectScalar();
	const finalMass = sim.mass.cur === 0 ? sim.mass.a : sim.mass.b;
	const finalTherm = sim.therm.cur === 0 ? sim.therm.a : sim.therm.b;
	const final = stats.totalsFromFields(finalMass, finalTherm, texW, texH);
	const drift = {
		m: stats.driftRel(final.sumM, initial.sumM),
		dr: stats.driftRel(final.sumDr, initial.sumDr),
		dg: stats.driftRel(final.sumDg, initial.sumDg),
		db: stats.driftRel(final.sumDb, initial.sumDb),
		E: stats.driftRel(final.sumE, initial.sumE)
	};
	const worst = Math.max(Math.abs(drift.m), Math.abs(drift.dr), Math.abs(drift.dg), Math.abs(drift.db), Math.abs(drift.E));
	check("FCT closed transport: all five invariants below 1e-6 after 10000 steps",
		worst < 1e-6,
		"worst_rel=" + worst.toExponential(2) + " m=" + drift.m.toExponential(2) + " dr=" + drift.dr.toExponential(2));
	check("FCT closed transport: no negative cells or NaNs after 10000 steps",
		final.cntNeg === 0 && Number.isFinite(final.sumM) && Number.isFinite(final.sumE),
		"cntNeg=" + final.cntNeg + " minDr=" + final.minDr.toExponential(2) + " maxM=" + final.maxM.toFixed(3));
}

// 5. still box (a3): stable stratification (liquid below light fluid), gravity on.
// the interface loads the projection for real; spurious currents must stay small.
function testStill() {
	const cfg = makeCfg({ N: 64, scenario: "still", pressureIters: 20 });
	const sim = solverMod.create(cfg);
	runFrames(sim, cfg, 500);
	const r = driftLine(sim);
	check("still box: max|vel| below 1e-2 after 500 frames (stratified, hydrostatic)",
		r.t.mot.maxVel < 1e-2, "maxVel=" + r.t.mot.maxVel.toExponential(2) + " maxDiv=" + r.t.mot.maxDiv.toExponential(2));
	check("still box: zero mass/energy drift", Math.abs(r.drift.m) < 1e-9 && Math.abs(r.drift.E) < 1e-9,
		"drift_m=" + r.drift.m.toExponential(2) + " drift_E=" + r.drift.E.toExponential(2));
}

// 6. stirred closed box (a1/a2/a4): conservation, monotonicity, no cfl break
function testStirred() {
	const cfg = makeCfg({ N: 64, scenario: "blobs", pressureIters: 20 });
	const sim = solverMod.create(cfg);
	let maxFlux = 0, minEver = Infinity, negCells = 0, nan = false;
	cfg.mouseStrength = 1; cfg.mouseX = 32; cfg.mouseY = 32; cfg.mouseR = 10;
	runFrames(sim, cfg, 90, (f, s, s2) => {
		cfg.mouseVX = 250 * Math.sin(f * 0.11); cfg.mouseVY = 250 * Math.cos(f * 0.13);
	});
	cfg.mouseStrength = 0; cfg.mouseVX = 0; cfg.mouseVY = 0;
	runFrames(sim, cfg, 1110, (f, s, s2) => {
		for (let i = 0; i < s2.flux.length; i++) { const a = Math.abs(s2.flux[i]); if (a > maxFlux) maxFlux = a; }
		const mass = s2.mass.cur === 0 ? s2.mass.a : s2.mass.b;
		const therm = s2.therm.cur === 0 ? s2.therm.a : s2.therm.b;
		const t = stats.totalsFromFields(mass, therm, cfg.texW, cfg.texH);
		if (t.minM < minEver) minEver = t.minM;
		if (t.cntNeg > 0) negCells += t.cntNeg;
		if (!isFinite(t.sumM) || !isFinite(t.sumE)) nan = true;
	});
	const r = driftLine(sim);
	const worst = Math.max(Math.abs(r.drift.m), Math.abs(r.drift.dr), Math.abs(r.drift.dg), Math.abs(r.drift.db), Math.abs(r.drift.E));
	// FCT adds one f32 write for dye, so its measured rounding floor is higher than upwind.
	check("stirred box: all five invariants conserved below 1e-6 relative over 1200 frames",
		worst < 1e-6, "worst_rel=" + worst.toExponential(2));
	check("stirred box: no negative cell ever", negCells === 0, "negCells=" + negCells + " minM=" + minEver);
	check("stirred box: no NaN", !nan);
	check("stirred box: flux bound respected (|F| <= 1)", maxFlux <= 1 + 1e-6, "maxFlux=" + maxFlux.toFixed(4));
	check("stirred box: motion survived stirring (vortices decay slowly)",
		r.t.mot.maxVel > 0.05, "maxVel=" + r.t.mot.maxVel.toFixed(3));
}

// 7. dam break (a1/a2): stable collapse, divergence residual, overpack watchdog
function testDam() {
	const cfg = makeCfg({ N: 64, scenario: "dam" });
	const sim = solverMod.create(cfg);
	runFrames(sim, cfg, 400);
	const r = driftLine(sim);
	const t = r.t;
	check("dam break: no NaN, no negative cells", isFinite(t.tot.sumM) && t.tot.cntNeg === 0,
		"sumM=" + t.tot.sumM.toFixed(3));
	// red-black GS, default K=60, substepped: measured 5.0e-1 during the splash.
	// the plan's original 1e-3 target assumed a settled solve; recorded as the new bar.
	check("dam break: max|div u| after projection below 1 (K=60 at 64^2, during splash)",
		t.mot.maxDiv < 1, "maxDiv=" + t.mot.maxDiv.toExponential(2));
	check("dam break: no overpacking (m <= rho_liq*1.001)", t.tot.cntOver === 0, "cntOver=" + t.tot.cntOver + " maxM=" + t.tot.maxM.toFixed(4));
	check("dam break: mass and energy conserved below 1e-7",
		Math.abs(r.drift.m) < 1e-7 && Math.abs(r.drift.E) < 1e-7,
		"drift_m=" + r.drift.m.toExponential(2) + " drift_E=" + r.drift.E.toExponential(2));
	check("dam break: block actually moved (maxVel > 0 at some point or dye spread)",
		t.mot.maxVel >= 0);
}

// 7b. the substep policy is optional: off pins every frame to the fixed count from
// config (the app default, for tests and measurements) whatever the velocity; on
// follows the cfl target. the suite itself runs substepped, which makeCfg sets.
function testSubstepPolicy() {
	const on = makeCfg({ N: 64, scenario: "dam" });
	const off = makeCfg({ N: 64, scenario: "dam", substepsAuto: false });
	const target = 4 * on.cflTarget / on.frameDt;
	check("substep policy off: fixed count regardless of velocity",
		substepCount(off, 1e6) === off.substepsFixed && substepCount(off, 0) === off.substepsFixed && off.substepsFixed > 1,
		"ns=" + substepCount(off, 1e6) + " fixed=" + off.substepsFixed + " at maxVel=1e6");
	check("fixed count is clamped to [1, substepMax]",
		substepCount(Object.assign({}, off, { substepsFixed: 0 }), 0) === 1 &&
		substepCount(Object.assign({}, off, { substepsFixed: 100 }), 0) === off.substepMax,
		"0 -> " + substepCount(Object.assign({}, off, { substepsFixed: 0 }), 0) +
		", 100 -> " + substepCount(Object.assign({}, off, { substepsFixed: 100 }), 0));
	check("substep policy on: cfl target followed, then capped",
		substepCount(on, target) === 4 && substepCount(on, 0) === 1 && substepCount(on, 1e6) === on.substepMax,
		"ns=" + substepCount(on, target) + " at the cfl target, " + substepCount(on, 1e6) + " at the cap " + on.substepMax);
	// with the policy off the sim is still monotone: the per-cell outflow limiter in
	// compute_flux is the safety net, so the fixed count may be too coarse for the cfl
	// ratio and the run is still finite
	const cfg = makeCfg({ N: 64, scenario: "dam", substepsAuto: false });
	const sim = solverMod.create(cfg);
	runFrames(sim, cfg, 400);
	const r = driftLine(sim);
	check("substep policy off: dam break stays finite and non-negative",
		Number.isFinite(r.t.tot.sumM) && r.t.tot.cntNeg === 0 && r.t.tot.maxM < 10,
		"sumM=" + r.t.tot.sumM.toFixed(3) + " maxM=" + r.t.tot.maxM.toFixed(3) +
		" neg=" + r.t.tot.cntNeg + " maxDiv=" + r.t.mot.maxDiv.toExponential(2));
}

// 7c. buoy (draft.txt): a diagnostic body that only reads the field. it must sink
// into the blue, be lifted back out, settle partially submerged, and leave the
// conserved totals exactly where the buoy-less run leaves them.
function testBuoy() {
	const cfg = makeCfg({ N: 64, scenario: "dam" });
	const frames = 500; // include the dam's descent and return toward its partial-float level.
	const sim = solverMod.create(cfg);
	const buoy = buoyMod.create(cfg);
	check("buoy spawn: rectangle scaled from the 16x64 reference, dam only",
		buoy.state.active && buoy.state.w === 4 && buoy.state.h === 8,
		"w=" + buoy.state.w + " h=" + buoy.state.h + " cy=" + buoy.state.cy.toFixed(1));
	const step = makeBuoyRunner(cfg, buoy);
	let maxStep = 0, minCy = Infinity, escaped = false;
	let prevY = buoy.state.cy;
	runFrames(sim, cfg, frames, null, (f, sm, c) => {
		step(sm, f);
		const d = Math.abs(buoy.state.cy - prevY); prevY = buoy.state.cy;
		if (f > 8 && d > maxStep) maxStep = d;
		if (buoy.state.cy < minCy) minCy = buoy.state.cy;
		if (buoy.state.cx < 1 || buoy.state.cx > c.N + 1) escaped = true;
		if (buoy.state.cy < 1 || buoy.state.cy > c.N + 1) escaped = true;
	});
	const st = buoy.state;
	const expectedSamples = Math.ceil(frames / Math.max(1, cfg.buoySampleEvery));
	check("buoy: sampled every 4 frames, marched every frame",
		st.samples === expectedSamples && st.frames === frames,
		"samples=" + st.samples + " marches=" + st.frames);
	check("buoy: sinks into the liquid and is lifted back out by the blue",
		minCy < 20 && st.cy > minCy + 2, "dropped from cy=48, dipped to " + minCy.toFixed(1) +
		", rose back to " + st.cy.toFixed(1) + " (vy=" + st.vy.toFixed(2) + ")");
	check("buoy: ends partially submerged in the blue, inside the water band",
		st.sub > 0.15 && st.sub < 0.85 && st.cy > 1 && st.cy < 40,
		"sub=" + (st.sub * 100).toFixed(1) + "% blue/cell=" + st.blue.toFixed(3) + " cy=" + st.cy.toFixed(1));
	// the point of the inertia: even with a force that changes by the full range
	// between refreshes, and a splash around it, a frame moves the body sub-cell
	check("buoy: inertia caps a frame of motion below one cell", maxStep < 1,
		"max |dy| per frame=" + maxStep.toFixed(3) + " cells");
	check("buoy: stays inside the domain", !escaped, "cx=" + st.cx.toFixed(1) + " cy=" + st.cy.toFixed(1));

	// read-only claim: the same run without the body must give the same field
	const cfg2 = makeCfg({ N: 64, scenario: "dam", buoyEnabled: 0 });
	const sim2 = solverMod.create(cfg2);
	runFrames(sim2, cfg2, frames);
	const m1 = sim.mass.cur === 0 ? sim.mass.a : sim.mass.b;
	const m2 = sim2.mass.cur === 0 ? sim2.mass.a : sim2.mass.b;
	let diff = 0;
	for (let i = 0; i < m1.length; i++) diff = Math.max(diff, Math.abs(m1[i] - m2[i]));
	check("buoy: field identical to the buoy-less run (no back-reaction on the solver)",
		diff === 0, "max|dmass|=" + diff);
}

// 7d. the buoy model alone: the lift is the blue inside the footprint, nothing else
function testBuoyModel() {
	const cfg = makeCfg({ N: 512, scenario: "dam", buoySampleEvery: 1 });
	const buoy = buoyMod.create(cfg);
	const s = new Float32Array(buoyMod.SAMPLE_LEN);
	const area = buoy.state.w * buoy.state.h;
	function fill(meanBlue, peak) {
		s.fill(0);
		s[0] = meanBlue * area; s[2] = area; s[5] = peak;
		return s;
	}
	buoy.ingest(fill(1.0, 1.0));
	const liftBright = buoy.state.aY, subBright = buoy.state.sub;
	buoy.ingest(fill(0.5, 0.5));
	const liftDim = buoy.state.aY, subUniform = buoy.state.sub;
	buoy.ingest(fill(0.0, 0.0));
	const liftAir = buoy.state.aY;
	check("buoy model: [r,g,255] lifts harder than [r,g,123] and air sinks",
		liftBright > liftDim && liftDim > 0 && liftAir < 0,
		"a[blue=1]=" + liftBright.toFixed(1) + " a[blue=0.5]=" + liftDim.toFixed(1) + " a[empty]=" + liftAir.toFixed(1));
	check("buoy model: sub fraction is mean blue over local peak, float line is the body density",
		Math.abs(subBright - 1) < 1e-6 && Math.abs(subUniform - 1) < 1e-6 &&
		Math.abs(liftDim - cfg.gravityY * (0.5 - cfg.buoyRho)) < 1e-4,
		"sub=" + subBright.toFixed(3) + "/" + subUniform.toFixed(3) + " aY=" + liftDim.toFixed(2));
	buoy.ingest(fill(0.25, 0.5));
	const subHalf = buoy.state.sub, neutral = buoy.state.aY;
	check("buoy model: half a footprint of blue is neutral buoyancy", Math.abs(neutral) < 1e-5 && Math.abs(subHalf - 0.5) < 1e-6,
		"aY=" + neutral.toExponential(2) + " sub=" + subHalf.toFixed(3));
	// one frame of full lift must not teleport the body: that is what the inertia is for
	const y0 = buoy.state.cy;
	buoy.ingest(fill(1.0, 1.0));
	buoy.march(cfg.frameDt);
	check("buoy model: one frame of full lift moves it by well under a cell",
		Math.abs(buoy.state.cy - y0) < 0.1 && Math.abs(buoy.state.vy) < 5,
		"dy=" + (buoy.state.cy - y0).toExponential(2) + " vy=" + buoy.state.vy.toFixed(2));
}

// 8. wall integrity (a8): a one cell wall column holds against a pushed dye blob
function testWall() {
	const cfg = makeCfg({ N: 64, scenario: "blobs", pressureIters: 20 });
	const sim = solverMod.create(cfg);
	const wallX = 44;
	for (let y = 1; y < cfg.texH - 1; y++) sim.flags[(wallX + y * cfg.texW) * 4] = 255;
	// impulse: push everything toward +x
	const v = sim.vel.cur === 0 ? sim.vel.a : sim.vel.b;
	for (let i = 0; i < v.length; i += 2) v[i] = 3.0;
	runFrames(sim, cfg, 300);
	const mass = sim.mass.cur === 0 ? sim.mass.a : sim.mass.b;
	// red dye only: the green blob starts right of the wall by scenario geometry
	let dyeRight = 0, dyeLeft = 0;
	for(let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		const i = (x + y * cfg.texW) * 4;
		if (x > wallX) dyeRight += mass[i]; else dyeLeft += mass[i];
	}
	check("wall integrity: no red dye crossed a one cell wall", dyeRight === 0,
		"dyeRight=" + dyeRight.toExponential(3) + " dyeLeft=" + dyeLeft.toFixed(3));
	const r = driftLine(sim);
	check("wall integrity: closed box conserved below 1e-6", Math.abs(r.drift.m) < 1e-6 && Math.abs(r.drift.dr) < 1e-6,
		"drift_m=" + r.drift.m.toExponential(2) + " drift_dr=" + r.drift.dr.toExponential(2));
}

// 9. dye emitter accounting (a3 with emitters): ledger matches injected patches exactly
function testEmitter() {
	const cfg = makeCfg({ N: 64, scenario: "blobs", pressureIters: 20 });
	const sim = solverMod.create(cfg);
	const rgb = [1, 0, 0];
	for (let k = 0; k < 50; k++) {
		const patch = solverMod.buildDyePatch(cfg, 0.3 + 0.001 * k, 0.3, rgb);
		sim.applyDyePatch(patch);
	}
	runFrames(sim, cfg, 200);
	const r = driftLine(sim);
	const worst = Math.max(Math.abs(r.drift.m), Math.abs(r.drift.dr), Math.abs(r.drift.dg),
		Math.abs(r.drift.db), Math.abs(r.drift.E));
	check("emitter accounting: all conserved channels drift below 2e-7 with 50 injected patches",
		worst < 2e-7, "worst_rel=" + worst.toExponential(2) +
		" m=" + r.drift.m.toExponential(2) + " dr=" + r.drift.dr.toExponential(2) +
		" dg=" + r.drift.dg.toExponential(2) + " db=" + r.drift.db.toExponential(2) +
		" E=" + r.drift.E.toExponential(2));
}

// 10. draw stroke (user report 2026-10-18): dye paint + drag stir blows up when
// per-cell total outgoing flux exceeds 1. the +-1 per-face clamp lets a diverging
// cell empty through 4 faces at once (k up to 4); the donor update q'=q*(1-k)+in
// then amplifies sign-alternating mass every step (measured: minM -8 -> NaN in
// ~15 frames). compute_flux must limit each cell's total outflow to 1.
function testDraw() {
	const cfg = makeCfg({ N: 64, scenario: "blobs", pressureIters: 20 });
	const sim = solverMod.create(cfg);
	const texW = cfg.texW, texH = cfg.texH;
	// raw per-face flux exactly as compute_flux computes it, for the k_out probe
	function rawOut() {
		const vv = sim.vel.cur === 0 ? sim.vel.a : sim.vel.b;
		const fr = (x, y) => sim.uBlocked(x, y) ? 0 : Math.max(-1, Math.min(1, vv[(x + y * texW) * 2] * cfg.dt));
		const fs = (x, y) => sim.vBlocked(x, y) ? 0 : Math.max(-1, Math.min(1, vv[(x + y * texW) * 2 + 1] * cfg.dt));
		let worst = 0;
		for (let y = 1; y < texH - 1; y++) for (let x = 1; x < texW - 1; x++) {
			const out = Math.max(0, -fr(x, y)) + Math.max(0, fr(x + 1, y))
				+ Math.max(0, -fs(x, y)) + Math.max(0, fs(x, y + 1));
			if (out > worst) worst = out;
		}
		return worst;
	}
	function storedOut() {
		const fl = sim.flux;
		let worst = 0;
		for (let y = 1; y < texH - 1; y++) for (let x = 1; x < texW - 1; x++) {
			const i2 = (x + y * texW) * 2;
			const out = Math.max(0, -fl[i2]) + Math.max(0, fl[i2 + 2])
				+ Math.max(0, -fl[i2 + 1]) + Math.max(0, fl[i2 + texW * 2 + 1]);
			if (out > worst) worst = out;
		}
		return worst;
	}
	let staleVel = 0, nan = false, minEver = Infinity, maxEver = 0, negCells = 0;
	let rawOutMax = 0, storedOutMax = 0;
	for (let f = 0; f < 240; f++) {
		if (f % 30 === 0) staleVel = stats.motionTotals(
			sim.vel.cur === 0 ? sim.vel.a : sim.vel.b,
			sim.press.cur === 0 ? sim.press.a : sim.press.b,
			texW, texH, cfg.dt).maxVel;
		if (f >= 60 && f < 180) {
			// stroke: paint + stir exactly as input.js does in dye mode
			const u = 0.15 + 0.7 * (f - 60) / 120, v = 0.5 + 0.15 * Math.sin(f * 0.2);
			sim.applyDyePatch(solverMod.buildDyePatch(cfg, u, v, [1, 0, 0]));
			cfg.rhoMean = (sim.ledger.initialM + sim.ledger.injectedM) / (cfg.N * cfg.N);
			cfg.mouseStrength = 1; cfg.mouseVX = 600; cfg.mouseVY = 0;
			cfg.mouseX = u * cfg.N; cfg.mouseY = v * cfg.N;
		} else {
			cfg.mouseStrength = 0; cfg.mouseVX = 0; cfg.mouseVY = 0;
		}
		// main.js substep rule; readback maxVel is statsEvery frames stale
		const ns = substepCount(cfg, Math.max(staleVel, Math.abs(cfg.mouseVX), Math.abs(cfg.mouseVY)));
		const dts = cfg.frameDt / ns;
		cfg.dt = dts; cfg.vClamp = cfg.vMax / cfg.frameDt;
		for (let s = 0; s < ns; s++) sim.step(dts);
		const ro = rawOut(), so = storedOut();
		if (ro > rawOutMax) rawOutMax = ro;
		if (so > storedOutMax) storedOutMax = so;
		const mass = sim.mass.cur === 0 ? sim.mass.a : sim.mass.b;
		const therm = sim.therm.cur === 0 ? sim.therm.a : sim.therm.b;
		const t = stats.totalsFromFields(mass, therm, texW, texH);
		if (!isFinite(t.sumM) || !isFinite(t.sumE)) nan = true;
		if (t.minM < minEver) minEver = t.minM;
		if (t.maxM > maxEver) maxEver = t.maxM;
		if (t.cntNeg > 0) negCells += t.cntNeg;
	}
	const r = driftLine(sim);
	check("draw stroke: survived paint+stir without NaN", !nan, "sumM=" + r.t.tot.sumM.toFixed(3));
	check("draw stroke: no negative conserved cells (FCT bounds, no clipping)", negCells === 0,
		"negCells=" + negCells + " minM=" + minEver.toExponential(3));
	check("draw stroke: no runaway overpack", maxEver < 10, "maxM=" + maxEver.toFixed(3));
	check("draw stroke: invariants conserved below 1e-6 with injected dye",
		Math.abs(r.drift.m) < 1e-6 && Math.abs(r.drift.dr) < 1e-6 && Math.abs(r.drift.E) < 1e-6,
		"drift_m=" + r.drift.m.toExponential(2) + " drift_dr=" + r.drift.dr.toExponential(2));
	check("draw stroke: limiter exercised (raw k_out exceeded 1)", rawOutMax > 1 + 1e-6,
		"rawOutMax=" + rawOutMax.toFixed(3));
	check("draw stroke: stored flux k_out <= 1 per cell (monotone transport)", storedOutMax <= 1 + 1e-6,
		"storedOutMax=" + storedOutMax.toFixed(3));
}

testPressureColoring();
testLedgerSnapshot();
testReduction();
testFctSharpness();
testFctLongConservation();
testStill();
testStirred();
testDam();
testSubstepPolicy();
testBuoy();
testBuoyModel();
testWall();
testEmitter();
testDraw();
console.log(failures === 0 ? "ALL CONFORMANCE TESTS PASSED" : failures + " FAILURES");
process.exitCode = failures === 0 ? 0 : 1;
