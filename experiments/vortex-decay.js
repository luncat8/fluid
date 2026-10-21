"use strict";
// 0.6.0 t5 decision metric: how much of a stirred vortex each velocity advection
// scheme keeps. viscosity is 0, so every joule that disappears is numerical
// diffusion and KE(t)/KE(0) is the number plan 4.2 ranks (adopt at >= 1.25x
// retained energy for <= 15 % extra frame time). peak vorticity and enstrophy
// come along because "soft vortices" is a sharpness complaint, not an energy one.
//
// the loop is velocity only: mass stays uniform, so the projection is constant
// coefficient and can neither cause nor hide the advection difference. dt=1/240
// with an 80 cells/s rim is what the auto policy runs at 60 fps, so the 0.33
// cell per step displacement is the product's, not a tuned one.
//
// run: node experiments/vortex-decay.js [--n 128] [--t 4] [--no-proj]
//      tee experiments/logs/<date>-vortex-decay.log

const assert = require("node:assert/strict");
const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/multigrid.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const grid = Fluid.core.grid;
const cfg0 = Fluid.core.config.defaults;
const DT = 1 / 240;
const PEAK = 80;			// rim speed in cells/s
const SIGMA_FRAC = 1 / 6;	// vortex radius as a fraction of N
const SCHEMES = [
	{ name: "sl", over: { advectMac: 0 } },
	{ name: "mac", over: { advectMac: 1, macLimiter: 1 } },
	{ name: "mac-raw", over: { advectMac: 1, macLimiter: 0 } }
];
// 0.6.0 t6: vorticity confinement is a force, not an advection scheme, so it
// rides on the shipping scheme as another row of the same stirred vortex. eps is
// in 1/s (F = eps*h*(N x omega)); the force adds energy on purpose, so the
// retained-KE row stops being a diffusion metric for these rows and the
// vorticity and enstrophy rows are the ones that rank them. the sweep that
// picked these two is in `2026-10-21-vortex-decay-conf.log`: eps=2 is the
// setting that costs no divergence, eps=8 is the sharp one, eps=32 is the
// over-driven row that shows where the force stops being free.
const CONF_EPS = [2, 8];

function make(N, over) {
	const cfg = Object.assign({}, cfg0, { N, scenario: "still" }, over);
	cfg.texW = N + 2;
	cfg.texH = N + 2;
	Fluid.core.config.prepareScenario(cfg);
	// closed box, no buoyancy, no heat, no viscosity: the only sink in the system
	// is the scheme under test.
	cfg.gravityX = 0;
	cfg.gravityY = 0;
	cfg.thermalSources = 0;
	cfg.viscosity = 0;
	cfg.substepsAuto = false;
	cfg.substepsFixed = 1;
	return { cfg, sim: solverMod.create(cfg) };
}
// gaussian vortex on the staggered faces: each component is evaluated at its own
// face position, and blocked faces stay 0 (the wall). analytically divergence
// free, so what the projection removes is the scheme's error, not the setup's.
function paintVortex(sim, cfg) {
	const N = cfg.N, texW = cfg.texW, texH = cfg.texH;
	const cx = N * 0.5, cy = N * 0.5, sigma = N * SIGMA_FRAC;
	const invS2 = 1 / (2 * sigma * sigma);
	// (r/sigma) exp(-r^2/2 sigma^2) peaks at exp(-1/2) when r = sigma
	const A = PEAK / Math.exp(-0.5);
	const vel = grid.curBuf(sim.vel);
	vel.fill(0);
	for (let y = 1; y < texH - 1; y++) {
		for (let x = 1; x < texW - 1; x++) {
			const vi = (x + y * texW) * 2;
			if (!sim.uBlocked(x, y)) {
				const dx = (x - 1) - cx, dy = (y - 0.5) - cy;
				vel[vi] = -A * (dy / sigma) * Math.exp(-(dx * dx + dy * dy) * invS2);
			}
			if (!sim.vBlocked(x, y)) {
				const dx = (x - 0.5) - cx, dy = (y - 1) - cy;
				vel[vi + 1] = A * (dx / sigma) * Math.exp(-(dx * dx + dy * dy) * invS2);
			}
		}
	}
}
// one pass over the field into a caller owned object: no per face allocation.
function measure(sim, cfg, out) {
	const vel = grid.curBuf(sim.vel), mass = grid.curBuf(sim.mass);
	const texW = cfg.texW, texH = cfg.texH;
	let ke = 0, peakVort = 0, enstrophy = 0, peakSpeed = 0, peakDiv = 0;
	for (let y = 1; y < texH - 1; y++) {
		for (let x = 1; x < texW - 1; x++) {
			const i = x + y * texW, vi = i * 2;
			const uL = vel[vi], uR = vel[vi + 2], vB = vel[vi + 1], vT = vel[vi + texW * 2 + 1];
			const uc = 0.5 * (uL + uR), vc = 0.5 * (vB + vT);
			ke += 0.5 * mass[i * 4 + 3] * (uc * uc + vc * vc);
			const w = (vT - vB) - (uR - uL);
			enstrophy += w * w;
			const aw = Math.abs(w);
			if (aw > peakVort) peakVort = aw;
			const s = Math.max(Math.abs(uL), Math.abs(uR), Math.abs(vB), Math.abs(vT));
			if (s > peakSpeed) peakSpeed = s;
			const d = Math.abs(uR - uL + vT - vB) / DT;
			if (d > peakDiv) peakDiv = d;
		}
	}
	out.t = cfg.time;
	out.ke = ke;
	out.peakVort = peakVort;
	out.enstrophy = enstrophy;
	out.peakSpeed = peakSpeed;
	out.peakDiv = peakDiv;
}
function run(name, N, T, proj, over) {
	const { cfg, sim } = make(N, over);
	sim.reset();
	paintVortex(sim, cfg);
	const iters = Fluid.core.config.solverIters(cfg);
	const steps = Math.round(T / DT);
	const perSec = Math.round(1 / DT);
	const m = { t: 0, ke: 0, peakVort: 0, enstrophy: 0, peakSpeed: 0, peakDiv: 0 };
	measure(sim, cfg, m);
	const rows = [Object.assign({}, m)];
	const r0Peak = m.peakSpeed;
	let limitTotal = 0, limitPeak = 0, clampTotal = 0, peakSpeedMax = m.peakSpeed;
	let advNs = 0n, forceNs = 0n;
	const t0 = Date.now();
	for (let s = 1; s <= steps; s++) {
		// the force slice carries the confinement pass, and it is timed on its
		// own for the same reason as the advection slice: the step number is
		// dominated by the projection.
		const tf = process.hrtime.bigint();
		sim.forces(DT);
		forceNs += process.hrtime.bigint() - tf;
		// the advection slice is timed on its own: the whole-step number is
		// dominated by the projection, and the device bar is a frame-time bar.
		const ta = process.hrtime.bigint();
		sim.advectVelocity(DT);
		advNs += process.hrtime.bigint() - ta;
		if (proj) {
			sim.poissonRhs(DT);
			sim.project(iters);
			sim.applyGrad(DT);
		}
		cfg.frame++;
		cfg.time += DT;
		limitTotal += sim.diag.macLimit;
		clampTotal += sim.diag.macClamp;
		if (sim.diag.macLimit > limitPeak) limitPeak = sim.diag.macLimit;
		if (s % perSec !== 0 && s !== steps) continue;
		measure(sim, cfg, m);
		if (m.peakSpeed > peakSpeedMax) peakSpeedMax = m.peakSpeed;
		rows.push(Object.assign({}, m));
	}
	const ms = (Date.now() - t0) / steps;
	const advMs = Number(advNs) / 1e6 / steps;
	const forceMs = Number(forceNs) / 1e6 / steps;
	// the raw row exists to show what the limiter prevents, so its overshoot is a
	// result, not a failure. every scheme the product could ship must hold 0.
	if (cfg.macLimiter || !cfg.advectMac) {
		assert.equal(clampTotal, 0, name + ": vClamp watchdog must stay 0, got " + clampTotal);
		assert.ok(peakSpeedMax <= r0Peak * (1 + 1e-6),
			name + ": the limited scheme must not create a new extremum, " + peakSpeedMax + " vs " + r0Peak);
	}
	// the confinement force is an energy source, so it may not create a new
	// extremum the way an advection scheme may not. what it may not do is push
	// the velocity into the absolute clamp: that is the blowup signal.
	const velClamp = sim.diag.velClamp;
	assert.equal(velClamp, 0, name + ": the velocity clamp watchdog must stay 0, got " + velClamp);
	return { name, rows, limitTotal, limitPeak, clampTotal, peakSpeedMax, velClamp, ms, advMs, forceMs, eps: cfg.vorticityConfinement };
}
function keep(v, v0) {
	return v0 > 0 ? v / v0 : 0;
}
function printRun(r) {
	const r0 = r.rows[0];
	console.log("  " + r.name.padEnd(8) + " KE0=" + r0.ke.toExponential(4) +
		" peakVort0=" + r0.peakVort.toFixed(3) + " peakSpeed0=" + r0.peakSpeed.toFixed(2) +
		" maxDiv0=" + r0.peakDiv.toExponential(2));
	for (let i = 1; i < r.rows.length; i++) {
		const row = r.rows[i];
		console.log("    t=" + row.t.toFixed(0).padStart(2) + "s" +
			" KE keep=" + keep(row.ke, r0.ke).toFixed(6) +
			" vort keep=" + keep(row.peakVort, r0.peakVort).toFixed(6) +
			" enst keep=" + keep(row.enstrophy, r0.enstrophy).toFixed(6) +
			" peakSpeed=" + row.peakSpeed.toFixed(2) +
			" maxDiv=" + row.peakDiv.toExponential(2));
	}
	console.log("    limiter moved " + r.limitTotal + " faces (" + r.limitPeak +
		" worst step) · clamp=" + r.clampTotal + " · peakSpeed max=" + r.peakSpeedMax.toFixed(2) +
		" · velClamp=" + r.velClamp +
		" · cpu=" + r.ms.toFixed(2) + " ms/step (advection " + r.advMs.toFixed(3) +
		", forces " + r.forceMs.toFixed(3) + ")");
}
// on a field with no curvature the correction is (src - back)/2 = 0, so mac must
// reproduce semi-lagrangian exactly. a nonzero difference in the interior is a
// bias, not a sharpening, and would invalidate every decay row below. the wall
// band is excluded from that and asserted separately: a backward trace from the
// face next to the wall reads the blocked face (0), so the raw correction
// extrapolates past the boundary value and the limiter is what holds it there.
function constantField(N) {
	const { cfg, sim } = make(N, {});
	sim.reset();
	const U = 37, texW = cfg.texW, texH = cfg.texH;
	const paint = () => {
		const vel = grid.curBuf(sim.vel);
		vel.fill(0);
		for (let y = 1; y < texH - 1; y++) {
			for (let x = 1; x < texW - 1; x++) {
				if (!sim.uBlocked(x, y)) vel[(x + y * texW) * 2] = U;
			}
		}
	};
	const read = () => {
		const vel = grid.curBuf(sim.vel);
		const out = new Float32Array(texW * texH);
		for (let y = 1; y < texH - 1; y++) {
			for (let x = 1; x < texW - 1; x++) out[x + y * texW] = vel[(x + y * texW) * 2];
		}
		return out;
	};
	const one = (mac, limiter) => {
		cfg.advectMac = mac;
		cfg.macLimiter = limiter;
		paint();
		sim.advectVelocity(DT);
		return read();
	};
	const sl = one(0, 1);
	const mac = one(1, 1);
	const raw = one(1, 0);
	// departure taps reach x-1, arrival taps reach x+1, so a face is clear of the
	// blocked texels at 1 and N+1 from x=3 to x=N-1
	let diff = 0, err = 0, wallFaces = 0, wallOver = 0, rawOver = 0;
	for (let y = 1; y < texH - 2; y++) {
		for (let x = 3; x < texW - 3; x++) {
			if (sim.uBlocked(x, y)) continue;
			const i = x + y * texW;
			diff = Math.max(diff, Math.abs(mac[i] - sl[i]));
			err = Math.max(err, Math.abs(sl[i] - U));
		}
	}
	for (let y = 1; y < texH - 1; y++) {
		for (let x = 1; x < texW - 1; x++) {
			if (sim.uBlocked(x, y) || (x >= 3 && x < texW - 3)) continue;
			wallFaces++;
			wallOver = Math.max(wallOver, mac[x + y * texW] - U);
			rawOver = Math.max(rawOver, raw[x + y * texW] - U);
		}
	}
	console.log("  constant field, interior: mac-sl max diff=" + diff.toExponential(2) +
		" sl error vs U=" + err.toExponential(2));
	console.log("  constant field, wall band: " + wallFaces + " faces, overshoot past U: limited=" +
		wallOver.toExponential(2) + " raw=" + rawOver.toExponential(2) +
		" (" + (rawOver / U * 100).toFixed(1) + " % per step, compounding)");
	assert.ok(diff === 0, "mac-cormack must be identical to semi-lagrangian on a constant field, diff " + diff);
	assert.ok(err === 0, "semi-lagrangian must preserve a constant field, error " + err);
	assert.ok(wallOver <= 0, "the limited scheme must not overshoot at the wall, " + wallOver);
	assert.ok(rawOver > 0, "the raw correction is expected to overshoot at the wall, that is why the limiter is mandatory");
}
function at(r, t) {
	for (const row of r.rows) if (Math.abs(row.t - t) < 1e-9) return row;
	return r.rows[r.rows.length - 1];
}
function decision(sl, mac, T) {
	console.log("== decision (plan 4.2: adopt at >= 1.25x retained energy for <= 15 % frame time) ==");
	const marks = T > 1 ? [1, T] : [T];
	for (const t of marks) {
		const a = at(sl, t), b = at(mac, t);
		const ratio = a.ke > 0 ? b.ke / a.ke : 0;
		const slKeep = keep(a.ke, sl.rows[0].ke);
		console.log("  t=" + t + "s  sl keeps " + slKeep.toFixed(4) +
			" of KE, mac keeps " + keep(b.ke, sl.rows[0].ke).toFixed(6) +
			"  ->  mac/sl=" + ratio.toFixed(3) + "x" +
			(ratio >= 1.25 ? "  clears the bar" : "  below the bar") +
			"  (ceiling " + (slKeep > 0 ? 1 / slKeep : Infinity).toFixed(3) +
			"x: no scheme can beat 1/keep, so a window where sl has lost under" +
			" 20 % cannot reach 1.25x at all)");
	}
	const cost = sl.ms > 0 ? (mac.ms - sl.ms) / sl.ms : 0;
	const advCost = sl.advMs > 0 ? (mac.advMs - sl.advMs) / sl.advMs : 0;
	const share = sl.ms > 0 ? 100 * sl.advMs / sl.ms : 0;
	console.log("  cpu step +" + (cost * 100).toFixed(1) + " % (" + sl.ms.toFixed(2) + " -> " +
		mac.ms.toFixed(2) + " ms/step); the advection slice alone is " + sl.advMs.toFixed(3) +
		" -> " + mac.advMs.toFixed(3) + " ms (+" + (advCost * 100).toFixed(0) + " %) and is " +
		share.toFixed(0) + " % of that step.");
	console.log("  the 15 % bar is a device number and the wgsl kernel does not exist, so" +
		" neither cpu number is the gate. on the device the projection is 67-87 % of the" +
		" substep and the whole vel group is 2 % of it (plan 3, `05_55` jet).");
}
// 0.6.0 t6: the confinement rows are read against the shipping scheme. the
// energy row is not a ranking here (the force adds energy on purpose); the
// ranking is the vorticity and the enstrophy it keeps, against a bounded
// divergence and an untouched velocity clamp.
function confinementDecision(sl, rows, T) {
	console.log("== confinement decision (plan 4.2 t6: a force on the shipping scheme) ==");
	for (const r of rows) {
		const a = at(sl, T), b = at(r, T);
		const vort = a.peakVort > 0 ? b.peakVort / a.peakVort : 0;
		const enst = a.enstrophy > 0 ? b.enstrophy / a.enstrophy : 0;
		const ke = a.ke > 0 ? b.ke / a.ke : 0;
		console.log("  eps=" + r.eps + " 1/s at t=" + T + "s: peak vorticity " + a.peakVort.toFixed(3) +
			" -> " + b.peakVort.toFixed(3) + " (" + vort.toFixed(3) + "x sl), enstrophy " +
			enst.toFixed(3) + "x, KE " + ke.toFixed(3) + "x (the force adds energy), maxDiv " +
			a.peakDiv.toExponential(2) + " -> " + b.peakDiv.toExponential(2) +
			", peak speed " + b.peakSpeed.toFixed(2) + " (initial " +
			sl.rows[0].peakSpeed.toFixed(2) + "), velClamp " + r.velClamp + ".");
	}
}

const args = process.argv.slice(2);
function opt(name, fallback) {
	const i = args.indexOf(name);
	return i >= 0 ? Number(args[i + 1]) : fallback;
}
const N = opt("--n", 128);
const T = opt("--t", 4);
// projection is the product path and the default. without it the momentum
// equation is inviscid Burgers: the vortex steepens instead of decaying, peak
// vorticity rises and maxDiv reaches 1e4, so that row attributes diffusion but
// is not a candidate anyone could ship.
const proj = !args.includes("--no-proj");
// the confinement strengths to sweep, in 1/s. `--eps 0,1,4` overrides.
const epsList = args.includes("--eps")
	? args[args.indexOf("--eps") + 1].split(",").map(Number)
	: CONF_EPS;

console.log("vortex-decay · N=" + N + " sigma=" + (N * SIGMA_FRAC).toFixed(1) +
	" rim=" + PEAK + " cells/s · dt=" + DT + " · " + Math.round(T / DT) + " steps (" + T + " s)");
console.log("== fixed points ==");
constantField(N);
console.log(proj ? "== with the shipping projection (mg 2 cycles, K20) =="
	: "== no projection (Burgers: steepens, attribution row only) ==");
const runs = SCHEMES.map(s => run(s.name, N, T, proj, s.over));
for (const r of runs) printRun(r);
if (proj) {
	// the confinement rows are a force on the shipping scheme, so they are read
	// on the product path. without the projection the momentum equation is
	// inviscid Burgers and the force has nothing to work against: that row
	// belongs to the advection schemes, not to the force.
	const confRuns = epsList.map(eps => run("sl+conf" + eps, N, T, proj,
		{ advectMac: 0, vorticityConfinement: eps }));
	for (const r of confRuns) printRun(r);
	decision(runs[0], runs[1], T);
	confinementDecision(runs[0], confRuns, T);
} else console.log("== no decision: the bar is read on the projection row (`node experiments/vortex-decay.js`) ==");
console.log("done");
