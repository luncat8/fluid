"use strict";
// 0.6.0 t5 decision metric: how much of a stirred vortex each velocity advection
// scheme keeps. viscosity is 0, so every joule that disappears is numerical
// diffusion and KE(t)/KE(0) is the number plan 4.2 ranks (adopt at >= 1.25x
// retained energy for <= 15 % extra frame time). peak vorticity and enstrophy
// come along because "soft vortices" is a sharpness complaint, not an energy one,
// and errL2 against the initial field because the vortex is a steady solution:
// it is the only row here that punishes a scheme for distorting the profile
// while keeping its integrals.
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
// retained-KE row stops being a diffusion metric for these rows. they are
// ranked on errL2 against the (steady) initial field, with the vorticity and
// enstrophy keeps read as the concentration the force buys that error with:
// every eps overshoots peak vorticity past its initial value, so "more
// vorticity" alone would rank the most distorted row first. the sweep that
// picked these two is in `2026-10-22-vortex-decay-conf.log` (eps 0.5..8):
// at 4 s eps=1 improves errL2 0.274 -> 0.201 with the divergence unchanged,
// eps=2 is the errL2 optimum (0.161, enstrophy keep 0.934) for 1.1x the
// divergence, and eps=4/8 are over-driven (errL2 turns back up, divergence
// 3x/15x, peak speed past its initial value). the 2026-10-21 conf rows and
// their eps=2/8 picks used a curl that was not a curl; see curlFixedPoints.
const CONF_EPS = [1, 2];

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
// errL2 is the distance from the initial field, relative to its norm: the
// gaussian vortex is a steady solution, so a scheme that merely concentrates
// the core (peak vorticity up, energy wherever) is wrong in a way the three
// keep ratios cannot rank. this ratio can.
function measure(sim, cfg, out, vel0) {
	const vel = grid.curBuf(sim.vel), mass = grid.curBuf(sim.mass);
	const texW = cfg.texW, texH = cfg.texH;
	let ke = 0, peakVort = 0, enstrophy = 0, peakSpeed = 0, peakDiv = 0;
	for (let y = 1; y < texH - 1; y++) {
		for (let x = 1; x < texW - 1; x++) {
			const i = x + y * texW, vi = i * 2;
			const uL = vel[vi], uR = vel[vi + 2], vB = vel[vi + 1], vT = vel[vi + texW * 2 + 1];
			const uc = 0.5 * (uL + uR), vc = 0.5 * (vB + vT);
			ke += 0.5 * mass[i * 4 + 3] * (uc * uc + vc * vc);
			// the solver's curl, not a second copy of it: t5/t6 first shipped
			// with (vT - vB) - (uR - uL) in both places, which is dv/dy-du/dx
			// (0 for solid-body rotation), so force and oracle agreed on a
			// quantity that was never vorticity. curlFixedPoints below pins
			// this one definition to analytic values instead.
			const w = sim.vorticityAt(x, y);
			enstrophy += w * w;
			const aw = Math.abs(w);
			if (aw > peakVort) peakVort = aw;
			const s = Math.max(Math.abs(uL), Math.abs(uR), Math.abs(vB), Math.abs(vT));
			if (s > peakSpeed) peakSpeed = s;
			const d = Math.abs(uR - uL + vT - vB) / DT;
			if (d > peakDiv) peakDiv = d;
		}
	}
	let errL2 = 0;
	if (vel0) {
		let num = 0, den = 0;
		for (let i = 0; i < vel.length; i++) {
			const d = vel[i] - vel0[i];
			num += d * d;
			den += vel0[i] * vel0[i];
		}
		errL2 = den > 0 ? Math.sqrt(num / den) : 0;
	}
	out.t = cfg.time;
	out.ke = ke;
	out.peakVort = peakVort;
	out.enstrophy = enstrophy;
	out.peakSpeed = peakSpeed;
	out.peakDiv = peakDiv;
	out.errL2 = errL2;
}
function run(name, N, T, proj, over) {
	const { cfg, sim } = make(N, over);
	sim.reset();
	paintVortex(sim, cfg);
	const vel0 = Float32Array.from(grid.curBuf(sim.vel));
	const iters = Fluid.core.config.solverIters(cfg);
	const steps = Math.round(T / DT);
	const perSec = Math.round(1 / DT);
	const m = { t: 0, ke: 0, peakVort: 0, enstrophy: 0, peakSpeed: 0, peakDiv: 0, errL2: 0 };
	measure(sim, cfg, m, vel0);
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
		measure(sim, cfg, m, vel0);
		if (m.peakSpeed > peakSpeedMax) peakSpeedMax = m.peakSpeed;
		rows.push(Object.assign({}, m));
	}
	const ms = (Date.now() - t0) / steps;
	const advMs = Number(advNs) / 1e6 / steps;
	const forceMs = Number(forceNs) / 1e6 / steps;
	// the raw row exists to show what the limiter prevents, so its overshoot is a
	// result, not a failure. every scheme the product could ship must hold 0.
	// confinement rows are exempt from the extremum assert: the force is an
	// energy source, so a spun-up core can pass its initial rim speed, and
	// whether that eps is over-driven is the decision block's call (the old
	// condition read as exempt in the t6 worklog but never was in code).
	if ((cfg.macLimiter || !cfg.advectMac) && !cfg.vorticityConfinement) {
		assert.equal(clampTotal, 0, name + ": vClamp watchdog must stay 0, got " + clampTotal);
		assert.ok(peakSpeedMax <= r0Peak * (1 + 1e-6),
			name + ": the limited scheme must not create a new extremum, " + peakSpeedMax + " vs " + r0Peak);
	}
	// what confinement may not do is push the velocity into the absolute
	// clamp: that is the blowup signal.
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
			" errL2=" + row.errL2.toFixed(4) +
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
// the curl has analytic values on fields with no scheme in them, and those are
// the assertions that would have caught the t5/t6 formula: solid-body rotation
// has curl 2*Omega everywhere, pure shear u=k*y has curl -k, and the gaussian
// vortex peaks at 2*A/sigma in its core. the wrong stencil read 0, 0 and a
// diagonal lobe at 37 % of the core value.
function curlFixedPoints(N) {
	const { cfg, sim } = make(N, {});
	sim.reset();
	const texW = cfg.texW, texH = cfg.texH, cx = N * 0.5, cy = N * 0.5;
	const mid = Math.floor(texW / 2);
	const paintLinear = (fu, fv) => {
		const vel = grid.curBuf(sim.vel);
		vel.fill(0);
		for (let y = 1; y < texH - 1; y++) {
			for (let x = 1; x < texW - 1; x++) {
				const vi = (x + y * texW) * 2;
				if (!sim.uBlocked(x, y)) vel[vi] = fu((x - 1) - cx, (y - 0.5) - cy);
				if (!sim.vBlocked(x, y)) vel[vi + 1] = fv((x - 0.5) - cx, (y - 1) - cy);
			}
		}
	};
	const OMEGA = 0.1, SHEAR = 0.3;
	paintLinear((dx, dy) => -OMEGA * dy, dx => OMEGA * dx);
	const solid = sim.vorticityAt(mid, mid);
	paintLinear((dx, dy) => SHEAR * dy, () => 0);
	const shear = sim.vorticityAt(mid, mid);
	paintLinear(() => 37, () => -11);
	const uniform = sim.vorticityAt(mid, mid);
	paintVortex(sim, cfg);
	const m = { t: 0, ke: 0, peakVort: 0, enstrophy: 0, peakSpeed: 0, peakDiv: 0 };
	measure(sim, cfg, m);
	const analytic = 2 * (PEAK / Math.exp(-0.5)) / (N * SIGMA_FRAC);
	console.log("  curl: solid-body " + solid.toFixed(4) + " (exact " + (2 * OMEGA).toFixed(4) +
		") · shear " + shear.toFixed(4) + " (exact " + (-SHEAR).toFixed(4) +
		") · uniform " + uniform.toExponential(2) + " (exact 0)" +
		" · gaussian peak " + m.peakVort.toFixed(3) + " (analytic " + analytic.toFixed(3) + ")");
	assert.ok(Math.abs(solid - 2 * OMEGA) < 1e-3, "curl of solid-body rotation must be 2*Omega, got " + solid);
	assert.ok(Math.abs(shear + SHEAR) < 1e-3, "curl of pure shear u=k*y must be -k, got " + shear);
	assert.ok(uniform === 0, "curl of a uniform field must be 0, got " + uniform);
	assert.ok(Math.abs(m.peakVort / analytic - 1) < 0.02,
		"measured peak vorticity must match the analytic core value, " + m.peakVort + " vs " + analytic);
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
	console.log("  the 15 % bar is a device number, so neither cpu number is the gate." +
		" on Ampere pinned ns=4 (`2026-10-23-04_53`), wgsl mac-cormack adds +16..19 us/substep" +
		" (-0.6 %..+0.6 % fps); default flip awaits the auto-policy row (`run_gpu_mac_k20.py`).");
}
// 0.6.0 t6/t17: confinement rows are read against their host advection scheme
// (sl for t6, mac for t17). the energy row is not a ranking here (the force adds
// energy on purpose); the ranking is vorticity/enstrophy/errL2 against bounded
// divergence, peak speed <= initial rim speed, and macLimit activity.
function confinementDecision(base, baseName, rows, T) {
	if (!rows.length) return;
	console.log("== confinement decision on " + baseName + " ==");
	for (const r of rows) {
		const a = at(base, T), b = at(r, T);
		const vort = a.peakVort > 0 ? b.peakVort / a.peakVort : 0;
		const enst = a.enstrophy > 0 ? b.enstrophy / a.enstrophy : 0;
		const ke = a.ke > 0 ? b.ke / a.ke : 0;
		const limNote = r.limitTotal > 0 || base.limitTotal > 0
			? ", macLimit " + base.limitTotal + " -> " + r.limitTotal
			: "";
		console.log("  eps=" + r.eps + " 1/s at t=" + T + "s: peak vorticity " + a.peakVort.toFixed(3) +
			" -> " + b.peakVort.toFixed(3) + " (" + vort.toFixed(3) + "x " + baseName + "), enstrophy " +
			enst.toFixed(3) + "x, KE " + ke.toFixed(3) + "x, errL2 " +
			a.errL2.toFixed(4) + " -> " + b.errL2.toFixed(4) + ", maxDiv " +
			a.peakDiv.toExponential(2) + " -> " + b.peakDiv.toExponential(2) +
			", max speed " + r.peakSpeedMax.toFixed(2) + " (initial " +
			base.rows[0].peakSpeed.toFixed(2) + "), velClamp " + r.velClamp + limNote + ".");
	}
}

const args = process.argv.slice(2);
function opt(name, fallback) {
	const i = args.indexOf(name);
	return i >= 0 ? Number(args[i + 1]) : fallback;
}
function parseEpsArg(flag, fallback) {
	const i = args.indexOf(flag);
	if (i < 0) return fallback;
	const raw = args[i + 1] || "";
	if (!raw || raw === "none") return [];
	return raw.split(",").map(Number);
}
const N = opt("--n", 128);
const T = opt("--t", 4);
// projection is the product path and the default. without it the momentum
// equation is inviscid Burgers: the vortex steepens instead of decaying, peak
// vorticity rises and maxDiv reaches 1e4, so that row attributes diffusion but
// is not a candidate anyone could ship.
const proj = !args.includes("--no-proj");
// the confinement strengths to sweep, in 1/s. `--eps 0,1,4` or `--eps none`
// overrides the sl sweep; `--mac-eps 0.25,0.5,1,2` sweeps confinement on mac (t17).
const epsList = parseEpsArg("--eps", CONF_EPS);
const macEpsList = parseEpsArg("--mac-eps", []);

console.log("vortex-decay · N=" + N + " sigma=" + (N * SIGMA_FRAC).toFixed(1) +
	" rim=" + PEAK + " cells/s · dt=" + DT + " · " + Math.round(T / DT) + " steps (" + T + " s)");
console.log("== fixed points ==");
constantField(N);
curlFixedPoints(N);
console.log(proj ? "== with the shipping projection (mg 2 cycles, K20) =="
	: "== no projection (Burgers: steepens, attribution row only) ==");
const runs = SCHEMES.map(s => run(s.name, N, T, proj, s.over));
for (const r of runs) printRun(r);
if (proj) {
	// the confinement rows are a force on the host scheme, so they are read
	// on the product path. without the projection the momentum equation is
	// inviscid Burgers and the force has nothing to work against: that row
	// belongs to the advection schemes, not to the force.
	const confRuns = epsList.map(eps => run("sl+conf" + eps, N, T, proj,
		{ advectMac: 0, vorticityConfinement: eps }));
	for (const r of confRuns) printRun(r);
	const macConfRuns = macEpsList.map(eps => run("mac+conf" + eps, N, T, proj,
		{ advectMac: 1, macLimiter: 1, vorticityConfinement: eps }));
	for (const r of macConfRuns) printRun(r);
	decision(runs[0], runs[1], T);
	confinementDecision(runs[0], "sl", confRuns, T);
	confinementDecision(runs[1], "mac", macConfRuns, T);
} else console.log("== no decision: the bar is read on the projection row (`node experiments/vortex-decay.js`) ==");
console.log("done");
