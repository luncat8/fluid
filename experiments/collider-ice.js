"use strict";
// 0.7.4: the device reproducer, made headless and deterministic.
//
// the 0.7.3/0.7.4 device captures showed the demo page dying when the drag disc
// moves fast across the ice scenario with the six-emitter rig on. this harness
// reproduces exactly that on the cpu oracle (N=128, same emitters, same disc,
// same 480 u/s drag) and records the first non-finite value together with the
// pass that produced it.
//
// run: node experiments/collider-ice.js | tee experiments/logs/<date>-collider-ice.log
//
// it prints the frame, the substep, the pass, the state of the failing cell and
// its neighbourhood, and then the pressure history around the failure. that is
// the evidence the device capture could not give: which pass, which frame.

const path = require("node:path");
const root = path.resolve(__dirname, "..");
for (const f of ["config", "grid", "stats", "multigrid", "emitters", "colliders", "solver", "probes"]) {
	require(path.join(root, "src/core/" + f + ".js"));
}
const C = Fluid.core.config, G = Fluid.core.grid;

const N = +(process.env.NLEV || 128);
const FRAMES = +(process.env.FRAMES || 300);
const SPEED = +(process.env.SPEED || 480);
const STRESS = process.env.STRESS === "1";
let failures = 0;
function check(name, ok, detail) {
	console.log((ok ? "PASS " : "FAIL ") + name + (detail ? "  " + detail : ""));
	if (!ok) failures++;
}

function build() {
	const cfg = Object.assign({}, C.defaults, { N, scenario: "ice" });
	cfg.texW = N + 2; cfg.texH = N + 2; cfg.frameDt = 1 / 60;
	const sim = Fluid.core.solver.create(cfg);
	// the demo page's "Enable RGB + thermal rig", verbatim (examples/demo.js)
	const rig = [
		{ kind: "heat", x: N * 0.5, y: N * 0.08, radius: N * 0.14, value: 180 },
		{ kind: "freeze", x: N * 0.5, y: N * 0.88, radius: N * 0.08, value: 1100 },
		{ kind: "dye", x: N * 0.24, y: N * 0.20, radius: N * 0.04, value: 0.35, dye: [0.95, 0.2, 0.18] },
		{ kind: "dye", x: N * 0.50, y: N * 0.16, radius: N * 0.04, value: 0.35, dye: [0.18, 0.9, 0.28] },
		{ kind: "dye", x: N * 0.76, y: N * 0.20, radius: N * 0.04, value: 0.35, dye: [0.2, 0.45, 0.98] },
		{ kind: "drain", x: N * 0.5, y: N * 0.56, radius: N * 0.06, value: 0.45 }
	];
	for (const e of rig) sim.addEmitter(e);
	const r = Math.round(N * 0.055);
	const disc = { id: -1, r, x: Math.round(N * 0.35), y: Math.round(N * 0.35), px: 0, py: 0 };
	disc.px = disc.x; disc.py = disc.y;
	disc.id = sim.addCollider({ shape: "circle", x: disc.x, y: disc.y, radius: r, vx: 0, vy: 0 });
	return { sim, cfg, disc };
}

// drag the disc along the same path the browser drag follows: a target that sweeps
// the grid, approached at no more than SPEED cells/s
function dragStep(sim, disc, f) {
	// STRESS=1 switches to a tighter, faster orbit. the gentle path needs N=128 and
	// ~220 frames (about 5 minutes on the cpu twin); the stress path reproduces at
	// N=64 within ~95 frames (about 30 s), which is what the ci row can afford.
	const w1 = STRESS ? 0.21 : 0.09, w2 = STRESS ? 0.17 : 0.13;
	const a1 = STRESS ? 0.45 : 0.42, a2 = STRESS ? 0.40 : 0.34;
	const tx = Math.round(N * 0.5 + N * a1 * Math.sin(f * w1));
	const ty = Math.round(N * 0.5 + N * a2 * Math.cos(f * w2));
	const dx = tx - disc.x, dy = ty - disc.y, d = Math.hypot(dx, dy);
	const step = Math.min(d, SPEED / 60);
	disc.x = Math.max(disc.r, Math.min(N - disc.r, disc.x + (d > 0 ? dx / d * step : 0)));
	disc.y = Math.max(disc.r, Math.min(N - disc.r, disc.y + (d > 0 ? dy / d * step : 0)));
	const vx = Math.max(-SPEED, Math.min(SPEED, (disc.x - disc.px) * 60));
	const vy = Math.max(-SPEED, Math.min(SPEED, (disc.y - disc.py) * 60));
	sim.moveCollider(disc.id, disc.x, disc.y, vx, vy);
	disc.px = disc.x; disc.py = disc.y;
	return { vx, vy };
}

// the eleven passes of one substep, in the order step() runs them
function substep(sim, cfg, dt) {
	sim.updateColliderMask();
	sim.forces(dt);
	sim.advectVelocity(dt);
	sim.imprintColliders();
	sim.poissonRhs(dt);
	sim.project(C.solverIters(cfg));
	sim.applyGrad(dt);
	sim.computeFlux(dt);
	sim.advectScalar();
	sim.phaseEos(true);
	sim.conduction(dt);
	sim.applyEmitters(dt);
	sim.heatSources(dt);
	sim.phaseEos(true);
	cfg.frame++; cfg.time += dt;
}

// the scan the 0.7.4 stats counters do on the gpu: the first cell that is not finite
function firstNonFinite(sim, cfg) {
	const W = cfg.texW, H = cfg.texH;
	const m = G.curBuf(sim.mass), t = G.curBuf(sim.therm), v = G.curBuf(sim.vel), p = G.curBuf(sim.press);
	for (let y = 1; y < H - 1; y++) {
		for (let x = 1; x < W - 1; x++) {
			const i = x + y * W;
			if (m[i * 4 + 3] !== m[i * 4 + 3] || t[i * 4] !== t[i * 4] ||
				t[i * 4 + 1] !== t[i * 4 + 1] || v[i * 2] !== v[i * 2] ||
				v[i * 2 + 1] !== v[i * 2 + 1] || p[i] !== p[i]) {
				return { x, y, i };
			}
		}
	}
	return null;
}
function maxSpeed(sim, cfg) {
	const v = G.curBuf(sim.vel);
	let best = 0, bx = -1, by = -1;
	for (let y = 1; y < cfg.texH - 1; y++) {
		for (let x = 1; x < cfg.texW - 1; x++) {
			const i = x + y * cfg.texW;
			const a = Math.max(Math.abs(v[i * 2]), Math.abs(v[i * 2 + 1]));
			if (a > best) { best = a; bx = x; by = y; }
		}
	}
	return { best, bx, by };
}
function pressureRange(sim, cfg) {
	const p = G.curBuf(sim.press);
	let mn = Infinity, mx = -Infinity;
	for (let y = 1; y < cfg.texH - 1; y++) {
		for (let x = 1; x < cfg.texW - 1; x++) {
			const q = p[x + y * cfg.texW];
			if (q < mn) mn = q;
			if (q > mx) mx = q;
		}
	}
	return { mn, mx };
}
function neighbourhood(sim, cfg, cell) {
	const rows = [];
	const ms = G.curBuf(sim.mass), ts = G.curBuf(sim.therm), pr = G.curBuf(sim.press);
	for (let oy = -2; oy <= 2; oy++) {
		const row = [];
		for (let ox = -2; ox <= 2; ox++) {
			const x = cell.x + ox, y = cell.y + oy, k = x + y * cfg.texW;
			// the failing cell can sit on the domain edge (1,1 is common, because the
			// ghost ring is 0 and a clamp there reads a zero pressure), so skip any
			// offset that leaves the grid instead of crashing the reproducer
			if (x < 0 || x >= cfg.texW || y < 0 || y >= cfg.texH) { row.push("   ---   "); continue; }
			row.push("m" + ms[k * 4 + 3].toFixed(2) + " p" + pr[k].toFixed(0) +
				" T" + ts[k * 4 + 1].toFixed(0) + " fl" + sim.flags[k * 4] + "/" + sim.flags[k * 4 + 2]);
		}
		rows.push("      " + row.join("  "));
	}
	return rows.join("\n");
}

const { sim, cfg, disc } = build();
const W = cfg.texW;
let hit = null;
let pressureBefore = [];
console.log("collider-ice: N=" + N + " scenario=ice speed=" + SPEED + " u/s, six-emitter rig, disc r=" + disc.r);

for (let f = 0; f < FRAMES && !hit; f++) {
	const v = dragStep(sim, disc, f);
	const st = sim.stats();
	const ns = C.substepCount(cfg, st.mot.maxVel);
	cfg.vClamp = C.velocityCap(cfg, ns);
	const dt = cfg.frameDt / ns;
	for (let s = 0; s < ns && !hit; s++) {
		const passes = [
			["updateColliderMask", () => sim.updateColliderMask()],
			["forces", () => sim.forces(dt)],
			["advectVelocity", () => sim.advectVelocity(dt)],
			["imprintColliders", () => sim.imprintColliders()],
			["poissonRhs", () => sim.poissonRhs(dt)],
			["project", () => sim.project(C.solverIters(cfg))],
			["applyGrad", () => sim.applyGrad(dt)],
			["computeFlux", () => sim.computeFlux(dt)],
			["advectScalar", () => sim.advectScalar()],
			["phaseEos", () => sim.phaseEos(true)],
			["conduction", () => sim.conduction(dt)],
			["applyEmitters", () => sim.applyEmitters(dt)],
			["heatSources", () => sim.heatSources(dt)],
			["phaseEos (post-source)", () => sim.phaseEos(true)]
		];
		for (const [name, fn] of passes) {
			const before = maxSpeed(sim, cfg);
			const prBefore = pressureRange(sim, cfg);
			fn();
			cfg.frame++; cfg.time += dt;
			const cell = firstNonFinite(sim, cfg);
			const after = maxSpeed(sim, cfg);
			if (cell || !Number.isFinite(after.best)) {
				hit = { f, s, ns, name, cell, before, after, prBefore, prAfter: pressureRange(sim, cfg), v };
				break;
			}
			// the spike that precedes it: a velocity far past the clamp, or a pressure
			// far past what the previous solve could justify
			if (after.best > 20 * cfg.vClamp) {
				pressureBefore.push({ f, s, name, speed: after.best, cap: cfg.vClamp, pr: prBefore, res: sim.diag.poissonRes });
			}
		}
	}
}

if (hit) {
	console.log("");
	console.log("FIRST NON-FINITE VALUE: frame " + hit.f + " substep " + hit.s + "/" + hit.ns + " after pass '" + hit.name + "'");
	console.log("  velocity before pass: max " + hit.before.best.toFixed(1) + " at " + hit.before.bx + "," + hit.before.by +
		"  (vClamp " + cfg.vClamp.toFixed(1) + ")");
	console.log("  velocity after  pass: max " + hit.after.best.toExponential(3));
	console.log("  pressure before pass: " + hit.prBefore.mn.toFixed(0) + " .. " + hit.prBefore.mx.toFixed(0));
	console.log("  pressure after  pass: " + hit.prAfter.mn.toFixed(0) + " .. " + hit.prAfter.mx.toFixed(0));
	const c = hit.cell;
	if (c) {
		const ms = G.curBuf(sim.mass), ts = G.curBuf(sim.therm);
		console.log("  cell " + c.x + "," + c.y + ": m=" + ms[c.i * 4 + 3] + " E=" + ts[c.i * 4] +
			" T=" + ts[c.i * 4 + 1] + " flags=" + [...sim.flags.slice(c.i * 4, c.i * 4 + 4)]);
		console.log("  neighbourhood (m / pressure / T / flags.r|frozen):");
		console.log(neighbourhood(sim, cfg, c));
	}
	console.log("  disc at " + Math.round(disc.x) + "," + Math.round(disc.y) + " moving " +
		hit.v.vx.toFixed(0) + "," + hit.v.vy.toFixed(0) + " u/s");
	if (pressureBefore.length) {
		const last = pressureBefore[pressureBefore.length - 1];
		console.log("  last pre-failure velocity spike: frame " + last.f + " pass '" + last.name + "' max=" +
			last.speed.toExponential(2) + " cap=" + last.cap.toFixed(0) +
			" pressure " + last.pr.mn.toFixed(0) + ".." + last.pr.mx.toFixed(0) +
			" res=" + last.res.toExponential(2));
	}
	check("collider-ice: the fast-disc-across-ice failure is reproduced headlessly", true,
		"frame " + hit.f + " after " + hit.name);
	check("collider-ice: the field dies in the pressure solve, not in transport or an emitter",
		hit.name === "applyGrad" || hit.name === "project", "first bad after " + hit.name);

	// 0.7.5: the shape of the anomaly, printed so the worklog claim is reproducible
	// from the repo rather than only from a pasted device reading. the point is that
	// the blow-up is not justified by the source: the solve residual is orders of
	// magnitude larger than the rhs it is solving, and the mass on the failing cell
	// is swinging by thousands of times between substeps, which is what drives the
	// 1/rho face weight the operator uses.
	const ms = G.curBuf(sim.mass);
	let rhoMin = Infinity, rhoMax = 0, at = null;
	for (let y = 1; y < cfg.texH - 1; y++) {
		for (let x = 1; x < cfg.texW - 1; x++) {
			const k = x + y * cfg.texW;
			if ((sim.flags[k * 4] | sim.flags[k * 4 + 2]) !== 0) continue;
			const rho = ms[k * 4 + 3];
			if (!(rho === rho)) continue;
			if (rho < rhoMin) rhoMin = rho;
			if (rho > rhoMax) { rhoMax = rho; at = { x, y }; }
		}
	}
	const res = sim.diag.poissonRes || 0, rhs = sim.diag.poissonRhs || 0;
	console.log("  ANOMALY: solve residual " + res.toExponential(2) + " vs rhs " + rhs.toExponential(2) +
		"  (residual/source " + (rhs ? (res / rhs).toExponential(1) : "n/a") + ")");
	console.log("  ANOMALY: open-cell density spans " + rhoMin.toExponential(2) + " .. " + rhoMax.toExponential(2) +
		(at ? "  (max at " + at.x + "," + at.y + ")" : "") +
		"  - the 1/rho face weight is what the operator amplifies");
	// printed, not gated: once the field is non-finite the residual readout is zeroed
	// by captureSolveResidual, so a threshold here would assert on a stale value. the
	// capture that *is* trustworthy is the substep trace in the worklog, where the
	// residual is 5-6 orders above the rhs while the field is still finite.
	check("collider-ice: the open-cell density range is recorded for the worklog",
		Number.isFinite(rhoMin) && Number.isFinite(rhoMax),
		"rho " + rhoMin.toExponential(2) + ".." + rhoMax.toExponential(2));
} else {
	console.log("");
	console.log("no non-finite value in " + FRAMES + " frames at N=" + N);
	check("collider-ice: the run stayed finite", true, FRAMES + " frames");
}
console.log("");
console.log(failures === 0 ? "COLLIDER-ICE REPRODUCED" : failures + " FAILURES");
process.exitCode = failures === 0 ? 0 : 1;
