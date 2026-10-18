"use strict";

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));

function makeCfg(N, fctLimiter) {
	const cfg = Object.assign({}, Fluid.core.config.defaults, {
		N,
		texW: N + 2,
		texH: N + 2,
		scenario: "still",
		fctLimiter,
		dt: 1
	});
	return cfg;
}

function measureFrontWidth(fctLimiter, N, steps) {
	const cfg = makeCfg(N, fctLimiter);
	const sim = solverMod.create(cfg);
	const mass = sim.mass.a;
	const therm = sim.therm.a;
	const front = Math.floor(N * 0.2);
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		const i = (x + y * cfg.texW) * 4;
		mass[i] = x >= front ? 0.8 : 0;
		mass[i + 1] = 0;
		mass[i + 2] = 0;
		mass[i + 3] = 1;
		therm[i] = 293;
	}
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 2; x <= N; x++) {
		sim.flux[(x + y * cfg.texW) * 2] = 0.5;
	}
	for (let step = 0; step < steps; step++) sim.advectScalar();
	const current = sim.mass.cur === 0 ? sim.mass.a : sim.mass.b;
	const low = sim.mass.cur === 0 ? sim.mass.b : sim.mass.a;
	let width = 0, min = Infinity, max = -Infinity, totalMass = 0, totalDye = 0, localViolations = 0;
	const y = Math.floor(N / 2);
	const center = front + Math.floor(steps / 2);
	for (let x = Math.max(1, center - 30); x <= Math.min(N, center + 30); x++) {
		const i = (x + y * cfg.texW) * 4;
		const dye = current[i];
		if (dye > 0.04 && dye < 0.76) width++;
		if (dye < min) min = dye;
		if (dye > max) max = dye;
	}
	for (let row = 1; row < cfg.texH - 1; row++) for (let x = 1; x < cfg.texW - 1; x++) {
		const i = (x + row * cfg.texW) * 4;
		totalMass += current[i + 3];
		totalDye += current[i];
		if (!fctLimiter) continue;
		let qMin = Infinity, qMax = -Infinity;
		for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
			const q = low[(x + ox + (row + oy) * cfg.texW) * 4];
			if (q < qMin) qMin = q;
			if (q > qMax) qMax = q;
		}
		if (current[i] < qMin - 1e-6 || current[i] > qMax + 1e-6) localViolations++;
	}
	return { width, min, max, totalMass, totalDye, localViolations };
}

function measureRandomBounds(N) {
	const cfg = makeCfg(N, 1);
	const sim = solverMod.create(cfg);
	let seed = 0x12345678;
	function random() {
		seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
		return seed / 4294967296;
	}
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		const i = (x + y * cfg.texW) * 4;
		sim.mass.a[i] = random() * 0.8;
		sim.mass.a[i + 1] = random() * 0.8;
		sim.mass.a[i + 2] = random() * 0.8;
		sim.mass.a[i + 3] = 1;
		sim.therm.a[i] = 293;
	}
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 2; x <= N; x++) sim.flux[(x + y * cfg.texW) * 2] = 0.2;
	for (let y = 2; y <= N; y++) for (let x = 1; x < cfg.texW - 1; x++) sim.flux[(x + y * cfg.texW) * 2 + 1] = -0.15;
	const source = sim.mass.a;
	let initialDye = 0;
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		initialDye += source[(x + y * cfg.texW) * 4] + source[(x + y * cfg.texW) * 4 + 1] + source[(x + y * cfg.texW) * 4 + 2];
	}
	sim.advectScalar();
	const final = sim.mass.cur === 0 ? sim.mass.a : sim.mass.b;
	const low = sim.mass.cur === 0 ? sim.mass.b : sim.mass.a;
	let violations = 0, min = Infinity, lowDye = 0, finalDye = 0;
	for (let y = 1; y < cfg.texH - 1; y++) for (let x = 1; x < cfg.texW - 1; x++) {
		const i = (x + y * cfg.texW) * 4;
		for (let c = 0; c < 3; c++) {
			const q = final[i + c];
			let qMin = Infinity, qMax = -Infinity;
			if (q < min) min = q;
			for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
				const qN = low[(x + ox + (y + oy) * cfg.texW) * 4 + c];
				if (qN < qMin) qMin = qN;
				if (qN > qMax) qMax = qN;
			}
			if (q < qMin - 1e-6 || q > qMax + 1e-6) violations++;
			lowDye += low[i + c];
			finalDye += q;
		}
	}
	return { violations, min, lowDye, finalDye, initialDye };
}

function compare(N, steps) {
	const upwind = measureFrontWidth(0, N, steps);
	const fct = measureFrontWidth(1, N, steps);
	return { upwind, fct, ratio: upwind.width / fct.width, random: measureRandomBounds(64) };
}

if (require.main === module) {
	const N = 256, steps = 240;
	const result = compare(N, steps);
	console.log("sharpness: dye front, N=" + N + " steps=" + steps + " flux=0.5");
	console.log("upwind_width=" + result.upwind.width + " fct_width=" + result.fct.width +
		" improvement=" + result.ratio.toFixed(2) + "x");
	console.log("fct_front_range=" + result.fct.min.toExponential(3) + ".." + result.fct.max.toFixed(6) +
		" total_dye=" + result.fct.totalDye.toFixed(6) + " total_mass=" + result.fct.totalMass.toFixed(6) +
		" local_bound_violations=" + result.fct.localViolations);
	const dyeDrift = Math.abs(result.fct.totalDye - result.upwind.totalDye) / result.upwind.totalDye;
	console.log("fct_vs_upwind_dye_drift=" + dyeDrift.toExponential(3));
	const r = result.random;
	const randomDrift = Math.abs(r.finalDye - r.lowDye) / r.initialDye;
	console.log("random_2d: min=" + r.min.toExponential(3) + " local_bound_violations=" + r.violations +
		" correction_drift=" + randomDrift.toExponential(3));
	const pass = result.ratio >= 3 && result.fct.min >= -1e-8 && result.fct.max <= 0.800001 &&
		result.fct.localViolations === 0 && dyeDrift < 1e-6 && r.min >= -1e-8 && r.violations === 0 && randomDrift < 1e-6;
	console.log(pass ? "PASS" : "FAIL");
	process.exitCode = pass ? 0 : 1;
}

module.exports = { makeCfg, measureFrontWidth, compare };
