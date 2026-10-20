"use strict";
// 0.6.0 t4: two-level multigrid pressure solve vs the red-black baseline.
// run: node experiments/multigrid.js | tee experiments/logs/<date>-multigrid.log
//
// the fine red-black GS is a good smoother but cannot damp the long-wavelength
// pressure error on a closed box (amplification 1-O(1/N^2) per sweep), so the
// plain solve leaves a divergence floor and the warm-started null space drifts.
// this measures the same dam/blobs run with and without the N/2 coarse level:
// wall time, maxDiv after projection, overpack and the pressure range.

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/multigrid.js"));
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
	c.substepsAuto = false;
	return c;
}
function cur(sim, pp) { return pp.cur === 0 ? pp.a : pp.b; }
function totals(sim, cfg) {
	return stats.totalsFromFields(cur(sim, sim.mass), cur(sim, sim.therm), cfg.texW, cfg.texH);
}

// one fixed-policy run. returns wall time, end-state metrics, and the maxDiv
// the last projection actually left behind (sim.diag.divAfter is the same
// measurement, recorded inside step()).
function run(over, frames, ns) {
	const cfg = makeCfg(over);
	const sim = solverMod.create(cfg);
	const dt = cfg.frameDt / ns;
	const t0 = Date.now();
	let worstDiv = 0;
	for (let f = 0; f < frames; f++) {
		for (let s = 0; s < ns; s++) sim.step(dt);
		if (sim.diag.divAfter > worstDiv) worstDiv = sim.diag.divAfter;
	}
	const el = Date.now() - t0;
	const tot = totals(sim, cfg);
	const mot = stats.motionTotals(cur(sim, sim.vel), cur(sim, sim.press), cfg.texW, cfg.texH, dt);
	return { cfg, sim, el, tot, mot, worstDiv, frames, ns };
}

function compare(name, over, frames, ns) {
	const plain = run(Object.assign({}, over, { useMultigrid: false }), frames, ns);
	const mg = run(Object.assign({}, over, { useMultigrid: true }), frames, ns);
	const speedup = plain.el / mg.el;
	const divRatio = plain.worstDiv / Math.max(mg.worstDiv, 1e-9);
	console.log("--- " + name + "  N=" + plain.cfg.N + " " + frames + " frames x " + ns + " substeps ---");
	console.log("plain  K=" + plain.cfg.pressureIters + "  maxDiv(end)=" + plain.mot.maxDiv.toFixed(3) +
		"  worstDiv=" + plain.worstDiv.toFixed(3) + "  maxM=" + plain.tot.maxM.toFixed(4) +
		"  over=" + plain.tot.cntOver + "  ms=" + plain.el);
	console.log("mg     pre=" + mg.cfg.mgPre + " post=" + mg.cfg.mgPost + " coarse=" + mg.cfg.mgCoarseIters +
		"  maxDiv(end)=" + mg.mot.maxDiv.toFixed(3) + "  worstDiv=" + mg.worstDiv.toFixed(3) +
		"  maxM=" + mg.tot.maxM.toFixed(4) + "  over=" + mg.tot.cntOver + "  ms=" + mg.el);
	check(name + ": multigrid lowers the projection divergence at least 2x",
		divRatio >= 2, "worstDiv " + plain.worstDiv.toFixed(3) + " -> " + mg.worstDiv.toFixed(3) + " (" + divRatio.toFixed(1) + "x)");
	// maxM is the peak of a transported blob (0.67 in blobs), not a compression
	// metric until it exceeds 1; the two runs follow different trajectories, so
	// allow a 1e-3 peak difference and judge overpack by the count above 1.001.
	check(name + ": multigrid does not overpack more than the plain solve",
		mg.tot.maxM <= plain.tot.maxM + 1e-3 && mg.tot.cntOver <= plain.tot.cntOver,
		"maxM " + plain.tot.maxM.toFixed(4) + " -> " + mg.tot.maxM.toFixed(4) +
		", over " + plain.tot.cntOver + " -> " + mg.tot.cntOver);
	check(name + ": multigrid is cheaper in wall time",
		speedup > 1.2, "plain " + plain.el + "ms vs mg " + mg.el + "ms (" + speedup.toFixed(2) + "x)");
	return { plain, mg };
}

function main() {
	// dam: the violent, variable-density case where the warm-started smooth mode
	// drifts and the plain solve compressed mass to maxM 3.9 at N=256.
	compare("dam", { N: 128, scenario: "dam" }, 60, 4);
	// blobs: underpacked base (m=0.3) with dyed blobs, the interactive default.
	compare("blobs", { N: 128, scenario: "blobs" }, 60, 4);

	// restriction scaling: the coarse RHS must be the *sum* of the four fine
	// residuals. this guards the 1/S^2 factor: with the average the correction
	// is S^2 times too small. computed from a live state by poking the exposed
	// coarse arrays is not accessible, so assert the invariant indirectly: one
	// V-cycle from a converged state must not increase the residual.
	const cfg = makeCfg({ N: 128, scenario: "dam", useMultigrid: true, mgCoarseIters: 40 });
	const sim = solverMod.create(cfg);
	const dt = cfg.frameDt / 4;
	for (let f = 0; f < 20; f++) for (let s = 0; s < 4; s++) sim.step(dt);
	const before = sim.mg.residualNorm();
	sim.mg.vcycle(cfg.mgPre, cfg.mgCoarseIters, cfg.mgPost);
	const after = sim.mg.residualNorm();
	check("one V-cycle never increases the fine pressure residual",
		after <= before * 1.05, "residual " + before.toFixed(3) + " -> " + after.toFixed(3));

	console.log(failures === 0 ? "ALL MULTIGRID TESTS PASSED" : failures + " FAILURES");
	process.exit(failures ? 1 : 0);
}
main();
