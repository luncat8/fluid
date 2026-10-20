"use strict";
// 0.6.0 t1 contract: profiler aggregation, ring reuse, generation guard and
// capacity guard on a fake device whose timestamp buffer is scripted.
// run: node experiments/profiler.js
const path = require("node:path");
globalThis.GPUBufferUsage = { QUERY_RESOLVE: 1, COPY_SRC: 2, MAP_READ: 4, COPY_DST: 8 };
globalThis.GPUMapMode = { READ: 1 };
const profilerMod = require(path.join(__dirname, "..", "src/gpu/profiler.js"));

let failures = 0;
function check(name, ok, detail) {
	console.log((ok ? "PASS " : "FAIL ") + name + (detail ? "  " + detail : ""));
	if (!ok) failures++;
}
// scripted timestamps: pair p measures (durationsNs[p]) ns; buffer filled at map time
let script = [];
function makeBuffer() {
	return {
		async mapAsync() {},
		getMappedRange() {
			const arr = new BigUint64Array(256);
			let t = 1000000n;
			for (let p = 0; p < script.length; p++) { arr[2 * p] = t; t += BigInt(script[p]); arr[2 * p + 1] = t; }
			return arr.buffer;
		},
		unmap() {}, destroy() {}
	};
}
const device = {
	features: { has: f => f === "timestamp-query" },
	limits: { timestampPeriod: 1 },
	createQuerySet: () => ({ destroy() {} }),
	createBuffer: () => makeBuffer()
};
const encoder = {
	beginComputePass: () => ({ end() {} }),
	resolveQuerySet() {}, copyBufferToBuffer() {}
};

(async () => {
	const prof = profilerMod.create(device);
	check("create with feature", !!prof);
	const G = profilerMod.GROUP;

	// frame 1: two substeps of 5 groups + stats + disp + render = 13 pairs
	// group durations per substep: vel 100us rhs 10us press 900us trans 200us phase 150us
	script = [];
	for (let s = 0; s < 2; s++) script.push(100000, 10000, 900000, 200000, 150000);
	script.push(50000, 80000, 20000); // stats, disp, render
	prof.beginFrame(2, 41);
	for (let s = 0; s < 2; s++) for (const g of [G.vel, G.rhs, G.press, G.trans, G.phase]) prof.pass(encoder, g, "g");
	prof.pass(encoder, G.stats, "stats");
	prof.pass(encoder, G.disp, "disp");
	prof.closePass();
	const tw = prof.tsWrites(G.render);
	check("render tsWrites handed out", !!tw && tw.beginningOfPassWriteIndex === 24);
	prof.finish(encoder);
	await prof.poll();
	const r = prof.results;
	check("press sum over 2 substeps", r.valid && Math.abs(r.ms[G.press] - 1.8) < 1e-9, "press=" + r.ms[G.press]);
	check("vel sum", Math.abs(r.ms[G.vel] - 0.2) < 1e-9, "vel=" + r.ms[G.vel]);
	check("render separate", Math.abs(r.ms[G.render] - 0.02) < 1e-9, "render=" + r.ms[G.render]);
	check("frame/substeps recorded", r.frame === 41 && r.substeps === 2, "frame=" + r.frame + " ns=" + r.substeps);
	check("total", Math.abs(r.totalMs - (0.2 + 0.02 + 1.8 + 0.4 + 0.3 + 0.05 + 0.08 + 0.02)) < 1e-9, "total=" + r.totalMs);

	// frame 2 goes to the other ring slot while frame 1 already consumed
	script = [300000];
	prof.beginFrame(1, 42);
	prof.pass(encoder, G.press, "press");
	prof.finish(encoder);
	await prof.poll();
	check("second frame overwrites results", prof.results.frame === 42 && Math.abs(prof.results.ms[G.press] - 0.3) < 1e-9);

	// generation guard: invalidate between finish and poll drops the sample
	script = [700000];
	prof.beginFrame(1, 43);
	prof.pass(encoder, G.vel, "vel");
	prof.finish(encoder);
	prof.invalidate();
	await prof.poll();
	check("stale-generation sample dropped", prof.results.frame === 42 && !prof.results.valid);

	// capacity: >128 pairs -> passes still returned, extra groups untimed
	script = new Array(128).fill(1000);
	prof.beginFrame(30, 44);
	let lastPass = null;
	for (let i = 0; i < 200; i++) lastPass = prof.pass(encoder, G.vel, "vel");
	check("over-capacity still returns a pass", !!lastPass);
	prof.finish(encoder);
	await prof.poll();
	check("capacity clamped to 128 pairs", Math.abs(prof.results.ms[G.vel] - 128 * 0.001) < 1e-9, "vel=" + prof.results.ms[G.vel]);

	// feature guards
	check("no features object -> null", profilerMod.create({}) === null);
	check("feature absent -> null", profilerMod.create({ features: { has: () => false } }) === null);
	check("null device -> null", profilerMod.create(null) === null);
	prof.destroy();
	console.log(failures ? failures + " failures" : "all pass");
	process.exit(failures ? 1 : 0);
})();
