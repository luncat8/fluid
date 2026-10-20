"use strict";
// 0.6.0 a5: what the field pool costs the device, measured from the allocation list.
// webgpu has no "how much did you allocate" query, so this walks the same lists
// src/gpu/fields.js uses to destroy the pool: `logical` is what we asked for, `padded`
// is the worst case for an implementation that rounds every texture row up to 256 bytes.
// Neither number includes the canvas backbuffer, the driver's own block rounding (a few
// percent on top), or the cpu mirrors of the initial state, which are freed after upload.
// The page's report prints the same pair from the same function (`gpu memory:`), so a
// hardware capture and this script cannot disagree about what the number means.
// run: node experiments/gpu-memory.js
//      node experiments/gpu-memory.js --n 1024      per-field breakdown
//      node experiments/gpu-memory.js --strict      over-bar rows exit non-zero

const path = require("node:path");
const root = path.resolve(__dirname, "..");
globalThis.GPUBufferUsage = { MAP_READ: 1, COPY_SRC: 2, COPY_DST: 4, INDEX: 8, VERTEX: 16, UNIFORM: 32, STORAGE: 64 };
globalThis.GPUTextureUsage = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/gpu/fields.js"));

const MB = 1024 * 1024;
// 0.6.0 t7 as a pricing model, not an adoption: the half-float candidate set of plan 4.3,
// plus dropping sfield's ping-pong copy (single writer, no stencil). the real gate for
// each field is drift_rel and the residual, which this script cannot see.
const WHATIF = {
	formats: {
		vel0: "rg16float", vel1: "rg16float",
		press0: "r16float", press1: "r16float", divsrc: "r16float",
		flux: "rg16float", limiterA: "rgba16float", limiterB: "rg16float",
		sfield: "rgba16float"
	},
	drop: ["sfieldTmp"]
};
// plan 0.6.0 a5. N=1024 is over it by design of the field set (8 full-res rgba32float
// fields are 61% of the footprint) and t7 half-float per field is the assigned fix, so
// that row is reported, not asserted, until t7 lands. --strict asserts both.
const BAR = { 256: null, 512: 60, 1024: 200 };
const ASSERTED = { 512: true };

// a device that only records: enough for createPool, no simulation
function countingDevice(sink) {
	return {
		createBuffer: d => { const b = { label: d.label, size: d.size }; sink.bufs.push(b); return b; },
		createTexture: d => {
			const t = { label: d.label, createView: () => ({}), destroy() { t.destroyed = true; } };
			sink.texs.push(t);
			return t;
		},
		queue: { writeBuffer() {}, writeTexture() {} }
	};
}
function mb(x) { return (x / MB).toFixed(1) + " MB"; }
function textureBytesWith(t, map) {
	const f = map.formats[t.label] || t.fluidSize[2];
	return t.fluidSize[0] * t.fluidSize[1] * Fluid.gpu.fields.texBpp(f);
}
// same walk as memoryReport, but through the what-if map: formats replaced, fields dropped
function whatifReport(pool, map) {
	let textures = 0, texCount = 0, paddedTex = 0;
	for (const t of Fluid.gpu.fields.poolTextures(pool)) {
		if (!t || !t.fluidSize) continue;
		if (map.drop && map.drop.indexOf(t.label) >= 0) continue;
		const h = t.fluidSize[1];
		const b = textureBytesWith(t, map);
		textures += b; texCount++;
		paddedTex += Math.ceil(b / h / 256) * 256 * h;
	}
	const m = Fluid.gpu.fields.memoryReport(pool);
	return { textures, buffers: m.buffers, total: textures + m.buffers, padded: paddedTex + m.buffers,
		texCount, bufCount: m.bufCount };
}
function breakdown(pool) {
	const fields = Fluid.gpu.fields;
	const rows = [];
	for (const t of fields.poolTextures(pool)) {
		if (!t || !t.fluidSize) continue;
		const bytes = t.fluidSize[0] * t.fluidSize[1] * fields.texBpp(t.fluidSize[2]);
		rows.push({ label: t.label, bytes, format: t.fluidSize[2] });
	}
	return rows.sort((a, b) => b.bytes - a.bytes);
}

const argv = process.argv.slice(2);
const onlyN = argv.includes("--n") ? Number(argv[argv.indexOf("--n") + 1]) : null;
const strict = argv.includes("--strict");
const cfg = Fluid.core.config.defaults;
let failures = 0;
console.log("field pool footprint, mgCoarsen=" + cfg.mgCoarsen + " (rows padded to 256 in the"
	+ " worst-case column; a5 is judged on that column)");
for (const N of [256, 512, 1024]) {
	if (onlyN && N !== onlyN) continue;
	const sink = { bufs: [], texs: [] };
	const pool = Fluid.gpu.fields.createPool(countingDevice(sink), N, cfg.mgCoarsen);
	const m = Fluid.gpu.fields.memoryReport(pool);
	if (!m) {
		console.log("FAIL no memory report for N=" + N);
		failures++;
		continue;
	}
	const worst = Math.max(m.total, m.padded);
	console.log("N=" + N + "  textures " + mb(m.textures) + " (" + m.texCount + ") + buffers "
		+ mb(m.buffers) + " (" + m.bufCount + ")  = " + mb(m.total) + " logical, " + mb(m.padded)
		+ " worst case (+" + (100 * (m.padded / m.total - 1)).toFixed(1) + "%)  texW=" + pool.texW
		+ " mgTexW=" + pool.mgTexW);
	// the allocation list and the destroy list must be the same list, or a field is
	// either leaked or uncounted; both are silent without this
	const covered = sink.texs.length === m.texCount && sink.bufs.length === m.bufCount;
	const bar = BAR[N];
	if (!covered) {
		console.log("  FAIL uncounted allocation: device saw " + sink.texs.length + "/"
			+ sink.bufs.length + ", report counted " + m.texCount + "/" + m.bufCount);
		failures++;
	}
	if (bar) {
		const over = worst / MB - bar;
		const pct = 100 * over / bar;
		const assertThis = ASSERTED[N] || strict;
		if (over > 0) {
			console.log("  " + (assertThis ? "FAIL" : "OVER") + " a5 bar " + bar + " MB by "
				+ over.toFixed(1) + " MB (" + pct.toFixed(1) + "%)"
				+ (assertThis ? "" : "  [reported, not asserted until t7]"));
			if (assertThis) failures++;
		} else {
			console.log("  PASS a5 bar " + bar + " MB: " + (-pct).toFixed(1) + "% headroom"
				+ (-pct < 5 ? " (MARGINAL: a new full-res field at this N would trip it)" : ""));
		}
		if (over > 0 || onlyN) {
			for (const r of breakdown(pool).slice(0, 10)) {
				console.log("    " + r.label.padEnd(10, " ") + (r.bytes / MB).toFixed(2) + " MB  " + r.format);
			}
		}
	}
	if (argv.includes("--whatif")) {
		const w = whatifReport(pool, WHATIF);
		console.log("  what-if t7 map (vel/press/divsrc/flux/limiter/sfield half float,"
			+ " sfieldTmp dropped): " + mb(w.total) + " logical, " + mb(w.padded) + " worst case"
			+ "  -" + ((m.total - w.total) / MB).toFixed(1) + " MB  (pricing only: the drift and"
			+ " residual checks decide each field)");
	}
	Fluid.gpu.fields.destroyPool(pool);
	const leaked = sink.texs.filter(t => !t.destroyed).length;
	if (leaked) { console.log("  FAIL " + leaked + " textures survived destroyPool"); failures++; }
}
console.log(failures === 0 ? "MEMORY CHECKS PASSED" : failures + " FAILURES");
process.exitCode = failures === 0 ? 0 : 1;
