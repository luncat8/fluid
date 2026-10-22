import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

// Optional pre-device WGSL check: parse every shader with a real WGSL parser
// and execute kernels on a CPU interpreter, comparing against the CPU oracle
// (src/core/solver.js) field by field. This catches syntax and index/geometry
// errors that the recording mock (gpu-mock.js) cannot, because the mock runs
// no shader. It is NOT a device row: timing and driver validation still need
// hardware, and the interpreter's f32 rounding is its own, so the tolerances
// are rounding-scale, not bit-exact.
//
// Optional dependency, installed outside the repo like gpu-parity.mjs:
//   npm install --prefix /tmp/wgsl-check wgsl_reflect
//   FLUID_WGSL_REFLECT=/tmp/wgsl-check/node_modules/wgsl_reflect/wgsl_reflect.module.js \
//     node experiments/wgsl-exec.mjs

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
for (const file of [
	"src/core/config.js", "src/core/grid.js", "src/core/stats.js",
	"src/core/multigrid.js", "src/core/solver.js", "src/gpu/shaders.js"
]) require(path.join(root, file));

let reflectLib;
try {
	reflectLib = await import(process.env.FLUID_WGSL_REFLECT || "wgsl_reflect/wgsl_reflect.module.js");
} catch (error) {
	console.error("wgsl-exec needs the optional wgsl_reflect package outside the repo:");
	console.error("  npm install --prefix /tmp/wgsl-check wgsl_reflect");
	console.error("  FLUID_WGSL_REFLECT=/tmp/wgsl-check/node_modules/wgsl_reflect/wgsl_reflect.module.js node experiments/wgsl-exec.mjs");
	console.error(error.message);
	process.exit(2);
}
const { WgslReflect, WgslExec, WgslParser } = reflectLib;

let failures = 0;
function check(name, ok, detail) {
	console.log((ok ? "PASS " : "FAIL ") + name + (detail ? "  " + detail : ""));
	if (!ok) failures++;
}

// ---- 1. every shader parses as WGSL (the mock only counts braces) ----
const sh = Fluid.gpu.shaders;
const PS = sh.paramsStruct;
// poissonReduceB is compiled without the params prefix (see passes.js)
const bare = new Set(["renderVert", "renderFrag", "poissonReduceB"]);
for (const name of Object.keys(sh)) {
	if (name === "paramsStruct") continue;
	const code = bare.has(name) ? sh[name] : PS + sh[name];
	let ok = true, detail = "";
	try { new WgslReflect(code); } catch (e) { ok = false; detail = String(e.message || e).slice(0, 160); }
	check("parse " + name, ok, detail);
}

// ---- 2. execute the mac-cormack pair (t16) against the cpu oracle ----
// one advection step from the same initial field, pass 1 (the plain
// semi-lagrangian kernel) into the scratch, pass 2 (the correction) out.
// fields chosen so every branch is live: a smooth vortex (correction on,
// limiter idle), a constant field (the wall-band overshoot the limiter
// exists for) and a discontinuous field (limiter and clamps active).
const grid = Fluid.core.grid;
const N = 32, DT = 1 / 60;
const cfg = Object.assign({}, Fluid.core.config.defaults, { N, scenario: "still" });
cfg.texW = N + 2; cfg.texH = N + 2;
cfg.gravityX = 0; cfg.gravityY = 0; cfg.thermalSources = 0;
const texW = cfg.texW, texH = cfg.texH;
const sim = Fluid.core.solver.create(cfg);
sim.reset();

function runMacCase(name, vel0, limiter) {
	grid.curBuf(sim.vel).set(vel0);
	cfg.advectMac = 1; cfg.macLimiter = limiter;
	sim.advectVelocity(DT);
	const cpu = Float32Array.from(grid.curBuf(sim.vel));
	const limitCount = sim.diag.macLimit;
	cfg.dt = DT;
	const pa = Fluid.core.config.makeParamsArray();
	Fluid.core.config.writeParams(pa, cfg);
	const uniform = { uniform: pa.buffer.slice(0) };
	const rg = d => ({ texture: [d.buffer], descriptor: { size: [texW, texH], format: "rg32float" } });
	const flags = { texture: [new Uint8Array(texW * texH * 4).buffer], descriptor: { size: [texW, texH], format: "rgba8uint" } };
	const src = Float32Array.from(vel0);
	const mac = new Float32Array(vel0.length);
	const dst = new Float32Array(vel0.length);
	const wx = Math.ceil(texW / 8), wy = Math.ceil(texH / 8);
	new WgslExec(WgslParser.Parse(PS + sh.advectVel))
		.dispatchWorkgroups("main", [wx, wy], { 0: { 0: uniform, 1: rg(src), 2: rg(mac), 3: flags } });
	new WgslExec(WgslParser.Parse(PS + sh.advectVelMac))
		.dispatchWorkgroups("main", [wx, wy], { 0: { 0: uniform, 1: rg(src), 2: rg(mac), 3: rg(dst), 4: flags } });
	let maxDiff = 0;
	for (let i = 0; i < cpu.length; i++) maxDiff = Math.max(maxDiff, Math.abs(dst[i] - cpu[i]));
	check("mac exec: " + name + " limiter=" + limiter, maxDiff < 1e-3,
		"maxAbsDiff=" + maxDiff.toExponential(3) + " cpu macLimit=" + limitCount);
	return limitCount;
}
function paintVortex() {
	const vel = new Float32Array(texW * texH * 2);
	const sigma = N / 6, cx = N * 0.5, cy = N * 0.5;
	const invS2 = 1 / (2 * sigma * sigma), A = 80 / Math.exp(-0.5);
	for (let y = 1; y < texH - 1; y++) for (let x = 1; x < texW - 1; x++) {
		const vi = (x + y * texW) * 2;
		if (!sim.uBlocked(x, y)) { const dx = (x - 1) - cx, dy = (y - 0.5) - cy; vel[vi] = -A * (dy / sigma) * Math.exp(-(dx * dx + dy * dy) * invS2); }
		if (!sim.vBlocked(x, y)) { const dx = (x - 0.5) - cx, dy = (y - 1) - cy; vel[vi + 1] = A * (dx / sigma) * Math.exp(-(dx * dx + dy * dy) * invS2); }
	}
	return vel;
}
function paintConstant() {
	const vel = new Float32Array(texW * texH * 2);
	for (let y = 1; y < texH - 1; y++) for (let x = 1; x < texW - 1; x++)
		if (!sim.uBlocked(x, y)) vel[(x + y * texW) * 2] = 37;
	return vel;
}
function paintRandom() {
	const vel = new Float32Array(texW * texH * 2);
	let s = 12345;
	const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff * 2 - 1; };
	for (let y = 1; y < texH - 1; y++) for (let x = 1; x < texW - 1; x++) {
		const vi = (x + y * texW) * 2;
		if (!sim.uBlocked(x, y)) vel[vi] = 120 * rnd();
		if (!sim.vBlocked(x, y)) vel[vi + 1] = 120 * rnd();
	}
	return vel;
}
runMacCase("gaussian vortex", paintVortex(), 1);
const wallLimited = runMacCase("constant U=37", paintConstant(), 1);
runMacCase("constant U=37", paintConstant(), 0);
const randomLimited = runMacCase("random faces", paintRandom(), 1);
runMacCase("random faces", paintRandom(), 0);
check("the limiter branch was actually exercised", wallLimited > 0 && randomLimited > 0,
	"wall=" + wallLimited + " random=" + randomLimited);

console.log(failures ? "FAILURES: " + failures : "ALL WGSL-EXEC CHECKS PASSED");
process.exit(failures ? 1 : 0);
