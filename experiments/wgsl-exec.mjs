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
	"src/core/multigrid.js", "src/core/solver.js", "src/core/probes.js", "src/core/emitters.js", "src/core/colliders.js",
	"src/gpu/fields.js", "src/gpu/shaders.js"
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

// ---- 3. execute t8 fused passes (advectScalar, phaseSources) ----
// fused WGSL kernels must be bit-identical (maxDiff === 0) to the unfused
// WGSL sequence they replace, and match the CPU oracle within f32 rounding.
{
	const n = 16, tw = n + 2, th = n + 2;
	const wx = Math.ceil(tw / 8), wy = Math.ceil(th / 8);
	const fcfg = Object.assign({}, Fluid.core.config.defaults, {
		N: n, texW: tw, texH: th, scenario: "cycle", dt: DT,
		fctLimiter: 0, thermalSources: 1, chargeLatent: 1
	});
	Fluid.core.config.prepareScenario(fcfg);
	fcfg.N = n; fcfg.texW = tw; fcfg.texH = th; fcfg.dt = DT; fcfg.fctLimiter = 0;
	fcfg.chargeLatent = 1; fcfg.TCeil = 265; fcfg.heatTarget = 390;
	const fsim = Fluid.core.solver.create(fcfg);
	fsim.reset();
	fcfg.dt = DT; fcfg.fctLimiter = 0; fcfg.chargeLatent = 1;
	fcfg.TCeil = 265; fcfg.heatTarget = 390;
	const pa = Fluid.core.config.makeParamsArray();
	Fluid.core.config.writeParams(pa, fcfg);
	const uniform = { uniform: pa.buffer.slice(0) };
	const rg = d => ({ texture: [d.buffer], descriptor: { size: [tw, th], format: "rg32float" } });
	const rgba = d => ({ texture: [d.buffer], descriptor: { size: [tw, th], format: "rgba32float" } });
	const u8 = d => ({ texture: [d.buffer], descriptor: { size: [tw, th], format: "rgba8uint" } });

	let seed = 24680;
	const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
	const vel = grid.curBuf(fsim.vel);
	const mass0 = Float32Array.from(grid.curBuf(fsim.mass));
	const therm0 = Float32Array.from(grid.curBuf(fsim.therm));
	for (let y = 1; y < th - 1; y++) for (let x = 1; x < tw - 1; x++) {
		const i4 = (x + y * tw) * 4, i2 = (x + y * tw) * 2;
		vel[i2] = (rnd() * 2 - 1) * 20;
		vel[i2 + 1] = (rnd() * 2 - 1) * 20;
		mass0[i4] = 0.2 * rnd(); mass0[i4 + 1] = 0.3 * rnd(); mass0[i4 + 2] = 0.4 * rnd();
		mass0[i4 + 3] = y === 2 && x === 2 ? 0 : 0.4 + 0.6 * rnd();
		const T = 250 + 160 * (1 - y / n) + 15 * (rnd() - 0.5);
		therm0[i4] = mass0[i4 + 3] * fcfg.cLiq * T;
		therm0[i4 + 1] = T;
		therm0[i4 + 2] = 5 * rnd();
		therm0[i4 + 3] = 3 * rnd();
	}
	fsim.computeFlux(DT);
	const flux = Float32Array.from(fsim.flux);
	grid.curBuf(fsim.mass).set(mass0);
	grid.curBuf(fsim.therm).set(therm0);
	fsim.advectScalar();
	const cpuMass = Float32Array.from(grid.curBuf(fsim.mass));
	const cpuTherm = Float32Array.from(grid.curBuf(fsim.therm));

	const mUnf = Float32Array.from(mass0), tUnf = Float32Array.from(therm0);
	const mFus = Float32Array.from(mass0), tFus = Float32Array.from(therm0);
	new WgslExec(WgslParser.Parse(PS + sh.advectMass))
		.dispatchWorkgroups("main", [wx, wy], { 0: { 0: uniform, 1: rg(flux), 2: rgba(mass0), 3: rgba(mUnf) } });
	new WgslExec(WgslParser.Parse(PS + sh.advectTherm))
		.dispatchWorkgroups("main", [wx, wy], { 0: { 0: uniform, 1: rg(flux), 2: rgba(therm0), 3: rgba(tUnf) } });
	new WgslExec(WgslParser.Parse(PS + sh.advectScalar))
		.dispatchWorkgroups("main", [wx, wy], {
			0: { 0: uniform, 1: rg(flux), 2: rgba(mass0), 3: rgba(therm0), 4: rgba(mFus), 5: rgba(tFus) }
		});
	let diffScalarM = 0, diffScalarT = 0, diffCpuM = 0, diffCpuT = 0;
	for (let i = 0; i < mFus.length; i++) {
		diffScalarM = Math.max(diffScalarM, Math.abs(mFus[i] - mUnf[i]));
		diffScalarT = Math.max(diffScalarT, Math.abs(tFus[i] - tUnf[i]));
		diffCpuM = Math.max(diffCpuM, Math.abs(mFus[i] - cpuMass[i]));
		diffCpuT = Math.max(diffCpuT, Math.abs(tFus[i] - cpuTherm[i]));
	}
	check("t8 advectScalar: bit-exact with unfused advectMass + advectTherm",
		diffScalarM === 0 && diffScalarT === 0,
		"massDiff=" + diffScalarM + " thermDiff=" + diffScalarT);
	check("t8 advectScalar: matches CPU oracle",
		diffCpuM < 1e-5 && diffCpuT < 5e-2,
		"cpuMassDiff=" + diffCpuM.toExponential(3) + " cpuThermDiff=" + diffCpuT.toExponential(3));

	const flagsPrev = new Uint8Array(tw * th * 4);
	flagsPrev[(3 + 1 * tw) * 4] = 1; // wall cell inside bottom heater band
	const sfieldPrev = new Float32Array(tw * th * 4);
	const accumPrev = new Float32Array(tw * th * 2);
	for (let i = 0; i < tw * th; i++) {
		sfieldPrev[i * 4 + 1] = 0.8;
		sfieldPrev[i * 4 + 2] = 0.1;
		accumPrev[i * 2] = 1.5;
		accumPrev[i * 2 + 1] = 0.25;
	}
	const tMid = Float32Array.from(therm0);
	const tOutUnf = new Float32Array(therm0.length), tOutFus = new Float32Array(therm0.length);
	const flUnf = new Uint8Array(flagsPrev.length), flFus = new Uint8Array(flagsPrev.length);
	const sfUnf = new Float32Array(sfieldPrev.length), sfFus = new Float32Array(sfieldPrev.length);
	const acUnf = new Float32Array(accumPrev.length), acFus = new Float32Array(accumPrev.length);

	const phaseAst = WgslParser.Parse(PS + sh.phaseEos);
	new WgslExec(WgslParser.Parse(PS + sh.heatSources))
		.dispatchWorkgroups("main", [wx, wy], {
			0: { 0: uniform, 1: rgba(mass0), 2: rgba(therm0), 3: rgba(tMid), 4: u8(flagsPrev) }
		});
	new WgslExec(phaseAst)
		.dispatchWorkgroups("main", [wx, wy], {
			0: {
				0: uniform, 1: rgba(mass0), 2: rgba(tMid), 3: rg(vel),
				4: u8(flagsPrev), 5: rgba(sfieldPrev), 6: rg(accumPrev),
				7: rgba(tOutUnf), 8: u8(flUnf), 9: rgba(sfUnf), 10: rg(acUnf)
			}
		}, { constants: { APPLY_SOURCES: 0 } });
	new WgslExec(phaseAst)
		.dispatchWorkgroups("main", [wx, wy], {
			0: {
				0: uniform, 1: rgba(mass0), 2: rgba(therm0), 3: rg(vel),
				4: u8(flagsPrev), 5: rgba(sfieldPrev), 6: rg(accumPrev),
				7: rgba(tOutFus), 8: u8(flFus), 9: rgba(sfFus), 10: rg(acFus)
			}
		}, { constants: { APPLY_SOURCES: 1 } });
	let diffPT = 0, diffPF = 0, diffPS = 0, diffPA = 0, heatedCells = 0;
	for (let i = 0; i < tOutFus.length; i++) {
		diffPT = Math.max(diffPT, Math.abs(tOutFus[i] - tOutUnf[i]));
		diffPF = Math.max(diffPF, Math.abs(flFus[i] - flUnf[i]));
		diffPS = Math.max(diffPS, Math.abs(sfFus[i] - sfUnf[i]));
	}
	for (let i = 0; i < acFus.length; i++) diffPA = Math.max(diffPA, Math.abs(acFus[i] - acUnf[i]));
	for (let i = 0; i < therm0.length; i += 4) if (tOutFus[i] !== therm0[i]) heatedCells++;
	const wallCellIdx = (3 + 1 * tw) * 4;
	const vacCellIdx = (2 + 2 * tw) * 4;
	check("t8 phaseSources: bit-exact with unfused heatSources -> phaseEos across all 4 outputs",
		diffPT === 0 && diffPF === 0 && diffPS === 0 && diffPA === 0 && heatedCells > 0,
		"therm=" + diffPT + " flags=" + diffPF + " sfield=" + diffPS + " accum=" + diffPA + " active=" + heatedCells);
	check("t8 phaseSources: skips wall and vacuum cells in heater band",
		tOutFus[wallCellIdx] === therm0[wallCellIdx] && tOutFus[vacCellIdx] === therm0[vacCellIdx],
		"wallUnchanged=" + (tOutFus[wallCellIdx] === therm0[wallCellIdx]) +
		" vacUnchanged=" + (tOutFus[vacCellIdx] === therm0[vacCellIdx]));

	grid.curBuf(fsim.mass).set(mass0);
	grid.curBuf(fsim.therm).set(therm0);
	grid.curBuf(fsim.vel).set(vel);
	fsim.flags.set(flagsPrev);
	fsim.sfield.set(sfieldPrev);
	fsim.ledger.latentChargedE = 0;
	fsim.ledger.frozenMomentum = 0;
	fsim.heatSources(DT);
	fsim.phaseEos(true);
	const cpuT2 = grid.curBuf(fsim.therm);
	let diffCpuPT = 0, diffCpuPF = 0;
	for (let y = 1; y < th - 1; y++) for (let x = 1; x < tw - 1; x++) {
		const i = (x + y * tw) * 4;
		for (let c = 0; c < 4; c++) {
			diffCpuPT = Math.max(diffCpuPT, Math.abs(tOutFus[i + c] - cpuT2[i + c]));
			diffCpuPF = Math.max(diffCpuPF, Math.abs(flFus[i + c] - fsim.flags[i + c]));
		}
	}
	check("t8 phaseSources: matches CPU heatSources + phaseEos",
		diffCpuPT < 1e-2 && diffCpuPF === 0,
		"cpuThermDiff=" + diffCpuPT.toExponential(3) + " cpuFlagsDiff=" + diffCpuPF);
}


// ---- 3. execute the probe kernel (0.7.0 t5) against the cpu sampler ----
// the kernel decides three things no other check can: the slot stride, the
// row-major texel index from (x, y) and the interior clamp. run it on a
// synthetic field that varies in both axes so a transposed index cannot hide.
function runProbeCase() {
	const P = Fluid.core.probes;
	const tw = texW, th = texH;
	const mass = new Float32Array(tw * th * 4);
	const therm = new Float32Array(tw * th * 4);
	const vel = new Float32Array(tw * th * 2);
	const press = new Float32Array(tw * th);
	for (let y = 0; y < th; y++) {
		for (let x = 0; x < tw; x++) {
			const i4 = (x + y * tw) * 4, i2 = (x + y * tw) * 2;
			const m = 0.5 + 0.3 * Math.sin(x * 0.3) * Math.cos(y * 0.2);
			mass[i4] = 0.2 * m; mass[i4 + 1] = 0.3 * m; mass[i4 + 2] = 0.5 * m; mass[i4 + 3] = m;
			therm[i4] = Fluid.core.solver.energyFromT(m, 300 + 0.5 * x + 0.25 * y, cfg);
			vel[i2] = 0.5 * x;
			vel[i2 + 1] = -0.25 * y;
			press[x + y * tw] = 10 * Math.sin(x * 0.1);
		}
	}
	const table = P.create();
	const ids = [
		P.add(table, cfg, 1, 1),        // bottom-left interior corner
		P.add(table, cfg, N, N),        // top-right cell, not the ghost row
		P.add(table, cfg, 7.6, 3.2),    // rounded, not truncated
		P.add(table, cfg, 2.4, 15.5)
	];
	// sentinel in a slot that is not active: the kernel must not touch it
	const idleBase = P.slotBase(9);
	table.upload[idleBase + P.P_M] = -7;
	const storage = table.upload.buffer.slice(0);
	const pa = Fluid.core.config.makeParamsArray();
	Fluid.core.config.writeParams(pa, cfg);
	const uniform = { uniform: pa.buffer.slice(0) };
	const rgba = a => ({ texture: [a.buffer], descriptor: { size: [tw, th], format: "rgba32float" } });
	const rg = a => ({ texture: [a.buffer], descriptor: { size: [tw, th], format: "rg32float" } });
	const r32 = a => ({ texture: [a.buffer], descriptor: { size: [tw, th], format: "r32float" } });
	new WgslExec(WgslParser.Parse(PS + sh.probeSample))
		.dispatchWorkgroups("main", [Fluid.core.probes.MAX / 8], {
			0: { 0: uniform, 1: rgba(mass), 2: rgba(therm), 3: rg(vel), 4: r32(press), 5: storage }
		});
	const raw = new Float32Array(storage);
	const cpu = P.makeSample();
	let maxDiff = 0, values = 0, mismatched = [];
	for (const id of ids) {
		P.sampleCpu(table, cfg, mass, therm, vel, press, tw, th, id, cpu);
		const b = P.slotBase(id);
		const got = [raw[b + P.P_M], raw[b + P.P_DR], raw[b + P.P_DG], raw[b + P.P_DB],
			raw[b + P.P_E], raw[b + P.P_U], raw[b + P.P_V], raw[b + P.P_P]];
		const want = [cpu.m, cpu.dr, cpu.dg, cpu.db, cpu.E, cpu.u, cpu.v, cpu.p];
		for (let i = 0; i < got.length; i++) {
			const d = Math.abs(got[i] - want[i]);
			if (d > maxDiff) maxDiff = d;
			if (d > 1e-3) mismatched.push(id + ":" + i + " " + got[i] + "!=" + want[i]);
		}
		values += got.length;
	}
	check("probe exec: every measured slot matches the cpu sampler (stride, index, clamp)",
		maxDiff < 1e-3 && values === ids.length * 8,
		"maxAbsDiff=" + maxDiff.toExponential(3) + " values=" + values +
		(mismatched.length ? " " + mismatched.slice(0, 3).join(" ") : ""));
	check("probe exec: an inactive slot is left alone", raw[idleBase + P.P_M] === -7,
		"sentinel=" + raw[idleBase + P.P_M]);
	// moving a probe changes the cell the kernel reads, and nothing else
	P.move(table, cfg, ids[2], 1, 1);
	const moved = table.upload.buffer.slice(0);
	P.sampleCpu(table, cfg, mass, therm, vel, press, tw, th, ids[2], cpu);
	new WgslExec(WgslParser.Parse(PS + sh.probeSample))
		.dispatchWorkgroups("main", [Fluid.core.probes.MAX / 8], {
			0: { 0: uniform, 1: rgba(mass), 2: rgba(therm), 3: rg(vel), 4: r32(press), 5: moved }
		});
	const movedRaw = new Float32Array(moved);
	const movedBase = P.slotBase(ids[2]);
	check("probe exec: moveProbe is picked up by the next dispatch (one write, no rebuild)",
		Math.abs(movedRaw[movedBase + P.P_M] - cpu.m) < 1e-3 &&
		Math.abs(movedRaw[movedBase + P.P_U] - cpu.u) < 1e-3,
		"m=" + movedRaw[movedBase + P.P_M].toFixed(4) + " u=" + movedRaw[movedBase + P.P_U].toFixed(4));
}


// ---- 4. execute the emitter kernels (0.7.0 t3) against the cpu oracle ----
// the three passes are full-field copies with cell-local additions. the check that
// matters is not just "same numbers": the total the gpu injected must equal the
// total the cpu wrote into the ledger, or the drift meters would lie while a game
// leaves a heater on.
function runEmitterCase() {
	const E = Fluid.core.emitters;
	const dtFrame = 1 / 15;
	cfg.dt = dtFrame / 4;
	cfg.substeps = 4;
	const table = E.create();
	E.add(table, cfg, { kind: "dye", x: 10, y: 12, radius: 6, value: 2.5, dye: [1, 0.3, 0] });
	E.add(table, cfg, { kind: "dye", x: 30, y: 5.5, radius: 7.25, value: 1.2, dye: [0.1, 0.9, 0.2] });
	E.add(table, cfg, { kind: "heat", x: 16, y: 16, radius: 9, value: 300 });
	E.add(table, cfg, { kind: "cold", x: 16, y: 25, radius: 5.5, value: 220 });
	E.add(table, cfg, { kind: "momentum", x: 8, y: 20, radius: 4.5, value: 150, dir: [0.6, -0.8] });
	E.remove(table, E.add(table, cfg, { kind: "dye", x: 2, y: 2, radius: 20, value: 5 }));
	const size = texW * texH;
	const baseM = new Float32Array(size * 4);
	const baseT = new Float32Array(size * 4);
	const baseV = new Float32Array(size * 2);
	for (let y = 0; y < texH; y++) {
		for (let x = 0; x < texW; x++) {
			const i4 = (x + y * texW) * 4, i2 = (x + y * texW) * 2;
			const m = 0.5 + 0.25 * Math.sin(x * 0.4) * Math.cos(y * 0.3);
			baseM[i4] = 0.2 * m; baseM[i4 + 1] = 0.3 * m; baseM[i4 + 2] = 0.5 * m; baseM[i4 + 3] = m;
			baseT[i4] = 800 * m;
			baseV[i2] = 0.5 * x;
			baseV[i2 + 1] = -0.25 * y;
		}
	}
	const cpuM = Float32Array.from(baseM), cpuT = Float32Array.from(baseT), cpuV = Float32Array.from(baseV);
	E.header(table, cfg);
	const out = {};
	E.sweep(table, cfg, dtFrame, { mass: cpuM, therm: cpuT, vel: cpuV, texW: texW, texH: texH }, out);
	const pa = Fluid.core.config.makeParamsArray();
	Fluid.core.config.writeParams(pa, cfg);
	const uniform = { uniform: pa.buffer.slice(0) };
	const tableBuf = table.upload.buffer.slice(0);
	const rgba = a => ({ texture: [a.buffer], descriptor: { size: [texW, texH], format: "rgba32float" } });
	const rg = a => ({ texture: [a.buffer], descriptor: { size: [texW, texH], format: "rg32float" } });
	const wx = Math.ceil(texW / 8), wy = Math.ceil(texH / 8);
	const dstM = new Float32Array(baseM.length);
	const dstT = new Float32Array(baseT.length);
	const dstV = new Float32Array(baseV.length);
	new WgslExec(WgslParser.Parse(PS + sh.emitterMass))
		.dispatchWorkgroups("main", [wx, wy], { 0: { 0: uniform, 1: rgba(baseM), 2: rgba(dstM), 3: tableBuf } });
	new WgslExec(WgslParser.Parse(PS + sh.emitterTherm))
		.dispatchWorkgroups("main", [wx, wy], { 0: { 0: uniform, 1: rgba(baseT), 2: rgba(dstT), 3: tableBuf } });
	new WgslExec(WgslParser.Parse(PS + sh.emitterVel))
		.dispatchWorkgroups("main", [wx, wy], { 0: { 0: uniform, 1: rg(baseV), 2: rg(dstV), 3: tableBuf } });
	const maxDiff = (a, b) => {
		let d = 0;
		for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
		return d;
	};
	const relDiff = (a, b) => {
		let d = 0, scale = 0;
		for (let i = 0; i < a.length; i++) {
			d = Math.max(d, Math.abs(a[i] - b[i]));
			scale = Math.max(scale, Math.abs(b[i]));
		}
		return scale > 0 ? d / scale : d;
	};
	const dm = maxDiff(dstM, cpuM), dt = maxDiff(dstT, cpuT), dv = maxDiff(dstV, cpuV);
	check("emitter exec: mass/therm/vel passes match the cpu oracle",
		relDiff(dstM, cpuM) < 1e-5 && relDiff(dstT, cpuT) < 1e-5 && relDiff(dstV, cpuV) < 1e-5,
		"mass=" + dm.toExponential(2) + " therm=" + dt.toExponential(2) + " vel=" + dv.toExponential(2));
	// what the gpu actually added, summed the same way the stats reduction sums
	let gpuIn = 0, gpuDr = 0, gpuE = 0;
	for (let i = 0; i < size; i++) {
		gpuIn += dstM[i * 4 + 3] - baseM[i * 4 + 3];
		gpuDr += dstM[i * 4] - baseM[i * 4];
		gpuE += dstT[i * 4] - baseT[i * 4];
	}
	const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);
	// energy is booked on two sides (dye/heat in, cold out), so the field change is
	// the difference; mass only ever comes in, and must match the injection alone
	check("emitter exec: the ledger totals are the mass and energy the gpu added",
		rel(gpuIn, out.inM) < 1e-6 && rel(gpuDr, out.inDr) < 1e-6 && rel(gpuE, out.inE - out.outE) < 1e-6,
		"m " + gpuIn.toFixed(6) + "/" + out.inM.toFixed(6) + " E " + gpuE.toFixed(1) + "/" +
		(out.inE - out.outE).toFixed(1));
	check("emitter exec: passes staged by kind, and the removed slot adds nothing",
		out.needMass && out.needTherm && out.needVel && table.live === 5,
		"mass=" + out.needMass + " therm=" + out.needTherm + " vel=" + out.needVel + " live=" + table.live);
	// the ghost ring must be copied through untouched: the interior clamp is what
	// keeps a source from writing into a border a stencil reads as zero
	let ghost = 0;
	for (let y = 0; y < texH; y++) for (let x = 0; x < texW; x++) {
		if (x >= 1 && x <= texW - 2 && y >= 1 && y <= texH - 2) continue;
		const i4 = (x + y * texW) * 4, i2 = (x + y * texW) * 2;
		ghost = Math.max(ghost, Math.abs(dstM[i4 + 3] - baseM[i4 + 3]), Math.abs(dstT[i4] - baseT[i4]),
			Math.abs(dstV[i2] - baseV[i2]));
	}
	check("emitter exec: ghost cells are copied, never sourced", ghost === 0, "maxDiff=" + ghost);
	// the cold sink is a removal, not a negative injection: the two ledgers differ
	check("emitter exec: a cold slot books removed energy, not injected",
		out.outE > 0 && out.inE > 0, "inE=" + out.inE.toFixed(0) + " outE=" + out.outE.toFixed(0));
}


// ---- 5. execute the collider velocity imprint (0.7.0 t4) against the cpu oracle ----
// the imprint is a set, not an add: running it twice must leave the same field, and
// the cells outside the patch rect must come through the copy untouched. that is what
// keeps a pushed body from compounding its own velocity every frame.
function runColliderCase() {
	const PATCH = Fluid.gpu.fields.PATCH;
	const patch = new Float32Array(PATCH * PATCH * 4);
	const ox = 12, oy = 9, pw = 7, ph = 5;
	for (let y = 0; y < PATCH; y++) for (let x = 0; x < PATCH; x++) {
		const i = (x + y * PATCH) * 4;
		patch[i] = 111; patch[i + 1] = -222; patch[i + 2] = 0;
		if (x < pw && y < ph) patch[i + 3] = (x + y) % 3 === 0 ? 1 : 0;
	}
	// the solver writes rest for the body's own cells and the body velocity for the
	// fluid next to it; the alpha channel is what selects a cell
	const bodyX = ox + 2, bodyY = oy + 2;
	{
		const i = (2 + 2 * PATCH) * 4;
		patch[i] = 0; patch[i + 1] = 0; patch[i + 3] = 1;
	}
	{
		const i = (3 + 2 * PATCH) * 4;
		patch[i] = 120; patch[i + 1] = -30; patch[i + 3] = 1;
	}
	cfg.paintX = ox; cfg.paintY = oy; cfg.paintW = pw; cfg.paintH = ph;
	const size = texW * texH;
	const src = new Float32Array(size * 2);
	for (let y = 0; y < texH; y++) for (let x = 0; x < texW; x++) {
		const i = (x + y * texW) * 2;
		src[i] = 0.25 * x; src[i + 1] = -0.5 * y;
	}
	const want = Float32Array.from(src);
	for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
		const a = (x + (y * PATCH)) * 4 + 3;
		if (patch[a] <= 0.5) continue;
		const gi = ((ox + x) + (oy + y) * texW) * 2;
		want[gi] = patch[(x + y * PATCH) * 4];
		want[gi + 1] = patch[(x + y * PATCH) * 4 + 1];
	}
	const pa = Fluid.core.config.makeParamsArray();
	Fluid.core.config.writeParams(pa, cfg);
	const uniform = { uniform: pa.buffer.slice(0) };
	const rg = a => ({ texture: [a.buffer], descriptor: { size: [texW, texH], format: "rg32float" } });
	const rgbaPatch = { texture: [patch.buffer], descriptor: { size: [PATCH, PATCH], format: "rgba32float" } };
	const out = new Float32Array(src.length);
	const wx = Math.ceil(texW / 8), wy = Math.ceil(texH / 8);
	new WgslExec(WgslParser.Parse(PS + sh.colliderVel))
		.dispatchWorkgroups("main", [wx, wy],
			{ 0: { 0: uniform, 1: rg(src), 2: rg(out), 3: rgbaPatch } });
	let diff = 0;
	for (let i = 0; i < out.length; i++) diff = Math.max(diff, Math.abs(out[i] - want[i]));
	check("collider exec: the imprint matches the cpu oracle cell for cell", diff === 0, "maxDiff=" + diff);
	const bi = (bodyX + bodyY * texW) * 2;
	const ai = ((bodyX + 1) + bodyY * texW) * 2;
	check("collider exec: the body's own cell is at rest and its neighbour has its velocity",
		out[bi] === 0 && out[bi + 1] === 0 && out[ai] === 120 && out[ai + 1] === -30,
		"body=" + out[bi] + "," + out[bi + 1] + " ahead=" + out[ai] + "," + out[ai + 1]);
	// a second dispatch from the first result: identical, so the write is a set
	const again = new Float32Array(src.length);
	new WgslExec(WgslParser.Parse(PS + sh.colliderVel))
		.dispatchWorkgroups("main", [wx, wy],
			{ 0: { 0: uniform, 1: rg(out), 2: rg(again), 3: rgbaPatch } });
	let twice = 0;
	for (let i = 0; i < again.length; i++) twice = Math.max(twice, Math.abs(again[i] - out[i]));
	check("collider exec: re-running the imprint is idempotent (set, not add)", twice === 0, "maxDiff=" + twice);
	// cells outside the rect keep the source velocity, which is the copy half
	const far = ((texW - 5) + (texH - 5) * texW) * 2;
	const farOk = out[far] === src[far] && out[far + 1] === src[far + 1];
	check("collider exec: cells outside the patch rect are copied through", farOk,
		"far=" + out[far] + "," + out[far + 1]);
	// and a body cell whose alpha is 0 stays fluid: the neighbours are painted, not the hole
	let skipped = 0;
	for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
		const a = (x + y * PATCH) * 4 + 3;
		if (patch[a] > 0.5) continue;
		const gi = ((ox + x) + (oy + y) * texW) * 2;
		if (out[gi] !== src[gi] || out[gi + 1] !== src[gi + 1]) skipped++;
	}
	check("collider exec: cells the paint did not mark keep their velocity", skipped === 0, "changed=" + skipped);
}


// ---- 6. execute the body push (0.7.0 t4b) against the cpu oracle ----
// a body takes cells; the taken cells give their dye and heat to the free neighbours.
// the check is not only "same numbers": the totals must not move, because this is the
// pass that makes a body look like it pushes water instead of sliding over a picture.
function runColliderPushCase() {
	const PATCH = Fluid.gpu.fields.PATCH;
	const C = Fluid.core.colliders;
	const cfg2 = Object.assign({}, cfg, { TRef: 293 });
	const table = C.create();
	// a small body mid-field: it takes 4 cells, each with free neighbours on both sides
	C.add(table, cfg2, { shape: "box", x: 20.5, y: 14.5, hw: 1.5, hh: 1.5 });
	const cells = texW * texH;
	const owner = new Uint8Array(cells), prev = new Uint8Array(cells);
	const rect = C.padRect(cfg2, C.rasterize(table, cfg2, owner, {}));
	// one cell of the body was already solid last frame, so it must not push again
	prev[(20) + 14 * texW] = 1;
	const geo = new Float32Array(PATCH * PATCH * 4);
	const taken = C.geometry(cfg2, owner, prev, rect, geo, PATCH);
	const pw = rect.x1 - rect.x0 + 1, ph = rect.y1 - rect.y0 + 1;
	cfg.paintX = rect.x0; cfg.paintY = rect.y0; cfg.paintW = pw; cfg.paintH = ph;
	// a field shaped so every branch is live: dye and heat, a body cell holding some,
	// free cells next to nothing
	const srcM = new Float32Array(cells * 4), srcT = new Float32Array(cells * 4);
	for (let y = 1; y < texH - 1; y++) for (let x = 1; x < texW - 1; x++) {
		const i = (x + y * texW) * 4;
		const m = 0.5 + 0.25 * Math.sin(x * 0.7) * Math.cos(y * 0.4);
		srcM[i] = 0.2 * m; srcM[i + 1] = 0.3 * m; srcM[i + 2] = 0.5 * m; srcM[i + 3] = m;
		srcT[i] = 800 * m; srcT[i + 1] = 300 + 20 * Math.sin(x * 0.2);
	}
	const cpuM = Float32Array.from(srcM), cpuT = Float32Array.from(srcT);
	const scratch = C.makeScratch(cells);
	C.displace(cfg2, geo, rect, cpuM, cpuT, scratch, PATCH);
	const pa = Fluid.core.config.makeParamsArray();
	Fluid.core.config.writeParams(pa, cfg);
	const uniform = { uniform: pa.buffer.slice(0) };
	const rgba = a => ({ texture: [a.buffer], descriptor: { size: [texW, texH], format: "rgba32float" } });
	const patchTex = { texture: [geo.buffer], descriptor: { size: [PATCH, PATCH], format: "rgba32float" } };
	const dstM = new Float32Array(srcM.length), dstT = new Float32Array(srcT.length);
	const wx = Math.ceil(texW / 8), wy = Math.ceil(texH / 8);
	new WgslExec(WgslParser.Parse(PS + sh.colliderPushMass))
		.dispatchWorkgroups("main", [wx, wy], { 0: { 0: uniform, 1: rgba(srcM), 2: rgba(dstM), 3: patchTex } });
	new WgslExec(WgslParser.Parse(PS + sh.colliderPushTherm))
		.dispatchWorkgroups("main", [wx, wy], { 0: { 0: uniform, 1: rgba(srcT), 2: rgba(dstT), 3: patchTex } });
	const maxDiff = (a, b) => {
		let d = 0;
		for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
		return d;
	};
	const dm = maxDiff(dstM, cpuM), dt = maxDiff(dstT, cpuT);
	check("push exec: dye and heat match the cpu oracle cell for cell",
		dm < 1e-5 && dt < 1e-5 && taken > 0, "mass=" + dm.toExponential(2) + " E=" + dt.toExponential(2) + " taken=" + taken);
	const sum = (f, stride) => {
		let t = 0;
		for (let i = 0; i < f.length; i += 4) t += f[i + stride];
		return t;
	};
	const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-9);
	check("push exec: the dye total and the energy total are unchanged",
		rel(sum(dstM, 3), sum(srcM, 3)) < 1e-6 && rel(sum(dstT, 0), sum(srcT, 0)) < 1e-6,
		"m " + rel(sum(dstM, 3), sum(srcM, 3)).toExponential(2) + " E " + rel(sum(dstT, 0), sum(srcT, 0)).toExponential(2));
	// the taken cells are empty and read the reference temperature; the ghost ring is
	// copied, not sourced
	let body = 0, ghost = 0;
	for (let y = rect.y0; y <= rect.y1; y++) for (let x = rect.x0; x <= rect.x1; x++) {
		if (geo[((x - rect.x0) + (y - rect.y0) * PATCH) * 4 + 2] < 0.5) continue;
		const i = (x + y * texW) * 4;
		body = Math.max(body, Math.abs(dstM[i + 3]), Math.abs(dstT[i]), Math.abs(dstT[i + 1] - cfg.TRef));
	}
	for (let y = 0; y < texH; y++) for (let x = 0; x < texW; x++) {
		if (x >= 1 && x <= texW - 2 && y >= 1 && y <= texH - 2) continue;
		const i = (x + y * texW) * 4;
		ghost = Math.max(ghost, Math.abs(dstM[i + 3] - srcM[i + 3]), Math.abs(dstT[i] - srcT[i]));
	}
	check("push exec: a taken cell is emptied and reads the reference temperature",
		body === 0 && ghost === 0, "body=" + body + " ghost=" + ghost);
}

runProbeCase();
runEmitterCase();
runColliderCase();
runColliderPushCase();

console.log(failures ? "FAILURES: " + failures : "ALL WGSL-EXEC CHECKS PASSED");
process.exit(failures ? 1 : 0);
