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
	"src/core/multigrid.js", "src/core/emitters.js", "src/core/solver.js", "src/core/probes.js", "src/gpu/shaders.js"
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

runProbeCase();

// ---- 5. 0.7.0 t3 emitter kernels: forces (momentum + rhoMean) and applyEmitters (all 5 scalar/thermal kinds) ----
function runEmitterExecCases() {
	const en = 16, etw = en + 2, eth = en + 2, edt = 1 / 120;
	const ecfg = Object.assign({}, Fluid.core.config.defaults, {
		N: en, texW: etw, texH: eth, dt: edt,
		scenario: "blobs", gravityX: 10, gravityY: 180
	});
	const es = Fluid.core.solver.create(ecfg);
	// paint one static wall cell inside an emitter disc and set a low-mass cell to exercise full-vacuum drain
	es.flags[(5 + 5 * etw) * 4] = 255;
	const mCur0 = grid.curBuf(es.mass), tCur0 = grid.curBuf(es.therm);
	const vacTestIdx = (12 + 12 * etw) * 4;
	mCur0[vacTestIdx] = 0.001; mCur0[vacTestIdx + 1] = 0.002; mCur0[vacTestIdx + 2] = 0.0005; mCur0[vacTestIdx + 3] = 0.003;
	tCur0[vacTestIdx] = Fluid.core.solver.energyFromT(0.003, ecfg.TAmb, ecfg);

	es.addEmitter({ kind: "dye", x: 5, y: 5, radius: 3.5, value: 0.8, dye: [0.9, 0.25, 0.1] });
	es.addEmitter({ kind: "drain", x: 12, y: 12, radius: 3.5, value: 1.5 });
	es.addEmitter({ kind: "heat", x: 5, y: 12, radius: 3.5, value: 120 });
	es.addEmitter({ kind: "cold", x: 12, y: 5, radius: 3.5, value: 110 });
	es.addEmitter({ kind: "freeze", x: 8, y: 8, radius: 3.0, value: 320 });
	es.addEmitter({ kind: "momentum", x: 8, y: 8, radius: 4.5, vx: 160, vy: -95 });

	const wx = Math.ceil(etw / 8), wy = Math.ceil(eth / 8);
	const rgba = a => ({ texture: [a.buffer], descriptor: { size: [etw, eth], format: "rgba32float" } });
	const rg = a => ({ texture: [a.buffer], descriptor: { size: [etw, eth], format: "rg32float" } });
	const u8x4 = a => ({ texture: [a.buffer], descriptor: { size: [etw, eth], format: "rgba8uint" } });

	// 5a. applyEmitters vs cpu.applyEmitters across all 5 scalar/thermal kinds
	const massBefore = Float32Array.from(grid.curBuf(es.mass));
	const thermBefore = Float32Array.from(grid.curBuf(es.therm));
	const flagsSnap = Uint8Array.from(es.flags);
	es.applyEmitters(edt);
	const cpuMass = Float32Array.from(grid.curBuf(es.mass));
	const cpuTherm = Float32Array.from(grid.curBuf(es.therm));

	const pa = Fluid.core.config.makeParamsArray();
	Fluid.core.config.writeParams(pa, ecfg);
	const uniform = { uniform: pa.buffer.slice(0) };
	const emitBuf = es.emitterTable.upload.buffer.slice(0);
	const gpuMassDst = new Float32Array(massBefore.length);
	const gpuThermDst = new Float32Array(thermBefore.length);

	new WgslExec(WgslParser.Parse(PS + sh.applyEmitters))
		.dispatchWorkgroups("main", [wx, wy], {
			0: {
				0: uniform,
				1: emitBuf,
				2: rgba(Float32Array.from(massBefore)),
				3: rgba(Float32Array.from(thermBefore)),
				4: u8x4(Uint8Array.from(flagsSnap)),
				5: rgba(gpuMassDst),
				6: rgba(gpuThermDst)
			}
		});

	let maxMassErr = 0, maxThermErr = 0;
	for (let i = 0; i < cpuMass.length; i++) {
		const dm = Math.abs(gpuMassDst[i] - cpuMass[i]);
		if (dm > maxMassErr) maxMassErr = dm;
		const dtErr = Math.abs(gpuThermDst[i] - cpuTherm[i]);
		if (dtErr > maxThermErr) maxThermErr = dtErr;
	}
	check("applyEmitters exec: WGSL matches CPU across dye/drain/heat/cold/freeze + wall/vacuum guards",
		maxMassErr < 1e-6 && maxThermErr < 1e-3 && gpuMassDst[vacTestIdx + 3] === 0,
		"maxMassErr=" + maxMassErr.toExponential(2) + " maxThermErr=" + maxThermErr.toExponential(2));

	// 5b. statsReduceB exec: copies emitterLedger[0..9] into stats[36..45] alongside Kahan cell partials
	// (emitterTally uses @builtin(local_invocation_index) + workgroupBarrier(), which WgslExec does not
	// coroutine-schedule — same reason statsReduceA/poissonReduceA/buoySample are parse-checked in section 1;
	// emitterTally shares evalScalarCell verbatim with applyEmitters via EMITTER_HELPERS).
	const gpuEmitLedger = Float32Array.from([
		es.ledger.injectedM, es.ledger.injectedDr, es.ledger.injectedDg, es.ledger.injectedDb, es.ledger.injectedE,
		es.ledger.removedM, es.ledger.removedDr, es.ledger.removedDg, es.ledger.removedDb, es.ledger.removedE,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	]);
	const statsPart = new Float32Array(4 * 40);
	const statsOut = new Float32Array(Fluid.core.stats.STATS_LEN);
	const dimsU = { uniform: new Uint32Array([4 * 40, 0, 0, 0]).buffer };
	new WgslExec(WgslParser.Parse(sh.statsReduceB))
		.dispatchWorkgroups("main", [1], {
			0: {
				0: statsPart.buffer,
				1: statsOut.buffer,
				2: dimsU,
				3: gpuEmitLedger.buffer
			}
		});
	let statsCopyExact = true;
	for (let k = 0; k < 10; k++) {
		if (statsOut[36 + k] !== gpuEmitLedger[k]) statsCopyExact = false;
	}
	check("statsReduceB exec: copies emitterLedger[0..9] into stats[36..45] intact",
		statsCopyExact && statsOut[36] > 0 && statsOut[41] > 0,
		"injM=" + statsOut[36].toFixed(4) + " remM=" + statsOut[41].toFixed(4));

	// 5c. forces vs cpu.forces with momentum emitter + live emitterLedger rhoMean offset
	const emitLedger = new Float32Array(Fluid.core.stats.EMIT_LEDGER_LEN);
	emitLedger[0] = es.ledger.injectedM;
	emitLedger[5] = es.ledger.removedM;
	ecfg.rhoBase = es.ledger.initialM / (en * en);
	ecfg.rhoMean = (es.ledger.initialM + es.ledger.injectedM - es.ledger.removedM) / (en * en);
	Fluid.core.config.writeParams(pa, ecfg);
	const uniformForces = { uniform: pa.buffer.slice(0) };

	const velBefore = Float32Array.from(grid.curBuf(es.vel));
	es.forces(edt);
	const cpuVel = Float32Array.from(grid.curBuf(es.vel));
	const gpuVelDst = new Float32Array(velBefore.length);

	new WgslExec(WgslParser.Parse(PS + sh.forces))
		.dispatchWorkgroups("main", [wx, wy], {
			0: {
				0: uniformForces,
				1: rg(Float32Array.from(velBefore)),
				2: rg(gpuVelDst),
				3: rgba(Float32Array.from(cpuMass)),
				4: rgba(Float32Array.from(cpuTherm)),
				5: u8x4(Uint8Array.from(es.flags)),
				6: es.emitterTable.upload.buffer.slice(0),
				7: emitLedger.buffer.slice(0)
			}
		});

	let maxVelErr = 0;
	for (let i = 0; i < cpuVel.length; i++) {
		const dv = Math.abs(gpuVelDst[i] - cpuVel[i]);
		if (dv > maxVelErr) maxVelErr = dv;
	}
	check("forces exec: WGSL momentum emitter + emitterLedger rhoMean matches CPU forces",
		maxVelErr < 1e-5,
		"maxVelErr=" + maxVelErr.toExponential(2));
}

runEmitterExecCases();

console.log(failures ? "FAILURES: " + failures : "ALL WGSL-EXEC CHECKS PASSED");
process.exit(failures ? 1 : 0);
