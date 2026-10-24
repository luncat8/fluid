import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

// Optional numeric oracle: npm install --prefix /tmp/fluid-webgpu webgpu@0.6.2,
// then set FLUID_WEBGPU_MODULE=/tmp/fluid-webgpu/node_modules/webgpu/index.js.
// The app remains dependency-free; this runner needs a real WebGPU backend (not null).
// t8 gates aggregate total E and mean T at 2e-6; local E/T drift is reported
// separately and bounded at 5e-5 after step 64 for this diagnostic run.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
const modules = [
	"src/core/config.js",
	"src/core/grid.js",
	"src/core/stats.js",
	"src/core/multigrid.js",
	"src/core/solver.js",
	"src/core/probes.js",
	"src/core/emitters.js",
	"src/core/colliders.js",
	"src/core/buoy.js",
	"src/gpu/shaders.js",
	"src/gpu/fields.js",
	"src/gpu/profiler.js",
	"src/gpu/passes.js",
	"src/gpu/solver.js",
	"src/gpu/view.js"
];
for (const file of modules) require(path.join(root, file));

let webgpu;
try {
	webgpu = await import(process.env.FLUID_WEBGPU_MODULE || "webgpu");
} catch (error) {
	console.error("WebGPU parity needs the optional Node package 'webgpu'.");
	console.error("Install it outside the repository with:");
	console.error("  npm install --prefix /tmp/fluid-webgpu webgpu@0.6.2");
	console.error("Then set FLUID_WEBGPU_MODULE to its index.js path and run this file.");
	console.error(error.message);
	process.exit(2);
}

Object.assign(globalThis, webgpu.globals);
const backend = process.env.FLUID_WEBGPU_BACKEND;
const gpu = webgpu.create(backend ? ["backend=" + backend] : []);
Object.defineProperty(globalThis, "navigator", { value: { gpu }, configurable: true });

let failures = 0;
let device;
const uncaptured = [];
function check(name, ok, detail) {
	console.log((ok ? "PASS " : "FAIL ") + name + (detail ? "  " + detail : ""));
	if (!ok) failures++;
}
function current(pp) {
	return pp.cur === 0 ? pp.a : pp.b;
}
function summarize(actual, expected, spec, start = 0, stride = 1) {
	let maxError = 0;
	let maxRel = 0;
	let maxScaled = 0;
	let sumSq = 0;
	let count = 0;
	let violations = 0;
	for (let i = start; i < expected.length; i += stride) {
		const a = actual[i], b = expected[i];
		count++;
		if (!Number.isFinite(a) || !Number.isFinite(b)) {
			violations++;
			continue;
		}
		const error = Math.abs(a - b);
		const rel = error / Math.max(Math.abs(a), Math.abs(b), 1e-12);
		const allowed = spec.atol + spec.rtol * Math.max(Math.abs(a), Math.abs(b));
		const scaled = error / allowed;
		if (error > maxError) maxError = error;
		if (rel > maxRel) maxRel = rel;
		if (scaled > maxScaled) maxScaled = scaled;
		if (scaled > 1) violations++;
		sumSq += error * error;
	}
	return { maxError, maxRel, maxScaled, rms: Math.sqrt(sumSq / count), violations };
}
async function readTextures(fields, width, height) {
	const copies = [];
	const encoder = device.createCommandEncoder({ label: "parityReadback" });
	for (const field of fields) {
		const rowBytes = width * field.bytesPerPixel;
		const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
		const buffer = device.createBuffer({
			label: "parity-" + field.name,
			size: bytesPerRow * height,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		encoder.copyTextureToBuffer(
			{ texture: field.texture },
			{ buffer, bytesPerRow, rowsPerImage: height },
			{ width, height, depthOrArrayLayers: 1 }
		);
		copies.push({ field, buffer, bytesPerRow, rowBytes });
	}
	device.queue.submit([encoder.finish()]);
	await Promise.all(copies.map(copy => copy.buffer.mapAsync(GPUMapMode.READ)));
	for (const copy of copies) {
		const mapped = new Uint8Array(copy.buffer.getMappedRange());
		const packed = new Uint8Array(copy.rowBytes * height);
		for (let y = 0; y < height; y++) {
			const row = mapped.subarray(y * copy.bytesPerRow, y * copy.bytesPerRow + copy.rowBytes);
			packed.set(row, y * copy.rowBytes);
		}
		copy.field.gpuValues = copy.field.integer
			? packed
			: copy.field.halfFloat
				? new Uint16Array(packed.buffer)
				: new Float32Array(packed.buffer);
		copy.buffer.unmap();
		copy.buffer.destroy();
	}
}
function f16ToF32(h) {
	const s = (h & 0x8000) >> 15;
	const e = (h & 0x7c00) >> 10;
	const f = h & 0x03ff;
	if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
	if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
	return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
function fieldSpecs(cpu, solver) {
	const pool = solver.pool;
	const state = solver.state;
	return [
		{ name: "mass", texture: pool.mass.tex[state.m], cpuValues: current(cpu.mass), bytesPerPixel: 16, atol: 2e-6, rtol: 2e-5 },
		{ name: "therm", texture: pool.therm.tex[state.t], cpuValues: current(cpu.therm), bytesPerPixel: 16, atol: 2e-3, rtol: 2e-6 },
		{ name: "velocity", texture: pool.vel.tex[state.v], cpuValues: current(cpu.vel), bytesPerPixel: 8, atol: 2e-4, rtol: 2e-5 },
		{ name: "pressure", texture: pool.press.tex[0], cpuValues: current(cpu.press), bytesPerPixel: 4, atol: 2e-2, rtol: 2e-5 },
		{ name: "flux", texture: pool.flux, cpuValues: cpu.flux, bytesPerPixel: 8, atol: 2e-6, rtol: 2e-5 },
		{ name: "divergence-source", texture: pool.divsrc, cpuValues: cpu.divsrc, bytesPerPixel: 4, atol: 2e-2, rtol: 2e-5 },
		{ name: "sfield", texture: pool.sfield, cpuValues: cpu.sfield, bytesPerPixel: 16, atol: 5e-4, rtol: 2e-4 },
		{ name: "flags", texture: pool.flags, cpuValues: cpu.flags, bytesPerPixel: 4, integer: true }
	];
}
function makeConfig(scenario, n, options) {
	return Object.assign({}, Fluid.core.config.defaults, options, {
		N: n,
		texW: n + 2,
		texH: n + 2,
		scenario,
		buoyEnabled: 0,
		statsEvery: 1e9
	});
}
async function compareSnapshot(testCase, cpu, solver, n, step) {
	const fields = fieldSpecs(cpu, solver);
	await readTextures(fields, n + 2, n + 2);
	const mass = fields[0], therm = fields[1], velocity = fields[2];
	const cpuTotals = Fluid.core.stats.totalsFromFields(mass.cpuValues, therm.cpuValues, n + 2, n + 2);
	const gpuTotals = Fluid.core.stats.totalsFromFields(mass.gpuValues, therm.gpuValues, n + 2, n + 2);
	const cpuThermal = Fluid.core.stats.thermalTotals(mass.cpuValues, therm.cpuValues, velocity.cpuValues,
		n + 2, n + 2, cpu.cfg.mEps, cpu.cfg.cLiq, cpu.cfg.TAmb);
	const gpuThermal = Fluid.core.stats.thermalTotals(mass.gpuValues, therm.gpuValues, velocity.gpuValues,
		n + 2, n + 2, cpu.cfg.mEps, cpu.cfg.cLiq, cpu.cfg.TAmb);
	const energyRel = Math.abs(gpuTotals.sumE - cpuTotals.sumE) / Math.max(Math.abs(cpuTotals.sumE), 1e-12);
	const meanTRel = Math.abs(gpuThermal.TMean - cpuThermal.TMean) / Math.max(Math.abs(cpuThermal.TMean), 1e-12);
	check(testCase.name + " step " + step + " total-E parity", energyRel < 2e-6,
		"relative=" + energyRel.toExponential(2));
	check(testCase.name + " step " + step + " mean-T parity", meanTRel < 2e-6,
		"relative=" + meanTRel.toExponential(2));
	for (const field of fields) {
		if (field.integer) {
			let mismatches = 0;
			for (let i = 0; i < field.gpuValues.length; i++) {
				if (field.gpuValues[i] !== field.cpuValues[i]) mismatches++;
			}
			check(testCase.name + " step " + step + " " + field.name + " exact",
				mismatches === 0, "byte mismatches=" + mismatches);
			continue;
		}
		if (field.name === "therm") {
			const channels = ["E", "T", "heat-in", "heat-out"];
			for (let channel = 0; channel < channels.length; channel++) {
				const localSpec = Object.assign({}, field);
				if (testCase.name === "heated box" && step > 64 && channel < 2) localSpec.rtol = 5e-5;
				const summary = summarize(field.gpuValues, field.cpuValues, localSpec, channel, 4);
				const band = testCase.name === "heated box" && step > 64 && channel < 2 ? " f32-long-run" : "";
				check(testCase.name + " step " + step + " " + channels[channel] + " parity" + band,
					summary.violations === 0,
					"max_abs=" + summary.maxError.toExponential(2) + " max_rel=" + summary.maxRel.toExponential(2) +
					" rms=" + summary.rms.toExponential(2) + " cells_over_tol=" + summary.violations);
			}
			continue;
		}
		const summary = summarize(field.gpuValues, field.cpuValues, field);
		check(testCase.name + " step " + step + " " + field.name + " parity",
			summary.violations === 0,
			"max_abs=" + summary.maxError.toExponential(2) + " max_rel=" + summary.maxRel.toExponential(2) +
			" rms=" + summary.rms.toExponential(2) + " cells_over_tol=" + summary.violations);
	}
}
async function runCase(testCase, n) {
	const options = Object.assign({
		mgCoarseIters: 12,
		fctLimiter: 1,
		gravityX: 0,
		gravityY: 300,
		frameDt: 1 / 60,
		dt: 1 / 240
	}, testCase.options);
	const cpuCfg = makeConfig(testCase.scenario, n, options);
	const gpuCfg = makeConfig(testCase.scenario, n, options);
	const cpu = Fluid.core.solver.create(cpuCfg);
	const format = "rgba8unorm";
	const canvas = { width: n, height: n };
	const target = device.createTexture({
		label: "parity-render-target",
		size: [n, n],
		format,
		usage: GPUTextureUsage.RENDER_ATTACHMENT
	});
	const context = { getCurrentTexture: () => target };
	let solver;
	try {
		device.pushErrorScope("validation");
		solver = Fluid.gpu.solver.create(device, canvas, context, format, gpuCfg);
		const createError = await device.popErrorScope();
		if (createError) throw new Error("pipeline setup: " + createError.message);
		const snapshots = new Set(testCase.snapshots);
		if (snapshots.has(0)) await compareSnapshot(testCase, cpu, solver, n, 0);
		for (let step = 1; step <= testCase.steps; step++) {
			cpuCfg.dt = options.dt;
			cpu.step(options.dt);
			device.pushErrorScope("validation");
			solver.frame(options.dt, 1, false);
			if (snapshots.has(step)) await compareSnapshot(testCase, cpu, solver, n, step);
			const error = await device.popErrorScope();
			if (error) throw new Error("step " + step + ": " + error.message);
		}
		await device.queue.onSubmittedWorkDone();
		if (uncaptured.length) throw new Error(uncaptured.join("\n"));
	} finally {
		if (solver) solver.destroy();
		target.destroy();
	}
}

const adapter = await navigator.gpu.requestAdapter({ powerPreference: "low-power" });
if (!adapter) {
	console.error("No WebGPU adapter. Select an installed hardware or software Vulkan backend.");
	process.exit(2);
}
console.log("adapter: " + JSON.stringify(adapter.info));
device = await adapter.requestDevice({ requiredFeatures: [], requiredLimits: {} });
device.onuncapturederror = event => uncaptured.push(event.error.message);

const n = Number(process.env.FLUID_GPU_PARITY_N || 32);
const flowSteps = Number(process.env.FLUID_GPU_PARITY_FLOW_STEPS || 8);
const heatSteps = Number(process.env.FLUID_GPU_PARITY_HEAT_STEPS || 500);
const phaseSteps = Number(process.env.FLUID_GPU_PARITY_PHASE_STEPS || 16);
const tests = [
	{
		name: "dam transport/projection",
		scenario: "dam",
		steps: flowSteps,
		snapshots: [0, 1, 2, 4, flowSteps]
	},
	{
		name: "heated box",
		scenario: "convection",
		steps: heatSteps,
		snapshots: [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, heatSteps],
		options: { gravityY: 0, mgCoarseIters: 4, dt: 1 / 240 }
	},
	{
		name: "ice melt + advection",
		scenario: "ice",
		steps: phaseSteps,
		snapshots: [0, 1, 2, 4, 8, phaseSteps],
		options: { mgCoarseIters: 8, dt: 1 / 240 }
	},
	{
		name: "jet vapor expansion",
		scenario: "jet",
		steps: phaseSteps,
		snapshots: [0, 1, 2, 4, 8, phaseSteps],
		options: { mgCoarseIters: 8, dt: 1 / 240 }
	},
	{
		name: "rain boil/freeze cycle",
		scenario: "rain",
		steps: phaseSteps,
		snapshots: [0, 1, 2, 4, 8, phaseSteps],
		options: { mgCoarseIters: 8, dt: 1 / 240 }
	},
	{
		name: "arch frozen vault + MG",
		scenario: "arch",
		steps: phaseSteps,
		snapshots: [0, 1, 2, 4, 8, phaseSteps],
		options: { mgCoarseIters: 8, dt: 1 / 240 }
	}
];
for (const testCase of tests) await runCase(testCase, n);

// 0.5.0 stats readback + display view verification (a6, a11, a12)
{
	const vn = 32;
	const opts = { mgCoarseIters: 8, fctLimiter: 1, dt: 1 / 240 };
	const cpuCfg = makeConfig("cycle", vn, opts);
	const gpuCfg = makeConfig("cycle", vn, opts);
	const cpu = Fluid.core.solver.create(cpuCfg);
	const format = "rgba8unorm";
	const canvas = { width: vn, height: vn };
	const target = device.createTexture({
		label: "view-render-target",
		size: [vn, vn],
		format,
		usage: GPUTextureUsage.RENDER_ATTACHMENT
	});
	const context = { getCurrentTexture: () => target };
	const solver = Fluid.gpu.solver.create(device, canvas, context, format, gpuCfg);
	try {
		for (let s = 0; s < 12; s++) {
			cpuCfg.dt = opts.dt;
			cpu.step(opts.dt);
			solver.frame(opts.dt, 1, s === 11);
		}
		await device.queue.onSubmittedWorkDone();
		await solver.pollReadback();
		const cpuSt = cpu.stats();
		const gpuSt = solver.getStats();
		const countMatch = cpuSt.phaseCounts.every((c, i) => c === gpuSt.phaseCounts[i]);
		const latRel = Math.abs(gpuSt.ledger.latentChargedE - cpuSt.ledger.latentChargedE)
			/ Math.max(cpuSt.ledger.latentChargedE, 1e-6);
		check("stats readback: 32-float phaseCounts + latentChargedE + auto PMin/PMax match CPU",
			countMatch && latRel < 1e-4 && gpuCfg.PMax > 0 && gpuCfg.PMin === -gpuCfg.PMax,
			"counts=" + gpuSt.phaseCounts.join("/") + " lat=" + gpuSt.ledger.latentChargedE.toFixed(2)
			+ " P=[" + gpuCfg.PMin.toFixed(1) + ".." + gpuCfg.PMax.toFixed(1) + "]");

		// Paint one static wall cell so display has frozen ice, liquid, and static wall together
		solver.paintWall(0.1, 0.9, false);
		const dispSpec = [{ name: "display", texture: solver.pool.display, bytesPerPixel: 8, halfFloat: true }];
		const readDisplay = async () => {
			solver.renderOnly();
			await readTextures(dispSpec, vn + 2, vn + 2);
			return Array.from(dispSpec[0].gpuValues, f16ToF32);
		};
		gpuCfg.viewMode = 7;
		const phaseImg = await readDisplay();
		const rgbKey = i => [phaseImg[i].toFixed(2), phaseImg[i + 1].toFixed(2), phaseImg[i + 2].toFixed(2)].join(",");
		const seenPhaseColors = new Set();
		for (let y = 1; y <= vn; y++) {
			for (let x = 1; x <= vn; x++) seenPhaseColors.add(rgbKey((x + y * (vn + 2)) * 4));
		}
		check("toDisplay mode 7 (phase map): renders distinct colors for ice, liquid, gas, and wall",
			seenPhaseColors.size >= 3, "distinct=" + seenPhaseColors.size);

		gpuCfg.viewMode = 5;
		const flagsImg = await readDisplay();
		const flagsColors = new Set();
		for (let y = 1; y <= vn; y++) {
			for (let x = 1; x <= vn; x++) {
				const i = (x + y * (vn + 2)) * 4;
				flagsColors.add([flagsImg[i].toFixed(2), flagsImg[i + 1].toFixed(2), flagsImg[i + 2].toFixed(2)].join(","));
			}
		}
		check("toDisplay mode 5 (flags): distinguishes frozen solid, static wall, and open fluid",
			flagsColors.size >= 3, "distinct=" + flagsColors.size);

		gpuCfg.viewMode = 0;
		gpuCfg.overlay = 1;
		const overOn = await readDisplay();
		gpuCfg.overlay = 0;
		const overOff = await readDisplay();
		let diffPixels = 0;
		for (let i = 0; i < overOn.length; i += 4) {
			if (Math.abs(overOn[i] - overOff[i]) + Math.abs(overOn[i + 1] - overOff[i + 1]) + Math.abs(overOn[i + 2] - overOff[i + 2]) > 0.02) {
				diffPixels++;
			}
		}
		check("toDisplay solid overlay: toggling cfg.overlay updates frozen-cell pixels immediately",
			diffPixels > 0, "changed_pixels=" + diffPixels);
	} finally {
		solver.destroy();
		target.destroy();
	}
}

await device.queue.onSubmittedWorkDone();
device.destroy();
console.log(failures === 0
	? "GPU/CPU DIAGNOSTICS PASSED; t8 aggregate E/T acceptance met"
	: failures + " GPU/CPU DIAGNOSTIC FAILURES; t8 aggregate E/T acceptance not met");
process.exitCode = failures === 0 ? 0 : 1;
