"use strict";
// Portable CPU, UI-contract and GPU-plumbing suites. The optional real-device
// parity run stays opt-in because it needs an installed WebGPU backend.
const { spawnSync } = require("node:child_process");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const QUICK_SUITES = [
	"eos.js",
	"sharpness.js",
	"transport-roundoff.js",
	"gpu-memory.js",
	"gpu-mock.js",
	"profiler.js",
	"browser-tests.js",
	"hud.js",
	"input.js",
	"controls.js",
	"ui.js",
	"performance.js"
];
const FULL_SUITES = [
	"conformance.js",
	"sharpness.js",
	"convection.js",
	"phase.js",
	"solid-growth.js",
	"eos.js",
	"buoy.js",
	"multigrid.js",
	"transport-roundoff.js",
	"gpu-memory.js",
	"gpu-mock.js",
	"profiler.js",
	"browser-tests.js",
	"hud.js",
	"input.js",
	"controls.js",
	"ui.js",
	"performance.js"
];
const args = process.argv.slice(2);
const withGpu = args.includes("--gpu");
const quick = args.includes("--quick");
const unknown = args.filter(arg => arg !== "--gpu" && arg !== "--quick");
if (unknown.length) {
	console.error("unknown option: " + unknown.join(" "));
	console.error("usage: node experiments/run-all.js [--quick] [--gpu]");
	process.exit(2);
}

const profile = quick ? QUICK_SUITES : FULL_SUITES;
const suites = withGpu ? profile.concat("gpu-parity.mjs") : profile;
const runStarted = process.hrtime.bigint();
const failures = [];
function elapsedMs(start) {
	return Number(process.hrtime.bigint() - start) / 1e6;
}
function elapsedText(ms) {
	return (ms / 1000).toFixed(2) + " s";
}

console.log("Fluid test suite · " + (quick ? "quick" : "full") + " profile · " + suites.length + " suites"
	+ (quick ? " (use default profile for the longer numerical histories)" : "")
	+ (withGpu ? " (including optional WebGPU parity)" : " (use --gpu for real-device parity)"));
for (let i = 0; i < suites.length; i++) {
	const name = suites[i];
	const file = path.join(__dirname, name);
	const started = process.hrtime.bigint();
	console.log("\n=== [" + (i + 1) + "/" + suites.length + "] " + name + " ===");
	const result = spawnSync(process.execPath, [file], {
		cwd: ROOT,
		stdio: "inherit"
	});
	const ms = elapsedMs(started);
	if (result.error) {
		failures.push(name + " (" + result.error.message + ")");
		console.log("FAIL " + name + " · " + elapsedText(ms));
		continue;
	}
	if (result.status !== 0) {
		const reason = result.signal ? "signal " + result.signal : "exit " + result.status;
		failures.push(name + " (" + reason + ")");
		console.log("FAIL " + name + " · " + reason + " · " + elapsedText(ms));
		continue;
	}
	console.log("PASS " + name + " · " + elapsedText(ms));
}

const totalMs = elapsedMs(runStarted);
if (failures.length) {
	console.log("\nFAILED " + (suites.length - failures.length) + "/" + suites.length
		+ " suites · " + elapsedText(totalMs));
	for (const failure of failures) console.log("  " + failure);
	process.exitCode = 1;
} else {
	console.log("\nPASS all " + suites.length + " suites · " + elapsedText(totalMs));
}
