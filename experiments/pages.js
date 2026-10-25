"use strict";
// Page wiring check (headless): every HTML runner must load files that exist and
// that define the Fluid.* namespaces its scripts use, and loading them in document
// order must not throw. Classic <script defer> tags execute in document order, so a
// module that is renamed, moved or dropped from one page otherwise fails only in a
// browser - a case no numeric suite can see.
// run: node experiments/pages.js
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { performance } = require("node:perf_hooks");

const ROOT = path.resolve(__dirname, "..");
const SKIP_DIRS = new Set([".git", "node_modules", "archive", "deferred", "experiments/logs"]);
let failures = 0;
function check(name, ok, detail) {
	console.log((ok ? "PASS " : "FAIL ") + name + (detail ? "  " + detail : ""));
	if (!ok) failures++;
}

// "Fluid.core.save" -> "core.save"; container writes such as
// "Fluid.core = Fluid.core || {}" have one level and are not symbols.
const SYMBOL = /\bFluid\.([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)/g;
const DEFINE = /\bFluid\.([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)\s*=(?!=)/g;
function collect(text, re) {
	const out = new Set();
	re.lastIndex = 0;
	let m;
	while ((m = re.exec(text))) out.add(m[1] + "." + m[2]);
	return out;
}
function readFile(rel) {
	return fs.readFileSync(path.join(ROOT, rel), "utf8");
}
function walk(dir, match, out) {
	for (const entry of fs.readdirSync(path.join(ROOT, dir), { withFileTypes: true })) {
		const rel = dir ? dir + "/" + entry.name : entry.name;
		if (entry.isDirectory()) {
			if (!SKIP_DIRS.has(rel) && !entry.name.startsWith(".")) walk(rel, match, out);
		} else if (match(entry.name)) out.push(rel);
	}
	return out;
}

// the defining module of every namespace in src/, so a missing import can name it
const definers = new Map();
for (const file of walk("src", name => name.endsWith(".js"), [])) {
	for (const sym of collect(readFile(file), DEFINE)) {
		if (!definers.has(sym)) definers.set(sym, []);
		definers.get(sym).push(file);
	}
}

const started = performance.now();
const pages = walk("", name => name.endsWith(".html"), []);
check("html runners found", pages.length > 0, pages.join(", "));
for (const page of pages) {
	const html = readFile(page);
	const dir = path.dirname(page);
	const scripts = [...html.matchAll(/<script\b[^>]*\bsrc\s*=\s*"([^"]+)"/g)].map(m => m[1]);
	const inline = [...html.matchAll(/<script(?![^>]*\bsrc\s*=)[^>]*>([\s\S]*?)<\/script>/g)]
		.map(m => m[1]).join("\n");

	// 1. every referenced file exists
	const missing = scripts.filter(src => !fs.existsSync(path.resolve(ROOT, dir, src)));
	check(page + ": every <script src> resolves", missing.length === 0,
		missing.length ? "missing " + missing.join(", ") : scripts.length + " scripts present");

	// 2. every namespace a loaded script or the inline script uses is defined by a
	//    loaded script (page-local definitions are covered by the load check below)
	const files = scripts.map(src => {
		const rel = path.relative(ROOT, path.resolve(ROOT, dir, src));
		return { rel, text: fs.existsSync(path.join(ROOT, rel)) ? readFile(rel) : "" };
	});
	const pageDefines = new Set();
	for (const entry of files) for (const sym of collect(entry.text, DEFINE)) pageDefines.add(sym);
	const broken = [];
	for (const user of files.concat([{ rel: page + "#inline", text: inline }])) {
		for (const sym of collect(user.text, SYMBOL)) {
			if (pageDefines.has(sym)) continue;
			const owns = definers.get(sym);
			if (!owns || owns.includes(user.rel)) continue;
			broken.push(sym + " needs " + owns[0] + " (used by " + user.rel + ")");
		}
	}
	check(page + ": all used namespaces are loaded", broken.length === 0,
		broken.length ? broken.join("; ") : "closure complete");

	// 3. loading the scripts in document order must not throw, and must leave the
	//    entry points the page itself calls in place
	const sandbox = makeSandbox();
	let loadError = "";
	for (const entry of files) {
		if (!entry.text) continue;
		try {
			vm.runInContext(entry.text, sandbox, { filename: entry.rel });
		} catch (err) {
			loadError = entry.rel + ": " + err.message;
			break;
		}
	}
	check(page + ": scripts load in a stub browser", loadError === "", loadError || "ok");
	const called = collect(inline, SYMBOL);
	const absent = [...called].filter(sym => !sandboxDefines(sandbox, sym));
	check(page + ": inline entry points exist", absent.length === 0,
		absent.length ? "missing " + absent.join(", ") : called.size + " symbols");
}
// the api-test page is the only runner that executes a suite itself: drive its DOM
// reporting with a canned result (engine.js runs the real 20-cycle suite), so a renamed
// result field or a missing table cannot regress unnoticed.
const API_TEST = "experiments/api-test.html";
async function runApiTestPage() {
	const html = readFile(API_TEST);
	const scripts = [...html.matchAll(/<script\b[^>]*\bsrc\s*=\s*"([^"]+)"/g)].map(m => m[1]);
	const inline = [...html.matchAll(/<script(?![^>]*\bsrc\s*=)[^>]*>([\s\S]*?)<\/script>/g)]
		.map(m => m[1]).join("\n");
	async function render(ok) {
		const sandbox = makeSandbox();
		for (const src of scripts) {
			const rel = path.relative(ROOT, path.resolve(ROOT, path.dirname(API_TEST), src));
			vm.runInContext(readFile(rel), sandbox, { filename: rel });
		}
		const result = {
			ok,
			passed: ok ? 2 : 1,
			total: 2,
			checks: [{ ok: true, name: "first check", detail: "d1" }, { ok, name: "second check", detail: "d2" }],
			cycles: [{
				cycle: 0, aliveTexDuring: 1, aliveBufBeforeSave: 2, aliveBufAfterSave: 2,
				texAfterDestroy: 0, bufAfterDestroy: 0, phaseBatches: 0
			}],
			totals: {
				texCreated: 3, texAlive: 0, bufCreated: 4, bufAlive: 0,
				validationErrors: 0, uncapturedErrors: 0, heapDelta: null
			}
		};
		sandbox.Fluid.gpu.device = { init: async () => ({ device: {}, context: {}, format: "bgra8unorm" }) };
		sandbox.Fluid.tests.apiTest = { runLifecycleSuite: async () => result };
		vm.runInContext(inline, sandbox, { filename: API_TEST + "#inline" });
		sandbox.dispatch("DOMContentLoaded");
		await new Promise(resolve => setImmediate(resolve));
		await new Promise(resolve => setImmediate(resolve));
		return sandbox;
	}
	const pass = await render(true);
	check(API_TEST + ": success result renders its tables",
		pass.elements("checksBody").children.length === 2
		&& pass.elements("cyclesBody").children.length === 1
		&& pass.elements("status").textContent === "PASS"
		&& pass.elements("summary").innerHTML.includes("ALL CHECKS PASSED")
		&& pass.elements("summary").innerHTML.includes("2/2")
		&& pass.elements("summary").innerHTML.includes("texCreated=3")
		&& pass.elements("summary").innerHTML.includes("validationErrors=0"),
		"status=" + pass.elements("status").textContent);
	const fail = await render(false);
	check(API_TEST + ": failure result is labelled bad",
		fail.elements("status").textContent === "FAIL"
		&& fail.elements("status").className === "bad"
		&& fail.elements("summary").innerHTML.includes("CHECKS FAILED"),
		"status=" + fail.elements("status").textContent + " class=" + fail.elements("status").className);
}
function finish() {
	console.log((failures ? "FAIL " : "PASS ") + "page wiring · " + pages.length + " pages · "
		+ ((performance.now() - started) / 1000).toFixed(2) + " s");
	process.exitCode = failures ? 1 : 0;
}
if (!pages.includes(API_TEST)) {
	finish();
} else {
	runApiTestPage().then(finish).catch(err => {
		console.log("FAIL " + API_TEST + ": page flow threw " + err.message);
		process.exitCode = 1;
	});
}

function sandboxDefines(sandbox, sym) {
	let node = sandbox.Fluid;
	for (const part of sym.split(".")) {
		if (!node) return false;
		node = node[part];
	}
	return node !== undefined && node !== null;
}
function makeSandbox() {
	const noop = () => {};
	const listeners = Object.create(null);
	const elements = Object.create(null);
	const element = () => ({
		style: {}, dataset: {}, classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
		listeners: Object.create(null), children: [],
		addEventListener(type, fn) { (this.listeners[type] || (this.listeners[type] = [])).push(fn); },
		removeEventListener: noop,
		appendChild(child) { this.children.push(child); return child; },
		removeChild: noop, setAttribute: noop, removeAttribute: noop, focus: noop, blur: noop, click: noop,
		getContext: () => null, querySelector: () => null, querySelectorAll: () => [],
		getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
		width: 0, height: 0, value: "", checked: false, textContent: "", innerHTML: "", className: ""
	});
	const byId = id => elements[id] || (elements[id] = element());
	const dispatch = type => {
		let result;
		for (const fn of listeners[type] || []) result = fn({ type });
		return result;
	};
	const sandbox = {
		console: { log: noop, warn: noop, error: noop, info: noop, debug: noop },
		setTimeout, clearTimeout, setInterval, clearInterval, queueMicrotask,
		performance, TextEncoder, TextDecoder, URL, URLSearchParams, Blob: function Blob() {},
		requestAnimationFrame: () => 0, cancelAnimationFrame: noop,
		addEventListener(type, fn) { (listeners[type] || (listeners[type] = [])).push(fn); },
		removeEventListener: noop, dispatchEvent: noop, dispatch, elements: byId,
		getComputedStyle: () => ({ getPropertyValue: () => "" }),
		matchMedia: () => ({ matches: false, addEventListener: noop, removeEventListener: noop }),
		localStorage: { getItem: () => null, setItem: noop, removeItem: noop },
		location: { search: "", hash: "", href: "file:///fluid/", pathname: "/fluid/" },
		navigator: { userAgent: "node", gpu: undefined, maxTouchPoints: 0 },
		document: {
			hidden: false, readyState: "complete", addEventListener: noop, removeEventListener: noop,
			getElementById: byId, createElement: () => element(),
			querySelector: () => null, querySelectorAll: () => [],
			documentElement: element(), body: element()
		},
		devicePixelRatio: 1,
		GPUBufferUsage: { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128 },
		GPUTextureUsage: { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 },
		GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
		GPUMapMode: { READ: 1, WRITE: 2 }
	};
	sandbox.window = sandbox;
	sandbox.globalThis = sandbox;
	sandbox.self = sandbox;
	return vm.createContext(sandbox);
}
