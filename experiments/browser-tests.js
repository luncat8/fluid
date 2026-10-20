"use strict";
// Exercise the browser report page's run/lock/copy flow with a recording GPU facade.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { performance } = require("node:perf_hooks");

const html = fs.readFileSync(path.join(__dirname, "browser-tests.html"), "utf8");
const inline = /<script>([\s\S]*?)<\/script>/.exec(html);
assert.ok(inline, "browser harness inline script exists");

class Element {
	constructor(id) {
		this.id = id;
		this.listeners = Object.create(null);
		this.style = {};
		this.checked = id.startsWith("case-");
		this.value = id === "grid" ? "512" : id === "frames" ? "1" : "";
		this.textContent = "";
		this.disabled = false;
	}
	addEventListener(type, fn) {
		(this.listeners[type] || (this.listeners[type] = [])).push(fn);
	}
	click() {
		let result;
		for (const fn of this.listeners.click || []) result = fn({ type: "click", target: this });
		return result;
	}
}

const ids = [
	"out", "status", "run", "copy", "save", "c", "error-panel", "grid", "frames",
	"case-blobs", "case-dam", "case-still", "case-convection", "case-draw", "copy-feedback"
];
const elements = Object.fromEntries(ids.map(id => [id, new Element(id)]));
let copied = "";
let paintCalls = 0;
const values = { m: 10, dr: 2, dg: 3, db: 4, E: 500 };
const stats = {
	frame: 0,
	statsFrame: 0,
	totals: Object.assign({}, values),
	expected: Object.assign({}, values),
	mins: { m: 0.3, dr: 0, dg: 0, db: 0, E: 300 },
	maxM: 0.8,
	cntNeg: 0,
	cntOver: 0,
	motion: { maxVel: 0, maxDiv: 0.1, pMin: -1, pMax: 1 }
};
const device = {
	queue: { onSubmittedWorkDone: () => Promise.resolve() },
	pushErrorScope() {},
	popErrorScope: () => Promise.resolve(null),
	lost: Promise.resolve({ reason: "destroyed" })
};
const config = {
	defaults: {
		N: 512, texW: 514, texH: 514, frameDt: 1 / 60, dt: 1 / 60,
		pressureIters: 60, fctLimiter: 1, gravityY: 300, substepMax: 24,
		substepsAuto: true, cflTarget: 0.8, vMax: 8, mgCoarsen: 4, mgCycles: 2,
		mgPre: 4, mgPost: 8, mouseVX: 0, mouseVY: 0
	},
	substepCount: () => 1,
	velocityCap: () => 480
};
const Fluid = {
	core: {
		config,
		stats: { driftRel: () => 0 }
	},
	gpu: {
		device: {
			init: async () => ({ device, adapter: { info: {} }, context: {}, format: "rgba8unorm" })
		},
		solver: {
			create(_device, _canvas, _context, _format, cfg) {
				let frame = 0;
				return {
					pool: { mgS: 4 },
					reset() { frame = 0; stats.frame = 0; stats.statsFrame = 0; },
					frame(dt, _ns, needStats) {
						cfg.dt = dt;
						frame++;
						stats.frame = frame;
						if (needStats) stats.statsFrame = frame;
					},
					getStats() { return stats; },
					paintDye() { paintCalls++; },
					pollReadback: async () => {},
					destroy() {}
				};
			}
		}
	}
};
const context = {
	Fluid,
	document: { getElementById: id => elements[id], createElement: () => new Element("created"), body: { appendChild() {} } },
	navigator: { clipboard: { writeText: async text => { copied = text; } } },
	location: { search: "" },
	performance,
	URLSearchParams,
	setTimeout,
	console: { log() {}, error() {} },
	crossOriginIsolated: true
};
vm.runInNewContext(inline[1], context, { filename: "browser-tests.html:inline" });

(async () => {
	const run = elements.run.click();
	assert.equal(elements.run.disabled, true, "run button locks while the suite is active");
	await run;
	assert.equal(elements.run.disabled, false, "run button unlocks when the suite ends");
	assert.match(elements.out.textContent, /scenarios=blobs,dam,still,convection,draw/);
	assert.match(elements.out.textContent, /Finished 5 scenarios/);
	assert.ok(paintCalls > 0, "scripted draw case adds the deterministic dye stroke");
	assert.match(elements.out.textContent, /crossOriginIsolated=true/);
	assert.equal(elements.copy.disabled, false, "copy is available after the report exists");
	await elements.copy.click();
	assert.equal(copied, elements.out.textContent, "copy carries the complete report");
	assert.equal(elements["copy-feedback"].textContent, "copied");
	const queryElements = Object.fromEntries(ids.map(id => [id, new Element(id)]));
	const queryContext = {
		document: { getElementById: id => queryElements[id] },
		location: { search: "?scenarios=blobs,draw" },
		URLSearchParams,
		console: { log() {}, error() {} }
	};
	vm.runInNewContext(inline[1], queryContext, { filename: "browser-tests.html:query" });
	const selected = ["blobs", "dam", "still", "convection", "draw"]
		.filter(name => queryElements["case-" + name].checked);
	assert.deepEqual(selected, ["blobs", "draw"], "scenario query prefill selects a reproducible subset");
	console.log("PASS browser diagnostics UI: five-case run, scenario query, scripted paint, in-flight lock, report and clipboard copy");
})().catch(error => {
	console.error(error);
	process.exitCode = 1;
});
