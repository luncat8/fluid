"use strict";
const assert = require("node:assert/strict");
const { probe, formats } = require("../src/gpu/device.js");
global.GPUTextureUsage = { TEXTURE_BINDING: 1, STORAGE_BINDING: 2, COPY_SRC: 4, COPY_DST: 8 };

function fixture(options = {}) {
	let created = 0;
	let destroyed = 0;
	let scopes = 0;
	let resolveLost;
	const device = {
		features: new Set(), limits: { maxTextureDimension2D: 8192 },
		lost: new Promise(resolve => { resolveLost = resolve; }),
		pushErrorScope(type) { assert.equal(type, "validation"); scopes++; },
		async popErrorScope() { scopes--; return options.validationError ? { message: "unsupported format" } : null; },
		createTexture(descriptor) {
			assert.equal(descriptor.usage, 15);
			assert.deepEqual(descriptor.size, [2, 2]);
			created++;
			if (options.createThrows) throw Error("texture creation failed");
			return { destroy() { destroyed++; } };
		}
	};
	const adapter = {
		info: { vendor: "test adapter" }, features: new Set(["optional-feature"]),
		async requestDevice(descriptor) {
			assert.deepEqual(descriptor, { requiredFeatures: [], requiredLimits: {} });
			if (options.deviceRejects) throw Error("device denied");
			return device;
		}
	};
	const environment = {
		navigator: { userAgent: "mock; not browser evidence", gpu: {
			async requestAdapter(descriptor) {
				assert.equal(descriptor.powerPreference, "high-performance");
				if (options.adapterRejects) throw Error("adapter denied");
				return options.nullAdapter ? null : adapter;
			}
		} },
		location: { protocol: "file:" }, isSecureContext: true
	};
	return { environment, device, resolveLost, counts: () => ({ created, destroyed, scopes }) };
}

(async () => {
	const good = fixture();
	const events = [];
	const result = await probe(good.environment, message => events.push(message));
	assert.equal(result.report.status, "passed");
	assert.deepEqual(result.report.formats.map(item => item.format), formats);
	assert.deepEqual(good.counts(), { created: 5, destroyed: 5, scopes: 0 });
	assert.deepEqual(result.report.deviceFeatures, []);
	good.device.onuncapturederror({ error: { message: "test validation error" } });
	assert.equal(result.report.status, "failed");
	good.resolveLost({ reason: "unknown", message: "test loss" });
	await Promise.resolve();
	assert.equal(events.length, 2);
	console.log("PASS success, all five formats, resource cleanup, core-only request, uncaptured error, device loss");
	for (const option of ["nullAdapter", "adapterRejects", "deviceRejects", "validationError", "createThrows"]) {
		const test = fixture({ [option]: true });
		const { report } = await probe(test.environment);
		assert.equal(report.status, "failed");
		assert.ok(report.errors.length);
		assert.equal(test.counts().scopes, 0);
		console.log("PASS " + option);
	}
	const missing = fixture();
	delete missing.environment.navigator.gpu;
	const { report } = await probe(missing.environment);
	assert.equal(report.status, "failed");
	assert.match(report.errors[0], /http:\/\/localhost:8000/);
	const intentional = fixture();
	const normal = await probe(intentional.environment);
	intentional.resolveLost({ reason: "destroyed", message: "" });
	await Promise.resolve();
	assert.equal(normal.report.status, "passed");
	console.log("PASS missing GPU fallback, intentional device destruction");
	console.log("Mocks only: target-browser file:// capability is NOT verified.");
})().catch(error => { console.error(error); process.exitCode = 1; });
