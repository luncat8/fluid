"use strict";
// Shared recording WebGPU mock for the node harnesses (gpu-mock.js, engine.js).
// It records the command stream, tracks live allocations and can leave canned data
// in a readback target when a buffer is copied. It never executes shaders, so it
// validates plumbing and ownership - never numerics.
globalThis.GPUBufferUsage = globalThis.GPUBufferUsage || { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128 };
globalThis.GPUTextureUsage = globalThis.GPUTextureUsage || { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
globalThis.GPUShaderStage = globalThis.GPUShaderStage || { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
globalThis.GPUMapMode = globalThis.GPUMapMode || { READ: 1, WRITE: 2 };

function newSink() {
	return { modules: [], writes: [], pipeOrder: [], bindings: [], copies: [], destroyed: [], textureWrites: [] };
}
const labelOf = o => (o && o.label) || "";
// patterns: source label -> Float32Array stamped into the copy destination, standing
// in for what a real kernel would have written.
function create(patterns) {
	const sink = newSink();
	sink.textureWrites = [];
	const pat = patterns || {};
	const counts = { buffers: 0, textures: 0, dispatches: 0, copies: 0, buffersAlive: 0, texturesAlive: 0 };
	function makeBuffer(desc) {
		counts.buffers++;
		counts.buffersAlive++;
		const buf = new ArrayBuffer(desc.size);
		const b = {
			label: desc.label, size: desc.size, usage: desc.usage, _buf: buf, destroyed: false,
			getMappedRange() { return buf; },
			mapAsync() { return Promise.resolve(); },
			unmap() {},
			destroy() {
				if (this.destroyed) return;
				this.destroyed = true;
				counts.buffersAlive--;
				sink.destroyed.push(desc.label);
			}
		};
		return b;
	}
	// the encoder is the one place that can catch the mistake a recording mock used
	// to hide: a dispatch with no bind group at group 0 (on a device that is a
	// validation error, and the pass then never runs - "nothing moves" with no number
	// to point at). setBindGroup(0, undefined) unbinds, which is what a group that was
	// never created looks like at the call site.
	function makeEncoder() {
		let group0 = null, pipeline = "(none)";
		const pass = {
			setPipeline(p) { pipeline = p.label; sink.pipeOrder.push(p.label); },
			setBindGroup(i, g) {
				if (i === 0) group0 = g || null;
				sink.bindings.push(g && g._entries ? g._entries.length : 0);
			},
			dispatchWorkgroups(x, y, z) {
				if (!group0) {
					throw new Error("mock: dispatch on pipeline \"" + pipeline + "\" with no bind group at group 0"
						+ " (x" + (x || 1) + "y" + (y || 1) + "z" + (z || 1) + ") - a real device rejects the whole command buffer");
				}
				counts.dispatches++;
			},
			end() {}
		};
		return {
			beginComputePass: () => pass,
			beginRenderPass: () => ({ setPipeline() {}, setBindGroup() {}, draw() {}, end() {} }),
			copyBufferToBuffer(src, so, dst, doff, size) {
				counts.copies++;
				sink.copies.push([labelOf(src), labelOf(dst), size]);
				const pattern = pat[labelOf(src)];
				if (pattern) new Float32Array(dst._buf, 0, pattern.length).set(pattern);
			},
			finish: () => ({})
		};
	}
	const device = {
		createBuffer: d => makeBuffer(d),
		createTexture: d => {
			counts.textures++;
			counts.texturesAlive++;
			const t = {
				label: d.label, format: d.format, createView: () => ({}),
				destroy() {
					if (t.destroyed) return;
					t.destroyed = true;
					counts.texturesAlive--;
					sink.destroyed.push(d.label);
				}
			};
			return t;
		},
		createSampler: () => ({}),
		createShaderModule: d => { sink.modules.push(d); return { code: d.code, label: d.label }; },
		createCommandEncoder: () => makeEncoder(),
		createBindGroupLayout: () => ({}),
		createPipelineLayout: () => ({}),
		createBindGroup: d => ({ _entries: d.entries, label: d.label }),
		createComputePipeline: d => {
			const layout = {};
			return { label: d.label, _module: d.compute.module, getBindGroupLayout: () => layout };
		},
		createRenderPipeline: d => ({ label: d.label, getBindGroupLayout: () => ({}) }),
		queue: {
			writeBuffer(buf, offset, src) {
				new Float32Array(buf._buf, offset, src.length).set(src);
				sink.writes.push(labelOf(buf));
			},
			// recorded, not applied: the harnesses that need the uploaded bytes
			// (collider imprints, paint patches) read them back from here
			writeTexture(dst, data, layout, size) {
				sink.textureWrites.push({
					label: labelOf(dst.texture), data, layout, size,
					origin: dst.origin ? [dst.origin.x || 0, dst.origin.y || 0] : [0, 0]
				});
			},
			submit() {},
			// resolved immediately: the harnesses that observe backpressure swap this
			// for a queue they complete by hand
			onSubmittedWorkDone() { return Promise.resolve(); }
		}
	};
	return { device, sink, counts };
}
function mockContext() {
	return { getCurrentTexture: () => ({ createView: () => ({}) }) };
}
module.exports = { create, newSink, mockContext };
