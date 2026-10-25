"use strict";
// Shared recording WebGPU mock for the node harnesses (gpu-mock.js, engine.js).
// It records the command stream, tracks live allocations and can leave canned data
// in a readback target when a buffer is copied. It never executes shaders, so it
// validates plumbing and ownership - never numerics.
globalThis.GPUBufferUsage = globalThis.GPUBufferUsage || { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128 };
globalThis.GPUTextureUsage = globalThis.GPUTextureUsage || { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
globalThis.GPUShaderStage = globalThis.GPUShaderStage || { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
globalThis.GPUMapMode = globalThis.GPUMapMode || { READ: 1, WRITE: 2 };

// structuredClone with transfer is the portable way to detach an ArrayBuffer
function detach(buffer) {
	if (!buffer || buffer.byteLength === 0) return;
	structuredClone(buffer, { transfer: [buffer] });
}
function newSink() {
	return { modules: [], writes: [], pipeOrder: [], bindings: [], copies: [], destroyed: [] };
}
const labelOf = o => (o && o.label) || "";
function texBpp(fmt) {
	if (fmt === "r32float" || fmt === "rgba8uint" || fmt === "rg16float") return 4;
	if (fmt === "rg32float" || fmt === "rgba16float") return 8;
	if (fmt === "r16float") return 2;
	return 16;
}
// patterns: source label -> Float32Array stamped into the copy destination, standing
// in for what a real kernel would have written.
function create(patterns) {
	const sink = newSink();
	const pat = patterns || {};
	const counts = { buffers: 0, textures: 0, dispatches: 0, copies: 0, maps: 0, buffersAlive: 0, texturesAlive: 0 };
	function makeBuffer(desc) {
		counts.buffers++;
		counts.buffersAlive++;
		const buf = new ArrayBuffer(desc.size);
		// map semantics follow the spec closely enough to catch a cached view: every
		// mapAsync takes a snapshot of the requested range, getMappedRange hands out
		// one ArrayBuffer per (offset,size) pair for that map, and unmap detaches them
		// all. a view kept across frames therefore throws or reads zeros here exactly
		// as it would on a driver, instead of silently appearing to work.
		let mappedBytes = null, mappedOff = 0, mappedSize = 0, mappedRanges = new Map();
		const b = {
			label: desc.label, size: desc.size, usage: desc.usage, _buf: buf, destroyed: false,
			mapAsync(mode, offset, size) {
				counts.maps++;
				mappedOff = offset || 0;
				mappedSize = size !== undefined ? size : buf.byteLength - mappedOff;
				mappedBytes = buf.slice(mappedOff, mappedOff + mappedSize);
				mappedRanges.clear();
				return Promise.resolve();
			},
			getMappedRange(offset, size) {
				if (!mappedBytes) throw new Error("getMappedRange before mapAsync: " + desc.label);
				const off = (offset || 0) - mappedOff;
				const len = size !== undefined ? size : mappedBytes.byteLength - off;
				const key = off + ":" + len;
				let range = mappedRanges.get(key);
				if (!range) {
					range = mappedBytes.slice(off, off + len);
					mappedRanges.set(key, range);
				}
				return range;
			},
			unmap() {
				for (const range of mappedRanges.values()) detach(range);
				mappedRanges.clear();
				mappedBytes = null;
			},
			destroy() {
				if (this.destroyed) return;
				this.destroyed = true;
				counts.buffersAlive--;
				sink.destroyed.push(desc.label);
			}
		};
		return b;
	}
	function makeEncoder() {
		const pass = {
			setPipeline(p) { sink.pipeOrder.push(p.label); },
			setBindGroup(i, g) { sink.bindings.push(g && g._entries ? g._entries.length : 0); },
			dispatchWorkgroups() { counts.dispatches++; },
			end() {}
		};
		return {
			beginComputePass: () => pass,
			beginRenderPass: () => ({ setPipeline() {}, setBindGroup() {}, draw() {}, end() {} }),
			copyBufferToBuffer(src, so, dst, doff, size) {
				counts.copies++;
				sink.copies.push([labelOf(src), labelOf(dst), size, so || 0, doff || 0]);
				const pattern = pat[labelOf(src)];
				if (pattern) {
					new Float32Array(dst._buf, doff || 0, pattern.length).set(pattern);
				} else if (src && src._buf && dst && dst._buf) {
					const srcOff = so || 0;
					const dstOff = doff || 0;
					const byteCnt = size !== undefined ? size : Math.min(src._buf.byteLength - srcOff, dst._buf.byteLength - dstOff);
					if (byteCnt > 0) {
						new Uint8Array(dst._buf, dstOff, byteCnt).set(new Uint8Array(src._buf, srcOff, byteCnt));
					}
				}
			},
			copyTextureToBuffer(src, dst, extent) {
				counts.copies++;
				const tex = src && src.texture;
				const buf = dst && dst.buffer;
				const w = Array.isArray(extent) ? extent[0] : (extent && extent.width) || (tex && tex.width) || 0;
				const h = Array.isArray(extent) ? extent[1] : (extent && extent.height) || (tex && tex.height) || 0;
				const bpp = (tex && tex._bpp) || 4;
				const rowBytes = w * bpp;
				const bytesPerRow = (dst && dst.bytesPerRow) || rowBytes;
				const dstBase = (dst && dst.offset) || 0;
				sink.copies.push([labelOf(tex), labelOf(buf), rowBytes * h]);
				if (tex && tex._buf && buf && buf._buf) {
					const srcBytes = new Uint8Array(tex._buf);
					const dstBytes = new Uint8Array(buf._buf);
					const texRowBytes = tex.width * bpp;
					for (let r = 0; r < h; r++) {
						const sOff = r * texRowBytes;
						const dOff = dstBase + r * bytesPerRow;
						dstBytes.set(srcBytes.subarray(sOff, sOff + rowBytes), dOff);
					}
				}
			},
			finish: () => ({})
		};
	}
	const device = {
		createBuffer: d => makeBuffer(d),
		createTexture: d => {
			counts.textures++;
			counts.texturesAlive++;
			const w = Array.isArray(d.size) ? d.size[0] : (d.size && d.size.width) || 0;
			const h = Array.isArray(d.size) ? d.size[1] : (d.size && d.size.height) || 0;
			const bpp = texBpp(d.format);
			const buf = new ArrayBuffer(w * h * bpp);
			const t = {
				label: d.label, format: d.format, width: w, height: h, _bpp: bpp, _buf: buf,
				createView: () => ({}),
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
		pushErrorScope() {},
		popErrorScope() { return Promise.resolve(null); },
		addEventListener() {},
		removeEventListener() {},
		queue: {
			writeBuffer(buf, offset, src, dataOffset, size) {
				if (ArrayBuffer.isView(src)) {
					const bpe = src.BYTES_PER_ELEMENT || 1;
					const elemOff = dataOffset || 0;
					const elemCnt = size !== undefined ? size : (src.length !== undefined ? src.length - elemOff : src.byteLength - elemOff);
					const bytes = new Uint8Array(src.buffer, src.byteOffset + elemOff * bpe, elemCnt * bpe);
					new Uint8Array(buf._buf, offset, bytes.byteLength).set(bytes);
				} else if (src instanceof ArrayBuffer) {
					const byteOff = dataOffset || 0;
					const byteCnt = size !== undefined ? size : src.byteLength - byteOff;
					new Uint8Array(buf._buf, offset, byteCnt).set(new Uint8Array(src, byteOff, byteCnt));
				}
				sink.writes.push(labelOf(buf));
			},
			writeTexture(dst, data, layout, extent) {
				const tex = dst && dst.texture;
				if (!tex || !tex._buf || !data) return;
				const srcBytes = ArrayBuffer.isView(data)
					? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
					: new Uint8Array(data);
				const w = Array.isArray(extent) ? extent[0] : (extent && extent.width) || tex.width;
				const h = Array.isArray(extent) ? extent[1] : (extent && extent.height) || tex.height;
				const bpp = tex._bpp || 4;
				const rowBytes = w * bpp;
				const srcStride = (layout && layout.bytesPerRow) || rowBytes;
				const srcBase = (layout && layout.offset) || 0;
				const dstBytes = new Uint8Array(tex._buf);
				const dstStride = tex.width * bpp;
				for (let r = 0; r < h; r++) {
					const sOff = srcBase + r * srcStride;
					const dOff = r * dstStride;
					dstBytes.set(srcBytes.subarray(sOff, sOff + rowBytes), dOff);
				}
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
