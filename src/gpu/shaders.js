(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	const shaders={};
	shaders.paramsStruct=`
struct Params {
	values: array<f32,128>
};
`;
	shaders.exchange=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massSrc: texture_storage_2d<rgba32float,read>;
@group(0) @binding(2) var massDst: texture_storage_2d<rgba32float,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let texW = i32(params.values[2]);
	let texH = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= texW - 1 || y < 1 || y >= texH - 1) {
		return;
	}
	let center = textureLoad(massSrc, vec2<i32>(x, y));
	var out = center;
	let dt = params.values[4];
	let conductivity = params.values[35];
	if (conductivity <= 0.0) {
		textureStore(massDst, vec2<i32>(x, y), out);
		return;
	}
	let coeff = conductivity * dt;
	// 4 neighbours offsets
	let offsets = array<vec2<i32>,4>(vec2<i32>(1,0), vec2<i32>(-1,0), vec2<i32>(0,1), vec2<i32>(0,-1));
	for (var k: i32 = 0; k < 4; k++) {
		let ox = offsets[k].x;
		let oy = offsets[k].y;
		let nx = x + ox;
		let ny = y + oy;
		if (nx < 1 || nx >= texW - 1 || ny < 1 || ny >= texH - 1) {
			continue;
		}
		let nb = textureLoad(massSrc, vec2<i32>(nx, ny));
		// per channel exchange with C=1 => limit 0.25*delta per face
		for (var c: i32 = 0; c < 4; c++) {
			var sVal: f32;
			var nVal: f32;
			if (c == 0) { sVal = center.r; nVal = nb.r; }
			else if (c == 1) { sVal = center.g; nVal = nb.g; }
			else if (c == 2) { sVal = center.b; nVal = nb.b; }
			else { sVal = center.a; nVal = nb.a; }
			let delta = nVal - sVal;
			var d = coeff * delta;
			let limit = 0.25 * delta;
			if (limit >= 0.0) {
				if (d > limit) { d = limit; }
				if (d < 0.0) { d = 0.0; }
			} else {
				if (d < limit) { d = limit; }
				if (d > 0.0) { d = 0.0; }
			}
			if (c == 0) { out.r = out.r + d; }
			else if (c == 1) { out.g = out.g + d; }
			else if (c == 2) { out.b = out.b + d; }
			else { out.a = out.a + d; }
		}
	}
	textureStore(massDst, vec2<i32>(x, y), out);
}
`;
	shaders.copyRGBA=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var srcTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(2) var dstTex: texture_storage_2d<rgba32float,write>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let texW = i32(params.values[2]);
	let texH = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 0 || x >= texW || y < 0 || y >= texH) { return; }
	let v = textureLoad(srcTex, vec2<i32>(x,y));
	textureStore(dstTex, vec2<i32>(x,y), v);
}
`;
	shaders.copyR=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var srcTex: texture_storage_2d<r32float,read>;
@group(0) @binding(2) var dstTex: texture_storage_2d<r32float,write>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let texW = i32(params.values[2]);
	let texH = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 0 || x >= texW || y < 0 || y >= texH) { return; }
	let v = textureLoad(srcTex, vec2<i32>(x,y));
	textureStore(dstTex, vec2<i32>(x,y), v);
}
`;
	shaders.toDisplay=`
struct Params {
	values: array<f32,128>
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(2) var thermTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(3) var displayTex: texture_storage_2d<rgba16float,write>;

fn tempColor(t: f32, tmin: f32, tmax: f32) -> vec4<f32> {
	let denom = max(tmax - tmin, 1e-6);
	let u = clamp((t - tmin) / denom, 0.0, 1.0);
	// blue->cyan->green->yellow->red
	var r: f32;
	var g: f32;
	var b: f32;
	if (u < 0.25) {
		let a = u / 0.25;
		r = 0.0; g = a; b = 1.0;
	} else if (u < 0.5) {
		let a = (u - 0.25)/0.25;
		r = 0.0; g = 1.0; b = 1.0 - a;
	} else if (u < 0.75) {
		let a = (u - 0.5)/0.25;
		r = a; g = 1.0; b = 0.0;
	} else {
		let a = (u - 0.75)/0.25;
		r = 1.0; g = 1.0 - a*0.5; b = 0.0;
	}
	return vec4<f32>(r,g,b,1.0);
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let texW = i32(params.values[2]);
	let texH = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 0 || x >= texW || y < 0 || y >= texH) { return; }
	let mass = textureLoad(massTex, vec2<i32>(x,y));
	let therm = textureLoad(thermTex, vec2<i32>(x,y));
	let viewMode = i32(params.values[80]);
	let mEps = params.values[58];
	let colorGain = 1.2;
	var out = vec4<f32>(0.05, 0.07, 0.09, 1.0);
	let m = mass.a;
	if (viewMode == 0) {
		if (m < mEps) {
			out = vec4<f32>(0.06, 0.07, 0.09, 1.0);
		} else {
			let rr = clamp(mass.r * colorGain, 0.0, 1.0);
			let gg = clamp(mass.g * colorGain, 0.0, 1.0);
			let bb = clamp(mass.b * colorGain, 0.0, 1.0);
			out = vec4<f32>(rr, gg, bb, 1.0);
			// vacuum blend to dark: reduce brightness if m low
			if (m < 0.2) {
				let f = clamp(m / 0.2, 0.0, 1.0);
				out = mix(vec4<f32>(0.06,0.07,0.09,1.0), out, f);
			}
		}
	} else if (viewMode == 1) {
		let T = therm.g;
		let TMin = params.values[81];
		let TMax = params.values[82];
		if (m < mEps) {
			out = vec4<f32>(0.06,0.07,0.09,1.0);
		} else {
			out = tempColor(T, TMin, TMax);
		}
	} else {
		// fallback pressure/velocity etc show as grey m
		let g = clamp(m, 0.0, 1.0);
		out = vec4<f32>(g,g,g,1.0);
	}
	textureStore(displayTex, vec2<i32>(x,y), out);
}
`;
	shaders.statsReduceA=`
struct Params {
	values: array<f32,128>
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(2) var thermTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(3) var<storage,read_write> partials: array<f32>;

var<workgroup> shTotM: array<f32,64>;
var<workgroup> shTotDr: array<f32,64>;
var<workgroup> shTotDg: array<f32,64>;
var<workgroup> shTotDb: array<f32,64>;
var<workgroup> shTotE: array<f32,64>;
var<workgroup> shMinM: array<f32,64>;
var<workgroup> shMinDr: array<f32,64>;
var<workgroup> shMinDg: array<f32,64>;
var<workgroup> shMinDb: array<f32,64>;
var<workgroup> shMinE: array<f32,64>;
var<workgroup> shCntNeg: array<f32,64>;
var<workgroup> shCntOver: array<f32,64>;
var<workgroup> shMaxM: array<f32,64>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
	let texW = i32(params.values[2]);
	let texH = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	var totM: f32 = 0.0;
	var totDr: f32 = 0.0;
	var totDg: f32 = 0.0;
	var totDb: f32 = 0.0;
	var totE: f32 = 0.0;
	var minM: f32 = 1e30;
	var minDr: f32 = 1e30;
	var minDg: f32 = 1e30;
	var minDb: f32 = 1e30;
	var minE: f32 = 1e30;
	var cntNeg: f32 = 0.0;
	var cntOver: f32 = 0.0;
	var maxM: f32 = -1e30;
	let interior = x >= 1 && x < texW - 1 && y >= 1 && y < texH - 1;
	if (interior) {
		let mass = textureLoad(massTex, vec2<i32>(x,y));
		let therm = textureLoad(thermTex, vec2<i32>(x,y));
		let m = mass.a;
		let dr = mass.r;
		let dg = mass.g;
		let db = mass.b;
		let E = therm.r;
		totM = m;
		totDr = dr;
		totDg = dg;
		totDb = db;
		totE = E;
		minM = m;
		minDr = dr;
		minDg = dg;
		minDb = db;
		minE = E;
		maxM = m;
		if (m < 0.0 || dr < 0.0 || dg < 0.0 || db < 0.0 || E < 0.0) { cntNeg = 1.0; }
		if (m > 1.001) { cntOver = 1.0; }
	} else {
		minM = 1e30; minDr = 1e30; minDg = 1e30; minDb = 1e30; minE = 1e30; maxM = -1e30;
	}
	shTotM[lid] = totM;
	shTotDr[lid] = totDr;
	shTotDg[lid] = totDg;
	shTotDb[lid] = totDb;
	shTotE[lid] = totE;
	shMinM[lid] = minM;
	shMinDr[lid] = minDr;
	shMinDg[lid] = minDg;
	shMinDb[lid] = minDb;
	shMinE[lid] = minE;
	shCntNeg[lid] = cntNeg;
	shCntOver[lid] = cntOver;
	shMaxM[lid] = maxM;
	workgroupBarrier();
	// reduction in shared memory: tree
	for (var stride: u32 = 32u; stride > 0u; stride = stride >> 1u) {
		if (lid < stride) {
			shTotM[lid] = shTotM[lid] + shTotM[lid + stride];
			shTotDr[lid] = shTotDr[lid] + shTotDr[lid + stride];
			shTotDg[lid] = shTotDg[lid] + shTotDg[lid + stride];
			shTotDb[lid] = shTotDb[lid] + shTotDb[lid + stride];
			shTotE[lid] = shTotE[lid] + shTotE[lid + stride];
			shMinM[lid] = min(shMinM[lid], shMinM[lid + stride]);
			shMinDr[lid] = min(shMinDr[lid], shMinDr[lid + stride]);
			shMinDg[lid] = min(shMinDg[lid], shMinDg[lid + stride]);
			shMinDb[lid] = min(shMinDb[lid], shMinDb[lid + stride]);
			shMinE[lid] = min(shMinE[lid], shMinE[lid + stride]);
			shCntNeg[lid] = shCntNeg[lid] + shCntNeg[lid + stride];
			shCntOver[lid] = shCntOver[lid] + shCntOver[lid + stride];
			shMaxM[lid] = max(shMaxM[lid], shMaxM[lid + stride]);
		}
		workgroupBarrier();
	}
	if (lid == 0u) {
		let wgX = nwg.x;
		let wgIndex = wid.y * wgX + wid.x;
		let base = wgIndex * 16u;
		partials[base + 0u] = shTotM[0];
		partials[base + 1u] = shTotDr[0];
		partials[base + 2u] = shTotDg[0];
		partials[base + 3u] = shTotDb[0];
		partials[base + 4u] = shTotE[0];
		partials[base + 5u] = shMinM[0];
		partials[base + 6u] = shMinDr[0];
		partials[base + 7u] = shMinDg[0];
		partials[base + 8u] = shMinDb[0];
		partials[base + 9u] = shMinE[0];
		partials[base + 10u] = shMaxM[0];
		partials[base + 11u] = shCntNeg[0];
		partials[base + 12u] = shCntOver[0];
		partials[base + 13u] = 0.0;
		partials[base + 14u] = 0.0;
		partials[base + 15u] = 0.0;
	}
}
`;
	shaders.statsReduceB=`
@group(0) @binding(0) var<storage,read> partials: array<f32>;
@group(0) @binding(1) var<storage,read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>; // x=numPartials*16, y=unused

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	if (gid.x != 0u) { return; }
	let num = dims.x / 16u;
	var totM: f32 = 0.0;
	var totDr: f32 = 0.0;
	var totDg: f32 = 0.0;
	var totDb: f32 = 0.0;
	var totE: f32 = 0.0;
	var minM: f32 = 1e30;
	var minDr: f32 = 1e30;
	var minDg: f32 = 1e30;
	var minDb: f32 = 1e30;
	var minE: f32 = 1e30;
	var maxM: f32 = -1e30;
	var cntNeg: f32 = 0.0;
	var cntOver: f32 = 0.0;
	for (var i: u32 = 0u; i < num; i++) {
		let base = i * 16u;
		totM = totM + partials[base + 0u];
		totDr = totDr + partials[base + 1u];
		totDg = totDg + partials[base + 2u];
		totDb = totDb + partials[base + 3u];
		totE = totE + partials[base + 4u];
		minM = min(minM, partials[base + 5u]);
		minDr = min(minDr, partials[base + 6u]);
		minDg = min(minDg, partials[base + 7u]);
		minDb = min(minDb, partials[base + 8u]);
		minE = min(minE, partials[base + 9u]);
		maxM = max(maxM, partials[base + 10u]);
		cntNeg = cntNeg + partials[base + 11u];
		cntOver = cntOver + partials[base + 12u];
	}
	if (minM == 1e30) { minM = 0.0; }
	if (minDr == 1e30) { minDr = 0.0; }
	if (minDg == 1e30) { minDg = 0.0; }
	if (minDb == 1e30) { minDb = 0.0; }
	if (minE == 1e30) { minE = 0.0; }
	if (maxM == -1e30) { maxM = 0.0; }
	stats[0] = totM;
	stats[1] = totDr;
	stats[2] = totDg;
	stats[3] = totDb;
	stats[4] = totE;
	stats[5] = minM;
	stats[6] = minDr;
	stats[7] = minDg;
	stats[8] = minDb;
	stats[9] = minE;
	stats[10] = maxM;
	stats[11] = cntNeg;
	stats[12] = cntOver;
	stats[13] = 0.0;
	stats[14] = 0.0;
	stats[15] = 0.0;
}
`;
	shaders.renderVert=`
struct VSOut {
	@builtin(position) pos: vec4<f32>,
	@location(0) uv: vec2<f32>
};
@vertex
fn main(@builtin(vertex_index) vi: u32) -> VSOut {
	var pos = array<vec2<f32>,3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);
	var uv = array<vec2<f32>,3>(
		vec2<f32>(0.0, 1.0),
		vec2<f32>(2.0, 1.0),
		vec2<f32>(0.0, -1.0)
	);
	var out: VSOut;
	out.pos = vec4<f32>(pos[vi], 0.0, 1.0);
	out.uv = uv[vi];
	return out;
}
`;
	shaders.renderFrag=`
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var displayTex: texture_2d<f32>;
@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
	let c = textureSample(displayTex, samp, uv);
	return c;
}
`;
	Fluid.gpu.shaders=shaders;
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.shaders;
})(globalThis);
