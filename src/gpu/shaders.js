(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	// every kernel mirrors src/core/solver.js (the numeric oracle) exactly:
	// same blocking predicates, same donor rules, same update order. if the two
	// disagree, 0.0.0-plan.md section 5 wins and both get fixed.
	// param offsets are Fluid.core.config.paramsOff.
	const shaders={};
	shaders.paramsStruct=`
struct Params { values: array<f32,128> };
`;
	// ---------- forces ----------
	shaders.forces=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var velSrc: texture_storage_2d<rg32float,read>;
@group(0) @binding(2) var velDst: texture_storage_2d<rg32float,write>;
@group(0) @binding(3) var massTex: texture_2d<f32>;
@group(0) @binding(4) var thermTex: texture_2d<f32>;
@group(0) @binding(5) var flagsTex: texture_2d<u32>;

fn isWall(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	return textureLoad(flagsTex, vec2<i32>(x, y), 0).r > 0u;
}
fn uBlocked(x: i32, y: i32, w: i32, h: i32) -> bool {
	return x <= 1 || x >= w - 1 || isWall(x - 1, y, w, h) || isWall(x, y, w, h);
}
fn vBlocked(x: i32, y: i32, w: i32, h: i32) -> bool {
	return y <= 1 || y >= h - 1 || isWall(x, y - 1, w, h) || isWall(x, y, w, h);
}
fn cellAccelX(x: i32, y: i32, w: i32, h: i32, gx: f32, rhoMean: f32, rhoRef: f32, beta: f32, tref: f32) -> f32 {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return 0.0; }
	let m = textureLoad(massTex, vec2<i32>(x, y), 0).a;
	let T = textureLoad(thermTex, vec2<i32>(x, y), 0).g;
	return gx * (rhoMean - m) / rhoRef - gx * beta * (T - tref);
}
fn cellAccelY(x: i32, y: i32, w: i32, h: i32, gy: f32, rhoMean: f32, rhoRef: f32, beta: f32, tref: f32) -> f32 {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return 0.0; }
	let m = textureLoad(massTex, vec2<i32>(x, y), 0).a;
	let T = textureLoad(thermTex, vec2<i32>(x, y), 0).g;
	return gy * (rhoMean - m) / rhoRef - gy * beta * (T - tref);
}
fn mouseFall(dx: f32, dy: f32, r: f32) -> f32 {
	let r2 = dx * dx + dy * dy;
	if (r2 >= r * r) { return 0.0; }
	let f = 1.0 - sqrt(r2) / r;
	return f * f;
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let dt = params.values[4];
	let gx = params.values[16];
	let gy = params.values[17];
	let rhoRef = params.values[18];
	let beta = params.values[19];
	let tref = params.values[20];
	let rhoMean = params.values[27];
	let ms = params.values[23];
	let mvx = params.values[24];
	let mvy = params.values[25];
	let mr = params.values[26];
	let vclamp = params.values[38];
	let src = textureLoad(velSrc, vec2<i32>(x, y));
	var u = 0.0;
	var v = 0.0;
	if (!uBlocked(x, y, w, h)) {
		u = src.r + dt * 0.5 * (cellAccelX(x - 1, y, w, h, gx, rhoMean, rhoRef, beta, tref) + cellAccelX(x, y, w, h, gx, rhoMean, rhoRef, beta, tref));
		if (ms > 0.0) {
			let f = mouseFall(f32(x - 1) - params.values[21], f32(y) - 0.5 - params.values[22], mr);
			u = u + dt * mvx * f * ms;
		}
		u = clamp(u, -vclamp, vclamp);
	}
	if (!vBlocked(x, y, w, h)) {
		v = src.g + dt * 0.5 * (cellAccelY(x, y - 1, w, h, gy, rhoMean, rhoRef, beta, tref) + cellAccelY(x, y, w, h, gy, rhoMean, rhoRef, beta, tref));
		if (ms > 0.0) {
			let f = mouseFall(f32(x) - 0.5 - params.values[21], f32(y - 1) - params.values[22], mr);
			v = v + dt * mvy * f * ms;
		}
		v = clamp(v, -vclamp, vclamp);
	}
	textureStore(velDst, vec2<i32>(x, y), vec4<f32>(u, v, 0.0, 0.0));
}
`;
	// ---------- semi-lagrangian velocity advection ----------
	shaders.advectVel=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var velSrc: texture_storage_2d<rg32float,read>;
@group(0) @binding(2) var velDst: texture_storage_2d<rg32float,write>;

fn loadU(t: texture_storage_2d<rg32float,read>, m: i32, l: i32, w: i32, h: i32) -> f32 {
	let mm = clamp(m, 0, w - 1);
	let ll = clamp(l, 0, h - 1);
	return textureLoad(t, vec2<i32>(mm, ll)).r;
}
fn loadV(t: texture_storage_2d<rg32float,read>, m: i32, l: i32, w: i32, h: i32) -> f32 {
	let mm = clamp(m, 0, w - 1);
	let ll = clamp(l, 0, h - 1);
	return textureLoad(t, vec2<i32>(mm, ll)).g;
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let n = w - 2;
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let dt = params.values[4];
	let src = textureLoad(velSrc, vec2<i32>(x, y));
	var u = 0.0;
	var v = 0.0;
	// u face at (x-1, y-0.5); sample the u lattice (integers x, half-offset y)
	let uc = src.r;
	let vc = 0.25 * (loadV(velSrc, x - 1, y, w, h) + loadV(velSrc, x - 1, y + 1, w, h)
		+ loadV(velSrc, x, y, w, h) + loadV(velSrc, x, y + 1, w, h));
	var px = f32(x - 1) - dt * uc;
	var py = f32(y) - 0.5 - dt * vc;
	px = clamp(px, 0.0, f32(n));
	py = clamp(py, 0.5, f32(n) - 0.5);
	let i0 = i32(floor(px));
	let fx = px - f32(i0);
	let l0 = i32(floor(py + 0.5));
	let fy = py + 0.5 - f32(l0);
	let u00 = loadU(velSrc, i0 + 1, l0, w, h);
	let u10 = loadU(velSrc, i0 + 2, l0, w, h);
	let u01 = loadU(velSrc, i0 + 1, l0 + 1, w, h);
	let u11 = loadU(velSrc, i0 + 2, l0 + 1, w, h);
	let a0 = u00 + (u10 - u00) * fx;
	let a1 = u01 + (u11 - u01) * fx;
	u = a0 + (a1 - a0) * fy;
	// v face at (x-0.5, y-1); sample the v lattice (half-offset x, integers y)
	let vc2 = src.g;
	let uc2 = 0.25 * (loadU(velSrc, x, y - 1, w, h) + loadU(velSrc, x + 1, y - 1, w, h)
		+ loadU(velSrc, x, y, w, h) + loadU(velSrc, x + 1, y, w, h));
	var px2 = f32(x) - 0.5 - dt * uc2;
	var py2 = f32(y - 1) - dt * vc2;
	px2 = clamp(px2, 0.5, f32(n) - 0.5);
	py2 = clamp(py2, 0.0, f32(n));
	let k0 = i32(floor(px2 + 0.5));
	let fx2 = px2 + 0.5 - f32(k0);
	let l1 = i32(floor(py2));
	let fy2 = py2 - f32(l1);
	let v00 = loadV(velSrc, k0, l1 + 1, w, h);
	let v10 = loadV(velSrc, k0 + 1, l1 + 1, w, h);
	let v01 = loadV(velSrc, k0, l1 + 2, w, h);
	let v11 = loadV(velSrc, k0 + 1, l1 + 2, w, h);
	let b0 = v00 + (v10 - v00) * fx2;
	let b1 = v01 + (v11 - v01) * fx2;
	v = b0 + (b1 - b0) * fy2;
	textureStore(velDst, vec2<i32>(x, y), vec4<f32>(u, v, 0.0, 0.0));
}
`;
	// ---------- poisson rhs with the liquid anti-overpack source S ----------
	shaders.poissonRhs=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var divsrc: texture_storage_2d<r32float,write>;
@group(0) @binding(2) var velTex: texture_2d<f32>;
@group(0) @binding(3) var massTex: texture_2d<f32>;
@group(0) @binding(4) var flagsTex: texture_2d<u32>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	var b = 0.0;
	if (textureLoad(flagsTex, vec2<i32>(x, y), 0).r == 0u) {
		let dt = params.values[4];
		let uL = textureLoad(velTex, vec2<i32>(x, y), 0).r;
		let uR = textureLoad(velTex, vec2<i32>(x + 1, y), 0).r;
		let vB = textureLoad(velTex, vec2<i32>(x, y), 0).g;
		let vT = textureLoad(velTex, vec2<i32>(x, y + 1), 0).g;
		let m = textureLoad(massTex, vec2<i32>(x, y), 0).a;
		var S = 0.0;
		let rhoLiq = params.values[55];
		if (m > rhoLiq) {
			let tau = params.values[37];
			let smax = params.values[36];
			S = clamp((m - rhoLiq) / (m * tau), 0.0, smax);
		}
		b = (uR - uL + vT - vB) / dt - S;
	}
	textureStore(divsrc, vec2<i32>(x, y), vec4<f32>(b, 0.0, 0.0, 0.0));
}
`;
	// ---------- one red-black gauss-seidel color sweep ----------
	// plain jacobi is unusable on the closed box: the checkerboard mode is an
	// exact eigenvalue -1 of the jacobi iteration there (see core solver note).
	shaders.projectStep=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var pressSrc: texture_storage_2d<r32float,read>;
@group(0) @binding(2) var pressDst: texture_storage_2d<r32float,write>;
@group(0) @binding(3) var divsrcTex: texture_2d<f32>;
@group(0) @binding(4) var massTex: texture_2d<f32>;
@group(0) @binding(5) var flagsTex: texture_2d<u32>;
override COLOR: u32;

fn faceW(x: i32, y: i32, nx: i32, ny: i32, w: i32, h: i32) -> f32 {
	if (nx < 1 || nx > w - 2 || ny < 1 || ny > h - 2) { return 0.0; }
	if (textureLoad(flagsTex, vec2<i32>(nx, ny), 0).r > 0u) { return 0.0; }
	let rhoVac = params.values[57];
	let ra = max(textureLoad(massTex, vec2<i32>(x, y), 0).a, rhoVac);
	let rb = max(textureLoad(massTex, vec2<i32>(nx, ny), 0).a, rhoVac);
	return 1.0 / (2.0 * ra * rb / (ra + rb));
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let src = textureLoad(pressSrc, vec2<i32>(x, y)).r;
	var p = src;
	if ((u32(x + y) & 1u) == COLOR) {
		if (textureLoad(flagsTex, vec2<i32>(x, y), 0).r > 0u) {
			p = 0.0;
		} else {
			let b = textureLoad(divsrcTex, vec2<i32>(x, y), 0).r;
			var wsum = 0.0;
			var sum = 0.0;
			var fw = faceW(x, y, x - 1, y, w, h);
			if (fw > 0.0) { wsum = wsum + fw; sum = sum + fw * textureLoad(pressSrc, vec2<i32>(x - 1, y)).r; }
			fw = faceW(x, y, x + 1, y, w, h);
			if (fw > 0.0) { wsum = wsum + fw; sum = sum + fw * textureLoad(pressSrc, vec2<i32>(x + 1, y)).r; }
			fw = faceW(x, y, x, y - 1, w, h);
			if (fw > 0.0) { wsum = wsum + fw; sum = sum + fw * textureLoad(pressSrc, vec2<i32>(x, y - 1)).r; }
			fw = faceW(x, y, x, y + 1, w, h);
			if (fw > 0.0) { wsum = wsum + fw; sum = sum + fw * textureLoad(pressSrc, vec2<i32>(x, y + 1)).r; }
			if (wsum > 0.0) { p = (sum - b) / wsum; }
		}
	}
	textureStore(pressDst, vec2<i32>(x, y), vec4<f32>(p, 0.0, 0.0, 0.0));
}
`;
	// ---------- apply pressure gradient ----------
	shaders.applyGrad=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var velSrc: texture_storage_2d<rg32float,read>;
@group(0) @binding(2) var velDst: texture_storage_2d<rg32float,write>;
@group(0) @binding(3) var pressTex: texture_2d<f32>;
@group(0) @binding(4) var massTex: texture_2d<f32>;
@group(0) @binding(5) var flagsTex: texture_2d<u32>;

fn isWall2(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	return textureLoad(flagsTex, vec2<i32>(x, y), 0).r > 0u;
}
fn uBlocked2(x: i32, y: i32, w: i32, h: i32) -> bool {
	return x <= 1 || x >= w - 1 || isWall2(x - 1, y, w, h) || isWall2(x, y, w, h);
}
fn vBlocked2(x: i32, y: i32, w: i32, h: i32) -> bool {
	return y <= 1 || y >= h - 1 || isWall2(x, y - 1, w, h) || isWall2(x, y, w, h);
}
fn invRho(ax: i32, ay: i32, bx: i32, by: i32, rhoVac: f32) -> f32 {
	let ra = max(textureLoad(massTex, vec2<i32>(ax, ay), 0).a, rhoVac);
	let rb = max(textureLoad(massTex, vec2<i32>(bx, by), 0).a, rhoVac);
	return 1.0 / (2.0 * ra * rb / (ra + rb));
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let dt = params.values[4];
	let rhoVac = params.values[57];
	let src = textureLoad(velSrc, vec2<i32>(x, y));
	var u = src.r;
	var v = src.g;
	if (!uBlocked2(x, y, w, h)) {
		let rf = invRho(x - 1, y, x, y, rhoVac);
		u = u - dt * (textureLoad(pressTex, vec2<i32>(x, y), 0).r - textureLoad(pressTex, vec2<i32>(x - 1, y), 0).r) * rf;
	}
	if (!vBlocked2(x, y, w, h)) {
		let rf = invRho(x, y - 1, x, y, rhoVac);
		v = v - dt * (textureLoad(pressTex, vec2<i32>(x, y), 0).r - textureLoad(pressTex, vec2<i32>(x, y - 1), 0).r) * rf;
	}
	textureStore(velDst, vec2<i32>(x, y), vec4<f32>(u, v, 0.0, 0.0));
}
`;
	// ---------- face volume flux, written once, read by both neighbours ----------
	shaders.computeFlux=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var velTex: texture_2d<f32>;
@group(0) @binding(2) var flagsTex: texture_2d<u32>;
@group(0) @binding(3) var fluxDst: texture_storage_2d<rg32float,write>;

fn isWall3(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	return textureLoad(flagsTex, vec2<i32>(x, y), 0).r > 0u;
}
fn rawFx(x: i32, y: i32, w: i32, h: i32, dt: f32) -> f32 {
	if (x <= 1 || x >= w - 1 || isWall3(x - 1, y, w, h) || isWall3(x, y, w, h)) { return 0.0; }
	return clamp(textureLoad(velTex, vec2<i32>(x, y), 0).r * dt, -1.0, 1.0);
}
fn rawFy(x: i32, y: i32, w: i32, h: i32, dt: f32) -> f32 {
	if (y <= 1 || y >= h - 1 || isWall3(x, y - 1, w, h) || isWall3(x, y, w, h)) { return 0.0; }
	return clamp(textureLoad(velTex, vec2<i32>(x, y), 0).g * dt, -1.0, 1.0);
}
fn outFlux(x: i32, y: i32, w: i32, h: i32, dt: f32) -> f32 {
	let fw = rawFx(x, y, w, h, dt);
	let fe = rawFx(x + 1, y, w, h, dt);
	let fs = rawFy(x, y, w, h, dt);
	let fN = rawFy(x, y + 1, w, h, dt);
	return max(0.0, -fw) + max(0.0, fe) + max(0.0, -fs) + max(0.0, fN);
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let dt = params.values[4];
	var fx = rawFx(x, y, w, h, dt);
	var fy = rawFy(x, y, w, h, dt);
	// monotone donor-cell needs total outflow <= 1 volume per cell per step. the
	// +-1 face clamp alone is not enough: a diverging cell empties through up to
	// 4 faces at once and the q'=q*(1-k)+in update then amplifies sign-alternating
	// mass every step (draw + stir exploded to NaN in ~15 frames). scale the faces
	// a cell owns as outflow so they sum to <= 1; each face is outflow for exactly
	// one of its two cells, so the stored value stays the single shared flux and
	// conservation is untouched. same limiter in src/core/solver.js.
	if (fx != 0.0) {
		let s = select(outFlux(x, y, w, h, dt), outFlux(x - 1, y, w, h, dt), fx > 0.0);
		if (s > 1.0) { fx = fx / s; }
	}
	if (fy != 0.0) {
		let s = select(outFlux(x, y, w, h, dt), outFlux(x, y - 1, w, h, dt), fy > 0.0);
		if (s > 1.0) { fy = fy / s; }
	}
	textureStore(fluxDst, vec2<i32>(x, y), vec4<f32>(fx, fy, 0.0, 0.0));
}
`;
	// ---------- donor cell upwind transport ----------
	// conservation argument: both cells sharing a face read the identical stored
	// flux float and the identical donor value, with opposite signs. never
	// recompute a flux here from velocity - that breaks exact conservation.
	shaders.advectMass=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var fluxTex: texture_storage_2d<rg32float,read>;
@group(0) @binding(2) var massSrc: texture_storage_2d<rgba32float,read>;
@group(0) @binding(3) var massDst: texture_storage_2d<rgba32float,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let fw = textureLoad(fluxTex, vec2<i32>(x, y)).r;
	let fe = textureLoad(fluxTex, vec2<i32>(x + 1, y)).r;
	let fs = textureLoad(fluxTex, vec2<i32>(x, y)).g;
	let fN = textureLoad(fluxTex, vec2<i32>(x, y + 1)).g;
	let q = textureLoad(massSrc, vec2<i32>(x, y));
	let qW = textureLoad(massSrc, vec2<i32>(x - 1, y));
	let qE = textureLoad(massSrc, vec2<i32>(x + 1, y));
	let qS = textureLoad(massSrc, vec2<i32>(x, y - 1));
	let qN = textureLoad(massSrc, vec2<i32>(x, y + 1));
	var out: vec4<f32>;
	out.r = q.r + fw * select(q.r, qW.r, fw > 0.0) - fe * select(qE.r, q.r, fe > 0.0)
		+ fs * select(q.r, qS.r, fs > 0.0) - fN * select(qN.r, q.r, fN > 0.0);
	out.g = q.g + fw * select(q.g, qW.g, fw > 0.0) - fe * select(qE.g, q.g, fe > 0.0)
		+ fs * select(q.g, qS.g, fs > 0.0) - fN * select(qN.g, q.g, fN > 0.0);
	out.b = q.b + fw * select(q.b, qW.b, fw > 0.0) - fe * select(qE.b, q.b, fe > 0.0)
		+ fs * select(q.b, qS.b, fs > 0.0) - fN * select(qN.b, q.b, fN > 0.0);
	out.a = q.a + fw * select(q.a, qW.a, fw > 0.0) - fe * select(qE.a, q.a, fe > 0.0)
		+ fs * select(q.a, qS.a, fs > 0.0) - fN * select(qN.a, q.a, fN > 0.0);
	textureStore(massDst, vec2<i32>(x, y), out);
}
`;
	shaders.advectTherm=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var fluxTex: texture_storage_2d<rg32float,read>;
@group(0) @binding(2) var thermSrc: texture_storage_2d<rgba32float,read>;
@group(0) @binding(3) var thermDst: texture_storage_2d<rgba32float,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let fw = textureLoad(fluxTex, vec2<i32>(x, y)).r;
	let fe = textureLoad(fluxTex, vec2<i32>(x + 1, y)).r;
	let fs = textureLoad(fluxTex, vec2<i32>(x, y)).g;
	let fN = textureLoad(fluxTex, vec2<i32>(x, y + 1)).g;
	let q = textureLoad(thermSrc, vec2<i32>(x, y));
	let qW = textureLoad(thermSrc, vec2<i32>(x - 1, y));
	let qE = textureLoad(thermSrc, vec2<i32>(x + 1, y));
	let qS = textureLoad(thermSrc, vec2<i32>(x, y - 1));
	let qN = textureLoad(thermSrc, vec2<i32>(x, y + 1));
	var out: vec4<f32>;
	out.r = q.r + fw * select(q.r, qW.r, fw > 0.0) - fe * select(qE.r, q.r, fe > 0.0)
		+ fs * select(q.r, qS.r, fs > 0.0) - fN * select(qN.r, q.r, fN > 0.0);
	out.g = q.g + fw * select(q.g, qW.g, fw > 0.0) - fe * select(qE.g, q.g, fe > 0.0)
		+ fs * select(q.g, qS.g, fs > 0.0) - fN * select(qN.g, q.g, fN > 0.0);
	out.b = q.b + fw * select(q.b, qW.b, fw > 0.0) - fe * select(qE.b, q.b, fe > 0.0)
		+ fs * select(q.b, qS.b, fs > 0.0) - fN * select(qN.b, q.b, fN > 0.0);
	out.a = q.a + fw * select(q.a, qW.a, fw > 0.0) - fe * select(qE.a, q.a, fe > 0.0)
		+ fs * select(q.a, qS.a, fs > 0.0) - fN * select(qN.a, q.a, fN > 0.0);
	textureStore(thermDst, vec2<i32>(x, y), out);
}
`;
	// ---------- Zalesak FCT for RGB dye densities ----------
	shaders.limiterPrepare=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var fluxTex: texture_2d<f32>;
@group(0) @binding(2) var massSrc: texture_2d<f32>;
@group(0) @binding(3) var massLow: texture_2d<f32>;
@group(0) @binding(4) var limiterA: texture_storage_2d<rgba32float,write>;
@group(0) @binding(5) var limiterB: texture_storage_2d<rg32float,write>;

fn antiFlux(F: f32, qL: vec3<f32>, qR: vec3<f32>) -> vec3<f32> {
	let a = abs(F);
	return 0.5 * a * (1.0 - a) * (qR - qL);
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let at = vec2<i32>(x, y);
	let q = textureLoad(massLow, at, 0).rgb;
	var qMin = q;
	var qMax = q;
	for (var oy = -1; oy <= 1; oy = oy + 1) {
		for (var ox = -1; ox <= 1; ox = ox + 1) {
			let qN = textureLoad(massLow, at + vec2<i32>(ox, oy), 0).rgb;
			qMin = min(qMin, qN);
			qMax = max(qMax, qN);
		}
	}
	let fw = textureLoad(fluxTex, at, 0).r;
	let fe = textureLoad(fluxTex, at + vec2<i32>(1, 0), 0).r;
	let fs = textureLoad(fluxTex, at, 0).g;
	let fN = textureLoad(fluxTex, at + vec2<i32>(0, 1), 0).g;
	let qC = textureLoad(massSrc, at, 0).rgb;
	let dW = antiFlux(fw, textureLoad(massSrc, at + vec2<i32>(-1, 0), 0).rgb, qC);
	let dE = antiFlux(fe, qC, textureLoad(massSrc, at + vec2<i32>(1, 0), 0).rgb);
	let dS = antiFlux(fs, textureLoad(massSrc, at + vec2<i32>(0, -1), 0).rgb, qC);
	let dN = antiFlux(fN, qC, textureLoad(massSrc, at + vec2<i32>(0, 1), 0).rgb);
	let zero = vec3<f32>(0.0);
	let pPlus = max(dW, zero) + max(-dE, zero) + max(dS, zero) + max(-dN, zero);
	let pMinus = max(-dW, zero) + max(dE, zero) + max(-dS, zero) + max(dN, zero);
	let e = vec3<f32>(1e-30);
	var rPlus = select(vec3<f32>(1.0), min(vec3<f32>(1.0), max(qMax - q, zero) / max(pPlus, e)), pPlus > e);
	var rMinus = select(vec3<f32>(1.0), min(vec3<f32>(1.0), max(q - qMin, zero) / max(pMinus, e)), pMinus > e);
	rPlus = select(rPlus * 0.99999, rPlus, rPlus >= vec3<f32>(1.0));
	rMinus = select(rMinus * 0.99999, rMinus, rMinus >= vec3<f32>(1.0));
	textureStore(limiterA, at, vec4<f32>(rPlus.r, rMinus.r, rPlus.g, rMinus.g));
	textureStore(limiterB, at, vec4<f32>(rPlus.b, rMinus.b, 0.0, 0.0));
}
`;
	shaders.limiterApply=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var fluxTex: texture_2d<f32>;
@group(0) @binding(2) var massSrc: texture_2d<f32>;
@group(0) @binding(3) var massLow: texture_2d<f32>;
@group(0) @binding(4) var limiterA: texture_2d<f32>;
@group(0) @binding(5) var limiterB: texture_2d<f32>;
@group(0) @binding(6) var massDst: texture_storage_2d<rgba32float,write>;

fn antiFlux(F: f32, qL: vec3<f32>, qR: vec3<f32>) -> vec3<f32> {
	let a = abs(F);
	return 0.5 * a * (1.0 - a) * (qR - qL);
}
fn rPlusAt(at: vec2<i32>) -> vec3<f32> {
	let a = textureLoad(limiterA, at, 0);
	let b = textureLoad(limiterB, at, 0);
	return vec3<f32>(a.r, a.b, b.r);
}
fn rMinusAt(at: vec2<i32>) -> vec3<f32> {
	let a = textureLoad(limiterA, at, 0);
	let b = textureLoad(limiterB, at, 0);
	return vec3<f32>(a.g, a.a, b.g);
}
fn limitAnti(d: vec3<f32>, plusL: vec3<f32>, minusL: vec3<f32>, plusR: vec3<f32>, minusR: vec3<f32>) -> vec3<f32> {
	let alphaPositive = min(minusL, plusR);
	let alphaNegative = min(plusL, minusR);
	return d * select(alphaNegative, alphaPositive, d > vec3<f32>(0.0));
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let at = vec2<i32>(x, y);
	let low = textureLoad(massLow, at, 0);
	let qC = textureLoad(massSrc, at, 0).rgb;
	let qW = textureLoad(massSrc, at + vec2<i32>(-1, 0), 0).rgb;
	let qE = textureLoad(massSrc, at + vec2<i32>(1, 0), 0).rgb;
	let qS = textureLoad(massSrc, at + vec2<i32>(0, -1), 0).rgb;
	let qN = textureLoad(massSrc, at + vec2<i32>(0, 1), 0).rgb;
	let fw = textureLoad(fluxTex, at, 0).r;
	let fe = textureLoad(fluxTex, at + vec2<i32>(1, 0), 0).r;
	let fs = textureLoad(fluxTex, at, 0).g;
	let fN = textureLoad(fluxTex, at + vec2<i32>(0, 1), 0).g;
	let plusC = rPlusAt(at);
	let minusC = rMinusAt(at);
	let westAt = at + vec2<i32>(-1, 0);
	let eastAt = at + vec2<i32>(1, 0);
	let southAt = at + vec2<i32>(0, -1);
	let northAt = at + vec2<i32>(0, 1);
	let dW = antiFlux(fw, qW, qC);
	let dE = antiFlux(fe, qC, qE);
	let dS = antiFlux(fs, qS, qC);
	let dN = antiFlux(fN, qC, qN);
	let west = limitAnti(dW, rPlusAt(westAt), rMinusAt(westAt), plusC, minusC);
	let east = limitAnti(dE, plusC, minusC, rPlusAt(eastAt), rMinusAt(eastAt));
	let south = limitAnti(dS, rPlusAt(southAt), rMinusAt(southAt), plusC, minusC);
	let north = limitAnti(dN, plusC, minusC, rPlusAt(northAt), rMinusAt(northAt));
	let dye = low.rgb + west - east + south - north;
	textureStore(massDst, at, vec4<f32>(dye, low.a));
}
`;
	// ---------- exact additive painting (ledger sums the same floats) ----------
	shaders.paintMass=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massSrc: texture_storage_2d<rgba32float,read>;
@group(0) @binding(2) var massDst: texture_storage_2d<rgba32float,write>;
@group(0) @binding(3) var patchTex: texture_storage_2d<rgba32float,read>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let x = i32(gid.x);
	let y = i32(gid.y);
	let ox = i32(params.values[8]);
	let oy = i32(params.values[9]);
	let pw = i32(params.values[10]);
	let ph = i32(params.values[11]);
	let inPatch = x >= ox && x < ox + pw && y >= oy && y < oy + ph;
	var q = textureLoad(massSrc, vec2<i32>(x, y));
	if (inPatch) {
		let p = textureLoad(patchTex, vec2<i32>(x - ox, y - oy));
		q = q + p;
	}
	textureStore(massDst, vec2<i32>(x, y), q);
}
`;
	shaders.paintTherm=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var thermSrc: texture_storage_2d<rgba32float,read>;
@group(0) @binding(2) var thermDst: texture_storage_2d<rgba32float,write>;
@group(0) @binding(3) var ePatchTex: texture_storage_2d<r32float,read>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let x = i32(gid.x);
	let y = i32(gid.y);
	let ox = i32(params.values[8]);
	let oy = i32(params.values[9]);
	let pw = i32(params.values[10]);
	let ph = i32(params.values[11]);
	let inPatch = x >= ox && x < ox + pw && y >= oy && y < oy + ph;
	var q = textureLoad(thermSrc, vec2<i32>(x, y));
	if (inPatch) {
		let dE = textureLoad(ePatchTex, vec2<i32>(x - ox, y - oy)).r;
		q = vec4<f32>(q.r + dE, q.g, q.b, q.a);
	}
	textureStore(thermDst, vec2<i32>(x, y), q);
}
`;
	// ---------- two level stats reduction, level 1 ----------
	shaders.statsReduceA=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(2) var thermTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(3) var velTex: texture_2d<f32>;
@group(0) @binding(4) var pressTex: texture_2d<f32>;
@group(0) @binding(5) var<storage,read_write> partials: array<f32>;

var<workgroup> shM: array<f32,64>;
var<workgroup> shDr: array<f32,64>;
var<workgroup> shDg: array<f32,64>;
var<workgroup> shDb: array<f32,64>;
var<workgroup> shE: array<f32,64>;
var<workgroup> shMinM: array<f32,64>;
var<workgroup> shMinDr: array<f32,64>;
var<workgroup> shMinDg: array<f32,64>;
var<workgroup> shMinDb: array<f32,64>;
var<workgroup> shMinE: array<f32,64>;
var<workgroup> shMaxM: array<f32,64>;
var<workgroup> shNeg: array<f32,64>;
var<workgroup> shOver: array<f32,64>;
var<workgroup> shMaxVel: array<f32,64>;
var<workgroup> shPMin: array<f32,64>;
var<workgroup> shPMax: array<f32,64>;
var<workgroup> shMaxDiv: array<f32,64>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let dt = params.values[4];
	let x = i32(gid.x);
	let y = i32(gid.y);
	var totM = 0.0; var totDr = 0.0; var totDg = 0.0; var totDb = 0.0; var totE = 0.0;
	var minM = 1e30; var minDr = 1e30; var minDg = 1e30; var minDb = 1e30; var minE = 1e30;
	var maxM = -1e30; var neg = 0.0; var over = 0.0;
	var maxVel = 0.0; var pMin = 1e30; var pMax = -1e30; var maxDiv = 0.0;
	if (x >= 1 && x < w - 1 && y >= 1 && y < h - 1) {
		let mass = textureLoad(massTex, vec2<i32>(x, y));
		let therm = textureLoad(thermTex, vec2<i32>(x, y));
		totM = mass.a; totDr = mass.r; totDg = mass.g; totDb = mass.b; totE = therm.r;
		minM = mass.a; minDr = mass.r; minDg = mass.g; minDb = mass.b; minE = therm.r;
		maxM = mass.a;
		if (mass.a < 0.0 || mass.r < 0.0 || mass.g < 0.0 || mass.b < 0.0 || therm.r < 0.0) { neg = 1.0; }
		if (mass.a > 1.001) { over = 1.0; }
		let uL = textureLoad(velTex, vec2<i32>(x, y), 0).r;
		let uR = textureLoad(velTex, vec2<i32>(x + 1, y), 0).r;
		let vB = textureLoad(velTex, vec2<i32>(x, y), 0).g;
		let vT = textureLoad(velTex, vec2<i32>(x, y + 1), 0).g;
		maxVel = max(max(abs(uL), abs(uR)), max(abs(vB), abs(vT)));
		let div = abs(uR - uL + vT - vB) / max(dt, 1e-9);
		maxDiv = div;
		let p = textureLoad(pressTex, vec2<i32>(x, y), 0).r;
		pMin = p; pMax = p;
	}
	shM[lid] = totM; shDr[lid] = totDr; shDg[lid] = totDg; shDb[lid] = totDb; shE[lid] = totE;
	shMinM[lid] = minM; shMinDr[lid] = minDr; shMinDg[lid] = minDg; shMinDb[lid] = minDb; shMinE[lid] = minE;
	shMaxM[lid] = maxM; shNeg[lid] = neg; shOver[lid] = over;
	shMaxVel[lid] = maxVel; shPMin[lid] = pMin; shPMax[lid] = pMax; shMaxDiv[lid] = maxDiv;
	workgroupBarrier();
	var stride = 32u;
	loop {
		if (stride == 0u) { break; }
		if (lid < stride) {
			shM[lid] = shM[lid] + shM[lid + stride];
			shDr[lid] = shDr[lid] + shDr[lid + stride];
			shDg[lid] = shDg[lid] + shDg[lid + stride];
			shDb[lid] = shDb[lid] + shDb[lid + stride];
			shE[lid] = shE[lid] + shE[lid + stride];
			shMinM[lid] = min(shMinM[lid], shMinM[lid + stride]);
			shMinDr[lid] = min(shMinDr[lid], shMinDr[lid + stride]);
			shMinDg[lid] = min(shMinDg[lid], shMinDg[lid + stride]);
			shMinDb[lid] = min(shMinDb[lid], shMinDb[lid + stride]);
			shMinE[lid] = min(shMinE[lid], shMinE[lid + stride]);
			shMaxM[lid] = max(shMaxM[lid], shMaxM[lid + stride]);
			shNeg[lid] = shNeg[lid] + shNeg[lid + stride];
			shOver[lid] = shOver[lid] + shOver[lid + stride];
			shMaxVel[lid] = max(shMaxVel[lid], shMaxVel[lid + stride]);
			shPMin[lid] = min(shPMin[lid], shPMin[lid + stride]);
			shPMax[lid] = max(shPMax[lid], shPMax[lid + stride]);
			shMaxDiv[lid] = max(shMaxDiv[lid], shMaxDiv[lid + stride]);
		}
		workgroupBarrier();
		stride = stride >> 1u;
	}
	if (lid == 0u) {
		let wgIndex = wid.y * nwg.x + wid.x;
		let base = wgIndex * 32u;
		partials[base + 0u] = shM[0];
		partials[base + 1u] = shDr[0];
		partials[base + 2u] = shDg[0];
		partials[base + 3u] = shDb[0];
		partials[base + 4u] = shE[0];
		partials[base + 5u] = shMinM[0];
		partials[base + 6u] = shMinDr[0];
		partials[base + 7u] = shMinDg[0];
		partials[base + 8u] = shMinDb[0];
		partials[base + 9u] = shMinE[0];
		partials[base + 10u] = shMaxM[0];
		partials[base + 11u] = shNeg[0];
		partials[base + 12u] = shOver[0];
		partials[base + 13u] = shMaxVel[0];
		partials[base + 14u] = shPMin[0];
		partials[base + 15u] = shPMax[0];
		partials[base + 16u] = shMaxDiv[0];
	}
}
`;
	// ---------- level 2 with kahan sums ----------
	// plain sequential f32 summation here was the source of the false hud [warn]:
	// measured error +0.035 on m=5055 and +25 on E=6.2e6 vs the f64 ledger.
	shaders.statsReduceB=`
@group(0) @binding(0) var<storage,read> partials: array<f32>;
@group(0) @binding(1) var<storage,read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	if (gid.x != 0u) { return; }
	let num = dims.x / 32u;
	var cM = 0.0; var sM = 0.0;
	var cDr = 0.0; var sDr = 0.0;
	var cDg = 0.0; var sDg = 0.0;
	var cDb = 0.0; var sDb = 0.0;
	var cE = 0.0; var sE = 0.0;
	var neg = 0.0; var over = 0.0;
	var minM = 1e30; var minDr = 1e30; var minDg = 1e30; var minDb = 1e30; var minE = 1e30;
	var maxM = -1e30; var maxVel = 0.0; var pMin = 1e30; var pMax = -1e30; var maxDiv = 0.0;
	for (var i = 0u; i < num; i++) {
		let base = i * 32u;
		var y = partials[base + 0u] - cM; var t = sM + y; cM = (t - sM) - y; sM = t;
		y = partials[base + 1u] - cDr; t = sDr + y; cDr = (t - sDr) - y; sDr = t;
		y = partials[base + 2u] - cDg; t = sDg + y; cDg = (t - sDg) - y; sDg = t;
		y = partials[base + 3u] - cDb; t = sDb + y; cDb = (t - sDb) - y; sDb = t;
		y = partials[base + 4u] - cE; t = sE + y; cE = (t - sE) - y; sE = t;
		neg = neg + partials[base + 11u];
		over = over + partials[base + 12u];
		minM = min(minM, partials[base + 5u]);
		minDr = min(minDr, partials[base + 6u]);
		minDg = min(minDg, partials[base + 7u]);
		minDb = min(minDb, partials[base + 8u]);
		minE = min(minE, partials[base + 9u]);
		maxM = max(maxM, partials[base + 10u]);
		maxVel = max(maxVel, partials[base + 13u]);
		pMin = min(pMin, partials[base + 14u]);
		pMax = max(pMax, partials[base + 15u]);
		maxDiv = max(maxDiv, partials[base + 16u]);
	}
	if (minM == 1e30) { minM = 0.0; }
	if (minDr == 1e30) { minDr = 0.0; }
	if (minDg == 1e30) { minDg = 0.0; }
	if (minDb == 1e30) { minDb = 0.0; }
	if (minE == 1e30) { minE = 0.0; }
	if (maxM == -1e30) { maxM = 0.0; }
	if (pMin == 1e30) { pMin = 0.0; }
	if (pMax == -1e30) { pMax = 0.0; }
	stats[0u] = sM; stats[1u] = sDr; stats[2u] = sDg; stats[3u] = sDb; stats[4u] = sE;
	stats[5u] = minM; stats[6u] = minDr; stats[7u] = minDg; stats[8u] = minDb; stats[9u] = minE;
	stats[10u] = maxM; stats[11u] = neg; stats[12u] = over;
	stats[13u] = maxVel; stats[14u] = pMin; stats[15u] = pMax; stats[16u] = maxDiv;
}
`;
	// ---------- buoy field sample: one workgroup over the body footprint ----------
	// six raw sums; the cpu twin is Fluid.core.buoy.sampleCpu. the body normalises by
	// its own footprint area, so a rectangle clipped by a domain wall reads less fluid.
	shaders.buoySample=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(2) var velTex: texture_2d<f32>;
@group(0) @binding(3) var<storage,read_write> buoySample: array<f32>;

var<workgroup> shBlue: array<f32,64>;
var<workgroup> shMass: array<f32,64>;
var<workgroup> shU: array<f32,64>;
var<workgroup> shV: array<f32,64>;
var<workgroup> shMax: array<f32,64>;
var<workgroup> shCnt: array<f32,64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32) {
	let texW = i32(params.values[2]);
	let texH = i32(params.values[3]);
	let x0 = i32(params.values[88]);
	let y0 = i32(params.values[89]);
	let bw = i32(params.values[90]);
	let bh = i32(params.values[91]);
	if (bw <= 0 || bh <= 0) {
		if (lid == 0u) {
			for (var i = 0u; i < 6u; i++) { buoySample[i] = 0.0; }
		}
		return;
	}
	let wu = u32(bw);
	let n = u32(bw * bh);
	var blue = 0.0; var mass = 0.0; var su = 0.0; var sv = 0.0; var mx = 0.0; var cnt = 0.0;
	var k = lid;
	loop {
		if (k >= n) { break; }
		let cx = x0 + i32(k % wu);
		let cy = y0 + i32(k / wu);
		if (cx >= 1 && cx < texW - 1 && cy >= 1 && cy < texH - 1) {
			let m = textureLoad(massTex, vec2<i32>(cx, cy));
			blue = blue + m.b;
			mass = mass + m.a;
			mx = max(mx, m.b);
			let v = textureLoad(velTex, vec2<i32>(cx, cy), 0);
			su = su + v.r;
			sv = sv + v.g;
			cnt = cnt + 1.0;
		}
		k = k + 64u;
	}
	shBlue[lid] = blue; shMass[lid] = mass; shU[lid] = su;
	shV[lid] = sv; shMax[lid] = mx; shCnt[lid] = cnt;
	workgroupBarrier();
	var stride = 32u;
	loop {
		if (stride == 0u) { break; }
		if (lid < stride) {
			shBlue[lid] = shBlue[lid] + shBlue[lid + stride];
			shMass[lid] = shMass[lid] + shMass[lid + stride];
			shU[lid] = shU[lid] + shU[lid + stride];
			shV[lid] = shV[lid] + shV[lid + stride];
			shMax[lid] = max(shMax[lid], shMax[lid + stride]);
			shCnt[lid] = shCnt[lid] + shCnt[lid + stride];
		}
		workgroupBarrier();
		stride = stride >> 1u;
	}
	if (lid == 0u) {
		buoySample[0u] = shBlue[0];
		buoySample[1u] = shMass[0];
		buoySample[2u] = shCnt[0];
		buoySample[3u] = shU[0];
		buoySample[4u] = shV[0];
		buoySample[5u] = shMax[0];
	}
}
`;
	// ---------- view modes ----------
	shaders.toDisplay=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(2) var thermTex: texture_storage_2d<rgba32float,read>;
@group(0) @binding(3) var displayTex: texture_storage_2d<rgba16float,write>;
@group(0) @binding(4) var velTex: texture_2d<f32>;
@group(0) @binding(5) var pressTex: texture_2d<f32>;
@group(0) @binding(6) var flagsTex: texture_2d<u32>;

fn tempColor(t: f32, tmin: f32, tmax: f32) -> vec3<f32> {
	let u = clamp((t - tmin) / max(tmax - tmin, 1e-6), 0.0, 1.0);
	if (u < 0.25) { return vec3<f32>(0.0, u / 0.25, 1.0); }
	if (u < 0.5) { return vec3<f32>(0.0, 1.0, 1.0 - (u - 0.25) / 0.25); }
	if (u < 0.75) { return vec3<f32>((u - 0.5) / 0.25, 1.0, 0.0); }
	return vec3<f32>(1.0, 1.0 - (u - 0.75) / 0.5, 0.0);
}
fn hueRgb(hue: f32) -> vec3<f32> {
	let k = fract(vec3<f32>(hue, hue - 0.3333, hue + 0.3333));
	let c = clamp(abs(k * 6.0 - 3.0) - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
	return c;
}
// the body rectangle is drawn over every view mode: a smoky hull that still shows
// the fluid inside it, tinted up to the sampled submerged fraction, amber outline.
fn buoyOverlay(x: i32, y: i32, base: vec4<f32>) -> vec4<f32> {
	if (params.values[93] < 0.5) { return base; }
	let bx = x - i32(params.values[88]);
	let by = y - i32(params.values[89]);
	let bw = i32(params.values[90]);
	let bh = i32(params.values[91]);
	if (bx < 0 || bx >= bw || by < 0 || by >= bh) { return base; }
	let wet = f32(by) < params.values[92] * f32(bh);
	var hull = mix(base.rgb, vec3<f32>(0.07, 0.08, 0.1), 0.55);
	if (wet) { hull = mix(hull, vec3<f32>(0.15, 0.4, 1.0), 0.35); }
	if (bx == 0 || by == 0 || bx == bw - 1 || by == bh - 1) {
		return vec4<f32>(1.0, 0.78, 0.25, 1.0);
	}
	return vec4<f32>(hull, 1.0);
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 0 || x >= w || y < 0 || y >= h) { return; }
	let mode = i32(params.values[80]);
	let mass = textureLoad(massTex, vec2<i32>(x, y));
	let therm = textureLoad(thermTex, vec2<i32>(x, y));
	let m = mass.a;
	var out = vec4<f32>(0.02, 0.03, 0.04, 1.0);
	if (mode == 0) {
		let gain = params.values[87];
		let dye = clamp(mass.rgb * gain, vec3<f32>(0.0), vec3<f32>(1.0));
		out = vec4<f32>(dye, 1.0);
	} else if (mode == 1) {
		if (m > params.values[58]) {
			out = vec4<f32>(tempColor(therm.g, params.values[81], params.values[82]), 1.0);
		}
	} else if (mode == 2) {
		let pmin = params.values[83];
		let pmax = params.values[84];
		let u = clamp((textureLoad(pressTex, vec2<i32>(x, y), 0).r - pmin) / max(pmax - pmin, 1e-6) * 2.0 - 1.0, -1.0, 1.0);
		out = vec4<f32>(max(u, 0.0), 0.12 + 0.1 * (1.0 - abs(u)), max(-u, 0.0), 1.0);
	} else if (mode == 3) {
		let vmax = max(params.values[85], 1.0);
		let u = textureLoad(velTex, vec2<i32>(x, y), 0).r;
		let v = textureLoad(velTex, vec2<i32>(x, y), 0).g;
		let uR = textureLoad(velTex, vec2<i32>(x + 1, y), 0).r;
		let vT = textureLoad(velTex, vec2<i32>(x, y + 1), 0).g;
		let cx = 0.5 * (u + uR);
		let cy = 0.5 * (v + vT);
		let sp = clamp(length(vec2<f32>(cx, cy)) / vmax, 0.0, 1.0);
		let hue = atan2(cy, cx) / 6.2831853 + 0.5;
		let c = hueRgb(hue);
		out = vec4<f32>(c * sp + vec3<f32>(0.02), 1.0);
	} else if (mode == 4) {
		let dt = max(params.values[4], 1e-9);
		let uL = textureLoad(velTex, vec2<i32>(x, y), 0).r;
		let uR = textureLoad(velTex, vec2<i32>(x + 1, y), 0).r;
		let vB = textureLoad(velTex, vec2<i32>(x, y), 0).g;
		let vT = textureLoad(velTex, vec2<i32>(x, y + 1), 0).g;
		let div = abs(uR - uL + vT - vB) / dt;
		let t = clamp(log2(1.0 + div) / 10.0, 0.0, 1.0);
		out = vec4<f32>(t * t, t, 0.15 + 0.5 * t, 1.0);
	} else {
		// flags: painted walls over a mass greyscale
		let wall = textureLoad(flagsTex, vec2<i32>(x, y), 0).r > 0u;
		let g = clamp(m, 0.0, 1.0);
		out = vec4<f32>(g * 0.6, g * 0.6, g * 0.7, 1.0);
		if (wall) { out = vec4<f32>(0.9, 0.25, 0.2, 1.0); }
	}
	textureStore(displayTex, vec2<i32>(x, y), buoyOverlay(x, y, out));
}
`;
	shaders.renderVert=`
struct VSOut {
	@builtin(position) pos: vec4<f32>,
	@location(0) uv: vec2<f32>
};
@vertex
fn main(@builtin(vertex_index) vi: u32) -> VSOut {
	// NDC Y is up (-1 bottom), framebuffer Y is down (0 top), texture sampled Y is down (0 top).
	// Physics Y is up (0 bottom, N top) stored with low Y at low texel Y (0 top of storage = bottom physics).
	// To make physics bottom appear at screen bottom (framebuffer bottom), sample low physics (texture top)
	// from NDC bottom. So NDC bottom (-1) -> uv 0 (texture top), NDC top (+1 interpolated) -> uv 1 (texture bottom).
	// The previous uv (0,1),(2,1),(0,-1) did the opposite (physics top at screen bottom) -> inverted draw.
	var pos = array<vec2<f32>,3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);
	var uv = array<vec2<f32>,3>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(2.0, 0.0),
		vec2<f32>(0.0, 2.0)
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
	// displayTex is (N+2)x(N+2) with a 1-texel ghost ring; inset UV so canvas maps to interior 1..N
	let dims = textureDimensions(displayTex);
	let n = vec2<f32>(f32(dims.x) - 2, f32(dims.y) - 2);
	let tex = vec2<f32>(f32(dims.x), f32(dims.y));
	let uv2 = (uv * n + vec2<f32>(1.0, 1.0)) / tex;
	return textureSample(displayTex, samp, uv2);
}
`;
	Fluid.gpu.shaders=shaders;
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.shaders;
})(globalThis);
