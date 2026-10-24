(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	// every kernel mirrors src/core/solver.js (the numeric oracle) exactly:
	// same blocking predicates, same donor rules, same update order. if the two
	// disagree, 0.0.0-plan.md section 5 wins and both get fixed.
	// param offsets are Fluid.core.config.paramsOff.
	const EOS_HELPERS=`
struct PhaseState {
	T: f32,
	phiL: f32,
	phiG: f32,
	phase: u32,
	rhoStar: f32,
	S: f32,
	c: f32,
	k: f32
};
fn evalPhase(m: f32, E: f32) -> PhaseState {
	var o: PhaseState;
	let TAmb = params.values[20];
	let kLiq = params.values[35];
	let SMax = params.values[36];
	let tau = params.values[37];
	let kSol = params.values[45];
	let kGas = params.values[46];
	let TEps = params.values[47];
	let cSol = params.values[48];
	let cLiq = params.values[49];
	let cGas = params.values[50];
	let Lf = params.values[51];
	let Lv = params.values[52];
	let TMelt = params.values[53];
	let TBoil = params.values[54];
	let rhoLiq = params.values[55];
	let rhoGas0 = params.values[56];
	let mEps = params.values[58];
	if (!(m >= mEps)) {
		o.T = TAmb; o.phiL = 0.0; o.phiG = 0.0; o.phase = 0u;
		o.rhoStar = rhoLiq; o.S = 0.0; o.c = cLiq; o.k = kLiq;
		return o;
	}
	let E1 = m * cSol * TMelt;
	let E2 = E1 + m * Lf;
	let E3 = E2 + m * cLiq * (TBoil - TMelt);
	let E4 = E3 + m * Lv;
	var T = TAmb;
	var phiL = 0.0;
	var phiG = 0.0;
	var ph = 0u;
	if (E < E1) {
		T = E / (m * cSol); ph = 1u;
	} else if (E < E2) {
		T = TMelt; phiL = (E - E1) / (m * Lf); ph = 2u;
	} else if (E < E3) {
		T = TMelt + (E - E2) / (m * cLiq); phiL = 1.0; ph = 3u;
	} else if (E < E4) {
		T = TBoil; phiG = (E - E3) / (m * Lv); phiL = 1.0; ph = 4u;
	} else {
		T = TBoil + (E - E4) / (m * cGas); phiG = 1.0; phiL = 1.0; ph = 5u;
	}
	let rhoGas = rhoGas0 * TBoil / max(T, TEps);
	let vStar = (1.0 - phiG) / rhoLiq + phiG / rhoGas;
	let rhoStar = 1.0 / vStar;
	var S = 0.0;
	if (phiG > 0.0) {
		S = (m - rhoStar) / (max(m, rhoStar) * tau);
	} else if (m > rhoLiq) {
		S = (m - rhoLiq) / (m * tau);
	}
	S = clamp(S, -SMax, SMax);
	o.T = T;
	o.phiL = phiL;
	o.phiG = phiG;
	o.phase = ph;
	o.rhoStar = rhoStar;
	o.S = S;
	o.c = (1.0 - phiL) * cSol + phiL * ((1.0 - phiG) * cLiq + phiG * cGas);
	o.k = (1.0 - phiL) * kSol + phiL * ((1.0 - phiG) * kLiq + phiG * kGas);
	return o;
}
fn eqDist(m: f32, E: f32, T: f32) -> f32 {
	let cSol = params.values[48];
	let cLiq = params.values[49];
	let cGas = params.values[50];
	let Lf = params.values[51];
	let Lv = params.values[52];
	let TMelt = params.values[53];
	let TBoil = params.values[54];
	var lo = 0.0;
	var hi = 0.0;
	if (T < TMelt) {
		lo = m * cSol * T; hi = lo;
	} else if (T == TMelt) {
		lo = m * cSol * TMelt; hi = lo + m * Lf;
	} else if (T < TBoil) {
		lo = m * (cSol * TMelt + Lf + cLiq * (T - TMelt)); hi = lo;
	} else if (T == TBoil) {
		lo = m * (cSol * TMelt + Lf + cLiq * (TBoil - TMelt)); hi = lo + m * Lv;
	} else {
		lo = m * (cSol * TMelt + Lf + cLiq * (TBoil - TMelt) + Lv + cGas * (T - TBoil)); hi = lo;
	}
	if (E < lo) { return lo - E; }
	if (E > hi) { return E - hi; }
	return 0.0;
}
`;
	// the fine-grid poisson operator, shared by the two kernels that need it: the
	// v-cycle's restrict (which moves the residual down) and the stats pass (which
	// reports its norm). the caller supplies the five pressure reads, because the
	// solver holds press as a storage texture and the stats pass as a sampled one.
	// mirrors computeResidual in src/core/multigrid.js, the cpu oracle.
	const POISSON_HELPERS=`
struct PoissonP { left: f32, right: f32, down: f32, up: f32, center: f32 };
struct PoissonOut { res: f32, rhs: f32 };
fn poissonOpen(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return false; }
	let fl = textureLoad(flagsTex, vec2<i32>(x, y), 0);
	return (fl.r | fl.b) == 0u;
}
// harmonic-mean face weight of the inverse density, zero across a wall or frozen cell
fn poissonFaceW(x: i32, y: i32, nx: i32, ny: i32, w: i32, h: i32) -> f32 {
	if (nx < 1 || nx > w - 2 || ny < 1 || ny > h - 2) { return 0.0; }
	let nf = textureLoad(flagsTex, vec2<i32>(nx, ny), 0);
	if ((nf.r | nf.b) > 0u) { return 0.0; }
	let rhoVac = params.values[57];
	let ra = max(textureLoad(massTex, vec2<i32>(x, y), 0).a, rhoVac);
	let rb = max(textureLoad(massTex, vec2<i32>(nx, ny), 0).a, rhoVac);
	return 1.0 / (2.0 * ra * rb / (ra + rb));
}
fn poissonResidual(p: PoissonP, x: i32, y: i32, w: i32, h: i32) -> PoissonOut {
	var out: PoissonOut;
	out.res = 0.0; out.rhs = 0.0;
	if (!poissonOpen(x, y, w, h)) { return out; }
	out.rhs = textureLoad(divsrcTex, vec2<i32>(x, y), 0).r;
	var wsum = 0.0;
	var sum = 0.0;
	var fw = poissonFaceW(x, y, x - 1, y, w, h);
	if (fw > 0.0) { wsum = wsum + fw; sum = sum + fw * p.left; }
	fw = poissonFaceW(x, y, x + 1, y, w, h);
	if (fw > 0.0) { wsum = wsum + fw; sum = sum + fw * p.right; }
	fw = poissonFaceW(x, y, x, y - 1, w, h);
	if (fw > 0.0) { wsum = wsum + fw; sum = sum + fw * p.down; }
	fw = poissonFaceW(x, y, x, y + 1, w, h);
	if (fw > 0.0) { wsum = wsum + fw; sum = sum + fw * p.up; }
	out.res = out.rhs - (sum - wsum * p.center);
	return out;
}
`;
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
@group(0) @binding(6) var<storage,read> emitters: array<f32>;
@group(0) @binding(7) var<storage,read> emitterLedger: array<f32>;

fn isBlocked(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	let f = textureLoad(flagsTex, vec2<i32>(x, y), 0);
	return (f.r | f.b) > 0u;
}
fn uBlocked(x: i32, y: i32, w: i32, h: i32) -> bool {
	return x <= 1 || x >= w - 1 || isBlocked(x - 1, y, w, h) || isBlocked(x, y, w, h);
}
fn vBlocked(x: i32, y: i32, w: i32, h: i32) -> bool {
	return y <= 1 || y >= h - 1 || isBlocked(x, y - 1, w, h) || isBlocked(x, y, w, h);
}
fn cellAccelX(x: i32, y: i32, w: i32, h: i32, gx: f32, rhoMean: f32, rhoRef: f32, beta: f32, tref: f32, mEps: f32) -> f32 {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return 0.0; }
	let m = textureLoad(massTex, vec2<i32>(x, y), 0).a;
	let T = textureLoad(thermTex, vec2<i32>(x, y), 0).g;
	let tb = select(0.0, gx * beta * (T - tref), m >= mEps);
	return gx * (rhoMean - m) / rhoRef + tb;
}
fn cellAccelY(x: i32, y: i32, w: i32, h: i32, gy: f32, rhoMean: f32, rhoRef: f32, beta: f32, tref: f32, mEps: f32) -> f32 {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return 0.0; }
	let m = textureLoad(massTex, vec2<i32>(x, y), 0).a;
	let T = textureLoad(thermTex, vec2<i32>(x, y), 0).g;
	let tb = select(0.0, gy * beta * (T - tref), m >= mEps);
	return gy * (rhoMean - m) / rhoRef + tb;
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
	let invN = params.values[1];
	let gx = params.values[16];
	let gy = params.values[17];
	let rhoRef = params.values[18];
	let beta = params.values[19];
	let tref = params.values[20];
	let rhoMean = params.values[95] + (emitterLedger[0] - emitterLedger[5]) * invN * invN;
	let ms = params.values[23];
	let mvx = params.values[24];
	let mvy = params.values[25];
	let mr = params.values[26];
	let vclamp = params.values[38];
	let mEps = params.values[58];
	let momCnt = i32(emitters[2]);
	let emMax = i32(emitters[1]);
	let src = textureLoad(velSrc, vec2<i32>(x, y));
	var u = 0.0;
	var v = 0.0;
	if (!uBlocked(x, y, w, h)) {
		u = src.r + dt * 0.5 * (cellAccelX(x - 1, y, w, h, gx, rhoMean, rhoRef, beta, tref, mEps) + cellAccelX(x, y, w, h, gx, rhoMean, rhoRef, beta, tref, mEps));
		if (ms > 0.0) {
			let f = mouseFall(f32(x - 1) - params.values[21], f32(y) - 0.5 - params.values[22], mr);
			u = u + dt * mvx * f * ms;
		}
		if (momCnt > 0) {
			let px = f32(x - 1);
			let py = f32(y) - 0.5;
			for (var s = 0; s < emMax; s = s + 1) {
				let b = 16 + s * 16;
				if (i32(emitters[b + 4]) == 4) {
					let dx = px - emitters[b];
					let dy = py - emitters[b + 1];
					let d2 = dx * dx + dy * dy;
					if (d2 < emitters[b + 3]) {
						let tf = max(0.0, 1.0 - d2 * emitters[b + 2]);
						let fw = tf * tf;
						u = u + dt * (emitters[b + 9] * fw);
					}
				}
			}
		}
		u = clamp(u, -vclamp, vclamp);
	}
	if (!vBlocked(x, y, w, h)) {
		v = src.g + dt * 0.5 * (cellAccelY(x, y - 1, w, h, gy, rhoMean, rhoRef, beta, tref, mEps) + cellAccelY(x, y, w, h, gy, rhoMean, rhoRef, beta, tref, mEps));
		if (ms > 0.0) {
			let f = mouseFall(f32(x) - 0.5 - params.values[21], f32(y - 1) - params.values[22], mr);
			v = v + dt * mvy * f * ms;
		}
		if (momCnt > 0) {
			let px = f32(x) - 0.5;
			let py = f32(y - 1);
			for (var s = 0; s < emMax; s = s + 1) {
				let b = 16 + s * 16;
				if (i32(emitters[b + 4]) == 4) {
					let dx = px - emitters[b];
					let dy = py - emitters[b + 1];
					let d2 = dx * dx + dy * dy;
					if (d2 < emitters[b + 3]) {
						let tf = max(0.0, 1.0 - d2 * emitters[b + 2]);
						let fw = tf * tf;
						v = v + dt * (emitters[b + 10] * fw);
					}
				}
			}
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
@group(0) @binding(3) var flagsTex: texture_2d<u32>;

fn isBlockedAV(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	let f = textureLoad(flagsTex, vec2<i32>(x, y), 0);
	return (f.r | f.b) > 0u;
}
fn uBlockedAV(x: i32, y: i32, w: i32, h: i32) -> bool {
	return x <= 1 || x >= w - 1 || isBlockedAV(x - 1, y, w, h) || isBlockedAV(x, y, w, h);
}
fn vBlockedAV(x: i32, y: i32, w: i32, h: i32) -> bool {
	return y <= 1 || y >= h - 1 || isBlockedAV(x, y - 1, w, h) || isBlockedAV(x, y, w, h);
}
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
	if (!uBlockedAV(x, y, w, h)) {
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
	}
	if (!vBlockedAV(x, y, w, h)) {
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
	}
	textureStore(velDst, vec2<i32>(x, y), vec4<f32>(u, v, 0.0, 0.0));
}
`;
	// ---------- mac-cormack correction (t16): pass 2 of velocity advection.
	// pass 1 is the plain semi-lagrangian kernel above, dispatched into the
	// macTmp scratch. this pass traces that field forward along the same face
	// velocity and corrects with c = f + (src - back)/2, limited into the
	// min/max of the four taps the backward sample used. mirrors
	// advectVelMac in src/core/solver.js: same traces, same clamps, same
	// limiter, same absolute velocity clamp (the cpu counts its hits in
	// diag.macClamp; the device has no counter, like the forces clamp).
	shaders.advectVelMac=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var velSrc: texture_storage_2d<rg32float,read>;
@group(0) @binding(2) var macSrc: texture_storage_2d<rg32float,read>;
@group(0) @binding(3) var velDst: texture_storage_2d<rg32float,write>;
@group(0) @binding(4) var flagsTex: texture_2d<u32>;

fn isBlockedM(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	let f = textureLoad(flagsTex, vec2<i32>(x, y), 0);
	return (f.r | f.b) > 0u;
}
fn uBlockedM(x: i32, y: i32, w: i32, h: i32) -> bool {
	return x <= 1 || x >= w - 1 || isBlockedM(x - 1, y, w, h) || isBlockedM(x, y, w, h);
}
fn vBlockedM(x: i32, y: i32, w: i32, h: i32) -> bool {
	return y <= 1 || y >= h - 1 || isBlockedM(x, y - 1, w, h) || isBlockedM(x, y, w, h);
}
fn loadUM(t: texture_storage_2d<rg32float,read>, m: i32, l: i32, w: i32, h: i32) -> f32 {
	let mm = clamp(m, 0, w - 1);
	let ll = clamp(l, 0, h - 1);
	return textureLoad(t, vec2<i32>(mm, ll)).r;
}
fn loadVM(t: texture_storage_2d<rg32float,read>, m: i32, l: i32, w: i32, h: i32) -> f32 {
	let mm = clamp(m, 0, w - 1);
	let ll = clamp(l, 0, h - 1);
	return textureLoad(t, vec2<i32>(mm, ll)).g;
}
struct SampM { val: f32, lo: f32, hi: f32 };
fn sampleUM(t: texture_storage_2d<rg32float,read>, px: f32, py: f32, w: i32, h: i32) -> SampM {
	let i0 = i32(floor(px));
	let fx = px - f32(i0);
	let l0 = i32(floor(py + 0.5));
	let fy = py + 0.5 - f32(l0);
	let u00 = loadUM(t, i0 + 1, l0, w, h);
	let u10 = loadUM(t, i0 + 2, l0, w, h);
	let u01 = loadUM(t, i0 + 1, l0 + 1, w, h);
	let u11 = loadUM(t, i0 + 2, l0 + 1, w, h);
	let a0 = u00 + (u10 - u00) * fx;
	let a1 = u01 + (u11 - u01) * fx;
	var s: SampM;
	s.val = a0 + (a1 - a0) * fy;
	s.lo = min(min(u00, u10), min(u01, u11));
	s.hi = max(max(u00, u10), max(u01, u11));
	return s;
}
fn sampleVM(t: texture_storage_2d<rg32float,read>, px: f32, py: f32, w: i32, h: i32) -> SampM {
	let k0 = i32(floor(px + 0.5));
	let fx = px + 0.5 - f32(k0);
	let l0 = i32(floor(py));
	let fy = py - f32(l0);
	let v00 = loadVM(t, k0, l0 + 1, w, h);
	let v10 = loadVM(t, k0 + 1, l0 + 1, w, h);
	let v01 = loadVM(t, k0, l0 + 2, w, h);
	let v11 = loadVM(t, k0 + 1, l0 + 2, w, h);
	let b0 = v00 + (v10 - v00) * fx;
	let b1 = v01 + (v11 - v01) * fx;
	var s: SampM;
	s.val = b0 + (b1 - b0) * fy;
	s.lo = min(min(v00, v10), min(v01, v11));
	s.hi = max(max(v00, v10), max(v01, v11));
	return s;
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
	let vclamp = params.values[38];
	let lim = params.values[61] > 0.0;
	let src = textureLoad(velSrc, vec2<i32>(x, y));
	var u = 0.0;
	var v = 0.0;
	if (!uBlockedM(x, y, w, h)) {
		let uc = src.r;
		let vc = 0.25 * (loadVM(velSrc, x - 1, y, w, h) + loadVM(velSrc, x - 1, y + 1, w, h)
			+ loadVM(velSrc, x, y, w, h) + loadVM(velSrc, x, y + 1, w, h));
		let fs = sampleUM(velSrc, clamp(f32(x - 1) - dt * uc, 0.0, f32(n)),
			clamp(f32(y) - 0.5 - dt * vc, 0.5, f32(n) - 0.5), w, h);
		let back = sampleUM(macSrc, clamp(f32(x - 1) + dt * uc, 0.0, f32(n)),
			clamp(f32(y) - 0.5 + dt * vc, 0.5, f32(n) - 0.5), w, h).val;
		var c = fs.val + 0.5 * (uc - back);
		if (lim) { c = clamp(c, fs.lo, fs.hi); }
		u = clamp(c, -vclamp, vclamp);
	}
	if (!vBlockedM(x, y, w, h)) {
		let vc2 = src.g;
		let uc2 = 0.25 * (loadUM(velSrc, x, y - 1, w, h) + loadUM(velSrc, x + 1, y - 1, w, h)
			+ loadUM(velSrc, x, y, w, h) + loadUM(velSrc, x + 1, y, w, h));
		let fs = sampleVM(velSrc, clamp(f32(x) - 0.5 - dt * uc2, 0.5, f32(n) - 0.5),
			clamp(f32(y - 1) - dt * vc2, 0.0, f32(n)), w, h);
		let back = sampleVM(macSrc, clamp(f32(x) - 0.5 + dt * uc2, 0.5, f32(n) - 0.5),
			clamp(f32(y - 1) + dt * vc2, 0.0, f32(n)), w, h).val;
		var c = fs.val + 0.5 * (vc2 - back);
		if (lim) { c = clamp(c, fs.lo, fs.hi); }
		v = clamp(c, -vclamp, vclamp);
	}
	textureStore(velDst, vec2<i32>(x, y), vec4<f32>(u, v, 0.0, 0.0));
}
`;
	// ---------- poisson rhs reading S one step behind from sfield ----------
	shaders.poissonRhs=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var divsrc: texture_storage_2d<r32float,write>;
@group(0) @binding(2) var velTex: texture_2d<f32>;
@group(0) @binding(3) var sfieldTex: texture_2d<f32>;
@group(0) @binding(4) var flagsTex: texture_2d<u32>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	var b = 0.0;
	let fl = textureLoad(flagsTex, vec2<i32>(x, y), 0);
	if ((fl.r | fl.b) == 0u) {
		let dt = params.values[4];
		let uL = textureLoad(velTex, vec2<i32>(x, y), 0).r;
		let uR = textureLoad(velTex, vec2<i32>(x + 1, y), 0).r;
		let vB = textureLoad(velTex, vec2<i32>(x, y), 0).g;
		let vT = textureLoad(velTex, vec2<i32>(x, y + 1), 0).g;
		let S = textureLoad(sfieldTex, vec2<i32>(x, y), 0).r;
		b = (uR - uL + vT - vB) / dt - S;
	}
	textureStore(divsrc, vec2<i32>(x, y), vec4<f32>(b, 0.0, 0.0, 0.0));
}
`;
	// ---------- one red-black gauss-seidel color sweep ----------
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
	let nf = textureLoad(flagsTex, vec2<i32>(nx, ny), 0);
	if ((nf.r | nf.b) > 0u) { return 0.0; }
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
		let fl = textureLoad(flagsTex, vec2<i32>(x, y), 0);
		if ((fl.r | fl.b) > 0u) {
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

fn isBlocked2(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	let f = textureLoad(flagsTex, vec2<i32>(x, y), 0);
	return (f.r | f.b) > 0u;
}
fn uBlocked2(x: i32, y: i32, w: i32, h: i32) -> bool {
	return x <= 1 || x >= w - 1 || isBlocked2(x - 1, y, w, h) || isBlocked2(x, y, w, h);
}
fn vBlocked2(x: i32, y: i32, w: i32, h: i32) -> bool {
	return y <= 1 || y >= h - 1 || isBlocked2(x, y - 1, w, h) || isBlocked2(x, y, w, h);
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

fn isBlocked3(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	let f = textureLoad(flagsTex, vec2<i32>(x, y), 0);
	return (f.r | f.b) > 0u;
}
fn rawFx(x: i32, y: i32, w: i32, h: i32, dt: f32) -> f32 {
	if (x <= 1 || x >= w - 1 || isBlocked3(x - 1, y, w, h) || isBlocked3(x, y, w, h)) { return 0.0; }
	return clamp(textureLoad(velTex, vec2<i32>(x, y), 0).r * dt, -1.0, 1.0);
}
fn rawFy(x: i32, y: i32, w: i32, h: i32, dt: f32) -> f32 {
	if (y <= 1 || y >= h - 1 || isBlocked3(x, y - 1, w, h) || isBlocked3(x, y, w, h)) { return 0.0; }
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
	if (fx != 0.0) {
		let s = select(outFlux(x, y, w, h, dt), outFlux(x - 1, y, w, h, dt), fx > 0.0);
		if (s > 0.999999) { fx = fx * (0.999999 / s); }
	}
	if (fy != 0.0) {
		let s = select(outFlux(x, y, w, h, dt), outFlux(x, y - 1, w, h, dt), fy > 0.0);
		if (s > 0.999999) { fy = fy * (0.999999 / s); }
	}
	textureStore(fluxDst, vec2<i32>(x, y), vec4<f32>(fx, fy, 0.0, 0.0));
}
`;
	// ---------- donor cell upwind transport ----------
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
	let remaining = 1.0 - (max(-fw, 0.0) + max(fe, 0.0) + max(-fs, 0.0) + max(fN, 0.0));
	let out = q * remaining + qW * max(fw, 0.0) + qE * max(-fe, 0.0)
		+ qS * max(fs, 0.0) + qN * max(-fN, 0.0);
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
	let remaining = 1.0 - (max(-fw, 0.0) + max(fe, 0.0) + max(-fs, 0.0) + max(fN, 0.0));
	let out = q * remaining + qW * max(fw, 0.0) + qE * max(-fe, 0.0)
		+ qS * max(fs, 0.0) + qN * max(-fN, 0.0);
	textureStore(thermDst, vec2<i32>(x, y), out);
}
`;
	// 0.6.0 t8: fused donor-cell transport for mass and therm (mirrors cpu advectScalar)
	shaders.advectScalar=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var fluxTex: texture_2d<f32>;
@group(0) @binding(2) var massSrc: texture_2d<f32>;
@group(0) @binding(3) var thermSrc: texture_2d<f32>;
@group(0) @binding(4) var massDst: texture_storage_2d<rgba32float,write>;
@group(0) @binding(5) var thermDst: texture_storage_2d<rgba32float,write>;

fn donorUpwind(src: texture_2d<f32>, x: i32, y: i32, rem: f32, wW: f32, wE: f32, wS: f32, wN: f32) -> vec4<f32> {
	return textureLoad(src, vec2<i32>(x, y), 0) * rem
		+ textureLoad(src, vec2<i32>(x - 1, y), 0) * wW
		+ textureLoad(src, vec2<i32>(x + 1, y), 0) * wE
		+ textureLoad(src, vec2<i32>(x, y - 1), 0) * wS
		+ textureLoad(src, vec2<i32>(x, y + 1), 0) * wN;
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let fw = textureLoad(fluxTex, vec2<i32>(x, y), 0).r;
	let fe = textureLoad(fluxTex, vec2<i32>(x + 1, y), 0).r;
	let fs = textureLoad(fluxTex, vec2<i32>(x, y), 0).g;
	let fN = textureLoad(fluxTex, vec2<i32>(x, y + 1), 0).g;
	let rem = 1.0 - (max(-fw, 0.0) + max(fe, 0.0) + max(-fs, 0.0) + max(fN, 0.0));
	let wW = max(fw, 0.0);
	let wE = max(-fe, 0.0);
	let wS = max(fs, 0.0);
	let wN = max(-fN, 0.0);
	textureStore(massDst, vec2<i32>(x, y), donorUpwind(massSrc, x, y, rem, wW, wE, wS, wN));
	textureStore(thermDst, vec2<i32>(x, y), donorUpwind(thermSrc, x, y, rem, wW, wE, wS, wN));
}
`;
	// ---------- 0.5.0 phase_eos + freezeVel (0.6.0 t8: APPLY_SOURCES=1 folds heatSources in) ----------
	shaders.phaseEos=EOS_HELPERS+`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_2d<f32>;
@group(0) @binding(2) var thermSrc: texture_2d<f32>;
@group(0) @binding(3) var velTex: texture_2d<f32>;
@group(0) @binding(4) var flagsSrc: texture_2d<u32>;
@group(0) @binding(5) var sfieldSrc: texture_2d<f32>;
@group(0) @binding(6) var accumSrc: texture_2d<f32>;
@group(0) @binding(7) var thermDst: texture_storage_2d<rgba32float,write>;
@group(0) @binding(8) var flagsDst: texture_storage_2d<rgba8uint,write>;
@group(0) @binding(9) var sfieldDst: texture_storage_2d<rgba32float,write>;
@group(0) @binding(10) var accumDst: texture_storage_2d<rg32float,write>;
override APPLY_SOURCES: u32 = 0u;

fn applyHeatSources(qIn: vec4<f32>, m: f32, wall: u32, y: i32) -> vec4<f32> {
	if (params.values[43] < 0.5 || m < params.values[58] || wall > 0u) { return qIn; }
	let band = max(i32(round(params.values[42])), 1);
	let n = i32(params.values[0]);
	let inBot = y <= band;
	let inTop = y > n - band;
	if (!inBot && !inTop) { return qIn; }
	let dt = params.values[4];
	var dE = 0.0;
	if (inBot && (params.values[44] == 0.0 || qIn.g < params.values[44])) {
		dE = dE + params.values[39] * dt;
	}
	if (inTop) {
		let own = evalPhase(m, qIn.r);
		let ceilT = select(params.values[20], params.values[60], params.values[60] > 0.0);
		dE = dE - ((params.values[40] * (qIn.g - ceilT)) * (m * own.c)) * dt;
	}
	if (dE == 0.0) { return qIn; }
	return vec4<f32>(qIn.r + dE, qIn.g, qIn.b + max(dE, 0.0), qIn.a + max(-dE, 0.0));
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x >= w || y >= h) { return; }
	let at = vec2<i32>(x, y);
	var q = textureLoad(thermSrc, at, 0);
	let prevFl = textureLoad(flagsSrc, at, 0);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) {
		textureStore(thermDst, at, q);
		textureStore(flagsDst, at, prevFl);
		textureStore(sfieldDst, at, vec4<f32>(0.0));
		textureStore(accumDst, at, vec4<f32>(0.0));
		return;
	}
	let m = textureLoad(massTex, at, 0).a;
	if (APPLY_SOURCES > 0u) { q = applyHeatSources(q, m, prevFl.r, y); }
	let s = evalPhase(m, q.r);
	let prevS = textureLoad(sfieldSrc, at, 0);
	let prevAcc = textureLoad(accumSrc, at, 0);
	let Lf = params.values[51];
	let Lv = params.values[52];
	let SMax = params.values[36];
	let mEps = params.values[58];
	var latAcc = prevAcc.r;
	if (params.values[59] > 0.5) {
		latAcc = latAcc + m * Lf * abs(s.phiL - prevS.g) + m * Lv * abs(s.phiG - prevS.b);
	}
	let frozen = s.phiL < 0.5 && s.phiG == 0.0 && m >= mEps;
	var momAcc = prevAcc.g;
	if (params.values[59] > 0.5 && frozen && prevFl.b == 0u) {
		let uL = textureLoad(velTex, at, 0).r;
		let uR = textureLoad(velTex, vec2<i32>(x + 1, y), 0).r;
		let vB = textureLoad(velTex, at, 0).g;
		let vT = textureLoad(velTex, vec2<i32>(x, y + 1), 0).g;
		let uc = 0.5 * (uL + uR);
		let vc = 0.5 * (vB + vT);
		momAcc = momAcc + m * sqrt(uc * uc + vc * vc);
	}
	let sat = abs(s.S) >= SMax;
	let rClean = select(prevFl.r & 3u, 255u, prevFl.r == 255u);
	textureStore(thermDst, at, vec4<f32>(q.r, s.T, q.b, q.a));
	textureStore(flagsDst, at, vec4<u32>(rClean, s.phase, select(0u, 1u, frozen), select(0u, 1u, sat)));
	textureStore(sfieldDst, at, vec4<f32>(s.S, s.phiL, s.phiG, 0.0));
	textureStore(accumDst, at, vec4<f32>(latAcc, momAcc, 0.0, 0.0));
}
`;
	shaders.freezeVel=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var velSrc: texture_2d<f32>;
@group(0) @binding(2) var flagsTex: texture_2d<u32>;
@group(0) @binding(3) var velDst: texture_storage_2d<rg32float,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let at = vec2<i32>(x, y);
	let src = textureLoad(velSrc, at, 0);
	let fC = textureLoad(flagsTex, at, 0).b > 0u;
	let fW = x <= 1 || textureLoad(flagsTex, vec2<i32>(x - 1, y), 0).b > 0u;
	let fS = y <= 1 || textureLoad(flagsTex, vec2<i32>(x, y - 1), 0).b > 0u;
	let u = select(src.r, 0.0, fC || fW);
	let v = select(src.g, 0.0, fC || fS);
	textureStore(velDst, at, vec4<f32>(u, v, 0.0, 0.0));
}
`;
	// ---------- conduction (0.5.0 phase-aware + enthalpy budget) ----------
	shaders.conduction=EOS_HELPERS+`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_2d<f32>;
@group(0) @binding(2) var thermSrc: texture_2d<f32>;
@group(0) @binding(3) var thermDst: texture_storage_2d<rgba32float,write>;
@group(0) @binding(4) var flagsTex: texture_2d<u32>;

fn isWallAt(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	return textureLoad(flagsTex, vec2<i32>(x, y), 0).r > 0u;
}
fn isStaticWallAt(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return true; }
	return (textureLoad(flagsTex, vec2<i32>(x, y), 0).r & 1u) > 0u;
}
fn faceExchange(own: PhaseState, nx: i32, ny: i32, w: i32, h: i32, mL: f32, EL: f32, dt: f32) -> f32 {
	if (nx < 1 || nx > w - 2 || ny < 1 || ny > h - 2) { return 0.0; }
	let at = vec2<i32>(nx, ny);
	if (textureLoad(flagsTex, at, 0).r > 0u) { return 0.0; }
	let mR = textureLoad(massTex, at, 0).a;
	if (mR < params.values[58]) { return 0.0; }
	let ER = textureLoad(thermSrc, at, 0).r;
	let nb = evalPhase(mR, ER);
	let dT = nb.T - own.T;
	if (dT == 0.0) { return 0.0; }
	let CL = mL * own.c;
	let CR = mR * nb.c;
	let Ch = (CL * CR) / (CL + CR);
	let kf = (2.0 * own.k * nb.k) / (own.k + nb.k);
	var dE = (kf * dT) * dt;
	let lim = (0.5 * Ch) * abs(dT);
	if (abs(dE) > lim) { dE = select(-lim, lim, dE > 0.0); }
	let bud = 0.25 * min(eqDist(mL, EL, nb.T), eqDist(mR, ER, own.T));
	if (abs(dE) > bud) { dE = select(-bud, bud, dE > 0.0); }
	return dE;
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let at = vec2<i32>(x, y);
	let dt = params.values[4];
	let m = textureLoad(massTex, at, 0).a;
	var q = textureLoad(thermSrc, at, 0);
	let E0 = q.r;
	var E = E0;
	if (m >= params.values[58] && !isWallAt(x, y, w, h)) {
		let own = evalPhase(m, E0);
		E = E + faceExchange(own, x - 1, y, w, h, m, E0, dt);
		E = E + faceExchange(own, x + 1, y, w, h, m, E0, dt);
		E = E + faceExchange(own, x, y - 1, w, h, m, E0, dt);
		E = E + faceExchange(own, x, y + 1, w, h, m, E0, dt);
		let wallCool = params.values[41];
		if (wallCool > 0.0 && (isStaticWallAt(x - 1, y, w, h) || isStaticWallAt(x + 1, y, w, h) || isStaticWallAt(x, y - 1, w, h) || isStaticWallAt(x, y + 1, w, h))) {
			let dE = (((-wallCool) * (q.g - params.values[20])) * (m * own.c)) * dt;
			E = E + dE;
			q.b = q.b + max(dE, 0.0);
			q.a = q.a + max(-dE, 0.0);
		}
	}
	q.r = E;
	q.g = evalPhase(m, E).T;
	textureStore(thermDst, at, q);
}
`;
	// ---------- heat sources (0.4.0 / 0.5.0 phase-aware capacity + ceiling target) ----------
	shaders.heatSources=EOS_HELPERS+`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_2d<f32>;
@group(0) @binding(2) var thermSrc: texture_2d<f32>;
@group(0) @binding(3) var thermDst: texture_storage_2d<rgba32float,write>;
@group(0) @binding(4) var flagsTex: texture_2d<u32>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) { return; }
	let at = vec2<i32>(x, y);
	var q = textureLoad(thermSrc, at, 0);
	if (params.values[43] > 0.5) {
		let m = textureLoad(massTex, at, 0).a;
		let wall = textureLoad(flagsTex, at, 0).r > 0u;
		if (m >= params.values[58] && !wall) {
			let own = evalPhase(m, q.r);
			let dt = params.values[4];
			let band = max(i32(round(params.values[42])), 1);
			let n = i32(params.values[0]);
			let ceilT = select(params.values[20], params.values[60], params.values[60] > 0.0);
			var dE = 0.0;
			if (y <= band && (params.values[44] == 0.0 || q.g < params.values[44])) { dE = dE + params.values[39] * dt; }
			if (y > n - band) { dE = dE - ((params.values[40] * (q.g - ceilT)) * (m * own.c)) * dt; }
			if (dE != 0.0) {
				let nextE = q.r + dE;
				q = vec4<f32>(nextE, evalPhase(m, nextE).T, q.b + max(dE, 0.0), q.a + max(-dE, 0.0));
			}
		}
	}
	textureStore(thermDst, at, q);
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
	// ---------- 0.7.0 t4 obstacles & moving colliders ----------
	shaders.paintSolid=EOS_HELPERS+`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_2d<f32>;
@group(0) @binding(2) var thermSrc: texture_2d<f32>;
@group(0) @binding(3) var flagsTex: texture_2d<u32>;
@group(0) @binding(4) var thermDst: texture_storage_2d<rgba32float,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x >= w || y >= h) { return; }
	let at = vec2<i32>(x, y);
	let q = textureLoad(thermSrc, at, 0);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) {
		textureStore(thermDst, at, q);
		return;
	}
	let cx = params.values[8];
	let cy = params.values[9];
	let r2 = params.values[10];
	let mode = params.values[11];
	let dx = (f32(x) - 0.5) - cx;
	let dy = (f32(y) - 0.5) - cy;
	let fl = textureLoad(flagsTex, at, 0);
	let m = textureLoad(massTex, at, 0).a;
	let mEps = params.values[58];
	var E = q.r;
	var T = q.g;
	var hIn = q.b;
	var hOut = q.a;
	var changed = false;
	if (dx * dx + dy * dy < r2 && (fl.r & 1u) == 0u && m >= mEps) {
		let tEps = params.values[47];
		let cSol = params.values[48];
		let cLiq = params.values[49];
		let Lf = params.values[51];
		let tMelt = params.values[53];
		if (mode > 0.0) {
			let tFreeze = max(tEps, tMelt - 15.0);
			let eTarget = m * (cSol * tFreeze);
			if (E > eTarget) {
				let dReal = E - eTarget;
				E = eTarget;
				hOut = hOut + dReal;
				changed = true;
			}
		} else if (fl.b > 0u) {
			let tThaw = max(tEps, tMelt + 2.0);
			let hLiq = ((cSol * tMelt) + Lf) + cLiq * (tThaw - tMelt);
			let eTarget = m * hLiq;
			if (E < eTarget) {
				let dReal = eTarget - E;
				E = eTarget;
				hIn = hIn + dReal;
				changed = true;
			}
		}
	}
	if (changed) {
		T = evalPhase(m, E).T;
	}
	textureStore(thermDst, at, vec4<f32>(E, T, hIn, hOut));
}
`;
	shaders.colliderMask=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> colliders: array<f32>;
@group(0) @binding(2) var flagsSrc: texture_2d<u32>;
@group(0) @binding(3) var flagsDst: texture_storage_2d<rgba8uint,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x >= w || y >= h) { return; }
	let at = vec2<i32>(x, y);
	let oldFl = textureLoad(flagsSrc, at, 0);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) {
		textureStore(flagsDst, at, vec4<u32>(0u, oldFl.g, oldFl.b, oldFl.a));
		return;
	}
	let nSlots = min(u32(colliders[1u]), 32u);
	let px = f32(x) - 0.5;
	let py = f32(y) - 0.5;
	var curColl = false;
	for (var s: u32 = 0u; s < nSlots; s = s + 1u) {
		let b = 16u + s * 16u;
		let shape = i32(colliders[b + 2u]);
		if (shape == 1) {
			let dx = px - colliders[b];
			let dy = py - colliders[b + 1u];
			if (dx * dx + dy * dy < colliders[b + 3u]) { curColl = true; }
		} else if (shape == 2) {
			if (abs(px - colliders[b]) <= colliders[b + 3u] && abs(py - colliders[b + 1u]) <= colliders[b + 4u]) {
				curColl = true;
			}
		}
	}
	let legacyWall = oldFl.r == 255u;
	let staticBit = oldFl.r & 1u;
	let wasStepped = !legacyWall && ((oldFl.r & 2u) != 0u) && ((oldFl.r & 4u) == 0u);
	let newlyOcc = curColl && !wasStepped;
	let newR = select(staticBit | select(0u, 2u, curColl) | select(0u, 4u, newlyOcc), 255u, legacyWall);
	textureStore(flagsDst, at, vec4<u32>(newR, oldFl.g, oldFl.b, oldFl.a));
}
`;
	shaders.copyFlags=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var flagsSrc: texture_2d<u32>;
@group(0) @binding(2) var flagsDst: texture_storage_2d<rgba8uint,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x >= w || y >= h) { return; }
	let at = vec2<i32>(x, y);
	textureStore(flagsDst, at, textureLoad(flagsSrc, at, 0));
}
`;
	shaders.colliderVel=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> colliders: array<f32>;
@group(0) @binding(2) var velSrc: texture_2d<f32>;
@group(0) @binding(3) var flagsTex: texture_2d<u32>;
@group(0) @binding(4) var velDst: texture_storage_2d<rg32float,write>;

fn sampleCollCell(cx: i32, cy: i32, w: i32, h: i32, nSlots: u32) -> vec3<f32> {
    if (cx < 1 || cx > w - 2 || cy < 1 || cy > h - 2) { return vec3<f32>(0.0, 0.0, 0.0); }
    let px = f32(cx) - 0.5;
    let py = f32(cy) - 0.5;
    var sx = 0.0;
    var sy = 0.0;
    var cnt = 0.0;
    for (var s: u32 = 0u; s < nSlots; s = s + 1u) {
        let b = 16u + s * 16u;
        let shape = i32(colliders[b + 2u]);
        if (shape == 1) {
            let dx = px - colliders[b];
            let dy = py - colliders[b + 1u];
            if (dx * dx + dy * dy < colliders[b + 3u]) {
                sx = sx + colliders[b + 5u];
                sy = sy + colliders[b + 6u];
                cnt = cnt + 1.0;
            }
        } else if (shape == 2) {
            if (abs(px - colliders[b]) <= colliders[b + 3u] && abs(py - colliders[b + 1u]) <= colliders[b + 4u]) {
                sx = sx + colliders[b + 5u];
                sy = sy + colliders[b + 6u];
                cnt = cnt + 1.0;
            }
        }
    }
    if (cnt > 0.0) {
        let inv = 1.0 / cnt;
        return vec3<f32>(sx * inv, sy * inv, 1.0);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(params.values[2]);
    let h = i32(params.values[3]);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) { return; }
    let at = vec2<i32>(x, y);
    if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) {
        textureStore(velDst, at, vec4<f32>(0.0));
        return;
    }
    let src = textureLoad(velSrc, at, 0);
    var u = src.r;
    var v = src.g;
    let vc = params.values[38];
    let nSlots = min(u32(colliders[1u]), 32u);
    // sample collider occupancy at the west, centre and south cells so the
    // u-face (west|centre) and v-face (south|centre) can be classified.
    // static walls and frozen cells keep the velocity advectVel already zeroed
    // at blocked faces; only moving-collider faces are overwritten here.
    let cW = sampleCollCell(x - 1, y, w, h, nSlots);
    let cC = sampleCollCell(x, y, w, h, nSlots);
    let cS = sampleCollCell(x, y - 1, w, h, nSlots);
    let westIn = cW.z > 0.0;
    let centIn = cC.z > 0.0;
    let southIn = cS.z > 0.0;
    // u face lies between the west cell and the centre cell
    if ((westIn && !centIn) || (!westIn && centIn)) {
        // moving-collider boundary: impose the rigid-body x-velocity (no
        // penetration) on the fluid side of the face.
        let ov = select(cC, cW, westIn);
        u = clamp(ov.x, -vc, vc);
    }
    if (westIn && centIn) {
        u = 0.0; // rigid-body interior: keep fluid velocity at zero
    }
    // v face lies between the south cell and the centre cell
    if ((southIn && !centIn) || (!southIn && centIn)) {
        // moving-collider boundary: impose the rigid-body y-velocity.
        let ov = select(cC, cS, southIn);
        v = clamp(ov.y, -vc, vc);
    }
    if (southIn && centIn) {
        v = 0.0; // rigid-body interior: keep fluid velocity at zero
    }
    textureStore(velDst, at, vec4<f32>(u, v, 0.0, 0.0));
}
`;

	// ---------- 0.7.0 t3 emitters: shared realized-delta cell kernel + 1-workgroup Kahan tally ----------
	const EMITTER_HELPERS=EOS_HELPERS+`
struct EmitCellOut {
	dr: f32, dg: f32, db: f32, m: f32,
	e: f32, t: f32, hIn: f32, hOut: f32,
	injM: f32, injDr: f32, injDg: f32, injDb: f32, injE: f32,
	remM: f32, remDr: f32, remDg: f32, remDb: f32, remE: f32,
	changed: u32
};

fn evalScalarCell(x: i32, y: i32, mIn: vec4<f32>, qIn: vec4<f32>, wall: u32) -> EmitCellOut {
	var o: EmitCellOut;
	o.dr = mIn.r; o.dg = mIn.g; o.db = mIn.b; o.m = mIn.a;
	o.e = qIn.r; o.t = qIn.g; o.hIn = qIn.b; o.hOut = qIn.a;
	o.injM = 0.0; o.injDr = 0.0; o.injDg = 0.0; o.injDb = 0.0; o.injE = 0.0;
	o.remM = 0.0; o.remDr = 0.0; o.remDg = 0.0; o.remDb = 0.0; o.remE = 0.0;
	o.changed = 0u;
	if (wall > 0u) { return o; }
	let nSlots = i32(emitters[1]);
	let dt = params.values[4];
	let mEps = params.values[58];
	let px = f32(x) - 0.5;
	let py = f32(y) - 0.5;
	for (var s = 0; s < nSlots; s = s + 1) {
		let b = 16 + s * 16;
		let kind = i32(emitters[b + 4]);
		let val = emitters[b + 5];
		if (kind != 0 && kind != 4 && val > 0.0) {
			let dx = px - emitters[b];
			let dy = py - emitters[b + 1];
			let d2 = dx * dx + dy * dy;
			if (d2 < emitters[b + 3]) {
				let tf = max(0.0, 1.0 - d2 * emitters[b + 2]);
				let fw = tf * tf;
				let amt = (val * dt) * fw;
				if (amt > 0.0) {
					if (kind == 1) {
						let ddr = emitters[b + 6] * amt;
						let ddg = emitters[b + 7] * amt;
						let ddb = emitters[b + 8] * amt;
						let dE = amt * emitters[b + 11];
						let mNew = o.m + amt;
						let drNew = o.dr + ddr;
						let dgNew = o.dg + ddg;
						let dbNew = o.db + ddb;
						let eNew = o.e + dE;
						o.injM = o.injM + (mNew - o.m);
						o.injDr = o.injDr + (drNew - o.dr);
						o.injDg = o.injDg + (dgNew - o.dg);
						o.injDb = o.injDb + (dbNew - o.db);
						o.injE = o.injE + (eNew - o.e);
						o.m = mNew; o.dr = drNew; o.dg = dgNew; o.db = dbNew; o.e = eNew;
						o.changed = 1u;
					} else if (kind == 5) {
						if (o.m > 0.0) {
							if (amt >= o.m) {
								o.remM = o.remM + o.m;
								o.remDr = o.remDr + o.dr;
								o.remDg = o.remDg + o.dg;
								o.remDb = o.remDb + o.db;
								o.remE = o.remE + o.e;
								o.m = 0.0; o.dr = 0.0; o.dg = 0.0; o.db = 0.0; o.e = 0.0;
							} else {
								let frac = clamp(amt / o.m, 0.0, 1.0);
								let mNew = max(0.0, o.m - amt);
								let drNew = max(0.0, o.dr - o.dr * frac);
								let dgNew = max(0.0, o.dg - o.dg * frac);
								let dbNew = max(0.0, o.db - o.db * frac);
								let eNew = max(0.0, o.e - o.e * frac);
								o.remM = o.remM + (o.m - mNew);
								o.remDr = o.remDr + (o.dr - drNew);
								o.remDg = o.remDg + (o.dg - dgNew);
								o.remDb = o.remDb + (o.db - dbNew);
								o.remE = o.remE + (o.e - eNew);
								o.m = mNew; o.dr = drNew; o.dg = dgNew; o.db = dbNew; o.e = eNew;
							}
							o.changed = 1u;
						}
					} else if (kind == 2) {
						if (o.m >= mEps) {
							var dE = amt;
							let hCap = emitters[b + 11];
							if (hCap > 0.0) {
								let eCeil = o.m * hCap;
								if (o.e >= eCeil) { dE = 0.0; }
								else { dE = min(dE, eCeil - o.e); }
							}
							if (dE > 0.0) {
								let eNew = o.e + dE;
								let dReal = eNew - o.e;
								o.e = eNew;
								o.hIn = o.hIn + dReal;
								o.changed = 1u;
							}
						}
					} else if (kind == 3 || kind == 6) {
						if (o.m >= mEps) {
							let eFloor = o.m * emitters[b + 11];
							if (o.e > eFloor) {
								let cool = min(amt, o.e - eFloor);
								let eNew = max(eFloor, o.e - cool);
								let dReal = o.e - eNew;
								o.e = eNew;
								o.hOut = o.hOut + dReal;
								o.changed = 1u;
							}
						}
					}
				}
			}
		}
	}
	if (o.changed > 0u) {
		let ps = evalPhase(o.m, o.e);
		o.t = ps.T;
	}
	return o;
}
`;
	shaders.applyEmitters=EMITTER_HELPERS+`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> emitters: array<f32>;
@group(0) @binding(2) var massSrc: texture_2d<f32>;
@group(0) @binding(3) var thermSrc: texture_2d<f32>;
@group(0) @binding(4) var flagsTex: texture_2d<u32>;
@group(0) @binding(5) var massDst: texture_storage_2d<rgba32float,write>;
@group(0) @binding(6) var thermDst: texture_storage_2d<rgba32float,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x >= w || y >= h) { return; }
	let at = vec2<i32>(x, y);
	let mIn = textureLoad(massSrc, at, 0);
	let qIn = textureLoad(thermSrc, at, 0);
	if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) {
		textureStore(massDst, at, mIn);
		textureStore(thermDst, at, qIn);
		return;
	}
	let fl = textureLoad(flagsTex, at, 0);
	let r = evalScalarCell(x, y, mIn, qIn, fl.r);
	if (r.changed == 0u) {
		textureStore(massDst, at, mIn);
		textureStore(thermDst, at, qIn);
	} else {
		textureStore(massDst, at, vec4<f32>(r.dr, r.dg, r.db, r.m));
		textureStore(thermDst, at, vec4<f32>(r.e, r.t, r.hIn, r.hOut));
	}
}
`;
	shaders.emitterTally=EMITTER_HELPERS+`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> emitters: array<f32>;
@group(0) @binding(2) var massSrc: texture_2d<f32>;
@group(0) @binding(3) var thermSrc: texture_2d<f32>;
@group(0) @binding(4) var flagsTex: texture_2d<u32>;
@group(0) @binding(5) var<storage,read_write> emitterLedger: array<f32>;

var<workgroup> shEmit: array<f32,640>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let nSlots = i32(emitters[1]);
	var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0; var acc4 = 0.0;
	var acc5 = 0.0; var acc6 = 0.0; var acc7 = 0.0; var acc8 = 0.0; var acc9 = 0.0;
	for (var s = 0; s < nSlots; s = s + 1) {
		let b = 16 + s * 16;
		let sk = i32(emitters[b + 4]);
		if ((sk == 1 || sk == 5) && emitters[b + 5] > 0.0) {
			let x0 = i32(emitters[b + 12]);
			let y0 = i32(emitters[b + 13]);
			let x1 = i32(emitters[b + 14]);
			let y1 = i32(emitters[b + 15]);
			let bw = x1 - x0 + 1;
			let bh = y1 - y0 + 1;
			if (bw > 0 && bh > 0) {
				let wu = u32(bw);
				let total = u32(bw * bh);
				var k = lid;
				loop {
					if (k >= total) { break; }
					let cx = x0 + i32(k % wu);
					let cy = y0 + i32(k / wu);
					var owned = true;
					for (var p = 0; p < s; p = p + 1) {
						let pb = 16 + p * 16;
						let pk = i32(emitters[pb + 4]);
						if ((pk == 1 || pk == 5) && emitters[pb + 5] > 0.0) {
							if (cx >= i32(emitters[pb + 12]) && cx <= i32(emitters[pb + 14]) &&
								cy >= i32(emitters[pb + 13]) && cy <= i32(emitters[pb + 15])) {
								owned = false;
							}
						}
					}
					if (owned && cx >= 1 && cx < w - 1 && cy >= 1 && cy < h - 1) {
						let at = vec2<i32>(cx, cy);
						let mIn = textureLoad(massSrc, at, 0);
						let qIn = textureLoad(thermSrc, at, 0);
						let fl = textureLoad(flagsTex, at, 0);
						let r = evalScalarCell(cx, cy, mIn, qIn, fl.r);
						acc0 = acc0 + r.injM;
						acc1 = acc1 + r.injDr;
						acc2 = acc2 + r.injDg;
						acc3 = acc3 + r.injDb;
						acc4 = acc4 + r.injE;
						acc5 = acc5 + r.remM;
						acc6 = acc6 + r.remDr;
						acc7 = acc7 + r.remDg;
						acc8 = acc8 + r.remDb;
						acc9 = acc9 + r.remE;
					}
					k = k + 64u;
				}
			}
		}
	}
	let base = lid * 10u;
	shEmit[base + 0u] = acc0; shEmit[base + 1u] = acc1; shEmit[base + 2u] = acc2;
	shEmit[base + 3u] = acc3; shEmit[base + 4u] = acc4; shEmit[base + 5u] = acc5;
	shEmit[base + 6u] = acc6; shEmit[base + 7u] = acc7; shEmit[base + 8u] = acc8;
	shEmit[base + 9u] = acc9;
	workgroupBarrier();
	var stride = 32u;
	loop {
		if (stride == 0u) { break; }
		if (lid < stride) {
			let a = lid * 10u;
			let o = (lid + stride) * 10u;
			for (var c = 0u; c < 10u; c = c + 1u) {
				shEmit[a + c] = shEmit[a + c] + shEmit[o + c];
			}
		}
		workgroupBarrier();
		stride = stride >> 1u;
	}
	if (lid == 0u) {
		for (var c = 0u; c < 10u; c = c + 1u) {
			let d = shEmit[c];
			if (d != 0.0) {
				let sOld = emitterLedger[c];
				let cOld = emitterLedger[10u + c];
				let yv = d - cOld;
				let tv = sOld + yv;
				emitterLedger[10u + c] = (tv - sOld) - yv;
				emitterLedger[c] = tv;
			}
		}
	}
}
`;
	// solve-time poisson residual: same operator the projection just used, reduced
	// before transport so a later density or freeze change cannot inflate it.
	// shares the stats partial buffer; statsReduceB must not overwrite slots 32/33.
	shaders.poissonReduceA=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var pressTex: texture_2d<f32>;
@group(0) @binding(2) var divsrcTex: texture_2d<f32>;
@group(0) @binding(3) var massTex: texture_2d<f32>;
@group(0) @binding(4) var flagsTex: texture_2d<u32>;
@group(0) @binding(5) var<storage,read_write> partials: array<f32>;
${POISSON_HELPERS}
var<workgroup> sh: array<f32,128>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	var maxRes = 0.0;
	var maxRhs = 0.0;
	if (x >= 1 && x < w - 1 && y >= 1 && y < h - 1) {
		var pp: PoissonP;
		pp.center = textureLoad(pressTex, vec2<i32>(x, y), 0).r;
		pp.left = textureLoad(pressTex, vec2<i32>(x - 1, y), 0).r;
		pp.right = textureLoad(pressTex, vec2<i32>(x + 1, y), 0).r;
		pp.down = textureLoad(pressTex, vec2<i32>(x, y - 1), 0).r;
		pp.up = textureLoad(pressTex, vec2<i32>(x, y + 1), 0).r;
		let cell = poissonResidual(pp, x, y, w, h);
		maxRes = abs(cell.res);
		maxRhs = abs(cell.rhs);
	}
	sh[lid] = maxRes;
	sh[64u + lid] = maxRhs;
	workgroupBarrier();
	var stride = 32u;
	loop {
		if (stride == 0u) { break; }
		if (lid < stride) {
			sh[lid] = max(sh[lid], sh[lid + stride]);
			sh[64u + lid] = max(sh[64u + lid], sh[64u + lid + stride]);
		}
		workgroupBarrier();
		stride = stride >> 1u;
	}
	if (lid == 0u) {
		let base = (wid.y * nwg.x + wid.x) * 40u;
		partials[base] = sh[0];
		partials[base + 1u] = sh[64u];
	}
}
`;
	shaders.poissonReduceB=`
@group(0) @binding(0) var<storage,read> partials: array<f32>;
@group(0) @binding(1) var<storage,read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(1)
fn main() {
	let num = dims.x / 40u;
	var maxRes = 0.0;
	var maxRhs = 0.0;
	for (var i = 0u; i < num; i++) {
		let base = i * 40u;
		maxRes = max(maxRes, partials[base]);
		maxRhs = max(maxRhs, partials[base + 1u]);
	}
	stats[32u] = maxRes;
	stats[33u] = maxRhs;
}
`;
	// ---------- two level stats reduction, level 1 (35 values -> stride 40) ----------
	shaders.statsReduceA=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_2d<f32>;
@group(0) @binding(2) var thermTex: texture_2d<f32>;
@group(0) @binding(3) var velTex: texture_2d<f32>;
@group(0) @binding(4) var pressTex: texture_2d<f32>;
@group(0) @binding(5) var flagsTex: texture_2d<u32>;
@group(0) @binding(6) var accumTex: texture_2d<f32>;
@group(0) @binding(7) var<storage,read_write> partials: array<f32>;
@group(0) @binding(8) var divsrcTex: texture_2d<f32>;
${POISSON_HELPERS}
var<workgroup> sh: array<f32,2240>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let dt = params.values[4];
	let x = i32(gid.x);
	let y = i32(gid.y);
	let inside = x >= 1 && x < w - 1 && y >= 1 && y < h - 1;
	var totM = 0.0; var totDr = 0.0; var totDg = 0.0; var totDb = 0.0; var totE = 0.0;
	var minM = 1e30; var minDr = 1e30; var minDg = 1e30; var minDb = 1e30; var minE = 1e30;
	var maxM = -1e30; var neg = 0.0; var over = 0.0;
	var maxVel = 0.0; var pMin = 1e30; var pMax = -1e30; var maxDiv = 0.0;
	var heatIn = 0.0; var heatOut = 0.0; var tMin = 1e30; var tMax = -1e30; var tSum = 0.0; var tCnt = 0.0; var kin = 0.0;
	var ph: u32 = 0u; var sSat = 0.0; var latE = 0.0; var fMom = 0.0;
	var maxRes = 0.0; var maxRhs = 0.0;
	if (inside) {
		let mass = textureLoad(massTex, vec2<i32>(x, y), 0);
		let therm = textureLoad(thermTex, vec2<i32>(x, y), 0);
		totM = mass.a; totDr = mass.r; totDg = mass.g; totDb = mass.b; totE = therm.r;
		minM = mass.a; minDr = mass.r; minDg = mass.g; minDb = mass.b; minE = therm.r;
		maxM = mass.a;
		if (mass.a < -1e-9 || mass.r < -1e-9 || mass.g < -1e-9 || mass.b < -1e-9 || therm.r < -1e-9) { neg = 1.0; }
		if (mass.a > 1.001) { over = 1.0; }
		let uL = textureLoad(velTex, vec2<i32>(x, y), 0).r;
		let uR = textureLoad(velTex, vec2<i32>(x + 1, y), 0).r;
		let vB = textureLoad(velTex, vec2<i32>(x, y), 0).g;
		let vT = textureLoad(velTex, vec2<i32>(x, y + 1), 0).g;
		let fl = textureLoad(flagsTex, vec2<i32>(x, y), 0);
		maxVel = max(max(abs(uL), abs(uR)), max(abs(vB), abs(vT)));
		if ((fl.r | fl.b) == 0u) {
			maxDiv = abs(uR - uL + vT - vB) / max(dt, 1e-9);
		}
		let p = textureLoad(pressTex, vec2<i32>(x, y), 0).r;
		pMin = p; pMax = p;
		// post-step residual: same pressure and divsrc, but mass and flags after
		// transport and phase. this is coefficient drift, not the solve. the solve
		// residual is poissonReduce, captured before those passes.
		var pp: PoissonP;
		pp.center = p;
		pp.left = textureLoad(pressTex, vec2<i32>(x - 1, y), 0).r;
		pp.right = textureLoad(pressTex, vec2<i32>(x + 1, y), 0).r;
		pp.down = textureLoad(pressTex, vec2<i32>(x, y - 1), 0).r;
		pp.up = textureLoad(pressTex, vec2<i32>(x, y + 1), 0).r;
		let cell = poissonResidual(pp, x, y, w, h);
		maxRes = abs(cell.res); maxRhs = abs(cell.rhs);
		heatIn = therm.b; heatOut = therm.a;
		if (mass.a >= params.values[58]) {
			tMin = therm.g; tMax = therm.g; tSum = therm.g; tCnt = 1.0;
			let uc = 0.5 * (uL + uR);
			let vc = 0.5 * (vB + vT);
			kin = 0.5 * mass.a * (uc * uc + vc * vc);
		}
		ph = fl.g;
		if (fl.a > 0u) { sSat = 1.0; }
		let ac = textureLoad(accumTex, vec2<i32>(x, y), 0);
		latE = ac.r;
		fMom = ac.g;
	}
	sh[0u * 64u + lid] = totM; sh[1u * 64u + lid] = totDr; sh[2u * 64u + lid] = totDg;
	sh[3u * 64u + lid] = totDb; sh[4u * 64u + lid] = totE;
	sh[5u * 64u + lid] = minM; sh[6u * 64u + lid] = minDr; sh[7u * 64u + lid] = minDg;
	sh[8u * 64u + lid] = minDb; sh[9u * 64u + lid] = minE;
	sh[10u * 64u + lid] = maxM; sh[11u * 64u + lid] = neg; sh[12u * 64u + lid] = over;
	sh[13u * 64u + lid] = maxVel; sh[14u * 64u + lid] = pMin; sh[15u * 64u + lid] = pMax; sh[16u * 64u + lid] = maxDiv;
	sh[17u * 64u + lid] = heatIn; sh[18u * 64u + lid] = heatOut;
	sh[19u * 64u + lid] = tMin; sh[20u * 64u + lid] = tMax; sh[21u * 64u + lid] = tSum;
	sh[22u * 64u + lid] = kin; sh[23u * 64u + lid] = tCnt;
	sh[24u * 64u + lid] = select(0.0, 1.0, inside && ph == 0u);
	sh[25u * 64u + lid] = select(0.0, 1.0, inside && ph == 1u);
	sh[26u * 64u + lid] = select(0.0, 1.0, inside && ph == 2u);
	sh[27u * 64u + lid] = select(0.0, 1.0, inside && ph == 3u);
	sh[28u * 64u + lid] = select(0.0, 1.0, inside && ph == 4u);
	sh[29u * 64u + lid] = select(0.0, 1.0, inside && ph == 5u);
	sh[30u * 64u + lid] = sSat;
	sh[31u * 64u + lid] = latE;
	sh[32u * 64u + lid] = fMom;
	sh[33u * 64u + lid] = maxRes; sh[34u * 64u + lid] = maxRhs;
	workgroupBarrier();
	var stride = 32u;
	loop {
		if (stride == 0u) { break; }
		if (lid < stride) {
			for (var k = 0u; k < 5u; k++) { sh[k * 64u + lid] = sh[k * 64u + lid] + sh[k * 64u + lid + stride]; }
			for (var k = 5u; k < 10u; k++) { sh[k * 64u + lid] = min(sh[k * 64u + lid], sh[k * 64u + lid + stride]); }
			sh[10u * 64u + lid] = max(sh[10u * 64u + lid], sh[10u * 64u + lid + stride]);
			sh[11u * 64u + lid] = sh[11u * 64u + lid] + sh[11u * 64u + lid + stride];
			sh[12u * 64u + lid] = sh[12u * 64u + lid] + sh[12u * 64u + lid + stride];
			sh[13u * 64u + lid] = max(sh[13u * 64u + lid], sh[13u * 64u + lid + stride]);
			sh[14u * 64u + lid] = min(sh[14u * 64u + lid], sh[14u * 64u + lid + stride]);
			sh[15u * 64u + lid] = max(sh[15u * 64u + lid], sh[15u * 64u + lid + stride]);
			sh[16u * 64u + lid] = max(sh[16u * 64u + lid], sh[16u * 64u + lid + stride]);
			sh[33u * 64u + lid] = max(sh[33u * 64u + lid], sh[33u * 64u + lid + stride]);
			sh[34u * 64u + lid] = max(sh[34u * 64u + lid], sh[34u * 64u + lid + stride]);
			sh[17u * 64u + lid] = sh[17u * 64u + lid] + sh[17u * 64u + lid + stride];
			sh[18u * 64u + lid] = sh[18u * 64u + lid] + sh[18u * 64u + lid + stride];
			sh[19u * 64u + lid] = min(sh[19u * 64u + lid], sh[19u * 64u + lid + stride]);
			sh[20u * 64u + lid] = max(sh[20u * 64u + lid], sh[20u * 64u + lid + stride]);
			for (var k = 21u; k < 33u; k++) { sh[k * 64u + lid] = sh[k * 64u + lid] + sh[k * 64u + lid + stride]; }
		}
		workgroupBarrier();
		stride = stride >> 1u;
	}
	if (lid == 0u) {
		let base = (wid.y * nwg.x + wid.x) * 40u;
		for (var k = 0u; k < 35u; k++) { partials[base + k] = sh[k * 64u]; }
	}
}
`;
	// ---------- level 2 with kahan sums (35 partial slots -> 46 stats slots) ----------
	shaders.statsReduceB=`
@group(0) @binding(0) var<storage,read> partials: array<f32>;
@group(0) @binding(1) var<storage,read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>;
@group(0) @binding(3) var<storage,read> emitterLedger: array<f32>;

fn kahanSlot(num: u32, slot: u32) -> f32 {
	var s = 0.0;
	var c = 0.0;
	for (var i = 0u; i < num; i++) {
		let y = partials[i * 40u + slot] - c;
		let t = s + y;
		c = (t - s) - y;
		s = t;
	}
	return s;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	if (gid.x != 0u) { return; }
	let num = dims.x / 40u;
	let sM = kahanSlot(num, 0u);
	let sDr = kahanSlot(num, 1u);
	let sDg = kahanSlot(num, 2u);
	let sDb = kahanSlot(num, 3u);
	let sE = kahanSlot(num, 4u);
	let sIn = kahanSlot(num, 17u);
	let sOut = kahanSlot(num, 18u);
	let sTS = kahanSlot(num, 21u);
	let sKin = kahanSlot(num, 22u);
	let tCnt = kahanSlot(num, 23u);
	var neg = 0.0; var over = 0.0;
	var minM = 1e30; var minDr = 1e30; var minDg = 1e30; var minDb = 1e30; var minE = 1e30;
	var maxM = -1e30; var maxVel = 0.0; var pMin = 1e30; var pMax = -1e30; var maxDiv = 0.0;
	var maxRes = 0.0; var maxRhs = 0.0;
	var tMin = 1e30; var tMax = -1e30;
	for (var i = 0u; i < num; i++) {
		let base = i * 40u;
		tMin = min(tMin, partials[base + 19u]);
		tMax = max(tMax, partials[base + 20u]);
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
		maxRes = max(maxRes, partials[base + 33u]);
		maxRhs = max(maxRhs, partials[base + 34u]);
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
	// 32/33 belong to the solve-time reduce. 34/35 are this pass's post-step drift.
	stats[34u] = maxRes; stats[35u] = maxRhs;
	stats[13u] = maxVel; stats[14u] = pMin; stats[15u] = pMax; stats[16u] = maxDiv;
	if (tMin == 1e30) { tMin = 0.0; }
	if (tMax == -1e30) { tMax = 0.0; }
	stats[17u] = sIn; stats[18u] = sOut;
	stats[19u] = tMin; stats[20u] = tMax;
	stats[21u] = select(0.0, sTS / tCnt, tCnt > 0.0);
	stats[22u] = sKin;
	for (var k = 0u; k < 9u; k++) { stats[23u + k] = kahanSlot(num, 24u + k); }
	for (var k = 0u; k < 10u; k++) { stats[36u + k] = emitterLedger[k]; }
}
`;
	// ---------- buoy field sample: one workgroup over the body footprint ----------
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
	// ---------- point queries (0.7.0 t5): one invocation per probe slot ----------
	// the layout is owned by src/core/probes.js; this kernel only fills the measured
	// slots (m, dye, E, u, v, p) so x/y/active stay cpu-owned and a move is one write.
	// T and phase are derived on the cpu from (m, E) by the shared caloric EOS.
	shaders.probeSample=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_2d<f32>;
@group(0) @binding(2) var thermTex: texture_2d<f32>;
@group(0) @binding(3) var velTex: texture_2d<f32>;
@group(0) @binding(4) var pressTex: texture_2d<f32>;
@group(0) @binding(5) var<storage,read_write> probes: array<f32>;

@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let idx = gid.x;
	if (idx >= 32u || probes[0u] < 1.0) { return; }
	let base = 16u + idx * 16u;
	if (probes[base + 15u] < 0.5) { return; }
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let cx = clamp(i32(round(probes[base])), 1, w - 2);
	let cy = clamp(i32(round(probes[base + 1u])), 1, h - 2);
	let at = vec2<i32>(cx, cy);
	let mass = textureLoad(massTex, at, 0);
	let therm = textureLoad(thermTex, at, 0);
	let vel = textureLoad(velTex, at, 0);
	probes[base + 2u] = mass.a;
	probes[base + 3u] = mass.r;
	probes[base + 4u] = mass.g;
	probes[base + 5u] = mass.b;
	probes[base + 6u] = therm.r;
	probes[base + 7u] = vel.r;
	probes[base + 8u] = vel.g;
	probes[base + 9u] = textureLoad(pressTex, at, 0).r;
}
`;
	// ---------- view modes (0..7 + composable solid overlay + buoy overlay) ----------
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
fn phaseColor(ph: u32) -> vec3<f32> {
	if (ph == 0u) { return vec3<f32>(0.03, 0.04, 0.06); }
	if (ph == 1u) { return vec3<f32>(0.72, 0.88, 1.00); }
	if (ph == 2u) { return vec3<f32>(0.20, 0.78, 0.85); }
	if (ph == 3u) { return vec3<f32>(0.12, 0.42, 0.88); }
	if (ph == 4u) { return vec3<f32>(1.00, 0.58, 0.10); }
	return vec3<f32>(0.95, 0.26, 0.45);
}
fn frozenAt(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return false; }
	return textureLoad(flagsTex, vec2<i32>(x, y), 0).b > 0u;
}
fn colliderAt(x: i32, y: i32, w: i32, h: i32) -> bool {
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return false; }
	let r = textureLoad(flagsTex, vec2<i32>(x, y), 0).r;
	return (r & 2u) > 0u && (r & 1u) == 0u;
}
fn solidOverlay(x: i32, y: i32, w: i32, h: i32, mode: i32, fl: vec4<u32>, base: vec4<f32>) -> vec4<f32> {
	var rgb = base.rgb;
	if (params.values[86] > 0.5 && mode != 5 && fl.b > 0u) {
		let edge = !frozenAt(x - 1, y, w, h) || !frozenAt(x + 1, y, w, h)
			|| !frozenAt(x, y - 1, w, h) || !frozenAt(x, y + 1, w, h);
		let hatch = ((x + y) & 3) == 0;
		rgb = mix(rgb, vec3<f32>(0.78, 0.92, 1.0), 0.12);
		if (hatch) { rgb = mix(rgb, vec3<f32>(0.82, 0.94, 1.0), 0.36); }
		if (edge) { rgb = mix(rgb, vec3<f32>(0.92, 0.98, 1.0), 0.82); }
	}
	if (fl.r > 0u && mode != 5 && mode != 7) {
		if ((fl.r & 1u) > 0u) {
			rgb = mix(rgb, vec3<f32>(0.45, 0.35, 0.30), 0.72);
		} else {
			let rim = !colliderAt(x - 1, y, w, h) || !colliderAt(x + 1, y, w, h)
				|| !colliderAt(x, y - 1, w, h) || !colliderAt(x, y + 1, w, h);
			rgb = mix(rgb, vec3<f32>(0.24, 0.30, 0.38), 0.84);
			if (rim) { rgb = mix(rgb, vec3<f32>(0.96, 0.74, 0.28), 0.90); }
		}
	}
	return vec4<f32>(rgb, 1.0);
}
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
	let fl = textureLoad(flagsTex, vec2<i32>(x, y), 0);
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
		let div = select(0.0, abs(uR - uL + vT - vB) / dt, (fl.r | fl.b) == 0u);
		let t = clamp(log2(1.0 + div) / 10.0, 0.0, 1.0);
		out = vec4<f32>(t * t, t, 0.15 + 0.5 * t, 1.0);
	} else if (mode == 6) {
		if (m > params.values[58]) {
			let u = clamp((therm.g - params.values[20]) / 50.0, -1.0, 1.0);
			out = vec4<f32>(max(u, 0.0), 0.1 + 0.1 * (1.0 - abs(u)), max(-u, 0.0), 1.0);
		}
	} else if (mode == 7) {
		var pc = phaseColor(fl.g);
		if (fl.r > 0u) {
			pc = select(vec3<f32>(0.78, 0.60, 0.24), vec3<f32>(0.55, 0.30, 0.22), (fl.r & 1u) > 0u);
		}
		out = vec4<f32>(pc, 1.0);
	} else {
		// mode 5: painted static walls (red), moving colliders (amber), frozen substance (ice blue)
		let g = clamp(m, 0.0, 1.0);
		out = vec4<f32>(g * 0.6, g * 0.6, g * 0.7, 1.0);
		if (fl.b > 0u) { out = vec4<f32>(0.55, 0.85, 1.0, 1.0); }
		if (fl.r > 0u) {
			out = select(vec4<f32>(0.96, 0.72, 0.24, 1.0), vec4<f32>(0.9, 0.25, 0.2, 1.0), (fl.r & 1u) > 0u);
		}
	}
	let over = solidOverlay(x, y, w, h, mode, fl, out);
	textureStore(displayTex, vec2<i32>(x, y), buoyOverlay(x, y, over));
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
	let dims = textureDimensions(displayTex);
	let n = vec2<f32>(f32(dims.x) - 2, f32(dims.y) - 2);
	let tex = vec2<f32>(f32(dims.x), f32(dims.y));
	let uv2 = (uv * n + vec2<f32>(1.0, 1.0)) / tex;
	return textureSample(displayTex, samp, uv2);
}
`;
	// ---------- multigrid coarse level (0.6.0 t4 + 0.5.0 solid compatibility) ----------
	shaders.mgCoarsen=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var massTex: texture_2d<f32>;
@group(0) @binding(2) var flagsTex: texture_2d<u32>;
@group(0) @binding(3) var massCDst: texture_storage_2d<rgba32float,write>;
@group(0) @binding(4) var flagsCDst: texture_storage_2d<rgba8uint,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let S = max(2, i32(params.values[94]));
	let nc = (w - 2) / S;
	let xc = i32(gid.x);
	let yc = i32(gid.y);
	if (xc < 1 || xc > nc || yc < 1 || yc > nc) { return; }
	var sum = 0.0;
	var openCount = 0u;
	for (var oy = 0; oy < S; oy++) {
		for (var ox = 0; ox < S; ox++) {
			let xf = S * (xc - 1) + 1 + ox;
			let yf = S * (yc - 1) + 1 + oy;
			let fl = textureLoad(flagsTex, vec2<i32>(xf, yf), 0);
			if ((fl.r | fl.b) > 0u) { continue; }
			sum = sum + textureLoad(massTex, vec2<i32>(xf, yf), 0).a;
			openCount = openCount + 1u;
		}
	}
	let meanM = select(0.0, sum / f32(openCount), openCount > 0u);
	let wall = select(0u, 255u, openCount == 0u);
	textureStore(massCDst, vec2<i32>(xc, yc), vec4<f32>(0.0, 0.0, 0.0, meanM));
	textureStore(flagsCDst, vec2<i32>(xc, yc), vec4<u32>(wall, 0u, 0u, 0u));
}
`;
	shaders.mgRestrict=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var pressFine: texture_storage_2d<r32float,read>;
@group(0) @binding(2) var divsrcTex: texture_2d<f32>;
@group(0) @binding(3) var massTex: texture_2d<f32>;
@group(0) @binding(4) var flagsTex: texture_2d<u32>;
@group(0) @binding(5) var divCDst: texture_storage_2d<r32float,write>;

${POISSON_HELPERS}
// press arrives as a struct: this kernel reads the fine field as a storage texture,
// the stats pass reads the same one sampled. the operator itself is shared above.
fn residualAt(x: i32, y: i32, w: i32, h: i32) -> f32 {
	var p: PoissonP;
	p.center = textureLoad(pressFine, vec2<i32>(x, y)).r;
	p.left = textureLoad(pressFine, vec2<i32>(x - 1, y)).r;
	p.right = textureLoad(pressFine, vec2<i32>(x + 1, y)).r;
	p.down = textureLoad(pressFine, vec2<i32>(x, y - 1)).r;
	p.up = textureLoad(pressFine, vec2<i32>(x, y + 1)).r;
	return poissonResidual(p, x, y, w, h).res;
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let S = max(2, i32(params.values[94]));
	let nc = (w - 2) / S;
	let xc = i32(gid.x);
	let yc = i32(gid.y);
	if (xc < 1 || xc > nc || yc < 1 || yc > nc) { return; }
	var sum = 0.0;
	for (var oy = 0; oy < S; oy++) {
		for (var ox = 0; ox < S; ox++) {
			sum = sum + residualAt(S * (xc - 1) + 1 + ox, S * (yc - 1) + 1 + oy, w, h);
		}
	}
	textureStore(divCDst, vec2<i32>(xc, yc), vec4<f32>(sum, 0.0, 0.0, 0.0));
}
`;
	shaders.mgZero=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var pressC0: texture_storage_2d<r32float,write>;
@group(0) @binding(2) var pressC1: texture_storage_2d<r32float,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return; }
	textureStore(pressC0, vec2<i32>(x, y), vec4<f32>(0.0, 0.0, 0.0, 0.0));
	textureStore(pressC1, vec2<i32>(x, y), vec4<f32>(0.0, 0.0, 0.0, 0.0));
}
`;
	shaders.mgProlong=`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var pressC: texture_2d<f32>;
@group(0) @binding(2) var pressFine: texture_storage_2d<r32float,read_write>;
@group(0) @binding(3) var flagsTex: texture_2d<u32>;
@group(0) @binding(4) var flagsCTex: texture_2d<u32>;

fn sampleCoarse(cx: i32, cy: i32, base: f32) -> f32 {
	if (textureLoad(flagsCTex, vec2<i32>(cx, cy), 0).r > 0u) { return base; }
	return textureLoad(pressC, vec2<i32>(cx, cy), 0).r;
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let w = i32(params.values[2]);
	let h = i32(params.values[3]);
	let S = max(2, i32(params.values[94]));
	let nc = (w - 2) / S;
	let x = i32(gid.x);
	let y = i32(gid.y);
	if (x < 1 || x > w - 2 || y < 1 || y > h - 2) { return; }
	let fl = textureLoad(flagsTex, vec2<i32>(x, y), 0);
	if ((fl.r | fl.b) > 0u) { return; }
	let hc = vec2<i32>((x - 1) / S + 1, (y - 1) / S + 1);
	let base = textureLoad(pressC, hc, 0).r;
	let invS = 1.0 / f32(S);
	let xf = (f32(x) - 0.5) * invS + 0.5;
	let yf = (f32(y) - 0.5) * invS + 0.5;
	let xc = i32(floor(xf));
	let yc = i32(floor(yf));
	let fx = xf - f32(xc);
	let fy = yf - f32(yc);
	let x0 = clamp(xc, 1, nc);
	let x1 = clamp(xc + 1, 1, nc);
	let y0 = clamp(yc, 1, nc);
	let y1 = clamp(yc + 1, 1, nc);
	let v00 = sampleCoarse(x0, y0, base);
	let v10 = sampleCoarse(x1, y0, base);
	let v01 = sampleCoarse(x0, y1, base);
	let v11 = sampleCoarse(x1, y1, base);
	let v0 = v00 + (v10 - v00) * fx;
	let v1 = v01 + (v11 - v01) * fx;
	let corr = v0 + (v1 - v0) * fy;
	let p = textureLoad(pressFine, vec2<i32>(x, y)).r;
	textureStore(pressFine, vec2<i32>(x, y), vec4<f32>(p + corr, 0.0, 0.0, 0.0));
}
`;
	Fluid.gpu.shaders=shaders;
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.shaders;
})(globalThis);
