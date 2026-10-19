"use strict";
const assert=require("node:assert/strict");
require("../src/core/config.js");
require("../src/core/grid.js");
require("../src/core/stats.js");
require("../src/core/solver.js");
require("../src/gpu/shaders.js");
const f=Math.fround;
const sum=a=>f(f(f(a[0]+a[1])+a[2])+a[3]);
// f32 arithmetic probe, not shader execution: exact-unit normalization can
// overshoot after four independently rounded faces are stored and re-summed.
const raw=[0.336681067943573,0.3143274486064911,0.12935927510261536,0.9799701571464539];
const old=raw.map(x=>f(x/sum(raw)));
assert.ok(sum(old)>1,"old limiter admits a negative donor weight");
let seed=42;
function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
function limited(a){
	const s=sum(a);
	return a.map(x=>s>f(0.999999)?f(x*f(f(0.999999)/s)):x);
}
for(let i=0;i<100000;i++){
	const a=i===0?raw:Array.from({length:4},()=>f(random()));
	assert.ok(sum(limited(a))<=1);
}
console.log("PASS f32 outflow margin: old overshoot reproduced; 100000 bounded samples");
for(const fctLimiter of [0,1]){
	const N=8;
	const cfg={...Fluid.core.config.defaults,N,texW:N+2,texH:N+2,scenario:"still",fctLimiter};
	const sim=Fluid.core.solver.create(cfg);
	const cell=4+4*cfg.texW;
	const v=sim.vel.a;
	v[cell*2]=-raw[0];v[(cell+1)*2]=raw[1];
	v[cell*2+1]=-raw[2];v[(cell+cfg.texW)*2+1]=raw[3];
	const before=sim.stats();
	sim.computeFlux(1);
	const flux=sim.flux;
	const outgoing=[-flux[cell*2],flux[(cell+1)*2],-flux[cell*2+1],flux[(cell+cfg.texW)*2+1]];
	assert.ok(sum(outgoing)<1);
	sim.advectScalar();
	const after=sim.stats();
	assert.equal(after.tot.cntNeg,0);
	assert.ok(Math.abs(after.tot.sumM-before.tot.sumM)<1e-5);
	assert.ok(Math.abs(after.tot.sumE-before.tot.sumE)/before.tot.sumE<1e-7);
}
// Keep both GPU transport kernels on the same positive-weight formulation.
for(const shader of [Fluid.gpu.shaders.advectMass,Fluid.gpu.shaders.advectTherm]){
	assert.match(shader,/let remaining = 1\.0 -/);
	assert.match(shader,/let out = q \* remaining/);
}
console.log("PASS production CPU transport positivity/conservation, FCT on/off; WGSL structure matches");
