"use strict";
const assert=require("node:assert/strict");
require("../src/core/config.js");
require("../src/core/stats.js");
require("../src/app/hud.js");
global.document={getElementById:()=>null};
const cfg={...Fluid.core.config.defaults,dt:1/300};
const sample={
	frame:100,statsFrame:91,totals:{m:1,dr:0,dg:0,db:0,E:1},expected:{m:1,dr:0,dg:0,db:0,E:1},
	mins:{m:0,dr:-1e-12,dg:0,db:0,E:0},maxM:30,cntNeg:10,cntOver:100,
	motion:{maxVel:600,maxDiv:1000,pMin:0,pMax:100000}
};
const solver={getStats:()=>sample,buoy:{state:{active:false}}};
const perf={simHz:42,renderHz:42,rafHz:144,simSpeed:0.7,cpuMs:2,queueMs:20};
const hud=Fluid.app.hud.create(solver,cfg,perf);
let snapshot=hud.buildSnapshot();
for(const text of ["dr=-1e-12","scenario=blobs","FCT=1","gravity=300","sim/s=42.0",
	"rAF/s=144.0","speed=0.70x","queue=20.00ms","CFL bound=4.0000","NEGATIVE VALUES",
	"SEVERE COMPRESSION","dam float: inactive (dam scenario only)"]){
	assert.ok(snapshot.includes(text),text);
}
assert.ok(!snapshot.includes("fps~"));
cfg.scenario="dam";
snapshot=hud.buildSnapshot();
assert.ok(snapshot.includes("dam float: disabled (y to enable)"));
console.log("PASS snapshot: exact minima, toggle context, honest rates, CFL/compression warnings, float applicability");

// 0.7.4: a non-finite field used to be the one failure every meter here was blind to
// (max/min drop NaN, and so does every `<`/`>`), so the banner has to lead the health
// line and the snapshot has to carry the count even with the health line stripped.
assert.ok(!hud.buildSnapshot().includes("NON-FINITE"), "a clean field prints no banner");
sample.cntNonFinite=0;
sample.nonFinite={mass:0,therm:0,vel:0,press:0};
sample.firstNonFiniteFrame=-1;
assert.ok(!hud.buildSnapshot().includes("NON-FINITE"), "explicit zero prints no banner");
sample.cntNonFinite=7;
sample.nonFinite={mass:0,therm:0,vel:7,press:2};
sample.firstNonFiniteFrame=691;
snapshot=hud.buildSnapshot();
assert.ok(snapshot.includes("NON-FINITE FIELD: 7 cells (m=0 T=0 u,v=7 p=2) first seen frame#691"),
	"the banner names the count, the field split and the frame it started");
assert.ok(snapshot.indexOf("NON-FINITE")<snapshot.indexOf("CFL bound="),
	"the banner leads the health line");
assert.ok(snapshot.includes("nonFinite=7 (m=0 T=0 u,v=7 p=2) firstNonFiniteFrame#691"),
	"the snapshot carries the count with the health line stripped");
console.log("PASS snapshot: a non-finite field leads the health line and is never silent");
