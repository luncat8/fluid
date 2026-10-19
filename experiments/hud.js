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
