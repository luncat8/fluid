"use strict";
// Range ownership and cfg -> DOM synchronization without a browser or GPU.
const assert=require("node:assert/strict");
require("../src/core/config.js");
require("../src/app/controls.js");
let checks=0;
function check(value,message){
	assert.ok(value,message);
	console.log("ok   "+message);
	checks++;
}
class El{
	constructor(){this.listeners={};this.value="";this.textContent="";}
	addEventListener(type,fn){(this.listeners[type]??=[]).push(fn);}
	fire(type){for(const fn of this.listeners[type]||[])fn({type,target:this});}
}
const ids=["controls-panel","ctrl-grid","ctrl-grid-val","ctrl-iters","ctrl-iters-val",
	"ctrl-substeps","ctrl-substeps-fixed","ctrl-substeps-fixed-val","ctrl-fct","ctrl-gravity","ctrl-buoy"];
const els=Object.fromEntries(ids.map(id=>[id,new El()]));
global.document={getElementById:id=>els[id]||null,querySelectorAll:()=>[]};
global.window=new El();
const cfg={...Fluid.core.config.defaults};
let resizes=0;
const controls=Fluid.app.controls.create(cfg,{resize(n){resizes++;cfg.N=n;}});
const grid=els["ctrl-grid"],out=els["ctrl-grid-val"];
let now=0;
function sync(){controls.update(now+=200);}
check(grid.value===cfg.N&&out.textContent===cfg.N,"grid and readout initialize from config");
grid.fire("pointerdown");
grid.value="768";
grid.fire("input");
sync();
check(grid.value==="768"&&out.textContent==="768"&&cfg.N===512,"sync cannot snap a held grid thumb back");
check(resizes===0,"input previews the grid without reallocating textures");
grid.fire("pointerup");
grid.fire("change");
sync();
check(resizes===1&&cfg.N===768&&Number(grid.value)===768,"release commits one resize");
grid.fire("change");
check(resizes===1,"same grid change does not rebuild twice");
cfg.N=640;
sync();
check(grid.value===640&&out.textContent===640,"external grid changes sync when no drag owns the thumb");
for(const [target,type] of [[grid,"pointercancel"],[grid,"lostpointercapture"],[grid,"blur"],
	[window,"pointerup"],[window,"pointercancel"],[window,"blur"]]){
	grid.fire("pointerdown");
	grid.value="896";
	grid.fire("input");
	target.fire(type);
	sync();
	check(grid.value===640&&out.textContent===640,"slider ownership is released on "+(target===window?"window ":"")+type);
}
check(resizes===1,"cancelled slider gestures do not commit a resize");
const auto=els["ctrl-substeps"];
check(auto.checked,"auto-CFL checkbox initializes from the config");
auto.checked=false;
auto.fire("change");
check(cfg.substepsAuto===false,"auto-CFL checkbox switches to fixed steps");
const fixed=els["ctrl-substeps-fixed"];
fixed.fire("pointerdown");
fixed.value="6";
fixed.fire("input");
sync();
check(cfg.substepsFixed===6&&els["ctrl-substeps-fixed-val"].textContent===6,"fixed-step slider updates the shared policy immediately");
fixed.fire("pointerup");
check(Fluid.core.config.substepCount(cfg,0)===6,"fixed-step slider feeds the solver substep policy");
const iters=els["ctrl-iters"];
iters.value="75";
iters.fire("input");
check(cfg.mgCoarseIters===75&&cfg.pressureIters===60,
	"pressure slider drives the coarse sweep count while multigrid is on");
cfg.mgCoarseIters=80;
sync();
check(iters.value===80&&els["ctrl-iters-val"].textContent===80,"keyboard pressure changes sync back to the slider");
cfg.useMultigrid=false;
sync();
check(iters.value===60,"turning multigrid off shows the red-black sweep count");
iters.value="40";
iters.fire("input");
check(cfg.pressureIters===40&&cfg.mgCoarseIters===80,"mg=0 slider drives the red-black sweep count");
cfg.useMultigrid=true;


const buoy=els["ctrl-buoy"];
check(buoy.disabled,"dam float is disabled outside its scenario");
cfg.scenario="dam";sync();
check(!buoy.disabled,"dam scenario enables the float control");
const fct=els["ctrl-fct"];
fct.checked=false;fct.fire("change");
check(cfg.fctLimiter===0,"FCT checkbox switches the actual solver configuration off");
fct.checked=true;fct.fire("change");
check(cfg.fctLimiter===1,"FCT checkbox switches the actual solver configuration on");
const gravity=els["ctrl-gravity"];
gravity.checked=false;gravity.fire("change");
check(cfg.gravityY===0,"gravity checkbox removes applied gravity");
gravity.checked=true;gravity.fire("change");
check(cfg.gravityY===Fluid.core.config.defaults.gravityY,"gravity checkbox restores applied gravity");
console.log("\nall "+checks+" checks pass");
