"use strict";
// Real input handlers with a minimal event surface; no browser or GPU required.
const assert=require("node:assert/strict");
require("../src/core/config.js");
require("../src/app/input.js");
let checks=0;
function check(value,message){
	assert.ok(value,message);
	console.log("ok   "+message);
	checks++;
}
class Target{
	constructor(){this.listeners={};}
	addEventListener(type,fn){(this.listeners[type]??=[]).push(fn);}
	fire(type,props={}){
		const e={type,key:"",pointerId:1,button:0,clientX:50,clientY:50,defaultPrevented:false,
			preventDefault(){this.defaultPrevented=true;},...props};
		for(const fn of this.listeners[type]||[])fn(e);
		return e;
	}
}
const win=new Target();
global.window=win;
global.document={getElementById(){return null;}};
const canvas=new Target();
const captures=new Set();
canvas.getBoundingClientRect=()=>({left:0,top:0,width:100,height:100});
canvas.setPointerCapture=id=>captures.add(id);
canvas.hasPointerCapture=id=>captures.has(id);
canvas.releasePointerCapture=id=>captures.delete(id);
let focused=0;
canvas.focus=()=>focused++;
const cfg={...Fluid.core.config.defaults};
const paint=[];
let resets=0,resizes=0,mode=-1;
const solver={
	paintDye:(u,v,c)=>paint.push({u,v,c}),
	paintWall:(u,v,erase)=>paint.push({u,v,erase}),
	reset(){resets++;},
	resize(n){resizes++;cfg.N=n;}
};
const input=Fluid.app.input.create(canvas,cfg,solver,id=>{mode=id;});
const key=(k,props)=>win.fire("keydown",{key:k,...props});

canvas.fire("pointerdown",{button:2});
check(!input.state.down&&captures.size===0,"right button does not paint or capture");
canvas.fire("pointerdown");
check(input.state.down&&captures.has(1)&&focused===1,"primary down captures and focuses the canvas");
check(cfg.mouseStrength===1&&input.state.x===0.5&&input.state.y===0.5,"primary down sets brush and coordinates");
const request=input.state.pending;
canvas.fire("pointerdown",{pointerId:2,clientX:80});
canvas.fire("pointermove",{pointerId:2,clientX:90});
canvas.fire("pointerup",{pointerId:2});
check(input.state.down&&input.state.pointerId===1&&input.state.x===0.5,"second pointer cannot steal, move or end a stroke");
canvas.fire("pointermove",{clientX:90,clientY:20});
check(input.state.x===0.9&&input.state.y===0.8,"owning pointer moves in simulation coordinates");
check(Math.hypot(cfg.mouseVX,cfg.mouseVY)<=300.0001,"flick velocity remains clamped and smoothed");
canvas.fire("pointerleave");
check(input.state.down,"captured stroke survives leaving the canvas");
input.flushPaint();
check(paint.length===1&&paint[0].u===0.9&&paint[0].v===0.8,"one paint per flush uses the latest point");
input.flushPaint();
check(paint.length===1,"a flushed request is not replayed");
canvas.fire("pointermove",{clientX:150,clientY:-10});
check(input.state.pending===request,"paint requests reuse preallocated storage");
check(input.state.x===1&&input.state.y===1,"captured movement outside the canvas is clamped");
canvas.fire("pointerup");
check(!input.state.down&&captures.size===0&&cfg.mouseStrength===0&&cfg.mouseVX===0&&cfg.mouseVY===0,"release clears capture and stirring");
input.flushPaint();
check(paint.length===2,"normal release preserves the final queued paint");
canvas.fire("pointerdown");
canvas.fire("lostpointercapture");
check(!input.state.down&&cfg.mouseStrength===0&&input.state.pending===null,"lost capture ends stirring and drops pending paint");
canvas.fire("pointerdown");
canvas.fire("pointercancel");
check(!input.state.down&&captures.size===0&&input.state.pending===null,"pointer cancellation ends stirring and drops pending paint");
canvas.fire("pointerdown");
key("h");
win.fire("blur");
check(!input.state.down&&cfg.mouseStrength===0&&captures.size===0,"window blur ends an active stroke");
check(input.state.pending===null&&!input.state.keys.h,"window blur drops unflushed paint and clears held keys");

key("5");
check(mode===5,"unfocused numeric shortcut routes to the mode writer");
for(const tag of ["input","select","textarea","button","a"]){
	const target={closest:selector=>selector.includes(tag)?{}:null};
	key("r",{target});
	key(" ",{target});
	key("2",{target});
}
check(resets===0&&!cfg._paused&&mode===5,"form controls and links keep their native shortcuts");
key("r",{target:{isContentEditable:true}});
check(resets===0,"editable content keeps its keys");
for(const flag of ["ctrlKey","metaKey","altKey","isComposing","defaultPrevented"])key("r",{[flag]:true});
check(resets===0,"browser shortcuts, composition and handled events are ignored");
key(" ");
key(" ",{repeat:true});
check(cfg._paused===true,"holding space toggles pause only once");
key("r",{repeat:true});
check(resets===0,"holding reset does not repeatedly rebuild the scenario");
key("r");
check(resets===1,"plain reset still works");
const iters=Fluid.core.config.solverIters(cfg);
key("+",{repeat:true});
check(Fluid.core.config.solverIters(cfg)===iters+5,"pressure adjustment still allows key repeat");
cfg.useMultigrid=false;
const rb=cfg.pressureIters;
key("-");
check(cfg.pressureIters===rb-5&&cfg.mgCoarseIters===iters+5,
	"with multigrid off the same keys move the red-black sweep count");
cfg.useMultigrid=true;
cfg.N=64;
key("[");
check(resizes===0,"minimum-grid shortcut does not rebuild an unchanged grid");
cfg.N=1024;
key("]");
check(resizes===0,"maximum-grid shortcut does not rebuild an unchanged grid");
key("[");
check(resizes===1&&cfg.N===960,"grid shortcut still resizes within bounds");
const local=new Target();
let hookResets=0,hookResizes=0;
const keys=Fluid.app.input.createKeys(local,cfg,solver,id=>{mode=id;},{
	onReset(){hookResets++;},onResize(n){hookResizes++;cfg.N=n;}
});
local.fire("keydown",{key:"r"});
local.fire("keydown",{key:"]"});
check(hookResets===1&&hookResizes===1&&resets===1&&resizes===1,"standalone keys prefer the embedder's reset/resize hooks");
local.fire("keydown",{key:" "});
local.fire("blur");
check(!keys.state.keys[" "],"standalone custom-target blur clears held keys");
keys.destroy();
const bare=Fluid.app.input.createKeys({},cfg,solver,id=>{mode=id;});
bare.destroy();
input.destroy();
check(true,"listener setup and teardown tolerate missing add/removeEventListener methods");
console.log("\nall "+checks+" checks pass");
