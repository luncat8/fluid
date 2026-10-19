"use strict";
const assert=require("node:assert/strict");
require("../src/core/config.js");
require("../src/app/performance.js");

async function testRates(){
	let now=0,resolve,reject;
	const queue={onSubmittedWorkDone(){return new Promise((a,b)=>{resolve=a;reject=b;});}};
	const perf=Fluid.app.performance.create(queue,()=>now);
	for(let i=0;i<144;i++){
		perf.tick(1/144);
		if(i%3===0){
			perf.submitted(1/60,2);
			assert.equal(perf.state.busy,true);
			now+=8;resolve();await Promise.resolve();
			assert.equal(perf.state.busy,false);
		}
	}
	assert.ok(Math.abs(perf.state.rafHz-144)<1e-9);
	assert.ok(Math.abs(perf.state.simHz-48)<2);
	assert.ok(perf.state.simSpeed<0.85);
	assert.equal(perf.state.queueMs,8);
	assert.equal(perf.state.cpuMs,2);
	perf.submitted(0,1);resolve();await Promise.resolve();
	for(let i=0;i<150;i++)perf.tick(1/144);
	assert.equal(perf.state.simHz,0);
	assert.equal(perf.state.renderHz,0);
	perf.submitted(1/60,1);reject(new Error("lost"));await Promise.resolve();
	assert.equal(perf.state.busy,false);
	assert.equal(perf.state.error.message,"lost");
	console.log("PASS windowed counts, idle/paused rates, queue latency and rejection");
}
async function testMainLoop(){
	let callback,resolveQueue,clock=0,simFrames=0,renders=0,flushes=0;
	let cfg;
	const queue={onSubmittedWorkDone(){return new Promise(resolve=>{resolveQueue=resolve;});}};
	const canvas={focus(){},width:0,height:0};
	const panel={hidden:true,textContent:""};
	global.document={readyState:"loading",getElementById:id=>id==="c"?canvas:panel};
	global.window={addEventListener(){}};
	global.performance={now:()=>clock};
	global.requestAnimationFrame=fn=>{callback=fn;return 1;};
	global.cancelAnimationFrame=()=>{};
	Fluid.gpu={device:{async init(){return {device:{queue}};}},solver:{create(d,c,ctx,f,config){
		cfg=config;
		return {
			getStats(){return {frame:simFrames,motion:{maxVel:0}};},
			frame(dt,ns){assert.equal(dt*ns,cfg.frameDt);simFrames++;},
			renderOnly(){renders++;},pollReadback(){}
		};
	}}};
	Fluid.app.viewMenu={create(){return {setMode(){}};}};
	Fluid.app.hud={create(){return {update(){},setSubsteps(){}};}};
	Fluid.app.input={create(){return {flushPaint(){flushes++;}};}};
	Fluid.app.controls={create(){return {update(){}};}};
	require("../src/app/main.js");
	await Fluid.app.main.boot();
	function tick(dt=1000/144){clock+=dt;callback(clock);}
	async function complete(){resolveQueue();await Promise.resolve();}
	for(let i=0;i<1440;i++){
		tick();
		if(resolveQueue)await complete();
	}
	assert.ok(simFrames>=599&&simFrames<=600,"144 Hz callbacks still produce 60 sim/s");
	const before=simFrames;
	// Hold one submission unresolved across many callbacks.
	tick(20);
	for(let i=0;i<100;i++)tick(20);
	assert.equal(simFrames,before+1,"no GPU queue growth while busy");
	await complete();
	tick(20);
	assert.equal(simFrames,before+2,"simulation resumes after completion");
	await complete();
	cfg._paused=true;
	const paused=simFrames,pausedRenders=renders;
	for(let i=0;i<30;i++)tick();
	assert.equal(simFrames,paused);
	assert.equal(renders,pausedRenders,"paused clean canvas does not submit redraws");
	cfg._dirty=true;tick();assert.equal(renders,pausedRenders+1);await complete();
	cfg._stepOnce=true;tick();assert.equal(simFrames,paused+1);await complete();
	tick();assert.equal(simFrames,paused+1,"step once advances exactly once");
	assert.equal(flushes,simFrames);
	Fluid.app.main.stop();
	console.log("PASS real app loop: 144 Hz pacing, backpressure, paused redraw, single step");
}
(async()=>{await testRates();await testMainLoop();})().catch(error=>{console.error(error);process.exitCode=1;});
