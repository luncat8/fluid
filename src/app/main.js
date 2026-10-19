(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	let solver=null;
	let hud=null;
	let inputState=null;
	let controls=null;
	let cfg=null;
	let rafId=0;
	let lastTime=0;
	let accumulator=0;
	let canvas=null;
	async function boot(){
		canvas=document.getElementById("c");
		if(!canvas)return;
		const errorPanel=document.getElementById("error-panel");
		const status=document.getElementById("status");
		if(status)status.textContent="Requesting WebGPU adapter...";
		const initRes=await Fluid.gpu.device.init(canvas,errorPanel);
		if(!initRes){
			if(status)status.textContent="WebGPU init failed - see error panel";
			return;
		}
		cfg=Object.assign({},Fluid.core.config.defaults);
		cfg.texW=cfg.N+2;cfg.texH=cfg.N+2;
		cfg._paused=false;cfg._stepOnce=false;cfg._dirty=true;
		if(status)status.textContent="Running";
		solver=Fluid.gpu.solver.create(initRes.device,canvas,initRes.context,initRes.format,cfg);
		const viewMenu=Fluid.app.viewMenu.create(cfg);
		hud=Fluid.app.hud.create(solver,cfg);
		inputState=Fluid.app.input.create(canvas,cfg,solver,viewMenu.setMode);
		controls=Fluid.app.controls.create(cfg,solver,viewMenu.setMode);
		canvas.width=cfg.N;
		canvas.height=cfg.N;
		lastTime=performance.now();
		accumulator=0;
		rafId=requestAnimationFrame(frame);
		canvas.focus();
	}
	function frame(now){
		let dtRaw=(now-lastTime)/1000;
		lastTime=now;
		if(dtRaw>0.1)dtRaw=0.1;
		if(dtRaw<=0)dtRaw=1e-6;
		if(hud)hud.tickFrame(dtRaw);
		const frameDt=cfg.frameDt;
		let didSim=false;
		if(!cfg._paused||cfg._stepOnce){
			accumulator+=dtRaw;
			if(cfg._stepOnce)accumulator=frameDt;
			if(accumulator>=frameDt){
				if(accumulator>frameDt*4)accumulator=frameDt*4;
				accumulator-=frameDt;
				// substeps: the shared policy in config.js. auto follows the last readback
				// maxVel (<= statsEvery frames stale) plus the live drag speed, which is
				// known now and must not wait for the readback (draw blowup trigger). off
				// runs a fixed count, never one step: at 1/60 s the cfl ratio is far past
				// what the scheme resolves and the liquid only sees the outflow limiter.
				const st=solver.getStats();
				const maxVel=Math.max(st.motion.maxVel,Math.abs(cfg.mouseVX),Math.abs(cfg.mouseVY));
				const ns=Fluid.core.config.substepCount(cfg,maxVel);
				const dts=frameDt/ns;
				if(hud)hud.setSubsteps(ns);
				cfg.vClamp=cfg.vMax/frameDt;
				if(inputState)inputState.flushPaint();
				const needStats=st.frame%cfg.statsEvery===0;
				solver.frame(dts,ns,needStats);
				cfg._stepOnce=false;
				cfg._dirty=false;
				didSim=true;
			}
		}
		if(!didSim&&(cfg._dirty||cfg._paused)){
			solver.renderOnly();
			cfg._dirty=false;
		}
		if(hud)hud.update(now);
		if(controls)controls.update(now);
		solver.pollReadback();
		rafId=requestAnimationFrame(frame);
	}
	function stop(){
		if(rafId)cancelAnimationFrame(rafId);
	}
	Fluid.app.main={boot,stop};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.main;
	if(typeof window!=="undefined"){
		if(document.readyState==="loading")window.addEventListener("DOMContentLoaded",boot);
		else boot();
	}
})(globalThis);
