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
	let perf=null;
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
		// A/B overrides for the browser (no rebuild): ?mg=0 falls back to the
		// plain red-black solve, ?s=2/8 picks another coarsening factor and
		// ?cycles=1/3 another cycle count. the HUD prints the active solver.
		const q=new URLSearchParams(typeof location!=="undefined"&&location.search?location.search:"");
		if(q.has("mg"))cfg.useMultigrid=q.get("mg")!=="0";
		if(q.has("s"))cfg.mgCoarsen=Math.max(2,Math.round(+q.get("s")||4));
		if(q.has("cycles")){
			cfg.mgCycles=Math.max(1,Math.round(+q.get("cycles")||2));
			cfg.mgCyclesPinned=1;
		}
		cfg.productCycleGate=1;
		if(q.has("pre"))cfg.mgPre=Math.max(1,Math.round(+q.get("pre")||4));
		if(q.has("post"))cfg.mgPost=Math.max(1,Math.round(+q.get("post")||8));
		if(q.has("k"))Fluid.core.config.setSolverIters(cfg,Math.max(1,Math.round(+q.get("k")||60)));
		cfg.texW=cfg.N+2;cfg.texH=cfg.N+2;
		cfg._paused=false;cfg._stepOnce=false;cfg._dirty=true;
		if(status)status.textContent="Running";
		solver=Fluid.gpu.solver.create(initRes.device,canvas,initRes.context,initRes.format,cfg);
		const viewMenu=Fluid.app.viewMenu.create(cfg);
		perf=Fluid.app.performance.create(initRes.device.queue);
		hud=Fluid.app.hud.create(solver,cfg,perf.state);
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
		const elapsed=dtRaw;
		if(dtRaw>0.1)dtRaw=0.1;
		if(dtRaw<=0)dtRaw=1e-6;
		perf.tick(Math.max(elapsed,1e-6));
		if(perf.state.error){
			const panel=document.getElementById("error-panel");
			panel.hidden=false;panel.textContent="GPU queue failed: "+perf.state.error.message;
			return;
		}
		const cpuStart=performance.now();
		const frameDt=cfg.frameDt;
		let didSim=false;
		if(!cfg._paused||cfg._stepOnce){
			// Preserve fractional ticks at high refresh rates, but drop long GPU backlog.
			accumulator=Math.min(accumulator+dtRaw,frameDt*2);
			if(cfg._stepOnce)accumulator=frameDt;
			if(accumulator>=frameDt&&!perf.state.busy){
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
				// speed the chosen substep count can transport without the
				// positivity limiter clipping outflow (config.velocityCap)
				cfg.vClamp=Fluid.core.config.velocityCap(cfg,ns);
				if(inputState)inputState.flushPaint();
				const needStats=st.frame%cfg.statsEvery===0;
				solver.frame(dts,ns,needStats);
				perf.submitted(frameDt,performance.now()-cpuStart);
				cfg._stepOnce=false;
				cfg._dirty=false;
				didSim=true;
			}
		}
		if(cfg._paused&&!cfg._stepOnce)accumulator=0;
		if(!didSim&&!perf.state.busy&&cfg._dirty){
			solver.renderOnly();
			perf.submitted(0,performance.now()-cpuStart);
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
