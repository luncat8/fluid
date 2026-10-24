(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	// the demo page is the kitchen sink for the engine (0.7.0): it applies the
	// browser-only A/B overrides from the query string, wires the DOM diagnostics and
	// leaves the frame loop, substep policy and readbacks to Fluid.engine.
	let sim=null;
	let hud=null;
	let controls=null;
	let rafId=0;
	let lastTime=0;
	let canvas=null;
	function queryOverrides(cfg){
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
	}
	async function boot(){
		canvas=document.getElementById("c");
		if(!canvas)return;
		const errorPanel=document.getElementById("error-panel");
		const status=document.getElementById("status");
		if(status)status.textContent="Requesting WebGPU adapter...";
		const cfg=Object.assign({},Fluid.core.config.defaults);
		queryOverrides(cfg);
		sim=await Fluid.engine.create(canvas,{cfg,errorPanel});
		if(!sim){
			if(status)status.textContent="WebGPU init failed - see error panel";
			return;
		}
		if(status)status.textContent="Running";
		const viewMenu=Fluid.app.viewMenu.create(sim.params);
		sim.bindViewMode(viewMenu.setMode);
		sim.attachInput(canvas);
		hud=Fluid.app.hud.create(sim.solver,sim.params,sim.perf);
		controls=Fluid.app.controls.create(sim.params,sim.solver,sim.setViewMode);
		lastTime=performance.now();
		rafId=requestAnimationFrame(frame);
		canvas.focus();
	}
	function frame(now){
		const dt=(now-lastTime)/1000;
		lastTime=now;
		const ns=sim.update(dt);
		if(sim.perf.error){
			const panel=document.getElementById("error-panel");
			panel.hidden=false;
			panel.textContent="GPU queue failed: "+sim.perf.error.message;
			return;
		}
		if(ns>0&&hud)hud.setSubsteps(ns);
		if(hud)hud.update(now);
		if(controls)controls.update(now);
		rafId=requestAnimationFrame(frame);
	}
	function stop(){
		if(rafId)cancelAnimationFrame(rafId);
		rafId=0;
		if(sim){
			sim.destroy();
			sim=null;
		}
	}
	Fluid.app.main={boot,stop};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.main;
	if(typeof window!=="undefined"){
		if(document.readyState==="loading")window.addEventListener("DOMContentLoaded",boot);
		else boot();
	}
})(globalThis);
