(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	let solver=null;
	let hud=null;
	let inputState=null;
	let cfg=null;
	let rafId=0;
	let lastTime=0;
	let accumulator=0;
	let canvas=null;
	async function boot(){
		canvas=document.getElementById("c");
		if(!canvas) return;
		const errorPanel=document.getElementById("error-panel");
		const status=document.getElementById("status");
		if(status) status.textContent="Requesting WebGPU adapter...";
		const initRes=await Fluid.gpu.device.init(canvas,errorPanel);
		if(!initRes){
			if(status) status.textContent="WebGPU init failed - see error panel";
			return;
		}
		const device=initRes.device;
		const context=initRes.context;
		const format=initRes.format;
		cfg=new Fluid.core.config.defaults.constructor?Object.assign({},Fluid.core.config.defaults):{};
		// ensure cfg is mutable copy
		cfg=Object.assign({},Fluid.core.config.defaults);
		cfg.texW=cfg.N+2; cfg.texH=cfg.N+2;
		cfg._paused=false; cfg._stepOnce=false; cfg._needStats=false;
		const probeReport=initRes.report;
		// show probe report in probe-output if exists
		const probeOut=document.getElementById("probe-output");
		if(probeOut){
			probeOut.textContent=JSON.stringify(probeReport,null,"\t");
		}
		if(status) status.textContent="Probe passed - starting lab";
		solver=Fluid.gpu.solver.create(device,canvas,context,format,cfg);
		hud=Fluid.app.hud.create(solver);
		inputState=Fluid.app.input.create(canvas,cfg,solver);
		// set canvas size attributes
		canvas.width=cfg.N;
		canvas.height=cfg.N;
		// resize observer: handle canvas CSS scaling
		window.addEventListener("resize",()=>{
			// no texture resize on window resize, only on N change
		});
		lastTime=performance.now();
		accumulator=0;
		rafId=requestAnimationFrame(frame);
		canvas.focus();
	}
	function frame(now){
		let dtRaw=(now-lastTime)/1000;
		lastTime=now;
		if(dtRaw>0.1) dtRaw=0.1;
		const dt=cfg.dt;
		let didStep=false;
		let stepStart=performance.now();
		if(!cfg._paused||cfg._stepOnce){
			accumulator+=dtRaw;
			let substepsCap=Fluid.core.config.defaults.substepMax;
			let sub=0;
			while(accumulator>=dt && sub<substepsCap){
				solver.step(dt);
				accumulator-=dt;
				sub++;
				didStep=true;
				cfg._stepOnce=false;
				if(cfg._paused) break;
			}
			if(sub>=substepsCap) accumulator=0;
		}
		// even when paused we need to render once to keep display fresh
		if(!didStep && cfg._paused){
			// still need to drive a render? Our step does compute+render; for paused we just render last state without stepping
			// do a no-op step with dt=0? Better just request a render-only pass
			// For simplicity, when paused we still call solver.step(0) or just keep existing display; but our solver.step always does exchange, so we shouldn't step when paused
			// Instead we trigger a single render without compute: we can call a lightweight render-only encoding
			// For 0.1.0, paused means no exchange diffusion; we can just not submit new frame and keep last canvas; but we need HUD updates
		} else if(!didStep){
			// if no substep due to accumulator <dt, we still want to render at display rate? Our design renders only on step, so fast display would stutter
			// Alternative: render every frame regardless. For 0.1.0, stepping and rendering are coupled, but we can still render each frame by doing an extra display pass without exchange
			// Simpler: if we didn't step this frame but time has passed, we still want smooth FPS display; we can just step with smaller dt leftover? For now keep coupled
		}
		let frameMs=performance.now()-stepStart;
		if(hud){
			hud.setFrameMs(frameMs);
			hud.update(now);
		}
		rafId=requestAnimationFrame(frame);
	}
	function stop(){
		if(rafId) cancelAnimationFrame(rafId);
	}
	Fluid.app.main={boot,stop};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.main;
	// auto boot when script loads after DOM
	if(typeof window!=="undefined"){
		if(document.readyState==="loading") window.addEventListener("DOMContentLoaded",boot);
		else boot();
	}
})(globalThis);
