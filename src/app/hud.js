(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	function createHud(solver){
		const perfEl=document.getElementById("perf");
		const consEl=document.getElementById("conservation");
		const stateEl=document.getElementById("state");
		const statusEl=document.getElementById("status");
		const errorPanel=document.getElementById("error-panel");
		const probeOut=document.getElementById("probe-output");
		let lastText=0;
		let fpsSmoothing=0;
		let frameMs=0;
		let lastFpsTime=performance.now();
		let frames=0;
		function driftClass(v){
			let a=Math.abs(v);
			if(a<1e-6) return "ok";
			if(a<1e-3) return "warn";
			return "bad";
		}
		function fmt(v){
			if(!isFinite(v)) return String(v);
			let a=Math.abs(v);
			if(a===0) return "0";
			if(a<1e-3||a>1e6) return v.toExponential(2);
			return v.toFixed(4);
		}
		function update(now){
			if(now-lastText<200) return;
			lastText=now;
			if(!solver) return;
			let cfg=solver.cfg;
			let s=solver.getStats();
			if(perfEl){
				let fps=Math.round(1000/ Math.max(frameMs,1));
				perfEl.textContent="fps "+fps+"  frame "+frameMs.toFixed(1)+"ms  N="+cfg.N+"  dt="+cfg.dt.toFixed(4)+"  jacobi="+cfg.jacobiIters+"  frame#"+s.frame;
			}
			if(consEl){
				let lines=[];
				let kinds=["m","dr","dg","db","E"];
				let totals=[s.totals.m,s.totals.dr,s.totals.dg,s.totals.db,s.totals.E];
				let expects=[s.expected.m,s.expected.dr,s.expected.dg,s.expected.db,s.expected.E];
				let mins=[s.mins.m,s.mins.dr,s.mins.dg,s.mins.db,s.mins.E];
				for(let i=0;i<5;i++){
					let drift=Fluid.core.stats.driftRel(totals[i],expects[i]);
					let cls=driftClass(drift);
					let color=cls==="ok"?"#7fd8a0":cls==="warn"?"#e0c56a":"#ef9587";
					lines.push(kinds[i]+": total="+fmt(totals[i])+" exp="+fmt(expects[i])+" drift="+fmt(drift)+" min="+fmt(mins[i])+"  ["+cls+"]");
				}
				lines.push("maxM="+fmt(s.maxM)+" negCells="+s.cntNeg+" overPack="+s.cntOver);
				consEl.textContent=lines.join("\n");
				consEl.style.color="#d5e3ed";
				// color spans not needed for now
			}
			if(stateEl){
				let modeName=Fluid.core.config.viewModeNames[cfg.viewMode]||String(cfg.viewMode);
				stateEl.textContent="mode="+modeName+" ("+cfg.viewMode+")  paused="+(cfg._paused?"yes":"no")+"  gravityY="+cfg.gravityY+"  conductivity="+cfg.conductivity+"  \nkeys: 0-5 view  space pause  . step  r reset  h help  +/- iter  [ ] size  g gravity  s dump  c color";
			}
			if(statusEl){
				let paused=cfg._paused?"paused":"running";
				let driftM=Fluid.core.stats.driftRel(s.totals.m,s.expected.m);
				let col=driftClass(driftM);
				statusEl.textContent="Probe passed \u00b7 "+paused+" \u00b7 drift m "+fmt(driftM)+" ["+col+"]";
			}
		}
		function setFrameMs(ms){
			frameMs=ms;
		}
		function showError(msg){
			if(errorPanel){
				errorPanel.hidden=false;
				errorPanel.textContent=msg;
			}
		}
		function showProbe(report){
			if(probeOut){
				probeOut.textContent=JSON.stringify(report,null,"\t");
			}
		}
		return {update,setFrameMs,showError,showProbe};
	}
	Fluid.app.hud={create:createHud};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.hud;
})(globalThis);
