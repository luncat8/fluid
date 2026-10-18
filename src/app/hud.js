(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	function createHud(solver,cfg){
		const perfEl=document.getElementById("perf");
		const consEl=document.getElementById("conservation");
		const stateEl=document.getElementById("state");
		const statusEl=document.getElementById("status");
		let lastText=0;
		let fpsEma=60;
		function driftClass(v){
			const a=Math.abs(v);
			if(a<1e-6)return "ok";
			if(a<1e-3)return "warn";
			return "bad";
		}
		function fmt(v){
			if(!isFinite(v))return String(v);
			const a=Math.abs(v);
			if(a===0)return "0";
			if(a<1e-3||a>=1e6)return v.toExponential(2);
			return v.toFixed(4);
		}
		function tickFrame(dtRaw){
			const fps=1/Math.max(dtRaw,1e-6);
			fpsEma=fpsEma*0.9+fps*0.1;
		}
		function setSubsteps(ns){
			substepsShown=ns;
		}
		function update(now){
			if(now-lastText<200)return;
			lastText=now;
			if(!solver)return;
			const s=solver.getStats();
			if(perfEl){
				perfEl.textContent="fps "+Math.round(fpsEma)+"  N="+cfg.N+"  dt="+cfg.dt.toFixed(4)+
					"  K="+cfg.pressureIters+"  maxVel="+fmt(s.motion.maxVel)+"  frame#"+s.frame;
			}
			if(consEl){
				const kinds=["m","dr","dg","db","E"];
				const totals=[s.totals.m,s.totals.dr,s.totals.dg,s.totals.db,s.totals.E];
				const expects=[s.expected.m,s.expected.dr,s.expected.dg,s.expected.db,s.expected.E];
				const mins=[s.mins.m,s.mins.dr,s.mins.dg,s.mins.db,s.mins.E];
				const lines=[];
				for(let i=0;i<5;i++){
					const drift=Fluid.core.stats.driftRel(totals[i],expects[i]);
					lines.push(kinds[i]+": total="+fmt(totals[i])+" exp="+fmt(expects[i])+
						" drift="+drift.toExponential(2)+" min="+fmt(mins[i])+" ["+driftClass(drift)+"]");
				}
				lines.push("maxM="+fmt(s.maxM)+" neg="+s.cntNeg+" over="+s.cntOver+
					" maxDiv="+fmt(s.motion.maxDiv)+" p["+fmt(s.motion.pMin)+".."+fmt(s.motion.pMax)+"]");
				consEl.textContent=lines.join("\n");
			}
			if(stateEl){
				const modeName=Fluid.core.config.viewModeNames[cfg.viewMode]||String(cfg.viewMode);
				const paintName=Fluid.core.config.paintModeNames[cfg.paintMode]||String(cfg.paintMode);
				stateEl.textContent="mode="+modeName+"  scenario="+cfg.scenario+"  paint="+paintName+
					"  gravity="+(cfg.gravityY!==0?"on":"off")+"  "+(cfg._paused?"PAUSED":"running")+
					"\nkeys: 0-5 view · space pause · . step · r reset · n scenario · b paint(dye/wall/erase) · c color · g gravity · +/- K · [ ] size · s dump · h help";
			}
			if(statusEl){
				const driftM=Fluid.core.stats.driftRel(s.totals.m,s.expected.m);
				statusEl.textContent=(cfg._paused?"paused":"running")+" · drift m "+driftM.toExponential(1)+" ["+driftClass(driftM)+"]";
			}
		}
		return {update,tickFrame,setSubsteps};
	}
	Fluid.app.hud={create:createHud};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.hud;
})(globalThis);
