(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	function createHud(solver,cfg){
		const perfEl=document.getElementById("perf");
		const consEl=document.getElementById("conservation");
		const stateEl=document.getElementById("state");
		const statusEl=document.getElementById("status");
		const copyBtn=document.getElementById("copyBtn");
		const copyFeedback=document.getElementById("copyFeedback");
		let lastText=0;
		let fpsEma=60;
		let substepsShown=1;
		// cache last rendered strings to avoid DOM thrash
		let lastPerf="", lastCons="", lastState="", lastStatus="";
		function driftClass(v){
			const a=Math.abs(v);
			if(a<1e-6)return "ok";
			if(a<1e-3)return "warn";
			return "bad";
		}
		// the body is diagnostic, so its readout lives here and not in the solver
		function buoyLine(){
			const b=solver&&solver.buoy?solver.buoy.state:null;
			if(!b)return "buoy: no solver";
			if(!b.active)return "buoy: off (dam scenario only, y to enable)";
			return "buoy  y="+b.cy.toFixed(1)+"  vy="+b.vy.toFixed(2)+"  submerged="+(b.sub*100).toFixed(0)+
				"%  blue/cell="+b.blue.toFixed(3)+"  sample#"+b.samples;
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
		// build a copyable diagnostics snapshot (used by button and 's' key)
		function buildSnapshot(){
			if(!solver) return "no solver";
			const s=solver.getStats();
			const kinds=["m","dr","dg","db","E"];
			const totals=[s.totals.m,s.totals.dr,s.totals.dg,s.totals.db,s.totals.E];
			const expects=[s.expected.m,s.expected.dr,s.expected.dg,s.expected.db,s.expected.E];
			const lines=[];
			for(let i=0;i<5;i++){
				const drift=Fluid.core.stats.driftRel(totals[i],expects[i]);
				lines.push(kinds[i]+": total="+totals[i]+" exp="+expects[i]+" drift="+drift+" ["+driftClass(drift)+"]");
			}
			lines.push("maxM="+s.maxM+" neg="+s.cntNeg+" over="+s.cntOver+" maxDiv="+s.motion.maxDiv+" p["+s.motion.pMin+".."+s.motion.pMax+"] maxVel="+s.motion.maxVel);
			lines.push("statsFrame#"+s.statsFrame+" N="+cfg.N+" dt="+cfg.dt+" K="+cfg.pressureIters+
				" ns="+substepsShown+(cfg.substepsAuto?"/"+cfg.substepMax+" auto":" fixed")+" fps~"+Math.round(fpsEma));
			lines.push(buoyLine());
			return lines.join("\n");
		}
		async function copyToClipboard(){
			const text=buildSnapshot();
			let ok=false;
			try{
				if(navigator.clipboard && navigator.clipboard.writeText){
					await navigator.clipboard.writeText(text);
					ok=true;
				} else {
					const ta=document.createElement("textarea");
					ta.value=text;
					ta.style.position="fixed"; ta.style.opacity="0";
					document.body.appendChild(ta);
					ta.select();
					ok=document.execCommand("copy");
					ta.remove();
				}
			}catch(e){ ok=false; }
			if(copyFeedback){
				copyFeedback.textContent= ok ? "copied" : "copy failed";
				setTimeout(()=>{ if(copyFeedback) copyFeedback.textContent=""; }, 1500);
			}
			// also log for debugging
			console.log(text);
		}
		if(copyBtn){
			copyBtn.addEventListener("click", copyToClipboard);
		}

		function update(now){
			if(now-lastText<200) return;
			lastText=now;
			if(!solver) return;
			const s=solver.getStats();
			if(perfEl){
				const txt="fps "+Math.round(fpsEma)+"  N="+cfg.N+"  dt="+cfg.dt.toFixed(4)+
					"  K="+cfg.pressureIters+"  ns="+substepsShown+(cfg.substepsAuto?"/"+cfg.substepMax+" auto":"/1 fixed")+
					"  maxVel="+fmt(s.motion.maxVel)+"  frame#"+s.frame;
				if(txt!==lastPerf){
					perfEl.textContent=txt;
					lastPerf=txt;
				}
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
				const txt=lines.join("\n");
				if(txt!==lastCons){
					consEl.textContent=txt;
					lastCons=txt;
				}
			}
			if(stateEl){
				const modeName=Fluid.core.config.viewModeNames[cfg.viewMode]||String(cfg.viewMode);
				const paintName=Fluid.core.config.paintModeNames[cfg.paintMode]||String(cfg.paintMode);
				const txt="mode="+modeName+"  scenario="+cfg.scenario+"  paint="+paintName+
					"  FCT="+(cfg.fctLimiter?"on":"off")+"  gravity="+(cfg.gravityY!==0?"on":"off")+"  "+(cfg._paused?"PAUSED":"running")+
					"\n"+buoyLine()+
					"\nkeys: 0-7 view · space pause · . step · r reset · n scenario · b paint · c color · f FCT · g gravity · u substeps · y buoy · +/- K · [ ] size · s dump · h help";
				if(txt!==lastState){
					stateEl.textContent=txt;
					lastState=txt;
				}
			}
			if(statusEl){
				const driftM=Fluid.core.stats.driftRel(s.totals.m,s.expected.m);
				const txt=(cfg._paused?"paused":"running")+" · drift m "+driftM.toExponential(1)+" ["+driftClass(driftM)+"]";
				if(txt!==lastStatus){
					statusEl.textContent=txt;
					lastStatus=txt;
				}
			}
		}
		return {update,tickFrame,setSubsteps,copyToClipboard};
	}
	Fluid.app.hud={create:createHud};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.hud;
})(globalThis);
