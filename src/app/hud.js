(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	const $=id=>document.getElementById(id);
	function createHud(solver,cfg,perf){
		const perfEl=$("perf");
		const consEl=$("conservation");
		const stateEl=$("state");
		const statusEl=$("status");
		const copyBtn=$("copyBtn");
		const copyFeedback=$("copyFeedback");
		let lastText=0;
		let substepsShown=1;
		// cache last rendered strings to avoid DOM thrash
		let lastPerf="",lastCons="",lastState="",lastStatus="";
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
			if(!b.active)return cfg.scenario==="dam"?"dam float: disabled (y to enable)":"dam float: inactive (dam scenario only)";
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
		function phaseLine(s){
			const pc=s.phaseCounts||[0,0,0,0,0,0];
			const lat=s.ledger?s.ledger.latentChargedE:0;
			const fMom=s.ledger?s.ledger.frozenMomentum:0;
			const sat=s.diag&&s.diag.sSaturated?s.diag.sSaturated:0;
			return "phases vac="+pc[0]+" sol="+pc[1]+" melt="+pc[2]+" liq="+pc[3]+" boil="+pc[4]+" gas="+pc[5]+
				" | latentE="+fmt(lat)+" fMom="+fmt(fMom)+" sSat="+sat;
		}
		function setSubsteps(ns){
			substepsShown=ns;
		}
		// fixed mode still runs substepsFixed steps; only auto has a live maximum
		function substepLabel(){
			return cfg.substepsAuto?substepsShown+"/"+cfg.substepMax+" auto":substepsShown+" fixed";
		}
		function performanceLine(){
			if(!perf)return "performance: unavailable";
			return "sim/s="+perf.simHz.toFixed(1)+" render/s="+perf.renderHz.toFixed(1)+
				" rAF/s="+perf.rafHz.toFixed(1)+" speed="+perf.simSpeed.toFixed(2)+"x"+
				" CPU encode="+perf.cpuMs.toFixed(2)+"ms queue="+perf.queueMs.toFixed(2)+"ms"+
				" vCap="+Math.round(cfg.vClamp);
		}
		// per-group GPU timestamps (0.6.0 t1); empty while the toggle is off
		function gpuTimingLine(){
			if(!cfg.gpuTiming)return "";
			const p=solver&&solver.getProfile?solver.getProfile():null;
			if(!p)return "gpu timing: unavailable (no timestamp-query)";
			if(!p.valid)return "gpu timing: waiting for a sample";
			const names=Fluid.gpu.profiler.GROUP_NAMES;
			const ns=Math.max(1,p.substeps);
			let line="gpu ms/substep";
			for(let i=0;i<5;i++)line+=" "+names[i]+"="+(p.ms[i]/ns).toFixed(2);
			line+=" | ms/frame disp="+p.ms[6].toFixed(2)+" render="+p.ms[7].toFixed(2)+
				" stats="+p.ms[5].toFixed(2)+" (frame#"+p.frame+" ns="+p.substeps+")";
			return line;
		}
		function cflBound(s){
			return 2*s.motion.maxVel*cfg.dt;
		}
		function recommendedK(){
			// with multigrid, K is the coarse sweep count, not the number of fine
			// sweeps; the plain-grid rule of thumb (0.18N+20) does not apply and
			// the "try more K" nudge would just spend time on a cheap level.
			if(cfg.useMultigrid)return 0;
			const want=Fluid.core.config.recommendedIters(cfg.N);
			if(cfg.pressureIters<want)return want;
			return 0;
		}
		function healthLine(s){
			const cfl=cflBound(s);
			const recK=recommendedK();
			const epsNeg=1e-9;
			const hasNeg=s.mins.m<-epsNeg||s.mins.dr<-epsNeg||s.mins.dg<-epsNeg||s.mins.db<-epsNeg||s.mins.E<-epsNeg||s.cntNeg>0;
			const mildOver=s.cntOver>0&&s.maxM<=1.05;
			const modOver=s.maxM>1.05&&s.maxM<=1.5;
			const severe=s.maxM>1.5||s.maxM>2;
			const capped=cfg.substepMax&&s.motion&&cfl>1&&substepsShown>=cfg.substepMax;
			const velCapped=cfg.vClamp>0&&s.motion.maxVel>=0.98*cfg.vClamp;
			let line="CFL bound="+fmt(cfl)+(cfl>1?" [transport limiting likely · try more substeps"+(capped?" (capped at "+cfg.substepMax+")":"")+"]":"");
			if(velCapped)line+=" | velocity capped at "+Math.round(cfg.vClamp)+" (substep budget)";
			if(hasNeg)line+=" | NEGATIVE VALUES: inspect minima (threshold 1e-9)";
			if(s.diag&&s.diag.sSaturated>0)line+=" | S saturated "+s.diag.sSaturated+" cells";
			if(s.diag&&s.diag.residual)line+=" | residual "+fmt(s.diag.residual);
			if(s.diag&&s.diag.poissonRel)line+=" | poisson residual "+fmt(s.diag.poissonRel);
			if(s.motion.maxDiv>1)line+=" | high divergence "+fmt(s.motion.maxDiv)+" (K="+Fluid.core.config.solverIters(cfg)+(cfg.useMultigrid?" coarse":"")+(recK?" → try "+recK:"")+")";
			else if(s.motion.maxDiv>0.5)line+=" | divergence "+fmt(s.motion.maxDiv);
			if(severe)line+=" | SEVERE COMPRESSION: conservation is not stability";
			else if(modOver)line+=" | compression maxM="+fmt(s.maxM);
			else if(mildOver)line+=" | mild overpack "+s.cntOver+" cells";
			return line;
		}
		// solver quality in the units plan 4.1 judges: the true poisson residual over
		// the rhs scale. divAfter (maxDiv above) floors at max|S| in boiling flow.
		function poissonLine(s){
			const d=s.diag;
			if(!d||d.poissonRes===undefined)return "poisson: n/a (no residual channel)";
			return "poisson: res="+d.poissonRes+" rhs="+d.poissonRhs+" true="+(d.poissonRhs>0?d.poissonRes/d.poissonRhs:0);
		}
		// build a copyable diagnostics snapshot (used by button and 's' key)
		function buildSnapshot(){
			if(!solver)return "no solver";
			const s=solver.getStats();
			const kinds=["m","dr","dg","db","E"];
			const totals=[s.totals.m,s.totals.dr,s.totals.dg,s.totals.db,s.totals.E];
			const expects=[s.expected.m,s.expected.dr,s.expected.dg,s.expected.db,s.expected.E];
			const lines=[];
			for(let i=0;i<5;i++){
				const drift=Fluid.core.stats.driftRel(totals[i],expects[i]);
				lines.push(kinds[i]+": total="+totals[i]+" exp="+expects[i]+" drift="+drift+" ["+driftClass(drift)+"]");
			}
			lines.push("min m="+s.mins.m+" dr="+s.mins.dr+" dg="+s.mins.dg+" db="+s.mins.db+" E="+s.mins.E);
			lines.push("maxM="+s.maxM+" neg="+s.cntNeg+" over="+s.cntOver+" maxDiv="+s.motion.maxDiv+" p["+s.motion.pMin+".."+s.motion.pMax+"] maxVel="+s.motion.maxVel);
			lines.push(poissonLine(s));
			lines.push(phaseLine(s));
			const solverTag=cfg.useMultigrid?"mgS"+cfg.mgCoarsen+"x"+cfg.mgCycles+"(pre"+cfg.mgPre+"/post"+cfg.mgPost+")":
				"rb";
			lines.push("statsFrame#"+s.statsFrame+" N="+cfg.N+" dt="+cfg.dt+" K="+Fluid.core.config.solverIters(cfg)+(cfg.useMultigrid?" coarse":"")+" "+solverTag+
				" ns="+substepLabel()+" scenario="+cfg.scenario+" FCT="+cfg.fctLimiter+" gravity="+cfg.gravityY);
			lines.push(performanceLine());
			const gtl=gpuTimingLine();
			if(gtl)lines.push(gtl);
			lines.push(healthLine(s));
			lines.push(buoyLine());
			return lines.join("\n");
		}
		async function copyToClipboard(){
			const text=buildSnapshot();
			let ok=false;
			try{
				if(navigator.clipboard&&navigator.clipboard.writeText){
					await navigator.clipboard.writeText(text);
					ok=true;
				}else{
					const ta=document.createElement("textarea");
					ta.value=text;
					ta.style.position="fixed";ta.style.opacity="0";
					document.body.appendChild(ta);
					ta.select();
					ok=document.execCommand("copy");
					ta.remove();
				}
			}catch(e){ok=false;}
			if(copyFeedback){
				copyFeedback.textContent=ok?"copied":"copy failed";
				setTimeout(()=>{if(copyFeedback)copyFeedback.textContent="";},1500);
			}
			console.log(text);
		}
		if(copyBtn){
			copyBtn.addEventListener("click",copyToClipboard);
		}
		function update(now){
			if(now-lastText<200)return;
			lastText=now;
			if(!solver)return;
			const s=solver.getStats();
			if(perfEl){
				const gtl=gpuTimingLine();
				const txt=performanceLine()+(gtl?"\n"+gtl:"")+"\nN="+cfg.N+"  dt="+cfg.dt.toFixed(4)+
					"  K="+Fluid.core.config.solverIters(cfg)+(cfg.useMultigrid?" coarse mg"+cfg.mgCoarsen+"x"+cfg.mgCycles:"")+
					"  ns="+substepLabel()+
					"  maxVel="+fmt(s.motion.maxVel)+"  frame#"+s.frame+" stats#"+s.statsFrame+
					"\n"+healthLine(s);
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
				if(s.thermal)lines.push("T["+fmt(s.thermal.TMin)+".."+fmt(s.thermal.TMax)+"] mean="+fmt(s.thermal.TMean)+
					" heatIn="+fmt(s.thermal.heatIn)+" heatOut="+fmt(s.thermal.heatOut)+" KE="+fmt(s.thermal.kinetic));
				lines.push(phaseLine(s));
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
					"  FCT="+(cfg.fctLimiter?"on":"off")+"  solids="+(cfg.overlay>0?"on":"off")+
					"  gravity="+(cfg.gravityY!==0?"on":"off")+"  "+(cfg._paused?"PAUSED":"running")+
					"\n"+buoyLine()+
					"\nkeys: 0-7 view · space pause · . step · r reset · n scenario · b paint · c color · f FCT · o solids · g gravity · u substeps · y buoy · +/- K · [ ] size · s dump · h help";
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
		return {update,setSubsteps,copyToClipboard,buildSnapshot};
	}
	Fluid.app.hud={create:createHud};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.hud;
})(globalThis);
