(function(){
	"use strict";
	// Example game-style integration: this file only touches the public engine API
	// (create/update/pause/reset/setViewMode/addProbe/sample/stats/attachInput) and
	// never reaches into the solver's fields. That is the point of the page: if the
	// API cannot draw this, the API is missing something.
	const $=id=>document.getElementById(id);
	const PHASE_NAMES=["vacuum","solid","melting","liquid","boiling","vapor"];
	const MAX_PROBES=8;
	// one UI sweep every 200 ms: the fluid runs at frame rate, the text does not
	const UI_PERIOD=200;
	let sim=null;
	let raf=0;
	let last=0;
	let lastUi=0;
	let probeCursor=0;
	let viewButtons=null;
	const emitterIds=[];
	const probes=[];
	const stats=Fluid.engine.makeStats();
	const meter={};
	function setText(el,value){
		if(el&&el.textContent!==value)el.textContent=value;
	}
	function fmt(v,digits){
		if(!isFinite(v))return "—";
		const a=Math.abs(v);
		if(a===0)return "0";
		if(a<1e-3||a>=1e6)return v.toExponential(1);
		return v.toFixed(digits===undefined?3:digits);
	}
	function driftClass(v){
		const a=Math.abs(v);
		if(a<1e-6)return "ok";
		if(a<1e-3)return "warn";
		return "bad";
	}
	function buildProbeRows(){
		const body=$("probe-body");
		for(let i=0;i<MAX_PROBES;i++){
			const tr=document.createElement("tr");
			const cells=[];
			for(let c=0;c<8;c++){
				const td=document.createElement("td");
				tr.appendChild(td);
				cells.push(td);
			}
			tr.hidden=true;
			body.appendChild(tr);
			probes.push({id:-1,row:tr,cells,sample:Fluid.engine.makeProbeSample()});
		}
	}
	function metersInit(){
		const box=$("meters");
		box.textContent="";
		for(const name of ["m","e","cells","phase","motion","perf"]){
			const div=document.createElement("div");
			if(name==="perf")div.id="perf-sim";
			box.appendChild(div);
			meter[name]=div;
		}
	}
	function buildControls(){
		const select=$("scenario");
		for(const name of Fluid.core.config.scenarioNames){
			const o=document.createElement("option");
			o.value=name;
			o.textContent=name;
			select.appendChild(o);
		}
		select.value=sim.params.scenario;
		select.addEventListener("change",()=>{
			sim.reset(select.value);
			sim.singleStep();
		});
		$("pause").addEventListener("click",()=>{
			sim.paused=!sim.paused;
			setText($("pause"),sim.paused?"Resume":"Pause");
		});
		$("step").addEventListener("click",()=>sim.singleStep());
		$("reset").addEventListener("click",()=>{
			sim.reset();
			sim.singleStep();
		});
		$("clear-probes").addEventListener("click",()=>{
			for(const p of probes){
				if(p.id>=0)sim.removeProbe(p.id);
				p.id=-1;
				p.row.hidden=true;
			}
		});
		const toggleEmit=$("toggle-emitters");
		if(toggleEmit){
			toggleEmit.addEventListener("click",()=>{
				if(emitterIds.length){
					while(emitterIds.length)sim.removeEmitter(emitterIds.pop());
				}else{
					const n=sim.params.N;
					emitterIds.push(
						sim.addEmitter({kind:"heat",x:n*0.5,y:n*0.08,radius:n*0.14,value:180}),
						sim.addEmitter({kind:"freeze",x:n*0.5,y:n*0.88,radius:n*0.08,value:1100}),
						sim.addEmitter({kind:"dye",x:n*0.24,y:n*0.20,radius:n*0.04,value:0.35,dye:[0.95,0.2,0.18]}),
						sim.addEmitter({kind:"dye",x:n*0.50,y:n*0.16,radius:n*0.04,value:0.35,dye:[0.18,0.9,0.28]}),
						sim.addEmitter({kind:"dye",x:n*0.76,y:n*0.20,radius:n*0.04,value:0.35,dye:[0.2,0.45,0.98]}),
						sim.addEmitter({kind:"drain",x:n*0.50,y:n*0.56,radius:n*0.06,value:0.45})
					);
				}
				setText(toggleEmit,emitterIds.length?"Disable emitters":"Enable RGB + thermal rig");
				setText($("emitter-status"),sim.emitterCount()+" active");
			});
		}
		const views=$("views");
		const names=Fluid.core.config.viewModeNames;
		const buttons=[];
		for(let i=0;i<names.length;i++){
			const b=document.createElement("button");
			b.type="button";
			b.textContent=names[i];
			b.addEventListener("click",()=>sim.setViewMode(i));
			views.appendChild(b);
			buttons.push(b);
		}
		return buttons;
	}
	function addProbe(u,v){
		let slot=null;
		for(const p of probes){
			if(p.id<0){slot=p;break;}
		}
		if(!slot){
			// the table is full: recycle the oldest row so right-click always answers
			slot=probes[probeCursor%MAX_PROBES];
			probeCursor++;
			if(slot.id>=0)sim.removeProbe(slot.id);
			slot.id=-1;
		}
		// probe coordinates are cells with +y up; the canvas is y-down
		slot.id=sim.addProbe(u*sim.params.N,(1-v)*sim.params.N);
		slot.sample.valid=false;
		slot.row.hidden=slot.id<0;
		return slot.id;
	}
	function refreshProbes(){
		for(const p of probes){
			const c=p.cells;
			if(p.id<0){
				p.row.hidden=true;
				continue;
			}
			p.row.hidden=false;
			c[0].textContent=String(p.id);
			if(!sim.sample(p.id,p.sample)){
				c[1].textContent="—";
				for(let i=2;i<8;i++)c[i].textContent="—";
				continue;
			}
			c[1].textContent=Math.round(p.sample.x)+","+Math.round(p.sample.y);
			c[2].textContent=fmt(p.sample.m);
			c[3].textContent=fmt(p.sample.T,1)+" K";
			c[4].textContent=PHASE_NAMES[p.sample.phase]||"?";
			c[5].textContent=fmt(p.sample.u,1);
			c[6].textContent=fmt(p.sample.v,1);
			c[7].textContent=fmt(p.sample.p,1);
		}
	}
	function refreshMeters(viewButtons){
		sim.stats(stats);
		const m=stats.expected.m?stats.totals.m/stats.expected.m-1:0;
		const e=stats.expected.E?stats.totals.E/stats.expected.E-1:0;
		meter.m.className=driftClass(m);
		meter.e.className=driftClass(e);
		meter.m.textContent="mass "+(m>=0?"+":"")+m.toExponential(2);
		meter.e.textContent="energy "+(e>=0?"+":"")+e.toExponential(2);
		meter.cells.textContent="neg="+stats.cntNeg+" over="+stats.cntOver+
			" maxM="+fmt(stats.maxM,3);
		meter.phase.textContent="cells liq="+stats.phaseCounts[3]+" gas="+stats.phaseCounts[5];
		meter.motion.textContent="maxVel="+fmt(stats.motion.maxVel,0)+" maxDiv="+fmt(stats.motion.maxDiv,2);
		meter.perf.textContent="sim/s="+sim.perf.simHz.toFixed(0)+" · substeps="+sim.substeps+
			" · queue="+sim.perf.queueMs.toFixed(1)+" ms";
		// the space key toggles the same flag, so the button follows sim.paused
		// instead of remembering what it last clicked
		setText($("pause"),sim.paused?"Resume":"Pause");
		setText($("emitter-status"),sim.emitterCount()+" active");
		// the buttons follow cfg.viewMode, the one value every writer (keys, clicks) sets
		for(let i=0;i<viewButtons.length;i++)viewButtons[i].classList.toggle("active",i===sim.params.viewMode);
	}
	function frame(now){
		const dt=(now-last)/1000;
		last=now;
		sim.update(dt);
		if(sim.perf.error){
			const panel=$("error-panel");
			panel.hidden=false;
			panel.textContent="GPU queue failed: "+sim.perf.error.message;
			return;
		}
		if(now-lastUi>=UI_PERIOD){
			lastUi=now;
			refreshProbes();
			refreshMeters(viewButtons);
		}
		raf=requestAnimationFrame(frame);
	}
	async function boot(){
		const canvas=$("c");
		const status=$("status");
		sim=await Fluid.engine.create(canvas,{
			cfg:{N:256,scenario:"dam"},
			errorPanel:$("error-panel")
		});
		if(!sim){
			status.textContent="WebGPU unavailable - see the error panel";
			return;
		}
		sim.attachInput(canvas);
		buildProbeRows();
		viewButtons=buildControls();
		metersInit();
		canvas.addEventListener("contextmenu",e=>{
			e.preventDefault();
			const r=canvas.getBoundingClientRect();
			const u=(e.clientX-r.left)/r.width;
			const v=1-(e.clientY-r.top)/r.height;
			addProbe(Math.max(0,Math.min(1,u)),Math.max(0,Math.min(1,v)));
		});
		status.textContent="N="+sim.params.N+" · "+sim.params.scenario+" · right-click to probe";
		last=performance.now();
		lastUi=0;
		raf=requestAnimationFrame(frame);
		canvas.focus();
	}
	window.addEventListener("pagehide",()=>{
		if(raf)cancelAnimationFrame(raf);
		if(sim)sim.destroy();
		sim=null;
	});
	if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",boot);
	else boot();
})(globalThis);
