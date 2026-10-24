(function(){
	"use strict";
	// Example game-style integration: this file only touches the public engine API
	// (create/update/pause/reset/setViewMode/addProbe/sample/stats/attachInput and
	// addEmitter/removeEmitter) and never reaches into the solver's fields. That is
	// the point of the page: if the API cannot draw this, the API is missing
	// something.
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
	const probes=[];
	// the playground's sources: three heaters along the floor, two cold plates under
	// the ceiling and one dye jet. all of them are plain engine objects, so a game
	// moves or removes them the same way.
	const emitters=[];
	const EMITTER_SPOTS=[
		{kind:"heat",x:0.28,y:0.03,r:0.06,value:220},
		{kind:"heat",x:0.50,y:0.03,r:0.06,value:280},
		{kind:"heat",x:0.72,y:0.03,r:0.06,value:220},
		{kind:"cold",x:0.35,y:0.97,r:0.08,value:120},
		{kind:"cold",x:0.65,y:0.97,r:0.08,value:120},
		{kind:"dye",x:0.50,y:0.42,r:0.05,value:0.5,dye:[0.1,0.6,1]}
	];
	const emitterSpec={kind:"dye",x:0,y:0,radius:0,value:0,dye:[0,0,0]};
	// the moving body (t4): a paddle the page drives across the tank. the engine owns
	// the geometry and the mask; the page only says where it is and how fast, so a game
	// substitutes its own physics for these two lines and keeps the push.
	const paddle={id:-1,x:0.5,dir:1,speed:0.22,y:0.42};
	const bodySpec={shape:"box",x:0,y:0,hw:0,hh:0,vx:0,vy:0};
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
	// one reused spec object: addEmitter copies it into a preallocated slot, so the
	// page allocates nothing per call either
	function installEmitters(){
		const N=sim.params.N;
		for(const spot of EMITTER_SPOTS){
			emitterSpec.kind=spot.kind;
			emitterSpec.x=Math.round(spot.x*N);
			emitterSpec.y=Math.round(spot.y*N);
			emitterSpec.radius=Math.max(2,spot.r*N);
			emitterSpec.value=spot.value;
			const dye=emitterSpec.dye;
			dye[0]=spot.dye?spot.dye[0]:1;
			dye[1]=spot.dye?spot.dye[1]:1;
			dye[2]=spot.dye?spot.dye[2]:1;
			const id=sim.addEmitter(emitterSpec);
			if(id>=0)emitters.push(id);
		}
	}
	function toggleEmitters(){
		if(emitters.length){
			for(const id of emitters)sim.removeEmitter(id);
			emitters.length=0;
		}else{
			installEmitters();
		}
		setText($("emitters"),emitters.length?"Emitters ("+emitters.length+")":"Emitters (off)");
	}
	function toggleBody(){
		const N=sim.params.N;
		if(paddle.id>=0){
			sim.removeCollider(paddle.id);
			paddle.id=-1;
		}else{
			bodySpec.x=Math.round(paddle.x*N);
			bodySpec.y=Math.round(paddle.y*N);
			bodySpec.hw=Math.max(2,Math.round(0.02*N));
			bodySpec.hh=Math.max(3,Math.round(0.06*N));
			bodySpec.vx=0;
			bodySpec.vy=0;
			paddle.id=sim.addCollider(bodySpec);
			if(paddle.id<0)paddle.id=-1;
		}
		setText($("body"),paddle.id>=0?"Body (on)":"Body (off)");
	}
	// a wall bounce in page units, then one moveCollider: the engine diffs the mask and
	// imprints the velocity on the water the paddle is about to reach
	function driveBody(dt){
		// a paused sim does not step, so moving the body would only queue a jump for
		// the resume frame (and a jump wider than a patch is a teleport, not a push)
		if(paddle.id<0||sim.paused)return;
		const N=sim.params.N;
		paddle.x+=paddle.dir*paddle.speed*dt;
		if(paddle.x>0.86){paddle.x=0.86;paddle.dir=-1;}
		else if(paddle.x<0.14){paddle.x=0.14;paddle.dir=1;}
		sim.moveCollider(paddle.id,Math.round(paddle.x*N),Math.round(paddle.y*N),paddle.dir*paddle.speed*N,0);
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
		$("emitters").addEventListener("click",toggleEmitters);
		$("body").addEventListener("click",toggleBody);
		$("clear-probes").addEventListener("click",()=>{
			for(const p of probes){
				if(p.id>=0)sim.removeProbe(p.id);
				p.id=-1;
				p.row.hidden=true;
			}
		});
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
		// the buttons follow cfg.viewMode, the one value every writer (keys, clicks) sets
		for(let i=0;i<viewButtons.length;i++)viewButtons[i].classList.toggle("active",i===sim.params.viewMode);
	}
	function frame(now){
		// a page that was hidden (or a slow first frame) must not fling the paddle
		const dt=Math.min((now-last)/1000,0.1);
		last=now;
		driveBody(dt);
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
			cfg:{N:256,scenario:"still"},
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
		toggleEmitters();
		canvas.addEventListener("contextmenu",e=>{
			e.preventDefault();
			const r=canvas.getBoundingClientRect();
			const u=(e.clientX-r.left)/r.width;
			const v=1-(e.clientY-r.top)/r.height;
			addProbe(Math.max(0,Math.min(1,u)),Math.max(0,Math.min(1,v)));
		});
		status.textContent="N="+sim.params.N+" · "+sim.params.scenario+" · right-click to probe · "+
			emitters.length+" sources";
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
