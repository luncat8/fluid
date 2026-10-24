(function(){
	"use strict";
	// Example game-style integration: this file only touches the public engine API
	// (create/update/pause/reset/setViewMode/addCollider/moveCollider/removeCollider/
	// addEmitter/addProbe/sample/stats/attachInput) and never reaches into solver fields.
	const $=id=>document.getElementById(id);
	const PHASE_NAMES=["vacuum","solid","melting","liquid","boiling","vapor"];
	const MAX_PROBES=8;
	const COLLIDER_MAX_SPEED=480;
	// one UI sweep every 200 ms: the fluid runs at frame rate, the text does not
	const UI_PERIOD=200;
	let sim=null;
	let raf=0;
	let last=0;
	let lastUi=0;
	let probeCursor=0;
	let viewButtons=null;
	let colliderId=-1;
	let collX=0,collY=0,collR=14,collPrevX=0,collPrevY=0,collN=256;
	let collDrag=false,collPtrId=-1;
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
	function clamp(v,lo,hi){
		return v<lo?lo:(v>hi?hi:v);
	}
	function spawnCollider(){
		if(!sim||colliderId>=0)return;
		const n=sim.params.N;
		collN=n;
		collR=Math.max(4,Math.round(n*0.055));
		collX=Math.round(n*0.58);
		collY=Math.round(n*0.14);
		collPrevX=collX;
		collPrevY=collY;
		colliderId=sim.addCollider({shape:"circle",x:collX,y:collY,radius:collR,vx:0,vy:0});
		syncColliderUi();
	}
	function clearCollider(){
		if(!sim||colliderId<0)return;
		sim.removeCollider(colliderId);
		colliderId=-1;
		collDrag=false;
		collPtrId=-1;
		syncColliderUi();
	}
	function syncColliderUi(){
		const btn=$("toggle-collider");
		if(btn)setText(btn,colliderId>=0?"Remove circle collider":"Spawn circle collider");
		const st=$("collider-status");
		if(st){
			setText(st,colliderId>=0
				?("r="+Math.round(collR)+" @ "+Math.round(collX)+","+Math.round(collY)+" (drag disc)")
				:"0 active");
		}
	}
	function attachColliderDrag(canvas){
		function pointerToCellX(e,rect){
			const w=rect.width>0?rect.width:1;
			return clamp(((e.clientX-rect.left)/w)*sim.params.N,collR,sim.params.N-collR);
		}
		function pointerToCellY(e,rect){
			const h=rect.height>0?rect.height:1;
			return clamp((1-(e.clientY-rect.top)/h)*sim.params.N,collR,sim.params.N-collR);
		}
		function stopDrag(){
			if(!collDrag)return;
			const pid=collPtrId;
			collDrag=false;
			collPtrId=-1;
			collPrevX=collX;
			collPrevY=collY;
			if(colliderId>=0&&sim)sim.moveCollider(colliderId,collX,collY,0,0);
			if(pid>=0&&canvas.hasPointerCapture&&canvas.hasPointerCapture(pid)){
				try{canvas.releasePointerCapture(pid);}catch(_){}
			}
			syncColliderUi();
		}
		canvas.addEventListener("pointerdown",e=>{
			if(!sim||colliderId<0)return;
			if(e.button!==undefined&&e.button!==0)return;
			const rect=canvas.getBoundingClientRect();
			const w=rect.width>0?rect.width:1,h=rect.height>0?rect.height:1;
			const px=((e.clientX-rect.left)/w)*sim.params.N;
			const py=(1-(e.clientY-rect.top)/h)*sim.params.N;
			const dx=px-collX,dy=py-collY;
			const hitR=collR+3;
			if(dx*dx+dy*dy>hitR*hitR)return;
			e.preventDefault();
			if(e.stopImmediatePropagation)e.stopImmediatePropagation();
			collDrag=true;
			collPtrId=e.pointerId!==undefined?e.pointerId:1;
			collX=clamp(px,collR,sim.params.N-collR);
			collY=clamp(py,collR,sim.params.N-collR);
			collPrevX=collX;
			collPrevY=collY;
			sim.moveCollider(colliderId,collX,collY,0,0);
			if(canvas.setPointerCapture&&e.pointerId!==undefined){
				try{canvas.setPointerCapture(e.pointerId);}catch(_){}
			}
			canvas.focus();
		});
		canvas.addEventListener("pointermove",e=>{
			if(!collDrag||(e.pointerId!==undefined&&e.pointerId!==collPtrId))return;
			e.preventDefault();
			if(e.stopImmediatePropagation)e.stopImmediatePropagation();
			const rect=canvas.getBoundingClientRect();
			collX=pointerToCellX(e,rect);
			collY=pointerToCellY(e,rect);
			if(sim&&sim.paused&&colliderId>=0)sim.moveCollider(colliderId,collX,collY,0,0);
		});
		canvas.addEventListener("pointerup",e=>{
			if(!collDrag||(e.pointerId!==undefined&&e.pointerId!==collPtrId))return;
			stopDrag();
		});
		canvas.addEventListener("pointercancel",e=>{
			if(!collDrag||(e.pointerId!==undefined&&e.pointerId!==collPtrId))return;
			stopDrag();
		});
		canvas.addEventListener("lostpointercapture",stopDrag);
		window.addEventListener("blur",stopDrag);
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
			setText($("status"),"N="+sim.params.N+" · "+sim.params.scenario+" · right-click to probe");
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
		const toggleColl=$("toggle-collider");
		if(toggleColl){
			toggleColl.addEventListener("click",()=>{
				if(colliderId>=0)clearCollider();
				else spawnCollider();
			});
		}
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
		// keys (space, [, ], n) mutate engine state directly; sync controls at 5 Hz
		setText($("pause"),sim.paused?"Resume":"Pause");
		const scenarioSel=$("scenario");
		if(scenarioSel&&scenarioSel.value!==sim.params.scenario)scenarioSel.value=sim.params.scenario;
		setText($("status"),"N="+sim.params.N+" · "+sim.params.scenario+" · right-click to probe");
		setText($("emitter-status"),sim.emitterCount()+" active");
		syncColliderUi();
		// the buttons follow cfg.viewMode, the one value every writer (keys, clicks) sets
		for(let i=0;i<viewButtons.length;i++)viewButtons[i].classList.toggle("active",i===sim.params.viewMode);
	}
	function stepColliderDrag(dt){
		if(!sim||colliderId<0)return;
		if(sim.params.N!==collN){
			const scale=sim.params.N/collN;
			collN=sim.params.N;
			collR=Math.max(4,Math.round(collR*scale));
			collX=clamp(collX*scale,collR,collN-collR);
			collY=clamp(collY*scale,collR,collN-collR);
			collPrevX=collX;
			collPrevY=collY;
			sim.setCollider(colliderId,{x:collX,y:collY,radius:collR,vx:0,vy:0});
			return;
		}
		if(!collDrag)return;
		const invDt=dt>1e-3?1/Math.min(dt,0.1):60;
		const vx=clamp((collX-collPrevX)*invDt,-COLLIDER_MAX_SPEED,COLLIDER_MAX_SPEED);
		const vy=clamp((collY-collPrevY)*invDt,-COLLIDER_MAX_SPEED,COLLIDER_MAX_SPEED);
		collPrevX=collX;
		collPrevY=collY;
		sim.moveCollider(colliderId,collX,collY,vx,vy);
	}
	function frame(now){
		const dt=(now-last)/1000;
		last=now;
		stepColliderDrag(dt);
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
		// register collider drag before attachInput so primary drags on the disc
		// move the rigid body without triggering fluid paint/stir
		attachColliderDrag(canvas);
		sim.attachInput(canvas);
		buildProbeRows();
		viewButtons=buildControls();
		metersInit();
		spawnCollider();
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
