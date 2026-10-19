(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	function createControls(cfg,solver){
		const panel=document.getElementById("controls-panel");
		if(!panel)return {update(){}};
		// scenario
		const scenSel=document.getElementById("ctrl-scenario");
		if(scenSel){
			const names=Fluid.core.config.scenarioNames;
			for(const n of names){
				const o=document.createElement("option");
				o.value=n;o.textContent=n;
				scenSel.appendChild(o);
			}
			scenSel.value=cfg.scenario;
			scenSel.addEventListener("change",()=>{
				cfg.scenario=scenSel.value;
				if(solver)solver.reset();
				cfg._dirty=true;
			});
		}
		// paint mode
		const paintBtns=document.querySelectorAll("[data-paint]");
		paintBtns.forEach(btn=>{
			btn.addEventListener("click",()=>{
				cfg.paintMode=parseInt(btn.dataset.paint,10);
				updatePaintHighlight();
			});
		});
		function updatePaintHighlight(){
			paintBtns.forEach(b=>{
				b.classList.toggle("active",parseInt(b.dataset.paint,10)===cfg.paintMode);
			});
		}
		updatePaintHighlight();
		// dye color
		const colorBtns=document.querySelectorAll("[data-dye-color]");
		const DYE_CSS=["#e04040","#40c040","#4060e0","#e0e040"];
		colorBtns.forEach(btn=>{
			const idx=parseInt(btn.dataset.dyeColor,10);
			btn.style.borderBottomColor=DYE_CSS[idx%DYE_CSS.length];
			btn.addEventListener("click",()=>{
				cfg.paintColor=idx;
				updateColorHighlight();
			});
		});
		function updateColorHighlight(){
			colorBtns.forEach(b=>{
				b.classList.toggle("active",parseInt(b.dataset.dyeColor,10)===(cfg.paintColor||0));
			});
		}
		// buttons: pause, step, reset
		const pauseBtn=document.getElementById("ctrl-pause");
		if(pauseBtn){
			pauseBtn.addEventListener("click",()=>{cfg._paused=!cfg._paused;});
		}
		const stepBtn=document.getElementById("ctrl-step");
		if(stepBtn){
			stepBtn.addEventListener("click",()=>{cfg._stepOnce=true;});
		}
		const resetBtn=document.getElementById("ctrl-reset");
		if(resetBtn){
			resetBtn.addEventListener("click",()=>{
				if(solver)solver.reset();
				cfg._dirty=true;
			});
		}
		// toggles: fct, gravity
		const fctChk=document.getElementById("ctrl-fct");
		if(fctChk){
			fctChk.checked=cfg.fctLimiter>0;
			fctChk.addEventListener("change",()=>{cfg.fctLimiter=fctChk.checked?1:0;cfg._dirty=true;});
		}
		const gravChk=document.getElementById("ctrl-gravity");
		if(gravChk){
			gravChk.checked=cfg.gravityY!==0;
			gravChk.addEventListener("change",()=>{
				cfg.gravityY=gravChk.checked?Fluid.core.config.defaults.gravityY:0;
			});
		}
		const subChk=document.getElementById("ctrl-substeps");
		if(subChk){
			subChk.checked=!!cfg.substepsAuto;
			subChk.addEventListener("change",()=>{cfg.substepsAuto=subChk.checked;});
		}
		// the body spawns from the scenario, so toggling it respawns instead of reset
		const buoyChk=document.getElementById("ctrl-buoy");
		if(buoyChk){
			buoyChk.checked=cfg.buoyEnabled>0;
			buoyChk.addEventListener("change",()=>{
				cfg.buoyEnabled=buoyChk.checked?1:0;
				if(solver&&solver.buoy)solver.buoy.reset();
				cfg._dirty=true;
			});
		}
		// a range input owns its value while the pointer is down: the 5 hz cfg -> dom
		// sync would otherwise snap the thumb back mid-drag and the change event then
		// never sees the value the pointer stopped on.
		const dragging=new Set();
		function trackSlider(el){
			if(!el)return;
			const release=()=>{dragging.delete(el);};
			el.addEventListener("pointerdown",()=>{dragging.add(el);});
			el.addEventListener("pointerup",release);
			el.addEventListener("pointercancel",release);
			el.addEventListener("blur",release);
			window.addEventListener("pointerup",release);
		}
		function syncSlider(el,out,value){
			if(!el||dragging.has(el))return;
			el.value=value;
			if(out)out.textContent=value;
		}
		// sliders: pressure iters, grid size, fixed substeps
		const kSlider=document.getElementById("ctrl-iters");
		const kVal=document.getElementById("ctrl-iters-val");
		if(kSlider){
			trackSlider(kSlider);
			kSlider.value=cfg.pressureIters;
			if(kVal)kVal.textContent=cfg.pressureIters;
			kSlider.addEventListener("input",()=>{
				cfg.pressureIters=parseInt(kSlider.value,10);
				if(kVal)kVal.textContent=cfg.pressureIters;
			});
		}
		const gridSlider=document.getElementById("ctrl-grid");
		const gridVal=document.getElementById("ctrl-grid-val");
		if(gridSlider){
			trackSlider(gridSlider);
			gridSlider.value=cfg.N;
			if(gridVal)gridVal.textContent=cfg.N;
			// the readout follows the thumb, the rebuild waits for release: a resize
			// reallocates every texture and rebuilds every bind group, so doing it per
			// input event stalls the page while the pointer moves
			gridSlider.addEventListener("input",()=>{
				if(gridVal)gridVal.textContent=gridSlider.value;
			});
			gridSlider.addEventListener("change",()=>{
				const n=parseInt(gridSlider.value,10);
				if(gridVal)gridVal.textContent=n;
				if(n===cfg.N||!solver)return;
				solver.resize(n);
				cfg._dirty=true;
			});
		}
		const fixedSlider=document.getElementById("ctrl-substeps-fixed");
		const fixedVal=document.getElementById("ctrl-substeps-fixed-val");
		if(fixedSlider){
			trackSlider(fixedSlider);
			fixedSlider.max=cfg.substepMax;
			fixedSlider.value=cfg.substepsFixed;
			if(fixedVal)fixedVal.textContent=cfg.substepsFixed;
			fixedSlider.addEventListener("input",()=>{
				cfg.substepsFixed=parseInt(fixedSlider.value,10);
				if(fixedVal)fixedVal.textContent=cfg.substepsFixed;
			});
		}
		let lastCtrlUpdate=0;
		function update(now){
			if(now-lastCtrlUpdate<200)return;
			lastCtrlUpdate=now;
			if(pauseBtn)pauseBtn.textContent=cfg._paused?"Resume":"Pause";
			if(fctChk)fctChk.checked=cfg.fctLimiter>0;
			if(gravChk)gravChk.checked=cfg.gravityY!==0;
			if(subChk)subChk.checked=!!cfg.substepsAuto;
			if(buoyChk)buoyChk.checked=cfg.buoyEnabled>0;
			syncSlider(kSlider,kVal,cfg.pressureIters);
			syncSlider(gridSlider,gridVal,cfg.N);
			syncSlider(fixedSlider,fixedVal,cfg.substepsFixed);
			if(scenSel)scenSel.value=cfg.scenario;
			updatePaintHighlight();
			updateColorHighlight();
		}
		return {update};
	}
	Fluid.app.controls={create:createControls};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.controls;
})(globalThis);
