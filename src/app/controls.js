(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	function createControls(cfg,solver){
		const panel=document.getElementById("controls-panel");
		if(!panel)return {update(){}};
		// view mode
		const viewSel=document.getElementById("ctrl-view");
		if(viewSel){
			const names=Fluid.core.config.viewModeNames;
			for(let i=0;i<names.length;i++){
				const o=document.createElement("option");
				o.value=i;o.textContent=names[i];
				viewSel.appendChild(o);
			}
			viewSel.value=cfg.viewMode;
			viewSel.addEventListener("change",()=>{cfg.viewMode=parseInt(viewSel.value,10);cfg._dirty=true;});
		}
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
		// sliders: pressure iters, grid size
		const kSlider=document.getElementById("ctrl-iters");
		const kVal=document.getElementById("ctrl-iters-val");
		if(kSlider){
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
			gridSlider.value=cfg.N;
			if(gridVal)gridVal.textContent=cfg.N;
			gridSlider.addEventListener("change",()=>{
				const n=parseInt(gridSlider.value,10);
				if(n!==cfg.N&&solver){
					solver.resize(n);
					cfg._dirty=true;
					if(gridVal)gridVal.textContent=n;
				}
			});
		}
		let lastCtrlUpdate=0;
		function update(now){
			if(now-lastCtrlUpdate<200)return;
			lastCtrlUpdate=now;
			if(viewSel)viewSel.value=cfg.viewMode;
			if(pauseBtn)pauseBtn.textContent=cfg._paused?"Resume":"Pause";
			if(fctChk)fctChk.checked=cfg.fctLimiter>0;
			if(gravChk)gravChk.checked=cfg.gravityY!==0;
			if(kSlider)kSlider.value=cfg.pressureIters;
			if(kVal)kVal.textContent=cfg.pressureIters;
			if(gridSlider)gridSlider.value=cfg.N;
			if(gridVal)gridVal.textContent=cfg.N;
			if(scenSel)scenSel.value=cfg.scenario;
			updatePaintHighlight();
			updateColorHighlight();
		}
		return {update};
	}
	Fluid.app.controls={create:createControls};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.controls;
})(globalThis);
