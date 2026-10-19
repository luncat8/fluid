(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	const GROUP="view-mode";
	// one media query for the page: hover switching is for a real pointing device,
	// touch and stylus-only sessions stay click-only. reading .matches in the
	// listener keeps a hybrid device correct if its primary pointer changes.
	const fineHover=typeof root.matchMedia==="function"?root.matchMedia("(hover: hover) and (pointer: fine)"):null;
	function createViewMenu(cfg){
		const container=document.getElementById("view-menu");
		if(!container)return {setMode(){}};
		const names=Fluid.core.config.viewModeNames;
		const inputs=[];
		for(let i=0;i<names.length;i++){
			const label=document.createElement("label");
			const input=document.createElement("input");
			input.type="radio";
			input.name=GROUP;
			input.value=String(i);
			input.checked=i===cfg.viewMode;
			label.appendChild(input);
			label.appendChild(document.createTextNode(names[i]));
			container.appendChild(label);
			inputs.push(input);
		}
		// the single writer: cfg.viewMode is the source of truth and every path
		// (click, keyboard, hover) ends here. the early return also makes repeated
		// pointerover events on the same label free.
		function setMode(id){
			const mode=parseInt(id,10);
			if(mode===cfg.viewMode)return;
			if(!(mode>=0&&mode<names.length))return;
			cfg.viewMode=mode;
			cfg._dirty=true;
			inputs[mode].checked=true;
		}
		// native radio semantics: click, space and arrow keys all arrive as change
		container.addEventListener("change",e=>{
			const t=e.target;
			if(!t||t.tagName!=="INPUT"||t.type!=="radio"||t.name!==GROUP)return;
			setMode(t.value);
		});
		// hover: pointerover bubbles from the radio or its label, so one listener
		// covers the row. the walk allocates nothing per event.
		container.addEventListener("pointerover",e=>{
			if(!fineHover||!fineHover.matches)return;
			let node=e.target;
			while(node&&node!==container){
				if(node.tagName==="LABEL"){
					const input=node.control;
					if(input&&input.name===GROUP){
						input.checked=true;
						setMode(input.value);
					}
					return;
				}
				node=node.parentNode;
			}
		});
		return {setMode};
	}
	Fluid.app.viewMenu={create:createViewMenu,GROUP};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.viewMenu;
})(globalThis);
