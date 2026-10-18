(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	function createInput(canvas,cfg,solver){
		const state={
			x:0,y:0,dx:0,dy:0,down:false,
			keys:{},
			paintMode:1,
			paintColor:0,
			help:false
		};
		const rect={l:0,t:0,w:1,h:1};
		function updateRect(){
			let r=canvas.getBoundingClientRect();
			rect.l=r.left; rect.t=r.top; rect.w=r.width; rect.h=r.height;
		}
		canvas.addEventListener("pointerdown",e=>{
			updateRect();
			state.down=true;
			canvas.setPointerCapture(e.pointerId);
			let u=(e.clientX-rect.l)/rect.w;
			let v=1-(e.clientY-rect.t)/rect.h;
			state.dx=0; state.dy=0;
			state.x=u; state.y=v;
			cfg.mouseX=u*cfg.N; cfg.mouseY=v*cfg.N; cfg.mouseStrength=1;
		});
		canvas.addEventListener("pointermove",e=>{
			if(!state.down) return;
			updateRect();
			let u=(e.clientX-rect.l)/rect.w;
			let v=1-(e.clientY-rect.t)/rect.h;
			let nu=u, nv=v;
			state.dx=nu-state.x; state.dy=nv-state.y;
			state.x=nu; state.y=nv;
			cfg.mouseX=nu*cfg.N; cfg.mouseY=nv*cfg.N;
			// paint dye on drag: handled in main loop via solver.paintAt if available
			if(solver&&solver.paintAt){
				let colors=[[0.8,0,0],[0,0.8,0],[0,0,0.8],[0.6,0.6,0]];
				let c=colors[state.paintColor%colors.length];
				solver.paintAt(nu,nv, cfg.N*0.04, c[0],c[1],c[2],0.5);
			}
		});
		canvas.addEventListener("pointerup",e=>{
			state.down=false;
			state.dx=0; state.dy=0;
			cfg.mouseStrength=0;
		});
		canvas.addEventListener("pointerleave",e=>{
			state.down=false;
			cfg.mouseStrength=0;
		});
		window.addEventListener("keydown",e=>{
			let k=e.key.toLowerCase();
			state.keys[k]=true;
			if(k==="0"||k==="1"||k==="2"||k==="3"||k==="4"||k==="5"){
				let m=parseInt(k,10);
				cfg.viewMode=m;
			}else if(k===" "){
				cfg._paused=!cfg._paused;
				e.preventDefault();
			}else if(k==="." ){
				cfg._stepOnce=true;
			}else if(k==="r"){
				if(solver) solver.reset();
			}else if(k==="h"){
				state.help=!state.help;
				let el=document.getElementById("help-overlay");
				if(el) el.hidden=!state.help;
			}else if(k==="g"){
				if(cfg.gravityY===0) cfg.gravityY=300; else cfg.gravityY=0;
			}else if(k==="s"){
				console.log("stats dump", solver? solver.getStats():null);
			}else if(k==="+"||k==="="){
				cfg.jacobiIters=Math.min(200,cfg.jacobiIters+5);
			}else if(k==="-"||k==="_"){
				cfg.jacobiIters=Math.max(1,cfg.jacobiIters-5);
			}else if(k==="[" ){
				let n=Math.max(64,cfg.N-64);
				if(solver) solver.resize(n);
			}else if(k==="]" ){
				let n=Math.min(1024,cfg.N+64);
				if(solver) solver.resize(n);
			}else if(k==="c"){
				state.paintColor=(state.paintColor+1)%4;
			}
		});
		window.addEventListener("keyup",e=>{
			let k=e.key.toLowerCase();
			state.keys[k]=false;
		});
		return state;
	}
	Fluid.app.input={create:createInput};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.input;
})(globalThis);
