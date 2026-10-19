(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	const DYE_COLORS=[[1,0,0],[0,1,0],[0,0,1],[1,1,0]];
	function createInput(canvas,cfg,solver,setMode){
		const state={
			x:0,y:0,dx:0,dy:0,down:false,
			lastX:0,lastY:0,lastT:0,
			keys:{},
			paintColor:0,
			help:false,
			pending:null
		};
		const rect={l:0,t:0,w:1,h:1};
		function updateRect(){
			const r=canvas.getBoundingClientRect();
			rect.l=r.left;rect.t=r.top;rect.w=r.width;rect.h=r.height;
		}
		function toUV(e){
			updateRect();
			const u=(e.clientX-rect.l)/rect.w;
			const v=1-(e.clientY-rect.t)/rect.h;
			return [u<0?0:(u>1?1:u),v<0?0:(v>1?1:v)];
		}
		function queuePaint(u,v){
			// preallocated request object, flushed once per frame by main
			if(!state.pending)state.pending={u:0,v:0,wall:false,erase:false};
			state.pending.u=u;
			state.pending.v=v;
			state.pending.wall=cfg.paintMode!==0;
			state.pending.erase=cfg.paintMode===2;
		}
		cfg.paintColor=cfg.paintColor||0;
		function flushPaint(){
			const p=state.pending;
			if(!p||!solver)return;
			if(p.wall){
				solver.paintWall(p.u,p.v,p.erase);
			}else{
				const c=DYE_COLORS[(cfg.paintColor||0)%DYE_COLORS.length];
				solver.paintDye(p.u,p.v,c);
			}
			state.pending=null;
			cfg._dirty=true;
		}
		canvas.addEventListener("pointerdown",e=>{
			canvas.setPointerCapture(e.pointerId);
			state.down=true;
			const [u,v]=toUV(e);
			state.x=u;state.y=v;state.lastX=u;state.lastY=v;state.lastT=performance.now();
			state.dx=0;state.dy=0;
			cfg.mouseX=u*cfg.N;cfg.mouseY=v*cfg.N;
			if(cfg.paintMode===0)cfg.mouseStrength=1;
			queuePaint(u,v);
			e.preventDefault();
		});
		canvas.addEventListener("pointermove",e=>{
			if(!state.down)return;
			const [u,v]=toUV(e);
			const now=performance.now();
			const dtm=Math.max((now-state.lastT)/1000,1e-3);
			// pointer velocity in cells/s, smoothed; clamped so a fast flick cannot inject huge momentum
			let vx=(u-state.lastX)*cfg.N/dtm;
			let vy=(v-state.lastY)*cfg.N/dtm;
			const sp=Math.sqrt(vx*vx+vy*vy);
			if(sp>600){vx*=600/sp;vy*=600/sp;}
			cfg.mouseVX=cfg.mouseVX*0.5+vx*0.5;
			cfg.mouseVY=cfg.mouseVY*0.5+vy*0.5;
			state.dx=u-state.x;state.dy=v-state.y;
			state.x=u;state.y=v;
			state.lastX=u;state.lastY=v;state.lastT=now;
			cfg.mouseX=u*cfg.N;cfg.mouseY=v*cfg.N;
			queuePaint(u,v);
		});
		function pointerEnd(){
			state.down=false;
			state.dx=0;state.dy=0;
			cfg.mouseStrength=0;
			cfg.mouseVX=0;cfg.mouseVY=0;
		}
		canvas.addEventListener("pointerup",pointerEnd);
		canvas.addEventListener("pointercancel",pointerEnd);
		canvas.addEventListener("pointerleave",pointerEnd);
		window.addEventListener("keydown",e=>{
			const k=e.key.toLowerCase();
			state.keys[k]=true;
			if(k>="0"&&k<="7"&&k.length===1){
				setMode(parseInt(k,10));
			}else if(k===" "){
				cfg._paused=!cfg._paused;
				e.preventDefault();
			}else if(k==="."){
				cfg._stepOnce=true;
			}else if(k==="r"){
				if(solver)solver.reset();
				cfg._dirty=true;
			}else if(k==="h"){
				state.help=!state.help;
				const el=document.getElementById("help-overlay");
				if(el)el.hidden=!state.help;
			}else if(k==="g"){
				cfg.gravityY=cfg.gravityY===0?Fluid.core.config.defaults.gravityY:0;
			}else if(k==="s"){
				console.log("stats dump",solver?solver.getStats():null);
			}else if(k==="+"||k==="="){
				cfg.pressureIters=Math.min(200,cfg.pressureIters+5);
			}else if(k==="-"||k==="_"){
				cfg.pressureIters=Math.max(1,cfg.pressureIters-5);
			}else if(k==="["){
				const n=Math.max(64,cfg.N-64);
				if(solver)solver.resize(n);
				cfg._dirty=true;
			}else if(k==="]"){
				const n=Math.min(1024,cfg.N+64);
				if(solver)solver.resize(n);
				cfg._dirty=true;
				}else if(k==="f"){
					cfg.fctLimiter=cfg.fctLimiter?0:1;
					cfg._dirty=true;
			}else if(k==="c"){
				cfg.paintColor=((cfg.paintColor||0)+1)%DYE_COLORS.length;
			}else if(k==="b"){
				cfg.paintMode=(cfg.paintMode+1)%3;
			}else if(k==="n"){
				const names=Fluid.core.config.scenarioNames;
				cfg.scenario=names[(names.indexOf(cfg.scenario)+1)%names.length];
				if(solver)solver.reset();
				cfg._dirty=true;
			}else if(k==="u"){
				cfg.substepsAuto=!cfg.substepsAuto;
			}else if(k==="y"){
				cfg.buoyEnabled=cfg.buoyEnabled>0?0:1;
				if(solver&&solver.buoy)solver.buoy.reset();
				cfg._dirty=true;
			}
		});
		window.addEventListener("keyup",e=>{
			state.keys[e.key.toLowerCase()]=false;
		});
		return {state,flushPaint};
	}
	Fluid.app.input={create:createInput,DYE_COLORS};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.input;
})(globalThis);
