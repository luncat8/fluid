(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	const DYE_COLORS=[[1,0,0],[0,1,0],[0,0,1],[1,1,0]];
	function createInput(canvas,cfg,solver,setMode,options){
		const opt=options||{};
		const pointerEnabled=opt.pointer!==false;
		const keysEnabled=opt.keys!==false;
		const win=typeof window!=="undefined"&&window&&window.addEventListener?window:null;
		const keyTarget=opt.keyTarget||win;
		const doReset=opt.onReset||(()=>{
			if(solver)solver.reset();
			cfg._dirty=true;
		});
		const doResize=opt.onResize||(n=>{
			if(solver&&n!==cfg.N)solver.resize(n);
			cfg._dirty=true;
		});
		const state={
			x:0,y:0,dx:0,dy:0,down:false,
			lastX:0,lastY:0,lastT:0,
			pointerId:null,
			keys:{},
			paintColor:0,
			help:false,
			pending:null
		};
		const paintRequest={u:0,v:0,wall:false,erase:false};
		const rect={l:0,t:0,w:1,h:1};
		function updateRect(){
			const r=canvas.getBoundingClientRect();
			rect.l=r.left;rect.t=r.top;rect.w=r.width;rect.h=r.height;
		}
		function updateUV(e){
			updateRect();
			const u=(e.clientX-rect.l)/rect.w;
			const v=1-(e.clientY-rect.t)/rect.h;
			state.x=u<0?0:(u>1?1:u);
			state.y=v<0?0:(v>1?1:v);
		}
		function queuePaint(u,v){
			// preallocated request object, flushed once per frame by main
			state.pending=paintRequest;
			state.pending.u=u;
			state.pending.v=v;
			state.pending.wall=cfg.paintMode!==0;
			state.pending.erase=cfg.paintMode===2;
		}
		// pointer and keyboard listeners have separate ownership so games can detach
		// one without disturbing the other.
		const pointerBindings=[];
		const keyBindings=[];
		function on(target,type,fn,bindings){
			if(!target||typeof target.addEventListener!=="function")return;
			target.addEventListener(type,fn);
			bindings.push([target,type,fn]);
		}
		function removeBindings(bindings){
			for(let i=0;i<bindings.length;i++){
				const b=bindings[i];
				if(typeof b[0].removeEventListener==="function")b[0].removeEventListener(b[1],b[2]);
			}
			bindings.length=0;
		}
		function destroyKeys(){
			removeBindings(keyBindings);
			for(const k in state.keys)state.keys[k]=false;
		}
		function destroy(){
			removeBindings(pointerBindings);
			destroyKeys();
			state.pending=null;
			state.down=false;
			state.pointerId=null;
			if(pointerEnabled){
				cfg.mouseStrength=0;
				cfg.mouseVX=0;cfg.mouseVY=0;
			}
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
		if(pointerEnabled){
		on(canvas,"pointerdown",e=>{
			if(state.down||e.button!==0)return;
			canvas.setPointerCapture(e.pointerId);
			canvas.focus();
			state.pointerId=e.pointerId;
			state.down=true;
			updateUV(e);
			const u=state.x,v=state.y;
			state.lastX=u;state.lastY=v;state.lastT=performance.now();
			state.dx=0;state.dy=0;
			cfg.mouseX=u*cfg.N;cfg.mouseY=v*cfg.N;
			if(cfg.paintMode===0)cfg.mouseStrength=1;
			queuePaint(u,v);
			e.preventDefault();
		},pointerBindings);
		on(canvas,"pointermove",e=>{
			if(!state.down||e.pointerId!==state.pointerId)return;
			const oldX=state.x,oldY=state.y;
			updateUV(e);
			const u=state.x,v=state.y;
			const now=performance.now();
			const dtm=Math.max((now-state.lastT)/1000,1e-3);
			// pointer velocity in cells/s, smoothed; clamped so a fast flick cannot inject huge momentum
			let vx=(u-state.lastX)*cfg.N/dtm;
			let vy=(v-state.lastY)*cfg.N/dtm;
			const sp=Math.sqrt(vx*vx+vy*vy);
			if(sp>600){vx*=600/sp;vy*=600/sp;}
			cfg.mouseVX=cfg.mouseVX*0.5+vx*0.5;
			cfg.mouseVY=cfg.mouseVY*0.5+vy*0.5;
			state.dx=u-oldX;state.dy=v-oldY;
			state.lastX=u;state.lastY=v;state.lastT=now;
			cfg.mouseX=u*cfg.N;cfg.mouseY=v*cfg.N;
			queuePaint(u,v);
		},pointerBindings);
		function pointerEnd(e){
			if(e&&e.pointerId!==state.pointerId)return;
			if(e&&e.type!=="pointerup")state.pending=null;
			const id=state.pointerId;
			state.pointerId=null;
			state.down=false;
			state.dx=0;state.dy=0;
			cfg.mouseStrength=0;
			cfg.mouseVX=0;cfg.mouseVY=0;
			if(id!==null&&canvas.hasPointerCapture(id))canvas.releasePointerCapture(id);
		}
			on(canvas,"pointerup",pointerEnd,pointerBindings);
			on(canvas,"pointercancel",pointerEnd,pointerBindings);
			on(canvas,"lostpointercapture",pointerEnd,pointerBindings);
			if(win)on(win,"blur",()=>{
				pointerEnd();
				state.pending=null;
			},pointerBindings);
		}
		if(keysEnabled&&keyTarget){
			const clearKeys=()=>{
				for(const k in state.keys)state.keys[k]=false;
			};
			if(win)on(win,"blur",clearKeys,keyBindings);
			if(keyTarget!==win)on(keyTarget,"blur",clearKeys,keyBindings);
			on(keyTarget,"keydown",e=>{
			if(e.defaultPrevented||e.ctrlKey||e.metaKey||e.altKey||e.isComposing)return;
			const target=e.target;
			if(target&&(target.isContentEditable||(typeof target.closest==="function"&&target.closest("input, select, textarea, button, a[href]"))))return;
			const k=e.key.toLowerCase();
			if(e.repeat&&k!=="+"&&k!=="="&&k!=="-"&&k!=="_")return;
			state.keys[k]=true;
			if(k>="0"&&k<="7"&&k.length===1){
				setMode(parseInt(k,10));
			}else if(k===" "){
				cfg._paused=!cfg._paused;
				e.preventDefault();
			}else if(k==="."){
				cfg._stepOnce=true;
			}else if(k==="r"){
				doReset();
			}else if(k==="h"){
				state.help=!state.help;
				const el=document.getElementById("help-overlay");
				if(el)el.hidden=!state.help;
			}else if(k==="g"){
				cfg.gravityY=cfg.gravityY===0?Fluid.core.config.defaults.gravityY:0;
			}else if(k==="s"){
				console.log("stats dump",solver?solver.getStats():null);
			}else if(k==="+"||k==="="){
				Fluid.core.config.setSolverIters(cfg,Math.min(200,Fluid.core.config.solverIters(cfg)+5));
			}else if(k==="-"||k==="_"){
				Fluid.core.config.setSolverIters(cfg,Math.max(1,Fluid.core.config.solverIters(cfg)-5));
			}else if(k==="["){
				doResize(Math.max(64,cfg.N-64));
			}else if(k==="]"){
				doResize(Math.min(1024,cfg.N+64));
			}else if(k==="f"){
				cfg.fctLimiter=cfg.fctLimiter?0:1;
				cfg._dirty=true;
			}else if(k==="o"){
				cfg.overlay=cfg.overlay>0?0:1;
				cfg._dirty=true;
			}else if(k==="c"){
				cfg.paintColor=((cfg.paintColor||0)+1)%DYE_COLORS.length;
			}else if(k==="b"){
				cfg.paintMode=(cfg.paintMode+1)%3;
			}else if(k==="n"){
				const names=Fluid.core.config.scenarioNames;
				cfg.scenario=names[(names.indexOf(cfg.scenario)+1)%names.length];
				doReset();
				// convection is a temperature show: land on the map that shows it
				if(cfg.scenario==="convection"&&setMode)setMode(Fluid.core.config.viewModes.temperature);
			}else if(k==="u"){
				cfg.substepsAuto=!cfg.substepsAuto;
			}else if(k==="y"){
				cfg.buoyEnabled=cfg.buoyEnabled>0?0:1;
				if(solver&&solver.buoy)solver.buoy.reset();
				cfg._dirty=true;
			}
			},keyBindings);
			on(keyTarget,"keyup",e=>{
				state.keys[e.key.toLowerCase()]=false;
			},keyBindings);
		}
		return {state,flushPaint,destroy,destroyKeys};
	}
	function createKeys(element,cfg,solver,setMode,options){
		const opt=options||{};
		return createInput(null,cfg,solver,setMode,{pointer:false,keyTarget:element,onReset:opt.onReset,onResize:opt.onResize});
	}
	Fluid.app.input={create:createInput,createKeys,DYE_COLORS};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.input;
})(globalThis);
