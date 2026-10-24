(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.engine=Fluid.engine||{};
	const config=Fluid.core.config;
	// key tables for stats(): index loops over arrays, no iterators, no literals per call
	const K_TOTALS=["m","dr","dg","db","E"];
	const K_MOTION=["maxVel","pMin","pMax","maxDiv"];
	const K_THERMAL=["TMin","TMax","TMean","kinetic","heatIn","heatOut"];
	const K_DIAG=["sSaturated","poissonRes","poissonRhs","poissonRel","poissonDrift","poissonDriftRhs","poissonDriftRel"];
	const GRID_MIN=64,GRID_MAX=1024;
	function makeStats(){
		return {
			frame:0,statsFrame:0,
			totals:{m:0,dr:0,dg:0,db:0,E:0},
			expected:{m:0,dr:0,dg:0,db:0,E:0},
			mins:{m:0,dr:0,dg:0,db:0,E:0},
			maxM:0,cntNeg:0,cntOver:0,
			motion:{maxVel:0,pMin:0,pMax:0,maxDiv:0},
			thermal:{TMin:0,TMax:0,TMean:0,kinetic:0,heatIn:0,heatOut:0},
			phaseCounts:[0,0,0,0,0,0],
			diag:{sSaturated:0,poissonRes:0,poissonRhs:0,poissonRel:0,poissonDrift:0,poissonDriftRhs:0,poissonDriftRel:0},
			ledger:null
		};
	}
	// fills a preallocated out object in place: a game polling stats at frame rate
	// must not create garbage, and the solver's own view is mutated by later readbacks.
	function copyStats(out,st){
		if(!st)return out;
		out.frame=st.frame||0;
		out.statsFrame=st.statsFrame||0;
		const t=st.totals,e=st.expected,mi=st.mins;
		for(let i=0;i<K_TOTALS.length;i++){
			const k=K_TOTALS[i];
			out.totals[k]=(t&&t[k])||0;
			out.expected[k]=(e&&e[k])||0;
			out.mins[k]=(mi&&mi[k])||0;
		}
		out.maxM=st.maxM||0;
		out.cntNeg=st.cntNeg||0;
		out.cntOver=st.cntOver||0;
		const mo=st.motion;
		for(let i=0;i<K_MOTION.length;i++)out.motion[K_MOTION[i]]=(mo&&mo[K_MOTION[i]])||0;
		const th=st.thermal;
		for(let i=0;i<K_THERMAL.length;i++)out.thermal[K_THERMAL[i]]=(th&&th[K_THERMAL[i]])||0;
		const dg=st.diag;
		for(let i=0;i<K_DIAG.length;i++)out.diag[K_DIAG[i]]=(dg&&dg[K_DIAG[i]])||0;
		const pc=st.phaseCounts;
		for(let i=0;i<6;i++)out.phaseCounts[i]=(pc&&pc[i])||0;
		out.ledger=st.ledger||null;
		return out;
	}
	function now(){
		return typeof performance!=="undefined"&&performance.now?performance.now():Date.now();
	}
	// the game-facing surface (0.7.0). create() is the only promise: device setup.
	// update() owns pacing (one frame in flight, fixed frame dt), step() advances
	// exactly one frame, and every query writes into caller-owned objects.
	async function create(canvas,options){
		const opt=options||{};
		let device=opt.device,context=opt.context,format=opt.format;
		if(!device){
			const init=await Fluid.gpu.device.init(canvas,opt.errorPanel);
			if(!init)return null;
			device=init.device;context=init.context;format=init.format;
		}
		const cfg=Object.assign({},config.defaults,opt.cfg||{});
		cfg.N=Math.max(GRID_MIN,Math.min(GRID_MAX,Math.round(cfg.N||512)));
		cfg.texW=cfg.N+2;
		cfg.texH=cfg.N+2;
		cfg._paused=!!cfg._paused;
		cfg._stepOnce=false;
		cfg._dirty=true;
		let solver=Fluid.gpu.solver.create(device,canvas,context,format,cfg);
		// one canvas texel per cell; the display texture is N-sized and the browser
		// scales it to the css box
		canvas.width=cfg.N;
		canvas.height=cfg.N;
		const pacing=Fluid.core.pacing.create(device.queue);
		const statsScratch=makeStats();
		const probeSample=Fluid.core.probes.makeSample();
		let accumulator=0;
		let destroyed=false;
		let inputState=null;
		let viewWriter=null;
		let visibilityHandler=null;
		function active(){
			return !destroyed&&solver!==null;
		}
		function pollReadback(){
			if(!active())return null;
			const p=solver.pollReadback();
			// the solver guards every map with pending/in-flight flags; a rejected
			// readback is already reported there, this only keeps node quiet.
			if(p&&p.catch)p.catch(()=>{});
			return p;
		}
		function render(){
			if(!active()||pacing.state.busy)return false;
			const t0=now();
			solver.renderOnly();
			pacing.submitted(0,now()-t0);
			cfg._dirty=false;
			return true;
		}
		// one fixed frame: the substep count comes from the shared policy, dt from
		// the caller (or the configured frame dt). step() never renders.
		function step(dt){
			if(!active())return 0;
			const frameDt=dt>0?dt:cfg.frameDt;
			cfg.frameDt=frameDt;
			const st=solver.getStats();
			// the auto policy reads the last readback plus the live drag speed, which
			// is known now and must not wait for the readback (draw blowup trigger)
			const maxVel=Math.max(st.motion.maxVel,Math.abs(cfg.mouseVX),Math.abs(cfg.mouseVY));
			const ns=config.substepCount(cfg,maxVel);
			cfg.vClamp=config.velocityCap(cfg,ns);
			const t0=now();
			if(inputState)inputState.flushPaint();
			const needStats=st.frame%cfg.statsEvery===0;
			solver.frame(frameDt/ns,ns,needStats);
			pacing.submitted(frameDt,now()-t0);
			cfg._stepOnce=false;
			cfg._dirty=false;
			sim.substeps=ns;
			return ns;
		}
		// the game loop entry: clamp the wall delta, accumulate, take at most one
		// fixed step per call and one submission in flight, then redraw if asked.
		// returns the substep count of the step it took, 0 when it only rendered
		// or did nothing.
		function update(dtReal){
			if(!active())return 0;
			let dt=dtReal;
			if(!(dt>0))dt=1e-6;
			if(dt>0.1)dt=0.1;
			pacing.tick(dt);
			let ns=0;
			if(!cfg._paused||cfg._stepOnce){
				// keep fractional ticks at high refresh rates, drop long GPU backlog:
				// two frame periods of debt is the most that can be repaid in order
				accumulator=Math.min(accumulator+dt,cfg.frameDt*2);
				if(cfg._stepOnce)accumulator=cfg.frameDt;
				if(accumulator>=cfg.frameDt&&!pacing.state.busy){
					accumulator-=cfg.frameDt;
					ns=step(cfg.frameDt);
				}
			}else{
				accumulator=0;
			}
			if(ns===0&&cfg._dirty)render();
			pollReadback();
			return ns;
		}
		function reset(scenario){
			if(!active())return false;
			if(typeof scenario==="string"&&scenario){
				if(config.scenarioNames.indexOf(scenario)<0)return false;
				cfg.scenario=scenario;
			}
			solver.reset();
			accumulator=0;
			cfg._dirty=true;
			return true;
		}
		function setGrid(size){
			if(!active())return 0;
			const n=Math.max(GRID_MIN,Math.min(GRID_MAX,Math.round(size)||cfg.N));
			if(n===cfg.N)return cfg.N;
			// resize repaints the new grid in the same task; mark that submission so
			// the next update() waits for it like any other frame
			solver.resize(n);
			pacing.submitted(0,0);
			accumulator=0;
			return cfg.N;
		}
		function setViewMode(mode){
			const m=parseInt(mode,10);
			if(!(m>=0&&m<config.viewModeNames.length))return false;
			// one writer per app: when the page owns a view control, that control
			// writes cfg.viewMode and this only asks it to.
			if(viewWriter){
				viewWriter(m);
				return true;
			}
			cfg.viewMode=m;
			cfg._dirty=true;
			return true;
		}
		function setOverlay(on){
			cfg.overlay=on?1:0;
			cfg._dirty=true;
			return cfg.overlay;
		}
		function addProbe(x,y){
			if(!active())return -1;
			return solver.addProbe(x,y);
		}
		function moveProbe(id,x,y){
			return active()?solver.moveProbe(id,x,y):false;
		}
		function removeProbe(id){
			return active()?solver.removeProbe(id):false;
		}
		function sampleProbe(id,out){
			const dst=out||probeSample;
			if(!active())return false;
			return solver.sampleProbe(id,dst);
		}
		// sources (0.7.0 t3): the spec object is copied into a preallocated slot at add
		// time, so a caller may reuse its own object every call and nothing allocates
		function addEmitter(spec){
			return active()?solver.addEmitter(spec):-1;
		}
		function moveEmitter(id,x,y){
			return active()?solver.moveEmitter(id,x,y):false;
		}
		function setEmitter(id,patch){
			return active()?solver.setEmitter(id,patch):false;
		}
		function removeEmitter(id){
			return active()?solver.removeEmitter(id):false;
		}
		// geometry (0.7.0 t4): static masks and moving bodies. a body's mask is applied
		// on the next step, because that is the frame the fluid is pushed in; the call
		// itself only writes the game's intent into the table.
		function setObstacles(mask,w,h){
			return active()?solver.setObstacles(mask,w,h):false;
		}
		function paintObstacle(x,y,radius,kind){
			return active()?solver.paintObstacle(x,y,radius,kind):false;
		}
		function addCollider(spec){
			return active()?solver.addCollider(spec):-1;
		}
		function moveCollider(id,x,y,vx,vy){
			return active()?solver.moveCollider(id,x,y,vx,vy):false;
		}
		function setCollider(id,patch){
			return active()?solver.setCollider(id,patch):false;
		}
		function removeCollider(id){
			return active()?solver.removeCollider(id):false;
		}
		function stats(out){
			const dst=out||statsScratch;
			if(active())copyStats(dst,solver.getStats());
			return dst;
		}
		function attachInput(element){
			// the input helper is a page module: an embedder that does not load
			// src/app/input.js keeps the engine, it just has no DOM binding
			if(!active()||!Fluid.app||!Fluid.app.input)return null;
			detachInput();
			inputState=Fluid.app.input.create(element||canvas,cfg,solver,setViewMode);
			return inputState;
		}
		function detachInput(){
			if(inputState&&inputState.destroy)inputState.destroy();
			inputState=null;
		}
		function destroy(){
			if(destroyed)return;
			destroyed=true;
			detachInput();
			if(visibilityHandler&&typeof document!=="undefined")document.removeEventListener("visibilitychange",visibilityHandler);
			visibilityHandler=null;
			if(solver)solver.destroy();
			solver=null;
			sim.solver=null;
			sim.device=null;
			sim.context=null;
			sim.destroyed=true;
		}
		const sim={
			canvas,device,context,format,
			params:cfg,
			solver,
			perf:pacing.state,
			destroyed:false,
			substeps:0,
			createStats:makeStats,
			makeProbeSample:Fluid.core.probes.makeSample,
			step,update,render,reset,setGrid,setViewMode,setOverlay,
			bindViewMode(fn){viewWriter=typeof fn==="function"?fn:null;return !!viewWriter;},
			addProbe,moveProbe,removeProbe,sample:sampleProbe,stats,attachInput,detachInput,pollReadback,
			addEmitter,moveEmitter,setEmitter,removeEmitter,
			setObstacles,paintObstacle,addCollider,moveCollider,setCollider,removeCollider,
			probeCount(){return active()?solver.probes.live:0;},
			emitterCount(){return active()?solver.emitters.live:0;},
			colliderCount(){return active()?solver.colliders.live:0;},
			singleStep(){cfg._stepOnce=true;},
			pause(){cfg._paused=true;},
			resume(){cfg._paused=false;},
			destroy
		};
		Object.defineProperty(sim,"paused",{
			get(){return !!cfg._paused;},
			set(v){cfg._paused=!!v;}
		});
		if(typeof document!=="undefined"&&document.addEventListener){
			visibilityHandler=()=>{
				// a hidden tab stops receiving rAF; drop the debt so the first frame
				// after it returns does not try to catch up
				if(document.hidden)accumulator=0;
			};
			document.addEventListener("visibilitychange",visibilityHandler);
		}
		return sim;
	}
	Fluid.engine={create,copyStats,makeStats,makeProbeSample:Fluid.core.probes.makeSample};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.engine;
})(globalThis);
