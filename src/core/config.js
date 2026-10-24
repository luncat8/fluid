(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const paramsOff={
		N:0,invN:1,texW:2,texH:3,dt:4,substeps:5,time:6,frame:7,
		paintX:8,paintY:9,paintW:10,paintH:11,
		gravityX:16,gravityY:17,rhoRef:18,beta:19,TRef:20,mouseX:21,mouseY:22,mouseStrength:23,
		mouseVX:24,mouseVY:25,mouseR:26,rhoMean:27,
		pressureIters:32,fctLimiter:33,viscosity:34,kLiq:35,SMax:36,tauPhase:37,vClamp:38,
		heatPower:39,coolTop:40,wallCool:41,heatBand:42,thermalOn:43,heatTarget:44,
		kSol:45,kGas:46,TEps:47,
		cSol:48,cLiq:49,cGas:50,Lf:51,Lv:52,TMelt:53,TBoil:54,rhoLiq:55,rhoGas0:56,rhoVac:57,mEps:58,
		chargeLatent:59,TCeil:60,macLimiter:61,
		viewMode:80,TMin:81,TMax:82,PMin:83,PMax:84,VMax:85,overlay:86,colorGain:87,
		buoyX0:88,buoyY0:89,buoyW:90,buoyH:91,buoySub:92,buoyShow:93,
		mgS:94,rhoBase:95
	};
	const defaults={
		N:512,
		texW:514,
		texH:514,
		dt:1/60,
		frameDt:1/60,
		substeps:1,
		time:0,
		frame:0,
		gravityX:0,
		gravityY:300,
		rhoRef:1.0,
		beta:0.002,
		TRef:293,
		TAmb:293,
		mouseX:0,
		mouseY:0,
		mouseStrength:0,
		mouseVX:0,
		mouseVY:0,
		mouseR:12,
		// red-black sweeps on the fine grid when the multigrid solve is off
		// (the ?mg=0 fallback). the multigrid path reads mgCoarseIters instead:
		// a fine sweep costs 16x the cells of a coarse sweep, so one shared
		// number cannot mean the same thing on both paths.
		pressureIters:60,
		fctLimiter:1,
		viscosity:0.0,
		// phase dependent conduction (0.5.0): the liquid value is the 0.4.0
		// coefficient; gas nearly insulating keeps steam plumes hot so they rise far.
		kSol:0.25,
		kLiq:0.2,
		kGas:0.02,
		TEps:0.001,
		// thermal sources (0.4.0): fixed power per bottom-band cell, cold plate
		// relaxation per top-band cell, wall exchange coefficient. only applied
		// while cfg.thermalSources is on (thermal/phase scenarios turn it on).
		heatPower:30,
		coolTop:0.5,
		wallCool:0,
		heatBand:2,
		thermalSources:0,
		heatTarget:0,
		TCeil:0,
		convBotT:343,
		convTopT:293,
		SMax:30,
		tauPhase:0.05,
		vClamp:480,
		cSol:2.0,
		cLiq:4.2,
		cGas:1.0,
		Lf:334,
		Lv:2260,
		TMelt:273,
		TBoil:373,
		rhoLiq:1.0,
		rhoGas0:0.02,
		rhoVac:0.001,
		mEps:0.0001,
		chargeLatent:1,
		viewMode:0,
		TMin:273,
		TMax:373,
		PMin:-100,
		PMax:100,
		VMax:400,
		overlay:1,
		colorGain:1.2,
		statsEvery:30,
		// frames between probe-table readbacks (0.7.0 t5). the sample age is bounded
		// by this plus the in-flight readback; a game that needs fresher values lowers
		// it and pays for one more small mapAsync per cadence frame.
		probeEvery:4,
		gpuTiming:0,
		cflTarget:0.8,
		substepMax:24,
		substepsAuto:true,
		substepsFixed:4,
		vMax:8,
		// velocity advection scheme (0.6.0 t5/t16). 0 = semi-lagrangian, the
		// shipping path on both cpu and gpu. 1 = mac-cormack with the
		// stencil-bounds limiter; pinned ns=4 device cost is +18 us/substep
		// (<= 0.6 % frame time), stays off until the 4-scenario auto-policy row
		// confirms higher peak speeds do not push ns below the 60 fps gate.
		advectMac:0,
		macLimiter:1,
		// 0.6.0 t8 pass fusion on the gpu: 1 fuses advectMass+advectTherm into
		// advectScalar and heatSources into the second phaseEos (bit-identical,
		// two fewer full-grid dispatches per substep). off until its device row
		// clears the 3 % frame-time bar (plan 4.3).
		fusePasses:0,
		// 0.6.0 t6: vorticity confinement, F = eps*h*(N x omega) with N the unit
		// gradient of |omega|. eps is in 1/s, 0 is off. a force, not a change to
		// the advection scheme: it costs one extra pass over the velocity field.
		// cpu only for now (the gpu path does not read it), so it stays off until
		// the wgsl pass exists and the device row says it is worth a dispatch.
		vorticityConfinement:0,
		useMultigrid:true,
		mgPre:4,
		mgPost:8,
		mgCoarsen:4,
		// global default. the product gate (productCycleGate, set by the app and the
		// browser harness) drops jet to 1 and leaves every other scenario here.
		mgCycles:2,
		// coarse-level red-black sweeps per V-cycle. 0.6.0 t11: 60 -> 20 removes
		// 160 of the 295 dispatches per projection (295 -> 135) and 1.6-2.5 ->
		// 1.1-1.8 ms of gpu press time at N=512, while the divergence and the true
		// Poisson residual are unchanged on the pinned-dt oracle rows.
		mgCoarseIters:20,
		buoyEnabled:1,
		buoyRefW:16,
		buoyRefH:64,
		buoyU:0.3,
		buoyV:0.75,
		buoyRho:0.25,
		buoyGain:1,
		buoyDrag:0.12,
		buoySampleEvery:4,
		buoyW:0,
		buoyH:0,
		buoyX0:0,
		buoyY0:0,
		buoySub:0,
		buoyShow:0,
		paintAmount:0.35,
		paintR:10,
		scenario:"blobs",
		paintMode:0
	};
	const viewModes={
		color:0,
		temperature:1,
		pressure:2,
		velocity:3,
		divergence:4,
		flags:5,
		thermal:6,
		phase:7
	};
	const viewModeNames=["color","temperature","pressure","velocity","divergence","flags","thermal","phase"];
	const scenarioNames=["blobs","dam","still","convection","ice","cycle","jet","rain","arch"];
	const paintModeNames=["dye","wall","erase"];
	const keyMap={
		"0":"color","1":"temperature","2":"pressure","3":"velocity","4":"divergence","5":"flags",
		"6":"thermal","7":"phase",
		" ":"pause",".":"step","r":"reset","h":"help","s":"dumpStats","d":"dumpField","g":"gravity",
		"+":"iterUp","-":"iterDown","=":"iterUp","_":"iterDown","[":"sizeDown","]":"sizeUp",
		"c":"cycleColor","b":"cyclePaint","n":"nextScenario","f":"toggleFct",
		"u":"toggleSubsteps","y":"toggleBuoy","o":"toggleOverlay"
	};
	function substepCount(cfg,maxVel){
		if(!cfg.substepsAuto){
			let n=Math.round(cfg.substepsFixed)||1;
			if(n<1)n=1;
			if(n>cfg.substepMax)n=cfg.substepMax;
			return n;
		}
		let ns=Math.ceil(2*maxVel*cfg.frameDt/cfg.cflTarget);
		if(!(ns>=1))ns=1;
		if(ns>cfg.substepMax)ns=cfg.substepMax;
		return ns;
	}
	function velocityCap(cfg,ns){
		const cap=Math.min(cfg.vMax,0.45*Math.max(1,ns));
		return cap/cfg.frameDt;
	}
	function scaleBeta(N){
		const l2=Math.log2(Math.max(1,N/64));
		return defaults.beta*Math.min(1.8,1+0.18*l2);
	}
	// per-scenario presentation/physics setup, applied by both solvers on reset
	// so cpu and gpu cannot drift. resets scenario-driven keys back to defaults first
	// so switching scenarios never leaks thermal sources or view bounds.
	function prepareScenario(cfg){
		const s=cfg.scenario;
		const band=Math.max(2,Math.round(cfg.N/32));
		if(cfg._prevScenario&&cfg._prevScenario!==s){
			cfg.gravityY=defaults.gravityY;
			cfg.heatPower=defaults.heatPower;
			cfg.coolTop=defaults.coolTop;
			cfg.TCeil=0;
			cfg.TMin=defaults.TMin;
			cfg.TMax=defaults.TMax;
			cfg.thermalSources=0;
			cfg.heatBand=defaults.heatBand;
			cfg.heatTarget=0;
			cfg.beta=defaults.beta;
		}
		cfg._prevScenario=s;
		if(s==="convection"){
			cfg.thermalSources=1;
			cfg.heatBand=band;
			cfg.TMin=cfg.convTopT;
			cfg.TMax=cfg.convBotT;
			cfg.heatTarget=cfg.convBotT;
			cfg.beta=scaleBeta(cfg.N);
		} else if(s==="ice"){
			cfg.thermalSources=1;
			cfg.heatBand=band;
			cfg.heatPower=260;
			cfg.coolTop=0.5;
			cfg.heatTarget=355;
			cfg.TMin=245;
			cfg.TMax=355;
			cfg.beta=scaleBeta(cfg.N);
		} else if(s==="cycle"){
			// balanced cooker: box-scaled gravity keeps identical Froude number across N,
			// heater melts ice and crosses boiling plateau, cold ceiling condenses vapor back.
			cfg.gravityY=Math.max(24,defaults.gravityY*(cfg.N/512));
			cfg.thermalSources=1;
			cfg.heatBand=band;
			cfg.heatPower=380;
			cfg.coolTop=6.5;
			cfg.heatTarget=420;
			cfg.TMin=255;
			cfg.TMax=420;
			cfg.beta=scaleBeta(cfg.N);
		} else if(s==="jet"){
			cfg.thermalSources=1;
			cfg.heatBand=band;
			cfg.heatPower=180;
			cfg.coolTop=1.2;
			cfg.heatTarget=520;
			cfg.TMin=290;
			cfg.TMax=520;
			cfg.beta=scaleBeta(cfg.N);
		} else if(s==="rain"){
			cfg.thermalSources=1;
			cfg.heatBand=band;
			cfg.heatPower=420;
			cfg.coolTop=3.0;
			cfg.heatTarget=405;
			cfg.TCeil=235;
			cfg.TMin=235;
			cfg.TMax=405;
			cfg.beta=scaleBeta(cfg.N);
		} else if(s==="arch"){
			cfg.thermalSources=1;
			cfg.heatBand=band;
			cfg.heatPower=220;
			cfg.coolTop=0.3;
			cfg.heatTarget=340;
			cfg.TMin=240;
			cfg.TMax=340;
			cfg.beta=scaleBeta(cfg.N);
		}
		// product policy, off unless the app or the browser harness sets the flag,
		// so an oracle that named mgCycles keeps that knob. ?cycles= sets
		// mgCyclesPinned and wins. jet is the only scenario that drops a cycle:
		// dam and blobs are 18x on the solve, cycle is 27x on its own trajectory.
		if(cfg.productCycleGate && !cfg.mgCyclesPinned){
			cfg.mgCycles=s==="jet"?1:defaults.mgCycles;
		}
	}
	function recommendedIters(N){
		let k=Math.ceil(N*0.18+20);
		if(k<20) k=20;
		if(k>200) k=200;
		return k;
	}
	// the active solver's sweep count: coarse-level sweeps for multigrid,
	// red-black sweeps for the ?mg=0 fallback. a hand-built cfg that only sets
	// pressureIters keeps the old single-knob meaning instead of reading as 1.
	function solverIters(cfg){
		const coarse=cfg.mgCoarseIters!==undefined?cfg.mgCoarseIters:cfg.pressureIters;
		const raw=cfg.useMultigrid?coarse:cfg.pressureIters;
		const n=Math.round(raw)||1;
		return n<1?1:n;
	}
	function setSolverIters(cfg,value){
		const n=Math.max(1,Math.round(value)||1);
		if(cfg.useMultigrid)cfg.mgCoarseIters=n;
		else cfg.pressureIters=n;
	}
	function makeParamsArray(){
		return new Float32Array(128);
	}
	function writeParams(a,cfg){
		a[paramsOff.N]=cfg.N;
		a[paramsOff.invN]=1/cfg.N;
		a[paramsOff.texW]=cfg.texW;
		a[paramsOff.texH]=cfg.texH;
		a[paramsOff.dt]=cfg.dt;
		a[paramsOff.substeps]=cfg.substeps;
		a[paramsOff.time]=cfg.time;
		a[paramsOff.frame]=cfg.frame;
		a[paramsOff.paintX]=cfg.paintX||0;
		a[paramsOff.paintY]=cfg.paintY||0;
		a[paramsOff.paintW]=cfg.paintW||0;
		a[paramsOff.paintH]=cfg.paintH||0;
		a[paramsOff.gravityX]=cfg.gravityX;
		a[paramsOff.gravityY]=cfg.gravityY;
		a[paramsOff.rhoRef]=cfg.rhoRef;
		a[paramsOff.beta]=cfg.beta;
		a[paramsOff.TRef]=cfg.TRef;
		a[paramsOff.mouseX]=cfg.mouseX;
		a[paramsOff.mouseY]=cfg.mouseY;
		a[paramsOff.mouseStrength]=cfg.mouseStrength;
		a[paramsOff.rhoMean]=cfg.rhoMean||0;
		a[paramsOff.mouseVX]=cfg.mouseVX||0;
		a[paramsOff.mouseVY]=cfg.mouseVY||0;
		a[paramsOff.mouseR]=cfg.mouseR;
		a[paramsOff.pressureIters]=solverIters(cfg);
		a[paramsOff.fctLimiter]=cfg.fctLimiter;
		a[paramsOff.viscosity]=cfg.viscosity;
		a[paramsOff.kLiq]=cfg.kLiq;
		a[paramsOff.kSol]=cfg.kSol;
		a[paramsOff.kGas]=cfg.kGas;
		a[paramsOff.TEps]=cfg.TEps;
		a[paramsOff.SMax]=cfg.SMax;
		a[paramsOff.tauPhase]=cfg.tauPhase;
		a[paramsOff.vClamp]=cfg.vClamp;
		a[paramsOff.heatPower]=cfg.heatPower;
		a[paramsOff.coolTop]=cfg.coolTop;
		a[paramsOff.wallCool]=cfg.wallCool;
		a[paramsOff.heatBand]=cfg.heatBand;
		a[paramsOff.thermalOn]=cfg.thermalSources||0;
		a[paramsOff.heatTarget]=cfg.heatTarget||0;
		a[paramsOff.cSol]=cfg.cSol;
		a[paramsOff.cLiq]=cfg.cLiq;
		a[paramsOff.cGas]=cfg.cGas;
		a[paramsOff.Lf]=cfg.Lf;
		a[paramsOff.Lv]=cfg.Lv;
		a[paramsOff.TMelt]=cfg.TMelt;
		a[paramsOff.TBoil]=cfg.TBoil;
		a[paramsOff.rhoLiq]=cfg.rhoLiq;
		a[paramsOff.rhoGas0]=cfg.rhoGas0;
		a[paramsOff.rhoVac]=cfg.rhoVac;
		a[paramsOff.mEps]=cfg.mEps;
		a[paramsOff.chargeLatent]=cfg.chargeLatent===undefined?1:cfg.chargeLatent;
		a[paramsOff.TCeil]=cfg.TCeil||0;
		a[paramsOff.macLimiter]=cfg.macLimiter===undefined?1:cfg.macLimiter;
		a[paramsOff.viewMode]=cfg.viewMode;
		a[paramsOff.TMin]=cfg.TMin;
		a[paramsOff.TMax]=cfg.TMax;
		a[paramsOff.PMin]=cfg.PMin;
		a[paramsOff.PMax]=cfg.PMax;
		a[paramsOff.VMax]=cfg.VMax;
		a[paramsOff.overlay]=cfg.overlay;
		a[paramsOff.colorGain]=cfg.colorGain;
		a[paramsOff.buoyX0]=cfg.buoyX0||0;
		a[paramsOff.buoyY0]=cfg.buoyY0||0;
		a[paramsOff.buoyW]=cfg.buoyW||0;
		a[paramsOff.buoyH]=cfg.buoyH||0;
		a[paramsOff.buoySub]=cfg.buoySub||0;
		a[paramsOff.buoyShow]=cfg.buoyShow||0;
		a[paramsOff.mgS]=cfg.mgCoarsen||2;
		a[paramsOff.rhoBase]=cfg.rhoBase!==undefined?cfg.rhoBase:(cfg.rhoMean||0);
	}
	Fluid.core.config={paramsOff,defaults,viewModes,viewModeNames,scenarioNames,paintModeNames,keyMap,substepCount,velocityCap,prepareScenario,recommendedIters,solverIters,setSolverIters,makeParamsArray,writeParams};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.config;
})(globalThis);
