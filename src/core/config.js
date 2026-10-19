(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const paramsOff={
		N:0,invN:1,texW:2,texH:3,dt:4,substeps:5,time:6,frame:7,
		paintX:8,paintY:9,paintW:10,paintH:11,
		gravityX:16,gravityY:17,rhoRef:18,beta:19,TRef:20,mouseX:21,mouseY:22,mouseStrength:23,
		mouseVX:24,mouseVY:25,mouseR:26,rhoMean:27,
		pressureIters:32,fctLimiter:33,viscosity:34,conductivity:35,SMax:36,tauPhase:37,vClamp:38,
		heatPower:39,coolTop:40,wallCool:41,heatBand:42,thermalOn:43,heatTarget:44,
		cSol:48,cLiq:49,cGas:50,Lf:51,Lv:52,TMelt:53,TBoil:54,rhoLiq:55,rhoGas0:56,rhoVac:57,mEps:58,
		viewMode:80,TMin:81,TMax:82,PMin:83,PMax:84,VMax:85,overlay:86,colorGain:87,
		buoyX0:88,buoyY0:89,buoyW:90,buoyH:91,buoySub:92,buoyShow:93,
		emitters:96
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
		pressureIters:60,
		fctLimiter:1,
		viscosity:0.0,
		conductivity:0.2,
		// thermal sources (0.4.0): fixed power per bottom-band cell, cold plate
		// relaxation per top-band cell, wall exchange coefficient. only applied
		// while cfg.thermalSources is on (the convection scenario turns it on).
		heatPower:30,
		coolTop:0.5,
		wallCool:0,
		heatBand:2,
		thermalSources:0,
		// convection demo endpoints: the scenario starts from this unstable
		// stratification (hot below cold) instead of uniform T_amb plus a heater
		// whose heat conduction spreads <1 cell in 10 s at the 0.2 default.
		// the heater/cold plate then sustain the gradient the rolls feed on.
		heatTarget:0,
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
		viewMode:0,
		TMin:273,
		TMax:373,
		PMin:-100,
		PMax:100,
		VMax:400,
		overlay:0,
		colorGain:1.2,
		statsEvery:30,
		cflTarget:0.8,
		substepMax:8,
		// the auto-cfl substep policy is optional. with it off the frame still runs
		// substepsFixed steps, not one: a single 1/60 s step at 512^2 lets the cfl
		// number reach 20+, the outflow limiter clamps most of the field and the
		// liquid reads as smeared and sticky. 4 is the smallest count that keeps the
		// frame budget playable while giving the solve a usable dt.
		substepsAuto:false,
		substepsFixed:4,
		vMax:8,
		useMultigrid:false,
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
	const scenarioNames=["blobs","dam","still","convection"];
	const paintModeNames=["dye","wall","erase"];
	const keyMap={
		"0":"color","1":"temperature","2":"pressure","3":"velocity","4":"divergence","5":"flags",
		"6":"thermal","7":"phase",
		" ":"pause",".":"step","r":"reset","h":"help","s":"dumpStats","d":"dumpField","g":"gravity",
		"+":"iterUp","-":"iterDown","=":"iterUp","_":"iterDown","[":"sizeDown","]":"sizeUp",
		"c":"cycleColor","b":"cyclePaint","n":"nextScenario","f":"toggleFct",
		"u":"toggleSubsteps","y":"toggleBuoy"
	};
	// steps per rendered frame. the single implementation: app, core experiments and
	// the gpu plumbing mock all call this, so the rule cannot drift between them.
	// auto follows the cfl target from the live velocity, clamped to the step budget;
	// off runs the fixed count from the slider, clamped the same way.
	function substepCount(cfg,maxVel){
		if(!cfg.substepsAuto){
			let n=Math.round(cfg.substepsFixed)||1;
			if(n<1)n=1;
			if(n>cfg.substepMax)n=cfg.substepMax;
			return n;
		}
		let ns=Math.ceil(maxVel*cfg.frameDt/cfg.cflTarget);
		if(!(ns>=1))ns=1;
		if(ns>cfg.substepMax)ns=cfg.substepMax;
		return ns;
	}
	// per-scenario presentation/physics setup, applied by both solvers on reset
	// so cpu and gpu cannot drift. convection gets a heater band that scales
	// with N (a fixed 2-cell band is invisible at 512) and a temperature view
	// range matched to its stratification, not the full 273..373 phase range:
	// with the full range a 60 K gradient used only ~half the colormap and the
	// old uniform start used none of it.
	function prepareScenario(cfg){
		const conv=cfg.scenario==="convection";
		cfg.thermalSources=conv?1:0;
		cfg.heatBand=conv?Math.max(2,Math.round(cfg.N/32)):defaults.heatBand;
		cfg.TMin=conv?cfg.convTopT:defaults.TMin;
		cfg.heatTarget=conv?cfg.convBotT:0;
		// velocities are in cells/s, so at fixed beta a 512 box turns over 8x slower
		// than a 64 box. scale buoyancy sub-linearly so turnover stays visible
		// without the 512 box exploding to 400 cells/s (measured with linear scale).
		// sqrt(N/64) gives ~2.8x at N=512 instead of 8x, capped at 4x.
		if(conv){
			const s=Math.sqrt(Math.max(1,cfg.N/64));
			cfg.beta=defaults.beta*Math.min(4,s);
		} else {
			cfg.beta=defaults.beta;
		}
		cfg.TMax=conv?cfg.convBotT:defaults.TMax;
	}
	function recommendedIters(N){
		// Gauss-Seidel needs O(N) iterations for similar residual; tuned to keep
		// maxDiv <0.2 at N=64/K=60 and scale to N=512. slider max is 200.
		let k=Math.round(28+N*0.15);
		if(k<20) k=20;
		if(k>200) k=200;
		return k;
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
		a[paramsOff.pressureIters]=cfg.pressureIters;
		a[paramsOff.fctLimiter]=cfg.fctLimiter;
		a[paramsOff.viscosity]=cfg.viscosity;
		a[paramsOff.conductivity]=cfg.conductivity;
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
	}
	Fluid.core.config={paramsOff,defaults,viewModes,viewModeNames,scenarioNames,paintModeNames,keyMap,substepCount,prepareScenario,recommendedIters,makeParamsArray,writeParams};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.config;
})(globalThis);
