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
		beta:0.0004,
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
		// the auto-cfl substep policy is optional. off = one step per frame at
		// dt=frameDt, which is what the measurements and tests are compared
		// against; the per-cell outflow limiter in compute_flux keeps that monotone.
		substepsAuto:false,
		vMax:8,
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
	const scenarioNames=["blobs","dam","still"];
	const paintModeNames=["dye","wall","erase"];
	const keyMap={
		"0":"color","1":"temperature","2":"pressure","3":"velocity","4":"divergence","5":"flags",
		"6":"thermal","7":"phase",
		" ":"pause",".":"step","r":"reset","h":"help","s":"dumpStats","d":"dumpField","g":"gravity",
		"+":"iterUp","-":"iterDown","=":"iterUp","_":"iterDown","[":"sizeDown","]":"sizeUp",
		"c":"cycleColor","b":"cyclePaint","n":"nextScenario","f":"toggleFct",
		"u":"toggleSubsteps","y":"toggleBuoy"
	};
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
	Fluid.core.config={paramsOff,defaults,viewModes,viewModeNames,scenarioNames,paintModeNames,keyMap,makeParamsArray,writeParams};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.config;
})(globalThis);
