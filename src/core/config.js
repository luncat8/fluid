(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const paramsOff={
		N:0,invN:1,texW:2,texH:3,dt:4,substeps:5,time:6,frame:7,
		gravityX:16,gravityY:17,rhoRef:18,beta:19,TRef:20,mouseX:21,mouseY:22,mouseStrength:23,
		jacobiIters:32,fctLimiter:33,viscosity:34,conductivity:35,SMax:36,tauPhase:37,
		cSol:48,cLiq:49,cGas:50,Lf:51,Lv:52,TMelt:53,TBoil:54,rhoLiq:55,rhoGas0:56,rhoVac:57,mEps:58,
		viewMode:80,TMin:81,TMax:82,PMin:83,PMax:84,VMax:85,overlay:86,
		emitters:96
	};
	const defaults={
		N:256,
		texW:258,
		texH:258,
		dt:1/60,
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
		jacobiIters:40,
		fctLimiter:1,
		viscosity:0.0,
		conductivity:0.2,
		SMax:30,
		tauPhase:0.05,
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
		statsEvery:30,
		cflTarget:0.8,
		substepMax:8,
		vMax:8,
		conductivityPlumbing:0.2,
		colorGain:1.0
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
	const keyMap={
		"0":"color","1":"temperature","2":"pressure","3":"velocity","4":"divergence","5":"flags",
		" ":"pause",".":"step","r":"reset","h":"help","s":"dumpStats","d":"dumpField","g":"gravity","w":"sparkline","p":"profileV","y":"profileH",
		"+":"iterUp","-":"iterDown","=":"iterUp","_":"iterDown","[":"sizeDown","]":"sizeUp"
	};
	function makeParamsArray(){
		const a=new Float32Array(128);
		return a;
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
		a[paramsOff.gravityX]=cfg.gravityX;
		a[paramsOff.gravityY]=cfg.gravityY;
		a[paramsOff.rhoRef]=cfg.rhoRef;
		a[paramsOff.beta]=cfg.beta;
		a[paramsOff.TRef]=cfg.TRef;
		a[paramsOff.mouseX]=cfg.mouseX;
		a[paramsOff.mouseY]=cfg.mouseY;
		a[paramsOff.mouseStrength]=cfg.mouseStrength;
		a[paramsOff.jacobiIters]=cfg.jacobiIters;
		a[paramsOff.fctLimiter]=cfg.fctLimiter;
		a[paramsOff.viscosity]=cfg.viscosity;
		a[paramsOff.conductivity]=cfg.conductivity;
		a[paramsOff.SMax]=cfg.SMax;
		a[paramsOff.tauPhase]=cfg.tauPhase;
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
	}
	Fluid.core.config={paramsOff,defaults,viewModes,viewModeNames,keyMap,makeParamsArray,writeParams};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.config;
})(globalThis);
