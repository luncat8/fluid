(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	function createSolver(device,canvas,context,format,cfg){
		const fields=Fluid.gpu.fields;
		const passes=Fluid.gpu.passes;
		const stats=Fluid.core.stats;
		const coreSolver=Fluid.core.solver;
		const coreProbes=Fluid.core.probes;
		const coreEmitters=Fluid.core.emitters;
		const coreColliders=Fluid.core.colliders;
		let pool=fields.createPool(device,cfg.N,cfg.mgCoarsen);
		const pipes=passes.createPipelines(device,format);
		let groups=passes.makeBindGroups(device,pipes,pool);
		// null when the module is absent or the device lacks timestamp-query
		const prof=Fluid.gpu.profiler?Fluid.gpu.profiler.create(device):null;
		const PROFILE_GROUP=Fluid.gpu.profiler?Fluid.gpu.profiler.GROUP:null;
		const paramsArray=Fluid.core.config.makeParamsArray();
		const buoy=Fluid.core.buoy.create(cfg);
		const buoyRaw=new Float32Array(Fluid.core.buoy.SAMPLE_LEN);
		const buoyZero=new Float32Array(Fluid.core.buoy.SAMPLE_LEN);
		let buoyPending=false;
		let buoyInFlight=false;
		let buoyGeneration=0;
		// point queries (0.7.0 t5): one table for every game probe, sampled on its own
		// cadence and copied back through the same single-in-flight rule as stats/buoy
		const probeTable=coreProbes.create();
		const probeRaw=new Float32Array(coreProbes.LEN);
		let probePending=false;
		let probeInFlight=false;
		let probeGeneration=0;
		let probeFrame=0;
		const emitterTable=coreEmitters?coreEmitters.create():null;
		const colliderTable=coreColliders?coreColliders.create():null;
		const phaseEventMod=Fluid.core.phaseEvents||(stats&&stats.phaseEvents)||null;
		const phaseEventState=phaseEventMod?phaseEventMod.create(cfg.texW,cfg.texH):null;
		const phaseEventRaw=new Float32Array(stats.PHASE_EVENT_LEN||560);
		let phaseChangeCb=null;
		let phasePending=false;
		let phaseInFlight=false;
		let phaseGeneration=0;
		let phaseFrame=0;
		let phaseSeedNeeded=false;
		const emitLedgerZero=new Float32Array(stats.EMIT_LEDGER_LEN||20);
		let paintInjM=0;
		const emitInjKnown={m:0,dr:0,dg:0,db:0,E:0};
		const emitRemKnown={m:0,dr:0,dg:0,db:0,E:0};
		const emitNetKnown={m:0,dr:0,dg:0,db:0,E:0};
		const pendingEmitNet={m:0,dr:0,dg:0,db:0,E:0};
		const state={m:0,t:0,v:0};
		let flagsMirror=new Uint8Array(cfg.texW*cfg.texH*4);
		let ledger=null;
		let destroyed=false;
		const readbackArray=new Float32Array(stats.STATS_LEN);
		let readbackInFlight=false;
		let pendingRead=false;
		let pendingFrame=0;
		let pendingGeneration=0;
		let statsGeneration=0;
		const pendingExpected={m:0,dr:0,dg:0,db:0,E:0};
		let heatNetKnown=0;
		let pendingHeatNet=0;
		let frameCount=0;
		let timeAcc=0;
		let pSpanSmooth=100;
		const phaseCounts=[0,0,0,0,0,0];
		const diag={sSaturated:0,poissonRes:0,poissonRhs:0,poissonRel:0,poissonDrift:0,poissonDriftRhs:0,poissonDriftRel:0};
		let statsCache=null;
		const statsView={frame:0,statsFrame:0,totals:null,expected:null,mins:null,maxM:0,cntNeg:0,cntOver:0,motion:null,thermal:null,phaseCounts,diag,ledger:null};
		function writeParams(){
			Fluid.core.config.writeParams(paramsArray,cfg);
			paramsArray[Fluid.core.config.paramsOff.mgS]=pool.mgS;
			device.queue.writeBuffer(pool.params,0,paramsArray);
		}
		const mgParamsArray=Fluid.core.config.makeParamsArray();
		function writeMgParams(){
			Fluid.core.config.writeParams(mgParamsArray,cfg);
			mgParamsArray[Fluid.core.config.paramsOff.texW]=pool.mgTexW;
			mgParamsArray[Fluid.core.config.paramsOff.texH]=pool.mgTexH;
			mgParamsArray[Fluid.core.config.paramsOff.mgS]=pool.mgS;
			device.queue.writeBuffer(pool.mgParams,0,mgParamsArray);
		}
		function runPrimePhase(charge){
			const needColl=!!(colliderTable&&(colliderTable.maskDirty||colliderTable.live>0||colliderTable.hasGpuMask));
			if(needColl&&colliderTable.dirty)uploadColliders();
			cfg.chargeLatent=charge?1:0;
			writeParams();
			const encoder=device.createCommandEncoder({label:"primePhase"});
			const cpass=encoder.beginComputePass({label:"primePhase"});
			if(needColl){
				passes.encodeColliderMask(cpass,pool,pipes,groups);
				colliderTable.maskDirty=false;
				colliderTable.hasGpuMask=colliderTable.live>0;
			}
			passes.encodePrimePhase(cpass,pool,pipes,groups,state);
			cpass.end();
			device.queue.submit([encoder.finish()]);
			cfg.chargeLatent=1;
		}
		function reset(){
			statsGeneration++;
			if(prof)prof.invalidate();
			pendingRead=false;
			buoyPending=false;
			buoyGeneration=statsGeneration;
			probePending=false;
			phasePending=false;
			phaseGeneration=statsGeneration;
			phaseSeedNeeded=false;
			coreProbes.invalidate(probeTable);
			device.queue.writeBuffer(pool.buoy,0,buoyZero);
			Fluid.core.config.prepareScenario(cfg);
			buoy.reset();
			const sc=coreSolver.buildScenario(cfg);
			fields.writeTextureRect(device,pool.mass.tex[0],cfg.texW,cfg.texH,"rgba32float",sc.massA);
			fields.writeTextureRect(device,pool.mass.tex[1],cfg.texW,cfg.texH,"rgba32float",sc.massA);
			fields.writeTextureRect(device,pool.mass.tex[2],cfg.texW,cfg.texH,"rgba32float",sc.massA);
			fields.writeTextureRect(device,pool.therm.tex[0],cfg.texW,cfg.texH,"rgba32float",sc.thermA);
			fields.writeTextureRect(device,pool.therm.tex[1],cfg.texW,cfg.texH,"rgba32float",sc.thermA);
			fields.clearTexture(device,pool.vel.tex[0],cfg.texW,cfg.texH,"rg32float");
			fields.clearTexture(device,pool.vel.tex[1],cfg.texW,cfg.texH,"rg32float");
			fields.clearTexture(device,pool.macTmp,cfg.texW,cfg.texH,"rg32float");
			fields.clearTexture(device,pool.press.tex[0],cfg.texW,cfg.texH,"r32float");
			fields.clearTexture(device,pool.press.tex[1],cfg.texW,cfg.texH,"r32float");
			fields.writeTextureRect(device,pool.flags,cfg.texW,cfg.texH,"rgba8uint",sc.flagsA);
			fields.writeTextureRect(device,pool.flagsTmp,cfg.texW,cfg.texH,"rgba8uint",sc.flagsA);
			fields.clearTexture(device,pool.sfield,cfg.texW,cfg.texH,"rgba32float");
			fields.clearTexture(device,pool.sfieldTmp,cfg.texW,cfg.texH,"rgba32float");
			fields.clearTexture(device,pool.accum,cfg.texW,cfg.texH,"rg32float");
			fields.clearTexture(device,pool.accumTmp,cfg.texW,cfg.texH,"rg32float");
			fields.clearTexture(device,pool.flux,cfg.texW,cfg.texH,"rg32float");
			fields.clearTexture(device,pool.divsrc,cfg.texW,cfg.texH,"r32float");
			fields.clearTexture(device,pool.mgPress.tex[0],pool.mgTexW,pool.mgTexH,"r32float");
			fields.clearTexture(device,pool.mgPress.tex[1],pool.mgTexW,pool.mgTexH,"r32float");
			flagsMirror.set(sc.flagsA);
			state.m=0;state.t=0;state.v=0;
			heatNetKnown=0;pendingHeatNet=0;
			paintInjM=0;
			emitInjKnown.m=0;emitInjKnown.dr=0;emitInjKnown.dg=0;emitInjKnown.db=0;emitInjKnown.E=0;
			emitRemKnown.m=0;emitRemKnown.dr=0;emitRemKnown.dg=0;emitRemKnown.db=0;emitRemKnown.E=0;
			emitNetKnown.m=0;emitNetKnown.dr=0;emitNetKnown.dg=0;emitNetKnown.db=0;emitNetKnown.E=0;
			pendingEmitNet.m=0;pendingEmitNet.dr=0;pendingEmitNet.dg=0;pendingEmitNet.db=0;pendingEmitNet.E=0;
			device.queue.writeBuffer(pool.emitterLedger,0,emitLedgerZero);
			pSpanSmooth=100;
			cfg.PMin=-100;cfg.PMax=100;
			const init=stats.totalsFromFields(sc.massA,sc.thermA,cfg.texW,cfg.texH);
			ledger=stats.createLedger(init);
			cfg.rhoMean=init.sumM/(cfg.N*cfg.N);
			cfg.rhoBase=cfg.rhoMean;
			frameCount=0;
			timeAcc=0;
			cfg.frame=0;
			cfg.time=0;
			for(let k=0;k<6;k++)phaseCounts[k]=0;
			if(phaseEventState)phaseEventState.prev.fill(255);
			for(let y=1;y<cfg.texH-1;y++){
				for(let x=1;x<cfg.texW-1;x++){
					const cell=x+y*cfg.texW;
					const i=cell*4;
					const ph=coreSolver.phaseOf(sc.massA[i+3],sc.thermA[i],cfg).phase;
					phaseCounts[ph]++;
					if(phaseEventState){
						const r=sc.flagsA[i];
						const isStatic=(r&1)!==0||r===255;
						phaseEventState.prev[cell]=isStatic?255:(ph>>>0);
					}
				}
			}
			if(phaseEventState){
				const b=phaseEventState.batch;
				b.frame=0;b.total=0;b.count=0;
				b.melted=0;b.froze=0;b.boiled=0;b.condensed=0;
				b.melt=0;b.freeze=0;b.boil=0;b.condense=0;
				b.matrix.fill(0);
				phaseEventState.seeded=true;
				device.queue.writeBuffer(pool.phasePrev,0,phaseEventState.prev);
			}
			diag.sSaturated=0;
			diag.poissonRes=0;diag.poissonRhs=0;diag.poissonRel=0;
			diag.poissonDrift=0;diag.poissonDriftRhs=0;diag.poissonDriftRel=0;
			const expected=stats.writeExpectedTotals({m:0,dr:0,dg:0,db:0,E:0},ledger);
			statsCache={
				frame:0,
				totals:{m:init.sumM,dr:init.sumDr,dg:init.sumDg,db:init.sumDb,E:init.sumE},
				expected,
				mins:{m:init.minM,dr:init.minDr,dg:init.minDg,db:init.minDb,E:init.minE},
				maxM:init.maxM,cntNeg:init.cntNeg,cntOver:init.cntOver,
				motion:{maxVel:0,pMin:0,pMax:0,maxDiv:0},
				thermal:{TMin:cfg.TAmb,TMax:cfg.TAmb,TMean:cfg.TAmb,kinetic:0,heatIn:0,heatOut:0}
			};
			writeMgParams();
			// a resize rebuilds the pool: the new probe/emitter/collider buffers start zeroed,
			// so the live tables have to be pushed again before primePhase or active slots stop
			uploadProbes();
			if(emitterTable){
				coreEmitters.syncConfig(emitterTable,cfg,false);
				uploadEmitters();
			}
			if(colliderTable){
				coreColliders.syncConfig(colliderTable,cfg,false);
				colliderTable.hasGpuMask=false;
				colliderTable.maskDirty=colliderTable.live>0;
				uploadColliders();
			}
			runPrimePhase(false);
			writeParams();
		}
		function uploadProbes(){
			device.queue.writeBuffer(pool.probes,0,probeTable.upload);
		}
		function uploadEmitters(){
			if(!emitterTable)return;
			emitterTable.dirty=false;
			device.queue.writeBuffer(pool.emitters,0,emitterTable.upload);
		}
		function uploadColliders(){
			if(!colliderTable)return;
			colliderTable.dirty=false;
			device.queue.writeBuffer(pool.colliders,0,colliderTable.upload);
		}
		function addCollider(spec){
			if(!colliderTable)return -1;
			const id=coreColliders.add(colliderTable,cfg,spec);
			if(id<0)return -1;
			uploadColliders();
			return id;
		}
		function moveCollider(id,x,y,vx,vy){
			if(!colliderTable||!coreColliders.move(colliderTable,cfg,id,x,y,vx,vy))return false;
			uploadColliders();
			return true;
		}
		function setCollider(id,patch){
			if(!colliderTable||!coreColliders.set(colliderTable,cfg,id,patch))return false;
			uploadColliders();
			return true;
		}
		function removeCollider(id){
			if(!colliderTable||!coreColliders.remove(colliderTable,id))return false;
			uploadColliders();
			return true;
		}
		function colliderCount(){
			return colliderTable?colliderTable.live:0;
		}
		function setObstacles(maskArray,w,h){
			if(!coreColliders||!coreColliders.applyObstacleMask(flagsMirror,cfg.texW,cfg.texH,maskArray,w,h))return false;
			fields.writeTextureRect(device,pool.flags,cfg.texW,cfg.texH,"rgba8uint",flagsMirror);
			runPrimePhase(false);
			return true;
		}
		function paintObstacle(x,y,radius,kind){
			if(!coreColliders||!Number.isFinite(x)||!Number.isFinite(y))return false;
			const k=coreColliders.parsePaintKind(kind);
			if(!k)return false;
			const r=Math.min(cfg.N,Math.max(0.5,Number.isFinite(radius)&&radius>0?radius:(cfg.paintR||8)));
			if(k==="wall"){
				if(!coreColliders.paintWallDisc(flagsMirror,cfg.texW,cfg.texH,x,y,r,false))return false;
				fields.writeTextureRect(device,pool.flags,cfg.texW,cfg.texH,"rgba8uint",flagsMirror);
				runPrimePhase(false);
				return true;
			}
			if(k==="solid"){
				cfg.paintX=x;cfg.paintY=y;cfg.paintW=r*r;cfg.paintH=1;
				cfg.chargeLatent=0;
				writeParams();
				const encoder=device.createCommandEncoder({label:"paintSolid"});
				const cpass=encoder.beginComputePass({label:"paintSolid"});
				passes.encodePaintSolid(cpass,pool,pipes,groups,state);
				passes.encodePrimePhase(cpass,pool,pipes,groups,state);
				cpass.end();
				device.queue.submit([encoder.finish()]);
				cfg.paintX=0;cfg.paintY=0;cfg.paintW=0;cfg.paintH=0;
				cfg.chargeLatent=1;
				return true;
			}
			if(!coreColliders.paintWallDisc(flagsMirror,cfg.texW,cfg.texH,x,y,r,true))return false;
			fields.writeTextureRect(device,pool.flags,cfg.texW,cfg.texH,"rgba8uint",flagsMirror);
			const needColl=!!(colliderTable&&(colliderTable.maskDirty||colliderTable.live>0||colliderTable.hasGpuMask));
			if(needColl&&colliderTable.dirty)uploadColliders();
			cfg.paintX=x;cfg.paintY=y;cfg.paintW=r*r;cfg.paintH=-1;
			cfg.chargeLatent=0;
			writeParams();
			const encoder=device.createCommandEncoder({label:"paintErase"});
			const cpass=encoder.beginComputePass({label:"paintErase"});
			if(needColl){
				passes.encodeColliderMask(cpass,pool,pipes,groups);
				colliderTable.maskDirty=false;
				colliderTable.hasGpuMask=colliderTable.live>0;
			}
			passes.encodePaintSolid(cpass,pool,pipes,groups,state);
			passes.encodePrimePhase(cpass,pool,pipes,groups,state);
			cpass.end();
			device.queue.submit([encoder.finish()]);
			cfg.paintX=0;cfg.paintY=0;cfg.paintW=0;cfg.paintH=0;
			cfg.chargeLatent=1;
			return true;
		}
		function addEmitter(spec){
			if(!emitterTable)return -1;
			const id=coreEmitters.add(emitterTable,cfg,spec);
			if(id<0)return -1;
			uploadEmitters();
			return id;
		}
		function moveEmitter(id,x,y){
			if(!emitterTable||!coreEmitters.move(emitterTable,cfg,id,x,y))return false;
			uploadEmitters();
			return true;
		}
		function setEmitter(id,patch){
			if(!emitterTable||!coreEmitters.set(emitterTable,cfg,id,patch))return false;
			uploadEmitters();
			return true;
		}
		function removeEmitter(id){
			if(!emitterTable||!coreEmitters.remove(emitterTable,id))return false;
			uploadEmitters();
			return true;
		}
		function emitterCount(){
			return emitterTable?emitterTable.live:0;
		}
		function addProbe(x,y){
			const id=coreProbes.add(probeTable,cfg,x,y,frameCount);
			if(id<0)return -1;
			uploadProbes();
			return id;
		}
		function moveProbe(id,x,y){
			if(!coreProbes.move(probeTable,cfg,id,x,y))return false;
			uploadProbes();
			return true;
		}
		function removeProbe(id){
			if(!coreProbes.remove(probeTable,id))return false;
			uploadProbes();
			return true;
		}
		function sampleProbe(id,out){
			return coreProbes.sample(probeTable,cfg,frameCount,id,out);
		}
		function paintDye(u,v,rgb){
			const patch=coreSolver.buildDyePatch(cfg,u,v,rgb);
			if(!patch)return;
			if(patch.w>fields.PATCH||patch.h>fields.PATCH)return;
			fields.writePatch(device,pool.patchMass,patch.w,patch.h,"rgba32float",patch.massPatch);
			fields.writePatch(device,pool.patchE,patch.w,patch.h,"r32float",patch.ePatch);
			cfg.paintX=patch.x0;cfg.paintY=patch.y0;cfg.paintW=patch.w;cfg.paintH=patch.h;
			cfg.chargeLatent=1;
			writeParams();
			const encoder=device.createCommandEncoder({label:"paint"});
			const cpass=encoder.beginComputePass({label:"paint"});
			passes.encodePaint(cpass,pool,pipes,groups,state);
			passes.encodePrimePhase(cpass,pool,pipes,groups,state);
			cpass.end();
			device.queue.submit([encoder.finish()]);
			cfg.paintX=0;cfg.paintY=0;cfg.paintW=0;cfg.paintH=0;
			const t=coreSolver.patchTotals(patch);
			paintInjM+=t.m;
			ledger.injectedM+=t.m;ledger.injectedDr+=t.dr;ledger.injectedDg+=t.dg;ledger.injectedDb+=t.db;ledger.injectedE+=t.E;
			cfg.rhoBase=(ledger.initialM+paintInjM)/(cfg.N*cfg.N);
			cfg.rhoMean=(ledger.initialM+ledger.injectedM-ledger.removedM)/(cfg.N*cfg.N);
		}
		function paintWall(u,v,erase){
			const patch=coreSolver.buildWallPatch(cfg,u,v,erase);
			if(!patch)return;
			for(let y=0;y<patch.h;y++){
				for(let x=0;x<patch.w;x++){
					const op=patch.flagsPatch[(x+y*patch.w)*4];
					if(op===0)continue;
					const gi=((patch.x0+x)+(patch.y0+y)*cfg.texW)*4;
					flagsMirror[gi]=op===2?0:1;
				}
			}
			fields.writeTextureRect(device,pool.flags,cfg.texW,cfg.texH,"rgba8uint",flagsMirror);
			runPrimePhase(false);
		}
		function frame(dts,substeps,needStats){
			buoy.march(dts*substeps);
			const needBuoy=buoy.state.active&&(frameCount%Math.max(1,cfg.buoySampleEvery)===0);
			const needProbe=probeTable.live>0&&(frameCount%Math.max(1,cfg.probeEvery)===0);
			const needPhase=phaseChangeCb!==null&&!phaseInFlight&&!phasePending;
			const seedPhasePreStep=needPhase&&phaseSeedNeeded&&phaseEventState!==null;
			if(seedPhasePreStep){
				phaseEventState.prev.fill(255);
				device.queue.writeBuffer(pool.phasePrev,0,phaseEventState.prev);
				phaseSeedNeeded=false;
			}
			if(emitterTable&&(coreEmitters.syncConfig(emitterTable,cfg,false)||emitterTable.dirty))uploadEmitters();
			if(colliderTable&&(coreColliders.syncConfig(colliderTable,cfg,false)||colliderTable.dirty))uploadColliders();
			const scalarEmit=!!(emitterTable&&emitterTable.scalarCount>0);
			const massDrainEmit=!!(emitterTable&&emitterTable.massDrainCount>0);
			const needCollMask=!!(colliderTable&&(colliderTable.maskDirty||(colliderTable.live===0&&colliderTable.hasGpuMask)));
			const movingColl=!!(colliderTable&&colliderTable.movingCount>0);
			cfg.dt=dts;
			cfg.substeps=substeps;
			cfg.frame=frameCount;
			cfg.time=timeAcc;
			cfg.chargeLatent=1;
			cfg.rhoBase=(ledger.initialM+paintInjM)/(cfg.N*cfg.N);
			cfg.rhoMean=(ledger.initialM+ledger.injectedM-ledger.removedM)/(cfg.N*cfg.N);
			writeParams();
			// profiling splits the one compute pass at timing-group boundaries;
			// the dispatch sequence is identical either way (0.6.0 t1).
			const kIters=Fluid.core.config.solverIters(cfg);
			const profiling=!!(prof&&cfg.gpuTiming>0);
			const encoder=device.createCommandEncoder({label:"frame"});
			if(profiling){
				prof.beginFrame(substeps,frameCount+1);
				if(needCollMask){
					passes.encodeColliderMask(prof.pass(encoder,PROFILE_GROUP.vel,"vel"),pool,pipes,groups);
					colliderTable.maskDirty=false;
					colliderTable.hasGpuMask=colliderTable.live>0;
				}
				if(seedPhasePreStep){
					passes.encodePhaseChange(prof.pass(encoder,PROFILE_GROUP.vel,"vel"),pool,pipes,groups,state);
				}
				for(let s=0;s<substeps;s++){
					passes.encodeStepProfiled(encoder,prof,pool,pipes,groups,state,kIters,cfg.fctLimiter>0,
						!!cfg.useMultigrid,cfg.mgPre||4,cfg.mgPost||8,cfg.mgCycles||1,
						needStats&&s===substeps-1,cfg.advectMac>0,cfg.fusePasses>0,scalarEmit,massDrainEmit,movingColl);
				}
				if(needStats||needBuoy||needProbe||needPhase){
					const cp=prof.pass(encoder,PROFILE_GROUP.stats,"stats");
					if(needStats)passes.encodeStats(cp,pool,pipes,groups,state);
					if(needBuoy)passes.encodeBuoySample(cp,pool,pipes,groups,state);
					if(needProbe)passes.encodeProbeSample(cp,pool,pipes,groups,state);
					if(needPhase)passes.encodePhaseChange(cp,pool,pipes,groups,state);
				}
				passes.encodeDisplay(prof.pass(encoder,PROFILE_GROUP.disp,"display"),pool,pipes,groups,state);
				prof.closePass();
			}else{
				const cpass=encoder.beginComputePass({label:"sim"});
				if(needCollMask){
					passes.encodeColliderMask(cpass,pool,pipes,groups);
					colliderTable.maskDirty=false;
					colliderTable.hasGpuMask=colliderTable.live>0;
				}
				if(seedPhasePreStep){
					passes.encodePhaseChange(cpass,pool,pipes,groups,state);
				}
				for(let s=0;s<substeps;s++){
					passes.encodeStep(cpass,pool,pipes,groups,state,kIters,cfg.fctLimiter>0,
						!!cfg.useMultigrid,cfg.mgPre||4,cfg.mgPost||8,cfg.mgCycles||1,
						needStats&&s===substeps-1,cfg.advectMac>0,cfg.fusePasses>0,scalarEmit,massDrainEmit,movingColl);
				}
				if(needStats){
					passes.encodeStats(cpass,pool,pipes,groups,state);
				}
				if(needBuoy){
					passes.encodeBuoySample(cpass,pool,pipes,groups,state);
				}
				if(needProbe){
					passes.encodeProbeSample(cpass,pool,pipes,groups,state);
				}
				if(needPhase){
					passes.encodePhaseChange(cpass,pool,pipes,groups,state);
				}
				passes.encodeDisplay(cpass,pool,pipes,groups,state);
				cpass.end();
			}
			if(needStats&&!readbackInFlight){
				encoder.copyBufferToBuffer(pool.stats,0,pool.statsRead,0,stats.STATS_LEN*4);
				stats.writeExpectedTotals(pendingExpected,ledger);
				pendingHeatNet=heatNetKnown;
				pendingEmitNet.m=emitNetKnown.m;pendingEmitNet.dr=emitNetKnown.dr;
				pendingEmitNet.dg=emitNetKnown.dg;pendingEmitNet.db=emitNetKnown.db;
				pendingEmitNet.E=emitNetKnown.E;
				pendingFrame=frameCount+1;
				pendingGeneration=statsGeneration;
				pendingRead=true;
			}
			if(needBuoy&&!buoyInFlight){
				encoder.copyBufferToBuffer(pool.buoy,0,pool.buoyRead,0,buoyZero.byteLength);
				buoyGeneration=statsGeneration;
				buoyPending=true;
			}
			if(needProbe&&!probeInFlight){
				encoder.copyBufferToBuffer(pool.probes,0,pool.probesRead,0,probeRaw.byteLength);
				probeGeneration=statsGeneration;
				probeFrame=frameCount+1;
				probePending=true;
			}
			if(needPhase){
				encoder.copyBufferToBuffer(pool.phaseEvents,0,pool.phaseEventsRead,0,phaseEventRaw.byteLength);
				phaseGeneration=statsGeneration;
				phaseFrame=frameCount+1;
				phasePending=true;
			}
			const view=context.getCurrentTexture().createView();
			const rdesc={
				label:"render",
				colorAttachments:[{view,loadOp:"clear",clearValue:{r:0.02,g:0.03,b:0.04,a:1},storeOp:"store"}]
			};
			if(profiling)rdesc.timestampWrites=prof.tsWrites(PROFILE_GROUP.render);
			const rpass=encoder.beginRenderPass(rdesc);
			passes.encodeRender(rpass,pipes,groups);
			rpass.end();
			if(profiling)prof.finish(encoder);
			device.queue.submit([encoder.finish()]);
			frameCount++;
			timeAcc+=dts*substeps;
			cfg.frame=frameCount;
			cfg.time=timeAcc;
		}
		function renderOnly(){
			if(colliderTable&&(coreColliders.syncConfig(colliderTable,cfg,false)||colliderTable.dirty))uploadColliders();
			const needCollMask=!!(colliderTable&&(colliderTable.maskDirty||(colliderTable.live===0&&colliderTable.hasGpuMask)));
			writeParams();
			const encoder=device.createCommandEncoder({label:"renderOnly"});
			const cpass=encoder.beginComputePass({label:"display"});
			if(needCollMask){
				passes.encodeColliderMask(cpass,pool,pipes,groups);
				colliderTable.maskDirty=false;
				colliderTable.hasGpuMask=colliderTable.live>0;
			}
			passes.encodeDisplay(cpass,pool,pipes,groups,state);
			cpass.end();
			const view=context.getCurrentTexture().createView();
			const rpass=encoder.beginRenderPass({
				label:"render",
				colorAttachments:[{view,loadOp:"clear",clearValue:{r:0.02,g:0.03,b:0.04,a:1},storeOp:"store"}]
			});
			passes.encodeRender(rpass,pipes,groups);
			rpass.end();
			device.queue.submit([encoder.finish()]);
		}
		async function pollReadback(){
			await pollStats();
			await pollBuoy();
			await pollProbe();
			await pollPhaseEvents();
			if(prof)await prof.poll();
		}
		function getProfile(){
			return prof?prof.results:null;
		}
		function onPhaseChange(cb){
			if(typeof cb!=="function"){
				phaseChangeCb=null;
				phasePending=false;
				return false;
			}
			if(!phaseChangeCb&&frameCount>0){
				phaseSeedNeeded=true;
			}
			phaseChangeCb=cb;
			return true;
		}
		async function pollPhaseEvents(){
			if(!phasePending||phaseInFlight)return;
			phasePending=false;
			phaseInFlight=true;
			const readPool=pool;
			const readGeneration=phaseGeneration;
			const readFrame=phaseFrame;
			let fired=false;
			try{
				await readPool.phaseEventsRead.mapAsync(GPUMapMode.READ);
				if(readGeneration===statsGeneration&&phaseEventMod&&phaseEventState){
					phaseEventRaw.set(new Float32Array(readPool.phaseEventsRead.getMappedRange(),0,phaseEventRaw.length));
					fired=phaseEventMod.ingest(phaseEventState,phaseEventRaw,readFrame);
				}
				readPool.phaseEventsRead.unmap();
			}catch(e){
				console.error("phaseEvents readback failed",e);
			}
			phaseInFlight=false;
			if(fired&&phaseChangeCb)phaseChangeCb(phaseEventState.batch);
		}
		async function pollBuoy(){
			if(!buoyPending||buoyInFlight)return;
			buoyPending=false;
			buoyInFlight=true;
			const readPool=pool;
			const readGeneration=buoyGeneration;
			try{
				await readPool.buoyRead.mapAsync(GPUMapMode.READ);
				if(readGeneration===statsGeneration){
					buoyRaw.set(new Float32Array(readPool.buoyRead.getMappedRange(),0,buoyRaw.length));
					buoy.ingest(buoyRaw);
				}
				readPool.buoyRead.unmap();
			}catch(e){
				console.error("buoy readback failed",e);
			}
			buoyInFlight=false;
		}
		async function pollProbe(){
			if(!probePending||probeInFlight)return;
			probePending=false;
			probeInFlight=true;
			const readPool=pool;
			const readGeneration=probeGeneration;
			const readFrame=probeFrame;
			try{
				await readPool.probesRead.mapAsync(GPUMapMode.READ);
				if(readGeneration===statsGeneration){
					probeRaw.set(new Float32Array(readPool.probesRead.getMappedRange(),0,probeRaw.length));
					coreProbes.ingest(probeTable,probeRaw,readFrame);
				}
				readPool.probesRead.unmap();
			}catch(e){
				console.error("probe readback failed",e);
			}
			probeInFlight=false;
		}
		async function pollStats(){
			if(!pendingRead||readbackInFlight)return;
			readbackInFlight=true;
			pendingRead=false;
			const readPool=pool;
			const readGeneration=pendingGeneration;
			const readFrame=pendingFrame;
			const expM=pendingExpected.m,expDr=pendingExpected.dr,expDg=pendingExpected.dg;
			const expDb=pendingExpected.db,expE=pendingExpected.E,expHeatNet=pendingHeatNet;
			const expEmitM=pendingEmitNet.m,expEmitDr=pendingEmitNet.dr,expEmitDg=pendingEmitNet.dg;
			const expEmitDb=pendingEmitNet.db,expEmitE=pendingEmitNet.E;
			try{
				await readPool.statsRead.mapAsync(GPUMapMode.READ);
				const buf=readPool.statsRead.getMappedRange();
				if(readGeneration===statsGeneration){
					readbackArray.set(new Float32Array(buf));
					statsCache.frame=readFrame;
					statsCache.totals.m=readbackArray[0];
					statsCache.totals.dr=readbackArray[1];
					statsCache.totals.dg=readbackArray[2];
					statsCache.totals.db=readbackArray[3];
					statsCache.totals.E=readbackArray[4];
					const eInjM=readbackArray[stats.IDX_EMIT_INJ_M],eInjDr=readbackArray[stats.IDX_EMIT_INJ_DR];
					const eInjDg=readbackArray[stats.IDX_EMIT_INJ_DG],eInjDb=readbackArray[stats.IDX_EMIT_INJ_DB];
					const eInjE=readbackArray[stats.IDX_EMIT_INJ_E];
					const eRemM=readbackArray[stats.IDX_EMIT_REM_M],eRemDr=readbackArray[stats.IDX_EMIT_REM_DR];
					const eRemDg=readbackArray[stats.IDX_EMIT_REM_DG],eRemDb=readbackArray[stats.IDX_EMIT_REM_DB];
					const eRemE=readbackArray[stats.IDX_EMIT_REM_E];
					const eNetM=eInjM-eRemM,eNetDr=eInjDr-eRemDr,eNetDg=eInjDg-eRemDg;
					const eNetDb=eInjDb-eRemDb,eNetE=eInjE-eRemE;
					statsCache.expected.m=expM-expEmitM+eNetM;
					statsCache.expected.dr=expDr-expEmitDr+eNetDr;
					statsCache.expected.dg=expDg-expEmitDg+eNetDg;
					statsCache.expected.db=expDb-expEmitDb+eNetDb;
					statsCache.mins.m=readbackArray[5];
					statsCache.mins.dr=readbackArray[6];
					statsCache.mins.dg=readbackArray[7];
					statsCache.mins.db=readbackArray[8];
					statsCache.mins.E=readbackArray[9];
					statsCache.maxM=readbackArray[10];
					statsCache.cntNeg=readbackArray[11];
					statsCache.cntOver=readbackArray[12];
					statsCache.motion.maxVel=readbackArray[13];
					const pMin=readbackArray[14],pMax=readbackArray[15];
					statsCache.motion.pMin=pMin;
					statsCache.motion.pMax=pMax;
					statsCache.motion.maxDiv=readbackArray[16];
					const span=Math.max(Math.abs(pMin),Math.abs(pMax),1);
					pSpanSmooth=span>pSpanSmooth?span:(pSpanSmooth*0.85+span*0.15);
					cfg.PMin=-pSpanSmooth;
					cfg.PMax=pSpanSmooth;
					const heatIn=readbackArray[17],heatOut=readbackArray[18];
					statsCache.expected.E=expE-expEmitE+eNetE-expHeatNet+(heatIn-heatOut);
					heatNetKnown=heatIn-heatOut;
					ledger.heatInE=heatIn;ledger.heatOutE=heatOut;
					ledger.injectedM+=eInjM-emitInjKnown.m;emitInjKnown.m=eInjM;
					ledger.injectedDr+=eInjDr-emitInjKnown.dr;emitInjKnown.dr=eInjDr;
					ledger.injectedDg+=eInjDg-emitInjKnown.dg;emitInjKnown.dg=eInjDg;
					ledger.injectedDb+=eInjDb-emitInjKnown.db;emitInjKnown.db=eInjDb;
					ledger.injectedE+=eInjE-emitInjKnown.E;emitInjKnown.E=eInjE;
					ledger.removedM+=eRemM-emitRemKnown.m;emitRemKnown.m=eRemM;
					ledger.removedDr+=eRemDr-emitRemKnown.dr;emitRemKnown.dr=eRemDr;
					ledger.removedDg+=eRemDg-emitRemKnown.dg;emitRemKnown.dg=eRemDg;
					ledger.removedDb+=eRemDb-emitRemKnown.db;emitRemKnown.db=eRemDb;
					ledger.removedE+=eRemE-emitRemKnown.E;emitRemKnown.E=eRemE;
					emitNetKnown.m=eNetM;emitNetKnown.dr=eNetDr;emitNetKnown.dg=eNetDg;
					emitNetKnown.db=eNetDb;emitNetKnown.E=eNetE;
					cfg.rhoMean=(ledger.initialM+ledger.injectedM-ledger.removedM)/(cfg.N*cfg.N);
					statsCache.thermal.TMin=readbackArray[19];
					statsCache.thermal.TMax=readbackArray[20];
					statsCache.thermal.TMean=readbackArray[21];
					statsCache.thermal.kinetic=readbackArray[22];
					statsCache.thermal.heatIn=heatIn;
					statsCache.thermal.heatOut=heatOut;
					for(let k=0;k<6;k++)phaseCounts[k]=Math.round(readbackArray[23+k]);
					diag.sSaturated=Math.round(readbackArray[29]);
					// solve residual is slots 32/33, captured before transport. slots 34/35
					// are the same pressure after transport and phase: coefficient drift.
					const poissonRes=readbackArray[stats.IDX_POISSON_RES],poissonRhs=readbackArray[stats.IDX_POISSON_RHS];
					const poissonDrift=readbackArray[stats.IDX_POISSON_DRIFT];
					diag.poissonRes=poissonRes;
					diag.poissonRhs=poissonRhs;
					diag.poissonRel=poissonRhs>0?poissonRes/poissonRhs:0;
					diag.poissonDrift=poissonDrift;
					const poissonDriftRhs=readbackArray[stats.IDX_POISSON_DRIFT_RHS];
					diag.poissonDriftRhs=poissonDriftRhs;
					// same formula as the pre-split true column: post-step residual
					// over the post-step rhs, not the solve rhs.
					diag.poissonDriftRel=poissonDriftRhs>0?poissonDrift/poissonDriftRhs:0;
					ledger.latentChargedE=readbackArray[30];
					ledger.frozenMomentum=readbackArray[31];
				}
				readPool.statsRead.unmap();
			}catch(e){
				console.error("stats readback failed",e);
			}
			readbackInFlight=false;
		}
		function getStats(){
			statsView.frame=frameCount;
			statsView.statsFrame=statsCache.frame;
			statsView.totals=statsCache.totals;
			statsView.expected=statsCache.expected;
			statsView.mins=statsCache.mins;
			statsView.maxM=statsCache.maxM;
			statsView.cntNeg=statsCache.cntNeg;
			statsView.cntOver=statsCache.cntOver;
			statsView.motion=statsCache.motion;
			statsView.thermal=statsCache.thermal;
			statsView.phaseCounts=phaseCounts;
			statsView.diag=diag;
			statsView.ledger=ledger;
			return statsView;
		}
		async function serialize(out,extra){
			if(destroyed)throw new Error("serialize: solver is destroyed");
			const saveMod=Fluid.core.save;
			if(!saveMod)throw new Error("Fluid.core.save module not loaded");
			const cfgSnap=saveMod.cfgState(cfg);
			const snapLay=saveMod.layout(cfgSnap.N,cfgSnap.mgCoarsen);
			const stageLay=saveMod.stagingLayout(cfgSnap.N,cfgSnap.mgCoarsen);
			const outBuf=saveMod.pack({
				cfg,
				cfgSnap,
				frame:frameCount,
				time:timeAcc,
				pSpanSmooth,
				paintInjM,
				heatNetKnown,
				accumulator:(extra&&extra.accumulator)||0,
				simSubsteps:(extra&&extra.simSubsteps)||cfg.substeps||0,
				emitInjKnown,
				emitRemKnown,
				emitNetKnown,
				statsCache,
				phaseCounts,
				diag,
				buoy,
				ledger,
				probes:probeTable,
				emitters:emitterTable,
				colliders:colliderTable,
				fields:{flagsMirror}
			},out);
			const texMap={
				mass:pool.mass.tex[state.m],
				therm:pool.therm.tex[state.t],
				vel:pool.vel.tex[state.v],
				press:pool.press.tex[0],
				flags:pool.flags,
				sfield:pool.sfield,
				accum:pool.accum,
				mgPress:pool.mgPress.tex[0]
			};
			const staging=device.createBuffer({
				label:"saveStaging",
				size:stageLay.totalBytes,
				usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST
			});
			try{
				const encoder=device.createCommandEncoder({label:"serialize"});
				for(let i=0;i<stageLay.list.length;i++){
					const e=stageLay.list[i];
					encoder.copyTextureToBuffer(
						{texture:texMap[e.name]},
						{buffer:staging,offset:e.offset,bytesPerRow:e.bytesPerRow,rowsPerImage:e.h},
						[e.w,e.h,1]
					);
				}
				encoder.copyBufferToBuffer(pool.emitterLedger,0,staging,stageLay.emitterLedgerOffset,stageLay.emitterLedgerBytes);
				device.queue.submit([encFinish(encoder)]);
				await staging.mapAsync(GPUMapMode.READ);
				const mapped=staging.getMappedRange();
				saveMod.repackStagingInto(mapped,stageLay,outBuf,snapLay);
				staging.unmap();
			}finally{
				try{staging.destroy();}catch(e){}
			}
			return outBuf;
		}
		function encFinish(encoder){
			return encoder.finish();
		}
		function deserialize(buf){
			if(destroyed)throw new Error("deserialize: solver is destroyed");
			const saveMod=Fluid.core.save;
			if(!saveMod)throw new Error("Fluid.core.save module not loaded");
			const snap=saveMod.unpack(buf,cfg);
			statsGeneration++;
			if(prof)prof.invalidate();
			pendingRead=false;
			buoyPending=false;
			buoyGeneration=statsGeneration;
			probePending=false;
			phasePending=false;
			phaseGeneration=statsGeneration;
			phaseSeedNeeded=false;
			const w=cfg.texW,h=cfg.texH;
			const f=snap.fields;
			fields.writeTextureRect(device,pool.mass.tex[0],w,h,"rgba32float",f.mass);
			fields.writeTextureRect(device,pool.mass.tex[1],w,h,"rgba32float",f.mass);
			fields.writeTextureRect(device,pool.mass.tex[2],w,h,"rgba32float",f.mass);
			fields.writeTextureRect(device,pool.therm.tex[0],w,h,"rgba32float",f.therm);
			fields.writeTextureRect(device,pool.therm.tex[1],w,h,"rgba32float",f.therm);
			fields.writeTextureRect(device,pool.vel.tex[0],w,h,"rg32float",f.vel);
			fields.writeTextureRect(device,pool.vel.tex[1],w,h,"rg32float",f.vel);
			fields.clearTexture(device,pool.macTmp,w,h,"rg32float");
			fields.writeTextureRect(device,pool.press.tex[0],w,h,"r32float",f.press);
			fields.writeTextureRect(device,pool.press.tex[1],w,h,"r32float",f.press);
			fields.writeTextureRect(device,pool.flags,w,h,"rgba8uint",f.flags);
			fields.writeTextureRect(device,pool.flagsTmp,w,h,"rgba8uint",f.flags);
			fields.writeTextureRect(device,pool.sfield,w,h,"rgba32float",f.sfield);
			fields.writeTextureRect(device,pool.sfieldTmp,w,h,"rgba32float",f.sfield);
			fields.writeTextureRect(device,pool.accum,w,h,"rg32float",f.accum);
			fields.writeTextureRect(device,pool.accumTmp,w,h,"rg32float",f.accum);
			fields.clearTexture(device,pool.flux,w,h,"rg32float");
			fields.clearTexture(device,pool.limiterA,w,h,"rgba32float");
			fields.clearTexture(device,pool.limiterB,w,h,"rg32float");
			fields.clearTexture(device,pool.divsrc,w,h,"r32float");
			fields.writeTextureRect(device,pool.mgPress.tex[0],pool.mgTexW,pool.mgTexH,"r32float",f.mgPress);
			fields.writeTextureRect(device,pool.mgPress.tex[1],pool.mgTexW,pool.mgTexH,"r32float",f.mgPress);
			flagsMirror.set(f.flagsMirror);
			state.m=0;state.t=0;state.v=0;
			const st=snap.state;
			frameCount=Math.round(st[saveMod.ST_FRAME])||0;
			timeAcc=snap.time;
			cfg.frame=frameCount;
			cfg.time=timeAcc;
			cfg.dt=snap.dt;
			cfg.substeps=Math.round(st[saveMod.ST_SUBSTEPS])||1;
			cfg.vClamp=st[saveMod.ST_VCLAMP]||cfg.vClamp;
			cfg.rhoMean=st[saveMod.ST_RHO_MEAN]||0;
			cfg.rhoBase=st[saveMod.ST_RHO_BASE]||0;
			cfg.PMin=st[saveMod.ST_PMIN];
			cfg.PMax=st[saveMod.ST_PMAX];
			pSpanSmooth=st[saveMod.ST_PSPAN]||100;
			paintInjM=st[saveMod.ST_PAINT_INJ_M]||0;
			heatNetKnown=st[saveMod.ST_HEAT_NET]||0;
			pendingHeatNet=heatNetKnown;
			cfg.mouseX=st[saveMod.ST_MOUSE_X];
			cfg.mouseY=st[saveMod.ST_MOUSE_Y];
			cfg.mouseStrength=st[saveMod.ST_MOUSE_STR];
			cfg.mouseVX=st[saveMod.ST_MOUSE_VX];
			cfg.mouseVY=st[saveMod.ST_MOUSE_VY];
			const keys=["m","dr","dg","db","E"];
			for(let i=0;i<5;i++){
				const k=keys[i];
				emitInjKnown[k]=st[saveMod.ST_EMIT_INJ+i]||0;
				emitRemKnown[k]=st[saveMod.ST_EMIT_REM+i]||0;
				emitNetKnown[k]=st[saveMod.ST_EMIT_NET+i]||0;
				pendingEmitNet[k]=emitNetKnown[k];
				statsCache.totals[k]=st[saveMod.ST_TOT+i]||0;
				statsCache.expected[k]=st[saveMod.ST_EXP+i]||0;
				statsCache.mins[k]=st[saveMod.ST_MIN+i]||0;
			}
			statsCache.frame=Math.round(st[saveMod.ST_STATS_FRAME])||0;
			statsCache.maxM=st[saveMod.ST_MAX_M]||0;
			statsCache.cntNeg=st[saveMod.ST_CNT_NEG]||0;
			statsCache.cntOver=st[saveMod.ST_CNT_OVER]||0;
			statsCache.motion.maxVel=st[saveMod.ST_MOT]||0;
			statsCache.motion.pMin=st[saveMod.ST_MOT+1]||0;
			statsCache.motion.pMax=st[saveMod.ST_MOT+2]||0;
			statsCache.motion.maxDiv=st[saveMod.ST_MOT+3]||0;
			statsCache.thermal.TMin=st[saveMod.ST_THERM]||0;
			statsCache.thermal.TMax=st[saveMod.ST_THERM+1]||0;
			statsCache.thermal.TMean=st[saveMod.ST_THERM+2]||0;
			statsCache.thermal.kinetic=st[saveMod.ST_THERM+3]||0;
			statsCache.thermal.heatIn=st[saveMod.ST_THERM+4]||0;
			statsCache.thermal.heatOut=st[saveMod.ST_THERM+5]||0;
			for(let k=0;k<6;k++)phaseCounts[k]=Math.round(st[saveMod.ST_PHASE_COUNTS+k])||0;
			diag.sSaturated=Math.round(st[saveMod.ST_DIAG])||0;
			diag.poissonRes=st[saveMod.ST_DIAG+1]||0;
			diag.poissonRhs=st[saveMod.ST_DIAG+2]||0;
			diag.poissonRel=st[saveMod.ST_DIAG+3]||0;
			diag.poissonDrift=st[saveMod.ST_DIAG+4]||0;
			diag.poissonDriftRhs=st[saveMod.ST_DIAG+5]||0;
			diag.poissonDriftRel=st[saveMod.ST_DIAG+6]||0;
			Object.assign(ledger,snap.ledger);
			saveMod.restoreTables(snap,cfg,probeTable,emitterTable,colliderTable,buoy);
			uploadProbes();
			if(emitterTable)uploadEmitters();
			if(colliderTable)uploadColliders();
			device.queue.writeBuffer(pool.emitterLedger,0,snap.tables.emitLedger);
			device.queue.writeBuffer(pool.buoy,0,buoy.sample);
			if(phaseEventMod&&phaseEventState){
				phaseEventMod.seedFromFlags(phaseEventState,f.flags,w,h);
				device.queue.writeBuffer(pool.phasePrev,0,phaseEventState.prev);
			}
			writeMgParams();
			writeParams();
			return snap;
		}
		function resize(newN){
			fields.destroyPool(pool);
			cfg.N=newN;
			cfg.texW=newN+2;
			cfg.texH=newN+2;
			pool=fields.createPool(device,newN,cfg.mgCoarsen);
			groups=passes.makeBindGroups(device,pipes,pool);
			flagsMirror=new Uint8Array(cfg.texW*cfg.texH*4);
			if(phaseEventMod&&phaseEventState)phaseEventMod.resize(phaseEventState,cfg.texW,cfg.texH);
			canvas.width=newN;
			canvas.height=newN;
			api.pool=pool;
			api.flagsMirror=flagsMirror;
			coreProbes.syncConfig(probeTable,cfg);
			if(emitterTable)coreEmitters.syncConfig(emitterTable,cfg,true);
			if(colliderTable)coreColliders.syncConfig(colliderTable,cfg,true);
			reset();
			renderOnly();
		}
		function destroy(){
			destroyed=true;
			phaseChangeCb=null;
			fields.destroyPool(pool);
			if(prof)prof.destroy();
		}
		reset();
		const api={pool,device,canvas,context,format,cfg,state,reset,frame,renderOnly,paintDye,paintWall,setObstacles,paintObstacle,pollReadback,getStats,getProfile,buoy,resize,destroy,flagsMirror,
			phaseEvents:phaseEventState,onPhaseChange,serialize,deserialize,
			probes:probeTable,addProbe,moveProbe,removeProbe,sampleProbe,
			emitters:emitterTable,addEmitter,moveEmitter,setEmitter,removeEmitter,emitterCount,
			colliders:colliderTable,addCollider,moveCollider,setCollider,removeCollider,colliderCount};
		Object.defineProperty(api,"saveSize",{
			get(){return Fluid.core.save?Fluid.core.save.byteLength(cfg.N,cfg.mgCoarsen):0;}
		});
		return api;
	}
	Fluid.gpu.solver={create:createSolver};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.solver;
})(globalThis);
