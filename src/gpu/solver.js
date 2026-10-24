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
		const state={m:0,t:0,v:0,pw:0,ph:0};
		let flagsMirror=new Uint8Array(cfg.texW*cfg.texH*4);
		let ledger=null;
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
			cfg.chargeLatent=charge?1:0;
			writeParams();
			const encoder=device.createCommandEncoder({label:"primePhase"});
			const cpass=encoder.beginComputePass({label:"primePhase"});
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
			flagsMirror.set(sc.flagsA);
			state.m=0;state.t=0;state.v=0;
			heatNetKnown=0;pendingHeatNet=0;
			pSpanSmooth=100;
			cfg.PMin=-100;cfg.PMax=100;
			const init=stats.totalsFromFields(sc.massA,sc.thermA,cfg.texW,cfg.texH);
			ledger=stats.createLedger(init);
			cfg.rhoMean=init.sumM/(cfg.N*cfg.N);
			frameCount=0;
			timeAcc=0;
			for(let k=0;k<6;k++)phaseCounts[k]=0;
			for(let y=1;y<cfg.texH-1;y++){
				for(let x=1;x<cfg.texW-1;x++){
					const i=(x+y*cfg.texW)*4;
					phaseCounts[coreSolver.phaseOf(sc.massA[i+3],sc.thermA[i],cfg).phase]++;
				}
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
			runPrimePhase(false);
			writeParams();
			// a resize rebuilds the pool: the new probe buffer starts zeroed, so the
			// live table has to be pushed again or every query silently stops
			uploadProbes();
		}
		function uploadProbes(){
			device.queue.writeBuffer(pool.probes,0,probeTable.upload);
		}
		function addProbe(x,y){
			const id=coreProbes.add(probeTable,cfg,x,y);
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
			const t=coreSolver.patchTotals(patch);
			ledger.injectedM+=t.m;ledger.injectedDr+=t.dr;ledger.injectedDg+=t.dg;ledger.injectedDb+=t.db;ledger.injectedE+=t.E;
			cfg.rhoMean=(ledger.initialM+ledger.injectedM-ledger.removedM)/(cfg.N*cfg.N);
		}
		function paintWall(u,v,erase){
			const patch=coreSolver.buildWallPatch(cfg,u,v,erase);
			if(!patch)return;
			for(let y=0;y<patch.h;y++){
				for(let x=0;x<patch.w;x++){
					const gi=((patch.x0+x)+(patch.y0+y)*cfg.texW)*4;
					const pi=(x+y*patch.w)*4;
					flagsMirror[gi]=patch.flagsPatch[pi];
				}
			}
			fields.writeTextureRect(device,pool.flags,cfg.texW,cfg.texH,"rgba8uint",flagsMirror);
			fields.writeTextureRect(device,pool.flagsTmp,cfg.texW,cfg.texH,"rgba8uint",flagsMirror);
			runPrimePhase(false);
		}
		function frame(dts,substeps,needStats){
			buoy.march(dts*substeps);
			const needBuoy=buoy.state.active&&(frameCount%Math.max(1,cfg.buoySampleEvery)===0);
			const needProbe=probeTable.live>0&&(frameCount%Math.max(1,cfg.probeEvery)===0);
			cfg.dt=dts;
			cfg.substeps=substeps;
			cfg.frame=frameCount;
			cfg.time=timeAcc;
			cfg.chargeLatent=1;
			cfg.rhoMean=(ledger.initialM+ledger.injectedM-ledger.removedM)/(cfg.N*cfg.N);
			writeParams();
			// profiling splits the one compute pass at timing-group boundaries;
			// the dispatch sequence is identical either way (0.6.0 t1).
			const kIters=Fluid.core.config.solverIters(cfg);
			const profiling=!!(prof&&cfg.gpuTiming>0);
			const encoder=device.createCommandEncoder({label:"frame"});
			if(profiling){
				prof.beginFrame(substeps,frameCount+1);
				for(let s=0;s<substeps;s++){
					passes.encodeStepProfiled(encoder,prof,pool,pipes,groups,state,kIters,cfg.fctLimiter>0,
						!!cfg.useMultigrid,cfg.mgPre||4,cfg.mgPost||8,cfg.mgCycles||1,
						needStats&&s===substeps-1,cfg.advectMac>0,cfg.fusePasses>0);
				}
				if(needStats||needBuoy||needProbe){
					const cp=prof.pass(encoder,PROFILE_GROUP.stats,"stats");
					if(needStats)passes.encodeStats(cp,pool,pipes,groups,state);
					if(needBuoy)passes.encodeBuoySample(cp,pool,pipes,groups,state);
					if(needProbe)passes.encodeProbeSample(cp,pool,pipes,groups,state);
				}
				passes.encodeDisplay(prof.pass(encoder,PROFILE_GROUP.disp,"display"),pool,pipes,groups,state);
				prof.closePass();
			}else{
				const cpass=encoder.beginComputePass({label:"sim"});
				for(let s=0;s<substeps;s++){
					passes.encodeStep(cpass,pool,pipes,groups,state,kIters,cfg.fctLimiter>0,
						!!cfg.useMultigrid,cfg.mgPre||4,cfg.mgPost||8,cfg.mgCycles||1,
						needStats&&s===substeps-1,cfg.advectMac>0,cfg.fusePasses>0);
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
				passes.encodeDisplay(cpass,pool,pipes,groups,state);
				cpass.end();
			}
			if(needStats&&!readbackInFlight){
				encoder.copyBufferToBuffer(pool.stats,0,pool.statsRead,0,stats.STATS_LEN*4);
				stats.writeExpectedTotals(pendingExpected,ledger);
				pendingHeatNet=heatNetKnown;
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
		}
		function renderOnly(){
			writeParams();
			const encoder=device.createCommandEncoder({label:"renderOnly"});
			const cpass=encoder.beginComputePass({label:"display"});
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
			if(prof)await prof.poll();
		}
		function getProfile(){
			return prof?prof.results:null;
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
					statsCache.expected.m=expM;
					statsCache.expected.dr=expDr;
					statsCache.expected.dg=expDg;
					statsCache.expected.db=expDb;
					statsCache.expected.E=expE;
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
					statsCache.expected.E=expE-expHeatNet+(heatIn-heatOut);
					heatNetKnown=heatIn-heatOut;
					ledger.heatInE=heatIn;ledger.heatOutE=heatOut;
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
		function resize(newN){
			fields.destroyPool(pool);
			cfg.N=newN;
			cfg.texW=newN+2;
			cfg.texH=newN+2;
			pool=fields.createPool(device,newN,cfg.mgCoarsen);
			groups=passes.makeBindGroups(device,pipes,pool);
			flagsMirror=new Uint8Array(cfg.texW*cfg.texH*4);
			canvas.width=newN;
			canvas.height=newN;
			api.pool=pool;
			api.flagsMirror=flagsMirror;
			reset();
			renderOnly();
		}
		function destroy(){
			fields.destroyPool(pool);
			if(prof)prof.destroy();
		}
		reset();
		const api={pool,device,canvas,context,format,cfg,state,reset,frame,renderOnly,paintDye,paintWall,pollReadback,getStats,getProfile,buoy,resize,destroy,flagsMirror,
			probes:probeTable,addProbe,moveProbe,removeProbe,sampleProbe};
		return api;
	}
	Fluid.gpu.solver={create:createSolver};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.solver;
})(globalThis);
