(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	function createSolver(device,canvas,context,format,cfg){
		const fields=Fluid.gpu.fields;
		const passes=Fluid.gpu.passes;
		const stats=Fluid.core.stats;
		const coreSolver=Fluid.core.solver;
		let pool=fields.createPool(device,cfg.N);
		let pipes=passes.createPipelines(device,format);
		let groups=passes.makeBindGroups(device,pipes,pool);
		const paramsArray=Fluid.core.config.makeParamsArray();
		// (m, v) ping-pong parity; press parity is invariant (red+black returns to tex[0])
		const state={m:0,v:0,pw:0,ph:0};
		const flagsMirror=new Uint8Array(cfg.texW*cfg.texH*4);
		let ledger=null;
		const readbackArray=new Float32Array(32);
		let readbackInFlight=false;
		let pendingRead=false;
		let frameCount=0;
		let timeAcc=0;
		let statsCache=null;
		function writeParams(){
			Fluid.core.config.writeParams(paramsArray,cfg);
			device.queue.writeBuffer(pool.params,0,paramsArray);
		}
		function reset(){
			const sc=coreSolver.buildScenario(cfg);
			fields.writeTextureRect(device,pool.mass.tex[0],cfg.texW,cfg.texH,"rgba32float",sc.massA);
			fields.writeTextureRect(device,pool.mass.tex[1],cfg.texW,cfg.texH,"rgba32float",sc.massA);
			fields.writeTextureRect(device,pool.therm.tex[0],cfg.texW,cfg.texH,"rgba32float",sc.thermA);
			fields.writeTextureRect(device,pool.therm.tex[1],cfg.texW,cfg.texH,"rgba32float",sc.thermA);
			fields.clearTexture(device,pool.vel.tex[0],cfg.texW,cfg.texH,"rg32float");
			fields.clearTexture(device,pool.vel.tex[1],cfg.texW,cfg.texH,"rg32float");
			fields.clearTexture(device,pool.press.tex[0],cfg.texW,cfg.texH,"r32float");
			fields.clearTexture(device,pool.press.tex[1],cfg.texW,cfg.texH,"r32float");
			fields.clearTexture(device,pool.flags,cfg.texW,cfg.texH,"rgba8uint");
			fields.clearTexture(device,pool.flux,cfg.texW,cfg.texH,"rg32float");
			fields.clearTexture(device,pool.divsrc,cfg.texW,cfg.texH,"r32float");
			flagsMirror.set(sc.flagsA);
			state.m=0;state.v=0;
			const init=stats.totalsFromFields(sc.massA,sc.thermA,cfg.texW,cfg.texH);
			ledger=stats.createLedger(init);
			cfg.rhoMean=init.sumM/(cfg.N*cfg.N);
			frameCount=0;
			timeAcc=0;
			statsCache={
				totals:{m:init.sumM,dr:init.sumDr,dg:init.sumDg,db:init.sumDb,E:init.sumE},
				mins:{m:init.minM,dr:init.minDr,dg:init.minDg,db:init.minDb,E:init.minE},
				maxM:init.maxM,cntNeg:init.cntNeg,cntOver:init.cntOver,
				motion:{maxVel:0,pMin:0,pMax:0,maxDiv:0}
			};
			writeParams();
		}
		// dye brush: the gpu adds exactly these floats and the ledger sums the
		// same floats, so injection accounting is exact by construction
		function paintDye(u,v,rgb){
			const patch=coreSolver.buildDyePatch(cfg,u,v,rgb);
			if(!patch)return;
			if(patch.w>fields.PATCH||patch.h>fields.PATCH)return;
			fields.writePatch(device,pool.patchMass,patch.w,patch.h,"rgba32float",patch.massPatch);
			fields.writePatch(device,pool.patchE,patch.w,patch.h,"r32float",patch.ePatch);
			cfg.paintX=patch.x0;cfg.paintY=patch.y0;cfg.paintW=patch.w;cfg.paintH=patch.h;
			writeParams();
			const encoder=device.createCommandEncoder({label:"paint"});
			const cpass=encoder.beginComputePass({label:"paint"});
			passes.encodePaint(cpass,pool,pipes,groups,state);
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
			// full texture upload from the authoritative cpu mirror (paint is rare)
			fields.writeTextureRect(device,pool.flags,cfg.texW,cfg.texH,"rgba8uint",flagsMirror);
		}
		// one frame: optional substeps, optional stats, always display + render
		function frame(dts,substeps,needStats){
			cfg.dt=dts;
			cfg.substeps=substeps;
			cfg.frame=frameCount;
			cfg.time=timeAcc;
			cfg.rhoMean=(ledger.initialM+ledger.injectedM-ledger.removedM)/(cfg.N*cfg.N);
			writeParams();
			const encoder=device.createCommandEncoder({label:"frame"});
			const cpass=encoder.beginComputePass({label:"sim"});
			for(let s=0;s<substeps;s++){
				passes.encodeStep(cpass,pool,pipes,groups,state,cfg.pressureIters);
			}
			if(needStats){
				passes.encodeStats(cpass,pool,pipes,groups,state);
			}
			passes.encodeDisplay(cpass,pool,pipes,groups,state);
			cpass.end();
			if(needStats&&!readbackInFlight){
				encoder.copyBufferToBuffer(pool.stats,0,pool.statsRead,0,128);
				pendingRead=true;
			}
			const view=context.getCurrentTexture().createView();
			const rpass=encoder.beginRenderPass({
				label:"render",
				colorAttachments:[{view,loadOp:"clear",clearValue:{r:0.02,g:0.03,b:0.04,a:1},storeOp:"store"}]
			});
			passes.encodeRender(rpass,pipes,groups);
			rpass.end();
			device.queue.submit([encoder.finish()]);
			frameCount++;
			timeAcc+=dts*substeps;
		}
		function renderOnly(){
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
		async function pollStats(){
			if(!pendingRead||readbackInFlight)return;
			readbackInFlight=true;
			pendingRead=false;
			try{
				await pool.statsRead.mapAsync(GPUMapMode.READ);
				const buf=pool.statsRead.getMappedRange();
				readbackArray.set(new Float32Array(buf));
				pool.statsRead.unmap();
				statsCache={
					totals:{m:readbackArray[0],dr:readbackArray[1],dg:readbackArray[2],db:readbackArray[3],E:readbackArray[4]},
					mins:{m:readbackArray[5],dr:readbackArray[6],dg:readbackArray[7],db:readbackArray[8],E:readbackArray[9]},
					maxM:readbackArray[10],cntNeg:readbackArray[11],cntOver:readbackArray[12],
					motion:{maxVel:readbackArray[13],pMin:readbackArray[14],pMax:readbackArray[15],maxDiv:readbackArray[16]}
				};
			}catch(e){
				console.error("stats readback failed",e);
			}
			readbackInFlight=false;
		}
		function getStats(){
			return {
				frame:frameCount,
				totals:statsCache.totals,
				expected:{
					m:stats.expectedTotal(ledger,"m"),
					dr:stats.expectedTotal(ledger,"dr"),
					dg:stats.expectedTotal(ledger,"dg"),
					db:stats.expectedTotal(ledger,"db"),
					E:stats.expectedTotal(ledger,"E")
				},
				mins:statsCache.mins,
				maxM:statsCache.maxM,
				cntNeg:statsCache.cntNeg,
				cntOver:statsCache.cntOver,
				motion:statsCache.motion,
				ledger
			};
		}
		function resize(newN){
			fields.destroyPool(pool);
			cfg.N=newN;
			cfg.texW=newN+2;
			cfg.texH=newN+2;
			pool=fields.createPool(device,newN);
			pipes=passes.createPipelines(device,format);
			groups=passes.makeBindGroups(device,pipes,pool);
			flagsMirror=new Uint8Array(cfg.texW*cfg.texH*4);
			canvas.width=newN;
			canvas.height=newN;
			reset();
		}
		function destroy(){
			fields.destroyPool(pool);
		}
		reset();
		return {pool,device,canvas,context,format,cfg,state,reset,frame,renderOnly,paintDye,paintWall,pollStats,getStats,resize,destroy,flagsMirror};
	}
	Fluid.gpu.solver={create:createSolver};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.solver;
})(globalThis);
