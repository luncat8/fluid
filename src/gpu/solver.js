(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	function createSolver(device,canvas,context,format,cfg){
		let pool=Fluid.gpu.fields.createPool(device,cfg.N);
		let pipes=Fluid.gpu.passes.createPipelines(device,pool,format);
		let groups=Fluid.gpu.passes.makeBindGroups(device,pipes,pool);
		let paramsArray=Fluid.core.config.makeParamsArray();
		// ledger
		let ledger=null;
		let initialTotals=null;
		const statsArray=new Float32Array(16);
		const readbackArray=new Float32Array(16);
		let readbackInFlight=false;
		let frameCount=0;
		let timeAcc=0;
		let statsCache={sumM:0,sumDr:0,sumDg:0,sumDb:0,sumE:0,minM:0,maxM:0,cntNeg:0,cntOver:0};
		let pendingRead=false;
		function writeParams(){
			Fluid.core.config.writeParams(paramsArray,cfg);
			device.queue.writeBuffer(pool.params,0,paramsArray);
		}
		function buildInitialArrays(N){
			let texW=N+2, texH=N+2;
			let total=texW*texH*4;
			let massA=new Float32Array(total);
			let thermA=new Float32Array(total);
			let flagsA=new Uint8Array(total);
			let velA=new Float32Array(texW*texH*2);
			let cLiq=cfg.cLiq, TAmb=cfg.TAmb;
			// fill background with vacuum-like low density but for dye test we use 0.05 ambient + blobs
			for(let y=0;y<texH;y++){
				for(let x=0;x<texW;x++){
					let idx=(x+y*texW)*4;
					if(x>=1&&x<texW-1&&y>=1&&y<texH-1){
						massA[idx]=0;
						massA[idx+1]=0;
						massA[idx+2]=0;
						massA[idx+3]=0.05;
						let E=massA[idx+3]*cLiq*TAmb;
						thermA[idx]=E;
						thermA[idx+1]=TAmb;
						thermA[idx+2]=1;
						thermA[idx+3]=0;
					}else{
						massA[idx]=0; massA[idx+1]=0; massA[idx+2]=0; massA[idx+3]=0;
						thermA[idx]=0; thermA[idx+1]=TAmb; thermA[idx+2]=0; thermA[idx+3]=0;
					}
				}
			}
			function paintBlob(cx,cy,radius,dr,dg,db,dm){
				for(let y=1;y<texH-1;y++){
					for(let x=1;x<texW-1;x++){
						let dx=(x-1+0.5)-cx;
						let dy=(y-1+0.5)-cy;
						let d=Math.sqrt(dx*dx+dy*dy);
						if(d<radius){
							let fall=1-Math.pow(d/radius,2);
							fall=fall*fall;
							let idx=(x+y*texW)*4;
							massA[idx]+=dr*fall;
							massA[idx+1]+=dg*fall;
							massA[idx+2]+=db*fall;
							massA[idx+3]+=dm*fall;
							// update E to keep T constant
							thermA[idx]=massA[idx+3]*cLiq*TAmb;
						}
					}
				}
			}
			// three blobs: r,g,b
			let cx1=N*0.3, cy1=N*0.5, rad=N*0.12;
			let cx2=N*0.7, cy2=N*0.5;
			let cx3=N*0.5, cy3=N*0.25;
			paintBlob(cx1,cy1,rad,0.8,0,0,0.6);
			paintBlob(cx2,cy2,rad,0,0.8,0,0.6);
			paintBlob(cx3,cy3,rad,0,0,0.8,0.6);
			// extra mix check: ensure dye per mass bounded
			return {massA,thermA,flagsA,velA};
		}
		function uploadArrays(arrs){
			let texW=pool.texW, texH=pool.texH;
			Fluid.gpu.fields.writeTextureRect(device,pool.mass.tex[0],texW,texH,"rgba32float",arrs.massA);
			Fluid.gpu.fields.writeTextureRect(device,pool.mass.tex[1],texW,texH,"rgba32float",arrs.massA);
			Fluid.gpu.fields.writeTextureRect(device,pool.therm.tex[0],texW,texH,"rgba32float",arrs.thermA);
			Fluid.gpu.fields.writeTextureRect(device,pool.therm.tex[1],texW,texH,"rgba32float",arrs.thermA);
			// clear other textures
			Fluid.gpu.fields.clearTexture(device,pool.vel.tex[0],texW,texH,"rg32float");
			Fluid.gpu.fields.clearTexture(device,pool.vel.tex[1],texW,texH,"rg32float");
			Fluid.gpu.fields.clearTexture(device,pool.press.tex[0],texW,texH,"r32float");
			Fluid.gpu.fields.clearTexture(device,pool.press.tex[1],texW,texH,"r32float");
			Fluid.gpu.fields.clearTexture(device,pool.flags.tex[0],texW,texH,"rgba8uint");
			Fluid.gpu.fields.clearTexture(device,pool.flags.tex[1],texW,texH,"rgba8uint");
			Fluid.gpu.fields.clearTexture(device,pool.flux,texW,texH,"rg32float");
			Fluid.gpu.fields.clearTexture(device,pool.divsrc,texW,texH,"r32float");
			Fluid.gpu.fields.clearTexture(device,pool.sfield,texW,texH,"r32float");
			Fluid.gpu.fields.clearTexture(device,pool.limiter,texW,texH,"rg32float");
		}
		function computeInitialTotals(arrs){
			let texW=pool.texW, texH=pool.texH;
			let sumM=0,sumDr=0,sumDg=0,sumDb=0,sumE=0;
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					let i=(x+y*texW)*4;
					sumDr+=arrs.massA[i];
					sumDg+=arrs.massA[i+1];
					sumDb+=arrs.massA[i+2];
					sumM+=arrs.massA[i+3];
					sumE+=arrs.thermA[i];
				}
			}
			return {sumM,sumDr,sumDg,sumDb,sumE,minM:0};
		}
		function reset(scenario){
			let arrs=buildInitialArrays(cfg.N);
			uploadArrays(arrs);
			initialTotals=computeInitialTotals(arrs);
			ledger=Fluid.core.stats.createLedger(initialTotals);
			frameCount=0;
			timeAcc=0;
			// reset stats cache
			statsCache.sumM=initialTotals.sumM;
			statsCache.sumDr=initialTotals.sumDr;
			statsCache.sumDg=initialTotals.sumDg;
			statsCache.sumDb=initialTotals.sumDb;
			statsCache.sumE=initialTotals.sumE;
			// ensure pool pointers reset
			pool.mass.cur=0; pool.therm.cur=0; pool.vel.cur=0; pool.press.cur=0; pool.flags.cur=0;
			groups=Fluid.gpu.passes.makeBindGroups(device,pipes,pool);
		}
		function paintAt(u,v,radius,dr,dg,db,dm){
			// paint into current mass texture via CPU readback/write cycle using staging array
			// simpler: we keep CPU arrays for interaction and re-upload; but for interactive we can do GPU paint via compute? For 0.1.0, implement CPU side staged paint
			// we will maintain a CPU mirror of mass arrays for painting
			// For now, use readback path not implemented; instead we do a GPU-side dye paint via direct writeTexture of small region
			// Approach: create a small Float32Array patch and writeTexture partial? queue.writeTexture supports origin offset via texture origin
			// We'll implement per-drag patch upload
			let texW=pool.texW, texH=pool.texH;
			let cx=Math.floor(u * cfg.N)+1;
			let cy=Math.floor(v * cfg.N)+1;
			let rad=Math.max(1, Math.floor(radius));
			// build patch bounding box
			let x0=Math.max(1,cx-rad), x1=Math.min(texW-2,cx+rad);
			let y0=Math.max(1,cy-rad), y1=Math.min(texH-2,cy+rad);
			let w=x1-x0+1, h=y1-y0+1;
			let patch=new Float32Array(w*h*4);
			// read current mass via copy to buffer not implemented, so we approximate by adding directly via staging that adds to existing via additive blending?
			// For 0.1.0 interactive paint, we can just add dye via a compute pass that adds falloff directly on GPU without readback
			// Simpler: we perform a direct write that overwrites with additive value using a dedicated paint compute shader (not yet implemented)
			// As fallback, we implement additive paint by creating a temporary addition texture approach: we upload patch as addition and use a copy+add shader
			// For now, ignore interactive paint complexity and just note not implemented
		}
		function ensureStatsReadback(encoder){
			if(readbackInFlight) return;
			if(frameCount % cfg.statsEvery !== 0) return;
			encoder.copyBufferToBuffer(pool.stats,0,pool.statsRead,0,64);
			pendingRead=true;
		}
		async function pollStats(){
			if(!pendingRead||readbackInFlight) return;
			readbackInFlight=true;
			pendingRead=false;
			try{
				await pool.statsRead.mapAsync(GPUMapMode.READ);
				let buf=pool.statsRead.getMappedRange();
				let src=new Float32Array(buf.slice(0));
				pool.statsRead.unmap();
				statsArray.set(src);
				readbackArray.set(src);
				statsCache.sumM=statsArray[0];
				statsCache.sumDr=statsArray[1];
				statsCache.sumDg=statsArray[2];
				statsCache.sumDb=statsArray[3];
				statsCache.sumE=statsArray[4];
				statsCache.minM=statsArray[5];
				statsCache.minDr=statsArray[6];
				statsCache.minDg=statsArray[7];
				statsCache.minDb=statsArray[8];
				statsCache.minE=statsArray[9];
				statsCache.maxM=statsArray[10];
				statsCache.cntNeg=statsArray[11];
				statsCache.cntOver=statsArray[12];
			}catch(e){
				// ignore
			}
			readbackInFlight=false;
		}
		function step(dt){
			cfg.dt=dt;
			cfg.frame=frameCount;
			cfg.time=timeAcc;
			writeParams();
			groups=Fluid.gpu.passes.makeBindGroups(device,pipes,pool);
			const encoder=device.createCommandEncoder({label:"frame"});
			cfg._needStats=(frameCount % cfg.statsEvery===0);
			Fluid.gpu.passes.encodeCompute(encoder,device,pool,pipes,groups,cfg);
			ensureStatsReadback(encoder);
			// render pass
			const view=context.getCurrentTexture().createView();
			const rpass=encoder.beginRenderPass({
				label:"render",
				colorAttachments:[{view,loadOp:"clear",clearValue:{r:0.05,g:0.06,b:0.08,a:1},storeOp:"store"}]
			});
			Fluid.gpu.passes.encodeRender(rpass,pool,pipes,groups);
			rpass.end();
			device.queue.submit([encoder.finish()]);
			// swap ping pong after submit (logically)
			pool.mass.cur=1-pool.mass.cur;
			// therm swap not needed but keep consistent for future
			frameCount++;
			timeAcc+=dt;
			if(pendingRead){
				// trigger async poll next frame
				setTimeout(()=>pollStats(),0);
			}
		}
		function getStats(){
			let ledgerExpM=ledger?Fluid.core.stats.expectedTotal(ledger,"m"):statsCache.sumM;
			let ledgerExpDr=ledger?Fluid.core.stats.expectedTotal(ledger,"dr"):statsCache.sumDr;
			let ledgerExpDg=ledger?Fluid.core.stats.expectedTotal(ledger,"dg"):statsCache.sumDg;
			let ledgerExpDb=ledger?Fluid.core.stats.expectedTotal(ledger,"db"):statsCache.sumDb;
			let ledgerExpE=ledger?Fluid.core.stats.expectedTotal(ledger,"E"):statsCache.sumE;
			return {
				frame:frameCount,
				totals:{m:statsCache.sumM,dr:statsCache.sumDr,dg:statsCache.sumDg,db:statsCache.sumDb,E:statsCache.sumE},
				expected:{m:ledgerExpM,dr:ledgerExpDr,dg:ledgerExpDg,db:ledgerExpDb,E:ledgerExpE},
				mins:{m:statsCache.minM,dr:statsCache.minDr,dg:statsCache.minDg,db:statsCache.minDb,E:statsCache.minE},
				maxM:statsCache.maxM,
				cntNeg:statsCache.cntNeg,
				cntOver:statsCache.cntOver,
				ledger,initialTotals
			};
		}
		function resize(newN){
			Fluid.gpu.fields.destroyPool(pool);
			cfg.N=newN;
			cfg.texW=newN+2;
			cfg.texH=newN+2;
			pool=Fluid.gpu.fields.createPool(device,newN);
			pipes=Fluid.gpu.passes.createPipelines(device,pool,format);
			groups=Fluid.gpu.passes.makeBindGroups(device,pipes,pool);
			canvas.width=newN;
			canvas.height=newN;
			canvas.style.width=Math.min(512,newN)+"px";
			reset();
		}
		function destroy(){
			Fluid.gpu.fields.destroyPool(pool);
		}
		// initial reset
		reset();
		writeParams();
		return {pool,device,canvas,context,format,cfg,pipes,groups,reset,step,getStats,resize,destroy,paintAt,pollStats,paramsArray,statsArray};
	}
	Fluid.gpu.solver={create:createSolver};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.solver;
})(globalThis);
