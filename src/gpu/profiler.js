(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	// optional timestamp-query profiling (0.6.0 t1): one query pair per timed
	// pass group, summed per group across the frame's substeps. never on the
	// required path: create() returns null without the feature, and solver.frame
	// only splits passes while cfg.gpuTiming is on. readbacks go through a
	// 2-slot MAP_READ ring with the same generation guard as stats/buoy, so a
	// sample that lands after reset/destroy is dropped.
	const GROUP={vel:0,rhs:1,press:2,trans:3,phase:4,stats:5,disp:6,render:7};
	const GROUP_NAMES=["vel","rhs","press","trans","phase","stats","disp","render"];
	const MAX_PAIRS=128;	// 5 step groups x 24 substeps + 3 frame groups = 123
	const QUERIES=MAX_PAIRS*2;
	function create(device){
		if(!device||!device.features||typeof device.features.has!=="function")return null;
		if(!device.features.has("timestamp-query"))return null;
		if(typeof device.createQuerySet!=="function")return null;
		const period=(device.limits&&device.limits.timestampPeriod)||1;
		const querySet=device.createQuerySet({label:"profQueries",type:"timestamp",count:QUERIES});
		const resolveBuf=device.createBuffer({label:"profResolve",size:QUERIES*8,
			usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
		const readBufs=[0,1].map(i=>device.createBuffer({label:"profRead"+i,size:QUERIES*8,
			usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}));
		const marks=[new Int32Array(MAX_PAIRS),new Int32Array(MAX_PAIRS)];
		const sums=new Float64Array(GROUP_NAMES.length);
		const pend=[{slot:0,pairs:0,gen:-1,frame:0,substeps:0},{slot:1,pairs:0,gen:-1,frame:0,substeps:0}];
		const results={valid:false,frame:0,substeps:0,period,ms:new Float64Array(GROUP_NAMES.length),totalMs:0};
		const tw={querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1};
		const desc={label:"",timestampWrites:tw};
		let openPass=null;
		let active=false;	// this frame's sample will be resolved
		let curSlot=0;
		let pairs=0;
		let frameNo=0;
		let frameSubsteps=0;
		let generation=0;
		let pendHead=0,pendCount=0;
		let mapping=false;
		const busy=[false,false];
		function beginFrame(substeps,frame){
			pairs=0;
			frameNo=frame;
			frameSubsteps=substeps;
			curSlot=busy[0]&&!busy[1]?1:0;
			active=!busy[curSlot];
		}
		// ends the open pass (if any) and opens a new compute pass for the group.
		// always returns a usable pass, timestamped only while a slot is free.
		function pass(encoder,group,label){
			closePass();
			desc.label=label;
			if(active&&pairs<MAX_PAIRS){
				tw.beginningOfPassWriteIndex=pairs*2;
				tw.endOfPassWriteIndex=pairs*2+1;
				marks[curSlot][pairs]=group;
				pairs++;
				desc.timestampWrites=tw;
			}else{
				desc.timestampWrites=undefined;
			}
			openPass=encoder.beginComputePass(desc);
			return openPass;
		}
		// timestampWrites fragment for the render pass descriptor (solver-owned)
		function tsWrites(group){
			if(!active||pairs>=MAX_PAIRS)return undefined;
			tw.beginningOfPassWriteIndex=pairs*2;
			tw.endOfPassWriteIndex=pairs*2+1;
			marks[curSlot][pairs]=group;
			pairs++;
			return tw;
		}
		function closePass(){
			if(openPass){openPass.end();openPass=null;}
		}
		function finish(encoder){
			closePass();
			if(!active||pairs===0)return;
			encoder.resolveQuerySet(querySet,0,pairs*2,resolveBuf,0);
			encoder.copyBufferToBuffer(resolveBuf,0,readBufs[curSlot],0,pairs*16);
			const item=pend[(pendHead+pendCount)%2];
			item.slot=curSlot;item.pairs=pairs;item.gen=generation;
			item.frame=frameNo;item.substeps=frameSubsteps;
			pendCount++;
			busy[curSlot]=true;
			active=false;
		}
		async function poll(){
			if(mapping||pendCount===0)return;
			mapping=true;
			const item=pend[pendHead];
			const buf=readBufs[item.slot];
			try{
				await buf.mapAsync(GPUMapMode.READ);
				if(item.gen===generation){
					const view=new BigUint64Array(buf.getMappedRange(),0,item.pairs*2);
					sums.fill(0);
					for(let p=0;p<item.pairs;p++){
						const ticks=view[p*2+1]-view[p*2];
						if(ticks>0n)sums[marks[item.slot][p]]+=Number(ticks)*period/1e6;
					}
					let total=0;
					for(let g=0;g<sums.length;g++){results.ms[g]=sums[g];total+=sums[g];}
					results.totalMs=total;
					results.frame=item.frame;
					results.substeps=item.substeps;
					results.valid=true;
				}
				buf.unmap();
			}catch(e){
				console.error("gpu timing readback failed",e);
			}
			busy[item.slot]=false;
			pendHead=(pendHead+1)%2;
			pendCount--;
			mapping=false;
		}
		function invalidate(){
			generation++;
			results.valid=false;
		}
		function destroy(){
			invalidate();
			for(const b of [resolveBuf,readBufs[0],readBufs[1]]){try{b.destroy();}catch(e){}}
			try{querySet.destroy();}catch(e){}
		}
		return {results,beginFrame,pass,tsWrites,closePass,finish,poll,invalidate,destroy};
	}
	Fluid.gpu.profiler={create,GROUP,GROUP_NAMES};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.profiler;
})(globalThis);
