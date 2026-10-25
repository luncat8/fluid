(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.tests=Fluid.tests||{};

	function instrumentDevice(device){
		const state={
			texCreated:0,texAlive:0,
			bufCreated:0,bufAlive:0,
			uncaptured:[],
			validationErrors:[]
		};
		const origCreateTex=device.createTexture.bind(device);
		const origCreateBuf=device.createBuffer.bind(device);
		const onUncaptured=ev=>{
			const msg=(ev&&ev.error&&ev.error.message)||String(ev);
			state.uncaptured.push(msg);
		};
		if(device.addEventListener)device.addEventListener("uncapturederror",onUncaptured);

		device.createTexture=function(desc){
			const tex=origCreateTex(desc);
			state.texCreated++;
			state.texAlive++;
			let dead=false;
			const origDestroy=tex.destroy?tex.destroy.bind(tex):null;
			tex.destroy=function(){
				if(!dead){
					dead=true;
					state.texAlive--;
				}
				if(origDestroy)return origDestroy();
			};
			return tex;
		};
		device.createBuffer=function(desc){
			const buf=origCreateBuf(desc);
			state.bufCreated++;
			state.bufAlive++;
			let dead=false;
			const origDestroy=buf.destroy?buf.destroy.bind(buf):null;
			buf.destroy=function(){
				if(!dead){
					dead=true;
					state.bufAlive--;
				}
				if(origDestroy)return origDestroy();
			};
			return buf;
		};
		function restore(){
			device.createTexture=origCreateTex;
			device.createBuffer=origCreateBuf;
			if(device.removeEventListener)device.removeEventListener("uncapturederror",onUncaptured);
		}
		return {state,restore};
	}

	function diffBytes(bufA,bufB){
		if(bufA.byteLength!==bufB.byteLength)return {equal:false,offset:-1,a:bufA.byteLength,b:bufB.byteLength};
		const a=new Uint8Array(bufA),b=new Uint8Array(bufB);
		for(let i=0;i<a.length;i++){
			if(a[i]!==b[i])return {equal:false,offset:i,a:a[i],b:b[i]};
		}
		return {equal:true,offset:-1,a:0,b:0};
	}

	async function pushScopes(device){
		if(!device.pushErrorScope)return;
		device.pushErrorScope("validation");
		device.pushErrorScope("out-of-memory");
	}
	async function popScopes(device,cycle,sink){
		if(!device.popErrorScope)return;
		const oom=await device.popErrorScope();
		const val=await device.popErrorScope();
		if(oom)sink.push("cycle "+cycle+" OOM: "+(oom.message||oom));
		if(val)sink.push("cycle "+cycle+" validation: "+(val.message||val));
	}

	function makeCanvas(N){
		if(typeof document!=="undefined"&&document.createElement){
			const c=document.createElement("canvas");
			c.width=N;c.height=N;
			return c;
		}
		return {
			width:N,height:N,
			getContext(){return {configure(){},getCurrentTexture(){return {createView(){return {};}};}};}
		};
	}

	async function runLifecycleSuite(opts){
		const options=opts||{};
		const cycles=options.cycles||20;
		const N=options.N||64;
		const stepFrames=options.stepFrames||5;
		const replayFrames=options.replayFrames||100;
		const onProgress=options.onProgress||null;
		const device=options.device;
		const context=options.context||null;
		const format=options.format||"rgba8unorm";
		if(!device)throw new Error("runLifecycleSuite requires a WebGPU or mock device");

		const inst=instrumentDevice(device);
		const st=inst.state;
		const checks=[];
		function check(ok,name,detail){
			checks.push({ok:!!ok,name,detail:detail||""});
		}

		const heapStart=(typeof performance!=="undefined"&&performance.memory)?performance.memory.usedJSHeapSize:null;
		const cycleRecords=[];
		let scratchOut=null;

		try{
			for(let c=0;c<cycles;c++){
				await pushScopes(device);
				const canvas=options.canvas||makeCanvas(N);
				const sim=await Fluid.engine.create(canvas,{
					device,context,format,
					cfg:{
						N,scenario:"ice",
						substepsAuto:false,substepsFixed:1,
						statsEvery:1,probeEvery:1
					}
				});
				const aliveTexDuring=st.texAlive;
				const aliveBufBeforeSave=st.bufAlive;

				const pid=sim.addProbe(N*0.5,N*0.5);
				const eid=sim.addEmitter({kind:"heat",x:N*0.35,y:N*0.35,radius:5,value:1800});
				const cid=sim.addCollider({shape:"circle",x:N*0.65,y:N*0.5,radius:5,vx:-4,vy:0});
				let phaseBatches=0;
				sim.onPhaseChange(()=>{phaseBatches++;});

				for(let f=0;f<stepFrames;f++){
					sim.step(1/120);
				}
				await sim.pollReadback();

				if(!scratchOut||scratchOut.byteLength!==sim.saveSize){
					scratchOut=new ArrayBuffer(sim.saveSize);
				}
				const snap=await sim.serialize(scratchOut);
				const aliveBufAfterSave=st.bufAlive;

				sim.step(1/120);
				sim.deserialize(snap);
				sim.step(1/120);
				await sim.pollReadback();

				sim.removeProbe(pid);
				sim.removeEmitter(eid);
				sim.removeCollider(cid);
				sim.destroy();
				if(device.queue&&device.queue.onSubmittedWorkDone){
					await device.queue.onSubmittedWorkDone();
				}
				await popScopes(device,c,st.validationErrors);

				cycleRecords.push({
					cycle:c,
					aliveTexDuring,
					aliveBufBeforeSave,
					aliveBufAfterSave,
					texAfterDestroy:st.texAlive,
					bufAfterDestroy:st.bufAlive,
					phaseBatches
				});
				if(onProgress)onProgress(c+1,cycles,cycleRecords[cycleRecords.length-1]);
			}

			const leakedCycles=cycleRecords.filter(r=>r.texAfterDestroy!==0||r.bufAfterDestroy!==0);
			const stagingLeaks=cycleRecords.filter(r=>r.aliveBufAfterSave!==r.aliveBufBeforeSave);
			check(cycleRecords.length===cycles,"20 create/step/serialize/deserialize/destroy cycles completed","cycles="+cycleRecords.length);
			check(leakedCycles.length===0,"live textures and buffers return to 0 after every destroy()","leakedCycles="+leakedCycles.length+" (texAlive="+st.texAlive+", bufAlive="+st.bufAlive+")");
			check(stagingLeaks.length===0,"transient serialize() staging buffer destroyed before serialize resolves","stagingLeaks="+stagingLeaks.length);
			check(st.validationErrors.length===0,"zero WebGPU validation/OOM errors across all cycles",st.validationErrors.join("; ")||"0 errors");
			check(st.uncaptured.length===0,"zero uncaptured device errors across all cycles",st.uncaptured.join("; ")||"0 errors");

			// Deterministic 100-step replay test on engine + device
			await pushScopes(device);
			const canvasR=options.canvas||makeCanvas(N);
			const simR=await Fluid.engine.create(canvasR,{
				device,context,format,
				cfg:{
					N,scenario:"ice",
					substepsAuto:false,substepsFixed:1,
					statsEvery:10,probeEvery:5
				}
			});
			simR.addProbe(N*0.5,N*0.5);
			simR.addEmitter({kind:"heat",x:N*0.5,y:N*0.25,radius:6,value:2400});
			simR.addCollider({shape:"circle",x:N*0.5,y:N*0.65,radius:5,vx:0,vy:0});
			for(let i=0;i<20;i++)simR.step(1/120);
			await simR.pollReadback();
			const midSnap=await simR.serialize();

			for(let i=0;i<replayFrames;i++)simR.step(1/120);
			await simR.pollReadback();
			const endSnapA=await simR.serialize();

			simR.reset();
			simR.deserialize(midSnap);
			for(let i=0;i<replayFrames;i++)simR.step(1/120);
			await simR.pollReadback();
			const endSnapB=await simR.serialize();
			simR.destroy();
			await popScopes(device,cycles,st.validationErrors);

			const cmpGpu=diffBytes(endSnapA,endSnapB);
			check(cmpGpu.equal,"engine 100-step replay from mid-run snapshot is byte-for-byte identical",
				cmpGpu.equal?("bytes="+endSnapA.byteLength):("mismatch at byte "+cmpGpu.offset+": "+cmpGpu.a+" vs "+cmpGpu.b));
			check(st.texAlive===0&&st.bufAlive===0,"replay simulation destroyed cleanly with 0 live resources",
				"texAlive="+st.texAlive+", bufAlive="+st.bufAlive);

			// Deterministic 100-step CPU solver replay check (exercises full PDE numerics)
			if(Fluid.core&&Fluid.core.solver&&Fluid.core.config){
				const cpuCfg=Object.assign({},Fluid.core.config.defaults,{N:32,texW:34,texH:34,scenario:"ice",dt:1/120});
				const cpu=Fluid.core.solver.create(cpuCfg);
				cpu.addEmitter({kind:"heat",x:16,y:8,radius:4,value:2400});
				cpu.addCollider({shape:"circle",x:16,y:20,radius:4,vx:2,vy:0});
				for(let i=0;i<15;i++)cpu.step(cpuCfg.dt);
				const cpuMid=cpu.serialize();
				for(let i=0;i<replayFrames;i++)cpu.step(cpuCfg.dt);
				const cpuEndA=cpu.serialize();

				cpu.reset();
				cpu.deserialize(cpuMid);
				for(let i=0;i<replayFrames;i++)cpu.step(cpuCfg.dt);
				const cpuEndB=cpu.serialize();
				const cmpCpu=diffBytes(cpuEndA,cpuEndB);
				check(cmpCpu.equal,"CPU reference solver 100-step replay from mid-run snapshot is byte-for-byte identical",
					cmpCpu.equal?("bytes="+cpuEndA.byteLength):("mismatch at byte "+cmpCpu.offset+": "+cmpCpu.a+" vs "+cmpCpu.b));
			}
		}finally{
			inst.restore();
		}

		const heapEnd=(typeof performance!=="undefined"&&performance.memory)?performance.memory.usedJSHeapSize:null;
		const heapDelta=heapStart!==null&&heapEnd!==null?(heapEnd-heapStart):null;
		const passed=checks.filter(c=>c.ok).length;
		return {
			ok:passed===checks.length,
			passed,
			total:checks.length,
			checks,
			cycles:cycleRecords,
			totals:{
				texCreated:st.texCreated,
				texAlive:st.texAlive,
				bufCreated:st.bufCreated,
				bufAlive:st.bufAlive,
				validationErrors:st.validationErrors.length,
				uncapturedErrors:st.uncaptured.length,
				heapDelta
			}
		};
	}

	Fluid.tests.apiTest={runLifecycleSuite,instrumentDevice,diffBytes};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.tests.apiTest;
})(globalThis);
