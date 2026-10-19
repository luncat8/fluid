(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.app=Fluid.app||{};
	function create(queue,clock){
		const now=clock||(()=>performance.now());
		const state={busy:false,error:null,rafHz:0,simHz:0,renderHz:0,simSpeed:0,cpuMs:0,queueMs:0};
		let elapsed=0,callbacks=0,sims=0,renders=0,simTime=0,cpuTime=0,submittedAt=0;
		function complete(){
			state.queueMs=now()-submittedAt;
			state.busy=false;
		}
		function failed(error){
			state.error=error;
			state.busy=false;
		}
		function tick(dt){
			elapsed+=dt;callbacks++;
			if(elapsed<0.5)return;
			state.rafHz=callbacks/elapsed;
			state.simHz=sims/elapsed;
			state.renderHz=renders/elapsed;
			state.simSpeed=simTime/elapsed;
			state.cpuMs=renders?cpuTime/renders:0;
			elapsed=0;callbacks=0;sims=0;renders=0;simTime=0;cpuTime=0;
		}
		function submitted(simDt,cpuMs){
			renders++;cpuTime+=cpuMs;
			if(simDt>0){sims++;simTime+=simDt;}
			submittedAt=now();
			state.busy=true;
			// Bound the queue to one application frame. Completion latency includes
			// driver/scheduling overhead; it is not a GPU timestamp measurement.
			queue.onSubmittedWorkDone().then(complete,failed);
		}
		return {state,tick,submitted};
	}
	Fluid.app.performance={create};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.app.performance;
})(globalThis);
