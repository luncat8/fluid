(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	// m uses three slots when dye FCT is enabled; t and v remain ping-pong fields.
	// Pressure parity is invariant because every red-black sweep returns to press[0].
	// Bind groups for every state combination are built once during setup/resize.
	function createPipelines(device,canvasFormat){
		const sh=Fluid.gpu.shaders;
		const PS=sh.paramsStruct;
		function comp(label,code,constants){
			const desc={
				label,
				layout:"auto",
				compute:{module:device.createShaderModule({label:label+"-mod",code}),entryPoint:"main"}
			};
			if(constants)desc.compute.constants=constants;
			return device.createComputePipeline(desc);
		}
		const projLayout=device.createBindGroupLayout({
			label:"project",
			entries:[
				{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
				{binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-only",format:"r32float"}},
				{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float"}},
				{binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},
				{binding:4,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},
				{binding:5,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"uint"}}
			]
		});
		const projMod=device.createShaderModule({label:"project-mod",code:PS+sh.projectStep});
		function projPipe(label,color){
			return device.createComputePipeline({
				label,
				layout:device.createPipelineLayout({bindGroupLayouts:[projLayout]}),
				compute:{module:projMod,entryPoint:"main",constants:{COLOR:color}}
			});
		}
		const vertMod=device.createShaderModule({label:"renderVert",code:sh.renderVert});
		const fragMod=device.createShaderModule({label:"renderFrag",code:sh.renderFrag});
		return {
			forces:comp("forces",PS+sh.forces),
			advectVel:comp("advectVel",PS+sh.advectVel),
			rhs:comp("rhs",PS+sh.poissonRhs),
			projRed:projPipe("projRed",0),
			projBlack:projPipe("projBlack",1),
			applyGrad:comp("applyGrad",PS+sh.applyGrad),
			flux:comp("flux",PS+sh.computeFlux),
			advectMass:comp("advectMass",PS+sh.advectMass),
			advectTherm:comp("advectTherm",PS+sh.advectTherm),
			deriveT:comp("deriveT",PS+sh.deriveT),
			conduction:comp("conduction",PS+sh.conduction),
			heatSources:comp("heatSources",PS+sh.heatSources),
			limiterPrepare:comp("limiterPrepare",PS+sh.limiterPrepare),
			limiterApply:comp("limiterApply",PS+sh.limiterApply),
			paintMass:comp("paintMass",PS+sh.paintMass),
			paintTherm:comp("paintTherm",PS+sh.paintTherm),
			statsA:comp("statsA",PS+sh.statsReduceA),
			statsB:comp("statsB",PS+sh.statsReduceB),
			buoySample:comp("buoySample",PS+sh.buoySample),
			toDisplay:comp("toDisplay",PS+sh.toDisplay),
			renderPipe:device.createRenderPipeline({
				label:"render",
				layout:"auto",
				vertex:{module:vertMod,entryPoint:"main"},
				fragment:{module:fragMod,entryPoint:"main",targets:[{format:canvasFormat}]},
				primitive:{topology:"triangle-list"}
			}),
			sampler:device.createSampler({label:"linear",magFilter:"linear",minFilter:"linear"}),
			projLayout
		};
	}
	function bg(device,label,layout,entries){
		return device.createBindGroup({label,layout,entries});
	}
	function viewRes(tex){
		return tex.createView();
	}
	function makeBindGroups(device,pipes,pool){
		const p0=pool.press.tex[0],p1=pool.press.tex[1];
		const flagsView=viewRes(pool.flags);
		const fluxView=viewRes(pool.flux);
		const limiterAView=viewRes(pool.limiterA);
		const limiterBView=viewRes(pool.limiterB);
		const groups={
			forces:[],advectVel:[],rhs:[],applyGrad:[],flux:[],advectMass:[],advectTherm:[],
			deriveT:[],conduction:[],heatSources:[],
			limiterPrepare:[],limiterApply:[],paintMass:[],paintTherm:[],projRed:[],projBlack:[],
			statsA:[],toDisplay:[],statsB:null,buoySample:[],render:null
		};
		// thermal passes run after scalar transport, so they key on the mass
		// slot that will be current then: [start mass slot][therm source][mass slot]
		function thermalGroups(label,pipe,withFlags){
			const out=[];
			for(let m=0;m<3;m++){
				out[m]=[];
				for(let t=0;t<2;t++){
					out[m][t]=[];
					for(let mm=0;mm<3;mm++){
						const entries=[
							{binding:0,resource:{buffer:pool.params}},
							{binding:1,resource:viewRes(pool.mass.tex[mm])},
							{binding:2,resource:viewRes(pool.therm.tex[t])},
							{binding:3,resource:viewRes(pool.therm.tex[1-t])}
						];
						if(withFlags)entries.push({binding:4,resource:flagsView});
						out[m][t][mm]=bg(device,label+m+t+mm,pipe.getBindGroupLayout(0),entries);
					}
				}
			}
			return out;
		}
		groups.deriveT=thermalGroups("deriveT",pipes.deriveT,false);
		groups.conduction=thermalGroups("conduction",pipes.conduction,true);
		groups.heatSources=thermalGroups("heatSources",pipes.heatSources,true);
		for(let m=0;m<3;m++){
			const nextM=(m+1)%3,finalM=(m+2)%3;
			const mm=pool.mass.tex[m],ml=pool.mass.tex[nextM];
			const mo=pool.mass.tex[finalM];
			groups.projRed[m]=bg(device,"projRed"+m,pipes.projLayout,[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(p0)},
				{binding:2,resource:viewRes(p1)},
				{binding:3,resource:viewRes(pool.divsrc)},
				{binding:4,resource:viewRes(mm)},
				{binding:5,resource:flagsView}
			]);
			groups.projBlack[m]=bg(device,"projBlack"+m,pipes.projLayout,[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(p1)},
				{binding:2,resource:viewRes(p0)},
				{binding:3,resource:viewRes(pool.divsrc)},
				{binding:4,resource:viewRes(mm)},
				{binding:5,resource:flagsView}
			]);
			groups.advectMass[m]=bg(device,"advectMass"+m,pipes.advectMass.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:fluxView},
				{binding:2,resource:viewRes(mm)},
				{binding:3,resource:viewRes(ml)}
			]);
			groups.limiterPrepare[m]=bg(device,"limiterPrepare"+m,pipes.limiterPrepare.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:fluxView},
				{binding:2,resource:viewRes(mm)},
				{binding:3,resource:viewRes(ml)},
				{binding:4,resource:limiterAView},
				{binding:5,resource:limiterBView}
			]);
			groups.limiterApply[m]=bg(device,"limiterApply"+m,pipes.limiterApply.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:fluxView},
				{binding:2,resource:viewRes(mm)},
				{binding:3,resource:viewRes(ml)},
				{binding:4,resource:limiterAView},
				{binding:5,resource:limiterBView},
				{binding:6,resource:viewRes(mo)}
			]);
			groups.paintMass[m]=bg(device,"paintMass"+m,pipes.paintMass.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(mm)},
				{binding:2,resource:viewRes(ml)},
				{binding:3,resource:viewRes(pool.patchMass)}
			]);
			groups.rhs[m]=[];
			groups.applyGrad[m]=[];
			groups.buoySample[m]=[];
			groups.statsA[m]=[];
			groups.toDisplay[m]=[];
			for(let t=0;t<2;t++){
				const tt=pool.therm.tex[t];
				groups.statsA[m][t]=[];
				groups.toDisplay[m][t]=[];
				for(let v=0;v<2;v++){
					const vv=pool.vel.tex[v],vn=pool.vel.tex[1-v];
					groups.forces[m]=groups.forces[m]||[];
					groups.forces[m][t]=groups.forces[m][t]||[];
					groups.forces[m][t][v]=bg(device,"forces"+m+t+v,pipes.forces.getBindGroupLayout(0),[
						{binding:0,resource:{buffer:pool.params}},
						{binding:1,resource:viewRes(vv)},
						{binding:2,resource:viewRes(vn)},
						{binding:3,resource:viewRes(mm)},
						{binding:4,resource:viewRes(tt)},
						{binding:5,resource:flagsView}
					]);
					if(t===0){
						groups.rhs[m][v]=bg(device,"rhs"+m+v,pipes.rhs.getBindGroupLayout(0),[
							{binding:0,resource:{buffer:pool.params}},
							{binding:1,resource:viewRes(pool.divsrc)},
							{binding:2,resource:viewRes(vv)},
							{binding:3,resource:viewRes(mm)},
							{binding:4,resource:flagsView}
						]);
						groups.applyGrad[m][v]=bg(device,"applyGrad"+m+v,pipes.applyGrad.getBindGroupLayout(0),[
							{binding:0,resource:{buffer:pool.params}},
							{binding:1,resource:viewRes(vv)},
							{binding:2,resource:viewRes(vn)},
							{binding:3,resource:viewRes(p0)},
							{binding:4,resource:viewRes(mm)},
							{binding:5,resource:flagsView}
						]);
						// the sample reads the (mass, vel) pair the sim left behind
						groups.buoySample[m][v]=bg(device,"buoySample"+m+v,pipes.buoySample.getBindGroupLayout(0),[
							{binding:0,resource:{buffer:pool.params}},
							{binding:1,resource:viewRes(mm)},
							{binding:2,resource:viewRes(vv)},
							{binding:3,resource:{buffer:pool.buoy}}
						]);
					}
					groups.statsA[m][t][v]=bg(device,"statsA"+m+t+v,pipes.statsA.getBindGroupLayout(0),[
						{binding:0,resource:{buffer:pool.params}},
						{binding:1,resource:viewRes(mm)},
						{binding:2,resource:viewRes(tt)},
						{binding:3,resource:viewRes(vv)},
						{binding:4,resource:viewRes(p0)},
						{binding:5,resource:{buffer:pool.statsPart}}
					]);
					groups.toDisplay[m][t][v]=bg(device,"toDisplay"+m+t+v,pipes.toDisplay.getBindGroupLayout(0),[
						{binding:0,resource:{buffer:pool.params}},
						{binding:1,resource:viewRes(mm)},
						{binding:2,resource:viewRes(tt)},
						{binding:3,resource:viewRes(pool.display)},
						{binding:4,resource:viewRes(vv)},
						{binding:5,resource:viewRes(p0)},
						{binding:6,resource:flagsView}
					]);
				}
			}
		}
		for(let t=0;t<2;t++){
			const tt=pool.therm.tex[t],tn=pool.therm.tex[1-t];
			groups.advectTherm[t]=bg(device,"advectTherm"+t,pipes.advectTherm.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:fluxView},
				{binding:2,resource:viewRes(tt)},
				{binding:3,resource:viewRes(tn)}
			]);
			groups.paintTherm[t]=bg(device,"paintTherm"+t,pipes.paintTherm.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(tt)},
				{binding:2,resource:viewRes(tn)},
				{binding:3,resource:viewRes(pool.patchE)}
			]);
		}
		for(let v=0;v<2;v++){
			const vv=pool.vel.tex[v],vn=pool.vel.tex[1-v];
			groups.advectVel[v]=bg(device,"advectVel"+v,pipes.advectVel.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:viewRes(vn)}
			]);
			groups.flux[v]=bg(device,"flux"+v,pipes.flux.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:flagsView},
				{binding:3,resource:fluxView}
			]);
		}
		groups.statsB=bg(device,"statsB",pipes.statsB.getBindGroupLayout(0),[
			{binding:0,resource:{buffer:pool.statsPart}},
			{binding:1,resource:{buffer:pool.stats}},
			{binding:2,resource:{buffer:pool.dims}}
		]);
		groups.render=bg(device,"render",pipes.renderPipe.getBindGroupLayout(0),[
			{binding:0,resource:pipes.sampler},
			{binding:1,resource:viewRes(pool.display)}
		]);
		return groups;
	}
	function encodeStep(cpass,pool,pipes,groups,state,iters,fctOn){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		const m=state.m,t=state.t;
		let v=state.v;
		cpass.setPipeline(pipes.forces);
		cpass.setBindGroup(0,groups.forces[m][t][v]);
		cpass.dispatchWorkgroups(wx,wy);
		v=1-v;
		cpass.setPipeline(pipes.advectVel);
		cpass.setBindGroup(0,groups.advectVel[v]);
		cpass.dispatchWorkgroups(wx,wy);
		v=1-v;
		cpass.setPipeline(pipes.rhs);
		cpass.setBindGroup(0,groups.rhs[m][v]);
		cpass.dispatchWorkgroups(wx,wy);
		for(let k=0;k<iters;k++){
			cpass.setPipeline(pipes.projRed);
			cpass.setBindGroup(0,groups.projRed[m]);
			cpass.dispatchWorkgroups(wx,wy);
			cpass.setPipeline(pipes.projBlack);
			cpass.setBindGroup(0,groups.projBlack[m]);
			cpass.dispatchWorkgroups(wx,wy);
		}
		cpass.setPipeline(pipes.applyGrad);
		cpass.setBindGroup(0,groups.applyGrad[m][v]);
		cpass.dispatchWorkgroups(wx,wy);
		v=1-v;
		cpass.setPipeline(pipes.flux);
		cpass.setBindGroup(0,groups.flux[v]);
		cpass.dispatchWorkgroups(wx,wy);
		cpass.setPipeline(pipes.advectMass);
		cpass.setBindGroup(0,groups.advectMass[m]);
		cpass.dispatchWorkgroups(wx,wy);
		cpass.setPipeline(pipes.advectTherm);
		cpass.setBindGroup(0,groups.advectTherm[t]);
		cpass.dispatchWorkgroups(wx,wy);
		let nextM=(m+1)%3;
		if(fctOn){
			cpass.setPipeline(pipes.limiterPrepare);
			cpass.setBindGroup(0,groups.limiterPrepare[m]);
			cpass.dispatchWorkgroups(wx,wy);
			cpass.setPipeline(pipes.limiterApply);
			cpass.setBindGroup(0,groups.limiterApply[m]);
			cpass.dispatchWorkgroups(wx,wy);
			state.m=(m+2)%3;
		}else{
			state.m=nextM;
		}
		// thermal chain (0.4.0): derive, conduction, sources, derive. four
		// ping-pongs after the transport flip return therm to 1-t, as before.
		const mf=state.m;
		let ts=1-t;
		cpass.setPipeline(pipes.deriveT);
		cpass.setBindGroup(0,groups.deriveT[m][ts][mf]);
		cpass.dispatchWorkgroups(wx,wy);
		ts=1-ts;
		cpass.setPipeline(pipes.conduction);
		cpass.setBindGroup(0,groups.conduction[m][ts][mf]);
		cpass.dispatchWorkgroups(wx,wy);
		ts=1-ts;
		cpass.setPipeline(pipes.heatSources);
		cpass.setBindGroup(0,groups.heatSources[m][ts][mf]);
		cpass.dispatchWorkgroups(wx,wy);
		ts=1-ts;
		cpass.setPipeline(pipes.deriveT);
		cpass.setBindGroup(0,groups.deriveT[m][ts][mf]);
		cpass.dispatchWorkgroups(wx,wy);
		state.t=1-ts;
		state.v=v;
	}
	function encodeStats(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.statsA);
		cpass.setBindGroup(0,groups.statsA[state.m][state.t][state.v]);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
		cpass.setPipeline(pipes.statsB);
		cpass.setBindGroup(0,groups.statsB);
		cpass.dispatchWorkgroups(1);
	}
	// one workgroup over the body footprint. the gpu solver dispatches it once per
	// refresh frame, after the substep loop, never inside it
	function encodeBuoySample(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.buoySample);
		cpass.setBindGroup(0,groups.buoySample[state.m][state.v]);
		cpass.dispatchWorkgroups(1);
	}
	function encodeDisplay(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.toDisplay);
		cpass.setBindGroup(0,groups.toDisplay[state.m][state.t][state.v]);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
	}
	function encodePaint(cpass,pool,pipes,groups,state){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		cpass.setPipeline(pipes.paintMass);
		cpass.setBindGroup(0,groups.paintMass[state.m]);
		cpass.dispatchWorkgroups(wx,wy);
		cpass.setPipeline(pipes.paintTherm);
		cpass.setBindGroup(0,groups.paintTherm[state.t]);
		cpass.dispatchWorkgroups(wx,wy);
		state.m=(state.m+1)%3;
		state.t=1-state.t;
	}
	function encodeRender(rpass,pipes,groups){
		rpass.setPipeline(pipes.renderPipe);
		rpass.setBindGroup(0,groups.render);
		rpass.draw(3);
	}
	Fluid.gpu.passes={createPipelines,makeBindGroups,encodeStep,encodeStats,encodeBuoySample,encodeDisplay,encodePaint,encodeRender};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.passes;
})(globalThis);
