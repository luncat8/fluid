(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	// ping-pong parity model: m = mass/therm parity (they always swap together in
	// advect_scalar / paint), v = vel parity (forces, advectVel and applyGrad each
	// swap it once, so 3 swaps = odd and v flips each substep), press parity is
	// invariant: one red-black sweep reads press[0] -> press[1] -> press[0]. all
	// bind groups for all (m,v) combos are built once here; the frame loop never allocates.
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
		// project red/black share one explicit bind group layout so one set of
		// groups drives both pipelines
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
			paintMass:comp("paintMass",PS+sh.paintMass),
			paintTherm:comp("paintTherm",PS+sh.paintTherm),
			statsA:comp("statsA",PS+sh.statsReduceA),
			statsB:comp("statsB",PS+sh.statsReduceB),
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
		const v0=pool.vel.tex[0],v1=pool.vel.tex[1];
		const p0=pool.press.tex[0],p1=pool.press.tex[1];
		const flagsView=viewRes(pool.flags);
		const groups={forces:[],advectVel:[],rhs:[],applyGrad:[],flux:[],advectMass:[],advectTherm:[],
			paintMass:[],paintTherm:[],projRed:[],projBlack:[],statsA:[],toDisplay:[],statsB:null,render:null};
		for(let m=0;m<2;m++){
			const mm=pool.mass.tex[m],tt=pool.therm.tex[m],mn=pool.mass.tex[1-m],tn=pool.therm.tex[1-m];
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
				{binding:1,resource:viewRes(pool.flux)},
				{binding:2,resource:viewRes(mm)},
				{binding:3,resource:viewRes(mn)}
			]);
			groups.advectTherm[m]=bg(device,"advectTherm"+m,pipes.advectTherm.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(pool.flux)},
				{binding:2,resource:viewRes(tt)},
				{binding:3,resource:viewRes(tn)}
			]);
			groups.paintMass[m]=bg(device,"paintMass"+m,pipes.paintMass.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(mm)},
				{binding:2,resource:viewRes(mn)},
				{binding:3,resource:viewRes(pool.patchMass)}
			]);
			groups.paintTherm[m]=bg(device,"paintTherm"+m,pipes.paintTherm.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(tt)},
				{binding:2,resource:viewRes(tn)},
				{binding:3,resource:viewRes(pool.patchE)}
			]);
			for(let v=0;v<2;v++){
				const vv=pool.vel.tex[v],vn=pool.vel.tex[1-v];
				groups.forces[m]=groups.forces[m]||[];
				groups.forces[m][v]=bg(device,"forces"+m+v,pipes.forces.getBindGroupLayout(0),[
					{binding:0,resource:{buffer:pool.params}},
					{binding:1,resource:viewRes(vv)},
					{binding:2,resource:viewRes(vn)},
					{binding:3,resource:viewRes(mm)},
					{binding:4,resource:viewRes(tt)},
					{binding:5,resource:flagsView}
				]);
				groups.rhs[m]=groups.rhs[m]||[];
				groups.rhs[m][v]=bg(device,"rhs"+m+v,pipes.rhs.getBindGroupLayout(0),[
					{binding:0,resource:{buffer:pool.params}},
					{binding:1,resource:viewRes(pool.divsrc)},
					{binding:2,resource:viewRes(vv)},
					{binding:3,resource:viewRes(mm)},
					{binding:4,resource:flagsView}
				]);
				groups.applyGrad[m]=groups.applyGrad[m]||[];
				groups.applyGrad[m][v]=bg(device,"applyGrad"+m+v,pipes.applyGrad.getBindGroupLayout(0),[
					{binding:0,resource:{buffer:pool.params}},
					{binding:1,resource:viewRes(vv)},
					{binding:2,resource:viewRes(vn)},
					{binding:3,resource:viewRes(p0)},
					{binding:4,resource:viewRes(mm)},
					{binding:5,resource:flagsView}
				]);
				groups.statsA[m]=groups.statsA[m]||[];
				groups.statsA[m][v]=bg(device,"statsA"+m+v,pipes.statsA.getBindGroupLayout(0),[
					{binding:0,resource:{buffer:pool.params}},
					{binding:1,resource:viewRes(mm)},
					{binding:2,resource:viewRes(tt)},
					{binding:3,resource:viewRes(vv)},
					{binding:4,resource:viewRes(p0)},
					{binding:5,resource:{buffer:pool.statsPart}}
				]);
				groups.toDisplay[m]=groups.toDisplay[m]||[];
				groups.toDisplay[m][v]=bg(device,"toDisplay"+m+v,pipes.toDisplay.getBindGroupLayout(0),[
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
		groups.advectVel[0]=bg(device,"advectVel0",pipes.advectVel.getBindGroupLayout(0),[
			{binding:0,resource:{buffer:pool.params}},
			{binding:1,resource:viewRes(v0)},
			{binding:2,resource:viewRes(v1)}
		]);
		groups.advectVel[1]=bg(device,"advectVel1",pipes.advectVel.getBindGroupLayout(0),[
			{binding:0,resource:{buffer:pool.params}},
			{binding:1,resource:viewRes(v1)},
			{binding:2,resource:viewRes(v0)}
		]);
		groups.flux[0]=bg(device,"flux0",pipes.flux.getBindGroupLayout(0),[
			{binding:0,resource:{buffer:pool.params}},
			{binding:1,resource:viewRes(v0)},
			{binding:2,resource:flagsView},
			{binding:3,resource:viewRes(pool.flux)}
		]);
		groups.flux[1]=bg(device,"flux1",pipes.flux.getBindGroupLayout(0),[
			{binding:0,resource:{buffer:pool.params}},
			{binding:1,resource:viewRes(v1)},
			{binding:2,resource:flagsView},
			{binding:3,resource:viewRes(pool.flux)}
		]);
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
	// one substep of the 0.0.0 section 8 graph. m and v flip on return (3 vel swaps = odd).
	function encodeStep(cpass,pool,pipes,groups,state,iters){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		let m=state.m, v=state.v;
		cpass.setPipeline(pipes.forces);
		cpass.setBindGroup(0,groups.forces[m][v]);
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
		cpass.setBindGroup(0,groups.advectTherm[m]);
		cpass.dispatchWorkgroups(wx,wy);
		state.m=1-m;
		state.v=v;
	}
	function encodeStats(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.statsA);
		cpass.setBindGroup(0,groups.statsA[state.m][state.v]);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
		cpass.setPipeline(pipes.statsB);
		cpass.setBindGroup(0,groups.statsB);
		cpass.dispatchWorkgroups(1);
	}
	function encodeDisplay(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.toDisplay);
		cpass.setBindGroup(0,groups.toDisplay[state.m][state.v]);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
	}
	function encodePaint(cpass,pool,pipes,groups,state){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		cpass.setPipeline(pipes.paintMass);
		cpass.setBindGroup(0,groups.paintMass[state.m]);
		cpass.dispatchWorkgroups(wx,wy);
		cpass.setPipeline(pipes.paintTherm);
		cpass.setBindGroup(0,groups.paintTherm[state.m]);
		cpass.dispatchWorkgroups(wx,wy);
		state.m=1-state.m;
	}
	function encodeRender(rpass,pipes,groups){
		rpass.setPipeline(pipes.renderPipe);
		rpass.setBindGroup(0,groups.render);
		rpass.draw(3);
	}
	Fluid.gpu.passes={createPipelines,makeBindGroups,encodeStep,encodeStats,encodeDisplay,encodePaint,encodeRender};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.passes;
})(globalThis);
