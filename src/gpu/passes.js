(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	function createPipelines(device,pool,canvasFormat){
		const cfg=Fluid.core.config;
		const sh=Fluid.gpu.shaders;
		function comp(label,code){
			return device.createComputePipeline({
				label,
				layout:"auto",
				compute:{module:device.createShaderModule({label: label+"-mod",code}),entryPoint:"main"}
			});
		}
		const exchangePipe=comp("exchange", sh.paramsStruct + sh.exchange);
		const copyRgbaPipe=comp("copyRgba", sh.paramsStruct + sh.copyRGBA);
		const toDisplayPipe=comp("toDisplay", sh.toDisplay);
		const statsAPipe=comp("statsA", sh.statsReduceA);
		const statsBPipe=comp("statsB", sh.statsReduceB);
		const vertMod=device.createShaderModule({label:"renderVert",code:sh.renderVert});
		const fragMod=device.createShaderModule({label:"renderFrag",code:sh.renderFrag});
		const renderPipe=device.createRenderPipeline({
			label:"render",
			layout:"auto",
			vertex:{module:vertMod,entryPoint:"main"},
			fragment:{module:fragMod,entryPoint:"main",targets:[{format:canvasFormat}]},
			primitive:{topology:"triangle-list"}
		});
		const sampler=device.createSampler({label:"linear",magFilter:"linear",minFilter:"linear"});
		return {exchangePipe,copyRgbaPipe,toDisplayPipe,statsAPipe,statsBPipe,renderPipe,sampler};
	}
	function makeBindGroups(device,pipes,pool){
		// create groups for current pool state; caller must rebuild on resize or swap
		const groups={};
		groups.exchange=device.createBindGroup({
			label:"exchange",
			layout:pipes.exchangePipe.getBindGroupLayout(0),
			entries:[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:pool.mass.tex[pool.mass.cur].createView()},
				{binding:2,resource:pool.mass.tex[1-pool.mass.cur].createView()}
			]
		});
		groups.copyMass=device.createBindGroup({
			label:"copyMass",
			layout:pipes.copyRgbaPipe.getBindGroupLayout(0),
			entries:[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:pool.mass.tex[pool.mass.cur].createView()},
				{binding:2,resource:pool.mass.tex[1-pool.mass.cur].createView()}
			]
		});
		groups.toDisplay=device.createBindGroup({
			label:"toDisplay",
			layout:pipes.toDisplayPipe.getBindGroupLayout(0),
			entries:[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:pool.mass.tex[1-pool.mass.cur].createView()},
				{binding:2,resource:pool.therm.tex[pool.therm.cur].createView()},
				{binding:3,resource:pool.display.createView()}
			]
		});
		groups.statsA=device.createBindGroup({
			label:"statsA",
			layout:pipes.statsAPipe.getBindGroupLayout(0),
			entries:[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:pool.mass.tex[1-pool.mass.cur].createView()},
				{binding:2,resource:pool.therm.tex[pool.therm.cur].createView()},
				{binding:3,resource:{buffer:pool.statsPart}}
			]
		});
		groups.statsB=device.createBindGroup({
			label:"statsB",
			layout:pipes.statsBPipe.getBindGroupLayout(0),
			entries:[
				{binding:0,resource:{buffer:pool.statsPart}},
				{binding:1,resource:{buffer:pool.stats}},
				{binding:2,resource:{buffer:pool.dims}}
			]
		});
		groups.render=device.createBindGroup({
			label:"render",
			layout:pipes.renderPipe.getBindGroupLayout(0),
			entries:[
				{binding:0,resource:pipes.sampler},
				{binding:1,resource:pool.display.createView()}
			]
		});
		return groups;
	}
	function encodeCompute(encoder,device,pool,pipes,groups,cfg){
		const cpass=encoder.beginComputePass({label:"sim"});
		// exchange pass: main diffusion
		cpass.setPipeline(pipes.exchangePipe);
		cpass.setBindGroup(0,groups.exchange);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
		// toDisplay
		cpass.setPipeline(pipes.toDisplayPipe);
		cpass.setBindGroup(0,groups.toDisplay);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
		// stats every statsEvery frames handled outside; we optionally compute each frame for simplicity and throttle HUD
		// For now compute stats each encode but caller controls flag
		// We will compute stats if needStats flag in cfg
		if(cfg._needStats){
			cpass.setPipeline(pipes.statsAPipe);
			cpass.setBindGroup(0,groups.statsA);
			cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
			cpass.setPipeline(pipes.statsBPipe);
			cpass.setBindGroup(0,groups.statsB);
			cpass.dispatchWorkgroups(1);
		}
		cpass.end();
		// copy back stats for readback if needed: copy stats -> statsRead staging handled after compute
	}
	function encodeRender(renderPass,pool,pipes,groups){
		renderPass.setPipeline(pipes.renderPipe);
		renderPass.setBindGroup(0,groups.render);
		renderPass.draw(3);
	}
	Fluid.gpu.passes={createPipelines,makeBindGroups,encodeCompute,encodeRender};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.passes;
})(globalThis);
