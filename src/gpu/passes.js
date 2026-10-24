(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
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
		// phaseEos (APPLY_SOURCES=0) and phaseSources (APPLY_SOURCES=1) share
		// groups.phaseB, so WebGPU requires an explicit bind group layout.
		const phaseLayout=device.createBindGroupLayout({
			label:"phase",
			entries:[
				{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
				{binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},
				{binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},
				{binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},
				{binding:4,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"uint"}},
				{binding:5,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},
				{binding:6,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},
				{binding:7,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float"}},
				{binding:8,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba8uint"}},
				{binding:9,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float"}},
				{binding:10,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rg32float"}}
			]
		});
		const phaseMod=device.createShaderModule({label:"phaseEos-mod",code:PS+sh.phaseEos});
		function phasePipe(label,applySources){
			return device.createComputePipeline({
				label,
				layout:device.createPipelineLayout({bindGroupLayouts:[phaseLayout]}),
				compute:{module:phaseMod,entryPoint:"main",constants:{APPLY_SOURCES:applySources}}
			});
		}
		const vertMod=device.createShaderModule({label:"renderVert",code:sh.renderVert});
		const fragMod=device.createShaderModule({label:"renderFrag",code:sh.renderFrag});
		return {
			forces:comp("forces",PS+sh.forces),
			advectVel:comp("advectVel",PS+sh.advectVel),
			advectVelMac:comp("advectVelMac",PS+sh.advectVelMac),
			rhs:comp("rhs",PS+sh.poissonRhs),
			projRed:projPipe("projRed",0),
			projBlack:projPipe("projBlack",1),
			applyGrad:comp("applyGrad",PS+sh.applyGrad),
			flux:comp("flux",PS+sh.computeFlux),
			advectMass:comp("advectMass",PS+sh.advectMass),
			advectTherm:comp("advectTherm",PS+sh.advectTherm),
			advectScalar:comp("advectScalar",PS+sh.advectScalar),
			phaseEos:phasePipe("phaseEos",0),
			phaseSources:phasePipe("phaseSources",1),
			freezeVel:comp("freezeVel",PS+sh.freezeVel),
			conduction:comp("conduction",PS+sh.conduction),
			heatSources:comp("heatSources",PS+sh.heatSources),
			limiterPrepare:comp("limiterPrepare",PS+sh.limiterPrepare),
			limiterApply:comp("limiterApply",PS+sh.limiterApply),
			paintMass:comp("paintMass",PS+sh.paintMass),
			paintTherm:comp("paintTherm",PS+sh.paintTherm),
			mgCoarsen:comp("mgCoarsen",PS+sh.mgCoarsen),
			mgRestrict:comp("mgRestrict",PS+sh.mgRestrict),
			mgZero:comp("mgZero",PS+sh.mgZero),
			mgProlong:comp("mgProlong",PS+sh.mgProlong),
			poissonReduceA:comp("poissonReduceA",PS+sh.poissonReduceA),
		poissonReduceB:comp("poissonReduceB",sh.poissonReduceB),
		statsA:comp("statsA",PS+sh.statsReduceA),
			statsB:comp("statsB",PS+sh.statsReduceB),
			buoySample:comp("buoySample",PS+sh.buoySample),
			probeSample:comp("probeSample",PS+sh.probeSample),
			emitterMass:comp("emitterMass",PS+sh.emitterMass),
			emitterTherm:comp("emitterTherm",PS+sh.emitterTherm),
			emitterVel:comp("emitterVel",PS+sh.emitterVel),
			colliderVel:comp("colliderVel",PS+sh.colliderVel),
			colliderPushMass:comp("colliderPushMass",PS+sh.colliderPushMass),
			colliderPushTherm:comp("colliderPushTherm",PS+sh.colliderPushTherm),
			toDisplay:comp("toDisplay",PS+sh.toDisplay),
			renderPipe:device.createRenderPipeline({
				label:"render",
				layout:"auto",
				vertex:{module:vertMod,entryPoint:"main"},
				fragment:{module:fragMod,entryPoint:"main",targets:[{format:canvasFormat}]},
				primitive:{topology:"triangle-list"}
			}),
			sampler:device.createSampler({label:"linear",magFilter:"linear",minFilter:"linear"}),
			projLayout,
			phaseLayout
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
		const flagsTmpView=viewRes(pool.flagsTmp);
		const sfieldView=viewRes(pool.sfield);
		const sfieldTmpView=viewRes(pool.sfieldTmp);
		const accumView=viewRes(pool.accum);
		const divsrcView=viewRes(pool.divsrc);
		const accumTmpView=viewRes(pool.accumTmp);
		const fluxView=viewRes(pool.flux);
		const limiterAView=viewRes(pool.limiterA);
		const limiterBView=viewRes(pool.limiterB);
		const groups={
			forces:[],advectVel:[],macFwd:[],advectVelMac:[],rhs:[],applyGrad:[],flux:[],advectMass:[],advectTherm:[],advectScalar:[],
			phaseA:[],phaseB:[],freezeTmp:[],freezeMain:[],conduction:[],heatSources:[],
		limiterPrepare:[],limiterApply:[],paintMass:[],paintTherm:[],projRed:[],projBlack:[],
		emitterMass:[],emitterTherm:[],emitterVel:[],colliderVel:[],pushMass:[],pushTherm:[],
		poissonA:[],poissonB:null,statsA:[],toDisplay:[],statsB:null,buoySample:[],probeSample:[],render:null
		};
		for(let mm=0;mm<3;mm++){
			groups.phaseA[mm]=[];
			groups.phaseB[mm]=[];
			groups.conduction[mm]=[];
			groups.heatSources[mm]=[];
			const massV=viewRes(pool.mass.tex[mm]);
			for(let t=0;t<2;t++){
				const tSrc=viewRes(pool.therm.tex[t]),tDst=viewRes(pool.therm.tex[1-t]);
				groups.phaseA[mm][t]=[];
				groups.phaseB[mm][t]=[];
				for(let v=0;v<2;v++){
					const velV=viewRes(pool.vel.tex[v]);
					groups.phaseA[mm][t][v]=bg(device,"phaseA"+mm+t+v,pipes.phaseLayout,[
						{binding:0,resource:{buffer:pool.params}},
						{binding:1,resource:massV},
						{binding:2,resource:tSrc},
						{binding:3,resource:velV},
						{binding:4,resource:flagsView},
						{binding:5,resource:sfieldView},
						{binding:6,resource:accumView},
						{binding:7,resource:tDst},
						{binding:8,resource:flagsTmpView},
						{binding:9,resource:sfieldTmpView},
						{binding:10,resource:accumTmpView}
					]);
					groups.phaseB[mm][t][v]=bg(device,"phaseB"+mm+t+v,pipes.phaseLayout,[
						{binding:0,resource:{buffer:pool.params}},
						{binding:1,resource:massV},
						{binding:2,resource:tSrc},
						{binding:3,resource:velV},
						{binding:4,resource:flagsTmpView},
						{binding:5,resource:sfieldTmpView},
						{binding:6,resource:accumTmpView},
						{binding:7,resource:tDst},
						{binding:8,resource:flagsView},
						{binding:9,resource:sfieldView},
						{binding:10,resource:accumView}
					]);
				}
				groups.conduction[mm][t]=bg(device,"conduction"+mm+t,pipes.conduction.getBindGroupLayout(0),[
					{binding:0,resource:{buffer:pool.params}},
					{binding:1,resource:massV},
					{binding:2,resource:tSrc},
					{binding:3,resource:tDst},
					{binding:4,resource:flagsTmpView}
				]);
				groups.heatSources[mm][t]=bg(device,"heatSources"+mm+t,pipes.heatSources.getBindGroupLayout(0),[
					{binding:0,resource:{buffer:pool.params}},
					{binding:1,resource:massV},
					{binding:2,resource:tSrc},
					{binding:3,resource:tDst},
					{binding:4,resource:flagsTmpView}
				]);
			}
		}
		for(let m=0;m<3;m++){
			const nextM=(m+1)%3,finalM=(m+2)%3;
			const mm=pool.mass.tex[m],ml=pool.mass.tex[nextM];
			const mo=pool.mass.tex[finalM];
			groups.poissonA[m]=bg(device,"poissonA"+m,pipes.poissonReduceA.getBindGroupLayout(0),[
			{binding:0,resource:{buffer:pool.params}},
			{binding:1,resource:viewRes(p0)},
			{binding:2,resource:divsrcView},
			{binding:3,resource:viewRes(mm)},
			{binding:4,resource:flagsView},
			{binding:5,resource:{buffer:pool.statsPart}}
		]);
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
			// the push reads the mass parity it writes: same src/dst pair as paintMass,
			// so state.m advances by one exactly like a paint frame does
			groups.pushMass[m]=bg(device,"pushMass"+m,pipes.colliderPushMass.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(mm)},
				{binding:2,resource:viewRes(ml)},
				{binding:3,resource:viewRes(pool.patchGeo)}
			]);
			groups.emitterMass[m]=bg(device,"emitterMass"+m,pipes.emitterMass.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(mm)},
				{binding:2,resource:viewRes(ml)},
				{binding:3,resource:{buffer:pool.emitters}}
			]);
			groups.advectScalar[m]=[];
			groups.applyGrad[m]=[];
			groups.buoySample[m]=[];
			groups.probeSample[m]=[];
			groups.statsA[m]=[];
			groups.toDisplay[m]=[];
			for(let t=0;t<2;t++){
				const tt=pool.therm.tex[t],tn=pool.therm.tex[1-t];
				groups.advectScalar[m][t]=bg(device,"advectScalar"+m+t,pipes.advectScalar.getBindGroupLayout(0),[
					{binding:0,resource:{buffer:pool.params}},
					{binding:1,resource:fluxView},
					{binding:2,resource:viewRes(mm)},
					{binding:3,resource:viewRes(tt)},
					{binding:4,resource:viewRes(ml)},
					{binding:5,resource:viewRes(tn)}
				]);
				groups.pushTherm[t]=bg(device,"pushTherm"+t,pipes.colliderPushTherm.getBindGroupLayout(0),[
					{binding:0,resource:{buffer:pool.params}},
					{binding:1,resource:viewRes(tt)},
					{binding:2,resource:viewRes(tn)},
					{binding:3,resource:viewRes(pool.patchGeo)}
				]);
				groups.statsA[m][t]=[];
				groups.toDisplay[m][t]=[];
				groups.probeSample[m][t]=[];
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
						groups.applyGrad[m][v]=bg(device,"applyGrad"+m+v,pipes.applyGrad.getBindGroupLayout(0),[
							{binding:0,resource:{buffer:pool.params}},
							{binding:1,resource:viewRes(vv)},
							{binding:2,resource:viewRes(vn)},
							{binding:3,resource:viewRes(p0)},
							{binding:4,resource:viewRes(mm)},
							{binding:5,resource:flagsView}
						]);
						groups.buoySample[m][v]=bg(device,"buoySample"+m+v,pipes.buoySample.getBindGroupLayout(0),[
							{binding:0,resource:{buffer:pool.params}},
							{binding:1,resource:viewRes(mm)},
							{binding:2,resource:viewRes(vv)},
							{binding:3,resource:{buffer:pool.buoy}}
						]);
					}
					// probes read the live parity of every field, so this group is
					// built for all 12 (mass, therm, vel) combinations, not only t=0
					groups.probeSample[m][t][v]=bg(device,"probeSample"+m+t+v,pipes.probeSample.getBindGroupLayout(0),[
						{binding:0,resource:{buffer:pool.params}},
						{binding:1,resource:viewRes(mm)},
						{binding:2,resource:viewRes(tt)},
						{binding:3,resource:viewRes(vv)},
						{binding:4,resource:viewRes(p0)},
						{binding:5,resource:{buffer:pool.probes}}
					]);
					groups.statsA[m][t][v]=bg(device,"statsA"+m+t+v,pipes.statsA.getBindGroupLayout(0),[
						{binding:0,resource:{buffer:pool.params}},
						{binding:1,resource:viewRes(mm)},
						{binding:2,resource:viewRes(tt)},
						{binding:3,resource:viewRes(vv)},
						{binding:4,resource:viewRes(p0)},
						{binding:5,resource:flagsView},
						{binding:6,resource:accumView},
						{binding:7,resource:{buffer:pool.statsPart}},
						{binding:8,resource:divsrcView}
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
			groups.emitterTherm[t]=bg(device,"emitterTherm"+t,pipes.emitterTherm.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(tt)},
				{binding:2,resource:viewRes(tn)},
				{binding:3,resource:{buffer:pool.emitters}}
			]);
		}
		for(let v=0;v<2;v++){
			const vv=pool.vel.tex[v],vn=pool.vel.tex[1-v];
			groups.advectVel[v]=bg(device,"advectVel"+v,pipes.advectVel.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:viewRes(vn)},
				{binding:3,resource:flagsView}
			]);
			// mac-cormack (t16): pass 1 is the semi-lagrangian kernel into the
			// scratch, pass 2 corrects from (src, scratch) into the ping-pong
			// partner. both indexed by the source parity, like advectVel.
			groups.macFwd[v]=bg(device,"macFwd"+v,pipes.advectVel.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:viewRes(pool.macTmp)},
				{binding:3,resource:flagsView}
			]);
			groups.advectVelMac[v]=bg(device,"advectVelMac"+v,pipes.advectVelMac.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:viewRes(pool.macTmp)},
				{binding:3,resource:viewRes(vn)},
				{binding:4,resource:flagsView}
			]);
			groups.rhs[v]=bg(device,"rhs"+v,pipes.rhs.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(pool.divsrc)},
				{binding:2,resource:viewRes(vv)},
				{binding:3,resource:sfieldView},
				{binding:4,resource:flagsView}
			]);
			groups.flux[v]=bg(device,"flux"+v,pipes.flux.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:flagsView},
				{binding:3,resource:fluxView}
			]);
			groups.emitterVel[v]=bg(device,"emitterVel"+v,pipes.emitterVel.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:viewRes(vn)},
				{binding:3,resource:{buffer:pool.emitters}}
			]);
			groups.colliderVel[v]=bg(device,"colliderVel"+v,pipes.colliderVel.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:viewRes(vn)},
				{binding:3,resource:viewRes(pool.patchVel)}
			]);
			groups.freezeTmp[v]=bg(device,"freezeTmp"+v,pipes.freezeVel.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:flagsTmpView},
				{binding:3,resource:viewRes(vn)}
			]);
			groups.freezeMain[v]=bg(device,"freezeMain"+v,pipes.freezeVel.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(vv)},
				{binding:2,resource:flagsView},
				{binding:3,resource:viewRes(vn)}
			]);
		}
		const mgMassView=viewRes(pool.mgMass);
		const mgFlagsView=viewRes(pool.mgFlags);
		const mgDivView=viewRes(pool.mgDiv);
		const mgP0=viewRes(pool.mgPress.tex[0]);
		const mgP1=viewRes(pool.mgPress.tex[1]);
		groups.mgZero=bg(device,"mgZero",pipes.mgZero.getBindGroupLayout(0),[
			{binding:0,resource:{buffer:pool.mgParams}},
			{binding:1,resource:mgP0},
			{binding:2,resource:mgP1}
		]);
		groups.projRedC=bg(device,"projRedC",pipes.projLayout,[
			{binding:0,resource:{buffer:pool.mgParams}},
			{binding:1,resource:mgP0},
			{binding:2,resource:mgP1},
			{binding:3,resource:mgDivView},
			{binding:4,resource:mgMassView},
			{binding:5,resource:mgFlagsView}
		]);
		groups.projBlackC=bg(device,"projBlackC",pipes.projLayout,[
			{binding:0,resource:{buffer:pool.mgParams}},
			{binding:1,resource:mgP1},
			{binding:2,resource:mgP0},
			{binding:3,resource:mgDivView},
			{binding:4,resource:mgMassView},
			{binding:5,resource:mgFlagsView}
		]);
		groups.mgCoarsen=[];
		groups.mgRestrict=[];
		for(let m=0;m<3;m++){
			const mm=viewRes(pool.mass.tex[m]);
			groups.mgCoarsen[m]=bg(device,"mgCoarsen"+m,pipes.mgCoarsen.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:mm},
				{binding:2,resource:flagsView},
				{binding:3,resource:mgMassView},
				{binding:4,resource:mgFlagsView}
			]);
			groups.mgRestrict[m]=bg(device,"mgRestrict"+m,pipes.mgRestrict.getBindGroupLayout(0),[
				{binding:0,resource:{buffer:pool.params}},
				{binding:1,resource:viewRes(p0)},
				{binding:2,resource:viewRes(pool.divsrc)},
				{binding:3,resource:mm},
				{binding:4,resource:flagsView},
				{binding:5,resource:mgDivView}
			]);
		}
		groups.mgProlong=bg(device,"mgProlong",pipes.mgProlong.getBindGroupLayout(0),[
			{binding:0,resource:{buffer:pool.params}},
			{binding:1,resource:mgP0},
			{binding:2,resource:viewRes(p0)},
			{binding:3,resource:flagsView},
			{binding:4,resource:mgFlagsView}
		]);
		groups.poissonB=bg(device,"poissonB",pipes.poissonReduceB.getBindGroupLayout(0),[
			{binding:0,resource:{buffer:pool.statsPart}},
			{binding:1,resource:{buffer:pool.stats}},
			{binding:2,resource:{buffer:pool.dims}}
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
	function encodePrimePhase(cpass,pool,pipes,groups,state){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		const m=state.m;
		let t=state.t,v=state.v;
		cpass.setPipeline(pipes.phaseEos);
		cpass.setBindGroup(0,groups.phaseA[m][t][v]);
		cpass.dispatchWorkgroups(wx,wy);
		t=1-t;
		cpass.setPipeline(pipes.freezeVel);
		cpass.setBindGroup(0,groups.freezeTmp[v]);
		cpass.dispatchWorkgroups(wx,wy);
		v=1-v;
		cpass.setPipeline(pipes.phaseEos);
		cpass.setBindGroup(0,groups.phaseB[m][t][v]);
		cpass.dispatchWorkgroups(wx,wy);
		t=1-t;
		cpass.setPipeline(pipes.freezeVel);
		cpass.setBindGroup(0,groups.freezeMain[v]);
		cpass.dispatchWorkgroups(wx,wy);
		state.t=t;
		state.v=1-v;
	}
	// one substep in five segments; io threads the ping-pong indices through
	// them so encodeStepProfiled can split passes at the segment boundaries
	// (0.6.0 t1) without changing the dispatch sequence.
	const stepIo={m:0,t:0,v:0};
	function encodeVelAdv(cpass,pool,pipes,groups,io,macOn){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		cpass.setPipeline(pipes.forces);
		cpass.setBindGroup(0,groups.forces[io.m][io.t][io.v]);
		cpass.dispatchWorkgroups(wx,wy);
		io.v=1-io.v;
		if(macOn){
			cpass.setPipeline(pipes.advectVel);
			cpass.setBindGroup(0,groups.macFwd[io.v]);
			cpass.dispatchWorkgroups(wx,wy);
			cpass.setPipeline(pipes.advectVelMac);
			cpass.setBindGroup(0,groups.advectVelMac[io.v]);
			cpass.dispatchWorkgroups(wx,wy);
			io.v=1-io.v;
			return;
		}
		cpass.setPipeline(pipes.advectVel);
		cpass.setBindGroup(0,groups.advectVel[io.v]);
		cpass.dispatchWorkgroups(wx,wy);
		io.v=1-io.v;
	}
	function encodeRhs(cpass,pool,pipes,groups,io){
		cpass.setPipeline(pipes.rhs);
		cpass.setBindGroup(0,groups.rhs[io.v]);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
	}
	function encodePoissonReduce(cpass,pool,pipes,groups,io){
		cpass.setPipeline(pipes.poissonReduceA);
		cpass.setBindGroup(0,groups.poissonA[io.m]);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
		cpass.setPipeline(pipes.poissonReduceB);
		cpass.setBindGroup(0,groups.poissonB);
		cpass.dispatchWorkgroups(1);
	}
	function encodePressure(cpass,pool,pipes,groups,io,iters,mgOn,mgPre,mgPost,mgCycles,captureResidual){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		const mx=pool.mgWorkgroupX,my=pool.mgWorkgroupY;
		const m=io.m;
		if(mgOn){
			const cycles=Math.max(1,Math.round(mgCycles||1));
			cpass.setPipeline(pipes.mgCoarsen);
			cpass.setBindGroup(0,groups.mgCoarsen[m]);
			cpass.dispatchWorkgroups(mx,my);
			for(let cy=0;cy<cycles;cy++){
				for(let k=0;k<mgPre;k++){
					cpass.setPipeline(pipes.projRed);
					cpass.setBindGroup(0,groups.projRed[m]);
					cpass.dispatchWorkgroups(wx,wy);
					cpass.setPipeline(pipes.projBlack);
					cpass.setBindGroup(0,groups.projBlack[m]);
					cpass.dispatchWorkgroups(wx,wy);
				}
				cpass.setPipeline(pipes.mgZero);
				cpass.setBindGroup(0,groups.mgZero);
				cpass.dispatchWorkgroups(mx,my);
				cpass.setPipeline(pipes.mgRestrict);
				cpass.setBindGroup(0,groups.mgRestrict[m]);
				cpass.dispatchWorkgroups(mx,my);
				const cIters=iters+(iters&1);
				for(let k=0;k<cIters;k++){
					cpass.setPipeline(pipes.projRed);
					cpass.setBindGroup(0,groups.projRedC);
					cpass.dispatchWorkgroups(mx,my);
					cpass.setPipeline(pipes.projBlack);
					cpass.setBindGroup(0,groups.projBlackC);
					cpass.dispatchWorkgroups(mx,my);
				}
				cpass.setPipeline(pipes.mgProlong);
				cpass.setBindGroup(0,groups.mgProlong);
				cpass.dispatchWorkgroups(wx,wy);
				for(let k=0;k<mgPost;k++){
					cpass.setPipeline(pipes.projRed);
					cpass.setBindGroup(0,groups.projRed[m]);
					cpass.dispatchWorkgroups(wx,wy);
					cpass.setPipeline(pipes.projBlack);
					cpass.setBindGroup(0,groups.projBlack[m]);
					cpass.dispatchWorkgroups(wx,wy);
				}
			}
		} else {
			for(let k=0;k<iters;k++){
				cpass.setPipeline(pipes.projRed);
				cpass.setBindGroup(0,groups.projRed[m]);
				cpass.dispatchWorkgroups(wx,wy);
				cpass.setPipeline(pipes.projBlack);
				cpass.setBindGroup(0,groups.projBlack[m]);
				cpass.dispatchWorkgroups(wx,wy);
			}
		}
		cpass.setPipeline(pipes.applyGrad);
		cpass.setBindGroup(0,groups.applyGrad[m][io.v]);
		cpass.dispatchWorkgroups(wx,wy);
		io.v=1-io.v;
		// after the solve, before transport: mass parity is still the one the
		// projection used. only the sampled frame's last substep asks for this.
		if(captureResidual)encodePoissonReduce(cpass,pool,pipes,groups,io);
	}
	function encodeTransport(cpass,pool,pipes,groups,state,io,fctOn,fuseOn){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		const m=io.m;
		cpass.setPipeline(pipes.flux);
		cpass.setBindGroup(0,groups.flux[io.v]);
		cpass.dispatchWorkgroups(wx,wy);
		if(fuseOn){
			cpass.setPipeline(pipes.advectScalar);
			cpass.setBindGroup(0,groups.advectScalar[m][io.t]);
			cpass.dispatchWorkgroups(wx,wy);
		}else{
			cpass.setPipeline(pipes.advectMass);
			cpass.setBindGroup(0,groups.advectMass[m]);
			cpass.dispatchWorkgroups(wx,wy);
			cpass.setPipeline(pipes.advectTherm);
			cpass.setBindGroup(0,groups.advectTherm[io.t]);
			cpass.dispatchWorkgroups(wx,wy);
		}
		if(fctOn){
			cpass.setPipeline(pipes.limiterPrepare);
			cpass.setBindGroup(0,groups.limiterPrepare[m]);
			cpass.dispatchWorkgroups(wx,wy);
			cpass.setPipeline(pipes.limiterApply);
			cpass.setBindGroup(0,groups.limiterApply[m]);
			cpass.dispatchWorkgroups(wx,wy);
			state.m=(m+2)%3;
		}else{
			state.m=(m+1)%3;
		}
		io.m=state.m;
	}
	function encodePhaseChain(cpass,pool,pipes,groups,state,io,fuseOn){
		// 0.5.0 phase + thermal chain: phaseEos + freezeVel -> conduction ->
		// heatSources -> phaseEos + freezeVel (or phaseSources when fuseOn).
		// phaseA->phaseB keeps flags/sfield/accum in their canonical textures;
		// state.t tracks therm parity (5 flips/substep unfused, 4 when fused).
		const wx=pool.workgroupX,wy=pool.workgroupY;
		const mf=io.m;
		let v=io.v;
		let ts=1-io.t;
		cpass.setPipeline(pipes.phaseEos);
		cpass.setBindGroup(0,groups.phaseA[mf][ts][v]);
		cpass.dispatchWorkgroups(wx,wy);
		ts=1-ts;
		cpass.setPipeline(pipes.freezeVel);
		cpass.setBindGroup(0,groups.freezeTmp[v]);
		cpass.dispatchWorkgroups(wx,wy);
		v=1-v;
		cpass.setPipeline(pipes.conduction);
		cpass.setBindGroup(0,groups.conduction[mf][ts]);
		cpass.dispatchWorkgroups(wx,wy);
		ts=1-ts;
		if(!fuseOn){
			cpass.setPipeline(pipes.heatSources);
			cpass.setBindGroup(0,groups.heatSources[mf][ts]);
			cpass.dispatchWorkgroups(wx,wy);
			ts=1-ts;
		}
		cpass.setPipeline(fuseOn?pipes.phaseSources:pipes.phaseEos);
		cpass.setBindGroup(0,groups.phaseB[mf][ts][v]);
		cpass.dispatchWorkgroups(wx,wy);
		ts=1-ts;
		cpass.setPipeline(pipes.freezeVel);
		cpass.setBindGroup(0,groups.freezeMain[v]);
		cpass.dispatchWorkgroups(wx,wy);
		v=1-v;
		state.t=ts;
		state.v=v;
		io.t=ts;
		io.v=v;
	}
	function encodeStep(cpass,pool,pipes,groups,state,iters,fctOn,mgOn,mgPre,mgPost,mgCycles,captureResidual,macOn,fuseOn){
		stepIo.m=state.m;stepIo.t=state.t;stepIo.v=state.v;
		encodeVelAdv(cpass,pool,pipes,groups,stepIo,macOn);
		encodeRhs(cpass,pool,pipes,groups,stepIo);
		encodePressure(cpass,pool,pipes,groups,stepIo,iters,mgOn,mgPre,mgPost,mgCycles,captureResidual);
		encodeTransport(cpass,pool,pipes,groups,state,stepIo,fctOn,fuseOn);
		encodePhaseChain(cpass,pool,pipes,groups,state,stepIo,fuseOn);
	}
	function encodeStepProfiled(encoder,prof,pool,pipes,groups,state,iters,fctOn,mgOn,mgPre,mgPost,mgCycles,captureResidual,macOn,fuseOn){
		const G=Fluid.gpu.profiler.GROUP;
		stepIo.m=state.m;stepIo.t=state.t;stepIo.v=state.v;
		encodeVelAdv(prof.pass(encoder,G.vel,"vel"),pool,pipes,groups,stepIo,macOn);
		encodeRhs(prof.pass(encoder,G.rhs,"rhs"),pool,pipes,groups,stepIo);
		encodePressure(prof.pass(encoder,G.press,"press"),pool,pipes,groups,stepIo,iters,mgOn,mgPre,mgPost,mgCycles,captureResidual);
		encodeTransport(prof.pass(encoder,G.trans,"trans"),pool,pipes,groups,state,stepIo,fctOn,fuseOn);
		encodePhaseChain(prof.pass(encoder,G.phase,"phase"),pool,pipes,groups,state,stepIo,fuseOn);
	}
	function encodeStats(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.statsA);
		cpass.setBindGroup(0,groups.statsA[state.m][state.t][state.v]);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
		cpass.setPipeline(pipes.statsB);
		cpass.setBindGroup(0,groups.statsB);
		cpass.dispatchWorkgroups(1);
	}
	function encodeBuoySample(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.buoySample);
		cpass.setBindGroup(0,groups.buoySample[state.m][state.v]);
		cpass.dispatchWorkgroups(1);
	}
	// 4 groups of 8 cover the 32 slots; inactive slots return immediately
	function encodeProbeSample(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.probeSample);
		cpass.setBindGroup(0,groups.probeSample[state.m][state.t][state.v]);
		cpass.dispatchWorkgroups(Fluid.core.probes.MAX/8);
	}
	function encodeDisplay(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.toDisplay);
		cpass.setBindGroup(0,groups.toDisplay[state.m][state.t][state.v]);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
	}
	// one pass per field the active sources touch, at the top of the frame: the
	// parities move exactly like encodePaint, so every later pass reads the new field
	function encodeEmitters(cpass,pool,pipes,groups,state,need){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		if(need.needMass){
			cpass.setPipeline(pipes.emitterMass);
			cpass.setBindGroup(0,groups.emitterMass[state.m]);
			cpass.dispatchWorkgroups(wx,wy);
			state.m=(state.m+1)%3;
		}
		if(need.needTherm){
			cpass.setPipeline(pipes.emitterTherm);
			cpass.setBindGroup(0,groups.emitterTherm[state.t]);
			cpass.dispatchWorkgroups(wx,wy);
			state.t=1-state.t;
		}
		if(need.needVel){
			cpass.setPipeline(pipes.emitterVel);
			cpass.setBindGroup(0,groups.emitterVel[state.v]);
			cpass.dispatchWorkgroups(wx,wy);
			state.v=1-state.v;
		}
	}
	// the contents of the cells a body takes go to the fluid around them. parities move
	// exactly like encodePaint, so the prime that follows reads what was just written
	function encodeColliderPush(cpass,pool,pipes,groups,state){
		const wx=pool.workgroupX,wy=pool.workgroupY;
		cpass.setPipeline(pipes.colliderPushMass);
		cpass.setBindGroup(0,groups.pushMass[state.m]);
		cpass.dispatchWorkgroups(wx,wy);
		cpass.setPipeline(pipes.colliderPushTherm);
		cpass.setBindGroup(0,groups.pushTherm[state.t]);
		cpass.dispatchWorkgroups(wx,wy);
		state.m=(state.m+1)%3;
		state.t=1-state.t;
	}
	// the body's velocity on the fluid it just displaced; callers set the patch rect
	// params and upload patchVel first
	function encodeColliderVel(cpass,pool,pipes,groups,state){
		cpass.setPipeline(pipes.colliderVel);
		cpass.setBindGroup(0,groups.colliderVel[state.v]);
		cpass.dispatchWorkgroups(pool.workgroupX,pool.workgroupY);
		state.v=1-state.v;
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
	Fluid.gpu.passes={createPipelines,makeBindGroups,encodePrimePhase,encodeStep,encodeStepProfiled,encodeEmitters,encodeColliderPush,encodeColliderVel,encodeStats,encodeBuoySample,encodeProbeSample,encodeDisplay,encodePaint,encodeRender};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.passes;
})(globalThis);
