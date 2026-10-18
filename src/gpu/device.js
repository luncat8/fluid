(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	const formats=["rg32float","rgba32float","rgba8uint","r32float","rgba16float"];
	const limitNames=["maxTextureDimension2D","maxStorageTexturesPerShaderStage","maxSampledTexturesPerShaderStage","maxComputeInvocationsPerWorkgroup","maxComputeWorkgroupStorageSize","maxComputeWorkgroupsPerDimension","maxUniformBufferBindingSize","maxStorageBufferBindingSize","maxBufferSize"];
	const fallback="From the repository root run python3 -m http.server 8000, then open http://localhost:8000/experiments/probe.html. If no adapter is available there either, check browser WebGPU support and GPU settings.";
	async function checkFormat(device,format){
		device.pushErrorScope("validation");
		let texture;
		let failure=null;
		try{
			texture=device.createTexture({
				label:"probe "+format,
				size:[2,2],format,
				usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST
			});
		}catch(error){
			failure=error.message;
		}finally{
			if(texture)texture.destroy();
		}
		const error=await device.popErrorScope();
		return {format,ok:!failure&&!error,error:failure||(error&&error.message)||null};
	}
	async function probe(environment,onError){
		const report={
			timestamp:new Date().toISOString(),
			userAgent:environment.navigator.userAgent,
			protocol:environment.location.protocol,
			isSecureContext:environment.isSecureContext,
			gpuAvailable:!!environment.navigator.gpu,
			status:"running",adapter:null,adapterFeatures:[],deviceFeatures:[],limits:{},formats:[],errors:[]
		};
		let device=null;
		function fail(message){
			report.errors.push(message);
			report.status="failed";
			if(onError)onError(message,report);
		}
		try{
			if(!report.gpuAvailable)throw new Error("navigator.gpu is unavailable. "+fallback);
			const adapter=await environment.navigator.gpu.requestAdapter({powerPreference:"high-performance"});
			if(!adapter)throw new Error("requestAdapter returned null. "+fallback);
			const info=adapter.info||{};
			report.adapter={};
			for(const key of ["vendor","architecture","device","description","isFallbackAdapter"]){
				report.adapter[key]=info[key]===undefined?null:info[key];
			}
			report.adapterFeatures=Array.from(adapter.features).sort();
			device=await adapter.requestDevice({requiredFeatures:[],requiredLimits:{}});
			device.onuncapturederror=event=>fail("Uncaptured WebGPU error: "+event.error.message);
			device.lost.then(info=>{
				if(info.reason!=="destroyed")fail("WebGPU device lost ("+info.reason+"): "+info.message);
			});
			report.deviceFeatures=Array.from(device.features).sort();
			for(const name of limitNames)report.limits[name]=device.limits[name];
			for(const format of formats)report.formats.push(await checkFormat(device,format));
			if(report.formats.some(result=>!result.ok))throw new Error("Required texture format validation failed.");
			if(report.status!=="failed")report.status="passed";
		}catch(error){
			fail(error.message);
		}
		return {device,report};
	}
	async function init(canvas,errorPanel){
		const errors=[];
		function showError(msg){
			errors.push(msg);
			if(errorPanel){
				errorPanel.hidden=false;
				errorPanel.textContent=errors.join("\n\n");
			}
			console.error(msg);
		}
		if(!navigator.gpu){
			showError("navigator.gpu is unavailable. "+fallback);
			return null;
		}
		let adapter=null;
		try{
			adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});
		}catch(e){
			showError("requestAdapter failed: "+e.message);
			return null;
		}
		if(!adapter){
			showError("requestAdapter returned null. "+fallback);
			return null;
		}
		let device=null;
		try{
			device=await adapter.requestDevice({requiredFeatures:[],requiredLimits:{}});
		}catch(e){
			showError("requestDevice failed: "+e.message);
			return null;
		}
		device.onuncapturederror=event=>{
			showError("Uncaptured WebGPU error: "+event.error.message);
		};
		device.lost.then(info=>{
			if(info.reason!=="destroyed")showError("WebGPU device lost ("+info.reason+"): "+info.message);
		});
		const format=navigator.gpu.getPreferredCanvasFormat();
		const context=canvas.getContext("webgpu");
		if(!context){
			showError("Canvas WebGPU context unavailable.");
			return null;
		}
		try{
			context.configure({device,format,alphaMode:"opaque"});
		}catch(e){
			showError("context.configure failed: "+e.message);
			return null;
		}
		const report={
			timestamp:new Date().toISOString(),
			userAgent:navigator.userAgent,
			protocol:location.protocol,
			isSecureContext:isSecureContext,
			adapterInfo:adapter.info||null,
			adapterFeatures:Array.from(adapter.features).sort(),
			deviceFeatures:Array.from(device.features).sort(),
			limits:{},
			format
		};
		for(const name of limitNames)report.limits[name]=device.limits[name];
		for(const f of formats){
			const r=await checkFormat(device,f);
			if(!r.ok)showError("Format "+f+" failed: "+r.error);
		}
		return {adapter,device,context,format,report,errors};
	}
	Fluid.gpu.device={probe,init,formats,fallback,limitNames};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.device;
})(globalThis);
