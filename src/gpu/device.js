(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	function showError(errors,errorPanel,msg){
		errors.push(msg);
		if(errorPanel){
			errorPanel.hidden=false;
			errorPanel.textContent=errors.join("\n\n");
		}
		console.error(msg);
	}
	async function init(canvas,errorPanel){
		const errors=[];
		if(!navigator.gpu){
			showError(errors,errorPanel,"navigator.gpu is unavailable. WebGPU is required; no fallback exists.");
			return null;
		}
		let adapter=null;
		try{
			adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});
		}catch(e){
			showError(errors,errorPanel,"requestAdapter failed: "+e.message);
			return null;
		}
		if(!adapter){
			showError(errors,errorPanel,"requestAdapter returned null. Check browser WebGPU support and GPU settings.");
			return null;
		}
		let device=null;
		try{
			device=await adapter.requestDevice({requiredFeatures:[],requiredLimits:{}});
		}catch(e){
			showError(errors,errorPanel,"requestDevice failed: "+e.message);
			return null;
		}
		// silent unless wired; validation errors are invisible otherwise
		device.onuncapturederror=event=>{
			showError(errors,errorPanel,"Uncaptured WebGPU error: "+event.error.message);
		};
		device.lost.then(info=>{
			if(info.reason!=="destroyed")showError(errors,errorPanel,"WebGPU device lost ("+info.reason+"): "+info.message);
		});
		const format=navigator.gpu.getPreferredCanvasFormat();
		const context=canvas.getContext("webgpu");
		if(!context){
			showError(errors,errorPanel,"Canvas WebGPU context unavailable.");
			return null;
		}
		try{
			context.configure({device,format,alphaMode:"opaque"});
		}catch(e){
			showError(errors,errorPanel,"context.configure failed: "+e.message);
			return null;
		}
		return {adapter,device,context,format,errors};
	}
	Fluid.gpu.device={init};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.device;
})(globalThis);
