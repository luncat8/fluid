(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	// view helpers reused by toDisplay colormaps; for 0.1.0 the shader does the mapping,
	// this file documents ranges and provides cpu-side colormap for legend (if needed)
	function tempToRgb(t,tmin,tmax){
		let u=Math.max(0,Math.min(1,(t-tmin)/Math.max(tmax-tmin,1e-6)));
		let r,g,b;
		if(u<0.25){ let a=u/0.25; r=0; g=a; b=1; }
		else if(u<0.5){ let a=(u-0.25)/0.25; r=0; g=1; b=1-a; }
		else if(u<0.75){ let a=(u-0.5)/0.25; r=a; g=1; b=0; }
		else{ let a=(u-0.75)/0.25; r=1; g=1-a*0.5; b=0; }
		return [r,g,b];
	}
	Fluid.gpu.view={tempToRgb};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.view;
})(globalThis);
