(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	// cpu twins of the to_display wgsl colormaps, for legends and tests. the
	// shader is the authority; if these disagree, fix these.
	function tempToRgb(t,tmin,tmax){
		const u=Math.max(0,Math.min(1,(t-tmin)/Math.max(tmax-tmin,1e-6)));
		if(u<0.25){const a=u/0.25;return [0,a,1];}
		if(u<0.5){const a=(u-0.25)/0.25;return [0,1,1-a];}
		if(u<0.75){const a=(u-0.5)/0.25;return [a,1,0];}
		const a=(u-0.75)/0.25;return [1,1-a*0.5,0];
	}
	function pressureToRgb(p,pmin,pmax){
		const u=Math.max(-1,Math.min(1,(p-pmin)/Math.max(pmax-pmin,1e-6)*2-1));
		return [Math.max(u,0),0.12+0.1*(1-Math.abs(u)),Math.max(-u,0)];
	}
	function velocityToRgb(vx,vy,vmax){
		const sp=Math.min(1,Math.sqrt(vx*vx+vy*vy)/Math.max(vmax,1));
		const hue=(Math.atan2(vy,vx)/(2*Math.PI)+0.5)%1;
		const h2rgb=h=>{
			const f=(x)=>Math.max(0,Math.min(1,Math.abs(((x%1)+1)%1*6-3)-1));
			return [f(h),f(h-1/3),f(h+1/3)];
		};
		const c=h2rgb(hue);
		return [c[0]*sp+0.02,c[1]*sp+0.02,c[2]*sp+0.02];
	}
	Fluid.gpu.view={tempToRgb,pressureToRgb,velocityToRgb};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.view;
})(globalThis);
