(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	const PATCH=128;
	function texUsage(){
		return GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
	}
	function makeTexture(device,label,format,texW,texH){
		return device.createTexture({
			label,
			size:[texW,texH],
			format,
			usage:texUsage()
		});
	}
	function createPool(device,N,mgCoarsen){
		const texW=N+2;
		const texH=N+2;
		const pool={
			N,texW,texH,
			vel:{tex:[],cur:0},
			mass:{tex:[],cur:0},
			therm:{tex:[],cur:0},
			press:{tex:[],cur:0},
			flags:null,
			flux:null,
			limiterA:null,
			limiterB:null,
			divsrc:null,
			sfield:null,
			mgMass:null,
			mgFlags:null,
			mgDiv:null,
			mgPress:{tex:[],cur:0},
			mgParams:null,
			mgNc:0,mgS:0,
			mgTexW:0,
			mgTexH:0,
			mgWorkgroupX:0,
			mgWorkgroupY:0,
			patchMass:null,
			patchE:null,
			display:null,
			params:null,
			stats:null,
			statsPart:null,
			statsRead:null,
			buoy:null,
			buoyRead:null,
			dims:null,
			numWorkgroups:0,
			workgroupX:0,workgroupY:0
		};
		pool.vel.tex[0]=makeTexture(device,"vel0","rg32float",texW,texH);
		pool.vel.tex[1]=makeTexture(device,"vel1","rg32float",texW,texH);
		pool.mass.tex[0]=makeTexture(device,"mass0","rgba32float",texW,texH);
		pool.mass.tex[1]=makeTexture(device,"mass1","rgba32float",texW,texH);
		pool.mass.tex[2]=makeTexture(device,"mass2","rgba32float",texW,texH);
		pool.therm.tex[0]=makeTexture(device,"therm0","rgba32float",texW,texH);
		pool.therm.tex[1]=makeTexture(device,"therm1","rgba32float",texW,texH);
		pool.press.tex[0]=makeTexture(device,"press0","r32float",texW,texH);
		pool.press.tex[1]=makeTexture(device,"press1","r32float",texW,texH);
		pool.flags=makeTexture(device,"flags","rgba8uint",texW,texH);
		pool.flux=makeTexture(device,"flux","rg32float",texW,texH);
		pool.limiterA=makeTexture(device,"limiterA","rgba32float",texW,texH);
		pool.limiterB=makeTexture(device,"limiterB","rg32float",texW,texH);
		pool.divsrc=makeTexture(device,"divsrc","r32float",texW,texH);
		// multigrid coarse level: one level at N/2. every field the coarse RBGS
		// and the residual restriction need, at (N/S+2)^2.
		let mgS=Math.max(2,Math.round(mgCoarsen||2));
		while(mgS>2&&Math.floor(N/mgS)<4)mgS=Math.floor(mgS/2);
		const mgNc=Math.floor(N/mgS);
		const mgTexW=mgNc+2;
		const mgTexH=mgNc+2;
		pool.mgNc=mgNc;pool.mgTexW=mgTexW;pool.mgTexH=mgTexH;pool.mgS=mgS;
		pool.mgWorkgroupX=Math.ceil(mgTexW/8);pool.mgWorkgroupY=Math.ceil(mgTexH/8);
		// mass must stay rgba32float: projectStep reads .a from it, and an
		// r32float texture would return 1.0 in .a (the extension channels are
		// (0,0,0,1)), silently making every coarse face weight 1/rhoVac.
		pool.mgMass=makeTexture(device,"mgMass","rgba32float",mgTexW,mgTexH);
		pool.mgFlags=makeTexture(device,"mgFlags","rgba8uint",mgTexW,mgTexH);
		pool.mgDiv=makeTexture(device,"mgDiv","r32float",mgTexW,mgTexH);
		pool.mgPress.tex[0]=makeTexture(device,"mgPress0","r32float",mgTexW,mgTexH);
		pool.mgPress.tex[1]=makeTexture(device,"mgPress1","r32float",mgTexW,mgTexH);
		pool.mgParams=device.createBuffer({label:"mgParams",size:512,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
		pool.sfield=makeTexture(device,"sfield","r32float",texW,texH);
		pool.patchMass=makeTexture(device,"patchMass","rgba32float",PATCH,PATCH);
		pool.patchE=makeTexture(device,"patchE","r32float",PATCH,PATCH);
		pool.display=makeTexture(device,"display","rgba16float",texW,texH);
		const workgroupX=Math.ceil(texW/8);
		const workgroupY=Math.ceil(texH/8);
		pool.workgroupX=workgroupX;
		pool.workgroupY=workgroupY;
		pool.numWorkgroups=workgroupX*workgroupY;
		pool.params=device.createBuffer({label:"params",size:512,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
		// stats layout: 32 slots (see src/core/stats.js IDX_*)
		const statsPartSize=pool.numWorkgroups*32*4;
		pool.statsPart=device.createBuffer({label:"statsPart",size:statsPartSize,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
		pool.stats=device.createBuffer({label:"stats",size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
		pool.statsRead=device.createBuffer({label:"statsRead",size:128,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
		// buoy_sample output + its readback twin: six floats (see src/core/buoy.js),
		// 128 bytes like the stats pair so both readbacks share one shape
		pool.buoy=device.createBuffer({label:"buoy",size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
		pool.buoyRead=device.createBuffer({label:"buoyRead",size:128,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
		pool.dims=device.createBuffer({label:"dims",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
		const dimsArr=new Uint32Array([pool.numWorkgroups*32,0,0,0]);
		device.queue.writeBuffer(pool.dims,0,dimsArr);
		return pool;
	}
	function destroyPool(pool){
		const tex=[pool.vel.tex[0],pool.vel.tex[1],pool.mass.tex[0],pool.mass.tex[1],pool.mass.tex[2],
			pool.therm.tex[0],pool.therm.tex[1],pool.press.tex[0],pool.press.tex[1],
			pool.flags,pool.flux,pool.limiterA,pool.limiterB,pool.divsrc,pool.sfield,pool.patchMass,pool.patchE,pool.display,
			pool.mgMass,pool.mgFlags,pool.mgDiv,pool.mgPress.tex[0],pool.mgPress.tex[1]];
		for(const t of tex){if(t&&t.destroy)try{t.destroy();}catch(e){}}
		for(const b of [pool.params,pool.stats,pool.statsPart,pool.statsRead,pool.buoy,pool.buoyRead,pool.dims,pool.mgParams]){
			if(b&&b.destroy)try{b.destroy();}catch(e){}
		}
	}
	function swap(pool,name){
		const p=pool[name];
		if(p&&p.cur!==undefined)p.cur=1-p.cur;
	}
	function curTex(pool,name){
		const p=pool[name];
		if(!p)return null;
		if(p.tex)return p.tex[p.cur];
		return p;
	}
	function writeTextureRect(device,texture,texW,texH,format,data){
		let bytesPerRow;
		if(format==="rg32float")bytesPerRow=texW*8;
		else if(format==="r32float")bytesPerRow=texW*4;
		else if(format==="rgba16float")bytesPerRow=texW*8;
		else if(format==="rgba8uint")bytesPerRow=texW*4;
		else bytesPerRow=texW*16;
		device.queue.writeTexture({texture},data,{bytesPerRow},[texW,texH]);
	}
	function writePatch(device,texture,w,h,format,data){
		writeTextureRect(device,texture,w,h,format,data);
	}
	function clearTexture(device,texture,texW,texH,format){
		const size=texW*texH;
		let data;
		if(format==="rgba32float")data=new Float32Array(size*4);
		else if(format==="rg32float")data=new Float32Array(size*2);
		else if(format==="r32float")data=new Float32Array(size);
		else if(format==="rgba16float")data=new Uint16Array(size*4);
		else if(format==="rgba8uint")data=new Uint8Array(size*4);
		else data=new Float32Array(size*4);
		writeTextureRect(device,texture,texW,texH,format,data);
	}
	Fluid.gpu.fields={createPool,destroyPool,swap,curTex,writeTextureRect,writePatch,clearTexture,PATCH};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.fields;
})(globalThis);
