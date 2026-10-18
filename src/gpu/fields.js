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
	function createPool(device,N){
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
			divsrc:null,
			sfield:null,
			patchMass:null,
			patchE:null,
			display:null,
			params:null,
			stats:null,
			statsPart:null,
			statsRead:null,
			dims:null,
			numWorkgroups:0,
			workgroupX:0,workgroupY:0
		};
		pool.vel.tex[0]=makeTexture(device,"vel0","rg32float",texW,texH);
		pool.vel.tex[1]=makeTexture(device,"vel1","rg32float",texW,texH);
		pool.mass.tex[0]=makeTexture(device,"mass0","rgba32float",texW,texH);
		pool.mass.tex[1]=makeTexture(device,"mass1","rgba32float",texW,texH);
		pool.therm.tex[0]=makeTexture(device,"therm0","rgba32float",texW,texH);
		pool.therm.tex[1]=makeTexture(device,"therm1","rgba32float",texW,texH);
		pool.press.tex[0]=makeTexture(device,"press0","r32float",texW,texH);
		pool.press.tex[1]=makeTexture(device,"press1","r32float",texW,texH);
		pool.flags=makeTexture(device,"flags","rgba8uint",texW,texH);
		pool.flux=makeTexture(device,"flux","rg32float",texW,texH);
		pool.divsrc=makeTexture(device,"divsrc","r32float",texW,texH);
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
		pool.dims=device.createBuffer({label:"dims",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
		const dimsArr=new Uint32Array([pool.numWorkgroups*32,0,0,0]);
		device.queue.writeBuffer(pool.dims,0,dimsArr);
		return pool;
	}
	function destroyPool(pool){
		const tex=[pool.vel.tex[0],pool.vel.tex[1],pool.mass.tex[0],pool.mass.tex[1],
			pool.therm.tex[0],pool.therm.tex[1],pool.press.tex[0],pool.press.tex[1],
			pool.flags,pool.flux,pool.divsrc,pool.sfield,pool.patchMass,pool.patchE,pool.display];
		for(const t of tex){if(t&&t.destroy)try{t.destroy();}catch(e){}}
		for(const b of [pool.params,pool.stats,pool.statsPart,pool.statsRead,pool.dims]){
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
