(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
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
			flags:{tex:[],cur:0},
			flux:null,
			divsrc:null,
			sfield:null,
			limiter:null,
			display:null,
			params:null,
			stats:null,
			statsPart:null,
			statsRead:null,
			statsStaging:null,
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
		pool.flags.tex[0]=makeTexture(device,"flags0","rgba8uint",texW,texH);
		pool.flags.tex[1]=makeTexture(device,"flags1","rgba8uint",texW,texH);
		pool.flux=makeTexture(device,"flux","rg32float",texW,texH);
		pool.divsrc=makeTexture(device,"divsrc","r32float",texW,texH);
		pool.sfield=makeTexture(device,"sfield","r32float",texW,texH);
		pool.limiter=makeTexture(device,"limiter","rg32float",texW,texH);
		pool.display=makeTexture(device,"display","rgba16float",texW,texH);
		const workgroupX=Math.ceil(texW/8);
		const workgroupY=Math.ceil(texH/8);
		pool.workgroupX=workgroupX;
		pool.workgroupY=workgroupY;
		pool.numWorkgroups=workgroupX*workgroupY;
		pool.params=device.createBuffer({label:"params",size:512,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
		const statsPartSize=pool.numWorkgroups*16*4;
		pool.statsPart=device.createBuffer({label:"statsPart",size:statsPartSize,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
		pool.stats=device.createBuffer({label:"stats",size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
		pool.statsRead=device.createBuffer({label:"statsRead",size:64,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
		pool.dims=device.createBuffer({label:"dims",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
		const dimsArr=new Uint32Array([pool.numWorkgroups*16,0,0,0]);
		device.queue.writeBuffer(pool.dims,0,dimsArr);
		return pool;
	}
	function destroyPool(pool){
		const all=[pool.vel.tex[0],pool.vel.tex[1],pool.mass.tex[0],pool.mass.tex[1],pool.therm.tex[0],pool.therm.tex[1],pool.press.tex[0],pool.press.tex[1],pool.flags.tex[0],pool.flags.tex[1],pool.flux,pool.divsrc,pool.sfield,pool.limiter,pool.display,pool.params,pool.stats,pool.statsPart,pool.statsRead,pool.dims];
		for(let t of all){ if(t&&t.destroy)try{t.destroy();}catch(e){} else if(t&&t.destroy===undefined&&t.label){} }
		// buffers need destroy too
		for(let b of [pool.params,pool.stats,pool.statsPart,pool.statsRead,pool.dims]){ if(b&&b.destroy)try{b.destroy();}catch(e){} }
	}
	function swap(pool,name){
		let p=pool[name];
		if(p&&p.cur!==undefined) p.cur=1-p.cur;
	}
	function curTex(pool,name){
		let p=pool[name];
		if(!p) return null;
		if(p.tex) return p.tex[p.cur];
		return p;
	}
	function nxtTex(pool,name){
		let p=pool[name];
		if(!p) return null;
		if(p.tex) return p.tex[1-p.cur];
		return null;
	}
	function writeTextureRect(device,texture,texW,texH,format,data){
		// data is Float32Array or Uint8Array sized texW*texH*components
		let bytesPerElement=4;
		if(format==="rgba8uint") bytesPerElement=1;
		else if(format==="rgba16float") bytesPerElement=2;
		let bytesPerRow=texW*4*bytesPerElement; // for rgba32float 16 per texel; but for rg32float 8, for r32float 4, for rgba8uint 4
		if(format==="rg32float") bytesPerRow=texW*8;
		else if(format==="r32float") bytesPerRow=texW*4;
		else if(format==="rgba16float") bytesPerRow=texW*8;
		else if(format==="rgba8uint") bytesPerRow=texW*4;
		device.queue.writeTexture({texture},data,{bytesPerRow},[texW,texH]);
	}
	function clearTexture(device,texture,texW,texH,format){
		let size=texW*texH;
		let data;
		if(format==="rgba32float") data=new Float32Array(size*4);
		else if(format==="rg32float") data=new Float32Array(size*2);
		else if(format==="r32float") data=new Float32Array(size);
		else if(format==="rgba16float") data=new Uint16Array(size*4);
		else if(format==="rgba8uint") data=new Uint8Array(size*4);
		else data=new Float32Array(size*4);
		writeTextureRect(device,texture,texW,texH,format,data);
	}
	Fluid.gpu.fields={createPool,destroyPool,swap,curTex,nxtTex,writeTextureRect,clearTexture};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.fields;
})(globalThis);
