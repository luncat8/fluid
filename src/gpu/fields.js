(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.gpu=Fluid.gpu||{};
	const PATCH=128;
	// the stats layout is owned by src/core/stats.js so the cpu oracle, the readback
	// and these buffers cannot disagree on the slot count
	const STATS=Fluid.core.stats;
	// probe table layout (slot count, stride, total floats) is owned by src/core/probes.js
	const PROBES=Fluid.core.probes;
	function texUsage(){
		return GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
	}
	// bytes per texel of every format the pool uses or could use (the 16-bit variants are
	// what 0.6.0 t7 would switch to; the memory audit has to price them before they exist)
	function texBpp(format){
		if(format==="r32float"||format==="rgba8uint")return 4;
		if(format==="rg32float"||format==="rgba16float")return 8;
		if(format==="r16float")return 2;
		if(format==="rg16float")return 4;
		return 16; // rgba32float
	}
	function makeTexture(device,label,format,texW,texH){
		const tex=device.createTexture({
			label,
			size:[texW,texH],
			format,
			usage:texUsage()
		});
		// the memory audit needs width/height/format back; a GPUTexture is opaque in
		// the spec, so we remember what we asked for instead of reading the object
		if(tex)tex.fluidSize=[texW,texH,format];
		return tex;
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
			flagsTmp:null,
			flux:null,
			limiterA:null,
			limiterB:null,
			divsrc:null,
			sfield:null,
			sfieldTmp:null,
			accum:null,
			accumTmp:null,
			mgMass:null,
			mgFlags:null,
			mgDiv:null,
			macTmp:null,
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
			probes:null,
			probesRead:null,
			emitters:null,
			emitterLedger:null,
			colliders:null,
			dims:null,
			numWorkgroups:0,
			workgroupX:0,workgroupY:0
		};
		pool.vel.tex[0]=makeTexture(device,"vel0","rg32float",texW,texH);
		pool.vel.tex[1]=makeTexture(device,"vel1","rg32float",texW,texH);
		// mac-cormack forward pass scratch (t16): written by the plain
		// semi-lagrangian kernel, read by the correction kernel. allocated
		// with the pool like every other field; unused while advectMac is 0.
		pool.macTmp=makeTexture(device,"macTmp","rg32float",texW,texH);
		pool.mass.tex[0]=makeTexture(device,"mass0","rgba32float",texW,texH);
		pool.mass.tex[1]=makeTexture(device,"mass1","rgba32float",texW,texH);
		pool.mass.tex[2]=makeTexture(device,"mass2","rgba32float",texW,texH);
		pool.therm.tex[0]=makeTexture(device,"therm0","rgba32float",texW,texH);
		pool.therm.tex[1]=makeTexture(device,"therm1","rgba32float",texW,texH);
		pool.press.tex[0]=makeTexture(device,"press0","r32float",texW,texH);
		pool.press.tex[1]=makeTexture(device,"press1","r32float",texW,texH);
		pool.flags=makeTexture(device,"flags","rgba8uint",texW,texH);
		pool.flagsTmp=makeTexture(device,"flagsTmp","rgba8uint",texW,texH);
		pool.flux=makeTexture(device,"flux","rg32float",texW,texH);
		pool.limiterA=makeTexture(device,"limiterA","rgba32float",texW,texH);
		pool.limiterB=makeTexture(device,"limiterB","rg32float",texW,texH);
		pool.divsrc=makeTexture(device,"divsrc","r32float",texW,texH);
		let mgS=Math.max(2,Math.round(mgCoarsen||2));
		while(mgS>2&&Math.floor(N/mgS)<4)mgS=Math.floor(mgS/2);
		const mgNc=Math.floor(N/mgS);
		const mgTexW=mgNc+2;
		const mgTexH=mgNc+2;
		pool.mgNc=mgNc;pool.mgTexW=mgTexW;pool.mgTexH=mgTexH;pool.mgS=mgS;
		pool.mgWorkgroupX=Math.ceil(mgTexW/8);pool.mgWorkgroupY=Math.ceil(mgTexH/8);
		pool.mgMass=makeTexture(device,"mgMass","rgba32float",mgTexW,mgTexH);
		pool.mgFlags=makeTexture(device,"mgFlags","rgba8uint",mgTexW,mgTexH);
		pool.mgDiv=makeTexture(device,"mgDiv","r32float",mgTexW,mgTexH);
		pool.mgPress.tex[0]=makeTexture(device,"mgPress0","r32float",mgTexW,mgTexH);
		pool.mgPress.tex[1]=makeTexture(device,"mgPress1","r32float",mgTexW,mgTexH);
		pool.mgParams=device.createBuffer({label:"mgParams",size:512,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
		// 0.5.0: sfield holds (S, phi_l, phi_g, 0); accum holds (latentChargedE, frozenMomentum)
		pool.sfield=makeTexture(device,"sfield","rgba32float",texW,texH);
		pool.sfieldTmp=makeTexture(device,"sfieldTmp","rgba32float",texW,texH);
		pool.accum=makeTexture(device,"accum","rg32float",texW,texH);
		pool.accumTmp=makeTexture(device,"accumTmp","rg32float",texW,texH);
		pool.patchMass=makeTexture(device,"patchMass","rgba32float",PATCH,PATCH);
		pool.patchE=makeTexture(device,"patchE","r32float",PATCH,PATCH);
		pool.display=makeTexture(device,"display","rgba16float",texW,texH);
		const workgroupX=Math.ceil(texW/8);
		const workgroupY=Math.ceil(texH/8);
		pool.workgroupX=workgroupX;
		pool.workgroupY=workgroupY;
		pool.numWorkgroups=workgroupX*workgroupY;
		pool.params=device.createBuffer({label:"params",size:512,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
		const PARTIAL_STRIDE=STATS.PARTIAL_STRIDE;
		// 40 partial floats per workgroup reduced into STATS_LEN output floats
		const statsPartSize=pool.numWorkgroups*PARTIAL_STRIDE*4;
		pool.statsPart=device.createBuffer({label:"statsPart",size:statsPartSize,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
		const statsBytes=STATS.STATS_LEN*4;
		pool.stats=device.createBuffer({label:"stats",size:statsBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
		pool.statsRead=device.createBuffer({label:"statsRead",size:statsBytes,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
		pool.buoy=device.createBuffer({label:"buoy",size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
		pool.buoyRead=device.createBuffer({label:"buoyRead",size:128,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
		const probeBytes=PROBES.LEN*4;
		pool.probes=device.createBuffer({label:"probes",size:probeBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
		pool.probesRead=device.createBuffer({label:"probesRead",size:probeBytes,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
		const emitBytes=(Fluid.core.emitters?Fluid.core.emitters.LEN:528)*4;
		pool.emitters=device.createBuffer({label:"emitters",size:emitBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
		const emitLedgerBytes=(STATS.EMIT_LEDGER_LEN||20)*4;
		pool.emitterLedger=device.createBuffer({label:"emitterLedger",size:emitLedgerBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
		const collBytes=(Fluid.core.colliders?Fluid.core.colliders.LEN:528)*4;
		pool.colliders=device.createBuffer({label:"colliders",size:collBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
		pool.dims=device.createBuffer({label:"dims",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
		const dimsArr=new Uint32Array([pool.numWorkgroups*PARTIAL_STRIDE,0,0,0]);
		device.queue.writeBuffer(pool.dims,0,dimsArr);
		return pool;
	}
	// every allocation the pool owns, in one place: destroy and the memory audit both
	// read these lists, so a new field cannot be destroyed-but-uncounted or the reverse.
	function poolTextures(pool){
		return [pool.vel.tex[0],pool.vel.tex[1],pool.macTmp,pool.mass.tex[0],pool.mass.tex[1],pool.mass.tex[2],
			pool.therm.tex[0],pool.therm.tex[1],pool.press.tex[0],pool.press.tex[1],
			pool.flags,pool.flagsTmp,pool.flux,pool.limiterA,pool.limiterB,pool.divsrc,
			pool.sfield,pool.sfieldTmp,pool.accum,pool.accumTmp,
			pool.patchMass,pool.patchE,pool.display,
			pool.mgMass,pool.mgFlags,pool.mgDiv,pool.mgPress.tex[0],pool.mgPress.tex[1]];
	}
	function poolBuffers(pool){
		return [pool.params,pool.stats,pool.statsPart,pool.statsRead,pool.buoy,pool.buoyRead,pool.probes,pool.probesRead,pool.emitters,pool.emitterLedger,pool.colliders,pool.dims,pool.mgParams];
	}
	function textureBytes(t){
		if(!t||!t.fluidSize)return 0;
		return t.fluidSize[0]*t.fluidSize[1]*texBpp(t.fluidSize[2])*(t.depthOrArrayLayers||1);
	}
	// Informational allocation estimate for the field pool. WebGPU has no allocation
	// query, so logical is what we asked for. `padded` assumes a 256-byte row pitch.
	// Not counted: transient writeTexture staging arrays, the GPU profiler's small
	// resolve/read buffers, the canvas backbuffer, and CPU mirrors freed after upload.
	function memoryReport(pool){
		if(!pool)return null;
		let textures=0,texCount=0,buffers=0,bufCount=0,paddedTex=0;
		for(const t of poolTextures(pool)){
			const b=textureBytes(t);
			if(!b)continue;
			const h=t.fluidSize[1];
			textures+=b;texCount++;
			paddedTex+=Math.ceil(b/h/256)*256*h;
		}
		if(!texCount)return null;
		for(const b of poolBuffers(pool)){
			if(!b||!b.size)continue;
			buffers+=b.size;bufCount++;
		}
		return {textures,buffers,total:textures+buffers,padded:paddedTex+buffers,texCount,bufCount};
	}
	function destroyPool(pool){
		for(const t of poolTextures(pool)){if(t&&t.destroy)try{t.destroy();}catch(e){}}
		for(const b of poolBuffers(pool)){if(b&&b.destroy)try{b.destroy();}catch(e){}}
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
		// writeTexture, unlike copyBufferToTexture, has no 256 row rule: tight pitch is legal
		device.queue.writeTexture({texture},data,{bytesPerRow:texW*texBpp(format)},[texW,texH]);
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
	Fluid.gpu.fields={createPool,destroyPool,swap,curTex,writeTextureRect,writePatch,clearTexture,memoryReport,poolTextures,poolBuffers,texBpp,PATCH};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.gpu.fields;
})(globalThis);
