(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const f32=Math.fround;

	// preallocated collider table (0.7.0 t4): 32 stable slot ids owned on the cpu
	// and mirrored into one flat f32 storage buffer. add/move/set/remove mutate
	// existing slot objects and the upload array in place — zero allocations on
	// step/update/move paths.
	//
	// buffer layout in f32 (LEN = 528 floats = 2112 bytes):
	//   [0] live slot count
	//   [1] max active slot index + 1 (kernel loop bound)
	//   [2] active moving slot count (enabled and (vx != 0 || vy != 0))
	//   [3] peak speed max(|vx|, |vy|) across enabled slots (cells/s, for CFL)
	//   [4..15] reserved
	//   slot i at 16 + i*16:
	//     0 x, 1 y, 2 shape (1=circle, 2=rect), 3 p0 (r2 or hx), 4 p1 (r or hy),
	//     5 vx, 6 vy, 7 x0, 8 y0, 9 x1, 10 y1, 11..15 reserved
	const MAX=32;
	const STRIDE=16;
	const HEADER=16;
	const LEN=HEADER+MAX*STRIDE;

	const SHAPE_NONE=0;
	const SHAPE_CIRCLE=1;
	const SHAPE_RECT=2;

	const C_X=0,C_Y=1,C_SHAPE=2,C_P0=3,C_P1=4,C_VX=5,C_VY=6;
	const C_X0=7,C_Y0=8,C_X1=9,C_Y1=10;

	function slotBase(id){
		return HEADER+id*STRIDE;
	}
	function clamp(v,lo,hi){
		if(v<lo)return lo;
		if(v>hi)return hi;
		return v;
	}
	function clampCoord(cfg,v){
		const n=cfg.N||64;
		if(!Number.isFinite(v))return n*0.5;
		return clamp(v,0,n);
	}
	function clampRadius(cfg,r){
		const n=cfg.N||64;
		if(!Number.isFinite(r)||r<=0)return clamp(cfg.paintR||8,0.5,n);
		return clamp(r,0.5,n);
	}
	function clampExtent(cfg,ext){
		const n=cfg.N||64;
		if(!Number.isFinite(ext)||ext<=0)return clamp((cfg.paintR||8)*2,1,n);
		return clamp(ext,1,n);
	}
	function resolveShape(spec,fallback){
		if(spec.shape!==undefined){
			if(spec.shape==="circle")return SHAPE_CIRCLE;
			if(spec.shape==="rect"||spec.shape==="box")return SHAPE_RECT;
			return SHAPE_NONE;
		}
		if(spec.w!==undefined||spec.h!==undefined){
			if(Number.isFinite(spec.w)&&spec.w>0&&Number.isFinite(spec.h)&&spec.h>0)return SHAPE_RECT;
			return SHAPE_NONE;
		}
		if(spec.radius!==undefined){
			if(Number.isFinite(spec.radius)&&spec.radius>0)return SHAPE_CIRCLE;
			return SHAPE_NONE;
		}
		return fallback||SHAPE_CIRCLE;
	}
	function writeSlot(table,cfg,id){
		const s=table.slots[id];
		const u=table.upload;
		const b=slotBase(id);
		if(!s.active||!s.enabled){
			for(let k=0;k<STRIDE;k++)u[b+k]=0;
			return;
		}
		const n=cfg.N||64;
		let p0=0,p1=0,rx=0,ry=0;
		if(s.shape===SHAPE_RECT){
			p0=f32(s.w*0.5);
			p1=f32(s.h*0.5);
			rx=p0;ry=p1;
		}else{
			p0=f32(s.radius*s.radius);
			p1=f32(s.radius);
			rx=s.radius;ry=s.radius;
		}
		const x0=clamp(Math.floor(s.x-rx+0.5),1,n);
		const y0=clamp(Math.floor(s.y-ry+0.5),1,n);
		const x1=clamp(Math.ceil(s.x+rx+0.5),1,n);
		const y1=clamp(Math.ceil(s.y+ry+0.5),1,n);
		u[b+C_X]=f32(s.x);
		u[b+C_Y]=f32(s.y);
		u[b+C_SHAPE]=s.shape;
		u[b+C_P0]=p0;
		u[b+C_P1]=p1;
		u[b+C_VX]=f32(s.vx);
		u[b+C_VY]=f32(s.vy);
		u[b+C_X0]=x0;
		u[b+C_Y0]=y0;
		u[b+C_X1]=x1;
		u[b+C_Y1]=y1;
	}
	function refreshHeader(table){
		const u=table.upload;
		let live=0,maxPlus1=0,moving=0,peak=0;
		for(let i=0;i<MAX;i++){
			const s=table.slots[i];
			if(!s.active)continue;
			live++;
			maxPlus1=i+1;
			if(!s.enabled)continue;
			const ax=Math.abs(s.vx),ay=Math.abs(s.vy);
			if(ax>0||ay>0){
				moving++;
				if(ax>peak)peak=ax;
				if(ay>peak)peak=ay;
			}
		}
		table.live=live;
		table.maxSlotPlus1=maxPlus1;
		table.movingCount=moving;
		table.peakSpeed=f32(peak);
		u[0]=live;
		u[1]=maxPlus1;
		u[2]=moving;
		u[3]=table.peakSpeed;
		table.dirty=true;
		table.maskDirty=true;
	}
	function create(){
		const slots=new Array(MAX);
		for(let i=0;i<MAX;i++){
			slots[i]={
				active:false,enabled:true,shape:SHAPE_NONE,
				x:0,y:0,radius:8,w:16,h:16,vx:0,vy:0
			};
		}
		return {
			live:0,maxSlotPlus1:0,movingCount:0,peakSpeed:0,
			slots,
			upload:new Float32Array(LEN),
			dirty:false,
			maskDirty:false,
			hasGpuMask:false,
			cfgN:-1
		};
	}
	function add(table,cfg,spec){
		if(!spec||typeof spec!=="object")return -1;
		const shape=resolveShape(spec,SHAPE_CIRCLE);
		if(!shape)return -1;
		let id=-1;
		for(let i=0;i<MAX;i++){
			if(!table.slots[i].active){id=i;break;}
		}
		if(id<0)return -1;
		const s=table.slots[id];
		s.active=true;
		s.enabled=spec.enabled!==undefined?Boolean(spec.enabled):true;
		s.shape=shape;
		s.x=clampCoord(cfg,spec.x);
		s.y=clampCoord(cfg,spec.y);
		s.radius=clampRadius(cfg,spec.radius);
		s.w=clampExtent(cfg,spec.w);
		s.h=clampExtent(cfg,spec.h);
		s.vx=Number.isFinite(spec.vx)?spec.vx:0;
		s.vy=Number.isFinite(spec.vy)?spec.vy:0;
		table.cfgN=cfg.N||64;
		writeSlot(table,cfg,id);
		refreshHeader(table);
		return id;
	}
	function move(table,cfg,id,x,y,vx,vy){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		if(!Number.isFinite(x)||!Number.isFinite(y))return false;
		s.x=clampCoord(cfg,x);
		s.y=clampCoord(cfg,y);
		s.vx=Number.isFinite(vx)?vx:0;
		s.vy=Number.isFinite(vy)?vy:0;
		writeSlot(table,cfg,id);
		refreshHeader(table);
		return true;
	}
	function set(table,cfg,id,patch){
		const s=table.slots[id];
		if(!s||!s.active||!patch||typeof patch!=="object")return false;
		if(patch.shape!==undefined||patch.w!==undefined||patch.h!==undefined||patch.radius!==undefined){
			const ns=resolveShape(patch,s.shape);
			if(!ns)return false;
			if(patch.radius!==undefined&&!(Number.isFinite(patch.radius)&&patch.radius>0))return false;
			if((patch.w!==undefined&&!(Number.isFinite(patch.w)&&patch.w>0))||
				(patch.h!==undefined&&!(Number.isFinite(patch.h)&&patch.h>0)))return false;
			s.shape=ns;
		}
		if(patch.x!==undefined){
			if(!Number.isFinite(patch.x))return false;
			s.x=clampCoord(cfg,patch.x);
		}
		if(patch.y!==undefined){
			if(!Number.isFinite(patch.y))return false;
			s.y=clampCoord(cfg,patch.y);
		}
		if(patch.radius!==undefined)s.radius=clampRadius(cfg,patch.radius);
		if(patch.w!==undefined)s.w=clampExtent(cfg,patch.w);
		if(patch.h!==undefined)s.h=clampExtent(cfg,patch.h);
		if(patch.vx!==undefined)s.vx=Number.isFinite(patch.vx)?patch.vx:0;
		if(patch.vy!==undefined)s.vy=Number.isFinite(patch.vy)?patch.vy:0;
		if(patch.enabled!==undefined)s.enabled=Boolean(patch.enabled);
		table.cfgN=cfg.N||64;
		writeSlot(table,cfg,id);
		refreshHeader(table);
		return true;
	}
	function remove(table,id){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		s.active=false;
		s.vx=0;s.vy=0;
		const b=slotBase(id);
		for(let k=0;k<STRIDE;k++)table.upload[b+k]=0;
		refreshHeader(table);
		return true;
	}
	function syncConfig(table,cfg,clampPos){
		const n=cfg.N||64;
		if(table.live===0){
			table.cfgN=n;
			return false;
		}
		if(!clampPos&&table.cfgN===n)return false;
		table.cfgN=n;
		for(let i=0;i<MAX;i++){
			const s=table.slots[i];
			if(!s.active)continue;
			if(clampPos){
				s.x=clampCoord(cfg,s.x);
				s.y=clampCoord(cfg,s.y);
				s.radius=clampRadius(cfg,s.radius);
				s.w=clampExtent(cfg,s.w);
				s.h=clampExtent(cfg,s.h);
			}
			writeSlot(table,cfg,i);
		}
		refreshHeader(table);
		return true;
	}
	function parsePaintKind(kind){
		if(kind===undefined||kind==="wall")return "wall";
		if(kind==="solid"||kind==="erase")return kind;
		return "";
	}
	// writes static wall bit (1) into flags.r across the interior 1..N while
	// preserving moving-collider bits (2|4). supports 1:1 N*N masks and game
	// tilemaps of arbitrary (w, h) resampled by cell centre.
	function applyObstacleMask(flags,texW,texH,maskArray,w,h){
		if(!maskArray||(typeof maskArray!=="object"))return false;
		const len=maskArray.length;
		if(!(len>0))return false;
		const N=texW-2,NH=texH-2;
		let mw=w,mh=h;
		if(mw===undefined&&mh===undefined&&len===N*NH){
			mw=N;mh=NH;
		}
		if(!Number.isFinite(mw)||!Number.isFinite(mh))return false;
		mw=Math.floor(mw);mh=Math.floor(mh);
		if(mw<1||mh<1||len<mw*mh)return false;
		for(let x=0;x<texW;x++){
			flags[x*4]=0;
			flags[(x+(texH-1)*texW)*4]=0;
		}
		for(let y=1;y<texH-1;y++){
			flags[(y*texW)*4]=0;
			flags[((texW-1)+y*texW)*4]=0;
			const my=mh===NH?(y-1):Math.min(mh-1,Math.floor((y-0.5)*mh/NH));
			const row=my*mw;
			for(let x=1;x<texW-1;x++){
				const mx=mw===N?(x-1):Math.min(mw-1,Math.floor((x-0.5)*mw/N));
				const wall=maskArray[row+mx]>0?1:0;
				const gi=(x+y*texW)*4;
				const oldR=flags[gi];
				const collBits=oldR===255?0:(oldR&6);
				flags[gi]=collBits|wall;
			}
		}
		return true;
	}
	// paints or erases static wall bit (1) in cell coordinates (x, y, radius),
	// preserving moving-collider bits (2|4).
	function paintWallDisc(flags,texW,texH,x,y,radius,erase){
		if(!Number.isFinite(x)||!Number.isFinite(y))return false;
		const N=texW-2;
		const r=clamp(Number.isFinite(radius)&&radius>0?radius:8,0.5,N);
		const r2=f32(r*r);
		const px=f32(x),py=f32(y);
		const x0=Math.max(1,Math.floor(x-r+0.5));
		const x1=Math.min(texW-2,Math.ceil(x+r+0.5));
		const y0=Math.max(1,Math.floor(y-r+0.5));
		const y1=Math.min(texH-2,Math.ceil(y+r+0.5));
		for(let cy=y0;cy<=y1;cy++){
			const dy=f32(f32(cy-0.5)-py);
			const dy2=f32(dy*dy);
			for(let cx=x0;cx<=x1;cx++){
				const dx=f32(f32(cx-0.5)-px);
				if(f32(f32(dx*dx)+dy2)>=r2)continue;
				const gi=(cx+cy*texW)*4;
				const oldR=flags[gi];
				const collBits=oldR===255?0:(oldR&6);
				flags[gi]=collBits|(erase?0:1);
			}
		}
		return true;
	}

	Fluid.core.colliders={
		create,add,move,set,remove,syncConfig,slotBase,
		parsePaintKind,applyObstacleMask,paintWallDisc,
		MAX,STRIDE,HEADER,LEN,
		SHAPE_NONE,SHAPE_CIRCLE,SHAPE_RECT,
		C_X,C_Y,C_SHAPE,C_P0,C_P1,C_VX,C_VY,C_X0,C_Y0,C_X1,C_Y1
	};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.colliders;
})(globalThis);
