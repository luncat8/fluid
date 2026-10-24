(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const f32=Math.fround;

	// preallocated emitter table (0.7.0 t3): 32 stable slot ids owned on the cpu
	// and mirrored into one flat f32 storage buffer. add/move/set/remove mutate
	// the existing slot objects and upload array in place — zero allocations on
	// the step/update path or on live moves.
	//
	// buffer layout in f32 (LEN = 528 floats = 2112 bytes):
	//   [0] live slot count
	//   [1] max active slot index + 1 (kernel loop bound)
	//   [2] active momentum slot count
	//   [3] active scalar/thermal slot count (dye, heat, cold, drain, freeze)
	//   [4] active mass/drain slot count (dye, drain)
	//   [5..15] reserved
	//   slot i at 16 + i*16:
	//     0 x, 1 y, 2 invR2, 3 r2, 4 kind, 5 value,
	//     6 dr, 7 dg, 8 db, 9 vx, 10 vy, 11 enthalpy,
	//     12 x0, 13 y0, 14 x1, 15 y1
	const MAX=32;
	const STRIDE=16;
	const HEADER=16;
	const LEN=HEADER+MAX*STRIDE;

	const KIND_NONE=0;
	const KIND_DYE=1;
	const KIND_HEAT=2;
	const KIND_COLD=3;
	const KIND_MOMENTUM=4;
	const KIND_DRAIN=5;
	const KIND_FREEZE=6;
	const KINDS={
		dye:KIND_DYE,
		heat:KIND_HEAT,
		cold:KIND_COLD,
		momentum:KIND_MOMENTUM,
		drain:KIND_DRAIN,
		freeze:KIND_FREEZE
	};
	const KIND_NAMES=["none","dye","heat","cold","momentum","drain","freeze"];

	const E_X=0,E_Y=1,E_INV_R2=2,E_R2=3,E_KIND=4,E_VAL=5;
	const E_DR=6,E_DG=7,E_DB=8,E_VX=9,E_VY=10,E_ENTHALPY=11;
	const E_X0=12,E_Y0=13,E_X1=14,E_Y1=15;

	// DIR_DEFAULT: (0, 1)*value; DIR_VEC: (dirX, dirY)*value; DIR_ANGLE: (cos, sin)*value
	const DIR_DEFAULT=0,DIR_VEC=1,DIR_ANGLE=2;

	function slotBase(id){
		return HEADER+id*STRIDE;
	}
	function parseKind(k){
		if(typeof k==="string")return KINDS[k]||0;
		return 0;
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
	function defaultValue(cfg,kind,dirMode){
		if(kind===KIND_DYE||kind===KIND_DRAIN)return 0.6;
		if(kind===KIND_HEAT||kind===KIND_COLD)return cfg.heatPower||30;
		if(kind===KIND_FREEZE)return (cfg.heatPower||30)*2;
		if(kind===KIND_MOMENTUM)return dirMode===DIR_VEC?1:200;
		return 0;
	}
	function specificEnthalpy(T,cfg){
		if(T<=cfg.TMelt)return cfg.cSol*T;
		if(T<cfg.TBoil)return cfg.cSol*cfg.TMelt+cfg.Lf+cfg.cLiq*(T-cfg.TMelt);
		return cfg.cSol*cfg.TMelt+cfg.Lf+cfg.cLiq*(cfg.TBoil-cfg.TMelt)+cfg.Lv+cfg.cGas*(T-cfg.TBoil);
	}
	function slotEnthalpy(s,cfg){
		const tEps=cfg.TEps||1;
		if(s.kind===KIND_DYE){
			const T=s.target>0?Math.max(tEps,s.target):cfg.TAmb;
			return f32(specificEnthalpy(T,cfg));
		}
		if(s.kind===KIND_HEAT){
			if(!(s.target>0))return 0;
			return f32(specificEnthalpy(Math.max(tEps,s.target),cfg));
		}
		if(s.kind===KIND_COLD){
			const defT=cfg.TMelt>0?cfg.TMelt+2:tEps;
			const T=s.target>0?Math.max(tEps,s.target):Math.max(tEps,defT);
			return f32(specificEnthalpy(T,cfg));
		}
		if(s.kind===KIND_FREEZE){
			const melt=cfg.TMelt>0?cfg.TMelt:273;
			const maxFreezeT=cfg.TMelt>0?Math.max(tEps,cfg.TMelt-1):tEps;
			const rawT=s.target>0?s.target:melt-25;
			const T=clamp(rawT,tEps,maxFreezeT);
			return f32(specificEnthalpy(T,cfg));
		}
		return 0;
	}
	function copyDye(s,dye){
		if(!dye)return;
		if(Array.isArray(dye)||ArrayBuffer.isView(dye)){
			if(dye.length>0&&Number.isFinite(dye[0]))s.dr=clamp(dye[0],0,1);
			if(dye.length>1&&Number.isFinite(dye[1]))s.dg=clamp(dye[1],0,1);
			if(dye.length>2&&Number.isFinite(dye[2]))s.db=clamp(dye[2],0,1);
		}else if(typeof dye==="object"){
			if(Number.isFinite(dye.r))s.dr=clamp(dye.r,0,1);
			if(Number.isFinite(dye.g))s.dg=clamp(dye.g,0,1);
			if(Number.isFinite(dye.b))s.db=clamp(dye.b,0,1);
		}
	}
	function applyDirection(s,spec,isInit){
		if(spec.vx!==undefined||spec.vy!==undefined){
			s.dirMode=DIR_VEC;
			if(spec.vx!==undefined)s.dirX=Number.isFinite(spec.vx)?spec.vx:0;
			if(spec.vy!==undefined)s.dirY=Number.isFinite(spec.vy)?spec.vy:0;
		}else if(spec.dx!==undefined||spec.dy!==undefined){
			s.dirMode=DIR_VEC;
			if(spec.dx!==undefined)s.dirX=Number.isFinite(spec.dx)?spec.dx:0;
			if(spec.dy!==undefined)s.dirY=Number.isFinite(spec.dy)?spec.dy:0;
		}else if(spec.angle!==undefined&&Number.isFinite(spec.angle)){
			s.dirMode=DIR_ANGLE;
			s.angle=spec.angle;
		}else if(isInit){
			s.dirMode=DIR_DEFAULT;
			s.dirX=0;s.dirY=1;s.angle=Math.PI*0.5;
		}
	}
	function writeSlot(table,cfg,id){
		const s=table.slots[id];
		const u=table.upload;
		const b=slotBase(id);
		if(!s.active){
			for(let k=0;k<STRIDE;k++)u[b+k]=0;
			return;
		}
		const n=cfg.N||64;
		const r=s.radius;
		const r2=f32(r*r);
		const invR2=r2>0?f32(1/r2):0;
		const enabled=s.enabled;
		const val=enabled?(s.kind===KIND_MOMENTUM?s.value:Math.max(0,s.value)):0;
		let fx=0,fy=0;
		if(enabled&&s.kind===KIND_MOMENTUM){
			if(s.dirMode===DIR_VEC){fx=f32(val*s.dirX);fy=f32(val*s.dirY);}
			else if(s.dirMode===DIR_ANGLE){fx=f32(val*Math.cos(s.angle));fy=f32(val*Math.sin(s.angle));}
			else{fx=0;fy=f32(val);}
		}
		// interior cell centres sit at (cx - 0.5, cy - 0.5) for cx, cy in 1..N
		const x0=clamp(Math.floor(s.x-r+0.5),1,n);
		const y0=clamp(Math.floor(s.y-r+0.5),1,n);
		const x1=clamp(Math.ceil(s.x+r+0.5),1,n);
		const y1=clamp(Math.ceil(s.y+r+0.5),1,n);
		u[b+E_X]=f32(s.x);
		u[b+E_Y]=f32(s.y);
		u[b+E_INV_R2]=invR2;
		u[b+E_R2]=r2;
		u[b+E_KIND]=s.kind;
		u[b+E_VAL]=f32(val);
		u[b+E_DR]=f32(s.dr);
		u[b+E_DG]=f32(s.dg);
		u[b+E_DB]=f32(s.db);
		u[b+E_VX]=fx;
		u[b+E_VY]=fy;
		u[b+E_ENTHALPY]=slotEnthalpy(s,cfg);
		u[b+E_X0]=x0;
		u[b+E_Y0]=y0;
		u[b+E_X1]=x1;
		u[b+E_Y1]=y1;
	}
	function refreshHeader(table){
		const u=table.upload;
		let live=0,maxPlus1=0,mom=0,scalar=0,md=0;
		for(let i=0;i<MAX;i++){
			const s=table.slots[i];
			if(!s.active)continue;
			live++;
			maxPlus1=i+1;
			if(!s.enabled)continue;
			const b=slotBase(i);
			if(s.kind===KIND_MOMENTUM){
				if(u[b+E_VX]!==0||u[b+E_VY]!==0)mom++;
			}else if(u[b+E_VAL]>0){
				scalar++;
				if(s.kind===KIND_DYE||s.kind===KIND_DRAIN)md++;
			}
		}
		table.live=live;
		table.maxSlotPlus1=maxPlus1;
		table.momentumCount=mom;
		table.scalarCount=scalar;
		table.massDrainCount=md;
		u[0]=live;
		u[1]=maxPlus1;
		u[2]=mom;
		u[3]=scalar;
		u[4]=md;
		table.dirty=true;
	}
	function snapshotCfg(table,cfg){
		table.cfgN=cfg.N;
		table.cfgTAmb=cfg.TAmb;
		table.cfgTMelt=cfg.TMelt;
		table.cfgTBoil=cfg.TBoil;
		table.cfgTEps=cfg.TEps;
		table.cfgCSol=cfg.cSol;
		table.cfgCLiq=cfg.cLiq;
		table.cfgCGas=cfg.cGas;
		table.cfgLf=cfg.Lf;
		table.cfgLv=cfg.Lv;
	}
	function cfgChanged(table,cfg){
		return table.cfgN!==cfg.N||table.cfgTAmb!==cfg.TAmb||
			table.cfgTMelt!==cfg.TMelt||table.cfgTBoil!==cfg.TBoil||table.cfgTEps!==cfg.TEps||
			table.cfgCSol!==cfg.cSol||table.cfgCLiq!==cfg.cLiq||table.cfgCGas!==cfg.cGas||
			table.cfgLf!==cfg.Lf||table.cfgLv!==cfg.Lv;
	}
	function create(){
		const slots=new Array(MAX);
		for(let i=0;i<MAX;i++){
			slots[i]={
				active:false,enabled:true,kind:KIND_NONE,
				x:0,y:0,radius:8,value:0,
				dr:0.95,dg:0.28,db:0.20,
				dirMode:DIR_DEFAULT,dirX:0,dirY:1,angle:Math.PI*0.5,
				target:0
			};
		}
		return {
			live:0,maxSlotPlus1:0,momentumCount:0,scalarCount:0,massDrainCount:0,
			slots,
			upload:new Float32Array(LEN),
			dirty:false,
			cfgN:-1,cfgTAmb:0,cfgTMelt:0,cfgTBoil:0,cfgTEps:0,
			cfgCSol:0,cfgCLiq:0,cfgCGas:0,cfgLf:0,cfgLv:0
		};
	}
	function resetSlotDefaults(s){
		s.enabled=true;
		s.dr=0.95;s.dg=0.28;s.db=0.20;
		s.dirMode=DIR_DEFAULT;s.dirX=0;s.dirY=1;s.angle=Math.PI*0.5;
		s.target=0;
	}
	function add(table,cfg,spec){
		if(!spec||typeof spec!=="object")return -1;
		const kind=parseKind(spec.kind);
		if(!kind)return -1;
		let id=-1;
		for(let i=0;i<MAX;i++){
			if(!table.slots[i].active){id=i;break;}
		}
		if(id<0)return -1;
		const s=table.slots[id];
		resetSlotDefaults(s);
		s.active=true;
		s.kind=kind;
		s.x=clampCoord(cfg,spec.x);
		s.y=clampCoord(cfg,spec.y);
		s.radius=clampRadius(cfg,spec.radius);
		applyDirection(s,spec,true);
		const rawVal=spec.value!==undefined?spec.value:defaultValue(cfg,kind,s.dirMode);
		s.value=Number.isFinite(rawVal)?(kind===KIND_MOMENTUM?rawVal:Math.max(0,rawVal)):0;
		if(spec.enabled!==undefined)s.enabled=Boolean(spec.enabled);
		if(spec.target!==undefined)s.target=Number.isFinite(spec.target)&&spec.target>0?spec.target:0;
		else if(spec.T!==undefined)s.target=Number.isFinite(spec.T)&&spec.T>0?spec.T:0;
		copyDye(s,spec.dye);
		snapshotCfg(table,cfg);
		writeSlot(table,cfg,id);
		refreshHeader(table);
		return id;
	}
	function move(table,cfg,id,x,y){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		if(!Number.isFinite(x)||!Number.isFinite(y))return false;
		s.x=clampCoord(cfg,x);
		s.y=clampCoord(cfg,y);
		writeSlot(table,cfg,id);
		table.dirty=true;
		return true;
	}
	function set(table,cfg,id,patch){
		const s=table.slots[id];
		if(!s||!s.active||!patch||typeof patch!=="object")return false;
		if(patch.kind!==undefined){
			const nk=parseKind(patch.kind);
			if(!nk)return false;
			s.kind=nk;
		}
		if(patch.x!==undefined&&Number.isFinite(patch.x))s.x=clampCoord(cfg,patch.x);
		if(patch.y!==undefined&&Number.isFinite(patch.y))s.y=clampCoord(cfg,patch.y);
		if(patch.radius!==undefined&&Number.isFinite(patch.radius))s.radius=clampRadius(cfg,patch.radius);
		applyDirection(s,patch,false);
		if(patch.value!==undefined&&Number.isFinite(patch.value)){
			s.value=s.kind===KIND_MOMENTUM?patch.value:Math.max(0,patch.value);
		}else if(s.kind!==KIND_MOMENTUM&&s.value<0){
			s.value=0;
		}
		if(patch.enabled!==undefined)s.enabled=Boolean(patch.enabled);
		if(patch.target!==undefined)s.target=Number.isFinite(patch.target)&&patch.target>0?patch.target:0;
		else if(patch.T!==undefined)s.target=Number.isFinite(patch.T)&&patch.T>0?patch.T:0;
		if(patch.dye!==undefined)copyDye(s,patch.dye);
		snapshotCfg(table,cfg);
		writeSlot(table,cfg,id);
		refreshHeader(table);
		return true;
	}
	function remove(table,id){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		s.active=false;
		const b=slotBase(id);
		for(let k=0;k<STRIDE;k++)table.upload[b+k]=0;
		refreshHeader(table);
		return true;
	}
	// re-derive enthalpy and grid bounds when scenario/EOS parameters or N change
	function syncConfig(table,cfg,clampPos){
		if(table.live===0){
			snapshotCfg(table,cfg);
			return false;
		}
		if(!clampPos&&!cfgChanged(table,cfg))return false;
		snapshotCfg(table,cfg);
		for(let i=0;i<MAX;i++){
			const s=table.slots[i];
			if(!s.active)continue;
			if(clampPos){
				s.x=clampCoord(cfg,s.x);
				s.y=clampCoord(cfg,s.y);
				s.radius=clampRadius(cfg,s.radius);
			}
			writeSlot(table,cfg,i);
		}
		refreshHeader(table);
		return true;
	}

	Fluid.core.emitters={
		create,add,move,set,remove,syncConfig,slotBase,specificEnthalpy,
		MAX,STRIDE,HEADER,LEN,
		KIND_NONE,KIND_DYE,KIND_HEAT,KIND_COLD,KIND_MOMENTUM,KIND_DRAIN,KIND_FREEZE,
		KINDS,KIND_NAMES,
		E_X,E_Y,E_INV_R2,E_R2,E_KIND,E_VAL,E_DR,E_DG,E_DB,E_VX,E_VY,E_ENTHALPY,E_X0,E_Y0,E_X1,E_Y1
	};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.emitters;
})(globalThis);
