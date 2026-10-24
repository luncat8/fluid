(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};

	// Fixed game-facing source table. The GPU applies scalar sources before thermal
	// ones; the CPU oracle uses the same order so clamped drains and energy floors
	// have one deterministic meaning. All conserved deltas are measured from the
	// stored f32 states, never from the requested rate alone.
	const MAX=32;
	const STRIDE=16;
	const HEADER=16;
	const LEN=HEADER+MAX*STRIDE;
	const LEDGER_LEN=20;
	const STATS_OFFSET=36;
	const H_LIVE=0,H_EAMB=1;
	const E_X=0,E_Y=1,E_RADIUS=2,E_KIND=3,E_VALUE=4,E_DR=5,E_DG=6,E_DB=7,E_DX=8,E_DY=9,E_TARGET=10,E_H=11,E_INVR2=12,E_ACTIVE=15;
	const KIND_DYE=0,KIND_DRAIN=1,KIND_HEAT=2,KIND_COLD=3,KIND_FREEZE=4,KIND_MOMENTUM=5;
	const kindNames=["dye","drain","heat","cold","freeze","momentum"];
	const DEFAULT_RADIUS=8;
	const f32=Math.fround;

	function slotBase(id){return HEADER+id*STRIDE;}
	function makeFrame(){
		return {inM:0,inDr:0,inDg:0,inDb:0,inE:0,
			outM:0,outDr:0,outDg:0,outDb:0,outE:0,heatIn:0,heatOut:0,
			needScalar:false,needThermal:false,needMomentum:false,any:false};
	}
	function create(){
		const table={live:0,momentumLive:0,eAmb:0,slots:new Array(MAX),upload:new Float32Array(LEN),frame:makeFrame()};
		for(let i=0;i<MAX;i++){
			table.slots[i]={active:false,kind:0,x:0,y:0,radius:0,value:0,dr:1,dg:1,db:1,dx:0,dy:1,target:0,h:0,invR2:1};
		}
		return table;
	}
	function kindOf(kind){
		if(typeof kind==="number")return Number.isInteger(kind)&&kind>=0&&kind<kindNames.length?kind:-1;
		return kindNames.indexOf(kind);
	}
	function isScalar(kind){return kind===KIND_DYE||kind===KIND_DRAIN;}
	function isThermal(kind){return kind===KIND_HEAT||kind===KIND_COLD||kind===KIND_FREEZE;}
	function clampCoord(cfg,v){
		if(!(v>1))return 1;
		return f32(Math.min(v,cfg.N));
	}
	function clampRadius(cfg,r){
		if(!(r>0.5))return 0.5;
		return f32(Math.min(r,cfg.N));
	}
	function number(v){return Number.isFinite(+v)?+v:0;}
	function nonNegative(v){return Math.max(0,number(v));}
	function specificEnergy(s,cfg){
		const eos=Fluid.core.solver.energyFromT;
		if(s.kind===KIND_DYE)return f32(eos(1,s.target>0?s.target:cfg.TRef,cfg));
		if(s.kind===KIND_HEAT)return s.target>0?f32(eos(1,s.target,cfg)):0;
		if(s.kind===KIND_COLD)return f32(eos(1,Math.max(cfg.TMelt+2,s.target),cfg));
		if(s.kind===KIND_FREEZE){
			const t=s.target>0?s.target:cfg.TMelt-25;
			return f32(eos(1,Math.max(cfg.TEps,Math.min(cfg.TMelt-1,t)),cfg));
		}
		return 0;
	}
	function writeSlot(table,cfg,id){
		const s=table.slots[id],b=slotBase(id),u=table.upload;
		s.h=specificEnergy(s,cfg);
		u[b+E_X]=s.x;u[b+E_Y]=s.y;u[b+E_RADIUS]=s.radius;u[b+E_KIND]=s.kind;
		u[b+E_VALUE]=s.value;u[b+E_DR]=s.dr;u[b+E_DG]=s.dg;u[b+E_DB]=s.db;
		u[b+E_DX]=s.dx;u[b+E_DY]=s.dy;u[b+E_TARGET]=s.target;u[b+E_H]=s.h;
		s.invR2=f32(1/f32(s.radius*s.radius));
		u[b+E_INVR2]=s.invR2;
		u[b+E_ACTIVE]=s.active?1:0;
		u[H_LIVE]=table.live;
	}
	function header(table,cfg){
		table.eAmb=f32(Fluid.core.solver.energyFromT(1,cfg.TRef,cfg));
		table.upload[H_LIVE]=table.live;
		table.upload[H_EAMB]=table.eAmb;
		for(let i=0;i<MAX;i++)if(table.slots[i].active)writeSlot(table,cfg,i);
		return table.eAmb;
	}
	function setTint(s,dye){
		if(!dye){s.dr=1;s.dg=1;s.db=1;return;}
		s.dr=f32(Math.min(1,nonNegative(dye[0])));
		s.dg=f32(Math.min(1,nonNegative(dye[1])));
		s.db=f32(Math.min(1,nonNegative(dye[2])));
	}
	function setDir(s,dir){
		if(!dir){s.dx=0;s.dy=1;return;}
		s.dx=f32(number(dir[0]));s.dy=f32(number(dir[1]));
	}
	function add(table,cfg,spec){
		if(!spec)return -1;
		const kind=kindOf(spec.kind===undefined?KIND_DYE:spec.kind);
		if(kind<0)return -1;
		let id=-1;
		for(let i=0;i<MAX;i++)if(!table.slots[i].active){id=i;break;}
		if(id<0)return -1;
		const s=table.slots[id];
		s.active=true;s.kind=kind;
		s.x=clampCoord(cfg,spec.x);s.y=clampCoord(cfg,spec.y);
		s.radius=clampRadius(cfg,spec.radius===undefined?DEFAULT_RADIUS:spec.radius);
		s.value=f32(nonNegative(spec.value));
		s.target=f32(nonNegative(spec.target));
		setTint(s,spec.dye);setDir(s,spec.dir);
		table.live++;
		if(kind===KIND_MOMENTUM)table.momentumLive++;
		writeSlot(table,cfg,id);
		return id;
	}
	function move(table,cfg,id,x,y){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		s.x=clampCoord(cfg,x);s.y=clampCoord(cfg,y);
		writeSlot(table,cfg,id);
		return true;
	}
	function set(table,cfg,id,patch){
		const s=table.slots[id];
		if(!s||!s.active||!patch)return false;
		let kind=s.kind;
		if(patch.kind!==undefined){
			kind=kindOf(patch.kind);
			if(kind<0)return false;
		}
		if(kind!==s.kind){
			if(s.kind===KIND_MOMENTUM)table.momentumLive--;
			if(kind===KIND_MOMENTUM)table.momentumLive++;
			s.kind=kind;
		}
		if(patch.x!==undefined)s.x=clampCoord(cfg,patch.x);
		if(patch.y!==undefined)s.y=clampCoord(cfg,patch.y);
		if(patch.radius!==undefined)s.radius=clampRadius(cfg,patch.radius);
		if(patch.value!==undefined)s.value=f32(nonNegative(patch.value));
		if(patch.target!==undefined)s.target=f32(nonNegative(patch.target));
		if(patch.dye)setTint(s,patch.dye);
		if(patch.dir)setDir(s,patch.dir);
		writeSlot(table,cfg,id);
		return true;
	}
	function remove(table,id){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		table.live--;
		if(s.kind===KIND_MOMENTUM)table.momentumLive--;
		s.active=false;
		table.upload[slotBase(id)+E_ACTIVE]=0;
		table.upload[H_LIVE]=table.live;
		return true;
	}
	// Pure quadratic radial falloff, deliberately without sqrt or division per cell.
	function falloffInv(cx,cy,ex,ey,invR2){
		const dx=f32(f32(cx)-ex),dy=f32(f32(cy)-ey);
		const d2=f32(f32(dx*dx)+f32(dy*dy));
		const t=Math.max(0,f32(1-f32(d2*invR2)));
		return f32(t*t);
	}
	function falloff(cx,cy,ex,ey,r){return falloffInv(cx,cy,ex,ey,f32(1/f32(r*r)));}
	function amount(s,dt,w){return f32(f32(s.value*dt)*w);}
	function resetFrame(out){
		out.inM=0;out.inDr=0;out.inDg=0;out.inDb=0;out.inE=0;
		out.outM=0;out.outDr=0;out.outDg=0;out.outDb=0;out.outE=0;
		out.heatIn=0;out.heatOut=0;
		out.needScalar=false;out.needThermal=false;out.needMomentum=false;out.any=false;
	}
	function classify(table,dt,out){
		resetFrame(out);
		if(!(dt>0))return out;
		for(let i=0;i<MAX;i++){
			const s=table.slots[i];
			if(!s.active)continue;
			if(isScalar(s.kind)){out.needScalar=true;out.needThermal=true;}
			else if(isThermal(s.kind))out.needThermal=true;
			else if(s.kind===KIND_MOMENTUM)out.needMomentum=true;
		}
		out.any=out.needScalar||out.needThermal;
		return out;
	}
	function prepare(table,cfg,dtFrame){
		header(table,cfg);
		return classify(table,dtFrame,table.frame);
	}
	function wallAt(fields,x,y,tw){return fields.flags&&fields.flags[(x+y*tw)*4]>0;}
	function addDye(s,amt,mass,therm,i,out){
		const om=mass[i+3],odr=mass[i],odg=mass[i+1],odb=mass[i+2],oe=therm[i];
		mass[i]=f32(odr+f32(s.dr*amt));mass[i+1]=f32(odg+f32(s.dg*amt));
		mass[i+2]=f32(odb+f32(s.db*amt));mass[i+3]=f32(om+amt);
		therm[i]=f32(oe+f32(amt*s.h));
		out.inM+=f32(mass[i+3]-om);out.inDr+=f32(mass[i]-odr);
		out.inDg+=f32(mass[i+1]-odg);out.inDb+=f32(mass[i+2]-odb);out.inE+=f32(therm[i]-oe);
	}
	function drain(s,amt,mass,therm,i,out){
		const om=mass[i+3];
		if(!(om>0))return;
		const odr=mass[i],odg=mass[i+1],odb=mass[i+2],oe=therm[i];
		if(amt>=om){
			mass[i]=0;mass[i+1]=0;mass[i+2]=0;mass[i+3]=0;therm[i]=0;
		}else{
			const scale=Math.max(0,f32(1-f32(amt/om)));
			mass[i]=f32(odr*scale);mass[i+1]=f32(odg*scale);mass[i+2]=f32(odb*scale);
			mass[i+3]=f32(om*scale);therm[i]=f32(oe*scale);
		}
		out.outM+=f32(om-mass[i+3]);out.outDr+=f32(odr-mass[i]);
		out.outDg+=f32(odg-mass[i+1]);out.outDb+=f32(odb-mass[i+2]);out.outE+=f32(oe-therm[i]);
	}
	function thermal(s,amt,mass,therm,i,out,cfg){
		const m=mass[i+3];
		if(m<cfg.mEps)return;
		const old=therm[i];
		let next=old;
		if(s.kind===KIND_HEAT){
			const room=s.target>0?Math.max(0,f32(f32(m*s.h)-old)):amt;
			next=f32(old+Math.min(amt,room));
		}else{
			const available=Math.max(0,f32(old-f32(m*s.h)));
			next=f32(old-Math.min(amt,available));
		}
		const d=f32(next-old);
		therm[i]=next;
		if(d>0){out.heatIn+=d;therm[i+2]=f32(therm[i+2]+d);}
		else if(d<0){out.heatOut-=d;therm[i+3]=f32(therm[i+3]-d);}
	}
	// CPU oracle and backend hook. Scalar slots run first, then thermal slots; the
	// tally shader repeats that order over the union of scalar bounding boxes.
	function sweep(table,cfg,dtFrame,fields,out){
		classify(table,dtFrame,out);
		if(!out.any||!fields)return out;
		const dt=f32(dtFrame),tw=fields.texW,th=fields.texH;
		const mass=fields.mass,therm=fields.therm;
		for(let pass=0;pass<2;pass++){
			for(let i=0;i<MAX;i++){
				const s=table.slots[i];
				if(!s.active||(pass===0?!isScalar(s.kind):!isThermal(s.kind)))continue;
				const r=s.radius;
				const x0=Math.max(1,Math.ceil(s.x-r)),x1=Math.min(tw-2,Math.floor(s.x+r));
				const y0=Math.max(1,Math.ceil(s.y-r)),y1=Math.min(th-2,Math.floor(s.y+r));
				for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){
					if(wallAt(fields,x,y,tw))continue;
					const w=falloffInv(x,y,s.x,s.y,s.invR2);
					if(w<=0)continue;
					const at=(x+y*tw)*4,amt=amount(s,dt,w);
					if(pass===0){
						if(s.kind===KIND_DYE)addDye(s,amt,mass,therm,at,out);
						else drain(s,amt,mass,therm,at,out);
					}else thermal(s,amt,mass,therm,at,out,cfg);
				}
			}
		}
		return out;
	}
	function momentum(table,cfg,x,y,axis){
		let a=0;
		for(let i=0;i<MAX;i++){
			const s=table.slots[i];
			if(!s.active||s.kind!==KIND_MOMENTUM)continue;
			const w=falloffInv(x,y,s.x,s.y,s.invR2);
			if(w>0)a=f32(a+f32(s.value*w*(axis===0?s.dx:s.dy)));
		}
		return a;
	}
	Fluid.core.emitters={create,add,move,set,remove,header,prepare,sweep,falloff,falloffInv,momentum,kindOf,
		kindNames,isScalar,isThermal,KIND_DYE,KIND_DRAIN,KIND_HEAT,KIND_COLD,KIND_FREEZE,KIND_MOMENTUM,
		MAX,STRIDE,HEADER,LEN,LEDGER_LEN,STATS_OFFSET,H_LIVE,H_EAMB,
		E_X,E_Y,E_RADIUS,E_KIND,E_VALUE,E_DR,E_DG,E_DB,E_DX,E_DY,E_TARGET,E_H,E_INVR2,E_ACTIVE,slotBase};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.emitters;
})(globalThis);
