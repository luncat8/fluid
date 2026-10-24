(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};

	// game-facing continuous sources (0.7.0 t3): a fixed table of preallocated slots,
	// copied into one storage buffer and applied once per frame by one pass per field.
	//
	// the contract that keeps the conservation meters meaningful: every kind adds a
	// quantity the cpu can sum over exactly the same cells with the same falloff, so
	// injected/removed totals are computed from the gpu's own numbers instead of an
	// analytic approximation of them. drain and freeze are absent because how much
	// they take depends on the cell's own mass/energy, which the cpu cannot know
	// without a readback; they need a gpu-side sink reduction first (plan t3b).
	//
	// storage layout in f32:
	//   [0]      live slot count
	//   [1]      specific energy of the substance at TRef (energy per unit mass)
	//   slot i   at 16 + i*16:
	//            0 x, 1 y, 2 radius, 3 kind, 4 value, 5 dr, 6 dg, 7 db,
	//            8 dirX, 9 dirY, 10..14 unused, 15 active flag
	// ids are slot indices and stay valid until remove, like the probe table.
	const MAX=8;
	const STRIDE=16;
	const HEADER=16;
	const LEN=HEADER+MAX*STRIDE;
	const H_LIVE=0,H_EAMB=1;
	const E_X=0,E_Y=1,E_RADIUS=2,E_KIND=3,E_VALUE=4,E_DR=5,E_DG=6,E_DB=7,E_DX=8,E_DY=9,E_ACTIVE=15;
	// kinds, in the units the fields already store:
	//   dye       mass per second at the centre, carrying that mass at ambient energy,
	//             tinted by dr/dg/db (the same thing the pointer paints, continuously)
	//   heat      energy per second
	//   cold      energy per second removed
	//   momentum  velocity per second along dirX/dirY (a force: work, never a source)
	const KIND_DYE=0,KIND_HEAT=1,KIND_COLD=2,KIND_MOMENTUM=3;
	const kindNames=["dye","heat","cold","momentum"];
	const DEFAULT_RADIUS=8;

	function slotBase(id){
		return HEADER+id*STRIDE;
	}
	function create(){
		const table={
			live:0,
			eAmb:0,
			slots:new Array(MAX),
			upload:new Float32Array(LEN),
			// the last prepare() result; the solver reads it before the next call
			frame:{inM:0,inDr:0,inDg:0,inDb:0,inE:0,outE:0,needMass:false,needTherm:false,needVel:false,any:false}
		};
		for(let i=0;i<MAX;i++){
			table.slots[i]={active:false,kind:0,x:0,y:0,radius:0,value:0,dr:1,dg:1,db:1,dx:0,dy:1};
		}
		return table;
	}
	function kindOf(kind){
		if(typeof kind==="number"){
			if(!(kind>=0&&kind<kindNames.length))return -1;
			return Math.round(kind);
		}
		return kindNames.indexOf(kind);
	}
	// emitters live in cell coordinates like probes: x right, y up, interior 1..N
	function clampCoord(cfg,v){
		if(!(v>1))return 1;
		if(v>cfg.N)return cfg.N;
		return v;
	}
	function clampRadius(cfg,r){
		if(!(r>0.5))return 0.5;
		if(r>cfg.N)return cfg.N;
		return r;
	}
	function writeSlot(table,id){
		const s=table.slots[id],b=slotBase(id),u=table.upload;
		u[b+E_X]=s.x;
		u[b+E_Y]=s.y;
		u[b+E_RADIUS]=s.radius;
		u[b+E_KIND]=s.kind;
		u[b+E_VALUE]=s.value;
		u[b+E_DR]=s.dr;
		u[b+E_DG]=s.dg;
		u[b+E_DB]=s.db;
		u[b+E_DX]=s.dx;
		u[b+E_DY]=s.dy;
		u[b+E_ACTIVE]=s.active?1:0;
		u[H_LIVE]=table.live;
	}
	// TRef is the ambient temperature the wgsl EOS uses (evalPhase reads values[20]),
	// so dye mass is injected at exactly the energy that EOS calls ambient: one
	// definition of the caloric curve, no second constant to keep in sync.
	function header(table,cfg){
		table.eAmb=Fluid.core.solver.energyFromT(1,cfg.TRef,cfg);
		table.upload[H_LIVE]=table.live;
		table.upload[H_EAMB]=table.eAmb;
		return table.eAmb;
	}
	function add(table,cfg,spec){
		if(!spec)return -1;
		const kind=kindOf(spec.kind===undefined?KIND_DYE:spec.kind);
		if(kind<0)return -1;
		let id=-1;
		for(let i=0;i<MAX;i++){
			if(table.slots[i].active)continue;
			id=i;
			break;
		}
		if(id<0)return -1;
		const s=table.slots[id];
		s.active=true;
		s.kind=kind;
		s.x=clampCoord(cfg,spec.x);
		s.y=clampCoord(cfg,spec.y);
		s.radius=clampRadius(cfg,spec.radius===undefined?DEFAULT_RADIUS:spec.radius);
		s.value=+spec.value||0;
		setTint(s,spec.dye);
		setDir(s,spec.dir);
		table.live++;
		writeSlot(table,id);
		return id;
	}
	function setTint(s,dye){
		if(!dye){
			s.dr=1;s.dg=1;s.db=1;
			return;
		}
		s.dr=+dye[0]||0;
		s.dg=+dye[1]||0;
		s.db=+dye[2]||0;
	}
	function setDir(s,dir){
		if(!dir){
			s.dx=0;s.dy=1;
			return;
		}
		s.dx=+dir[0]||0;
		s.dy=+dir[1]||0;
	}
	function move(table,cfg,id,x,y){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		s.x=clampCoord(cfg,x);
		s.y=clampCoord(cfg,y);
		writeSlot(table,id);
		return true;
	}
	// patch copies what it names and leaves the rest; moving x/y is move(), because
	// that is the call the frame loop makes for a game object that walks.
	function set(table,cfg,id,patch){
		const s=table.slots[id];
		if(!s||!s.active||!patch)return false;
		if(patch.kind!==undefined){
			const kind=kindOf(patch.kind);
			if(kind<0)return false;
			s.kind=kind;
		}
		if(patch.x!==undefined)s.x=clampCoord(cfg,patch.x);
		if(patch.y!==undefined)s.y=clampCoord(cfg,patch.y);
		if(patch.radius!==undefined)s.radius=clampRadius(cfg,patch.radius);
		if(patch.value!==undefined)s.value=+patch.value||0;
		if(patch.dye)setTint(s,patch.dye);
		if(patch.dir)setDir(s,patch.dir);
		writeSlot(table,id);
		return true;
	}
	function remove(table,id){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		table.live--;
		s.active=false;
		writeSlot(table,id);
		table.upload[H_LIVE]=table.live;
		return true;
	}
	// the kernel's weight: f32 arithmetic because the kernel runs in f32, and the
	// inclusion test uses the same rounded d2 so a cell on the rim is never counted
	// by the ledger and skipped by the gpu.
	function falloff(cx,cy,ex,ey,r){
		const dx=Math.fround(cx-ex),dy=Math.fround(cy-ey);
		const d2=Math.fround(Math.fround(dx*dx)+Math.fround(dy*dy));
		if(!(d2<Math.fround(r*r)))return 0;
		const f=Math.fround(1-Math.fround(Math.sqrt(d2)/r));
		return Math.fround(f*f);
	}
	// one sweep over the live slots and their cells. writes the exact injected and
	// removed totals into `out`, and when `fields` is given also adds the same
	// numbers to a cpu field: that is the oracle the wgsl kernel is checked against.
	function sweep(table,cfg,dtFrame,fields,out){
		out.inM=0;out.inDr=0;out.inDg=0;out.inDb=0;out.inE=0;out.outE=0;
		out.needMass=false;out.needTherm=false;out.needVel=false;
		out.any=false;
		if(!(dtFrame>0))return out;
		const tw=fields?fields.texW:cfg.texW,th=fields?fields.texH:cfg.texH;
		const mass=fields?fields.mass:null,therm=fields?fields.therm:null,vel=fields?fields.vel:null;
		const eAmb=table.eAmb;
		for(let i=0;i<MAX;i++){
			const s=table.slots[i];
			if(!s.active)continue;
			const kind=s.kind,r=s.radius;
			if(kind===KIND_DYE){
				out.needMass=true;
				out.needTherm=true;
			}else if(kind===KIND_HEAT||kind===KIND_COLD){
				out.needTherm=true;
			}else{
				out.needVel=true;
			}
			const x0=Math.max(1,Math.ceil(s.x-r)),x1=Math.min(tw-2,Math.floor(s.x+r));
			const y0=Math.max(1,Math.ceil(s.y-r)),y1=Math.min(th-2,Math.floor(s.y+r));
			for(let y=y0;y<=y1;y++){
				for(let x=x0;x<=x1;x++){
					const fall=falloff(x,y,s.x,s.y,r);
					if(fall<=0)continue;
					if(kind===KIND_DYE){
						const dm=Math.fround(s.value*fall*dtFrame);
						out.inM+=dm;
						out.inDr+=s.dr*dm;
						out.inDg+=s.dg*dm;
						out.inDb+=s.db*dm;
						const dE=dm*eAmb;
						out.inE+=dE;
						if(mass){
							const i4=(x+y*tw)*4;
							mass[i4]+=s.dr*dm;
							mass[i4+1]+=s.dg*dm;
							mass[i4+2]+=s.db*dm;
							mass[i4+3]+=dm;
							therm[i4]+=dE;
						}
					}else if(kind===KIND_HEAT||kind===KIND_COLD){
						const dE=Math.fround((kind===KIND_HEAT?s.value:-s.value)*fall*dtFrame);
						if(dE>0)out.inE+=dE;
						else out.outE-=dE;
						if(therm)therm[(x+y*tw)*4]+=dE;
					}else{
						const dv=Math.fround(s.value*fall*dtFrame);
						if(vel){
							const i2=(x+y*tw)*2;
							vel[i2]+=s.dx*dv;
							vel[i2+1]+=s.dy*dv;
						}
					}
				}
			}
		}
		out.any=out.needMass||out.needTherm||out.needVel;
		return out;
	}
	// the frame path: header + this frame's exact totals, no allocation
	function prepare(table,cfg,dtFrame){
		header(table,cfg);
		return sweep(table,cfg,dtFrame,null,table.frame);
	}
	Fluid.core.emitters={create,add,move,set,remove,header,prepare,sweep,falloff,kindOf,
		kindNames,KIND_DYE,KIND_HEAT,KIND_COLD,KIND_MOMENTUM,
		MAX,STRIDE,HEADER,LEN,H_LIVE,H_EAMB,
		E_X,E_Y,E_RADIUS,E_KIND,E_VALUE,E_DR,E_DG,E_DB,E_DX,E_DY,E_ACTIVE,slotBase};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.emitters;
})(globalThis);
