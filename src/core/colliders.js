(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};

	// solid bodies a game moves through the fluid (0.7.0 t4). the geometry is analytic
	// and cpu-owned, so the mask is rasterized into a per-cell owner index and only the
	// cells that changed are uploaded: a body that does not move costs nothing, and the
	// delta is exact instead of a gpu reduction the cpu would have to read back.
	//
	// shapes: "circle" (radius) and "box" (half extents hw/hh). positions are cell
	// coordinates like probes and sources: x right, y up, interior 1..N.
	//
	// a collider owns the wall flag only while it covers the cell; releasing a cell
	// restores whatever the static mask says, so a painted wall cannot be erased by a
	// body driving over it. velocities are read at the moment a cell is newly occupied
	// and imprinted on the fluid next to it - that, and the explicit conservative scalar
	// push on supported moves, is what makes a body push water instead of sliding through
	// it. A teleport wider than the push patch skips displacement rather than inventing
	// substance.
	const MAX=8;
	const SHAPE_CIRCLE=0,SHAPE_BOX=1;
	const shapeNames=["circle","box"];
	const DEFAULT_RADIUS=8;
	// Radius-2 disk stencil, in a fixed order shared by geometry and displacement.
	// Keeping both coordinates in the arrays avoids direction-dependent neighbor bugs.
	const NX=[0,-1,0,1,-2,-1,1,2,-1,0,1,0];
	const NY=[-2,-1,-1,-1,0,0,0,0,1,1,1,2];

	function create(){
		const table={live:0,slots:new Array(MAX)};
		for(let i=0;i<MAX;i++){
			table.slots[i]={active:false,shape:0,x:0,y:0,radius:0,hw:0,hh:0,vx:0,vy:0};
		}
		return table;
	}
	function shapeOf(shape){
		if(typeof shape==="number"){
			if(!(shape>=0&&shape<shapeNames.length))return -1;
			return Math.round(shape);
		}
		return shapeNames.indexOf(shape);
	}
	function clampCoord(cfg,v){
		if(!(v>1))return 1;
		if(v>cfg.N)return cfg.N;
		return v;
	}
	function clampSize(cfg,v,dflt){
		const s=v===undefined?dflt:v;
		if(!(s>0.5))return 0.5;
		if(s>cfg.N)return cfg.N;
		return s;
	}
	function add(table,cfg,spec){
		if(!spec)return -1;
		const shape=shapeOf(spec.shape===undefined?SHAPE_CIRCLE:spec.shape);
		if(shape<0)return -1;
		let id=-1;
		for(let i=0;i<MAX;i++){
			if(table.slots[i].active)continue;
			id=i;
			break;
		}
		if(id<0)return -1;
		const s=table.slots[id];
		s.active=true;
		s.shape=shape;
		s.x=clampCoord(cfg,spec.x);
		s.y=clampCoord(cfg,spec.y);
		s.radius=clampSize(cfg,spec.radius,DEFAULT_RADIUS);
		s.hw=clampSize(cfg,spec.hw,DEFAULT_RADIUS);
		s.hh=clampSize(cfg,spec.hh,DEFAULT_RADIUS);
		s.vx=+spec.vx||0;
		s.vy=+spec.vy||0;
		table.live++;
		return id;
	}
	function move(table,cfg,id,x,y,vx,vy){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		s.x=clampCoord(cfg,x);
		s.y=clampCoord(cfg,y);
		if(vx!==undefined)s.vx=+vx||0;
		if(vy!==undefined)s.vy=+vy||0;
		return true;
	}
	function set(table,cfg,id,patch){
		const s=table.slots[id];
		if(!s||!s.active||!patch)return false;
		if(patch.shape!==undefined){
			const shape=shapeOf(patch.shape);
			if(shape<0)return false;
			s.shape=shape;
		}
		if(patch.x!==undefined)s.x=clampCoord(cfg,patch.x);
		if(patch.y!==undefined)s.y=clampCoord(cfg,patch.y);
		if(patch.radius!==undefined)s.radius=clampSize(cfg,patch.radius,s.radius);
		if(patch.hw!==undefined)s.hw=clampSize(cfg,patch.hw,s.hw);
		if(patch.hh!==undefined)s.hh=clampSize(cfg,patch.hh,s.hh);
		if(patch.vx!==undefined)s.vx=+patch.vx||0;
		if(patch.vy!==undefined)s.vy=+patch.vy||0;
		return true;
	}
	function remove(table,id){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		s.active=false;
		s.vx=0;s.vy=0;
		table.live--;
		return true;
	}
	// every live body into `owner` (0 = free, id+1 = the lowest slot that covers the
	// cell). the bounds actually written come back through `rect`; the caller clears
	// that rect before the call so the array is never swept in full.
	function rasterize(table,cfg,owner,rect){
		const tw=cfg.texW,th=cfg.texH;
		let bx0=Infinity,by0=Infinity,bx1=-Infinity,by1=-Infinity;
		for(let i=0;i<MAX;i++){
			const s=table.slots[i];
			if(!s.active)continue;
			let x0,x1,y0,y1;
			if(s.shape===SHAPE_CIRCLE){
				const r=s.radius;
				x0=Math.max(1,Math.ceil(s.x-r));x1=Math.min(tw-2,Math.floor(s.x+r));
				y0=Math.max(1,Math.ceil(s.y-r));y1=Math.min(th-2,Math.floor(s.y+r));
				for(let y=y0;y<=y1;y++){
					for(let x=x0;x<=x1;x++){
						const dx=Math.fround(x-s.x),dy=Math.fround(y-s.y);
						if(!(Math.fround(dx*dx+dy*dy)<=Math.fround(r*r)))continue;
						const k=x+y*tw;
						if(!owner[k])owner[k]=i+1;
					}
				}
			}else{
				const hw=s.hw,hh=s.hh;
				x0=Math.max(1,Math.ceil(s.x-hw));x1=Math.min(tw-2,Math.floor(s.x+hw));
				y0=Math.max(1,Math.ceil(s.y-hh));y1=Math.min(th-2,Math.floor(s.y+hh));
				for(let y=y0;y<=y1;y++){
					for(let x=x0;x<=x1;x++){
						if(!(Math.abs(x-s.x)<=hw&&Math.abs(y-s.y)<=hh))continue;
						const k=x+y*tw;
						if(!owner[k])owner[k]=i+1;
					}
				}
			}
			if(x1<x0||y1<y0)continue;
			if(x0<bx0)bx0=x0;
			if(y0<by0)by0=y0;
			if(x1>bx1)bx1=x1;
			if(y1>by1)by1=y1;
		}
		if(bx1<bx0||by1<by0)return emptyRect(rect);
		rect.x0=bx0;rect.y0=by0;rect.x1=bx1;rect.y1=by1;
		return rect;
	}
	function emptyRect(rect){
		rect.x0=1;rect.y0=1;rect.x1=0;rect.y1=0;
		return rect;
	}
	// Radius-2 margin around a rect, clamped to the interior. The push sends contents
	// up to two cells from newly occupied cells, so all receivers need to be uploaded.
	function padRect(cfg,rect){
		rect.x0=Math.max(1,rect.x0-2);rect.y0=Math.max(1,rect.y0-2);
		rect.x1=Math.min(cfg.texW-2,rect.x1+2);rect.y1=Math.min(cfg.texH-2,rect.y1+2);
		return rect;
	}
	// what a body takes, and from whom (0.7.0 t4b). fills the patch-local geometry the
	// two push passes read, in the third channel of a 4-float stride:
	//   > 0  a cell the body takes this frame, carrying its count of free neighbours
	//   -1   a cell that was already body (it keeps what it holds)
	//    0   free: this cell gathers the share of every neighbour that was just taken
	// `rect` is the patch the caller will upload, padded (padRect): a taken cell whose
	// radius-2 receivers are not all inside the rect keeps its contents instead of losing
	// the share that could not be delivered, so no rect mistake can destroy substance.
	// the cell's contents are split equally between its free radius-2 neighbours. returns how
	// many cells were taken.
	// `pitch` is the row stride of the patch buffer: the gpu passes read a PATCH-wide
	// texture, so the solver asks for PATCH and the cpu-only callers can stay tight.
	function geometry(cfg,owner,prev,rect,geo,pitch){
		const tw=cfg.texW,th=cfg.texH;
		const x0=rect.x0,y0=rect.y0,x1=rect.x1,y1=rect.y1;
		const w=x1-x0+1,h=y1-y0+1;
		if(w<=0||h<=0)return 0;
		const stride=pitch||w;
		geo.fill(0,0,stride*h*4);
		let taken=0;
		for(let y=y0;y<=y1;y++){
			for(let x=x0;x<=x1;x++){
				const i=x+y*tw;
				if(!owner[i])continue;
				const p=((x-x0)+(y-y0)*stride)*4+2;
				if(prev[i]){geo[p]=-1;continue;}
				let deg=0,inside=true;
				for(let n=0;n<NX.length;n++){
					const nx=x+NX[n],ny=y+NY[n];
					// the ghost ring is not a receiver
					if(nx<1||nx>tw-2||ny<1||ny>th-2)continue;
					if(nx<x0||nx>x1||ny<y0||ny>y1){inside=false;break;}
					if(!owner[nx+ny*tw])deg++;
				}
				if(!inside||deg===0){geo[p]=-1;continue;}
				geo[p]=deg;
				taken++;
			}
		}
		return taken;
	}
	// the cpu half of the push, cell for cell the same walk the shaders run: a taken
	// cell is emptied and each free neighbour adds its share of it. reads the pre-push
	// contents through `scratch` (which must hold the rect), so no order matters.
	// temperature is derived: the emptied cell gets the reference temperature and the
	// next phase pass reads it back off the caloric curve.
	function displace(cfg,geo,rect,mass,therm,scratch,pitch){
		const tw=cfg.texW;
		const x0=rect.x0,y0=rect.y0,w=rect.x1-x0+1,h=rect.y1-y0+1;
		const stride=pitch||w;
		const cells=stride*h;
		const sm=scratch.mass,st=scratch.therm;
		for(let y=0;y<h;y++){
			for(let x=0;x<w;x++){
				const si=((x0+x)+(y0+y)*tw)*4,pi=(x+y*stride)*4;
				sm[pi]=mass[si];sm[pi+1]=mass[si+1];sm[pi+2]=mass[si+2];sm[pi+3]=mass[si+3];
				st[pi]=therm[si];st[pi+1]=therm[si+1];st[pi+2]=therm[si+2];st[pi+3]=therm[si+3];
			}
		}
		const code=(x,y)=>{
			const cx=x-x0,cy=y-y0;
			if(cx<0||cy<0||cx>=w||cy>=h)return 0;
			return geo[(cx+cy*stride)*4+2];
		};
		const tAmb=cfg.TRef;
		for(let y=0;y<h;y++){
			for(let x=0;x<w;x++){
				const i=(x0+x)+(y0+y)*tw,pi=(x+y*stride)*4;
				const c=code(x0+x,y0+y);
				if(c>=0.5){
					mass[i*4]=0;mass[i*4+1]=0;mass[i*4+2]=0;mass[i*4+3]=0;
					therm[i*4]=0;therm[i*4+1]=tAmb;therm[i*4+2]=0;therm[i*4+3]=0;
					continue;
				}
				if(c<=-0.5)continue;
				let m0=sm[pi],m1=sm[pi+1],m2=sm[pi+2],m3=sm[pi+3];
				let e0=st[pi],e1=st[pi+1];
				for(let n=0;n<NX.length;n++){
					const nx=x0+x+NX[n],ny=y0+y+NY[n];
					const g=code(nx,ny);
					if(g<0.5)continue;
					const np=(nx-x0+(ny-y0)*stride)*4,s=1/g;
					m0+=sm[np]*s;m1+=sm[np+1]*s;m2+=sm[np+2]*s;m3+=sm[np+3]*s;
					e0+=st[np]*s;e1+=st[np+1]*s;
				}
				mass[i*4]=m0;mass[i*4+1]=m1;mass[i*4+2]=m2;mass[i*4+3]=m3;
				therm[i*4]=e0;therm[i*4+1]=e1;therm[i*4+2]=st[pi+2];therm[i*4+3]=st[pi+3];
			}
		}
		return cells;
	}
	// a scratch pair for displace(): one of these per caller, sized to the widest rect
	// that path can push (the gpu path only pushes rects that fit its patch)
	function makeScratch(cells){
		const n=(cells||128*128)*4;
		return {mass:new Float32Array(n),therm:new Float32Array(n)};
	}
	Fluid.core.colliders={create,add,move,set,remove,rasterize,padRect,geometry,displace,makeScratch,
		emptyRect,shapeOf,shapeNames,MAX,SHAPE_CIRCLE,SHAPE_BOX};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.colliders;
})(globalThis);
