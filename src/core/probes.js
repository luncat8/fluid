(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};

	// game-facing point queries (0.7.0 t5): a fixed table of 32 slots, read by one
	// workgroup and copied back on the buoy cadence. the table is cpu-owned; the
	// kernel only overwrites the measured values, so a move is one buffer write and
	// never a table rebuild.
	//
	// storage layout in f32:
	//   [0]      live slot count; 0 skips the dispatch and the readback
	//   [1..15]  header pad
	//   slot i   at 16 + i*16:
	//            0 x, 1 y, 2 m, 3 dr, 4 dg, 5 db, 6 E, 7 u, 8 v, 9 p,
	//            10..14 unused (future fields), 15 active flag
	// ids are slot indices and stay valid until remove, so a game can hold them.
	const MAX=32;
	const STRIDE=16;
	const HEADER=16;
	const LEN=HEADER+MAX*STRIDE;
	const P_X=0,P_Y=1,P_M=2,P_DR=3,P_DG=4,P_DB=5,P_E=6,P_U=7,P_V=8,P_P=9;
	const P_ACTIVE=15;
	// one shared scratch: sample() derives T and phase with the same closed form the
	// cpu oracle and the wgsl evalPhase use, and writes only plain numbers into out.
	const phaseScratch={T:0,phiL:0,phiG:0,phase:0,rhoStar:0,S:0,c:0,k:0};
	function slotBase(id){
		return HEADER+id*STRIDE;
	}
	function makeSample(){
		return {valid:false,age:0,id:-1,x:0,y:0,m:0,dr:0,dg:0,db:0,E:0,T:0,phase:0,u:0,v:0,p:0};
	}
	function create(){
		const table={
			live:0,
			slots:new Array(MAX),
			upload:new Float32Array(LEN),
			raw:new Float32Array(LEN),
			sampledFrame:-1,
			samples:0
		};
		for(let i=0;i<MAX;i++)table.slots[i]={active:false,x:0,y:0,bornFrame:-1};
		return table;
	}
	// probes live in cell coordinates, the same space as the buoy rectangle; the
	// kernel clamps to the interior, so an edge probe reads the nearest real cell.
	function clampCoord(cfg,v){
		if(!(v>0))v=1;
		if(v>cfg.N)v=cfg.N;
		return v;
	}
	function setSlot(table,id,x,y,active){
		const s=table.slots[id];
		s.active=active;s.x=x;s.y=y;
		const b=slotBase(id);
		table.upload[b+P_X]=active?x:0;
		table.upload[b+P_Y]=active?y:0;
		table.upload[b+P_ACTIVE]=active?1:0;
		table.upload[0]=table.live;
	}
	function add(table,cfg,x,y,bornFrame){
		for(let i=0;i<MAX;i++){
			if(table.slots[i].active)continue;
			table.live++;
			setSlot(table,i,clampCoord(cfg,x),clampCoord(cfg,y),true);
			table.slots[i].bornFrame=bornFrame!==undefined?Math.max(bornFrame,table.sampledFrame):table.sampledFrame;
			table.upload[0]=table.live;
			return i;
		}
		return -1;
	}
	function move(table,cfg,id,x,y){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		setSlot(table,id,clampCoord(cfg,x),clampCoord(cfg,y),true);
		return true;
	}
	function remove(table,id){
		const s=table.slots[id];
		if(!s||!s.active)return false;
		table.live--;
		setSlot(table,id,0,0,false);
		s.bornFrame=-1;
		table.upload[0]=table.live;
		return true;
	}
	function syncConfig(table,cfg){
		if(table.live===0)return false;
		for(let i=0;i<MAX;i++){
			const s=table.slots[i];
			if(!s.active)continue;
			setSlot(table,i,clampCoord(cfg,s.x),clampCoord(cfg,s.y),true);
		}
		return true;
	}
	// a map is in flight across a reset or resize: the numbers belong to the old
	// field. the table keeps its slots, only the sample gets marked stale.
	function invalidate(table){
		table.sampledFrame=-1;
		table.samples=0;
		for(let i=0;i<MAX;i++)table.slots[i].bornFrame=-1;
	}
	function ingest(table,raw,frame){
		table.raw.set(raw);
		table.sampledFrame=frame;
		table.samples++;
	}
	function derive(cfg,out){
		Fluid.core.solver.phaseInto(phaseScratch,out.m,out.E,cfg);
		out.T=phaseScratch.T;
		out.phase=phaseScratch.phase;
	}
	// last completed sample of slot id into out. returns false when the probe is gone
	// or has never been read back; a caller that ignores the age is reading stale data.
	function sample(table,cfg,frame,id,out){
		const s=table.slots[id];
		if(!s||!s.active||table.sampledFrame<0||table.sampledFrame<=s.bornFrame){
			out.valid=false;
			out.id=id;
			return false;
		}
		const b=slotBase(id);
		out.id=id;
		out.x=s.x;out.y=s.y;
		out.m=table.raw[b+P_M];
		out.dr=table.raw[b+P_DR];
		out.dg=table.raw[b+P_DG];
		out.db=table.raw[b+P_DB];
		out.E=table.raw[b+P_E];
		out.u=table.raw[b+P_U];
		out.v=table.raw[b+P_V];
		out.p=table.raw[b+P_P];
		out.age=Math.max(0,frame-table.sampledFrame);
		derive(cfg,out);
		out.valid=true;
		return true;
	}
	// cpu twin of the gpu probeSample kernel on plain fields: the oracle the node
	// tests compare the layout and the clamps against. no readback, no table state.
	function sampleCpu(table,cfg,mass,therm,vel,press,texW,texH,id,out){
		const s=table.slots[id];
		if(!s||!s.active){
			out.valid=false;
			out.id=id;
			return false;
		}
		let cx=Math.round(s.x),cy=Math.round(s.y);
		if(cx<1)cx=1;
		if(cx>texW-2)cx=texW-2;
		if(cy<1)cy=1;
		if(cy>texH-2)cy=texH-2;
		const i4=(cx+cy*texW)*4,i2=(cx+cy*texW)*2;
		out.id=id;
		out.x=s.x;out.y=s.y;
		out.m=mass[i4+3];
		out.dr=mass[i4];out.dg=mass[i4+1];out.db=mass[i4+2];
		out.E=therm[i4];
		out.u=vel[i2];out.v=vel[i2+1];
		out.p=press[cx+cy*texW];
		out.age=0;
		derive(cfg,out);
		out.valid=true;
		return true;
	}
	Fluid.core.probes={create,add,move,remove,syncConfig,invalidate,ingest,sample,sampleCpu,makeSample,
		MAX,STRIDE,HEADER,LEN,P_X,P_Y,P_M,P_DR,P_DG,P_DB,P_E,P_U,P_V,P_P,P_ACTIVE,slotBase};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.probes;
})(globalThis);
