(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};

	// diagnostic floating body for the dam scenario (draft.txt): "buoy stays
	// partially submerged in blue, the amount of blue inside it corresponds to the
	// buoy volume". the lift is the blue tracer per cell area inside the rectangle,
	// so [r,g,255] pushes harder than [r,g,123]. that is an approximation of
	// displaced volume on purpose, and it stays outside the physics solver: the body
	// only reads the field, so no conserved quantity and no ledger entry changes.
	//
	// the six floats below are the whole contract between the field and the body:
	// raw sums over the cell block. src/gpu/shaders.js (buoy_sample) and sampleCpu
	// here produce the same layout; the integration is shared and runs on the cpu.

	const S_BLUE=0,S_MASS=1,S_COUNT=2,S_UX=3,S_UY=4,S_MAXBLUE=5;
	const SAMPLE_LEN=6;

	// the draft's 16x64 is the reference rectangle on the reference grid (N=512);
	// the body keeps that fraction of the view at any N
	function scaledSize(cfg){
		const s=cfg.N/512;
		const w=Math.max(4,Math.round(cfg.buoyRefW*s));
		const h=Math.max(4,Math.round(cfg.buoyRefH*s));
		return {w:Math.min(w,cfg.N),h:Math.min(h,cfg.N)};
	}
	function scenarioHasBody(cfg){
		return cfg.buoyEnabled>0&&cfg.scenario==="dam";
	}
	function clampAbs(v,lim){
		if(v>lim)return lim;
		if(v<-lim)return -lim;
		return v;
	}
	function create(cfg){
		const state={active:false,cx:0,cy:0,x0:0,y0:0,w:0,h:0,vx:0,vy:0,
			aY:0,sub:0,blue:0,fluid:0,peak:0,ux:0,uy:0,frames:0,samples:0};
		const sample=new Float32Array(SAMPLE_LEN);
		function commit(){
			// the rectangle is a block of whole cells: the params block, the gpu sample
			// kernel and the overlay all work from this rounded origin, so they agree
			state.x0=Math.round(state.cx-state.w*0.5);
			state.y0=Math.round(state.cy-state.h*0.5);
			cfg.buoyW=state.w;cfg.buoyH=state.h;
			cfg.buoyX0=state.x0;cfg.buoyY0=state.y0;
		}
		// the body exists only where a scenario places it
		function reset(){
			state.active=scenarioHasBody(cfg);
			cfg.buoyShow=state.active?1:0;
			cfg.buoySub=0;
			sample.fill(0);
			state.frames=0;state.samples=0;
			state.vx=0;state.vy=0;state.aY=0;
			state.sub=0;state.blue=0;state.fluid=0;state.peak=0;state.ux=0;state.uy=0;
			if(!state.active){cfg.buoyW=0;cfg.buoyH=0;cfg.buoyX0=0;cfg.buoyY0=0;return;}
			const size=scaledSize(cfg);
			state.w=size.w;state.h=size.h;
			state.cx=cfg.buoyU*cfg.N;
			state.cy=cfg.buoyV*cfg.N;
			commit();
		}
		// one field sample. called every cfg.buoySampleEvery frames, never in the
		// substep loop: the force is held between refreshes and the body inertia
		// (march) is what makes the motion smooth.
		function ingest(fresh){
			if(fresh)for(let i=0;i<SAMPLE_LEN;i++)sample[i]=fresh[i];
			state.samples++;
			const area=Math.max(state.w*state.h,1);
			const cnt=sample[S_COUNT];
			state.blue=sample[S_BLUE]/area;
			state.fluid=sample[S_MASS]/area;
			state.peak=sample[S_MAXBLUE];
			state.ux=cnt>0?sample[S_UX]/cnt:0;
			state.uy=cnt>0?sample[S_UY]/cnt:0;
			// submerged fraction of the footprint, read off the local peak so it does
			// not depend on how much dye the scenario happens to carry
			state.sub=state.peak>1e-6?Math.min(1,state.blue/state.peak):0;
			// archimedes, approximate: lift = blue inside, weight = the body density.
			// +y is up in this grid, gravityY is a positive down magnitude.
			state.aY=cfg.buoyGain*Math.abs(cfg.gravityY)*(state.blue-cfg.buoyRho);
			cfg.buoySub=state.sub;
		}
		function march(dt){
			if(!state.active)return;
			state.frames++;
			state.vy+=state.aY*dt;
			// drag toward the mean fluid velocity inside the footprint: that is both
			// the coupling that carries the body with the water and the damping that
			// stops it ringing around the float line
			const kd=Math.min(1,dt/Math.max(cfg.buoyDrag,1e-6));
			state.vx+=(state.ux-state.vx)*kd;
			state.vy+=(state.uy-state.vy)*kd;
			const vc=cfg.vMax/Math.max(dt,1e-6);
			state.vx=clampAbs(state.vx,vc);
			state.vy=clampAbs(state.vy,vc);
			state.cx+=state.vx*dt;
			state.cy+=state.vy*dt;
			bounds();
			commit();
		}
		// the domain walls are inelastic; the body never leaves the interior cells
		function bounds(){
			const lx=1+state.w*0.5,hx=cfg.N+1-state.w*0.5;
			const ly=1+state.h*0.5,hy=cfg.N+1-state.h*0.5;
			if(state.cx<lx){state.cx=lx;if(state.vx<0)state.vx=0;}
			else if(state.cx>hx){state.cx=hx;if(state.vx>0)state.vx=0;}
			if(state.cy<ly){state.cy=ly;if(state.vy<0)state.vy=0;}
			else if(state.cy>hy){state.cy=hy;if(state.vy>0)state.vy=0;}
		}
		// cpu twin of the gpu buoy_sample kernel: the same sums, no normalising here
		// (the body divides by its own footprint area, so a rect clipped by a domain
		// wall simply reads less fluid).
		function sampleCpu(mass,vel,texW,texH,out){
			out.fill(0);
			const w=state.w,h=state.h,x0=state.x0,y0=state.y0;
			if(!state.active||w<=0||h<=0)return;
			let blue=0,m=0,su=0,sv=0,peak=0,cnt=0;
			for(let row=0;row<h;row++){
				const cy=y0+row;
				if(cy<1||cy>=texH-1)continue;
				for(let col=0;col<w;col++){
					const cx=x0+col;
					if(cx<1||cx>=texW-1)continue;
					const i4=(cx+cy*texW)*4;
					const db=mass[i4+2];
					blue+=db;m+=mass[i4+3];
					if(db>peak)peak=db;
					const i2=(cx+cy*texW)*2;
					su+=vel[i2];sv+=vel[i2+1];
					cnt++;
				}
			}
			out[S_BLUE]=blue;out[S_MASS]=m;out[S_COUNT]=cnt;
			out[S_UX]=su;out[S_UY]=sv;out[S_MAXBLUE]=peak;
		}
		reset();
		return {state,sample,SAMPLE_LEN,reset,ingest,march,sampleCpu};
	}
	Fluid.core.buoy={create,scaledSize,scenarioHasBody,SAMPLE_LEN};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.buoy;
})(globalThis);
