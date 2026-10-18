(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const grid=Fluid.core.grid;

	// layout (0.0.0 section 4/7): cell (x,y) spans [x-1,x] x [y-1,y] in cell units,
	// interior x,y in 1..N, one ghost ring around. texel (x,y) of vel/flux holds the
	// left/bottom face values of cell (x,y). this file is the numeric oracle for the
	// scheme in 0.0.0 section 5; the wgsl in src/gpu implements the same formulas.

	function buildScenario(cfg){
		const N=cfg.N,texW=N+2,texH=N+2;
		const massA=new Float32Array(texW*texH*4);
		const thermA=new Float32Array(texW*texH*4);
		const flagsA=new Uint8Array(texW*texH*4);
		const cLiq=cfg.cLiq,TAmb=cfg.TAmb;
		for(let y=1;y<texH-1;y++){
			for(let x=1;x<texW-1;x++){
				const i=(x+y*texW)*4;
				let m=0.3;
				if(cfg.scenario==="still")m=1.0;
				if(cfg.scenario==="dam"&&x<Math.ceil(N*0.45)&&x>=Math.ceil(N*0.1)&&y<=Math.ceil(N*0.6))m=1.0;
				massA[i+3]=m;
				if(cfg.scenario==="dam"&&m>0.5)massA[i+2]=0.5*m;
				thermA[i]=m*cLiq*TAmb;
				thermA[i+1]=TAmb;
				thermA[i+2]=1;
				thermA[i+3]=0;
			}
		}
		if(cfg.scenario==="blobs"){
			paintBlob(massA,thermA,texW,texH,N*0.3,N*0.5,N*0.12,[0.8,0,0],0.6,cLiq,TAmb);
			paintBlob(massA,thermA,texW,texH,N*0.7,N*0.5,N*0.12,[0,0.8,0],0.6,cLiq,TAmb);
			paintBlob(massA,thermA,texW,texH,N*0.5,N*0.25,N*0.12,[0,0,0.8],0.6,cLiq,TAmb);
		}
		return {massA,thermA,flagsA,texW,texH};
	}
	function paintBlob(massA,thermA,texW,texH,cx,cy,radius,rgb,dm,cLiq,TAmb){
		for(let y=1;y<texH-1;y++){
			for(let x=1;x<texW-1;x++){
				const dx=(x-0.5)-cx,dy=(y-0.5)-cy;
				const d=Math.sqrt(dx*dx+dy*dy);
				if(d>=radius)continue;
				let fall=1-(d/radius)*(d/radius);
				fall=fall*fall;
				const i=(x+y*texW)*4;
				massA[i]+=rgb[0]*fall;
				massA[i+1]+=rgb[1]*fall;
				massA[i+2]+=rgb[2]*fall;
				massA[i+3]+=dm*fall;
				thermA[i]=massA[i+3]*cLiq*TAmb;
			}
		}
	}
	// interactive dye brush: an exact additive patch. the gpu adds these floats and
	// the ledger sums these floats, so injection accounting is exact by construction.
	function buildDyePatch(cfg,u,v,rgb){
		const texW=cfg.texW,texH=cfg.texH;
		const px=u*cfg.N,py=v*cfg.N;
		const cx=Math.floor(px)+1,cy=Math.floor(py)+1;
		const r=Math.min(60,Math.max(1,cfg.paintR));
		const x0=Math.max(1,cx-r),x1=Math.min(texW-2,cx+r);
		const y0=Math.max(1,cy-r),y1=Math.min(texH-2,cy+r);
		const w=x1-x0+1,h=y1-y0+1;
		if(w<=0||h<=0)return null;
		const massPatch=new Float32Array(w*h*4);
		const ePatch=new Float32Array(w*h);
		const cLiq=cfg.cLiq,TAmb=cfg.TAmb;
		for(let y=0;y<h;y++){
			for(let x=0;x<w;x++){
				const dx=(x0+x-0.5)-px,dy=(y0+y-0.5)-py;
				const d=Math.sqrt(dx*dx+dy*dy);
				if(d>=r)continue;
				let fall=1-(d/r)*(d/r);
				fall=fall*fall;
				const dm=cfg.paintAmount*fall;
				const i=(x+y*w)*4;
				massPatch[i]=rgb[0]*dm;
				massPatch[i+1]=rgb[1]*dm;
				massPatch[i+2]=rgb[2]*dm;
				massPatch[i+3]=dm;
				ePatch[x+y*w]=dm*cLiq*TAmb;
			}
		}
		return {x0,y0,w,h,massPatch,ePatch};
	}
	function patchTotals(patch){
		let m=0,dr=0,dg=0,db=0,E=0;
		for(let i=0;i<patch.massPatch.length;i+=4){
			dr+=patch.massPatch[i];dg+=patch.massPatch[i+1];db+=patch.massPatch[i+2];m+=patch.massPatch[i+3];
		}
		for(let i=0;i<patch.ePatch.length;i++)E+=patch.ePatch[i];
		return {m,dr,dg,db,E};
	}
	function buildWallPatch(cfg,u,v,erase){
		const texW=cfg.texW,texH=cfg.texH;
		const px=u*cfg.N,py=v*cfg.N;
		const cx=Math.floor(px)+1,cy=Math.floor(py)+1;
		const r=Math.min(60,Math.max(1,cfg.paintR));
		const x0=Math.max(1,cx-r),x1=Math.min(texW-2,cx+r);
		const y0=Math.max(1,cy-r),y1=Math.min(texH-2,cy+r);
		const w=x1-x0+1,h=y1-y0+1;
		if(w<=0||h<=0)return null;
		const flagsPatch=new Uint8Array(w*h*4);
		const v8=erase?0:255;
		for(let y=0;y<h;y++){
			for(let x=0;x<w;x++){
				const dx=(x0+x-0.5)-px,dy=(y0+y-0.5)-py;
				if(dx*dx+dy*dy>=r*r)continue;
				flagsPatch[(x+y*w)*4]=v8;
			}
		}
		return {x0,y0,w,h,flagsPatch};
	}

	function create(cfg){
		const texW=cfg.texW,texH=cfg.texH,N=cfg.N;
		const mass=grid.allocPingPong(texW,texH,16,Float32Array);
		const therm=grid.allocPingPong(texW,texH,16,Float32Array);
		const vel=grid.allocPingPong(texW,texH,8,Float32Array);
		const press=grid.allocPingPong(texW,texH,4,Float32Array);
		const flux=new Float32Array(texW*texH*2);
		const divsrc=new Float32Array(texW*texH);
		const flags=new Uint8Array(texW*texH*4);
		const acc=Fluid.core.stats;
		let ledger=null;
		function isWall(x,y){
			if(x<1||x>texW-2||y<1||y>texH-2)return true;
			return flags[(x+y*texW)*4]>0;
		}
		function uBlocked(x,y){
			return x<=1||x>=texW-1||isWall(x-1,y)||isWall(x,y);
		}
		function vBlocked(x,y){
			return y<=1||y>=texH-1||isWall(x,y-1)||isWall(x,y);
		}
		function reset(){
			const sc=buildScenario(cfg);
			mass.a.set(sc.massA);mass.b.fill(0);
			therm.a.set(sc.thermA);therm.b.fill(0);
			vel.a.fill(0);vel.b.fill(0);
			press.a.fill(0);press.b.fill(0);
			flux.fill(0);divsrc.fill(0);
			flags.set(sc.flagsA);
			mass.cur=0;therm.cur=0;vel.cur=0;press.cur=0;
			const tot=acc.totalsFromFields(mass.a,therm.a,texW,texH);
			ledger=acc.createLedger(tot);
			cfg.frame=0;cfg.time=0;
		}
		function forces(dt){
			const src=grid.curBuf(vel),dst=grid.nxtBuf(vel);
			const ms=grid.curBuf(mass),ts=grid.curBuf(therm);
			const R=cfg.mouseR,R2=R*R;
			// mean-subtracted gravity: the uniform part is what the projection would remove
			// anyway; subtracting it exactly stops the residual-divergence mass feedback
			// measured in experiments/conformance.js (runaway overpack at walls).
			const rhoMean=(ledger.initialM+ledger.injectedM-ledger.removedM)/(N*N);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const vi=(x+y*texW)*2;
					let u=0,v=0;
					if(!uBlocked(x,y)){
						u=src[vi]+dt*0.5*(cellAx(ms,ts,x-1,y,rhoMean)+cellAx(ms,ts,x,y,rhoMean));
						if(cfg.mouseStrength>0){
							const f=mouseFall((x-1)-cfg.mouseX,(y-0.5)-cfg.mouseY,R,R2);
							u+=dt*cfg.mouseVX*f*cfg.mouseStrength;
						}
						u=clampAbs(u,cfg.vClamp);
					}
					if(!vBlocked(x,y)){
						v=src[vi+1]+dt*0.5*(cellAy(ms,ts,x,y-1,rhoMean)+cellAy(ms,ts,x,y,rhoMean));
						if(cfg.mouseStrength>0){
							const f=mouseFall((x-0.5)-cfg.mouseX,(y-1)-cfg.mouseY,R,R2);
							v+=dt*cfg.mouseVY*f*cfg.mouseStrength;
						}
						v=clampAbs(v,cfg.vClamp);
					}
					dst[vi]=u;dst[vi+1]=v;
				}
			}
			grid.swap(vel);
		}
		function cellAx(ms,ts,x,y,rhoMean){
			if(x<1||x>texW-2||y<1||y>texH-2)return 0;
			const i=(x+y*texW)*4;
			return cfg.gravityX*(rhoMean-ms[i+3])/cfg.rhoRef-cfg.gravityX*cfg.beta*(ts[i+1]-cfg.TRef);
		}
		function cellAy(ms,ts,x,y,rhoMean){
			if(x<1||x>texW-2||y<1||y>texH-2)return 0;
			const i=(x+y*texW)*4;
			return cfg.gravityY*(rhoMean-ms[i+3])/cfg.rhoRef-cfg.gravityY*cfg.beta*(ts[i+1]-cfg.TRef);
		}
		function mouseFall(dx,dy,R,R2){
			const r2=dx*dx+dy*dy;
			if(r2>=R2)return 0;
			const f=1-Math.sqrt(r2)/R;
			return f*f;
		}
		function clampAbs(v,lim){
			if(v>lim)return lim;
			if(v<-lim)return -lim;
			return v;
		}
		function loadU(buf,m,l){
			const mm=m<0?0:(m>texW-1?texW-1:m);
			const ll=l<0?0:(l>texH-1?texH-1:l);
			return buf[(mm+ll*texW)*2];
		}
		function loadV(buf,m,l){
			const mm=m<0?0:(m>texW-1?texW-1:m);
			const ll=l<0?0:(l>texH-1?texH-1:l);
			return buf[(mm+ll*texW)*2+1];
		}
		function advectVel(dt){
			const src=grid.curBuf(vel),dst=grid.nxtBuf(vel);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const vi=(x+y*texW)*2;
					let u=0,v=0;
					if(!uBlocked(x,y)){
						const uc=src[vi];
						const vc=0.25*(loadV(src,x-1,y)+loadV(src,x-1,y+1)+loadV(src,x,y)+loadV(src,x,y+1));
						let px=(x-1)-dt*uc,py=(y-0.5)-dt*vc;
						px=px<0?0:(px>N?N:px);
						py=py<0.5?0.5:(py>N-0.5?N-0.5:py);
						const i0=Math.floor(px),fx=px-i0;
						const l0=Math.floor(py+0.5),fy=py+0.5-l0;
						const u00=loadU(src,i0+1,l0),u10=loadU(src,i0+2,l0);
						const u01=loadU(src,i0+1,l0+1),u11=loadU(src,i0+2,l0+1);
						u=(u00+(u10-u00)*fx)+((u01+(u11-u01)*fx)-(u00+(u10-u00)*fx))*fy;
					}
					if(!vBlocked(x,y)){
						const vc=src[vi+1];
						const uc=0.25*(loadU(src,x,y-1)+loadU(src,x+1,y-1)+loadU(src,x,y)+loadU(src,x+1,y));
						let px=(x-0.5)-dt*uc,py=(y-1)-dt*vc;
						px=px<0.5?0.5:(px>N-0.5?N-0.5:px);
						py=py<0?0:(py>N?N:py);
						const k0=Math.floor(px+0.5),fx=px+0.5-k0;
						const l0=Math.floor(py),fy=py-l0;
						const v00=loadV(src,k0,l0+1),v10=loadV(src,k0+1,l0+1);
						const v01=loadV(src,k0,l0+2),v11=loadV(src,k0+1,l0+2);
						v=(v00+(v10-v00)*fx)+((v01+(v11-v01)*fx)-(v00+(v10-v00)*fx))*fy;
					}
					dst[vi]=u;dst[vi+1]=v;
				}
			}
			grid.swap(vel);
		}
		// rhs with the expansion source S (0.0.0 section 5.4/5.5, minimal liquid form):
		// a cell packed above rho_liq expands. without this the liquid has no
		// resistance to compaction, the projection residual slowly overpacks the
		// pile and gravity runs it away (measured: dam break at 64^2 doubles
		// max|vel| every ~2 steps and NaNs by step 100). gas rho* arrives in 0.5.0.
		function poissonRhs(dt){
			const vv=grid.curBuf(vel);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=x+y*texW;
					divsrc[i]=0;
					if(isWall(x,y))continue;
					const uL=vv[(i)*2],uR=vv[(i+1)*2];
					const vB=vv[(i)*2+1],vT=vv[(i+texW)*2+1];
					const m=grid.curBuf(mass)[i*4+3];
					let S=0;
					if(m>cfg.rhoLiq){
						S=(m-cfg.rhoLiq)/(m*cfg.tauPhase);
						if(S>cfg.SMax)S=cfg.SMax;
					}
					divsrc[i]=(uR-uL+vT-vB)/dt-S;
				}
			}
		}
		function project(iters){
			const ms=grid.curBuf(mass);
			for(let k=0;k<iters;k++){
				const src=grid.curBuf(press),dst=grid.nxtBuf(press);
				for(let y=1;y<texH-1;y++){
					for(let x=1;x<texW-1;x++){
						const i=x+y*texW;
						if(isWall(x,y)){dst[i]=0;continue;}
						let wsum=0,sum=0;
						let w=faceW(ms,x,y,x-1,y);
						if(w>0){wsum+=w;sum+=w*src[i-1];}
						w=faceW(ms,x,y,x+1,y);
						if(w>0){wsum+=w;sum+=w*src[i+1];}
						w=faceW(ms,x,y,x,y-1);
						if(w>0){wsum+=w;sum+=w*src[i-texW];}
						w=faceW(ms,x,y,x,y+1);
						if(w>0){wsum+=w;sum+=w*src[i+texW];}
						dst[i]=wsum>0?(sum-divsrc[i])/wsum:0;
					}
				}
				grid.swap(press);
			}
		}
		// w = 1/rho_f (harmonic mean); 0 drops the face: wall neighbour or domain edge
		function faceW(ms,x,y,nx,ny){
			if(nx<1||nx>texW-2||ny<1||ny>texH-2)return 0;
			if(flags[(nx+ny*texW)*4]>0)return 0;
			const ra=Math.max(ms[(x+y*texW)*4+3],cfg.rhoVac);
			const rb=Math.max(ms[(nx+ny*texW)*4+3],cfg.rhoVac);
			return 1/(2*ra*rb/(ra+rb));
		}
		function applyGrad(dt){
			const vs=grid.curBuf(vel),vd=grid.nxtBuf(vel);
			const ps=grid.curBuf(press);
			const ms=grid.curBuf(mass);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const vi=(x+y*texW)*2;
					const i=x+y*texW;
					let u=vs[vi],v=vs[vi+1];
					if(!uBlocked(x,y)){
						const rf=faceInvRho(ms,x-1,y,x,y);
						u-=dt*(ps[i]-ps[i-1])*rf;
					}
					if(!vBlocked(x,y)){
						const rf=faceInvRho(ms,x,y-1,x,y);
						v-=dt*(ps[i]-ps[i-texW])*rf;
					}
					vd[vi]=u;vd[vi+1]=v;
				}
			}
			grid.swap(vel);
		}
		function faceInvRho(ms,ax,ay,bx,by){
			const ra=Math.max(ms[(ax+ay*texW)*4+3],cfg.rhoVac);
			const rb=Math.max(ms[(bx+by*texW)*4+3],cfg.rhoVac);
			return 1/(2*ra*rb/(ra+rb));
		}
		function computeFlux(dt){
			const vv=grid.curBuf(vel);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const vi=(x+y*texW)*2;
					const fi=(x+y*texW)*2;
					let fx=0,fy=0;
					if(!uBlocked(x,y)){
						fx=vv[vi]*dt;
						if(fx>1)fx=1;else if(fx<-1)fx=-1;
					}
					if(!vBlocked(x,y)){
						fy=vv[vi+1]*dt;
						if(fy>1)fy=1;else if(fy<-1)fy=-1;
					}
					flux[fi]=fx;flux[fi+1]=fy;
				}
			}
		}
		function advectScalar(){
			const mS=grid.curBuf(mass),mD=grid.nxtBuf(mass);
			const tS=grid.curBuf(therm),tD=grid.nxtBuf(therm);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i4=(x+y*texW)*4;
					const i2=(x+y*texW)*2;
					const fw=flux[i2],fe=flux[i2+2];
					const fs=flux[i2+1],fn=flux[i2+texW*2+1];
					for(let c=0;c<4;c++){
						const qm=mS[i4+c],qt=tS[i4+c];
						const qmW=mS[i4-4+c],qmE=mS[i4+4+c],qmS=mS[i4-texW*4+c],qmN=mS[i4+texW*4+c];
						const qtW=tS[i4-4+c],qtE=tS[i4+4+c],qtS=tS[i4-texW*4+c],qtN=tS[i4+texW*4+c];
					const dm=fw*(fw>0?qmW:qm)-fe*(fe>0?qm:qmE)+fs*(fs>0?qmS:qm)-fn*(fn>0?qm:qmN);
					mD[i4+c]=qm+dm;
						let dtn=fw*(fw>0?qtW:qt)-fe*(fe>0?qt:qtE)+fs*(fs>0?qtS:qt)-fn*(fn>0?qt:qtN);
						tD[i4+c]=qt+dtn;
					}
				}
			}
			grid.swap(mass);grid.swap(therm);
		}
		function step(dt){
			forces(dt);
			advectVel(dt);
			poissonRhs(dt);
			project(cfg.pressureIters);
			applyGrad(dt);
			computeFlux(dt);
			advectScalar();
			cfg.frame++;
			cfg.time+=dt;
		}
		function stats(){
			const tot=acc.totalsFromFields(grid.curBuf(mass),grid.curBuf(therm),texW,texH);
			const mot=acc.motionTotals(grid.curBuf(vel),grid.curBuf(press),texW,texH,cfg.dt);
			return {frame:cfg.frame,tot,mot,ledger};
		}
		function applyDyePatch(patch){
			const buf=grid.curBuf(mass),tbuf=grid.curBuf(therm);
			for(let y=0;y<patch.h;y++){
				for(let x=0;x<patch.w;x++){
					const i=((patch.x0+x)+(patch.y0+y)*texW)*4;
					const p=(x+y*patch.w)*4;
					buf[i]+=patch.massPatch[p];
					buf[i+1]+=patch.massPatch[p+1];
					buf[i+2]+=patch.massPatch[p+2];
					buf[i+3]+=patch.massPatch[p+3];
					tbuf[i]+=patch.ePatch[x+y*patch.w];
				}
			}
			const t=patchTotals(patch);
			ledger.injectedM+=t.m;ledger.injectedDr+=t.dr;ledger.injectedDg+=t.dg;ledger.injectedDb+=t.db;ledger.injectedE+=t.E;
		}
		function applyWallPatch(patch){
			for(let y=0;y<patch.h;y++){
				for(let x=0;x<patch.w;x++){
					const i=((patch.x0+x)+(patch.y0+y)*texW)*4;
					flags[i]=patch.flagsPatch[(x+y*patch.w)*4];
				}
			}
		}
		reset();
		return {cfg,texW,texH,N,mass,therm,vel,press,flux,divsrc,flags,ledger,
			isWall,uBlocked,vBlocked,reset,step,stats,forces,advectVel,poissonRhs,project,applyGrad,computeFlux,advectScalar,
			applyDyePatch,applyWallPatch};
	}
	Fluid.core.solver={create,buildScenario,buildDyePatch,buildWallPatch,patchTotals,paintBlob};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.solver;
})(globalThis);
