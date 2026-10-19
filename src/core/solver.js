(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const grid=Fluid.core.grid;
	const f32=Math.fround;

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
				if(cfg.scenario==="still"||cfg.scenario==="convection")m=1.0;
				if(cfg.scenario==="dam"&&x<Math.ceil(N*0.45)&&x>=Math.ceil(N*0.1)&&y<=Math.ceil(N*0.6))m=1.0;
				massA[i+3]=m;
				if(cfg.scenario==="dam"&&m>0.5)massA[i+2]=0.5*m;
				thermA[i]=m*cLiq*TAmb;
				thermA[i+1]=TAmb;
				// b/a are the cumulative heater/cold-plate source sums (0.4.0):
				// donor transport carries them with the fluid, so their field
				// totals are the exact source ledger and survive advection.
				thermA[i+2]=0;
				thermA[i+3]=0;
			}
		}
		if(cfg.scenario==="convection"){
			// seed the rolls: a smooth warm/cold perturbation at uniform energy
			// density per unit mass, so buoyancy has something to amplify
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=(x+y*texW)*4;
					const T=TAmb+3*Math.sin(6.2831853*(x-0.5)/N)*Math.sin(3.14159265*(y-0.5)/N);
					thermA[i]=massA[i+3]*cLiq*T;
					thermA[i+1]=T;
				}
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
		// Preallocated dye-FCT scratch: per-cell (R+,R-) for RGB and final mass result.
		const fctLimiters=new Float32Array(texW*texH*6);
		const massFctTmp=new Float32Array(texW*texH*4);
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
			cfg.thermalSources=cfg.scenario==="convection"?1:0;
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
				for(let color=0;color<2;color++){
					const src=grid.curBuf(press),dst=grid.nxtBuf(press);
					for(let y=1;y<texH-1;y++){
						for(let x=1;x<texW-1;x++){
							const i=x+y*texW;
							if(((x+y)&1)!==color){dst[i]=src[i];continue;}
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
		function rawFx(vv,dt,x,y){
			if(uBlocked(x,y))return 0;
			const u=vv[(x+y*texW)*2]*dt;
			return u>1?1:(u<-1?-1:u);
		}
		function rawFy(vv,dt,x,y){
			if(vBlocked(x,y))return 0;
			const v=vv[(x+y*texW)*2+1]*dt;
			return v>1?1:(v<-1?-1:v);
		}
		function outFlux(vv,dt,x,y){
			const fw=rawFx(vv,dt,x,y),fe=rawFx(vv,dt,x+1,y),fs=rawFy(vv,dt,x,y),fn=rawFy(vv,dt,x,y+1);
			return (fw<0?-fw:0)+(fe>0?fe:0)+(fs<0?-fs:0)+(fn>0?fn:0);
		}
		function computeFlux(dt){
			const vv=grid.curBuf(vel);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i2=(x+y*texW)*2;
					let fx=rawFx(vv,dt,x,y),fy=rawFy(vv,dt,x,y);
					// monotone donor-cell needs total outflow <= 1 volume per cell per
					// step. the +-1 face clamp alone is not enough: a diverging cell
					// empties through up to 4 faces at once and the q'=q*(1-k)+in
					// update then amplifies sign-alternating mass every step (draw +
					// stir exploded to NaN in ~15 frames). scale the faces a cell owns
					// as outflow so they sum to <= 1; each face is outflow for exactly
					// one of its two cells, so the stored value stays the single shared
					// flux and conservation is untouched. same limiter in the wgsl.
					if(fx!==0){
						const s=fx>0?outFlux(vv,dt,x-1,y):outFlux(vv,dt,x,y);
						if(s>1)fx/=s;
					}
					if(fy!==0){
						const s=fy>0?outFlux(vv,dt,x,y-1):outFlux(vv,dt,x,y);
						if(s>1)fy/=s;
					}
					flux[i2]=fx;flux[i2+1]=fy;
				}
			}
		}
		function scalarValue(mbuf,index,field){
			return mbuf[index*4+field];
		}
		// Lax-Wendroff minus donor-cell flux; only dye densities use this scalar correction.
		function antiFlux(mSrc,faceFlux,left,right,field){
			const a=Math.abs(faceFlux);
			return 0.5*a*(1-a)*(scalarValue(mSrc,right,field)-scalarValue(mSrc,left,field));
		}
		function limiterIndex(index,field){
			return index*6+field*2;
		}
		// Zalesak plus/minus allowances use the local 3x3 extrema of the low-order state.
		function limiterPrepare(mSrc,mLow){
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=x+y*texW,i2=i*2;
					const fw=flux[i2],fe=flux[i2+2],fs=flux[i2+1],fn=flux[i2+texW*2+1];
					for(let field=0;field<3;field++){
						const q=scalarValue(mLow,i,field);
						let qMin=q,qMax=q;
						for(let oy=-1;oy<=1;oy++){
							for(let ox=-1;ox<=1;ox++){
								const qn=scalarValue(mLow,i+ox+oy*texW,field);
								if(qn<qMin)qMin=qn;
								if(qn>qMax)qMax=qn;
							}
						}
						const dW=antiFlux(mSrc,fw,i-1,i,field);
						const dE=antiFlux(mSrc,fe,i,i+1,field);
						const dS=antiFlux(mSrc,fs,i-texW,i,field);
						const dN=antiFlux(mSrc,fn,i,i+texW,field);
						const pPlus=Math.max(dW,0)+Math.max(-dE,0)+Math.max(dS,0)+Math.max(-dN,0);
						const pMinus=Math.max(-dW,0)+Math.max(dE,0)+Math.max(-dS,0)+Math.max(dN,0);
						const base=limiterIndex(i,field);
						let rPlus=pPlus>0?Math.min(1,Math.max(0,qMax-q)/pPlus):1;
						let rMinus=pMinus>0?Math.min(1,Math.max(0,q-qMin)/pMinus):1;
						if(rPlus<1)rPlus*=0.99999;
						if(rMinus<1)rMinus*=0.99999;
						fctLimiters[base]=rPlus;
						fctLimiters[base+1]=rMinus;
					}
				}
			}
		}
		function limitedAnti(mSrc,faceFlux,left,right,field){
			const d=antiFlux(mSrc,faceFlux,left,right,field);
			if(d===0)return 0;
			const leftR=limiterIndex(left,field),rightR=limiterIndex(right,field);
			const ratio=d>0?Math.min(fctLimiters[leftR+1],fctLimiters[rightR]):Math.min(fctLimiters[leftR],fctLimiters[rightR+1]);
			return ratio*d;
		}
		function limiterApply(mSrc,mLow){
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=x+y*texW,i2=i*2,i4=i*4;
					const fw=flux[i2],fe=flux[i2+2],fs=flux[i2+1],fn=flux[i2+texW*2+1];
					massFctTmp[i4]=mLow[i4];
					massFctTmp[i4+1]=mLow[i4+1];
					massFctTmp[i4+2]=mLow[i4+2];
					massFctTmp[i4+3]=mLow[i4+3];
					for(let field=0;field<3;field++){
						const west=limitedAnti(mSrc,fw,i-1,i,field);
						const east=limitedAnti(mSrc,fe,i,i+1,field);
						const south=limitedAnti(mSrc,fs,i-texW,i,field);
						const north=limitedAnti(mSrc,fn,i,i+texW,field);
						massFctTmp[i4+field]=mLow[i4+field]+west-east+south-north;
					}
				}
			}
			grid.nxtBuf(mass).set(massFctTmp);
			grid.swap(mass);
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
						const dtn=fw*(fw>0?qtW:qt)-fe*(fe>0?qt:qtE)+fs*(fs>0?qtS:qt)-fn*(fn>0?qt:qtN);
						tD[i4+c]=qt+dtn;
					}
				}
			}
			grid.swap(mass);grid.swap(therm);
			if(!cfg.fctLimiter)return;
			const mLow=grid.curBuf(mass),mSrc=grid.nxtBuf(mass);
			limiterPrepare(mSrc,mLow);
			limiterApply(mSrc,mLow);
		}
		const thermDiag={clamped:0};
		// therm.g is a cache, never a transported quantity: T = E/(m*c) for cells
		// with substance, T_amb for vacuum (no capacity, no buoyancy there)
		function deriveT(){
			const ms=grid.curBuf(mass),ts=grid.curBuf(therm);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=(x+y*texW)*4;
					const m=ms[i+3];
					ts[i+1]=m>=cfg.mEps?f32(ts[i]/f32(m*cfg.cLiq)):cfg.TAmb;
				}
			}
		}
		// clamped pairwise exchange (0.0.0 5.7). Ping-pong is mandatory: both
		// cells read the same pre-update state and compute opposite f32 face fluxes;
		// summing four fluxes into each f32 cell leaves only a small rounding floor.
		function conduction(dt){
			const ms=grid.curBuf(mass);
			const src=grid.curBuf(therm),dst=grid.nxtBuf(therm);
			thermDiag.clamped=0;
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=(x+y*texW)*4;
					let E=src[i];
					if(ms[i+3]>=cfg.mEps&&!isWall(x,y)){
						const T=src[i+1],C=f32(ms[i+3]*cfg.cLiq);
						E=f32(E+faceExchange(ms,src,x,y,x-1,y,T,dt));
						E=f32(E+faceExchange(ms,src,x,y,x+1,y,T,dt));
						E=f32(E+faceExchange(ms,src,x,y,x,y-1,T,dt));
						E=f32(E+faceExchange(ms,src,x,y,x,y+1,T,dt));
						if(cfg.wallCool>0&&(isWall(x-1,y)||isWall(x+1,y)||isWall(x,y-1)||isWall(x,y+1))){
							const dT=f32(T-cfg.TAmb);
							const dE=f32(f32(f32(f32(-cfg.wallCool)*dT)*C)*dt);
							E=f32(E+dE);
							if(dE>0)ledger.heatInE+=dE;else ledger.heatOutE-=dE;
							dst[i+2]=src[i+2]+(dE>0?dE:0);
							dst[i+3]=src[i+3]+(dE<0?-dE:0);
						}else{
							dst[i+2]=src[i+2];dst[i+3]=src[i+3];
						}
					}else{
						dst[i+2]=src[i+2];dst[i+3]=src[i+3];
					}
					dst[i]=E;
					dst[i+1]=ms[i+3]>=cfg.mEps?f32(E/f32(ms[i+3]*cfg.cLiq)):cfg.TAmb;
				}
			}
			grid.swap(therm);
		}
		// one face of the conduction update for cell (x,y) against (nx,ny);
		// the neighbour computes the exact negation from the same inputs
		function faceExchange(ms,ts,x,y,nx,ny,T,dt){
			if(nx<1||nx>texW-2||ny<1||ny>texH-2)return 0;
			const j=(nx+ny*texW)*4;
			if(flags[j]>0){
				return 0; // wall faces exchange through wallCool above, not pairwise
			}
			if(ms[j+3]<cfg.mEps)return 0; // vacuum does not conduct
			const dT=f32(ts[j+1]-T);
			if(dT===0)return 0;
			const CL=f32(ms[(x+y*texW)*4+3]*cfg.cLiq),CR=f32(ms[j+3]*cfg.cLiq);
			const Ch=f32(f32(CL*CR)/f32(CL+CR));
			let dE=f32(f32(cfg.conductivity*dT)*dt);
			const limit=f32(f32(f32(0.5)*Ch)*Math.abs(dT));
			if(Math.abs(dE)>limit){dE=dE>0?limit:-limit;thermDiag.clamped++;}
			return dE;
		}
		// local sources, in place is safe (no pairwise coupling). bottom band:
		// fixed power per cell; top band: cold plate relaxing T toward T_amb.
		// every joule is charged to the ledger and to the therm.b/a accumulators
		// the gpu stats reduction reads back.
		function heatSources(dt){
			if(!cfg.thermalSources)return;
			const ms=grid.curBuf(mass),ts=grid.curBuf(therm);
			const band=Math.max(1,Math.round(cfg.heatBand));
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					if(ms[(x+y*texW)*4+3]<cfg.mEps||isWall(x,y))continue;
					const i=(x+y*texW)*4;
					let dE=0;
					if(y<=band)dE=f32(dE+f32(cfg.heatPower*dt));
					if(y>N-band){
						const C=f32(ms[i+3]*cfg.cLiq);
						const dT=f32(ts[i+1]-cfg.TAmb);
						const cooling=f32(f32(f32(cfg.coolTop*dT)*C)*dt);
						dE=f32(dE-cooling);
					}
					if(dE===0)continue;
					ts[i]=f32(ts[i]+dE);
					ts[i+1]=f32(ts[i]/f32(ms[i+3]*cfg.cLiq));
					if(dE>0){ledger.heatInE+=dE;ts[i+2]+=dE;}
					else{ledger.heatOutE-=dE;ts[i+3]-=dE;}
				}
			}
		}
		function step(dt){
			forces(dt);
			advectVel(dt);
			poissonRhs(dt);
			project(cfg.pressureIters);
			applyGrad(dt);
			computeFlux(dt);
			advectScalar();
			deriveT();
			conduction(dt);
			heatSources(dt);
			deriveT();
			cfg.frame++;
			cfg.time+=dt;
		}
		function stats(){
			const tot=acc.totalsFromFields(grid.curBuf(mass),grid.curBuf(therm),texW,texH);
			const mot=acc.motionTotals(grid.curBuf(vel),grid.curBuf(press),texW,texH,cfg.dt);
			const thermal=acc.thermalTotals(grid.curBuf(mass),grid.curBuf(therm),grid.curBuf(vel),texW,texH,cfg.mEps,cfg.cLiq,cfg.TAmb);
			return {frame:cfg.frame,tot,mot,thermal,ledger};
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
					if(buf[i+3]>=cfg.mEps)tbuf[i+1]=tbuf[i]/(buf[i+3]*cfg.cLiq);
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
		return {cfg,texW,texH,N,mass,therm,vel,press,flux,divsrc,flags,ledger,thermDiag,
			isWall,uBlocked,vBlocked,reset,step,stats,forces,advectVel,poissonRhs,project,applyGrad,computeFlux,advectScalar,
			deriveT,conduction,heatSources,
			applyDyePatch,applyWallPatch};
	}
	Fluid.core.solver={create,buildScenario,buildDyePatch,buildWallPatch,patchTotals,paintBlob};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.solver;
})(globalThis);
