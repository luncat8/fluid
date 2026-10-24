(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const grid=Fluid.core.grid;
	const f32=Math.fround;

	// phase bytes are the caloric branch (0.0.0 section 5.5); the frozen face
	// mask is a separate derived predicate (flags.b), never shared with walls.
	const PHASE_VAC=0,PHASE_SOLID=1,PHASE_MELTING=2,PHASE_LIQUID=3,PHASE_BOILING=4,PHASE_GAS=5;
	// one closed form of (m,E) for every consumer: T, phase fractions, phase
	// byte, target density rho*, expansion source S, and the branch-local c/k.
	// E=0 is solid at T=0; plateaus hold T constant at T_melt/T_boil while phi
	// sweeps. writes into the caller's object; the solver passes a scratch.
	function phaseInto(o,m,E,cfg){
		if(!(m>=cfg.mEps)){
			o.T=cfg.TAmb;o.phiL=0;o.phiG=0;o.phase=PHASE_VAC;
			o.rhoStar=cfg.rhoLiq;o.S=0;o.c=cfg.cLiq;o.k=cfg.kLiq;
			return o;
		}
		const E1=m*cfg.cSol*cfg.TMelt;
		const E2=E1+m*cfg.Lf;
		const E3=E2+m*cfg.cLiq*(cfg.TBoil-cfg.TMelt);
		const E4=E3+m*cfg.Lv;
		let T,phiL=0,phiG=0,phase;
		if(E<E1){T=E/(m*cfg.cSol);phase=PHASE_SOLID;}
		else if(E<E2){T=cfg.TMelt;phiL=(E-E1)/(m*cfg.Lf);phase=PHASE_MELTING;}
		else if(E<E3){T=cfg.TMelt+(E-E2)/(m*cfg.cLiq);phiL=1;phase=PHASE_LIQUID;}
		else if(E<E4){T=cfg.TBoil;phiG=(E-E3)/(m*cfg.Lv);phiL=1;phase=PHASE_BOILING;}
		else{T=cfg.TBoil+(E-E4)/(m*cfg.cGas);phiG=1;phiL=1;phase=PHASE_GAS;}
		const rhoGas=cfg.rhoGas0*cfg.TBoil/Math.max(T,cfg.TEps);
		const vStar=(1-phiG)/cfg.rhoLiq+phiG/rhoGas;
		const rhoStar=1/vStar;
		let S=0;
		if(phiG>0)S=(m-rhoStar)/(Math.max(m,rhoStar)*cfg.tauPhase);
		else if(m>cfg.rhoLiq)S=(m-cfg.rhoLiq)/(m*cfg.tauPhase);
		if(S>cfg.SMax)S=cfg.SMax;
		else if(S<-cfg.SMax)S=-cfg.SMax;
		const c=(1-phiL)*cfg.cSol+phiL*((1-phiG)*cfg.cLiq+phiG*cfg.cGas);
		const k=(1-phiL)*cfg.kSol+phiL*((1-phiG)*cfg.kLiq+phiG*cfg.kGas);
		o.T=T;o.phiL=phiL;o.phiG=phiG;o.phase=phase;
		o.rhoStar=rhoStar;o.S=S;o.c=c;o.k=k;
		return o;
	}
	function phaseOf(m,E,cfg){
		return phaseInto({T:0,phiL:0,phiG:0,phase:0,rhoStar:0,S:0,c:0,k:0},m,E,cfg);
	}
	function energyFromT(m,T,cfg){
		if(T<=cfg.TMelt)return m*cfg.cSol*T;
		if(T<cfg.TBoil)return m*(cfg.cSol*cfg.TMelt+cfg.Lf+cfg.cLiq*(T-cfg.TMelt));
		return m*(cfg.cSol*cfg.TMelt+cfg.Lf+cfg.cLiq*(cfg.TBoil-cfg.TMelt)+cfg.Lv+cfg.cGas*(T-cfg.TBoil));
	}
	function eqDist(m,E,T,cfg){
		let lo,hi;
		if(T<cfg.TMelt){lo=hi=m*cfg.cSol*T;}
		else if(T===cfg.TMelt){lo=m*cfg.cSol*cfg.TMelt;hi=lo+m*cfg.Lf;}
		else if(T<cfg.TBoil){lo=m*(cfg.cSol*cfg.TMelt+cfg.Lf+cfg.cLiq*(T-cfg.TMelt));hi=lo;}
		else if(T===cfg.TBoil){lo=m*(cfg.cSol*cfg.TMelt+cfg.Lf+cfg.cLiq*(cfg.TBoil-cfg.TMelt));hi=lo+m*cfg.Lv;}
		else{lo=m*(cfg.cSol*cfg.TMelt+cfg.Lf+cfg.cLiq*(cfg.TBoil-cfg.TMelt)+cfg.Lv+cfg.cGas*(T-cfg.TBoil));hi=lo;}
		if(E<lo)return lo-E;
		if(E>hi)return E-hi;
		return 0;
	}

	function setCell(massA,thermA,texW,x,y,r,g,b,m,T,cfg){
		const i=(x+y*texW)*4;
		massA[i]=r*m;massA[i+1]=g*m;massA[i+2]=b*m;massA[i+3]=m;
		thermA[i]=energyFromT(m,T,cfg);
		thermA[i+1]=T;
		thermA[i+2]=0;
		thermA[i+3]=0;
	}
	function buildScenario(cfg){
		const N=cfg.N,texW=N+2,texH=N+2;
		const massA=new Float32Array(texW*texH*4);
		const thermA=new Float32Array(texW*texH*4);
		const flagsA=new Uint8Array(texW*texH*4);
		const sc=cfg.scenario,TAmb=cfg.TAmb;
		if(sc==="ice"){
			const x0=Math.max(2,Math.round(N*0.28)),x1=Math.min(N-1,Math.round(N*0.72));
			const y0=Math.max(2,Math.round(N*0.10)),y1=Math.min(N-1,Math.round(N*0.48));
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					if(x>=x0&&x<=x1&&y>=y0&&y<=y1)setCell(massA,thermA,texW,x,y,0.12,0.62,0.85,1.0,245,cfg);
					else setCell(massA,thermA,texW,x,y,0.03,0.08,0.20,1.0,283,cfg);
				}
			}
			return {massA,thermA,flagsA,texW,texH};
		}
		if(sc==="cycle"){
			// closed water cycle: central frozen ice block sitting in a warm water pool
			// with open channels on both sides so plumes escape upward, boil into the
			// headspace, condense against the cold ceiling and rain back down.
			const x0=Math.max(2,Math.round(N*0.30)),x1=Math.min(N-1,Math.round(N*0.70));
			const yIce0=Math.max(3,Math.round(N*0.05)),yIce1=Math.max(yIce0+2,Math.round(N*0.18));
			const yPool=Math.max(yIce1+2,Math.round(N*0.26));
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const wave=2*Math.sin(6.2831853*3*(x-0.5)/N);
					if(x>=x0&&x<=x1&&y>=yIce0&&y<=yIce1)setCell(massA,thermA,texW,x,y,0.12,0.70,0.92,1.0,258,cfg);
					else if(y<=yPool)setCell(massA,thermA,texW,x,y,0.06,0.34,0.82,1.0,366+wave,cfg);
					else setCell(massA,thermA,texW,x,y,0.02,0.03,0.06,0.05,305,cfg);
				}
			}
			return {massA,thermA,flagsA,texW,texH};
		}
		if(sc==="jet"){
			const x0=Math.max(2,Math.round(N*0.40)),x1=Math.min(N-1,Math.round(N*0.60));
			const y1=Math.max(3,Math.round(N*0.16));
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					if(x>=x0&&x<=x1&&y<=y1)setCell(massA,thermA,texW,x,y,0.95,0.36,0.08,0.22,500,cfg);
					else setCell(massA,thermA,texW,x,y,0.03,0.06,0.12,0.14,293,cfg);
				}
			}
			return {massA,thermA,flagsA,texW,texH};
		}
		if(sc==="rain"){
			const yPool=Math.max(3,Math.round(N*0.22));
			const yTop=Math.min(N-1,Math.round(N*0.84));
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const wave=2*Math.sin(6.2831853*4*(x-0.5)/N);
					if(y<=yPool)setCell(massA,thermA,texW,x,y,0.12,0.48,0.88,0.95,365+wave,cfg);
					else if(y>=yTop)setCell(massA,thermA,texW,x,y,0.55,0.80,0.95,0.08,242,cfg);
					else setCell(massA,thermA,texW,x,y,0.03,0.06,0.10,0.06,285,cfg);
				}
			}
			return {massA,thermA,flagsA,texW,texH};
		}
		if(sc==="arch"){
			// frozen solid arch holding back an elevated pool of warm red fluid:
			// stays rigid while frozen, then melts from the outside in until the crown breaches.
			const xL=Math.max(2,Math.round(N*0.18)),xR=Math.min(N-1,Math.round(N*0.82));
			const legW=Math.max(2,Math.round(N*0.10));
			const ySpring=Math.max(4,Math.round(N*0.36));
			const thick=Math.max(3,Math.round(N*0.09));
			const span=Math.max(1,(xR-xL)*0.5),cx=0.5*(xL+xR);
			const rise=Math.max(2,Math.round(N*0.16));
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const nx=(x-cx)/span;
					const archTop=Math.abs(nx)<=1?ySpring+rise*(1-nx*nx):0;
					const inLeg=x>=xL&&x<=xR&&(x<xL+legW||x>xR-legW)&&y<=ySpring;
					const inVault=Math.abs(nx)<=1&&y>=archTop-thick&&y<=archTop;
					const inPool=x>xL+legW&&x<xR-legW&&y>archTop&&y<=ySpring+rise+Math.max(4,Math.round(N*0.18));
					if(inLeg||inVault)setCell(massA,thermA,texW,x,y,0.10,0.75,0.95,1.0,242,cfg);
					else if(inPool)setCell(massA,thermA,texW,x,y,0.90,0.24,0.12,1.0,305,cfg);
					else setCell(massA,thermA,texW,x,y,0.04,0.07,0.14,0.35,298,cfg);
				}
			}
			return {massA,thermA,flagsA,texW,texH};
		}
		for(let y=1;y<texH-1;y++){
			for(let x=1;x<texW-1;x++){
				const i=(x+y*texW)*4;
				let m=0.3;
				if(sc==="still"||sc==="convection")m=1.0;
				if(sc==="dam"&&x<Math.ceil(N*0.45)&&x>=Math.ceil(N*0.1)&&y<=Math.ceil(N*0.6))m=1.0;
				massA[i+3]=m;
				if(sc==="dam"&&m>0.5)massA[i+2]=0.5*m;
				thermA[i]=energyFromT(m,TAmb,cfg);
				thermA[i+1]=TAmb;
				thermA[i+2]=0;
				thermA[i+3]=0;
			}
		}
		if(sc==="convection"){
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=(x+y*texW)*4;
					const grad=cfg.convBotT+(cfg.convTopT-cfg.convBotT)*(y-0.5)/N;
					const T=grad+4*Math.sin(6.2831853*(x-0.5)/N)*Math.sin(3.14159265*(y-0.5)/N);
					thermA[i]=energyFromT(massA[i+3],T,cfg);
					thermA[i+1]=T;
				}
			}
		}
		if(sc==="blobs"){
			paintBlob(massA,thermA,texW,texH,N*0.3,N*0.5,N*0.12,[0.8,0,0],0.6,cfg);
			paintBlob(massA,thermA,texW,texH,N*0.7,N*0.5,N*0.12,[0,0.8,0],0.6,cfg);
			paintBlob(massA,thermA,texW,texH,N*0.5,N*0.25,N*0.12,[0,0,0.8],0.6,cfg);
		}
		return {massA,thermA,flagsA,texW,texH};
	}
	function paintBlob(massA,thermA,texW,texH,cx,cy,radius,rgb,dm,cfg){
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
				thermA[i]=energyFromT(massA[i+3],cfg.TAmb,cfg);
			}
		}
	}
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
				ePatch[x+y*w]=energyFromT(dm,cfg.TAmb,cfg);
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
		const sfield=new Float32Array(texW*texH*4);
		const flags=new Uint8Array(texW*texH*4);
		const fctLimiters=new Float32Array(texW*texH*6);
		const massFctTmp=new Float32Array(texW*texH*4);
		const acc=Fluid.core.stats;
		let ledger=null;
		const diag={divBefore:0,divAfter:0,pressureResidual:0,sSaturated:0,
			poissonRes:0,poissonRhs:0,poissonDrift:0,poissonDriftRhs:0,poissonDriftRel:0,
			macLimit:0,macClamp:0,velClamp:0};
		// mac-cormack scratch: the forward-advected field plus a stencil and a
		// counter reused per face. the field is allocated on first use so the
		// default (semi-lagrangian) path does not pay for it; its ghost ring stays
		// 0 exactly like the velocity field's, which the sampler reads on clamps.
		const macTaps=new Float32Array(4);
		const macCnt={limit:0,clamp:0};
		let macTmp=null;
		function macBuf(){
			if(!macTmp)macTmp=new Float32Array(texW*texH*2);
			return macTmp;
		}
		// 0.6.0 t6: vorticity confinement scratch (omega, |omega| and the cell
		// centred force), allocated on first use so the shipping path (eps=0)
		// does not pay for it.
		let confW=null,confS=null,confFx=null,confFy=null;
		const emitMod=Fluid.core.emitters;
		const emitterTable=emitMod?emitMod.create():null;
		let mg=null;
		if(cfg.useMultigrid && Fluid.core.multigrid){
			mg=Fluid.core.multigrid.create(cfg,texW,texH,{
				press, divsrc,
				massCur:()=>grid.curBuf(mass),
				flags
			});
		}
		function isWall(x,y){
			if(x<1||x>texW-2||y<1||y>texH-2)return true;
			return flags[(x+y*texW)*4]>0;
		}
		function isFrozen(x,y){
			if(x<1||x>texW-2||y<1||y>texH-2)return false;
			return flags[(x+y*texW)*4+2]>0;
		}
		function isBlocked(x,y){
			return isWall(x,y)||isFrozen(x,y);
		}
		function uBlocked(x,y){
			return x<=1||x>=texW-1||isBlocked(x-1,y)||isBlocked(x,y);
		}
		function vBlocked(x,y){
			return y<=1||y>=texH-1||isBlocked(x,y-1)||isBlocked(x,y);
		}
		function reset(){
			Fluid.core.config.prepareScenario(cfg);
			const sc=buildScenario(cfg);
			mass.a.set(sc.massA);mass.b.fill(0);
			therm.a.set(sc.thermA);therm.b.fill(0);
			vel.a.fill(0);vel.b.fill(0);
			press.a.fill(0);press.b.fill(0);
			flux.fill(0);divsrc.fill(0);sfield.fill(0);
			flags.set(sc.flagsA);
			mass.cur=0;therm.cur=0;vel.cur=0;press.cur=0;
			const tot=acc.totalsFromFields(mass.a,therm.a,texW,texH);
			ledger=acc.createLedger(tot);
			cfg.frame=0;cfg.time=0;
			diag.poissonRes=0;diag.poissonRhs=0;diag.poissonDrift=0;diag.poissonDriftRhs=0;diag.poissonDriftRel=0;
			diag.macLimit=0;diag.macClamp=0;
			diag.velClamp=0;
			if(emitterTable)emitMod.syncConfig(emitterTable,cfg,false);
			phaseEos(false);
		}
		function captureSolveResidual(){
			const mgOp=Fluid.core.multigrid;
			if(!mgOp)return;
			diag.poissonRes=mgOp.computeResidual(grid.curBuf(press),divsrc,grid.curBuf(mass),
				flags,texW,texH,cfg.rhoVac,null);
			diag.poissonRhs=diag.divBefore;
		}
		function forces(dt){
			const src=grid.curBuf(vel),dst=grid.nxtBuf(vel);
			const ms=grid.curBuf(mass),ts=grid.curBuf(therm);
			const R=cfg.mouseR,R2=R*R;
			const rhoMean=(ledger.initialM+ledger.injectedM-ledger.removedM)/(N*N);
			if(emitterTable)emitMod.syncConfig(emitterTable,cfg,false);
			const emU=emitterTable?emitterTable.upload:null;
			const momCnt=emU?emU[2]:0;
			const emMax=emU?emU[1]:0;
			const dtf=f32(dt);
			const conf=cfg.vorticityConfinement>0;
			if(conf)confine(src);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const ci=x+y*texW,vi=ci*2;
					let u=0,v=0;
					if(!uBlocked(x,y)){
						u=src[vi]+dt*0.5*(cellAx(ms,ts,x-1,y,rhoMean)+cellAx(ms,ts,x,y,rhoMean));
						if(cfg.mouseStrength>0){
							const f=mouseFall((x-1)-cfg.mouseX,(y-0.5)-cfg.mouseY,R,R2);
							u+=dt*cfg.mouseVX*f*cfg.mouseStrength;
						}
						if(momCnt>0){
							const px=f32(x-1),py=f32(y-0.5);
							for(let s=0;s<emMax;s++){
								const b=16+s*16;
								if(emU[b+4]!==4)continue;
								const dx=f32(px-emU[b]),dy=f32(py-emU[b+1]);
								const d2=f32(f32(dx*dx)+f32(dy*dy));
								if(d2>=emU[b+3])continue;
								const tf=Math.max(0,f32(1-f32(d2*emU[b+2])));
								const fw=f32(tf*tf);
								u=f32(f32(u)+f32(dtf*f32(emU[b+9]*fw)));
							}
						}
						// the u face (x,y) sits between the cells (x-1,y) and (x,y)
						if(conf)u+=dt*0.5*(confFx[ci-1]+confFx[ci]);
						u=clampAbs(u,cfg.vClamp);
					}
					if(!vBlocked(x,y)){
						v=src[vi+1]+dt*0.5*(cellAy(ms,ts,x,y-1,rhoMean)+cellAy(ms,ts,x,y,rhoMean));
						if(cfg.mouseStrength>0){
							const f=mouseFall((x-0.5)-cfg.mouseX,(y-1)-cfg.mouseY,R,R2);
							v+=dt*cfg.mouseVY*f*cfg.mouseStrength;
						}
						if(momCnt>0){
							const px=f32(x-0.5),py=f32(y-1);
							for(let s=0;s<emMax;s++){
								const b=16+s*16;
								if(emU[b+4]!==4)continue;
								const dx=f32(px-emU[b]),dy=f32(py-emU[b+1]);
								const d2=f32(f32(dx*dx)+f32(dy*dy));
								if(d2>=emU[b+3])continue;
								const tf=Math.max(0,f32(1-f32(d2*emU[b+2])));
								const fw=f32(tf*tf);
								v=f32(f32(v)+f32(dtf*f32(emU[b+10]*fw)));
							}
						}
						// the v face (x,y) sits between the cells (x,y-1) and (x,y)
						if(conf)v+=dt*0.5*(confFy[ci-texW]+confFy[ci]);
						v=clampAbs(v,cfg.vClamp);
					}
					dst[vi]=u;dst[vi+1]=v;
				}
			}
			grid.swap(vel);
		}
		// 0.6.0 t6 vorticity confinement: F = eps*h*(N x omega), N the unit
		// gradient of |omega|. in 2d that force is tangential, so it spins a
		// vortex up instead of tearing it apart, and it is an energy source:
		// the velocity clamp is the watchdog and must stay at 0.
		// cell centred velocity from its two faces. a blocked face is dropped,
		// not read as 0: the stored 0 is a wall condition, so averaging it in
		// would halve the centre velocity next to every wall and read as shear.
		function centreU(src,x,y){
			const vi=(x+y*texW)*2;
			const wl=!uBlocked(x,y),er=!uBlocked(x+1,y);
			if(wl&&er)return 0.5*(src[vi]+src[vi+2]);
			if(wl)return src[vi];
			if(er)return src[vi+2];
			return 0;
		}
		function centreV(src,x,y){
			const vi=(x+y*texW)*2;
			const sb=!vBlocked(x,y),nt=!vBlocked(x,y+1);
			if(sb&&nt)return 0.5*(src[vi+1]+src[vi+texW*2+1]);
			if(sb)return src[vi+1];
			if(nt)return src[vi+texW*2+1];
			return 0;
		}
		// z component of the curl at the cell centre: dv/dx - du/dy from the
		// neighbouring cell centred velocities, which is the average of the
		// cell's four corner curls and the only centred form a staggered grid
		// has. (vT-vB)-(uR-uL) is not it: that is dv/dy-du/dx, which is 0 for
		// solid-body rotation and for pure shear. the oracle asserts those
		// fixed points. a blocked or ghost neighbour mirrors this cell (free
		// slip), so the difference degrades to a half one-sided one instead of
		// reading a wall as a resolved shear.
		function vorticity(src,x,y){
			const xe=x<texW-2&&!isBlocked(x+1,y)?x+1:x;
			const xw=x>1&&!isBlocked(x-1,y)?x-1:x;
			const yn=y<texH-2&&!isBlocked(x,y+1)?y+1:y;
			const ys=y>1&&!isBlocked(x,y-1)?y-1:y;
			return 0.5*(centreV(src,xe,y)-centreV(src,xw,y))-0.5*(centreU(src,x,yn)-centreU(src,x,ys));
		}
		function vorticityAt(x,y){
			return vorticity(grid.curBuf(vel),x,y);
		}
		// cell centred force from the velocity field of this step. two passes
		// over the grid: omega and |omega|, then the force.
		function confine(src){
			if(!confS){
				confW=new Float32Array(texW*texH);
				confS=new Float32Array(texW*texH);
				confFx=new Float32Array(texW*texH);
				confFy=new Float32Array(texW*texH);
			}
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=x+y*texW;
					const w=isBlocked(x,y)?0:vorticity(src,x,y);
					confW[i]=w;
					confS[i]=w<0?-w:w;
				}
			}
			const eps=cfg.vorticityConfinement;
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=x+y*texW;
					const w=confW[i];
					if(w===0){confFx[i]=0;confFy[i]=0;continue;}
					// |omega| mirrors across a wall too, for the same reason
					// the curl does: the 0 stored in a blocked cell would
					// otherwise point the gradient into the wall.
					const xm=x>1&&!isBlocked(x-1,y)?i-1:i,xp=x<texW-2&&!isBlocked(x+1,y)?i+1:i;
					const ym=y>1&&!isBlocked(x,y-1)?i-texW:i,yp=y<texH-2&&!isBlocked(x,y+1)?i+texW:i;
					const gx=0.5*(confS[xp]-confS[xm]);
					const gy=0.5*(confS[yp]-confS[ym]);
					const len=Math.sqrt(gx*gx+gy*gy);
					if(len<1e-12){confFx[i]=0;confFy[i]=0;continue;}
					confFx[i]=eps*(gy/len)*w;
					confFy[i]=-eps*(gx/len)*w;
				}
			}
		}
		function cellAx(ms,ts,x,y,rhoMean){
			if(x<1||x>texW-2||y<1||y>texH-2)return 0;
			const i=(x+y*texW)*4;
			const m=ms[i+3];
			const tb=m>=cfg.mEps?cfg.gravityX*cfg.beta*(ts[i+1]-cfg.TRef):0;
			return cfg.gravityX*(rhoMean-m)/cfg.rhoRef+tb;
		}
		function cellAy(ms,ts,x,y,rhoMean){
			if(x<1||x>texW-2||y<1||y>texH-2)return 0;
			const i=(x+y*texW)*4;
			const m=ms[i+3];
			const tb=m>=cfg.mEps?cfg.gravityY*cfg.beta*(ts[i+1]-cfg.TRef):0;
			return cfg.gravityY*(rhoMean-m)/cfg.rhoRef+tb;
		}
		function mouseFall(dx,dy,R,R2){
			const r2=dx*dx+dy*dy;
			if(r2>=R2)return 0;
			const f=1-Math.sqrt(r2)/R;
			return f*f;
		}
		function clampAbs(v,lim){
			if(v>lim){diag.velClamp++;return lim;}
			if(v<-lim){diag.velClamp++;return -lim;}
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
		function clampTo(v,lo,hi){
			return v<lo?lo:(v>hi?hi:v);
		}
		// bilinear taps of one component on its own staggered face grid. `taps`
		// (caller owned, 4 floats) receives the stencil so the mac-cormack limiter
		// can bound its correction without sampling twice; pass null to skip it.
		function sampleU(src,px,py,taps){
			const i0=Math.floor(px),fx=px-i0;
			const l0=Math.floor(py+0.5),fy=py+0.5-l0;
			const u00=loadU(src,i0+1,l0),u10=loadU(src,i0+2,l0);
			const u01=loadU(src,i0+1,l0+1),u11=loadU(src,i0+2,l0+1);
			if(taps){taps[0]=u00;taps[1]=u10;taps[2]=u01;taps[3]=u11;}
			const a0=u00+(u10-u00)*fx,a1=u01+(u11-u01)*fx;
			return a0+(a1-a0)*fy;
		}
		function sampleV(src,px,py,taps){
			const k0=Math.floor(px+0.5),fx=px+0.5-k0;
			const l0=Math.floor(py),fy=py-l0;
			const v00=loadV(src,k0,l0+1),v10=loadV(src,k0+1,l0+1);
			const v01=loadV(src,k0,l0+2),v11=loadV(src,k0+1,l0+2);
			if(taps){taps[0]=v00;taps[1]=v10;taps[2]=v01;taps[3]=v11;}
			const a0=v00+(v10-v00)*fx,a1=v01+(v11-v01)*fx;
			return a0+(a1-a0)*fy;
		}
		// the trace velocity of a face is its own component plus the cross
		// component averaged from the four cells that share it. both advection
		// schemes read it, so it is one definition.
		function traceU(src,x,y){
			return 0.25*(loadV(src,x-1,y)+loadV(src,x-1,y+1)+loadV(src,x,y)+loadV(src,x,y+1));
		}
		function traceV(src,x,y){
			return 0.25*(loadU(src,x,y-1)+loadU(src,x+1,y-1)+loadU(src,x,y)+loadU(src,x+1,y));
		}
		// one semi-lagrangian trace of src into dst, which is either the ping pong
		// partner (the shipping path) or the mac-cormack scratch (pass 1 of 2). one
		// definition, so the two schemes cannot drift on the forward pass.
		function slInto(src,dst,dt){
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const vi=(x+y*texW)*2;
					let u=0,v=0;
					if(!uBlocked(x,y)){
						const px=clampTo((x-1)-dt*src[vi],0,N);
						const py=clampTo((y-0.5)-dt*traceU(src,x,y),0.5,N-0.5);
						u=sampleU(src,px,py,null);
					}
					if(!vBlocked(x,y)){
						const px=clampTo((x-0.5)-dt*traceV(src,x,y),0.5,N-0.5);
						const py=clampTo((y-1)-dt*src[vi+1],0,N);
						v=sampleV(src,px,py,null);
					}
					dst[vi]=u;dst[vi+1]=v;
				}
			}
		}
		function advectVel(dt){
			slInto(grid.curBuf(vel),grid.nxtBuf(vel),dt);
			grid.swap(vel);
		}
		// mac-cormack (0.6.0 t5): the forward semi-lagrangian pass diffuses, so a
		// second pass traces that result forward along the same face velocity and
		// the difference cancels the smearing. c = f + (src - back)/2 creates new
		// extrema, so it is limited into the min/max of the four taps the forward
		// sample used, which keeps a face inside its own first-order stencil.
		// macLimit counts the faces the limiter moved and macClamp the vClamp hits.
		// under the shipping projection the cuts are roundoff scale (the unlimited
		// scheme agrees to six digits); what the limiter actually holds is the face
		// next to a blocked face, where the backward trace reads that face's 0 and
		// the raw correction extrapolates past the boundary value every step.
		function macLimit(f,srcVal,back,lim,taps,cnt){
			const c=f+0.5*(srcVal-back);
			if(!lim)return c;
			const lo=Math.min(taps[0],taps[1],taps[2],taps[3]);
			const hi=Math.max(taps[0],taps[1],taps[2],taps[3]);
			if(c<lo){cnt.limit++;return lo;}
			if(c>hi){cnt.limit++;return hi;}
			return c;
		}
		function macWatch(c,cnt){
			if(c>cfg.vClamp){cnt.clamp++;return cfg.vClamp;}
			if(c<-cfg.vClamp){cnt.clamp++;return -cfg.vClamp;}
			return c;
		}
		function advectVelMac(dt){
			const src=grid.curBuf(vel),dst=grid.nxtBuf(vel),fwd=macBuf();
			const taps=macTaps,cnt=macCnt,lim=cfg.macLimiter?1:0;
			cnt.limit=0;cnt.clamp=0;
			slInto(src,fwd,dt);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const vi=(x+y*texW)*2;
					let u=0,v=0;
					if(!uBlocked(x,y)){
						const uc=src[vi],vc=traceU(src,x,y);
						const f=sampleU(src,clampTo((x-1)-dt*uc,0,N),clampTo((y-0.5)-dt*vc,0.5,N-0.5),lim?taps:null);
						const back=sampleU(fwd,clampTo((x-1)+dt*uc,0,N),clampTo((y-0.5)+dt*vc,0.5,N-0.5),null);
						u=macWatch(macLimit(f,uc,back,lim,taps,cnt),cnt);
					}
					if(!vBlocked(x,y)){
						const vc=src[vi+1],uc=traceV(src,x,y);
						const f=sampleV(src,clampTo((x-0.5)-dt*uc,0.5,N-0.5),clampTo((y-1)-dt*vc,0,N),lim?taps:null);
						const back=sampleV(fwd,clampTo((x-0.5)+dt*uc,0.5,N-0.5),clampTo((y-1)+dt*vc,0,N),null);
						v=macWatch(macLimit(f,vc,back,lim,taps,cnt),cnt);
					}
					dst[vi]=u;dst[vi+1]=v;
				}
			}
			diag.macLimit=cnt.limit;
			diag.macClamp=cnt.clamp;
			grid.swap(vel);
		}
		function advectVelocity(dt){
			if(cfg.advectMac)advectVelMac(dt);
			else advectVel(dt);
		}
		function poissonRhs(dt){
			const vv=grid.curBuf(vel);
			let maxB=0;
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=x+y*texW;
					divsrc[i]=0;
					if(isBlocked(x,y))continue;
					const uL=vv[(i)*2],uR=vv[(i+1)*2];
					const vB=vv[(i)*2+1],vT=vv[(i+texW)*2+1];
					const b=(uR-uL+vT-vB)/dt-sfield[i*4];
					divsrc[i]=b;
					const ab=Math.abs(b);
					if(ab>maxB) maxB=ab;
				}
			}
			diag.divBefore=maxB;
		}
		function project(iters){
			if(mg && cfg.useMultigrid){
				const cycles=Math.max(1,Math.round(cfg.mgCycles||1));
				for(let c=0;c<cycles;c++)mg.vcycle(cfg.mgPre||4,iters,cfg.mgPost||8);
				captureSolveResidual();
				return;
			}
			const ms=grid.curBuf(mass);
			for(let k=0;k<iters;k++){
				for(let color=0;color<2;color++){
					const src=grid.curBuf(press),dst=grid.nxtBuf(press);
					for(let y=1;y<texH-1;y++){
						for(let x=1;x<texW-1;x++){
							const i=x+y*texW;
							if(((x+y)&1)!==color){dst[i]=src[i];continue;}
							if(isBlocked(x,y)){dst[i]=0;continue;}
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
			captureSolveResidual();
		}
		function faceW(ms,x,y,nx,ny){
			if(nx<1||nx>texW-2||ny<1||ny>texH-2)return 0;
			if(isBlocked(nx,ny))return 0;
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
					if(fx!==0){
						const s=fx>0?outFlux(vv,dt,x-1,y):outFlux(vv,dt,x,y);
						if(s>0.999999)fx*=0.999999/s;
					}
					if(fy!==0){
						const s=fy>0?outFlux(vv,dt,x,y-1):outFlux(vv,dt,x,y);
						if(s>0.999999)fy*=0.999999/s;
					}
					flux[i2]=fx;flux[i2+1]=fy;
				}
			}
		}
		function scalarValue(mbuf,index,field){
			return mbuf[index*4+field];
		}
		function antiFlux(mSrc,faceFlux,left,right,field){
			const a=Math.abs(faceFlux);
			return 0.5*a*(1-a)*(scalarValue(mSrc,right,field)-scalarValue(mSrc,left,field));
		}
		function limiterIndex(index,field){
			return index*6+field*2;
		}
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
					const remaining=1-(Math.max(-fw,0)+Math.max(fe,0)+Math.max(-fs,0)+Math.max(fn,0));
					for(let c=0;c<4;c++){
						const qm=mS[i4+c],qt=tS[i4+c];
						const qmW=mS[i4-4+c],qmE=mS[i4+4+c],qmS=mS[i4-texW*4+c],qmN=mS[i4+texW*4+c];
						const qtW=tS[i4-4+c],qtE=tS[i4+4+c],qtS=tS[i4-texW*4+c],qtN=tS[i4+texW*4+c];
						mD[i4+c]=qm*remaining+qmW*Math.max(fw,0)+qmE*Math.max(-fe,0)+qmS*Math.max(fs,0)+qmN*Math.max(-fn,0);
						tD[i4+c]=qt*remaining+qtW*Math.max(fw,0)+qtE*Math.max(-fe,0)+qtS*Math.max(fs,0)+qtN*Math.max(-fn,0);
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
		const phaseScratch={T:0,phiL:0,phiG:0,phase:0,rhoStar:0,S:0,c:0,k:0};
		const phaseN={T:0,phiL:0,phiG:0,phase:0,rhoStar:0,S:0,c:0,k:0};
		// phase_eos (0.0.0 5.5): pass 1 computes state + pre-zeroing frozenMomentum so
		// adjacent simultaneous freezes are order-independent and match GPU ping-pong;
		// pass 2 zeroes every face touching any frozen cell.
		function phaseEos(chargeLatent){
			const ms=grid.curBuf(mass),ts=grid.curBuf(therm),vs=grid.curBuf(vel);
			diag.sSaturated=0;
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=(x+y*texW)*4,vi=(x+y*texW)*2;
					const m=ms[i+3];
					phaseInto(phaseScratch,m,ts[i],cfg);
					const s=phaseScratch;
					ts[i+1]=s.T;
					if(chargeLatent){
						ledger.latentChargedE+=m*cfg.Lf*Math.abs(s.phiL-sfield[i+1])+m*cfg.Lv*Math.abs(s.phiG-sfield[i+2]);
					}
					sfield[i]=s.S;sfield[i+1]=s.phiL;sfield[i+2]=s.phiG;
					const sat=s.S>=cfg.SMax||s.S<=-cfg.SMax;
					if(sat)diag.sSaturated++;
					flags[i+1]=s.phase;
					const frozen=s.phiL<0.5&&s.phiG===0&&m>=cfg.mEps;
					if(frozen&&flags[i+2]===0){
						const uc=0.5*(vs[vi]+vs[vi+2]),vc=0.5*(vs[vi+1]+vs[vi+texW*2+1]);
						ledger.frozenMomentum+=m*Math.sqrt(uc*uc+vc*vc);
					}
					flags[i+2]=frozen?1:0;
					flags[i+3]=sat?1:0;
				}
			}
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					if(flags[(x+y*texW)*4+2]===0)continue;
					const vi=(x+y*texW)*2;
					vs[vi]=0;vs[vi+1]=0;vs[vi+2]=0;vs[vi+texW*2+1]=0;
				}
			}
		}
		function conduction(dt){
			const ms=grid.curBuf(mass);
			const src=grid.curBuf(therm),dst=grid.nxtBuf(therm);
			thermDiag.clamped=0;
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const i=(x+y*texW)*4;
					const m=ms[i+3],E0=src[i];
					let E=E0;
					if(m>=cfg.mEps&&!isWall(x,y)){
						phaseInto(phaseScratch,m,E0,cfg);
						E=f32(E+faceExchange(ms,src,phaseScratch,x,y,x-1,y,m,E0,dt));
						E=f32(E+faceExchange(ms,src,phaseScratch,x,y,x+1,y,m,E0,dt));
						E=f32(E+faceExchange(ms,src,phaseScratch,x,y,x,y-1,m,E0,dt));
						E=f32(E+faceExchange(ms,src,phaseScratch,x,y,x,y+1,m,E0,dt));
						if(cfg.wallCool>0&&(isWall(x-1,y)||isWall(x+1,y)||isWall(x,y-1)||isWall(x,y+1))){
							const dT=f32(src[i+1]-cfg.TAmb);
							const C=f32(m*phaseScratch.c);
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
					phaseInto(phaseScratch,m,E,cfg);
					dst[i+1]=phaseScratch.T;
				}
			}
			grid.swap(therm);
		}
		function faceExchange(ms,ts,own,x,y,nx,ny,mL,EL,dt){
			if(nx<1||nx>texW-2||ny<1||ny>texH-2)return 0;
			const j=(nx+ny*texW)*4;
			if(flags[j]>0)return 0;
			const mR=ms[j+3];
			if(mR<cfg.mEps)return 0;
			phaseInto(phaseN,mR,ts[j],cfg);
			const dT=f32(phaseN.T-own.T);
			if(dT===0)return 0;
			const CL=f32(mL*own.c),CR=f32(mR*phaseN.c);
			const Ch=f32(f32(CL*CR)/f32(CL+CR));
			const kf=f32(f32(2*own.k*phaseN.k)/f32(own.k+phaseN.k));
			let dE=f32(f32(kf*dT)*dt);
			const limit=f32(f32(f32(0.5)*Ch)*Math.abs(dT));
			if(Math.abs(dE)>limit){dE=dE>0?limit:-limit;thermDiag.clamped++;}
			const bud=f32(f32(0.25)*Math.min(eqDist(mL,EL,phaseN.T,cfg),eqDist(mR,ts[j],own.T,cfg)));
			if(Math.abs(dE)>bud){dE=dE>0?bud:-bud;thermDiag.clamped++;}
			return dE;
		}
		function heatSources(dt){
			if(!cfg.thermalSources)return;
			const ms=grid.curBuf(mass),ts=grid.curBuf(therm);
			const band=Math.max(1,Math.round(cfg.heatBand));
			const ceilT=cfg.TCeil>0?cfg.TCeil:cfg.TAmb;
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					if(ms[(x+y*texW)*4+3]<cfg.mEps||isWall(x,y))continue;
					const i=(x+y*texW)*4;
					phaseInto(phaseScratch,ms[i+3],ts[i],cfg);
					let dE=0;
					if(y<=band&&(!cfg.heatTarget||ts[i+1]<cfg.heatTarget))dE=f32(dE+f32(cfg.heatPower*dt));
					if(y>N-band){
						const C=f32(ms[i+3]*phaseScratch.c);
						const dT=f32(ts[i+1]-ceilT);
						const cooling=f32(f32(f32(cfg.coolTop*dT)*C)*dt);
						dE=f32(dE-cooling);
					}
					if(dE===0)continue;
					ts[i]=f32(ts[i]+dE);
					phaseInto(phaseScratch,ms[i+3],ts[i],cfg);
					ts[i+1]=phaseScratch.T;
					if(dE>0){ledger.heatInE+=dE;ts[i+2]+=dE;}
					else{ledger.heatOutE-=dE;ts[i+3]-=dE;}
				}
			}
		}
		// 0.7.0 t3 scalar/thermal emitters: realized texel deltas (mNew - m, eNew - E)
		// are charged directly so f32 quantization at high base mass/enthalpy cannot
		// accumulate systematic drift over tens of thousands of substeps.
		function applyEmitters(dt){
			if(!emitterTable)return;
			emitMod.syncConfig(emitterTable,cfg,false);
			const emU=emitterTable.upload;
			if(!(emU[3]>0))return;
			const emMax=emU[1];
			const ms=grid.curBuf(mass),ts=grid.curBuf(therm);
			const dtf=f32(dt),mEps=f32(cfg.mEps);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					if(isWall(x,y))continue;
					const i=(x+y*texW)*4;
					let dr=ms[i],dg=ms[i+1],db=ms[i+2],m=ms[i+3];
					let E=ts[i],hIn=ts[i+2],hOut=ts[i+3];
					let changed=false;
					const px=f32(x-0.5),py=f32(y-0.5);
					for(let s=0;s<emMax;s++){
						const b=16+s*16;
						const kind=emU[b+4]|0;
						if(kind===0||kind===4)continue;
						const val=emU[b+5];
						if(!(val>0))continue;
						const dx=f32(px-emU[b]),dy=f32(py-emU[b+1]);
						const d2=f32(f32(dx*dx)+f32(dy*dy));
						if(d2>=emU[b+3])continue;
						const tf=Math.max(0,f32(1-f32(d2*emU[b+2])));
						const fw=f32(tf*tf);
						const amt=f32(f32(val*dtf)*fw);
						if(!(amt>0))continue;
						if(kind===1){
							const ddr=f32(emU[b+6]*amt),ddg=f32(emU[b+7]*amt),ddb=f32(emU[b+8]*amt);
							const dE=f32(amt*emU[b+11]);
							const mNew=f32(m+amt),drNew=f32(dr+ddr),dgNew=f32(dg+ddg),dbNew=f32(db+ddb),eNew=f32(E+dE);
							ledger.injectedM+=f32(mNew-m);
							ledger.injectedDr+=f32(drNew-dr);
							ledger.injectedDg+=f32(dgNew-dg);
							ledger.injectedDb+=f32(dbNew-db);
							ledger.injectedE+=f32(eNew-E);
							m=mNew;dr=drNew;dg=dgNew;db=dbNew;E=eNew;
							changed=true;
						}else if(kind===5){
							if(m>0){
								if(amt>=m){
									ledger.removedM+=m;
									ledger.removedDr+=dr;
									ledger.removedDg+=dg;
									ledger.removedDb+=db;
									ledger.removedE+=E;
									m=0;dr=0;dg=0;db=0;E=0;
								}else{
									const frac=Math.min(1,Math.max(0,f32(amt/m)));
									const mNew=Math.max(0,f32(m-amt));
									const drNew=Math.max(0,f32(dr-f32(dr*frac)));
									const dgNew=Math.max(0,f32(dg-f32(dg*frac)));
									const dbNew=Math.max(0,f32(db-f32(db*frac)));
									const eNew=Math.max(0,f32(E-f32(E*frac)));
									ledger.removedM+=f32(m-mNew);
									ledger.removedDr+=f32(dr-drNew);
									ledger.removedDg+=f32(dg-dgNew);
									ledger.removedDb+=f32(db-dbNew);
									ledger.removedE+=f32(E-eNew);
									m=mNew;dr=drNew;dg=dgNew;db=dbNew;E=eNew;
								}
								changed=true;
							}
						}else if(kind===2){
							if(m>=mEps){
								let dE=amt;
								const hCap=emU[b+11];
								if(hCap>0){
									const eCeil=f32(m*hCap);
									if(E>=eCeil)dE=0;
									else dE=Math.min(dE,f32(eCeil-E));
								}
								if(dE>0){
									const eNew=f32(E+dE);
									const dReal=f32(eNew-E);
									E=eNew;
									hIn=f32(hIn+dReal);
									ledger.heatInE+=dReal;
									changed=true;
								}
							}
						}else if(kind===3||kind===6){
							if(m>=mEps){
								const eFloor=f32(m*emU[b+11]);
								if(E>eFloor){
									const cool=Math.min(amt,f32(E-eFloor));
									const eNew=Math.max(eFloor,f32(E-cool));
									const dReal=f32(E-eNew);
									E=eNew;
									hOut=f32(hOut+dReal);
									ledger.heatOutE+=dReal;
									changed=true;
								}
							}
						}
					}
					if(changed){
						ms[i]=dr;ms[i+1]=dg;ms[i+2]=db;ms[i+3]=m;
						ts[i]=E;ts[i+2]=hIn;ts[i+3]=hOut;
						phaseInto(phaseScratch,m,E,cfg);
						ts[i+1]=phaseScratch.T;
					}
				}
			}
		}
		function addEmitter(spec){
			return emitterTable?emitMod.add(emitterTable,cfg,spec):-1;
		}
		function moveEmitter(id,x,y){
			return emitterTable?emitMod.move(emitterTable,cfg,id,x,y):false;
		}
		function setEmitter(id,patch){
			return emitterTable?emitMod.set(emitterTable,cfg,id,patch):false;
		}
		function removeEmitter(id){
			return emitterTable?emitMod.remove(emitterTable,id):false;
		}
		function emitterCount(){
			return emitterTable?emitterTable.live:0;
		}
		function step(dt){
			forces(dt);
			advectVelocity(dt);
			poissonRhs(dt);
			project(Fluid.core.config.solverIters(cfg));
			applyGrad(dt);
			const motPost=acc.motionTotals(grid.curBuf(vel),grid.curBuf(press),texW,texH,dt);
			diag.divAfter=motPost.maxDiv;
			diag.pressureResidual=diag.divBefore>0?motPost.maxDiv/diag.divBefore:0;
			computeFlux(dt);
			advectScalar();
			phaseEos(true);
			conduction(dt);
			applyEmitters(dt);
			heatSources(dt);
			phaseEos(true);
			cfg.frame++;
			cfg.time+=dt;
		}
		function stats(){
			const tot=acc.totalsFromFields(grid.curBuf(mass),grid.curBuf(therm),texW,texH);
			const mot=acc.motionTotals(grid.curBuf(vel),grid.curBuf(press),texW,texH,cfg.dt);
			// poissonRes is the system just solved (captured in project, before transport).
			// the recompute here is the post-step operator: coefficient drift, same
			// number the gpu writes to slots 34/35. the ratio uses the post-step rhs,
			// which is what the pre-split `true` column was, so a re-capture's drift
			// is comparable to those logs. a freeze can inflate it; that is the point.
			const mgOp=Fluid.core.multigrid;
			const driftSink={rhs:0};
			const poissonDrift=mgOp?mgOp.computeResidual(grid.curBuf(press),divsrc,grid.curBuf(mass),
				flags,texW,texH,cfg.rhoVac,null,driftSink):0;
			const poissonRes=diag.poissonRes||0;
			const poissonRhs=diag.poissonRhs||0;
			const poissonRel=poissonRhs>0?poissonRes/poissonRhs:0;
			const poissonDriftRhs=driftSink.rhs;
			const poissonDriftRel=poissonDriftRhs>0?poissonDrift/poissonDriftRhs:0;
			diag.poissonDrift=poissonDrift;
			diag.poissonDriftRhs=poissonDriftRhs;
			diag.poissonDriftRel=poissonDriftRel;
			const thermal=acc.thermalTotals(grid.curBuf(mass),grid.curBuf(therm),grid.curBuf(vel),texW,texH,cfg.mEps,cfg.cLiq,cfg.TAmb);
			const phaseCounts=[0,0,0,0,0,0];
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					phaseCounts[flags[(x+y*texW)*4+1]]++;
				}
			}
			return {frame:cfg.frame,tot,mot,thermal,ledger,phaseCounts,
				diag:{divBefore:diag.divBefore,divAfter:diag.divAfter,residual:diag.pressureResidual,
					sSaturated:diag.sSaturated,poissonRes,poissonRhs,poissonRel,
					poissonDrift,poissonDriftRhs,poissonDriftRel,
					macLimit:diag.macLimit,macClamp:diag.macClamp,velClamp:diag.velClamp}};
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
			phaseEos(true);
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
		return {cfg,texW,texH,N,mass,therm,vel,press,flux,divsrc,sfield,flags,ledger,thermDiag,diag,mg,emitterTable,
			isWall,isFrozen,isBlocked,uBlocked,vBlocked,reset,step,stats,forces,advectVelocity,poissonRhs,project,applyGrad,computeFlux,advectScalar,
			phaseEos,conduction,applyEmitters,heatSources,vorticityAt,
			applyDyePatch,applyWallPatch,addEmitter,moveEmitter,setEmitter,removeEmitter,emitterCount};
	}
	Fluid.core.solver={create,buildScenario,buildDyePatch,buildWallPatch,patchTotals,paintBlob,
		phaseOf,phaseInto,energyFromT,eqDist,PHASE_VAC,PHASE_SOLID,PHASE_MELTING,PHASE_LIQUID,PHASE_BOILING,PHASE_GAS};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.solver;
})(globalThis);
