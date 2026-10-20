(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const grid=Fluid.core.grid;

	// two-level V-cycle for the variable-coefficient pressure solve.
	// both painted walls (flags.r) and frozen substance (flags.b) block coupling.
	// a coarse macro-cell is blocked only when all SxS fine cells are blocked so no
	// open fine residual is ever dropped across a cavity boundary, and prolongation
	// mirrors across blocked coarse cells so constant Neumann modes stay constant.
	function cellBlocked(flagsArr,i4){
		return flagsArr[i4]>0||flagsArr[i4+2]>0;
	}
	// the poisson operator for one level: harmonic-mean face weights, zero across a
	// wall or a frozen cell. shared by the v-cycle and by the frame-end residual
	// readout, so the reported number can never drift from the system solved.
	function faceWeight(massArr,texW,texH,x,y,nx,ny,flagsArr,rhoVac){
		if(nx<1||nx>texW-2||ny<1||ny>texH-2)return 0;
		if(cellBlocked(flagsArr,(nx+ny*texW)*4))return 0;
		const ra=Math.max(massArr[(x+y*texW)*4+3],rhoVac);
		const rb=Math.max(massArr[(nx+ny*texW)*4+3],rhoVac);
		return 1/(2*ra*rb/(ra+rb));
	}
	// out[i]=b-(L p-wsum*p) per open cell, 0 in blocked cells, and the return value
	// is max|out| — the true poisson residual (0.6.0 4.1). `out` may be null when
	// only the norm is wanted, so a diagnostic readout needs no scratch field.
	function computeResidual(pressArr,divArr,massArr,flagsArr,texW,texH,rhoVac,out){
		let maxA=0;
		for(let y=1;y<texH-1;y++){
			for(let x=1;x<texW-1;x++){
				const i=x+y*texW;
				let r=0;
				if(!cellBlocked(flagsArr,i*4)){
					let wsum=0,sum=0,w;
					w=faceWeight(massArr,texW,texH,x,y,x-1,y,flagsArr,rhoVac);
					if(w>0){wsum+=w;sum+=w*pressArr[i-1];}
					w=faceWeight(massArr,texW,texH,x,y,x+1,y,flagsArr,rhoVac);
					if(w>0){wsum+=w;sum+=w*pressArr[i+1];}
					w=faceWeight(massArr,texW,texH,x,y,x,y-1,flagsArr,rhoVac);
					if(w>0){wsum+=w;sum+=w*pressArr[i-texW];}
					w=faceWeight(massArr,texW,texH,x,y,x,y+1,flagsArr,rhoVac);
					if(w>0){wsum+=w;sum+=w*pressArr[i+texW];}
					r=divArr[i]-(sum-wsum*pressArr[i]);
				}
				if(out)out[i]=r;
				const a=r<0?-r:r;
				if(a>maxA)maxA=a;
			}
		}
		return maxA;
	}
	function create(cfg,texW,texH,refs){
		const N=cfg.N;
		let S=Math.max(2,Math.round(cfg.mgCoarsen||2));
		while(S>2 && Math.floor(N/S)<4)S=Math.floor(S/2);
		const Nc=Math.floor(N/S);
		const texWc=Nc+2, texHc=Nc+2;
		const pressC=grid.allocPingPong(texWc,texHc,4,Float32Array);
		const divC=new Float32Array(texWc*texHc);
		const massC=new Float32Array(texWc*texHc*4);
		const flagsC=new Uint8Array(texWc*texHc*4);
		const resFine=new Float32Array(texW*texH);
		const corrFine=new Float32Array(texW*texH);
		const faceW=(massArr,texW_,texH_,x,y,nx,ny,flagsArr)=>
			faceWeight(massArr,texW_,texH_,x,y,nx,ny,flagsArr,cfg.rhoVac);
		function coarsen(){
			const massFine=refs.massCur(),flagsFine=refs.flags;
			for(let yc=1;yc<texHc-1;yc++){
				for(let xc=1;xc<texWc-1;xc++){
					let sum=0,openCount=0;
					for(let oy=0;oy<S;oy++)for(let ox=0;ox<S;ox++){
						const fi=S*(xc-1)+1+ox+(S*(yc-1)+1+oy)*texW;
						if(cellBlocked(flagsFine,fi*4))continue;
						sum+=massFine[fi*4+3];
						openCount++;
					}
					const ic=(xc+yc*texWc)*4;
					massC[ic+3]=openCount>0?sum/openCount:0;
					flagsC[ic]=openCount===0?255:0;
					flagsC[ic+2]=0;
				}
			}
		}
		function smooth(pressArr,divArr,massArr,flagsArr,texW_,texH_,iters){
			for(let k=0;k<iters;k++){
				for(let color=0;color<2;color++){
					for(let y=1;y<texH_-1;y++){
						for(let x=1;x<texW_-1;x++){
							if(((x+y)&1)!==color)continue;
							const i=x+y*texW_;
							if(cellBlocked(flagsArr,i*4)){pressArr[i]=0;continue;}
							let wsum=0,sum=0,w;
							w=faceW(massArr,texW_,texH_,x,y,x-1,y,flagsArr);
							if(w>0){wsum+=w;sum+=w*pressArr[i-1];}
							w=faceW(massArr,texW_,texH_,x,y,x+1,y,flagsArr);
							if(w>0){wsum+=w;sum+=w*pressArr[i+1];}
							w=faceW(massArr,texW_,texH_,x,y,x,y-1,flagsArr);
							if(w>0){wsum+=w;sum+=w*pressArr[i-texW_];}
							w=faceW(massArr,texW_,texH_,x,y,x,y+1,flagsArr);
							if(w>0){wsum+=w;sum+=w*pressArr[i+texW_];}
							pressArr[i]=wsum>0?(sum-divArr[i])/wsum:0;
						}
					}
				}
			}
		}
		function restrictResidual(){
			for(let yc=1;yc<texHc-1;yc++){
				for(let xc=1;xc<texWc-1;xc++){
					let sum=0;
					for(let oy=0;oy<S;oy++)for(let ox=0;ox<S;ox++){
						const xf=S*(xc-1)+1+ox,yf=S*(yc-1)+1+oy;
						sum+=resFine[xf+yf*texW];
					}
					divC[xc+yc*texWc]=sum;
				}
			}
			for(let x=0;x<texWc;x++){divC[x]=0;divC[x+(texHc-1)*texWc]=0;}
			for(let y=0;y<texHc;y++){divC[y*texWc]=0;divC[texWc-1+y*texWc]=0;}
		}
		function sampleCoarse(coarse,cx,cy,fallback){
			const idx=cx+cy*texWc;
			return flagsC[idx*4]>0?fallback:coarse[idx];
		}
		function prolongAndCorrect(){
			const pressFine=grid.curBuf(refs.press);
			const coarse=grid.curBuf(pressC);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const hc=Math.floor((x-1)/S)+1+(Math.floor((y-1)/S)+1)*texWc;
					const base=coarse[hc];
					const xf=(x-0.5)/S+0.5, yf=(y-0.5)/S+0.5;
					const xc=Math.floor(xf), yc=Math.floor(yf);
					const fx=xf-xc, fy=yf-yc;
					const x0=Math.max(1,Math.min(Nc,xc)), x1=Math.max(1,Math.min(Nc,xc+1));
					const y0=Math.max(1,Math.min(Nc,yc)), y1=Math.max(1,Math.min(Nc,yc+1));
					const v00=sampleCoarse(coarse,x0,y0,base), v10=sampleCoarse(coarse,x1,y0,base);
					const v01=sampleCoarse(coarse,x0,y1,base), v11=sampleCoarse(coarse,x1,y1,base);
					const v0=v00+(v10-v00)*fx, v1=v01+(v11-v01)*fx;
					corrFine[x+y*texW]=v0+(v1-v0)*fy;
				}
			}
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					if(cellBlocked(refs.flags,(x+y*texW)*4))continue;
					pressFine[x+y*texW]+=corrFine[x+y*texW];
				}
			}
		}
		function vcycle(preSmooth,coarseIters,postSmooth){
			const p=grid.curBuf(refs.press);
			coarsen();
			smooth(p,refs.divsrc,refs.massCur(),refs.flags,texW,texH,preSmooth);
			computeResidual(p,refs.divsrc,refs.massCur(),refs.flags,texW,texH,cfg.rhoVac,resFine);
			restrictResidual();
			pressC.a.fill(0);pressC.b.fill(0);pressC.cur=0;
			smooth(grid.curBuf(pressC),divC,massC,flagsC,texWc,texHc,coarseIters);
			prolongAndCorrect();
			smooth(p,refs.divsrc,refs.massCur(),refs.flags,texW,texH,postSmooth);
		}
		// the true poisson residual of the current fine pressure field, operator units
		function residualNorm(){
			return computeResidual(grid.curBuf(refs.press),refs.divsrc,refs.massCur(),
				refs.flags,texW,texH,cfg.rhoVac,resFine);
		}
		return {vcycle,residualNorm,pressC,divC,massC,flagsC,Nc,texWc,texHc,S};
	}
	Fluid.core.multigrid={create,cellBlocked,computeResidual};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.multigrid;
})(globalThis);
