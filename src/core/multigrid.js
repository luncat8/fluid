(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const grid=Fluid.core.grid;

	// two-level V-cycle for the variable-coefficient pressure solve.
	// the fine red-black GS is an excellent smoother but cannot reduce the
	// long-wavelength pressure error: on a closed box the slowest mode has
	// amplification 1-O(1/N^2) per sweep, so 60 sweeps at N=512 leave a
	// divergence floor of 2-3 cells/s and the warm-started null space drifts
	// (measured: a 4 s dam run at N=256 climbs to maxM 3.9, maxDiv 190).
	// the coarse level solves exactly those modes.
	//
	// scaling: the fine operator is A p = sum_faces w (p_n - p) with
	// w = 1/rho_f (harmonic mean). a coarse cell covers S x S fine cells, so
	// the same physical pressure curvature gives S^2 times the raw stencil:
	// A_c = (1/S^2) * raw. we keep the raw stencil (mass-based w, same code as
	// the fine smoother) and feed it the *sum* of the S^2 fine residuals,
	// which is S^2 times their average. with the average the correction is
	// S^2 times too small and the cycle is worse than plain GS (measured
	// N=128: 4 cycles, sum 0.18 vs average 12.7 residual).
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
		function faceW(massArr,texW_,x,y,nx,ny,flagsArr){
			if(nx<1||nx>texW_-2||ny<1||ny>texW_-2)return 0;
			if(flagsArr[(nx+ny*texW_)*4]>0)return 0;
			const ra=Math.max(massArr[(x+y*texW_)*4+3],cfg.rhoVac);
			const rb=Math.max(massArr[(nx+ny*texW_)*4+3],cfg.rhoVac);
			return 1/(2*ra*rb/(ra+rb));
		}
		function coarsen(){
			const massFine=refs.massCur(),flagsFine=refs.flags;
			for(let yc=1;yc<texHc-1;yc++){
				for(let xc=1;xc<texWc-1;xc++){
					let sum=0,anyWall=0;
					for(let oy=0;oy<S;oy++)for(let ox=0;ox<S;ox++){
						const fi=S*(xc-1)+1+ox+(S*(yc-1)+1+oy)*texW;
						sum+=massFine[fi*4+3];
						if(flagsFine[fi*4]>0)anyWall=1;
					}
					const ic=(xc+yc*texWc)*4;
					massC[ic+3]=sum/(S*S);
					flagsC[ic]=anyWall?255:0;
				}
			}
		}
		// one red-black gauss-seidel sweep set, in place, on either level
		function smooth(pressArr,divArr,massArr,flagsArr,texW_,texH_,iters){
			for(let k=0;k<iters;k++){
				for(let color=0;color<2;color++){
					for(let y=1;y<texH_-1;y++){
						for(let x=1;x<texW_-1;x++){
							if(((x+y)&1)!==color)continue;
							const i=x+y*texW_;
							if(flagsArr[i*4]>0){pressArr[i]=0;continue;}
							let wsum=0,sum=0,w;
							w=faceW(massArr,texW_,x,y,x-1,y,flagsArr);
							if(w>0){wsum+=w;sum+=w*pressArr[i-1];}
							w=faceW(massArr,texW_,x,y,x+1,y,flagsArr);
							if(w>0){wsum+=w;sum+=w*pressArr[i+1];}
							w=faceW(massArr,texW_,x,y,x,y-1,flagsArr);
							if(w>0){wsum+=w;sum+=w*pressArr[i-texW_];}
							w=faceW(massArr,texW_,x,y,x,y+1,flagsArr);
							if(w>0){wsum+=w;sum+=w*pressArr[i+texW_];}
							pressArr[i]=wsum>0?(sum-divArr[i])/wsum:0;
						}
					}
				}
			}
		}
		function computeResidual(pressArr,divArr,massArr,flagsArr,texW_,texH_,out){
			for(let y=1;y<texH_-1;y++){
				for(let x=1;x<texW_-1;x++){
					const i=x+y*texW_;
					if(flagsArr[i*4]>0){out[i]=0;continue;}
					let wsum=0,sum=0,w;
					w=faceW(massArr,texW_,x,y,x-1,y,flagsArr);
					if(w>0){wsum+=w;sum+=w*pressArr[i-1];}
					w=faceW(massArr,texW_,x,y,x+1,y,flagsArr);
					if(w>0){wsum+=w;sum+=w*pressArr[i+1];}
					w=faceW(massArr,texW_,x,y,x,y-1,flagsArr);
					if(w>0){wsum+=w;sum+=w*pressArr[i-texW_];}
					w=faceW(massArr,texW_,x,y,x,y+1,flagsArr);
					if(w>0){wsum+=w;sum+=w*pressArr[i+texW_];}
					out[i]=divArr[i]-(sum-wsum*pressArr[i]);
				}
			}
		}
		// sum of the S^2 fine residuals per coarse cell (see scaling note above)
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
		// bilinear interpolation of the coarse correction, fine cell centers map
		// to (x-0.5)/S + 0.5 in coarse index space
		function prolongAndCorrect(){
			const pressFine=grid.curBuf(refs.press);
			const coarse=grid.curBuf(pressC);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const xf=(x-0.5)/S+0.5, yf=(y-0.5)/S+0.5;
					const xc=Math.floor(xf), yc=Math.floor(yf);
					const fx=xf-xc, fy=yf-yc;
					const x0=Math.max(1,Math.min(Nc,xc)), x1=Math.max(1,Math.min(Nc,xc+1));
					const y0=Math.max(1,Math.min(Nc,yc)), y1=Math.max(1,Math.min(Nc,yc+1));
					const v00=coarse[x0+y0*texWc], v10=coarse[x1+y0*texWc];
					const v01=coarse[x0+y1*texWc], v11=coarse[x1+y1*texWc];
					const v0=v00+(v10-v00)*fx, v1=v01+(v11-v01)*fx;
					corrFine[x+y*texW]=v0+(v1-v0)*fy;
				}
			}
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					if(refs.flags[(x+y*texW)*4]>0)continue;
					pressFine[x+y*texW]+=corrFine[x+y*texW];
				}
			}
		}
		function vcycle(preSmooth,coarseIters,postSmooth){
			const p=grid.curBuf(refs.press);
			coarsen();
			smooth(p,refs.divsrc,refs.massCur(),refs.flags,texW,texH,preSmooth);
			computeResidual(p,refs.divsrc,refs.massCur(),refs.flags,texW,texH,resFine);
			restrictResidual();
			pressC.a.fill(0);pressC.b.fill(0);pressC.cur=0;
			smooth(grid.curBuf(pressC),divC,massC,flagsC,texWc,texHc,coarseIters);
			prolongAndCorrect();
			smooth(p,refs.divsrc,refs.massCur(),refs.flags,texW,texH,postSmooth);
		}
		function residualNorm(){
			const p=grid.curBuf(refs.press);
			computeResidual(p,refs.divsrc,refs.massCur(),refs.flags,texW,texH,resFine);
			let maxA=0;
			for(let y=1;y<texH-1;y++)for(let x=1;x<texW-1;x++){
				const v=Math.abs(resFine[x+y*texW]);
				if(v>maxA)maxA=v;
			}
			return maxA;
		}
		return {vcycle,residualNorm,pressC,divC,massC,flagsC,Nc,texWc,texHc,S};
	}
	Fluid.core.multigrid={create};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.multigrid;
})(globalThis);
