(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const grid=Fluid.core.grid;
	// 2-level V-cycle for the variable-coefficient pressure Poisson.
	// This is the CPU prototype for 0.6.0 t4. It reuses the same faceW
	// as src/core/solver.js (harmonic mean density) so the residual is
	// computed with the identical operator that the fine RB uses. Coarse
	// solves are RB Gauss-Seidel on the restricted residual. The cycle
	// is intentionally simple: 2 pre, coarse 16, 2 post — the point is to
	// measure residual reduction per unit time versus plain RB, not to
	// tune a production solver yet.
	function create(cfg,texW,texH,refs){
		const N=cfg.N;
		const Nc=Math.floor(N/2);
		const texWc=Nc+2, texHc=Nc+2;
		const pressC=grid.allocPingPong(texWc,texHc,4,Float32Array);
		const divC=new Float32Array(texWc*texHc);
		const massC=new Float32Array(texWc*texHc*4);
		const flagsC=new Uint8Array(texWc*texHc*4);
		const resFine=new Float32Array(texW*texH);
		const corrFine=new Float32Array(texW*texH);
		function isWallFine(x,y){
			if(x<1||x>texW-2||y<1||y>texH-2) return true;
			return refs.flags[(x+y*texW)*4]>0;
		}
		function isWallCoarse(x,y){
			if(x<1||x>texWc-2||y<1||y>texHc-2) return true;
			return flagsC[(x+y*texWc)*4]>0;
		}
		function faceW(massArr,texW_,x,y,nx,ny,flagsArr){
			if(nx<1||nx>texW_-2||ny<1||ny>texW_-2) return 0;
			if(flagsArr[(nx+ny*texW_)*4]>0) return 0;
			const ra=Math.max(massArr[(x+y*texW_)*4+3],cfg.rhoVac);
			const rb=Math.max(massArr[(nx+ny*texW_)*4+3],cfg.rhoVac);
			return 1/(2*ra*rb/(ra+rb));
		}
		function restrictMassAndFlags(){
			// average 2x2 fine blocks into coarse; flags = any wall in block
			for(let yc=1; yc<texHc-1; yc++){
				for(let xc=1; xc<texWc-1; xc++){
					let sum=0, anyWall=0;
					for(let oy=0; oy<2; oy++) for(let ox=0; ox<2; ox++){
						const xf=2*xc-1+ox, yf=2*yc-1+oy;
						const i=(xf+yf*texW)*4;
						sum+=refs.massCur()[i+3];
						if(refs.flags[i]>0) anyWall=1;
					}
					const ic=(xc+yc*texWc)*4;
					massC[ic+3]=sum/4;
					flagsC[ic]=anyWall?255:0;
				}
			}
			// ghost ring remains 0
		}
		function smooth(pressPP,divArr,massArr,flagsArr,texW_,texH_,iters){
			const ms=massArr;
			for(let k=0;k<iters;k++){
				for(let color=0;color<2;color++){
					const src=grid.curBuf(pressPP), dst=grid.nxtBuf(pressPP);
					for(let y=1;y<texH_-1;y++){
						for(let x=1;x<texW_-1;x++){
							const i=x+y*texW_;
							if(((x+y)&1)!==color){ dst[i]=src[i]; continue; }
							if(flagsArr[(x+y*texW_)*4]>0){ dst[i]=0; continue; }
							let wsum=0,sum=0;
							let w=faceW(massArr,texW_,x,y,x-1,y,flagsArr);
							if(w>0){ wsum+=w; sum+=w*src[i-1]; }
							w=faceW(massArr,texW_,x,y,x+1,y,flagsArr);
							if(w>0){ wsum+=w; sum+=w*src[i+1]; }
							w=faceW(massArr,texW_,x,y,x,y-1,flagsArr);
							if(w>0){ wsum+=w; sum+=w*src[i-texW_]; }
							w=faceW(massArr,texW_,x,y,x,y+1,flagsArr);
							if(w>0){ wsum+=w; sum+=w*src[i+texW_]; }
							const b=divArr[i];
							dst[i]=wsum>0?(sum-b)/wsum:0;
						}
					}
					grid.swap(pressPP);
				}
			}
		}
		function computeResidual(pressArr,divArr,massArr,flagsArr,texW_,texH_,out){
			// r = b - A p, with A = sum w(p_nb)-wsum p_c
			for(let y=1;y<texH_-1;y++){
				for(let x=1;x<texW_-1;x++){
					const i=x+y*texW_;
					if(flagsArr[(x+y*texW_)*4]>0){ out[i]=0; continue; }
					let wsum=0,sum=0;
					let w=faceW(massArr,texW_,x,y,x-1,y,flagsArr);
					if(w>0){ wsum+=w; sum+=w*pressArr[i-1]; }
					w=faceW(massArr,texW_,x,y,x+1,y,flagsArr);
					if(w>0){ wsum+=w; sum+=w*pressArr[i+1]; }
					w=faceW(massArr,texW_,x,y,x,y-1,flagsArr);
					if(w>0){ wsum+=w; sum+=w*pressArr[i-texW_]; }
					w=faceW(massArr,texW_,x,y,x,y+1,flagsArr);
					if(w>0){ wsum+=w; sum+=w*pressArr[i+texW_]; }
					const Ap=sum - wsum*pressArr[i];
					out[i]=divArr[i] - Ap;
				}
			}
		}
		function restrictResidual(){
			for(let yc=1; yc<texHc-1; yc++){
				for(let xc=1; xc<texWc-1; xc++){
					let sum=0;
					for(let oy=0; oy<2; oy++) for(let ox=0; ox<2; ox++){
						const xf=2*xc-1+ox, yf=2*yc-1+oy;
						sum+=resFine[xf+yf*texW];
					}
					divC[xc+yc*texWc]=sum/4;
				}
			}
			// ghost zero
			for(let x=0;x<texWc;x++){ divC[x]=0; divC[x+(texHc-1)*texWc]=0; }
			for(let y=0;y<texHc;y++){ divC[y*texWc]=0; divC[texWc-1+y*texWc]=0; }
		}
		function prolongAndCorrect(){
			const pressFine=grid.curBuf(refs.press);
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					const xc=Math.floor((x+1)/2), yc=Math.floor((y+1)/2);
					const e=grid.curBuf(pressC)[xc+yc*texWc];
					corrFine[x+y*texW]=e;
				}
			}
			for(let y=1;y<texH-1;y++){
				for(let x=1;x<texW-1;x++){
					if(isWallFine(x,y)) continue;
					pressFine[x+y*texW]+=corrFine[x+y*texW];
				}
			}
		}
		function vcycle(preSmooth,coarseIters,postSmooth){
			// fine pre-smooth (operates on refs.press using refs.divsrc and refs.mass)
			restrictMassAndFlags();
			smooth(refs.press,refs.divsrc,refs.massCur(),refs.flags,texW,texH,preSmooth);
			computeResidual(grid.curBuf(refs.press),refs.divsrc,refs.massCur(),refs.flags,texW,texH,resFine);
			restrictResidual();
			// coarse solve: zero initial error
			for(let i=0;i<pressC.a.length;i++) pressC.a[i]=0, pressC.b[i]=0;
			pressC.cur=0;
			smooth(pressC,divC,massC,flagsC,texWc,texHc,coarseIters);
			prolongAndCorrect();
			smooth(refs.press,refs.divsrc,refs.massCur(),refs.flags,texW,texH,postSmooth);
		}
		function residualNorm(){
			const p=grid.curBuf(refs.press);
			computeResidual(p,refs.divsrc,refs.massCur(),refs.flags,texW,texH,resFine);
			let maxA=0;
			for(let y=1;y<texH-1;y++) for(let x=1;x<texW-1;x++){
				const v=Math.abs(resFine[x+y*texW]);
				if(v>maxA) maxA=v;
			}
			return maxA;
		}
		return {vcycle,residualNorm,pressC,divC,massC,flagsC,Nc,texWc,texHc};
	}
	Fluid.core.multigrid={create};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.multigrid;
})(globalThis);
