(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	const STATS_FIELDS=["m","dr","dg","db","E"];
	const IDX_TOT_M=0,IDX_TOT_DR=1,IDX_TOT_DG=2,IDX_TOT_DB=3,IDX_TOT_E=4;
	const IDX_MIN_M=5,IDX_MIN_DR=6,IDX_MIN_DG=7,IDX_MIN_DB=8,IDX_MIN_E=9;
	const IDX_MAX_M=10,IDX_CNT_NEG=11,IDX_CNT_OVER=12;
	const IDX_MAX_VEL=13,IDX_P_MIN=14,IDX_P_MAX=15,IDX_MAX_DIV=16;
	// cumulative thermal source sums carried in therm.b / therm.a (0.4.0), then
	// temperature range and kinetic energy reconstructed from the face velocities
	const IDX_HEAT_IN=17,IDX_HEAT_OUT=18,IDX_T_MIN=19,IDX_T_MAX=20,IDX_T_MEAN=21,IDX_KINETIC=22;
	// 0.5.0 phase histogram (6 caloric branches), S saturation watchdog, and
	// cumulative latent/frozen-momentum diagnostics in slots 23..31
	const IDX_PH_VAC=23,IDX_PH_SOL=24,IDX_PH_MELT=25,IDX_PH_LIQ=26,IDX_PH_BOIL=27,IDX_PH_GAS=28;
	const IDX_S_SAT=29,IDX_LATENT=30,IDX_FROZEN_MOM=31;
	// 0.6.0: solve-time poisson residual (max|b - L p| and max|b|) in 32/33.
	// the ratio is the plan 4.1 judge. unlike divAfter it does not floor at max|S|.
	// 34/35 are the same pressure read after transport and phase: coefficient drift,
	// not solver error. a boiling spike lives there.
	const IDX_POISSON_RES=32,IDX_POISSON_RHS=33;
	const IDX_POISSON_DRIFT=34,IDX_POISSON_DRIFT_RHS=35;
	// 0.7.0 t3: cumulative GPU emitter mass/dye/enthalpy injections and drain
	// removals, Kahan-accumulated in pool.emitterLedger and copied into 36..45
	// by statsReduceB alongside totals 0..4 so pollStats pairs them exactly.
	const IDX_EMIT_INJ_M=36,IDX_EMIT_INJ_DR=37,IDX_EMIT_INJ_DG=38,IDX_EMIT_INJ_DB=39,IDX_EMIT_INJ_E=40;
	const IDX_EMIT_REM_M=41,IDX_EMIT_REM_DR=42,IDX_EMIT_REM_DG=43,IDX_EMIT_REM_DB=44,IDX_EMIT_REM_E=45;
	const STATS_LEN=46;
	const PARTIAL_STRIDE=40;
	const EMIT_LEDGER_LEN=20;
	// negatives below -1e-9 are real; smaller is f32/FCT roundoff (≈1e-28 seen at N=512).
	// over threshold 1.001 catches any liquid overpack; severity is decided in the hud.
	const NEG_EPS=1e-9;
	const OVER_EPS=1.001;
	function totalsFromFields(mass,therm,texW,texH){
		let sumM=0,sumDr=0,sumDg=0,sumDb=0,sumE=0;
		let minM=Infinity,minDr=Infinity,minDg=Infinity,minDb=Infinity,minE=Infinity;
		let maxM=-Infinity;
		let cntNeg=0,cntOver=0;
		for(let y=1;y<texH-1;y++){
			for(let x=1;x<texW-1;x++){
				let i=(x+y*texW)*4;
				let dr=mass[i],dg=mass[i+1],db=mass[i+2],m=mass[i+3];
				let E=therm[i];
				sumM+=m;sumDr+=dr;sumDg+=dg;sumDb+=db;sumE+=E;
				if(m<minM)minM=m;
				if(dr<minDr)minDr=dr;
				if(dg<minDg)minDg=dg;
				if(db<minDb)minDb=db;
				if(E<minE)minE=E;
				if(m>maxM)maxM=m;
				if(m<-NEG_EPS||dr<-NEG_EPS||dg<-NEG_EPS||db<-NEG_EPS||E<-NEG_EPS)cntNeg++;
				if(m>OVER_EPS)cntOver++;
			}
		}
		if(minM===Infinity)minM=0;
		if(minDr===Infinity)minDr=0;
		if(minDg===Infinity)minDg=0;
		if(minDb===Infinity)minDb=0;
		if(minE===Infinity)minE=0;
		if(maxM===-Infinity)maxM=0;
		return {sumM,sumDr,sumDg,sumDb,sumE,minM,minDr,minDg,minDb,minE,maxM,cntNeg,cntOver};
	}
	function motionTotals(vel,press,texW,texH,dt){
		// vel: rg per texel, r = u left face, g = v bottom face; press: r per texel
		let maxVel=0,pMin=Infinity,pMax=-Infinity,maxDiv=0;
		let invDt=1/Math.max(dt,1e-9);
		for(let y=1;y<texH-1;y++){
			for(let x=1;x<texW-1;x++){
				let i=(x+y*texW)*2;
				let iR=(x+1+y*texW)*2;
				let iT=(x+(y+1)*texW)*2;
				let uL=vel[i],uR=vel[iR],vB=vel[i+1],vT=vel[iT+1];
				let f=Math.max(Math.abs(uL),Math.abs(uR),Math.abs(vB),Math.abs(vT));
				if(f>maxVel)maxVel=f;
				let div=Math.abs(uR-uL+vT-vB)*invDt;
				if(div>maxDiv)maxDiv=div;
				let p=press[x+y*texW];
				if(p<pMin)pMin=p;
				if(p>pMax)pMax=p;
			}
		}
		if(pMin===Infinity)pMin=0;
		if(pMax===-Infinity)pMax=0;
		return {maxVel,pMin,pMax,maxDiv};
	}
	function packStatsArray(arr,tot,motion,poisson){
		arr[IDX_TOT_M]=tot.sumM;
		arr[IDX_TOT_DR]=tot.sumDr;
		arr[IDX_TOT_DG]=tot.sumDg;
		arr[IDX_TOT_DB]=tot.sumDb;
		arr[IDX_TOT_E]=tot.sumE;
		arr[IDX_MIN_M]=tot.minM;
		arr[IDX_MIN_DR]=tot.minDr;
		arr[IDX_MIN_DG]=tot.minDg;
		arr[IDX_MIN_DB]=tot.minDb;
		arr[IDX_MIN_E]=tot.minE;
		arr[IDX_MAX_M]=tot.maxM;
		arr[IDX_CNT_NEG]=tot.cntNeg;
		arr[IDX_CNT_OVER]=tot.cntOver;
		if(motion){
			arr[IDX_MAX_VEL]=motion.maxVel;
			arr[IDX_P_MIN]=motion.pMin;
			arr[IDX_P_MAX]=motion.pMax;
			arr[IDX_MAX_DIV]=motion.maxDiv;
		}
		if(poisson){
			arr[IDX_POISSON_RES]=poisson.resMax;
			arr[IDX_POISSON_RHS]=poisson.rhsMax;
		}
	}
	function driftRel(actual,expected){
		let denom=Math.max(Math.abs(expected),1e-12);
		return (actual-expected)/denom;
	}
	// temperature range over cells with substance, and kinetic energy from the
	// mac faces: cell velocity is the average of its two faces per axis
	// (0.0.0/0.4.0: the same reconstruction on cpu and gpu, closure depends on it)
	function thermalTotals(mass,therm,vel,texW,texH,mEps,cLiq,TAmb){
		let TMin=Infinity,TMax=-Infinity,TSum=0,TCount=0,kinetic=0;
		for(let y=1;y<texH-1;y++){
			for(let x=1;x<texW-1;x++){
				const i=(x+y*texW)*4,iv=(x+y*texW)*2;
				const m=mass[i+3];
				if(m<mEps)continue;
				const T=therm[i+1];
				if(T<TMin)TMin=T;
				if(T>TMax)TMax=T;
				TSum+=T;TCount++;
				const uc=0.5*(vel[iv]+vel[iv+2]),vc=0.5*(vel[iv+1]+vel[iv+texW*2+1]);
				kinetic+=0.5*m*(uc*uc+vc*vc);
			}
		}
		if(TCount===0){TMin=TAmb;TMax=TAmb;}
		return {TMin,TMax,TMean:TCount?TSum/TCount:TAmb,kinetic};
	}
	function createLedger(initial){
		return {
			initialM:initial.sumM,initialDr:initial.sumDr,initialDg:initial.sumDg,initialDb:initial.sumDb,initialE:initial.sumE,
			injectedM:0,injectedDr:0,injectedDg:0,injectedDb:0,injectedE:0,
			removedM:0,removedDr:0,removedDg:0,removedDb:0,removedE:0,
			boundaryM:0,boundaryDr:0,boundaryDg:0,boundaryDb:0,boundaryE:0,
			heatInE:0,heatOutE:0,
			// 0.5.0 diagnostics: m*L traversed through latent plateaus, and the
			// momentum frozen cells lost (not conserved, explains "where the splash went")
			latentChargedE:0,frozenMomentum:0
		};
	}
	function expectedTotal(ledger,kind){
		if(kind==="m")return ledger.initialM+ledger.injectedM-ledger.removedM-ledger.boundaryM;
		if(kind==="dr")return ledger.initialDr+ledger.injectedDr-ledger.removedDr-ledger.boundaryDr;
		if(kind==="dg")return ledger.initialDg+ledger.injectedDg-ledger.removedDg-ledger.boundaryDg;
		if(kind==="db")return ledger.initialDb+ledger.injectedDb-ledger.removedDb-ledger.boundaryDb;
		if(kind==="E")return ledger.initialE+ledger.injectedE-ledger.removedE-ledger.boundaryE+(ledger.heatInE||0)-(ledger.heatOutE||0);
		return 0;
	}
	function writeExpectedTotals(out,ledger){
		out.m=expectedTotal(ledger,"m");
		out.dr=expectedTotal(ledger,"dr");
		out.dg=expectedTotal(ledger,"dg");
		out.db=expectedTotal(ledger,"db");
		out.E=expectedTotal(ledger,"E");
		return out;
	}
	function formatDrift(v){
		let a=Math.abs(v);
		if(a<1e-6)return "ok";
		if(a<1e-3)return "warn";
		return "bad";
	}
	Fluid.core.stats={STATS_LEN,PARTIAL_STRIDE,EMIT_LEDGER_LEN,IDX_TOT_M,IDX_TOT_DR,IDX_TOT_DG,IDX_TOT_DB,IDX_TOT_E,IDX_MIN_M,IDX_MIN_DR,IDX_MIN_DG,IDX_MIN_DB,IDX_MIN_E,IDX_MAX_M,IDX_CNT_NEG,IDX_CNT_OVER,IDX_MAX_VEL,IDX_P_MIN,IDX_P_MAX,IDX_MAX_DIV,IDX_HEAT_IN,IDX_HEAT_OUT,IDX_T_MIN,IDX_T_MAX,IDX_T_MEAN,IDX_KINETIC,IDX_PH_VAC,IDX_PH_SOL,IDX_PH_MELT,IDX_PH_LIQ,IDX_PH_BOIL,IDX_PH_GAS,IDX_S_SAT,IDX_LATENT,IDX_FROZEN_MOM,IDX_POISSON_RES,IDX_POISSON_RHS,IDX_POISSON_DRIFT,IDX_POISSON_DRIFT_RHS,IDX_EMIT_INJ_M,IDX_EMIT_INJ_DR,IDX_EMIT_INJ_DG,IDX_EMIT_INJ_DB,IDX_EMIT_INJ_E,IDX_EMIT_REM_M,IDX_EMIT_REM_DR,IDX_EMIT_REM_DG,IDX_EMIT_REM_DB,IDX_EMIT_REM_E,STATS_FIELDS,totalsFromFields,motionTotals,thermalTotals,packStatsArray,driftRel,createLedger,expectedTotal,writeExpectedTotals,formatDrift};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.stats;
})(globalThis);
