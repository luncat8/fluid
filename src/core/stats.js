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
	const STATS_LEN=32;
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
	function packStatsArray(arr,tot,motion){
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
			heatInE:0,heatOutE:0
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
	Fluid.core.stats={STATS_LEN,IDX_TOT_M,IDX_TOT_DR,IDX_TOT_DG,IDX_TOT_DB,IDX_TOT_E,IDX_MIN_M,IDX_MIN_DR,IDX_MIN_DG,IDX_MIN_DB,IDX_MIN_E,IDX_MAX_M,IDX_CNT_NEG,IDX_CNT_OVER,IDX_MAX_VEL,IDX_P_MIN,IDX_P_MAX,IDX_MAX_DIV,IDX_HEAT_IN,IDX_HEAT_OUT,IDX_T_MIN,IDX_T_MAX,IDX_T_MEAN,IDX_KINETIC,STATS_FIELDS,totalsFromFields,motionTotals,thermalTotals,packStatsArray,driftRel,createLedger,expectedTotal,writeExpectedTotals,formatDrift};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.stats;
})(globalThis);
