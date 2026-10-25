(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};

	// Versioned little-endian binary snapshot (0.7.0 t6):
	//   256 B header | 52 cfg/policy keys (f64) | 96-float CPU state (f32) |
	//   24-key ledger (f64) | 3 tables (probes, emitters+ledger, colliders, buoy) |
	//   9 tight field blocks (mass, therm, vel, press, flags, flagsMirror, sfield, accum, mgPress)
	const MAGIC=0x44495546; // "FUID" little-endian
	const VERSION=1;
	const HEADER_BYTES=256;
	const STATE_LEN=96;
	const PROBE_MAX=32,PROBE_LEN=528,PROBE_META_LEN=36;
	const EMIT_MAX=32,EMIT_LEN=528,EMIT_SLOT_FIELDS=16,EMIT_META_LEN=EMIT_MAX*EMIT_SLOT_FIELDS,EMIT_LEDGER_LEN=20;
	const COLL_MAX=32,COLL_LEN=528,COLL_SLOT_FIELDS=10,COLL_META_LEN=COLL_MAX*COLL_SLOT_FIELDS;
	const BUOY_SAMPLE_PAD=8;

	// 52 physics and step-policy keys that determine trajectory. View-only keys
	// (viewMode, overlay, colorGain, TMin/TMax/PMin/PMax/VMax, gpuTiming, paint*)
	// do not alter the solver step and are not part of the replay contract.
	const CFG_KEYS=[
		"frameDt","cflTarget","substepMax","substepsAuto","substepsFixed","vMax",
		"gravityX","gravityY","rhoRef","beta","TRef","TAmb","mouseR",
		"pressureIters","fctLimiter","viscosity","kSol","kLiq","kGas","TEps",
		"heatPower","coolTop","wallCool","heatBand","thermalSources","heatTarget","TCeil","convBotT","convTopT",
		"SMax","tauPhase","cSol","cLiq","cGas","Lf","Lv","TMelt","TBoil","rhoLiq","rhoGas0","rhoVac","mEps",
		"advectMac","macLimiter","fusePasses","vorticityConfinement",
		"useMultigrid","mgPre","mgPost","mgCoarsen","mgCycles","mgCoarseIters"
	];

	const LEDGER_KEYS=[
		"initialM","initialDr","initialDg","initialDb","initialE",
		"injectedM","injectedDr","injectedDg","injectedDb","injectedE",
		"removedM","removedDr","removedDg","removedDb","removedE",
		"boundaryM","boundaryDr","boundaryDg","boundaryDb","boundaryE",
		"heatInE","heatOutE","latentChargedE","frozenMomentum"
	];

	const FIELD_NAMES=["mass","therm","vel","press","flags","flagsMirror","sfield","accum","mgPress"];

	// CPU state indices inside the 96-float block
	const ST_FRAME=0,ST_TIME=1,ST_DT=2,ST_SUBSTEPS=3,ST_VCLAMP=4,ST_RHO_MEAN=5,ST_RHO_BASE=6;
	const ST_PMIN=7,ST_PMAX=8,ST_PSPAN=9,ST_PAINT_INJ_M=10,ST_HEAT_NET=11,ST_ACCUM=12,ST_SIM_SUBSTEPS=13;
	const ST_MOUSE_X=14,ST_MOUSE_Y=15,ST_MOUSE_STR=16,ST_MOUSE_VX=17,ST_MOUSE_VY=18;
	const ST_EMIT_INJ=19,ST_EMIT_REM=24,ST_EMIT_NET=29;
	const ST_STATS_FRAME=34,ST_COLL_GPU_MASK=35;
	const ST_TOT=36,ST_EXP=41,ST_MIN=46,ST_MAX_M=51,ST_CNT_NEG=52,ST_CNT_OVER=53;
	const ST_MOT=54,ST_THERM=58,ST_PHASE_COUNTS=64,ST_DIAG=70;
	const ST_BUOY=77;

	function align256(n){
		return Math.ceil(n/256)*256;
	}
	function mgDims(N,mgCoarsen){
		let mgS=Math.max(2,Math.round(mgCoarsen||4));
		while(mgS>2&&Math.floor(N/mgS)<4)mgS=Math.floor(mgS/2);
		const mgNc=Math.floor(N/mgS);
		return {mgS,mgNc,mgTexW:mgNc+2,mgTexH:mgNc+2};
	}
	function numVal(v){
		if(typeof v==="boolean")return v?1:0;
		const n=Number(v);
		return Number.isFinite(n)?n:0;
	}
	function layout(N,mgCoarsen){
		const n=Math.round(N)||64;
		const texW=n+2,texH=n+2;
		const mg=mgDims(n,mgCoarsen);
		const cells=texW*texH;
		const mgCells=mg.mgTexW*mg.mgTexH;
		let off=HEADER_BYTES;
		const cfgOff=off;off+=CFG_KEYS.length*8;
		const stateOff=off;off+=STATE_LEN*8;
		const ledgerOff=off;off+=LEDGER_KEYS.length*8;
		const probeUpOff=off;off+=PROBE_LEN*4;
		const probeRawOff=off;off+=PROBE_LEN*4;
		const probeMetaOff=off;off+=PROBE_META_LEN*4;
		const emitUpOff=off;off+=EMIT_LEN*4;
		const emitLedgerOff=off;off+=EMIT_LEDGER_LEN*4;
		const collUpOff=off;off+=COLL_LEN*4;
		const buoySampleOff=off;off+=BUOY_SAMPLE_PAD*4;
		// align to 8 bytes before f64 slot metadata arrays
		if(off&7)off+=4;
		const emitMetaOff=off;off+=EMIT_META_LEN*8;
		const collMetaOff=off;off+=COLL_META_LEN*8;
		const massBytes=cells*16;
		const thermBytes=cells*16;
		const velBytes=cells*8;
		const pressBytes=cells*4;
		const flagsBytes=cells*4;
		const flagsMirrorBytes=cells*4;
		const sfieldBytes=cells*16;
		const accumBytes=cells*8;
		const mgPressBytes=mgCells*4;
		const massOff=off;off+=massBytes;
		const thermOff=off;off+=thermBytes;
		const velOff=off;off+=velBytes;
		const pressOff=off;off+=pressBytes;
		const flagsOff=off;off+=flagsBytes;
		const flagsMirrorOff=off;off+=flagsMirrorBytes;
		const sfieldOff=off;off+=sfieldBytes;
		const accumOff=off;off+=accumBytes;
		const mgPressOff=off;off+=mgPressBytes;
		return {
			N:n,texW,texH,
			mgS:mg.mgS,mgNc:mg.mgNc,mgTexW:mg.mgTexW,mgTexH:mg.mgTexH,
			cfgOff,stateOff,ledgerOff,
			probeUpOff,probeRawOff,probeMetaOff,
			emitUpOff,emitLedgerOff,emitMetaOff,
			collUpOff,collMetaOff,buoySampleOff,
			fields:{
				mass:{off:massOff,bytes:massBytes,len:cells*4,type:"f32"},
				therm:{off:thermOff,bytes:thermBytes,len:cells*4,type:"f32"},
				vel:{off:velOff,bytes:velBytes,len:cells*2,type:"f32"},
				press:{off:pressOff,bytes:pressBytes,len:cells,type:"f32"},
				flags:{off:flagsOff,bytes:flagsBytes,len:cells*4,type:"u8"},
				flagsMirror:{off:flagsMirrorOff,bytes:flagsMirrorBytes,len:cells*4,type:"u8"},
				sfield:{off:sfieldOff,bytes:sfieldBytes,len:cells*4,type:"f32"},
				accum:{off:accumOff,bytes:accumBytes,len:cells*2,type:"f32"},
				mgPress:{off:mgPressOff,bytes:mgPressBytes,len:mgCells,type:"f32"}
			},
			totalBytes:off
		};
	}
	function byteLength(N,mgCoarsen){
		return layout(N,mgCoarsen).totalBytes;
	}
	// 256-aligned layout for single-submit GPU texture + emitterLedger capture
	function stagingLayout(N,mgCoarsen){
		const lay=layout(N,mgCoarsen);
		const specs=[
			{name:"mass",w:lay.texW,h:lay.texH,bpp:16},
			{name:"therm",w:lay.texW,h:lay.texH,bpp:16},
			{name:"vel",w:lay.texW,h:lay.texH,bpp:8},
			{name:"press",w:lay.texW,h:lay.texH,bpp:4},
			{name:"flags",w:lay.texW,h:lay.texH,bpp:4},
			{name:"sfield",w:lay.texW,h:lay.texH,bpp:16},
			{name:"accum",w:lay.texW,h:lay.texH,bpp:8},
			{name:"mgPress",w:lay.mgTexW,h:lay.mgTexH,bpp:4}
		];
		let off=0;
		const tex={};
		const list=[];
		for(let i=0;i<specs.length;i++){
			const s=specs[i];
			const rowBytes=s.w*s.bpp;
			const bytesPerRow=align256(rowBytes);
			const byteLen=bytesPerRow*s.h;
			const entry={name:s.name,w:s.w,h:s.h,bpp:s.bpp,rowBytes,bytesPerRow,offset:off,byteLength:byteLen};
			tex[s.name]=entry;
			list.push(entry);
			off+=align256(byteLen);
		}
		const emitterLedgerOffset=off;
		const emitterLedgerBytes=EMIT_LEDGER_LEN*4;
		off+=align256(emitterLedgerBytes);
		return {tex,list,emitterLedgerOffset,emitterLedgerBytes,totalBytes:off};
	}
	// Snapshot cfg tunables and runtime values synchronously before an async readback
	// so a slider moved while mapAsync is in flight cannot mislabel the captured fields.
	function cfgState(cfg){
		const values=new Float64Array(CFG_KEYS.length);
		for(let i=0;i<CFG_KEYS.length;i++)values[i]=numVal(cfg[CFG_KEYS[i]]);
		const scenarioNames=Fluid.core.config.scenarioNames;
		const scenarioIdx=Math.max(0,scenarioNames.indexOf(cfg.scenario));
		return {
			N:cfg.N,
			mgCoarsen:cfg.mgCoarsen,
			scenario:cfg.scenario,
			scenarioIdx,
			frame:cfg.frame||0,
			time:cfg.time||0,
			dt:cfg.dt||(1/60),
			frameDt:cfg.frameDt||(1/60),
			substeps:cfg.substeps||1,
			substepsAuto:cfg.substepsAuto?1:0,
			substepsFixed:Math.round(cfg.substepsFixed)||4,
			substepMax:Math.round(cfg.substepMax)||24,
			vClamp:cfg.vClamp||480,
			rhoMean:cfg.rhoMean||0,
			rhoBase:cfg.rhoBase!==undefined?cfg.rhoBase:(cfg.rhoMean||0),
			PMin:cfg.PMin!==undefined?cfg.PMin:-100,
			PMax:cfg.PMax!==undefined?cfg.PMax:100,
			mouseX:cfg.mouseX||0,
			mouseY:cfg.mouseY||0,
			mouseStrength:cfg.mouseStrength||0,
			mouseVX:cfg.mouseVX||0,
			mouseVY:cfg.mouseVY||0,
			values
		};
	}
	function resolveOutBuffer(out,neededBytes){
		if(out===undefined||out===null)return new ArrayBuffer(neededBytes);
		if(out instanceof ArrayBuffer){
			if(out.byteLength<neededBytes){
				throw new RangeError("serialize: out buffer too small (need "+neededBytes+" B, got "+out.byteLength+" B)");
			}
			return out;
		}
		if(ArrayBuffer.isView(out)){
			if(out.byteLength<neededBytes){
				throw new RangeError("serialize: out view too small (need "+neededBytes+" B, got "+out.byteLength+" B)");
			}
			if(out.byteOffset===0&&out.buffer.byteLength===out.byteLength)return out.buffer;
		}
		throw new TypeError("serialize: out must be an ArrayBuffer");
	}
	function resolveInView(buf){
		if(buf instanceof ArrayBuffer)return {buffer:buf,byteOffset:0,byteLength:buf.byteLength};
		if(ArrayBuffer.isView(buf))return {buffer:buf.buffer,byteOffset:buf.byteOffset,byteLength:buf.byteLength};
		throw new TypeError("deserialize: expected ArrayBuffer or TypedArray");
	}
	function writeHeader(buf,byteOffset,lay,cs,frame,time){
		const dv=new DataView(buf,byteOffset,HEADER_BYTES);
		for(let i=0;i<HEADER_BYTES;i+=4)dv.setUint32(i,0,true);
		dv.setUint32(0,MAGIC,true);
		dv.setUint32(4,VERSION,true);
		dv.setUint32(8,lay.N,true);
		dv.setUint32(12,lay.texW,true);
		dv.setUint32(16,lay.texH,true);
		dv.setUint32(20,lay.mgNc,true);
		dv.setUint32(24,lay.mgTexW,true);
		dv.setUint32(28,lay.mgTexH,true);
		dv.setUint32(32,lay.mgS,true);
		dv.setUint32(36,lay.totalBytes,true);
		dv.setUint32(40,cs.scenarioIdx>>>0,true);
		dv.setUint32(44,(frame>>>0),true);
		dv.setUint32(48,cs.substepsAuto>>>0,true);
		dv.setUint32(52,cs.substepsFixed>>>0,true);
		dv.setUint32(56,cs.substepMax>>>0,true);
		dv.setFloat64(64,cs.frameDt,true);
		dv.setFloat64(72,time,true);
		dv.setFloat64(80,cs.dt,true);
		dv.setUint32(88,CFG_KEYS.length,true);
		dv.setUint32(92,STATE_LEN,true);
		dv.setUint32(96,LEDGER_KEYS.length,true);
		dv.setUint32(100,FIELD_NAMES.length,true);
	}
	function packTables(buf,lay,probes,emitters,emitterLedger,colliders,buoy){
		if(probes){
			new Float32Array(buf,lay.probeUpOff,PROBE_LEN).set(probes.upload);
			new Float32Array(buf,lay.probeRawOff,PROBE_LEN).set(probes.raw);
			const meta=new Float32Array(buf,lay.probeMetaOff,PROBE_META_LEN);
			meta[0]=probes.live||0;
			meta[1]=probes.sampledFrame!==undefined?probes.sampledFrame:-1;
			meta[2]=probes.samples||0;
			meta[3]=0;
			for(let i=0;i<PROBE_MAX;i++){
				const s=probes.slots[i];
				meta[4+i]=(s&&s.bornFrame!==undefined)?s.bornFrame:-1;
			}
		}
		if(emitters){
			new Float32Array(buf,lay.emitUpOff,EMIT_LEN).set(emitters.upload);
			const em=new Float64Array(buf,lay.emitMetaOff,EMIT_META_LEN);
			for(let i=0;i<EMIT_MAX;i++){
				const s=emitters.slots[i];
				const b=i*EMIT_SLOT_FIELDS;
				if(!s){continue;}
				em[b]=s.active?1:0;
				em[b+1]=s.enabled?1:0;
				em[b+2]=s.kind||0;
				em[b+3]=s.x||0;
				em[b+4]=s.y||0;
				em[b+5]=s.radius||8;
				em[b+6]=s.value||0;
				em[b+7]=s.hasValue?1:0;
				em[b+8]=s.dr||0;
				em[b+9]=s.dg||0;
				em[b+10]=s.db||0;
				em[b+11]=s.dirMode||0;
				em[b+12]=s.dirX||0;
				em[b+13]=s.dirY||0;
				em[b+14]=s.angle||0;
				em[b+15]=s.target||0;
			}
		}
		if(emitterLedger){
			new Float32Array(buf,lay.emitLedgerOff,EMIT_LEDGER_LEN).set(emitterLedger);
		}
		if(colliders){
			new Float32Array(buf,lay.collUpOff,COLL_LEN).set(colliders.upload);
			const cm=new Float64Array(buf,lay.collMetaOff,COLL_META_LEN);
			for(let i=0;i<COLL_MAX;i++){
				const s=colliders.slots[i];
				const b=i*COLL_SLOT_FIELDS;
				if(!s){continue;}
				cm[b]=s.active?1:0;
				cm[b+1]=s.enabled?1:0;
				cm[b+2]=s.shape||0;
				cm[b+3]=s.x||0;
				cm[b+4]=s.y||0;
				cm[b+5]=s.radius||8;
				cm[b+6]=s.w||16;
				cm[b+7]=s.h||16;
				cm[b+8]=s.vx||0;
				cm[b+9]=s.vy||0;
			}
		}
		if(buoy&&buoy.sample){
			const bs=new Float32Array(buf,lay.buoySampleOff,BUOY_SAMPLE_PAD);
			bs.set(buoy.sample);
		}
	}
	function restoreTables(snap,cfg,probes,emitters,colliders,buoy){
		if(probes){
			probes.upload.set(snap.tables.probeUp);
			probes.raw.set(snap.tables.probeRaw);
			const meta=snap.tables.probeMeta;
			probes.live=Math.round(meta[0])||0;
			probes.sampledFrame=Math.round(meta[1]);
			probes.samples=Math.round(meta[2])||0;
			for(let i=0;i<PROBE_MAX;i++){
				const s=probes.slots[i];
				const b=16+i*16;
				s.active=probes.upload[b+15]>0.5;
				s.x=probes.upload[b];
				s.y=probes.upload[b+1];
				s.bornFrame=Math.round(meta[4+i]);
			}
		}
		if(emitters){
			emitters.upload.set(snap.tables.emitUp);
			const em=snap.tables.emitMeta;
			for(let i=0;i<EMIT_MAX;i++){
				const s=emitters.slots[i];
				const b=i*EMIT_SLOT_FIELDS;
				s.active=em[b]>0.5;
				s.enabled=em[b+1]>0.5;
				s.kind=Math.round(em[b+2])||0;
				s.x=em[b+3];
				s.y=em[b+4];
				s.radius=em[b+5];
				s.value=em[b+6];
				s.hasValue=em[b+7]>0.5;
				s.dr=em[b+8];
				s.dg=em[b+9];
				s.db=em[b+10];
				s.dirMode=Math.round(em[b+11])||0;
				s.dirX=em[b+12];
				s.dirY=em[b+13];
				s.angle=em[b+14];
				s.target=em[b+15];
			}
			emitters.live=Math.round(emitters.upload[0])||0;
			emitters.maxSlotPlus1=Math.round(emitters.upload[1])||0;
			emitters.momentumCount=Math.round(emitters.upload[2])||0;
			emitters.scalarCount=Math.round(emitters.upload[3])||0;
			emitters.massDrainCount=Math.round(emitters.upload[4])||0;
			emitters.dirty=false;
			if(Fluid.core.emitters)Fluid.core.emitters.syncConfig(emitters,cfg,false);
			emitters.dirty=false;
		}
		if(colliders){
			colliders.upload.set(snap.tables.collUp);
			const cm=snap.tables.collMeta;
			for(let i=0;i<COLL_MAX;i++){
				const s=colliders.slots[i];
				const b=i*COLL_SLOT_FIELDS;
				s.active=cm[b]>0.5;
				s.enabled=cm[b+1]>0.5;
				s.shape=Math.round(cm[b+2])||0;
				s.x=cm[b+3];
				s.y=cm[b+4];
				s.radius=cm[b+5];
				s.w=cm[b+6];
				s.h=cm[b+7];
				s.vx=cm[b+8];
				s.vy=cm[b+9];
			}
			colliders.live=Math.round(colliders.upload[0])||0;
			colliders.maxSlotPlus1=Math.round(colliders.upload[1])||0;
			colliders.movingCount=Math.round(colliders.upload[2])||0;
			colliders.peakSpeed=colliders.upload[3]||0;
			colliders.cfgN=cfg.N||64;
			colliders.dirty=false;
			colliders.maskDirty=false;
			colliders.hasGpuMask=snap.state[ST_COLL_GPU_MASK]>0.5;
		}
		if(buoy){
			const st=snap.state;
			const bs=buoy.state;
			bs.active=st[ST_BUOY]>0.5;
			bs.cx=st[ST_BUOY+1];bs.cy=st[ST_BUOY+2];
			bs.x0=Math.round(st[ST_BUOY+3]);bs.y0=Math.round(st[ST_BUOY+4]);
			bs.w=Math.round(st[ST_BUOY+5]);bs.h=Math.round(st[ST_BUOY+6]);
			bs.vx=st[ST_BUOY+7];bs.vy=st[ST_BUOY+8];bs.aY=st[ST_BUOY+9];
			bs.sub=st[ST_BUOY+10];bs.blue=st[ST_BUOY+11];bs.fluid=st[ST_BUOY+12];bs.peak=st[ST_BUOY+13];
			bs.ux=st[ST_BUOY+14];bs.uy=st[ST_BUOY+15];
			bs.frames=Math.round(st[ST_BUOY+16])||0;
			bs.samples=Math.round(st[ST_BUOY+17])||0;
			for(let i=0;i<buoy.SAMPLE_LEN;i++)buoy.sample[i]=snap.tables.buoySample[i];
			cfg.buoyShow=bs.active?1:0;
			cfg.buoyW=bs.w;cfg.buoyH=bs.h;
			cfg.buoyX0=bs.x0;cfg.buoyY0=bs.y0;
			cfg.buoySub=bs.sub;
		}
	}
	function packState(stateArr,cs,src){
		stateArr.fill(0);
		const frame=src.frame!==undefined?src.frame:cs.frame;
		const time=src.time!==undefined?src.time:cs.time;
		stateArr[ST_FRAME]=frame;
		stateArr[ST_TIME]=time;
		stateArr[ST_DT]=cs.dt;
		stateArr[ST_SUBSTEPS]=cs.substeps;
		stateArr[ST_VCLAMP]=cs.vClamp;
		stateArr[ST_RHO_MEAN]=cs.rhoMean;
		stateArr[ST_RHO_BASE]=cs.rhoBase;
		stateArr[ST_PMIN]=cs.PMin;
		stateArr[ST_PMAX]=cs.PMax;
		stateArr[ST_PSPAN]=src.pSpanSmooth!==undefined?src.pSpanSmooth:100;
		stateArr[ST_PAINT_INJ_M]=src.paintInjM||0;
		stateArr[ST_HEAT_NET]=src.heatNetKnown||0;
		stateArr[ST_ACCUM]=src.accumulator||0;
		stateArr[ST_SIM_SUBSTEPS]=src.simSubsteps||cs.substeps||0;
		stateArr[ST_MOUSE_X]=cs.mouseX;
		stateArr[ST_MOUSE_Y]=cs.mouseY;
		stateArr[ST_MOUSE_STR]=cs.mouseStrength;
		stateArr[ST_MOUSE_VX]=cs.mouseVX;
		stateArr[ST_MOUSE_VY]=cs.mouseVY;
		const keys=["m","dr","dg","db","E"];
		for(let i=0;i<5;i++){
			const k=keys[i];
			stateArr[ST_EMIT_INJ+i]=(src.emitInjKnown&&src.emitInjKnown[k])||0;
			stateArr[ST_EMIT_REM+i]=(src.emitRemKnown&&src.emitRemKnown[k])||0;
			stateArr[ST_EMIT_NET+i]=(src.emitNetKnown&&src.emitNetKnown[k])||0;
		}
		stateArr[ST_COLL_GPU_MASK]=(src.colliders&&src.colliders.hasGpuMask)?1:0;
		const sc=src.statsCache;
		if(sc){
			stateArr[ST_STATS_FRAME]=sc.frame||0;
			for(let i=0;i<5;i++){
				const k=keys[i];
				stateArr[ST_TOT+i]=(sc.totals&&sc.totals[k])||0;
				stateArr[ST_EXP+i]=(sc.expected&&sc.expected[k])||0;
				stateArr[ST_MIN+i]=(sc.mins&&sc.mins[k])||0;
			}
			stateArr[ST_MAX_M]=sc.maxM||0;
			stateArr[ST_CNT_NEG]=sc.cntNeg||0;
			stateArr[ST_CNT_OVER]=sc.cntOver||0;
			if(sc.motion){
				stateArr[ST_MOT]=sc.motion.maxVel||0;
				stateArr[ST_MOT+1]=sc.motion.pMin||0;
				stateArr[ST_MOT+2]=sc.motion.pMax||0;
				stateArr[ST_MOT+3]=sc.motion.maxDiv||0;
			}
			if(sc.thermal){
				stateArr[ST_THERM]=sc.thermal.TMin||0;
				stateArr[ST_THERM+1]=sc.thermal.TMax||0;
				stateArr[ST_THERM+2]=sc.thermal.TMean||0;
				stateArr[ST_THERM+3]=sc.thermal.kinetic||0;
				stateArr[ST_THERM+4]=sc.thermal.heatIn||0;
				stateArr[ST_THERM+5]=sc.thermal.heatOut||0;
			}
		}
		if(src.phaseCounts){
			for(let i=0;i<6;i++)stateArr[ST_PHASE_COUNTS+i]=src.phaseCounts[i]||0;
		}
		if(src.diag){
			const d=src.diag;
			stateArr[ST_DIAG]=d.sSaturated||0;
			stateArr[ST_DIAG+1]=d.poissonRes||0;
			stateArr[ST_DIAG+2]=d.poissonRhs||0;
			stateArr[ST_DIAG+3]=d.poissonRel||0;
			stateArr[ST_DIAG+4]=d.poissonDrift||0;
			stateArr[ST_DIAG+5]=d.poissonDriftRhs||0;
			stateArr[ST_DIAG+6]=d.poissonDriftRel||0;
		}
		if(src.buoy&&src.buoy.state){
			const bs=src.buoy.state;
			stateArr[ST_BUOY]=bs.active?1:0;
			stateArr[ST_BUOY+1]=bs.cx||0;stateArr[ST_BUOY+2]=bs.cy||0;
			stateArr[ST_BUOY+3]=bs.x0||0;stateArr[ST_BUOY+4]=bs.y0||0;
			stateArr[ST_BUOY+5]=bs.w||0;stateArr[ST_BUOY+6]=bs.h||0;
			stateArr[ST_BUOY+7]=bs.vx||0;stateArr[ST_BUOY+8]=bs.vy||0;stateArr[ST_BUOY+9]=bs.aY||0;
			stateArr[ST_BUOY+10]=bs.sub||0;stateArr[ST_BUOY+11]=bs.blue||0;stateArr[ST_BUOY+12]=bs.fluid||0;stateArr[ST_BUOY+13]=bs.peak||0;
			stateArr[ST_BUOY+14]=bs.ux||0;stateArr[ST_BUOY+15]=bs.uy||0;
			stateArr[ST_BUOY+16]=bs.frames||0;stateArr[ST_BUOY+17]=bs.samples||0;
		}
	}
	function pack(src,outBuffer){
		const cs=src.cfgSnap||cfgState(src.cfg);
		const lay=layout(cs.N,cs.mgCoarsen);
		const buf=resolveOutBuffer(outBuffer,lay.totalBytes);
		const frame=src.frame!==undefined?src.frame:cs.frame;
		const time=src.time!==undefined?src.time:cs.time;
		writeHeader(buf,0,lay,cs,frame,time);
		new Float64Array(buf,lay.cfgOff,CFG_KEYS.length).set(cs.values);
		const stateArr=new Float64Array(buf,lay.stateOff,STATE_LEN);
		packState(stateArr,cs,src);
		const ledgerArr=new Float64Array(buf,lay.ledgerOff,LEDGER_KEYS.length);
		const lg=src.ledger||{};
		for(let i=0;i<LEDGER_KEYS.length;i++)ledgerArr[i]=Number(lg[LEDGER_KEYS[i]])||0;
		packTables(buf,lay,src.probes,src.emitters,src.emitterLedger,src.colliders,src.buoy);
		const f=src.fields;
		if(f){
			for(let i=0;i<FIELD_NAMES.length;i++){
				const name=FIELD_NAMES[i];
				const data=f[name];
				if(!data)continue;
				const fl=lay.fields[name];
				if(fl.type==="u8")new Uint8Array(buf,fl.off,fl.len).set(data);
				else new Float32Array(buf,fl.off,fl.len).set(data);
			}
		}
		return buf;
	}
	function repackRows(srcBuf,srcOff,bytesPerRow,rowBytes,rows,dstBuf,dstOff){
		const src=new Uint8Array(srcBuf,srcOff,bytesPerRow*rows);
		const dst=new Uint8Array(dstBuf,dstOff,rowBytes*rows);
		if(bytesPerRow===rowBytes){
			dst.set(src);
			return;
		}
		for(let r=0;r<rows;r++){
			const s=r*bytesPerRow;
			dst.set(src.subarray(s,s+rowBytes),r*rowBytes);
		}
	}
	function repackStagingInto(stagingBuf,stageLay,snapBuf,snapLay){
		for(let i=0;i<stageLay.list.length;i++){
			const e=stageLay.list[i];
			const fl=snapLay.fields[e.name];
			repackRows(stagingBuf,e.offset,e.bytesPerRow,e.rowBytes,e.h,snapBuf,fl.off);
		}
		const ledSrc=new Float32Array(stagingBuf,stageLay.emitterLedgerOffset,EMIT_LEDGER_LEN);
		new Float32Array(snapBuf,snapLay.emitLedgerOff,EMIT_LEDGER_LEN).set(ledSrc);
	}
	function unpack(buf,cfg){
		const view=resolveInView(buf);
		if(view.byteLength<HEADER_BYTES){
			throw new RangeError("deserialize: truncated buffer (need >= "+HEADER_BYTES+" B, got "+view.byteLength+" B)");
		}
		const dv=new DataView(view.buffer,view.byteOffset,view.byteLength);
		const magic=dv.getUint32(0,true);
		if(magic!==MAGIC){
			throw new Error("deserialize: invalid magic 0x"+magic.toString(16));
		}
		const ver=dv.getUint32(4,true);
		if(ver!==VERSION){
			throw new Error("deserialize: unsupported version "+ver+" (expected "+VERSION+")");
		}
		const savedN=dv.getUint32(8,true);
		if(cfg&&savedN!==(cfg.N>>>0)){
			throw new RangeError("deserialize: mismatched N (expected "+cfg.N+", got "+savedN+")");
		}
		const savedMgS=dv.getUint32(32,true);
		const lay=layout(savedN,cfg?cfg.mgCoarsen:savedMgS);
		if(savedMgS!==lay.mgS){
			throw new Error("deserialize: mismatched mgCoarsen (expected "+lay.mgS+", got "+savedMgS+")");
		}
		const totalBytes=dv.getUint32(36,true);
		if(totalBytes!==lay.totalBytes||view.byteLength<lay.totalBytes){
			throw new RangeError("deserialize: truncated or mismatched snapshot size (expected "+lay.totalBytes+" B, got "+view.byteLength+" B)");
		}
		const scenarioNames=Fluid.core.config.scenarioNames;
		const scenarioIdx=dv.getUint32(40,true);
		const savedScenario=scenarioNames[scenarioIdx]||"";
		if(cfg&&savedScenario!==cfg.scenario){
			throw new Error("deserialize: mismatched scenario (expected "+cfg.scenario+", got "+(savedScenario||scenarioIdx)+")");
		}
		// Copy into an aligned slice when the caller passed an unaligned view
		const aligned=(view.byteOffset&7)===0
			?view.buffer
			:view.buffer.slice(view.byteOffset,view.byteOffset+lay.totalBytes);
		const base=(view.byteOffset&7)===0?view.byteOffset:0;
		const savedCfg=new Float64Array(aligned,base+lay.cfgOff,CFG_KEYS.length);
		if(cfg){
			for(let i=0;i<CFG_KEYS.length;i++){
				const k=CFG_KEYS[i];
				const expected=numVal(cfg[k]);
				const got=savedCfg[i];
				const tol=1e-6*Math.max(1,Math.abs(expected),Math.abs(got));
				if(Math.abs(expected-got)>tol){
					throw new Error("deserialize: mismatched "+k+" (expected "+expected+", got "+got+")");
				}
			}
		}
		const frame=dv.getUint32(44,true);
		const time=dv.getFloat64(72,true);
		const dt=dv.getFloat64(80,true);
		const state=new Float64Array(aligned,base+lay.stateOff,STATE_LEN);
		const ledgerArr=new Float64Array(aligned,base+lay.ledgerOff,LEDGER_KEYS.length);
		const ledger={};
		for(let i=0;i<LEDGER_KEYS.length;i++)ledger[LEDGER_KEYS[i]]=ledgerArr[i];
		const tables={
			probeUp:new Float32Array(aligned,base+lay.probeUpOff,PROBE_LEN),
			probeRaw:new Float32Array(aligned,base+lay.probeRawOff,PROBE_LEN),
			probeMeta:new Float32Array(aligned,base+lay.probeMetaOff,PROBE_META_LEN),
			emitUp:new Float32Array(aligned,base+lay.emitUpOff,EMIT_LEN),
			emitLedger:new Float32Array(aligned,base+lay.emitLedgerOff,EMIT_LEDGER_LEN),
			emitMeta:new Float64Array(aligned,base+lay.emitMetaOff,EMIT_META_LEN),
			collUp:new Float32Array(aligned,base+lay.collUpOff,COLL_LEN),
			collMeta:new Float64Array(aligned,base+lay.collMetaOff,COLL_META_LEN),
			buoySample:new Float32Array(aligned,base+lay.buoySampleOff,BUOY_SAMPLE_PAD)
		};
		const fields={};
		for(let i=0;i<FIELD_NAMES.length;i++){
			const name=FIELD_NAMES[i];
			const fl=lay.fields[name];
			fields[name]=fl.type==="u8"
				?new Uint8Array(aligned,base+fl.off,fl.len)
				:new Float32Array(aligned,base+fl.off,fl.len);
		}
		return {
			N:savedN,texW:lay.texW,texH:lay.texH,
			mgS:lay.mgS,mgNc:lay.mgNc,mgTexW:lay.mgTexW,mgTexH:lay.mgTexH,
			scenario:savedScenario,frame,time,dt,
			cfgValues:savedCfg,state,ledger,tables,fields
		};
	}

	Fluid.core.save={
		MAGIC,VERSION,HEADER_BYTES,STATE_LEN,CFG_KEYS,LEDGER_KEYS,FIELD_NAMES,
		ST_FRAME,ST_TIME,ST_DT,ST_SUBSTEPS,ST_VCLAMP,ST_RHO_MEAN,ST_RHO_BASE,
		ST_PMIN,ST_PMAX,ST_PSPAN,ST_PAINT_INJ_M,ST_HEAT_NET,ST_ACCUM,ST_SIM_SUBSTEPS,
		ST_MOUSE_X,ST_MOUSE_Y,ST_MOUSE_STR,ST_MOUSE_VX,ST_MOUSE_VY,
		ST_EMIT_INJ,ST_EMIT_REM,ST_EMIT_NET,ST_STATS_FRAME,ST_COLL_GPU_MASK,
		ST_TOT,ST_EXP,ST_MIN,ST_MAX_M,ST_CNT_NEG,ST_CNT_OVER,
		ST_MOT,ST_THERM,ST_PHASE_COUNTS,ST_DIAG,ST_BUOY,
		mgDims,layout,byteLength,stagingLayout,cfgState,
		resolveOutBuffer,writeHeader,packState,packTables,restoreTables,
		pack,repackRows,repackStagingInto,unpack
	};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.save;
})(globalThis);
