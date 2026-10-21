"use strict";
// CPU-oracle trial for a third projection level. The GPU implementation and
// shipping defaults remain unchanged until a candidate clears these same-state
// quality rows and gets a pinned hardware capture.
// run: node experiments/mg-three-level.js
//      node experiments/mg-three-level.js --n512
//
// Each candidate warms on the exact two-level K60 reference, then measures one
// step from that shared starting state. The oracle's true residual is the gate;
// timing is exploratory CPU cost, not a prediction of GPU speed.

const path = require("node:path");
const root = path.resolve(__dirname, "..");
require(path.join(root, "src/core/config.js"));
require(path.join(root, "src/core/grid.js"));
require(path.join(root, "src/core/stats.js"));
require(path.join(root, "src/core/multigrid.js"));
const solverMod = require(path.join(root, "src/core/solver.js"));
const cfg0 = Fluid.core.config.defaults;
const DT = 1 / 240;
const REFERENCE = { mgCycles: 2, mgPre: 4, mgPost: 8, mgCoarseIters: 60 };

const CANDIDATES = [
	{ name: "two-level K20", levels: 2, innerPre: 0, innerPost: 0, K: 20 },
	{ name: "three L2/4 K20", levels: 3, innerPre: 2, innerPost: 4, K: 20 },
	{ name: "three L4/8 K20", levels: 3, innerPre: 4, innerPost: 8, K: 20 },
	{ name: "three L4/8 K40", levels: 3, innerPre: 4, innerPost: 8, K: 40 }
];

function run(scenario,N,warmup,candidate){
	const cfg=Object.assign({},cfg0,{N,scenario,mgOracleLevels:3,mgLevels:2});
	cfg.texW=N+2;cfg.texH=N+2;
	Fluid.core.config.prepareScenario(cfg);
	const sim=solverMod.create(cfg);
	Object.assign(cfg,REFERENCE);
	for(let i=0;i<warmup;i++)sim.step(DT);
	Object.assign(cfg,{mgLevels:candidate.levels,mgInnerPre:candidate.innerPre,mgInnerPost:candidate.innerPost,
		mgCycles:2,mgPre:4,mgPost:8,mgCoarseIters:candidate.K});
	const t0=Date.now();
	sim.step(DT);
	const ms=Date.now()-t0;
	const b=Math.max(sim.diag.divBefore,1e-9);
	const r=sim.mg.residualNorm();
	const stats=sim.stats();
	return {trueResidual:r/b,rel:sim.diag.pressureResidual,divBefore:b,divAfter:sim.diag.divAfter,
		maxM:stats.tot.maxM,over:stats.tot.cntOver,ms};
}

function report(scenario,N,warmup){
	console.log("== "+scenario+" N="+N+" ("+warmup+" reference warmup steps) ==");
	let baseline=null;
	for(const candidate of CANDIDATES){
		const r=run(scenario,N,warmup,candidate);
		if(candidate.levels===2)baseline=r;
		const delta=baseline?((r.trueResidual/baseline.trueResidual-1)*100):0;
		console.log("  "+candidate.name.padEnd(18)+
			" true="+r.trueResidual.toExponential(3)+
			" rel="+r.rel.toExponential(3)+
			" div="+r.divBefore.toExponential(3)+" -> "+r.divAfter.toExponential(3)+
			" maxM="+r.maxM.toFixed(4)+" over="+r.over+
			" cpu="+r.ms+"ms/step"+
			(candidate.levels===3?" true-vs-2L="+delta.toFixed(1)+"%":""));
	}
}

const n512=process.argv.includes("--n512");
const all=process.argv.includes("--all");
const scenarioArg=process.argv.find(arg=>arg.startsWith("--scenario="));
const warmupArg=process.argv.find(arg=>arg.startsWith("--warmup="));
const scenarios=scenarioArg?[scenarioArg.slice("--scenario=".length)]:
	(all?["dam","blobs","jet","cycle"]:["dam","cycle"]);
const warmup=warmupArg?Math.max(1,Number(warmupArg.slice("--warmup=".length))||180):180;
console.log("mg-three-level · pinned dt="+DT+" · same-state oracle · two-level K20 baseline");
for(const scenario of scenarios)report(scenario,n512?512:128,n512?90:warmup);
console.log("done");
