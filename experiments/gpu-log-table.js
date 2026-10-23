// Table of the hardware captures in experiments/logs, for the plan 3 rule that a
// number without a pinned step policy is not comparable: rows are grouped by
// (scenario, dt, ns) and only compared inside a group, so an auto-policy drift
// shows up as "no peer" instead of a fake speedup.
//
//	node experiments/gpu-log-table.js experiments/logs/2026-10-20-12_*.log
//	node experiments/gpu-log-table.js --csv <logs...>
"use strict";
const fs = require("node:fs");

const RE = {
	settings: /^grid N=(\d+).*pressureK=(\d+) coarse solver=(\S+) substeps=(auto|\d+ fixed)(?: FCT=\d+)?(?: mac=(\d+)\/\d+)?(?: fuse=(\d+))?/,
	step: /^statsFrame#.* N=(\d+) dt=(\S+) K=(\d+).*mgS(\d+)x(\d+).*ns=(\d+)\/(\d+) (auto|fixed) scenario=(\S+)/,
	fps: /^frames\/s=([\d.]+)/,
	gpu: /^gpu timing \((\d+) samples\): vel=([\d.]+).*press=([\d.]+) trans=([\d.]+) phase=([\d.]+).*step sum=([\d.]+)/,
	peaks: /^sampled peaks: maxDiv=(\S+) maxM=(\S+) over=(\d+)/,
	poisson: /^poisson: res=(\S+) rhs=(\S+) true=(\S+)(?: drift=(\S+))?/,
	peaksPoisson: /poissonTrue=(\S+)/,
	peaksDrift: / drift=(\S+) at frame=/
};

function number(v) {
	const n = Number(v);
	return Number.isFinite(n) ? n : 0;
}

function parse(file) {
	const rows = [];
	let head = null, row = null;
	for (const line of fs.readFileSync(file, "utf8").split("\n")) {
		const sc = /^--- (\S+) · (\S+) ---/.exec(line);
		if (sc) { row = { scenario: sc[1], ok: sc[2] }; rows.push(row); continue; }
		if (!row) {
			const h = RE.settings.exec(line);
			if (h) head = {
				N: +h[1], K: +h[2], solver: h[3], substeps: h[4],
				mac: h[5] === undefined ? 0 : +h[5],
				fuse: h[6] === undefined ? 0 : +h[6]
			};
			continue;
		}
		const st = RE.step.exec(line);
		if (st) {
			Object.assign(row, {
				dt: number(st[2]), K: +st[3], pre: +st[4], cycles: +st[5],
				ns: +st[6], nsMax: +st[7], policy: st[8]
			});
			continue;
		}
		const fps = RE.fps.exec(line);
		if (fps) { row.fps = number(fps[1]); continue; }
		const gpu = RE.gpu.exec(line);
		if (gpu) {
			Object.assign(row, {
				vel: number(gpu[2]), press: number(gpu[3]),
				trans: number(gpu[4]), phase: number(gpu[5]), sum: number(gpu[6])
			});
			continue;
		}
		const peaks = RE.peaks.exec(line);
		if (peaks) { Object.assign(row, { maxDiv: number(peaks[1]), maxM: number(peaks[2]), over: +peaks[3] }); }
		const pp = RE.peaksPoisson.exec(line);
		if (pp) row.peakTrue = number(pp[1]);
		const pd = RE.peaksDrift.exec(line);
		if (pd) row.peakDrift = number(pd[1]);
		const pz = RE.poisson.exec(line);
		if (pz) Object.assign(row, { res: number(pz[1]), rhs: number(pz[2]), trueRel: number(pz[3]), drift: pz[4] === undefined ? undefined : number(pz[4]) });
	}
	const stamp = /(?:(\d\d)-)?(\d\d_\d\d)-GPU-browser-tests/.exec(file);
	const tag = stamp ? (stamp[1] ? stamp[1] + "/" + stamp[2] : stamp[2]) : file;
	for (const row of rows) {
		row.log = tag;
		row.N = head ? head.N : row.N;
		row.mac = head ? head.mac : 0;
		row.fuse = head ? head.fuse : 0;
		if (!row.K) row.K = head ? head.K : 0;
		if (!row.solver) row.solver = head ? head.solver : "?";
	}
	return rows;
}

function key(row) {
	return row.scenario + " dt=" + row.dt.toPrecision(4) + " ns=" + row.ns + " N=" + row.N;
}

function fmt(v, digits) {
	return v === undefined ? "-" : v.toFixed(digits);
}

function main() {
	const args = process.argv.slice(2);
	const csv = args[0] === "--csv";
	const rows = args.filter(a => !a.startsWith("--")).flatMap(parse);
	if (!rows.length) { console.error("no rows: pass experiments/logs/*.log files"); process.exitCode = 1; return; }
	const groups = new Map();
	for (const row of rows) {
		const k = key(row);
		if (!groups.has(k)) groups.set(k, []);
		groups.get(k).push(row);
	}
	const head = [
		"log", "scenario", "K", "cyc", "mac", "fuse", "policy", "ns", "fps",
		"vel", "press", "trans", "phase", "step", "press%", "maxDiv", "endTrue", "peakTrue", "drift"
	];
	const lines = [head.join(csv ? "," : "\t")];
	for (const [k, list] of groups) {
		lines.push("");
		lines.push("# " + k + (list.length < 2 ? "  (no peer: not comparable)" : "  · % vs first row"));
		if (list.some(r => r.peakDrift !== undefined) && list.some(r => r.peakDrift === undefined && r.peakTrue !== undefined)) {
			lines.push("# peakTrue on a row with no drift column is the post-step number; do not rank it against the solve");
		}
		const base = list[0];
		for (const row of list) {
			const cells = [
				row.log, row.scenario, row.K, row.cycles, row.mac, row.fuse, row.policy, row.ns,
				fmt(row.fps, 1), fmt(row.vel, 3), fmt(row.press, 3), fmt(row.trans, 3), fmt(row.phase, 3), fmt(row.sum, 3),
				fmt(100 * row.press / row.sum, 1),
				fmt(row.maxDiv, 3),
				row.trueRel === undefined ? "-" : row.trueRel.toExponential(2),
				row.peakTrue === undefined ? "-" : row.peakTrue.toExponential(2),
				row.peakDrift === undefined ? "-" : row.peakDrift.toExponential(2)
			];
			if (list.length > 1 && row !== base) {
				cells[8] += " (" + (100 * (row.fps / base.fps - 1)).toFixed(1) + "%)";
				cells[13] += " (" + (100 * (row.sum / base.sum - 1)).toFixed(1) + "%)";
				const ratio = row.maxDiv / base.maxDiv;
				cells[15] += " (" + ratio.toFixed(ratio >= 10 ? 1 : 2) + "x)";
			}
			lines.push(cells.join(csv ? "," : "\t"));
		}
	}
	console.log(lines.join("\n"));
}

main();
