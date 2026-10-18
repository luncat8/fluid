(function () {
	"use strict";
	const output = document.getElementById("probe-output");
	const status = document.getElementById("status");
	const errorPanel = document.getElementById("error-panel");
	const runButton = document.getElementById("run-probe");
	const saveButton = document.getElementById("save-probe");
	let activeDevice = null;
	let report = null;

	function display(current) {
		report = current;
		output.textContent = JSON.stringify(report, null, "\t");
		status.textContent = report.status === "passed" ? "Probe passed · simulation not implemented" : "Probe " + report.status;
		errorPanel.hidden = report.errors.length === 0;
		errorPanel.textContent = report.errors.join("\n\n");
	}

	async function run() {
		runButton.disabled = true;
		saveButton.disabled = true;
		status.textContent = "Requesting WebGPU adapter…";
		errorPanel.hidden = true;
		if (activeDevice) activeDevice.destroy();
		activeDevice = null;
		try {
			const result = await Fluid.gpu.device.probe(window, (message, current) => display(current));
			activeDevice = result.device;
			display(result.report);
			console.log("Fluid capability probe", output.textContent);
			saveButton.disabled = false;
		} finally {
			runButton.disabled = false;
		}
	}

	function save() {
		const blob = new Blob([JSON.stringify(report, null, "\t") + "\n"], { type: "text/plain" });
		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");
		link.href = url;
		link.download = report.timestamp.slice(0, 10) + "-probe.log";
		link.click();
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	}

	runButton.addEventListener("click", run);
	saveButton.addEventListener("click", save);
	window.addEventListener("pagehide", () => { if (activeDevice) activeDevice.destroy(); });
	run();
})();
