/**
 * \brief If defined save H_max map.
 */

#define SAVE_ETA_MAX_MAP

#include "SingleLineGridSimulation.h"
#include "DiskOffloadSimulator.h"
#include <cstdlib>
#include <iostream>
#include <chrono> // Для измерения времени

#include "ASCParser.h"
#include "Sandbox.h"

#include <fstream>
static std::ofstream _diag_log("_diag_stages.txt");
#define DIAG(msg) do { _diag_log << msg << std::endl; _diag_log.flush(); } while(0)

int main(int argc, char** argv) {
	DIAG("STAGE 0: main entered");
	auto sandBox = new Sandbox();
	DIAG("STAGE 1: Sandbox constructed");

	// Special tests via string argument (string-keyed; processed AFTER flag parsing below
	// so flags like --threads / --timesteps / --disk-skip-reference apply to them too).
	std::string special_cmd;
	if (argc > 1) {
		std::string arg(argv[1]);
		if (arg == "coupling" || arg == "mixed" || arg == "dgvalidation" || arg == "dgvsmc"
			|| arg == "dgprobe" || arg == "dgprobelegacy" || arg == "dgsquare" || arg == "dgsquarenolim"
			|| arg == "hybridflat" || arg == "hybridwave" || arg == "hybridisland" || arg == "dgislandprobe" || arg == "nesteddg"
			|| arg == "hybrids7" || arg == "hybrids7sph" || arg == "hybrids7cart"
			|| arg == "hybrids7test" || arg == "hybrids7probe" || arg == "hybrids7probecart" || arg == "hybrids7short" || arg == "hybrids7mid"
			|| (arg.size() > 4 && arg.substr(0, 4) == "disk")) {
			special_cmd = arg;
		}
	}

	// Numeric test index + optional flags:
	//   --save-interval=N : override Sandbox SetSaveInterval (large value → only final snapshot saved)
	//   --outdir=PATH     : override Result/test{N} folder
	//   --threads=N       : override OpenMP num_threads (for perf benchmarking)
	//   --subgrids=N      : override num_subgrids (e.g., =1 to use single big grid with OMP-inside)
	//   --timesteps=N     : override scenario timesteps (for perf probing)
	//   --disk-skip-reference : skip in-memory reference in DiskOffloadTest (benchmark mode)
	int testIndex = 18;
	int saveIntervalOverride = -1;
	int threadsOverride = -1;
	int subgridsOverride = -1;
	std::string outdirOverride;
	for (int i = 1; i < argc; ++i) {
		std::string arg(argv[i]);
		// Skip the special_cmd argument (if argv[1] matched, don't reprocess as flag/test number).
		if (i == 1 && !special_cmd.empty()) continue;
		if (arg.rfind("--save-interval=", 0) == 0) { saveIntervalOverride = std::atoi(arg.c_str() + 16); continue; }
		if (arg.rfind("--outdir=", 0) == 0) { outdirOverride = arg.substr(9); continue; }
		if (arg.rfind("--threads=", 0) == 0) { threadsOverride = std::atoi(arg.c_str() + 10); continue; }
		if (arg.rfind("--subgrids=", 0) == 0) { subgridsOverride = std::atoi(arg.c_str() + 11); continue; }
		if (arg.rfind("--timesteps=", 0) == 0) { sandBox->SetTimestepsOverride(std::atoi(arg.c_str() + 12)); continue; }
		if (arg == "--disk-skip-reference") { sandBox->SetDiskSkipReference(true); continue; }
		char* end = nullptr;
		long parsed = std::strtol(arg.c_str(), &end, 10);
		if (end != arg.c_str() && end && *end == '\0') { testIndex = static_cast<int>(parsed); continue; }
		std::cerr << "Invalid argument: " << arg << std::endl;
		return 1;
	}

	sandBox->SetSaveIntervalOverride(saveIntervalOverride);
	sandBox->SetOutdirOverride(outdirOverride);
	sandBox->SetThreadsOverride(threadsOverride);
	sandBox->SetSubgridsOverride(subgridsOverride);

	// Dispatch special string-keyed commands (after flags applied).
	if (!special_cmd.empty()) {
		if (special_cmd == "coupling") { sandBox->CouplingTest(); return 0; }
		if (special_cmd == "mixed") { sandBox->MixedSolverTest(); return 0; }
		if (special_cmd == "dgvalidation") { sandBox->DGValidationTest(); return 0; }
		if (special_cmd == "dgvsmc") { sandBox->DGvsMacCormackTest(); return 0; }
		if (special_cmd == "dgprobe") { sandBox->DGStabilityProbe(false); return 0; }
		if (special_cmd == "dgprobelegacy") { sandBox->DGStabilityProbe(true); return 0; }
		if (special_cmd == "dgsquare") { sandBox->DGSquarePulseProbe(true); return 0; }
		if (special_cmd == "dgsquarenolim") { sandBox->DGSquarePulseProbe(false); return 0; }
		if (special_cmd == "hybridflat") { sandBox->HybridFlatTest(); return 0; }
	if (special_cmd == "hybridwave") { sandBox->HybridWaveTest(); return 0; }
	if (special_cmd == "hybridisland") { sandBox->HybridIslandTest(); return 0; }
		if (special_cmd == "dgislandprobe") { sandBox->DGIslandProbe(); return 0; }
		if (special_cmd == "nesteddg") { sandBox->NestedDGTest(); return 0; }
		if (special_cmd == "hybrids7" || special_cmd == "hybrids7sph") { sandBox->HybridS7Test(30); return 0; }
		if (special_cmd == "hybrids7cart") { sandBox->HybridS7Test(31); return 0; }
		if (special_cmd == "hybrids7test") { DIAG("STAGE 2H: dispatch hybrids7test"); sandBox->HybridS7Test(30, 5000); DIAG("STAGE 3H: hybrids7test returned"); return 0; }
		if (special_cmd == "hybrids7probe") { sandBox->HybridS7Test(30, 300); return 0; }
		if (special_cmd == "hybrids7short") { sandBox->HybridS7Test(30, 600); return 0; }
		if (special_cmd == "hybrids7mid") { sandBox->HybridS7Test(30, 2500); return 0; }
		if (special_cmd == "hybrids7probecart") { sandBox->HybridS7Test(31, 300); return 0; }
		if (special_cmd.size() > 4 && special_cmd.substr(0, 4) == "disk") {
			int diskTest = std::atoi(special_cmd.substr(4).c_str());
			sandBox->DiskOffloadTest(diskTest);
			return 0;
		}
		std::cerr << "Unknown special command: " << special_cmd << std::endl;
		return 1;
	}

	DIAG("STAGE 2: about to call SimulatorTest(" + std::to_string(testIndex) + ")");
	sandBox->SimulatorTest(testIndex);
	DIAG("STAGE 3: SimulatorTest returned");
	return 0;
}
 
