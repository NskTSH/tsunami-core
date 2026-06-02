// main.cpp - CUDA версия симулятора MacCormack
//
// Use: MacCormackSimulationCUDA.exe [test_number] [--precision=fp32|fp64]
//   test_number: 1-5 (default: 2)
//   --precision: fp32 (single) or fp64 (double, default)
//
// Сборка: nvcc -O3 -arch=sm_70 -std=c++17 main.cpp ScenarioBuilderCUDA.cpp \
//   SingleLineGridSimulationCUDA.cu MacCormackCUDA.cu -o simulation_cuda

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <string>
#include "SingleLineGridSimulationCUDA.cuh"
#include "ScenarioBuilderCUDA.h"
#include "SingleLineGridSimulationSubgridCUDA.h"
#include "ASCParser.h"
#include "DGSolverCUDA.cuh"  // DG-on-GPU port (added alongside MacCormack; --dgtest)

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [test_number] [--precision=fp32|fp64]" << std::endl;
    std::cout << "  test_number:" << std::endl;
    std::cout << "    1-5   Cartesian tests (synthetic)" << std::endl;
    std::cout << "    6     Japan Sea (S2 ellipsoidal source, reads Data/JapanSea.asc)" << std::endl;
    std::cout << "    11-15 Spherical+Coriolis variants of tests 1-5 (45°N)" << std::endl;
    std::cout << "    16    Japan Sea with Spherical coords (no Coriolis by default)" << std::endl;
    std::cout << "    900   Pacific/Kamchatka full domain, Coriolis ON  (needs Data/pacific_1min/)" << std::endl;
    std::cout << "    901   Pacific/Kamchatka full domain, Coriolis OFF" << std::endl;
    std::cout << "    902   Pacific/Kamchatka 3000-crop probe, Coriolis ON" << std::endl;
    std::cout << "    903   Pacific/Kamchatka 3000-crop probe, Coriolis OFF" << std::endl;
    std::cout << "    904   Pacific synthetic smoke (no data files), Coriolis ON" << std::endl;
    std::cout << "    905   Pacific synthetic smoke (no data files), Coriolis OFF" << std::endl;
    std::cout << "  --precision=fp32: single precision (faster, lower accuracy)" << std::endl;
    std::cout << "  --precision=fp64: double precision (default, matches CPU)" << std::endl;
    std::cout << std::endl;
    std::cout << "Available tests:" << std::endl;
    std::cout << "  1  - Square pulse (120x120, 500 steps)" << std::endl;
    std::cout << "  2  - Cosine pulse from center (1000x1000, 4500 steps)" << std::endl;
    std::cout << "  3  - Plane wave with variable depth (1000x1000, 8000 steps)" << std::endl;
    std::cout << "  4  - Triangle island - Luneberg lens (1000x1000, 9000 steps)" << std::endl;
    std::cout << "  5  - Circular island (1000x1000, 6000 steps)" << std::endl;
    std::cout << "  11 - Test 1 with Spherical coords + Coriolis (45 deg)" << std::endl;
    std::cout << "  12 - Test 2 with Spherical coords + Coriolis (45 deg)" << std::endl;
    std::cout << "  13 - Test 3 with Spherical coords + Coriolis (45 deg)" << std::endl;
    std::cout << "  14 - Test 4 with Spherical coords + Coriolis (45 deg)" << std::endl;
    std::cout << "  15 - Test 5 with Spherical coords + Coriolis (45 deg)" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== MacCormack Simulation CUDA ===" << std::endl;
    std::cout << std::endl;

    int testNumber = 2;
    bool use_fp32 = false;
    int override_timesteps = -1;  // -1 = use scenario default
    int override_save_interval = -1;  // -1 = use default per-test
    int num_subgrids = 1;  // 1 = monolithic path (MacCormackCUDA<Real>); >1 = legacy SubgridCUDA
    bool force_legacy = false;  // --use-legacy: route N=1 also through legacy path (apples-to-apples benchmark)
    bool gpu_ooc = false;       // --gpu-ooc: enable per-step H2D+D2H roundtrip (Case 3 OOC benchmark)
    bool etamax_only = false;   // --etamax-only: save ONLY final eta_max, no velocities; report compute vs save time
    std::string override_outdir;  // empty = use default Result/test{N}[_fp32]

    for (int k = 1; k < argc; ++k) {
        std::string arg(argv[k]);
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        if (arg == "--dgtest") {
            // DG-on-GPU port self-test (Phase 0+). Independent of the MacCormack path.
            return RunDGCudaTest();
        }
        if (arg == "--s7hybrid") {
            // REAL S7 (Japan Sea) MacCormack+DG hybrid on the GPU. Reads Data/JapanSea.asc.
            return RunS7HybridGPU();
        }
        if (arg == "--s7full") {
            // NATIVE-resolution S7: 8-band MC|DG|MC hybrid (band #3 = DG). Reads Data/JapanSea.asc.
            return RunS7HybridFullGPU();
        }
        if (arg == "--precision=fp32" || arg == "--precision=float" || arg == "--fp32") {
            use_fp32 = true;
            continue;
        }
        if (arg == "--precision=fp64" || arg == "--precision=double" || arg == "--fp64") {
            use_fp32 = false;
            continue;
        }
        if (arg.rfind("--timesteps=", 0) == 0) {
            override_timesteps = std::atoi(arg.c_str() + 12);
            continue;
        }
        if (arg.rfind("--outdir=", 0) == 0) {
            override_outdir = arg.substr(9);
            continue;
        }
        if (arg.rfind("--save-interval=", 0) == 0) {
            override_save_interval = std::atoi(arg.c_str() + 16);
            continue;
        }
        if (arg.rfind("--subgrids=", 0) == 0) {
            num_subgrids = std::atoi(arg.c_str() + 11);
            if (num_subgrids < 1) num_subgrids = 1;
            continue;
        }
        if (arg == "--use-legacy" || arg == "--legacy") {
            force_legacy = true;
            continue;
        }
        if (arg == "--gpu-ooc" || arg == "--ooc") {
            gpu_ooc = true;
            force_legacy = true;  // OOC mode goes through legacy SubgridCUDA path
            continue;
        }
        if (arg == "--etamax-only") {
            etamax_only = true;
            continue;
        }
        // Numeric: test number. 1-6 Cartesian, 11-16 Spherical+Coriolis,
        // 18=S7 Japan Sea Spherical, 19=S7 Japan Sea Cartesian,
        // 900-903 Pacific/Kamchatka (Spherical + Sommerfeld BC), 904/905 Pacific smoke.
        int n = std::atoi(arg.c_str());
        if ((n >= 1 && n <= 6) || (n >= 11 && n <= 16) || n == 18 || n == 19 ||
            (n >= 900 && n <= 905)) {
            testNumber = n;
            continue;
        }
        std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "Running Test " << testNumber
              << " (precision=" << (use_fp32 ? "fp32" : "fp64")
              << ", subgrids=" << num_subgrids << ")" << std::endl;
    std::cout << std::endl;

    // === Legacy subgrid path (num_subgrids > 1 OR --use-legacy, FP64-only) ===
    // Routes Test 6 (Japan Sea S2) through SingleLineGridSimulationSubgridCUDA.
    // For apples-to-apples benchmark of subgrid overhead, use --use-legacy with --subgrids=1.
    if (num_subgrids > 1 || force_legacy) {
        if (testNumber != 6) {
            std::cerr << "ERROR: --subgrids=" << num_subgrids
                      << " currently supported only for testNumber=6 (Japan Sea S2)."
                      << " Other tests use monolithic path (--subgrids=1)." << std::endl;
            return 1;
        }
        if (use_fp32) {
            std::cerr << "WARNING: legacy subgrid path is FP64-only. Ignoring --precision=fp32."
                      << std::endl;
        }

        std::string folder = override_outdir.empty()
            ? ("Result/test" + std::to_string(testNumber) + "_subgrid_n" + std::to_string(num_subgrids))
            : override_outdir;
        std::cout << "Output folder: " << folder << std::endl;

        // Parameters mirror Data/JapanSeaOneSourceS2.par (used by CPU BuildTest6 +
        // new CUDA monolithic path). InitializeFromParser will set up host grid
        // via Grid::SetParamsTest6 (S2 source hardcoded) and create N subgrids.
        ASCParser parser;
        parser.Read("Data/JapanSea.asc");
        const double g_val = 9.81;
        const double tau_val = 0.5;
        const int default_timesteps = 25000;
        const int timesteps = (override_timesteps > 0) ? override_timesteps : default_timesteps;
        const int save_interval = (override_save_interval > 0) ? override_save_interval : 999999;
        const double min_calc_value = 8.0;

        auto start_time = std::chrono::high_resolution_clock::now();

        SingleLineGridSimulationSubgridCUDA simulation;
        simulation.SetOutOfCoreMode(gpu_ooc);
        simulation.InitializeFromParser(parser, g_val, tau_val,
                                        timesteps, num_subgrids,
                                        save_interval, min_calc_value);
        simulation.Run();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << std::endl << "SubgridCUDA completed in "
                  << elapsed.count() << " seconds." << std::endl;
        return 0;
    }

    auto scenario = ScenarioBuilderCUDA::BuildScenario(testNumber);

    SingleLineGridSimulationCUDA simulation;
    simulation.SetUseFloat32(use_fp32);

    ScenarioBuilderCUDA::ApplyScenarioToSimulation(scenario, simulation);

    // Production benchmark mode: save ONLY the final eta_max, no velocities,
    // and report pure-compute vs compute+save time separately (overrides scenario).
    if (etamax_only) {
        simulation.SetEtaMaxOnly(true);
        simulation.SetSaveVelocities(false);
        std::cout << "Mode: --etamax-only (final eta_max only; compute/save timed separately)" << std::endl;
    }

    // Папка результата (совпадает с CPU: Result/test{N}/, для FP32 — Result/test{N}_fp32/).
    std::string folder;
    if (!override_outdir.empty()) {
        folder = override_outdir;
    } else {
        folder = "Result/test" + std::to_string(testNumber);
        if (use_fp32) folder += "_fp32";
    }
    simulation.SetResultFolder(folder);

    if (override_timesteps > 0) {
        simulation.OverrideTimesteps(override_timesteps);
        std::cout << "Timesteps override: " << override_timesteps << std::endl;
    }

    simulation.Initialize();

    // Save interval: match CPU Sandbox defaults to keep regression file sets aligned.
    // CPU Sandbox.h: test 6/16 → default branch (save every 10); test 18/19 (S7) → save every 500.
    // Pacific (900-903): large interval — at 65M cells each eta frame is ~1.4 GB of text,
    // so save every 10 would dump ~700 GB over 5000 steps. 904/905 (small smoke) → every 100.
    int save_interval;
    if (testNumber == 18 || testNumber == 19) save_interval = 500;
    else if (testNumber >= 900 && testNumber <= 903) save_interval = 500;
    else if (testNumber == 904 || testNumber == 905) save_interval = 100;
    else save_interval = 10;
    if (override_save_interval > 0) save_interval = override_save_interval;
    simulation.SetSaveInterval(save_interval);
    std::cout << "Save interval: " << save_interval << std::endl;

    simulation.Run();

    return 0;
}
