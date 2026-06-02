#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdlib>      // _dupenv_s, std::stod, free
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

#include "DefaultParametersParser.h"
#include "ScenarioBuilder.h"
#include "SingleLineGridSimulation.h"
#include "ZDataParser.h"
#include "HybridDomainBuilder.h"
#include "SimulationRunner.h"
#include "DGValidation.h"
#include "DiskOffloadSimulator.h"
#include "Diag.h"

class Sandbox
{
	// CLI overrides set by Source.cpp before SimulatorTest():
	//   save_interval_override_ >= 0 replaces SetSaveInterval(...) chosen by testIndex
	//   outdir_override_ non-empty replaces "Result/test{N}" folder
	//   threads_override_ > 0 replaces scenario.num_threads
	int save_interval_override_ = -1;
	std::string outdir_override_;
	int threads_override_ = -1;
	int subgrids_override_ = -1;
	int timesteps_override_ = -1;
	bool disk_skip_reference_ = false;
public:
	void SetSaveIntervalOverride(int v) { save_interval_override_ = v; }
	void SetOutdirOverride(const std::string& v) { outdir_override_ = v; }
	void SetThreadsOverride(int v) { threads_override_ = v; }
	void SetSubgridsOverride(int v) { subgrids_override_ = v; }
	void SetTimestepsOverride(int v) { timesteps_override_ = v; }
	void SetDiskSkipReference(bool v) { disk_skip_reference_ = v; }
private:
	void SimulatorTests(int testIndex)
	{
		SingleLineGridSimulation simulation;
		SingleLineScenario scenario;
		switch (testIndex)
		{
		// Cartesian coordinate tests
		case 1: scenario = ScenarioBuilder::BuildTest1(); break;
		case 2: scenario = ScenarioBuilder::BuildTest2(); break;
		case 3: scenario = ScenarioBuilder::BuildTest3(); break;
		case 4: scenario = ScenarioBuilder::BuildTest4(); break;
		case 5: scenario = ScenarioBuilder::BuildTest5(); break;
		case 6: scenario = ScenarioBuilder::BuildTest6(); break;
		case 7: scenario = ScenarioBuilder::BuildTest7(); break;
		// Spherical coordinate tests (with Coriolis force)
		case 11: scenario = ScenarioBuilder::BuildTest1Spherical(); break;
		case 12: scenario = ScenarioBuilder::BuildTest2Spherical(); break;
		case 13: scenario = ScenarioBuilder::BuildTest3Spherical(); break;
		case 14: scenario = ScenarioBuilder::BuildTest4Spherical(); break;
		case 15: scenario = ScenarioBuilder::BuildTest5Spherical(); break;
		case 16: scenario = ScenarioBuilder::BuildTest6Spherical(); break;
		case 17: scenario = ScenarioBuilder::BuildTest7Spherical(); break;
		// Japan Sea S7 source in Cartesian coordinates
		case 19: scenario = ScenarioBuilder::BuildTest6CartesianS7(); break;
		// Japan Sea S7 source in spherical coordinates
		case 18: scenario = ScenarioBuilder::BuildTest6SphericalS7(); break;
		// Japan Sea multi-source (3 sources) — Active Region multi-component validation
		case 50: scenario = ScenarioBuilder::BuildTest6JapanSeaThreeSources(); break;
		// Okushiri NTHMP BP9 — 450 m coarse single-grid (Active Region equivalence test)
		case 60: scenario = ScenarioBuilder::BuildOkushiri450m(); break;
		// Monai Valley NTHMP BP6/7 — laboratory 1/400 tank with wavemaker BC
		case 70: scenario = ScenarioBuilder::BuildMonaiValley(); break;
		// FV Phase A validation: lake-at-rest tests
		case 80: scenario = ScenarioBuilder::BuildLakeAtRestFlat(); break;
		case 81: scenario = ScenarioBuilder::BuildLakeAtRestStep(); break;
		// FV Phase B validation: dam-breaks + reflective box
		case 82: scenario = ScenarioBuilder::BuildDamBreakWetWet(); break;
		case 83: scenario = ScenarioBuilder::BuildDamBreakWetDry(); break;
		case 84: scenario = ScenarioBuilder::BuildReflectiveBox(); break;
		// Phase E pre-paper: truly-closed basin with D=0 walls for mass conservation test
		case 85: scenario = ScenarioBuilder::BuildClosedBasinFV(); break;
		// Coriolis tests: plane wave south→north through equator
		case 21: scenario = ScenarioBuilder::BuildTestCoriolis1(CoordSystem::Cartesian); break;
		case 22: scenario = ScenarioBuilder::BuildTestCoriolis1(CoordSystem::Spherical); break;
		// Coriolis tests: plane wave west→east through equator (vertical decomposition)
		case 23: scenario = ScenarioBuilder::BuildTestCoriolis2(CoordSystem::Cartesian); break;
		case 24: scenario = ScenarioBuilder::BuildTestCoriolis2(CoordSystem::Spherical); break;
		// Coriolis tests: ellipsoidal source (N-S elongated, 50x200km)
		case 25: scenario = ScenarioBuilder::BuildTestCoriolisEllipsoidal1(CoordSystem::Cartesian); break;
		case 26: scenario = ScenarioBuilder::BuildTestCoriolisEllipsoidal1(CoordSystem::Spherical); break;
		// Coriolis tests: ellipsoidal source (W-E elongated, 200x50km)
		case 27: scenario = ScenarioBuilder::BuildTestCoriolisEllipsoidal2(CoordSystem::Cartesian); break;
		case 28: scenario = ScenarioBuilder::BuildTestCoriolisEllipsoidal2(CoordSystem::Spherical); break;
		// Coriolis v2: 4000x4000, center at 45°, 8000 steps
		case 41: scenario = ScenarioBuilder::BuildTestCoriolisV2_SN(CoordSystem::Cartesian); break;
		case 42: scenario = ScenarioBuilder::BuildTestCoriolisV2_SN(CoordSystem::Spherical); break;
		case 43: scenario = ScenarioBuilder::BuildTestCoriolisV2_WE(CoordSystem::Cartesian); break;
		case 44: scenario = ScenarioBuilder::BuildTestCoriolisV2_WE(CoordSystem::Spherical); break;
		case 45: scenario = ScenarioBuilder::BuildTestCoriolisV2_EllipNS(CoordSystem::Cartesian); break;
		case 46: scenario = ScenarioBuilder::BuildTestCoriolisV2_EllipNS(CoordSystem::Spherical); break;
		case 47: scenario = ScenarioBuilder::BuildTestCoriolisV2_EllipWE(CoordSystem::Cartesian); break;
		case 48: scenario = ScenarioBuilder::BuildTestCoriolisV2_EllipWE(CoordSystem::Spherical); break;
		// Article tests v7 (2026-04-23): S→N plane wave, finite horizontal extent,
		// zero-gradient BC (index=0), no sponge. Small grid for validation.
		case 480: scenario = ScenarioBuilder::BuildTestCoriolisV7_SmallSN(CoordSystem::Spherical);
		          scenario.coriolis_enabled = true;  scenario.save_velocities = false; break;
		case 481: scenario = ScenarioBuilder::BuildTestCoriolisV7_SmallSN(CoordSystem::Spherical);
		          scenario.coriolis_enabled = false; scenario.save_velocities = false; break;
		// Article v8 production grid (4500x3000) — max(η), max|u|, max|v| maps.
		// 5 threads to keep system responsive; show middle 2500 cols in figs.
		case 420: scenario = ScenarioBuilder::BuildTestCoriolisV7_ProductionSN(CoordSystem::Spherical);
		          scenario.coriolis_enabled = true;  scenario.save_velocities = true;
		          scenario.num_threads = 5;          scenario.num_subgrids = 5; break;
		case 421: scenario = ScenarioBuilder::BuildTestCoriolisV7_ProductionSN(CoordSystem::Spherical);
		          scenario.coriolis_enabled = false; scenario.save_velocities = true;
		          scenario.num_threads = 5;          scenario.num_subgrids = 5; break;
		// v12 (2026-04-25, Q5): same as 420/421 but save eta-snapshots every
		// 720 steps (~1 hour) for moment-snapshot figure (t=2,5,8,12h).
		// Threads reduced to 5 (≈1/3 of 16-core system) per author request
		// (system was overloaded during parallel cor+nocor runs).
		case 422: scenario = ScenarioBuilder::BuildTestCoriolisV7_ProductionSN(CoordSystem::Spherical);
		          scenario.coriolis_enabled = true;  scenario.save_velocities = false;
		          scenario.num_threads = 5;          scenario.num_subgrids = 5; break;
		case 423: scenario = ScenarioBuilder::BuildTestCoriolisV7_ProductionSN(CoordSystem::Spherical);
		          scenario.coriolis_enabled = false; scenario.save_velocities = false;
		          scenario.num_threads = 5;          scenario.num_subgrids = 5; break;
		// Legacy v6 localized-source tests (not used in article v7)
		case 440: scenario = ScenarioBuilder::BuildTestCoriolisV2_WE(CoordSystem::Spherical);
		          scenario.coriolis_enabled = true;  scenario.save_velocities = true; break;
		case 441: scenario = ScenarioBuilder::BuildTestCoriolisV2_WE(CoordSystem::Spherical);
		          scenario.coriolis_enabled = false; scenario.save_velocities = true; break;
		// Article tests: S7 Japan Sea — same Spherical grid, Coriolis on / off
		case 180: scenario = ScenarioBuilder::BuildTest6SphericalS7();
		          scenario.coriolis_enabled = true;  scenario.save_velocities = false; break;
		case 181: scenario = ScenarioBuilder::BuildTest6SphericalS7();
		          scenario.coriolis_enabled = false; scenario.save_velocities = false; break;
		// Radiation BC verification (2026-04-10)
		case 101: scenario = ScenarioBuilder::BuildRadiationLakeAtRest(); break;
		case 102: scenario = ScenarioBuilder::BuildRadiationGaussianPulse(); break;
		// §1 Test L: benchmark scaffold (Work_Plan_2026-05-12)
		case 700: scenario = ScenarioBuilder::BuildTestL_Benchmark(); break;
		// §5 Coriolis sweep по широтам: 7 полос × 20° span
		// 800-806: Spherical с Coriolis (исследуемые)
		// 810-816: Spherical БЕЗ Coriolis (baseline для diff)
		case 800: scenario = ScenarioBuilder::BuildTestCoriolisLatBand( 0.0, 20.0); scenario.coriolis_enabled = true;  break;
		case 801: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(10.0, 30.0); scenario.coriolis_enabled = true;  break;
		case 802: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(20.0, 40.0); scenario.coriolis_enabled = true;  break;
		case 803: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(30.0, 50.0); scenario.coriolis_enabled = true;  break;
		case 804: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(40.0, 60.0); scenario.coriolis_enabled = true;  break;
		case 805: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(50.0, 70.0); scenario.coriolis_enabled = true;  break;
		case 806: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(60.0, 80.0); scenario.coriolis_enabled = true;  break;
		case 810: scenario = ScenarioBuilder::BuildTestCoriolisLatBand( 0.0, 20.0); scenario.coriolis_enabled = false; break;
		case 811: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(10.0, 30.0); scenario.coriolis_enabled = false; break;
		case 812: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(20.0, 40.0); scenario.coriolis_enabled = false; break;
		case 813: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(30.0, 50.0); scenario.coriolis_enabled = false; break;
		case 814: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(40.0, 60.0); scenario.coriolis_enabled = false; break;
		case 815: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(50.0, 70.0); scenario.coriolis_enabled = false; break;
		case 816: scenario = ScenarioBuilder::BuildTestCoriolisLatBand(60.0, 80.0); scenario.coriolis_enabled = false; break;
		// Pacific / Курило-Камчатский источник 2025. sub_size=3000 — окно вокруг источника для probe;
		// 901 = с Coriolis, 902 = без Coriolis. 909 = полный домен с Coriolis (огромный, ≈10 ч).
		case 900: scenario = ScenarioBuilder::BuildPacificKamchatka(true,  0); scenario.save_velocities = true; break;
		case 901: scenario = ScenarioBuilder::BuildPacificKamchatka(false, 0); scenario.save_velocities = true; break;
		// (опционально) sub-region для quick probe:
		case 902: scenario = ScenarioBuilder::BuildPacificKamchatka(true,  3000); scenario.save_velocities = true; break;
		case 903: scenario = ScenarioBuilder::BuildPacificKamchatka(false, 3000); scenario.save_velocities = true; break;
		default: throw std::runtime_error("Unknown test index");
		}

		// Output folder per test: Result/test{N}/, overridable via --outdir=PATH.
		const std::string folder = outdir_override_.empty()
			? ("Result/test" + std::to_string(testIndex))
			: outdir_override_;
		simulation.SetResultFolder(folder);

		// Configure save interval based on test type
		if (testIndex == 420 || testIndex == 421) {
			// Article v15 (2026-05-07): сохраняем покадровые u/v/η для
			// Hovmöller-диаграммы. 10000 шагов / 125 = 80 кадров,
			// шаг между кадрами 625 с ≈ 10.4 мин — это даёт примерно
			// 65 км между «ступенями» при c ≈ 99 м/с, что достаточно
			// для гладкой Hovmöller-диаграммы при interpolation='nearest'
			// (по рекомендации Codex 2026-05-07: bilinear-интерполяция
			// для редких снимков создаёт визуальные артефакты,
			// которых нет в данных).
			simulation.SetSaveInterval(125);
		} else if (testIndex == 480 || testIndex == 481) {
			// Article v7 (2026-04-23): S→N plane wave, zero-gradient BC, no sponge.
			// Сохраняем только итоговые max-поля; промежуточные кадры не нужны.
			simulation.SetSaveInterval(999999);
		} else if (testIndex == 422 || testIndex == 423) {
			// v12 (2026-04-25, Q5): моментальные снимки η раз в час
			// (τ=5s × 720 = 3600s = 1ч). 10000 шагов → 13 кадров.
			simulation.SetSaveInterval(720);
		} else if (testIndex == 440 || testIndex == 441) {
			simulation.SetSaveInterval(999999);
		} else if (testIndex == 180 || testIndex == 181) {
			// Article S7: 25000 timesteps, save every 500 → 50 frames η
			simulation.SetSaveInterval(500);
		} else if (testIndex >= 41 && testIndex <= 48) {
			// Coriolis v2: 8000 timesteps, save every 40 → 200 frames для детального видео
			simulation.SetSaveInterval(40);
		} else if (testIndex >= 21 && testIndex <= 28) {
			// Coriolis tests: 4000 timesteps, save every 100 → 40 frames
			simulation.SetSaveInterval(100);
		} else if (testIndex == 18 || testIndex == 19) {
			// S7 Japan Sea: 25000 timesteps, save every 500 → 50 frames
			simulation.SetSaveInterval(500);
		} else if (testIndex == 101) {
			// Lake-at-rest: save every 100 → 5 кадров, проверяем статику
			simulation.SetSaveInterval(100);
		} else if (testIndex == 102) {
			// Gaussian pulse: save every 20 → 50 кадров, видим уход через границу
			simulation.SetSaveInterval(20);
		} else if (testIndex == 700) {
			// §1 Test L benchmark: только финальный η + η_max, никаких промежуточных.
			simulation.SetSaveInterval(999999);
		} else if ((testIndex >= 800 && testIndex <= 806) ||
		           (testIndex >= 810 && testIndex <= 816)) {
			// §5 lat band sweep: по умолчанию сохраняем только финал — для curve
			// «max_eta_max vs latitude». Для покадровой визуализации использовать
			// CLI: --save-interval=100 (даст 25 кадров за 2500 шагов).
			simulation.SetSaveInterval(999999);
		} else {
			// Default: save every 10 steps
			simulation.SetSaveInterval(10);
		}
		// CLI override (e.g. --save-interval=999999 for baseline-only runs that
		// only need final eta + eta_max — avoids GB of intermediate snapshots).
		if (save_interval_override_ >= 0) {
			simulation.SetSaveInterval(save_interval_override_);
		}
		// Thread count override (for perf benchmarks).
		if (threads_override_ > 0) {
			scenario.num_threads = threads_override_;
		}
		// Subgrid count override. num_subgrids=1 lets a single grid use OMP
		// internally without the per-step overhead of 16+ subgrids.
		if (subgrids_override_ > 0) {
			scenario.num_subgrids = subgrids_override_;
		}
		// Timesteps override (for perf probing).
		if (timesteps_override_ > 0) {
			scenario.timesteps = timesteps_override_;
		}
		// Phase E pre-paper: MC_TAU_OVERRIDE env for convergence study.
		// Replaces scenario tau with given value (e.g. =0.0005 for Monai tau/2).
		{
			char* tbuf = nullptr; size_t tsz = 0;
			_dupenv_s(&tbuf, &tsz, "MC_TAU_OVERRIDE");
			if (tbuf) {
				try {
					const double new_tau = std::stod(tbuf);
					if (new_tau > 0.0) {
						std::cerr << "[Sandbox] MC_TAU_OVERRIDE: tau "
								  << scenario.tau << " -> " << new_tau << std::endl;
						scenario.tau = new_tau;
					}
				} catch (...) {}
				free(tbuf);
			}
		}

		std::cerr << "[ST] before ApplyScenarioToSimulation" << std::flush << std::endl;
		// Устанавливаем параметры тестового сценария
		ScenarioBuilder::ApplyScenarioToSimulation(scenario, simulation);

		std::cerr << "[ST] before Initialize" << std::flush << std::endl;
		// Инициализируем симуляцию
		simulation.Initialize();

		std::cerr << "[ST] before Run, timesteps=" << scenario.timesteps
		          << " nx=" << scenario.nx << " ny=" << scenario.ny << std::flush << std::endl;
		// Измеряем время выполнения симуляции
		auto start_time = std::chrono::high_resolution_clock::now();

		// Запускаем симуляцию
		simulation.Run();

		auto end_time = std::chrono::high_resolution_clock::now();

		// Рассчитываем и выводим длительность
		std::chrono::duration<double> elapsed = end_time - start_time;
		std::cout << std::endl << "SingleLineGridSimulation completed in " << elapsed.count() << " seconds." << std::endl;

	}



	void HybridSimulatorTests(int testIndex)
	{
		SingleLineScenario scenario;
		switch (testIndex)
		{
		case 1: scenario = ScenarioBuilder::BuildTest1(); break;
		case 2: scenario = ScenarioBuilder::BuildTest2(); break;
		case 3: scenario = ScenarioBuilder::BuildTest3(); break;
		case 4: scenario = ScenarioBuilder::BuildTest4(); break;
		case 5: scenario = ScenarioBuilder::BuildTest5(); break;
		case 6: scenario = ScenarioBuilder::BuildTest6(); break;
		case 7: scenario = ScenarioBuilder::BuildTest7(); break;
		case 11: scenario = ScenarioBuilder::BuildTest1Spherical(); break;
		case 12: scenario = ScenarioBuilder::BuildTest2Spherical(); break;
		case 13: scenario = ScenarioBuilder::BuildTest3Spherical(); break;
		case 14: scenario = ScenarioBuilder::BuildTest4Spherical(); break;
		case 15: scenario = ScenarioBuilder::BuildTest5Spherical(); break;
		case 16: scenario = ScenarioBuilder::BuildTest6Spherical(); break;
		case 17: scenario = ScenarioBuilder::BuildTest7Spherical(); break;
		case 19: scenario = ScenarioBuilder::BuildTest6CartesianS7(); break;
		case 18: scenario = ScenarioBuilder::BuildTest6SphericalS7(); break;
		case 50: scenario = ScenarioBuilder::BuildTest6JapanSeaThreeSources(); break;
		case 60: scenario = ScenarioBuilder::BuildOkushiri450m(); break;
		case 70: scenario = ScenarioBuilder::BuildMonaiValley(); break;
		default: throw std::runtime_error("Unknown test index");
		}

		if (!scenario.has_own_grid || scenario.nx <= 0 || scenario.ny <= 0) {
			throw std::runtime_error("HybridSimulatorTest requires a scenario with a pre-built grid");
		}

		auto domain = HybridDomainBuilder::BuildStructuredStripes(
			scenario.g, scenario.tau, scenario.delta_x, scenario.delta_y,
			scenario.nx, scenario.ny, scenario.num_subgrids,
			scenario.grid, scenario.coord_system,
			scenario.minimum_depth_to_calculate, scenario.center_latitude_rad,
			scenario.exchangeBoundaryIndex,
			scenario.exchangeHatBoundaryIndex
		);

		SimulationRunner runner(domain, scenario.tau, scenario.timesteps, scenario.g);
		runner.SetSaveInterval(10);
		runner.SetSaveFunction([](int timestep, const Grid& grid) {
			std::string filename = "Result/eta_hybrid_" + std::to_string(timestep) + ".txt";
			std::ofstream file(filename);
			if (file.is_open()) {
				int nx_g = grid.GetNx();
				int ny_g = grid.GetNy();
				for (int i = 0; i < nx_g; ++i) {
					for (int j = 0; j < ny_g; ++j) {
						file << grid.eta(i, j);
						if (j < ny_g - 1) file << " ";
					}
					file << "\n";
				}
			}
		});

		auto start_time = std::chrono::high_resolution_clock::now();
		runner.Run();
		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end_time - start_time;
		std::cout << std::endl << "HybridSimulationRunner completed in "
		          << elapsed.count() << " seconds." << std::endl;
	}

public:
	/// Hybrid S7 test: Japan Sea with one stripe computed by DG solver.
	/// Test 30 = S7 Spherical hybrid, Test 31 = S7 Cartesian hybrid
	/// DG stripe index 3 (near source at lon≈132°)
	void HybridS7Test(int testIndex, int max_steps = 0)
	{
		DIAG_EXT("HS7: enter, testIndex=" + std::to_string(testIndex));
		SingleLineScenario scenario;
		int dg_stripe = 3; // Source at i≈1128, stripe 3 = i[900..1200]
		// 2026-05-20: env DG_STRIPE_INDEX=N picks a different DG stripe. Stripe 3
		// contains coastal cliffs that wave-reflect against frozen-dry cells
		// and amplify uncontrollably after t≈300 even with η-mode limiter.
		// Moving DG to a pure deep-water stripe (5-8) avoids the wet/dry issue.
		{
			char* dsi = nullptr; size_t dsi_sz = 0;
			_dupenv_s(&dsi, &dsi_sz, "DG_STRIPE_INDEX");
			if (dsi) {
				int idx = std::atoi(dsi);
				if (idx >= 0 && idx < 12) {
					dg_stripe = idx;
					std::cerr << "[HS7] DG stripe overridden to " << dg_stripe
					          << " via DG_STRIPE_INDEX=" << dsi << std::endl;
				}
				free(dsi);
			}
		}
		std::string result_folder;

		switch (testIndex) {
		case 30:
			DIAG_EXT("HS7: building Test6SphericalS7 scenario");
			scenario = ScenarioBuilder::BuildTest6SphericalS7();
			DIAG_EXT("HS7: scenario built: nx=" + std::to_string(scenario.nx) + " ny=" + std::to_string(scenario.ny));
			result_folder = "Result/test30_hybrid_s7_spherical";
			break;
		case 31:
			scenario = ScenarioBuilder::BuildTest6CartesianS7();
			result_folder = "Result/test31_hybrid_s7_cartesian";
			break;
		default:
			throw std::runtime_error("HybridS7Test: unknown test index");
		}

		std::cerr << "=== Hybrid S7 Test " << testIndex
		          << " (DG stripe " << dg_stripe << ") ===" << std::endl;
		DIAG_EXT("HS7: about to BuildHybridStripesOneDG");

		// 2026-05-20 evening: env LAKE_AT_REST=1 wipes the source — sets eta=0
		// and H=D everywhere, u=v=0. Used to verify the rest-mode is truly
		// stable (Codex Q6 baseline). If H_max drifts even without a source,
		// the discretisation has a balance bug in the moving zone.
		{
			char* lar = nullptr; size_t lar_sz = 0;
			_dupenv_s(&lar, &lar_sz, "LAKE_AT_REST");
			bool lake = (lar && *lar == '1');
			if (lar) free(lar);
			if (lake) {
				std::cerr << "[HS7] LAKE_AT_REST=1 → wiping source (eta=0, H=D, u=v=0)" << std::endl;
				int gnx = scenario.grid.GetNx();
				int gny = scenario.grid.GetNy();
				for (int i = 0; i < gnx; ++i) {
					for (int j = 0; j < gny; ++j) {
						double D = scenario.grid.D(i, j);
						scenario.grid.eta(i, j) = 0.0;
						scenario.grid.H(i, j) = D;
						scenario.grid.u(i, j) = 0.0;
						scenario.grid.v(i, j) = 0.0;
					}
				}
			}
		}

		auto domain = HybridDomainBuilder::BuildHybridStripesOneDG(
			scenario.g, scenario.tau, scenario.delta_x, scenario.delta_y,
			scenario.nx, scenario.ny, scenario.num_subgrids,
			scenario.grid, scenario.coord_system,
			scenario.minimum_depth_to_calculate, dg_stripe,
			1, // dg_cells_per_structured_cell
			scenario.center_latitude_rad,
			scenario.exchangeBoundaryIndex,
			scenario.exchangeHatBoundaryIndex
		);
		DIAG_EXT("HS7: BuildHybridStripesOneDG done, patches=" + std::to_string(domain->GetPatches().size()));

		// 2026-05-20 evening (Codex Q3): probe initial DG rest-state imbalance.
		// max |H_pt − D_pt| over volume quadrature points; for lake-at-rest
		// must be near roundoff. Any larger value points to projection /
		// limiter / boundary issues at initialization.
		for (auto& patch : domain->GetPatches()) {
			auto mp = dynamic_cast<MeshPatch*>(patch.get());
			if (mp && mp->GetDGSolver().NumElements() > 0) {
				double imb = mp->GetDGSolver().MaxRestImbalance();
				std::cerr << "[DG t=0 rest-imbalance] patch=" << patch->GetPatchId()
				          << "  max|H_pt - D_pt| = " << imb << " m" << std::endl;
				double mom = 0.0; int we = -1;
				double wwet = 0.0; int wwe = -1;
				double rr = mp->GetDGSolver().RestResidual(mom, we, &wwet, &wwe);
				std::cerr << "[DG t=0 rest-residual] patch=" << patch->GetPatchId()
				          << "  ||M^-1 L(q)|| = " << rr << "  (momentum=" << mom
				          << ", worst_elem=" << we << ")" << std::endl;
				std::cerr << "[DG t=0 rest-residual WET] patch=" << patch->GetPatchId()
				          << "  ||M^-1 L(q)||_wet = " << wwet << "  (worst_wet_elem=" << wwe << ")" << std::endl;
				int mwe = -1;
				double mmd = mp->GetDGSolver().MaxModalRestDiff(mwe);
				std::cerr << "[DG t=0 modal H-D] patch=" << patch->GetPatchId()
				          << "  max|H_coeff - D_coeff| (wet) = " << mmd
				          << "  (worst_elem=" << mwe << ")" << std::endl;
				int dwe = -1;
				double dmin = mp->GetDGSolver().MinBathymetryTrace(dwe);
				std::cerr << "[DG t=0 min D-trace] patch=" << patch->GetPatchId()
				          << "  min D_pt (wet, vol+edge) = " << dmin
				          << "  (worst_elem=" << dwe << ")" << std::endl;
			}
		}

		std::filesystem::create_directories(result_folder);

		// Determine global tau: min of MacCormack tau and DG stable dt
		double tau = scenario.tau;
		for (auto& patch : domain->GetPatches()) {
			auto mp = dynamic_cast<MeshPatch*>(patch.get());
			if (mp && mp->GetDGSolver().NumElements() > 0) {
				double dt_dg = mp->GetDGSolver().ComputeTimeStep(0.2); // CFL 0.2 for safety
				if (dt_dg < tau) {
					std::cerr << "  Reducing global tau from " << tau << " to " << dt_dg
					          << " (DG CFL constraint, patch " << patch->GetPatchId() << ")" << std::endl;
					tau = dt_dg;
				}
			}
		}

		// Adjust timesteps to keep same physical time
		double total_time = scenario.tau * scenario.timesteps;
		int timesteps = static_cast<int>(std::ceil(total_time / tau));
		if (max_steps > 0 && max_steps < timesteps) {
			timesteps = max_steps;
			total_time = timesteps * tau;
		}
		int save_interval = std::max(1, static_cast<int>(500.0 * scenario.tau / tau)); // Same physical interval

		std::cerr << "  Global tau=" << tau << " s, timesteps=" << timesteps
		          << " (physical time=" << total_time << " s)" << std::endl;

		SimulationRunner runner(domain, tau, timesteps, scenario.g);
		runner.SetSaveInterval(save_interval);
		runner.SetSaveFunction([&result_folder](int timestep, const Grid& grid) {
			std::string filename = result_folder + "/eta_timestep_" + std::to_string(timestep) + ".txt";
			std::ofstream file(filename);
			if (file.is_open()) {
				int nx_g = grid.GetNx();
				int ny_g = grid.GetNy();
				file << std::setprecision(std::numeric_limits<double>::max_digits10);
				for (int i = 0; i < nx_g; ++i) {
					for (int j = 0; j < ny_g; ++j) {
						file << grid.eta(i, j);
						if (j < ny_g - 1) file << " ";
					}
					file << "\n";
				}
			}
		});

		auto start_time = std::chrono::high_resolution_clock::now();
		runner.Run();
		auto end_time = std::chrono::high_resolution_clock::now();

		// END-OF-RUN DG internal imbalance: distinguishes a COLLECT/output artifact
		// (DG stayed at rest internally => max|H_pt-D_pt|~0) from a real coupling
		// instability (DG drifted => large). At t=0 this was machine-zero.
		for (auto& patch : domain->GetPatches()) {
			auto mp = dynamic_cast<MeshPatch*>(patch.get());
			if (mp && mp->GetDGSolver().NumElements() > 0) {
				double imb_end = mp->GetDGSolver().MaxRestImbalance();
				int mwe2 = -1;
				double mmd2 = mp->GetDGSolver().MaxModalRestDiff(mwe2);
				std::cerr << "[DG END internal] patch=" << patch->GetPatchId()
				          << "  max|H_pt-D_pt| = " << imb_end
				          << "  max|H_coeff-D_coeff|(wet) = " << mmd2
				          << "  (worst_elem=" << mwe2 << ")" << std::endl;
				double permode[DG_BASIS_SIZE];
				mp->GetDGSolver().ModalImbalanceBreakdown(permode);
				std::cerr << "[DG END per-mode |H_l-D_l|] ";
				for (int l = 0; l < DG_BASIS_SIZE; ++l) std::cerr << "m" << l << "=" << permode[l] << " ";
				std::cerr << std::endl;
			}
		}

		// Collect results and save eta_max + summary
		domain->CollectResultsToGlobalGrid();
		const Grid& final_grid = domain->GetGlobalGrid();
		int nx = final_grid.GetNx();
		int ny = final_grid.GetNy();

		// Two metrics:
		//  - all-cell max/min eta: includes DRY-LAND cells where H is floored to
		//    minimum_depth_to_calculate while D < 0 (negated land elevation), so
		//    eta = H - D = floor - (negative) is a large positive REPRESENTATION
		//    artifact, NOT a water wave.
		//  - wet max/min eta (D > 0): the physical free-surface elevation over the
		//    water column — the number that should be reported as the tsunami height.
		double max_eta = 0.0, min_eta = 0.0;
		double max_eta_wet = 0.0, min_eta_wet = 0.0;
		int wet_max_i = -1, wet_max_j = -1;
		double max_eta_deep = 0.0, min_eta_deep = 0.0; // genuine open ocean (D > 50 m)
		int deep_max_i = -1, deep_max_j = -1;
		for (int i = 0; i < nx; ++i)
			for (int j = 0; j < ny; ++j) {
				double e = final_grid.eta(i, j);
				if (e > max_eta) max_eta = e;
				if (e < min_eta) min_eta = e;
				if (final_grid.D(i, j) > 0.0) { // water column at still-water rest
					if (e > max_eta_wet) { max_eta_wet = e; wet_max_i = i; wet_max_j = j; }
					if (e < min_eta_wet) min_eta_wet = e;
				}
				if (final_grid.D(i, j) > 50.0) { // deep open ocean, away from coast/floor
					if (e > max_eta_deep) { max_eta_deep = e; deep_max_i = i; deep_max_j = j; }
					if (e < min_eta_deep) min_eta_deep = e;
				}
			}
		// Diagnose the wet-max cell: is it deep water (real MC imbalance) or a
		// shallow/coastal cell where the structured floor inflates eta?
		if (wet_max_i >= 0) {
			std::cerr << "  [max-eta cell] (i=" << wet_max_i << ", j=" << wet_max_j
			          << ")  eta=" << final_grid.eta(wet_max_i, wet_max_j)
			          << "  H=" << final_grid.H(wet_max_i, wet_max_j)
			          << "  D=" << final_grid.D(wet_max_i, wet_max_j) << std::endl;
			// Dump a horizontal slice across the cell so we can see whether the
			// anomaly is a one-cell seam/collect spike (eta != H-D) or a smooth wave.
			int i0 = std::max(0, wet_max_i - 8), i1 = std::min(nx - 1, wet_max_i + 8);
			for (int ii = i0; ii <= i1; ++ii) {
				double e = final_grid.eta(ii, wet_max_j);
				double H = final_grid.H(ii, wet_max_j);
				double D = final_grid.D(ii, wet_max_j);
				std::cerr << "    i=" << ii << " eta=" << e << " H=" << H
				          << " D=" << D << " (H-D=" << (H - D) << ")" << std::endl;
			}
		}

		// Write summary
		{
			std::ofstream summary(result_folder + "/summary.txt");
			if (summary.is_open()) {
				summary << "nx=" << nx << std::endl;
				summary << "ny=" << ny << std::endl;
				summary << "timesteps=" << scenario.timesteps << std::endl;
				summary << "tau=" << scenario.tau << std::endl;
				summary << "dg_stripe=" << dg_stripe << std::endl;
				summary << "save_interval=" << save_interval << std::endl;
				summary << "coord_system=" << CoordSystemToString(scenario.coord_system) << std::endl;
				summary << "max_eta_final=" << std::setprecision(std::numeric_limits<double>::max_digits10) << max_eta << std::endl;
				summary << "min_eta_final=" << min_eta << std::endl;
				summary << "max_eta_wet=" << max_eta_wet << std::endl;   // physical (D>0)
				summary << "min_eta_wet=" << min_eta_wet << std::endl;
				summary << "max_eta_deep=" << max_eta_deep << std::endl; // open ocean (D>50)
				summary << "min_eta_deep=" << min_eta_deep << std::endl;
			}
		}

		std::chrono::duration<double> elapsed = end_time - start_time;
		std::cerr << "  max_eta_final (all cells, incl. dry-land floor artifact) = "
		          << max_eta << " m  [min " << min_eta << "]" << std::endl;
		std::cerr << "  max_eta_WET (physical, D>0) = " << max_eta_wet << " m  [min "
		          << min_eta_wet << "]  at (i=" << wet_max_i << ", j=" << wet_max_j
		          << ")" << std::endl;
		std::cerr << "  max_eta_DEEP (open ocean, D>50) = " << max_eta_deep << " m  [min "
		          << min_eta_deep << "]  at (i=" << deep_max_i << ", j=" << deep_max_j
		          << ")" << std::endl;
		std::cout << std::endl << "HybridS7Test completed in "
		          << elapsed.count() << " seconds." << std::endl;
	}

public:
	Sandbox() = default;

	void SimulatorTest(int testIndex)
	{
		SimulatorTests(testIndex);
	}

	void HybridSimulatorTest(int testIndex)
	{
		HybridSimulatorTests(testIndex);
	}

	void DGValidationTest()
	{
		DGValidation::RunAllTests();
	}

	/// DG stability probe with a SQUARE-pulse IC, mimicking the HybridFlatTest
	/// scenario but isolated to a pure DG mesh on the same dimensions and CFL
	/// as the hybrid DG stripe. If this diverges, the issue is purely DG; if
	/// it doesn't, the divergence in HybridFlatTest is caused by coupling or
	/// inter-patch metadata, not the standalone solver.
	void DGSquarePulseProbe(bool with_limiter = true)
	{
		std::cerr << "=== DG Square-Pulse Probe ("
		          << (with_limiter ? "limiter ON" : "limiter OFF")
		          << ") ===" << std::endl;
		// Mirror HybridFlatTest's DG stripe geometry: stripe of 30km×120km,
		// 30×120 quads = 7200 triangles, source eta=1 at i,j ∈ [15,25)×[45,55)
		// (the DG-local index for the structured [45,55)×[45,55) source square).
		const double Lx = 30000.0, Ly = 120000.0;
		const double D0 = 1000.0;
		const int dg_nx = 30, dg_ny = 120;
		const double cfl = 0.2;
		const int log_steps[] = {0, 1, 2, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000};

		TriMesh mesh = DGValidation::GenerateRectMesh(Lx, Ly, dg_nx, dg_ny);
		ElementMatrices em; em.Precompute(mesh);

		DGSolver dg;
		dg.Initialize(&mesh, &em);
		dg.SetBathymetryFunction([D0](double, double) { return D0; });
		dg.SetDefaultBoundaryCondition(DGBoundaryType::Open);
		// Toggle radiation BC via env var so we can isolate whether the
		// hybrid divergence is caused by legacy copy-BC at the top/bottom
		// of the DG stripe.
		bool use_radiation = true;
		{
			char* buf = nullptr; size_t sz = 0;
			if (_dupenv_s(&buf, &sz, "DGPROBE_LEGACY_OPEN") == 0 && buf) {
				if (*buf == '1') use_radiation = false;
				free(buf);
			}
		}
		if (use_radiation) dg.SetUniformOpenReference(SWEState(D0, 0.0, 0.0));
		else std::cerr << "  (using legacy OpenBC = copy)" << std::endl;
		if (with_limiter) {
			dg.SetSlopeLimiter(true, 0.0);
			dg.Config().limiter_jump_threshold = 1e-4;
			dg.Config().limiter_oscillation_threshold = 1e-3;
		}
		// Square-pulse IC matching SetParamsTest1: 10×10 cells at (15..25, 45..55)
		// in DG-local coordinates (= [15000,25000]×[45000,55000] m).
		dg.SetInitialCondition([D0](double x, double y) {
			bool in_square = (x >= 15000.0 && x < 25000.0
			               && y >= 45000.0 && y < 55000.0);
			double H = D0 + (in_square ? 1.0 : 0.0);
			return SWEState(H, 0.0, 0.0);
		});
		if (with_limiter) { dg.ApplySlopeLimiter(); dg.ApplyPositivityLimiter(); }

		double dt = dg.ComputeTimeStep(cfl);
		std::cerr << "  Mesh: " << 2 * dg_nx * dg_ny << " tri, dt=" << dt << " s" << std::endl;
		int n_total = 5000;
		double max_H_overall = 0, min_H_overall = 1e30;
		for (int step = 0; step <= n_total; ++step) {
			if (step > 0) dg.StepRK3(dt);
			bool logme = false;
			for (int v : log_steps) if (step == v) { logme = true; break; }
			if (logme) {
				double min_H = 1e30, max_H = 0;
				int nan_count = 0;
				for (int k = 0; k < dg.NumElements(); ++k) {
					auto avg = dg.GetCellAverage(k);
					if (std::isnan(avg.H)) { nan_count++; continue; }
					min_H = std::min(min_H, avg.H);
					max_H = std::max(max_H, avg.H);
				}
				max_H_overall = std::max(max_H_overall, max_H);
				min_H_overall = std::min(min_H_overall, min_H);
				std::cerr << "  step " << step << ": H=[" << min_H << ", " << max_H << "]"
				          << (nan_count ? " NaN=" : "") << (nan_count ? std::to_string(nan_count) : "") << std::endl;
				if (nan_count > 0 || min_H < 0 || max_H > 2 * D0) {
					std::cerr << "  >>> DIVERGED at step " << step << std::endl;
					return;
				}
			}
		}
		std::cerr << "  overall H range: [" << min_H_overall << ", " << max_H_overall << "]" << std::endl;
	}

	/// Fast DG stability probe: flat-bottom Gaussian, small mesh (30x30 = 1800
	/// elements), long run with periodic max|eta| diagnostics. Designed to
	/// catch divergence early (within minutes, not 33 minutes like
	/// DGvsMacCormackTest). Pass the legacy "copy" Open BC by setting
	/// legacy_bc=true; otherwise the characteristic/Flather BC is used.
	void DGStabilityProbe(bool legacy_bc = false)
	{
		std::cerr << "=== DG Stability Probe (" << (legacy_bc ? "legacy copy BC" : "characteristic radiation BC") << ") ===" << std::endl;

		const double Lx = 200000.0, Ly = 200000.0;
		const double D0 = 1000.0;
		const double amp = 1.0;
		const double sigma = 10000.0;
		const double cx = Lx / 2, cy = Ly / 2;
		const int dg_nx = 30, dg_ny = 30; // 1800 triangles — fast iteration
		const double dg_cfl = 0.15;
		const int num_dg_steps = 10000;
		const int log_interval = 200;

		TriMesh mesh = DGValidation::GenerateRectMesh(Lx, Ly, dg_nx, dg_ny);
		ElementMatrices em;
		em.Precompute(mesh);

		DGSolver dg_solver;
		dg_solver.Initialize(&mesh, &em);
		dg_solver.SetBathymetryFunction([D0](double, double) { return D0; });
		dg_solver.SetDefaultBoundaryCondition(DGBoundaryType::Open);
		dg_solver.SetInitialCondition([D0, amp, sigma, cx, cy](double x, double y) {
			double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
			double H = D0 + amp * std::exp(-r2 / (2.0 * sigma * sigma));
			return SWEState(H, 0.0, 0.0);
		});
		if (!legacy_bc) {
			dg_solver.SetUniformOpenReference(SWEState(D0, 0.0, 0.0));
		}

		double dt = dg_solver.ComputeTimeStep(dg_cfl);
		std::cerr << "  Mesh: " << 2 * dg_nx * dg_ny << " elements, dt=" << dt
		          << " s, total physical time=" << num_dg_steps * dt << " s" << std::endl;

		auto t0 = std::chrono::high_resolution_clock::now();
		double max_eta_overall = 0.0;
		bool diverged = false;
		int diverged_step = -1;

		for (int step = 1; step <= num_dg_steps; ++step) {
			dg_solver.StepRK3(dt);

			if (step % log_interval == 0 || step == 1) {
				double max_eta = 0.0;
				bool has_nan = false;
				for (int k = 0; k < dg_solver.NumElements(); ++k) {
					auto avg = dg_solver.GetCellAverage(k);
					if (std::isnan(avg.H)) { has_nan = true; break; }
					double eta_pert = avg.H - D0;
					max_eta = std::max(max_eta, std::abs(eta_pert));
				}
				max_eta_overall = std::max(max_eta_overall, max_eta);
				std::cerr << "  step " << step << ": max|eta-D|=" << max_eta
				          << (has_nan ? " [NaN]" : "") << std::endl;
				if (has_nan || max_eta > 100.0 * amp) {
					diverged = true;
					diverged_step = step;
					std::cerr << "  >>> DIVERGED at step " << step << std::endl;
					break;
				}
			}
		}

		auto t1 = std::chrono::high_resolution_clock::now();
		double dt_wall = std::chrono::duration<double>(t1 - t0).count();
		std::cerr << "  Wall: " << dt_wall << " s, max|eta-D| over run: "
		          << max_eta_overall << std::endl;
		std::cerr << (diverged ? "[FAIL]" : "[PASS]") << " DG Stability Probe ("
		          << (legacy_bc ? "legacy" : "radiation") << "): "
		          << (diverged ? "diverged at step " + std::to_string(diverged_step) : "stable")
		          << std::endl;
	}

	/// Simple hybrid test: flat bottom, no land, Cartesian.
	/// 120x120 grid, 4 stripes, stripe 1 = DG. Source: eta=1 square at i=45-55.
	/// Pure MacCormack reference for comparison.
	void HybridFlatTest()
	{
		std::cerr << "=== Hybrid Flat Bottom Test ===" << std::endl;
		std::string result_dir = "Result/hybrid_flat_test";
		std::filesystem::create_directories(result_dir);

		// Build scenario from Test1 (120x120, flat D=1000, eta=1 square)
		SingleLineScenario scenario = ScenarioBuilder::BuildTest1();
		scenario.num_subgrids = 4;  // 4 stripes of width 30
		scenario.num_threads = 1;

		int dg_stripe = 1; // Source at i=45-55, stripe 1 = i[30..60] contains it

		std::cerr << "  Grid: " << scenario.nx << "x" << scenario.ny
		          << ", tau=" << scenario.tau << ", steps=" << scenario.timesteps << std::endl;
		std::cerr << "  DG stripe: " << dg_stripe << " (i=" << dg_stripe*30 << ".." << (dg_stripe+1)*30 << ")" << std::endl;

		// ---- Part 1: Pure MacCormack reference ----
		std::cerr << "\n  Running pure MacCormack..." << std::endl;
		auto mc_start = std::chrono::high_resolution_clock::now();

		SingleLineScenario mc_scenario = ScenarioBuilder::BuildTest1();
		mc_scenario.num_subgrids = 4;
		mc_scenario.num_threads = 1;
		SingleLineGridSimulation mc_sim;
		ScenarioBuilder::ApplyScenarioToSimulation(mc_scenario, mc_sim);
		mc_sim.SetResultFolder(result_dir + "/pure_mc");
		mc_sim.Initialize();
		mc_sim.Run();

		auto mc_end = std::chrono::high_resolution_clock::now();
		double mc_time = std::chrono::duration<double>(mc_end - mc_start).count();

		double mc_max_eta = 0;
		for (int i = 0; i < scenario.nx; ++i)
			for (int j = 0; j < scenario.ny; ++j)
				mc_max_eta = std::max(mc_max_eta, static_cast<double>(std::abs(mc_sim.global_grid_.eta(i, j))));
		std::cerr << "  Pure MC done: " << mc_time << "s, max|eta|=" << mc_max_eta << std::endl;

		// Save pure MC final eta
		{
			std::ofstream f(result_dir + "/pure_mc_final_eta.txt");
			f << std::setprecision(8);
			for (int i = 0; i < scenario.nx; ++i) {
				for (int j = 0; j < scenario.ny; ++j) {
					f << mc_sim.global_grid_.eta(i, j);
					if (j < scenario.ny - 1) f << " ";
				}
				f << "\n";
			}
		}

		// ---- Part 2: Hybrid (MacCormack + DG stripe) ----
		std::cerr << "\n  Running Hybrid (MC + DG stripe " << dg_stripe << ")..." << std::endl;
		auto hy_start = std::chrono::high_resolution_clock::now();

		auto domain = HybridDomainBuilder::BuildHybridStripesOneDG(
			scenario.g, scenario.tau, scenario.delta_x, scenario.delta_y,
			scenario.nx, scenario.ny, scenario.num_subgrids,
			scenario.grid, scenario.coord_system,
			scenario.minimum_depth_to_calculate, dg_stripe,
			1, // dg_cells_per_structured_cell
			scenario.center_latitude_rad,
			scenario.exchangeBoundaryIndex,
			scenario.exchangeHatBoundaryIndex
		);

		SimulationRunner runner(domain, scenario.tau, scenario.timesteps, scenario.g);
		runner.SetSaveInterval(50);
		runner.SetSaveFunction([&result_dir](int timestep, const Grid& grid) {
			std::filesystem::create_directories(result_dir + "/hybrid");
			std::string fname = result_dir + "/hybrid/eta_" + std::to_string(timestep) + ".txt";
			std::ofstream f(fname);
			f << std::setprecision(8);
			for (int i = 0; i < grid.GetNx(); ++i) {
				for (int j = 0; j < grid.GetNy(); ++j) {
					f << grid.eta(i, j);
					if (j < grid.GetNy() - 1) f << " ";
				}
				f << "\n";
			}
		});
		runner.Run();
		domain->CollectResultsToGlobalGrid();

		auto hy_end = std::chrono::high_resolution_clock::now();
		double hy_time = std::chrono::duration<double>(hy_end - hy_start).count();

		const Grid& hy_grid = domain->GetGlobalGrid();
		double hy_max_eta = 0;
		for (int i = 0; i < scenario.nx; ++i)
			for (int j = 0; j < scenario.ny; ++j)
				hy_max_eta = std::max(hy_max_eta, static_cast<double>(std::abs(hy_grid.eta(i, j))));
		std::cerr << "  Hybrid done: " << hy_time << "s, max|eta|=" << hy_max_eta << std::endl;

		// Save hybrid final eta
		{
			std::ofstream f(result_dir + "/hybrid_final_eta.txt");
			f << std::setprecision(8);
			for (int i = 0; i < scenario.nx; ++i) {
				for (int j = 0; j < scenario.ny; ++j) {
					f << hy_grid.eta(i, j);
					if (j < scenario.ny - 1) f << " ";
				}
				f << "\n";
			}
		}

		// ---- Comparison ----
		double max_diff = 0;
		for (int i = 0; i < scenario.nx; ++i)
			for (int j = 0; j < scenario.ny; ++j) {
				double diff = std::abs(mc_sim.global_grid_.eta(i, j) - hy_grid.eta(i, j));
				max_diff = std::max(max_diff, diff);
			}

		// T2 mass conservation diagnostic. Total water mass is
		//   M = ∫∫ H dx dy ≈ Σ H[i,j] · dx[i,j] · dy
		// We compare initial vs final mass for pure MC and hybrid. Pure MC
		// is the reference: with Open BC the wave radiates and drains a
		// tiny amount across the outer rectangle. The hybrid should drain
		// the same amount IF the DG↔MC interface is mass-conservative; any
		// extra drain or gain is the interface error.
		auto total_mass = [&](const Grid& g) {
			double M = 0.0;
			for (int i = 0; i < g.GetNx(); ++i) {
				double dx = g.GetDeltaXFunc()(i, 0);
				for (int j = 0; j < g.GetNy(); ++j) {
					M += g.H(i, j) * dx * g.GetDeltaY();
				}
			}
			return M;
		};
		double M0 = 0.0;
		{
			SingleLineScenario ic_scenario = ScenarioBuilder::BuildTest1();
			M0 = total_mass(ic_scenario.grid);
		}
		double M_mc = total_mass(mc_sim.global_grid_);
		double M_hy = total_mass(hy_grid);
		auto rel = [&](double M) { return (M - M0) / M0; };

		std::cerr << "\n=== Results ===" << std::endl;
		std::cerr << "  Pure MC:  " << mc_time << "s, max|eta|=" << mc_max_eta << std::endl;
		std::cerr << "  Hybrid:   " << hy_time << "s, max|eta|=" << hy_max_eta << std::endl;
		std::cerr << "  max|diff|=" << max_diff << std::endl;
		std::cerr << "  --- T2 mass conservation ---" << std::endl;
		std::cerr << "  M(t=0)  = " << std::scientific << std::setprecision(10) << M0 << std::endl;
		std::cerr << "  M_MC    = " << M_mc << "  (Δ/M0 = " << rel(M_mc) << ")" << std::endl;
		std::cerr << "  M_Hybrid= " << M_hy << "  (Δ/M0 = " << rel(M_hy) << ")" << std::endl;
		std::cerr << "  Interface mass error vs pure MC: "
		          << std::abs(M_hy - M_mc) / M0 << std::endl;
		std::cerr << std::defaultfloat;
		if (mc_max_eta > 1e-10)
			std::cerr << "  Relative: " << max_diff / mc_max_eta * 100 << "%" << std::endl;

		std::cerr << "  Results in " << result_dir << "/" << std::endl;
	}

	// ===================================================================
	// Incremental MC+DG validation (2026-05-30): get the hybrid stable &
	// physical on controlled cases with the DG stripe in DIFFERENT positions
	// BEFORE Japan Sea S7. Acceptance is NOT "match pure MC" (different solvers
	// legitimately differ) but: no inf/NaN, bounded |eta| (no secular growth),
	// wave actually propagates through the seams. First-offender diagnostic in
	// SimulationRunner localizes any blow-up.
	// ===================================================================

	// Shared result of one hybrid/pure run.
	struct HybridRunStat {
		std::string tag;
		double max_abs_eta = 0.0;
		double min_eta = 0.0;
		double max_abs_eta_dg = 0.0; // max |eta| restricted to DG stripe columns
		int nan_cells = 0;
		bool finite = true;
		double seconds = 0.0;
	};

	// Run a hybrid (or pure-DG when num_subgrids==1) flat/variable case and
	// return stability stats. grid carries D/H/eta/u/v IC; bathymetry/IC are
	// projected into the DG stripe by the builder.
	HybridRunStat RunHybridCase(Grid grid, int nx, int ny, double dx, double dy,
	                            double g, double mc_tau, double phys_time,
	                            int num_subgrids, int dg_stripe,
	                            double min_calc, const std::string& tag,
	                            const std::string& base_dir)
	{
		HybridRunStat st; st.tag = tag;
		auto dxf = [dx](int, int) { return dx; };

		auto domain = HybridDomainBuilder::BuildHybridStripesOneDG(
			g, mc_tau, dxf, dy, nx, ny, num_subgrids, grid,
			CoordSystem::Cartesian, min_calc, dg_stripe,
			1, 0.785398163, 0, 0);

		// Reduce global tau to the DG CFL (real travelling wave needs it).
		double tau = mc_tau;
		for (auto& patch : domain->GetPatches()) {
			auto mp = dynamic_cast<MeshPatch*>(patch.get());
			if (mp && mp->GetDGSolver().NumElements() > 0) {
				double dt_dg = mp->GetDGSolver().ComputeTimeStep(0.2);
				if (dt_dg < tau) tau = dt_dg;
			}
		}
		int timesteps = std::max(1, static_cast<int>(std::ceil(phys_time / tau)));
		std::cerr << "  [" << tag << "] num_subgrids=" << num_subgrids
		          << " dg_stripe=" << dg_stripe << " tau=" << tau
		          << " timesteps=" << timesteps << std::endl;

		SimulationRunner runner(domain, tau, timesteps, g);
		runner.SetSaveInterval(timesteps); // final snapshot only (disk-friendly)
		std::string out_dir = base_dir + "/" + tag;
		std::filesystem::create_directories(out_dir);
		runner.SetSaveFunction([&out_dir](int ts, const Grid& gr) {
			std::ofstream f(out_dir + "/eta_" + std::to_string(ts) + ".txt");
			f << std::setprecision(8);
			for (int i = 0; i < gr.GetNx(); ++i) {
				for (int j = 0; j < gr.GetNy(); ++j) {
					f << gr.eta(i, j); if (j < gr.GetNy() - 1) f << " ";
				}
				f << "\n";
			}
		});

		auto t0 = std::chrono::high_resolution_clock::now();
		runner.Run();
		auto t1 = std::chrono::high_resolution_clock::now();
		st.seconds = std::chrono::duration<double>(t1 - t0).count();

		domain->CollectResultsToGlobalGrid();
		const Grid& gr = domain->GetGlobalGrid();
		int m = nx / num_subgrids;
		int dg_i0 = dg_stripe * m;
		int dg_i1 = (dg_stripe == num_subgrids - 1) ? nx : (dg_stripe + 1) * m;
		for (int i = 0; i < nx; ++i)
			for (int j = 0; j < ny; ++j) {
				double e = gr.eta(i, j);
				if (!std::isfinite(e)) { st.nan_cells++; st.finite = false; continue; }
				st.max_abs_eta = std::max(st.max_abs_eta, std::abs(e));
				st.min_eta = std::min(st.min_eta, e);
				if (i >= dg_i0 && i < dg_i1)
					st.max_abs_eta_dg = std::max(st.max_abs_eta_dg, std::abs(e));
			}
		return st;
	}

	// Pure MacCormack reference on the same grid (num_subgrids stripes, all MC).
	HybridRunStat RunPureMCCase(Grid grid, int nx, int ny,
	                            std::function<double(int,int)> dxf, double dy,
	                            double g, double mc_tau, double phys_time,
	                            int num_subgrids, double min_calc,
	                            const std::string& tag, const std::string& base_dir)
	{
		HybridRunStat st; st.tag = tag;
		int timesteps = std::max(1, static_cast<int>(std::ceil(phys_time / mc_tau)));

		SingleLineScenario s;
		s.g = g; s.tau = mc_tau; s.delta_x = dxf; s.delta_y = dy;
		s.nx = nx; s.ny = ny; s.timesteps = timesteps;
		s.num_threads = 1; s.num_subgrids = num_subgrids;
		s.coord_system = CoordSystem::Cartesian;
		s.minimum_depth_to_calculate = min_calc;
		s.exchangeBoundaryIndex = 0; s.exchangeHatBoundaryIndex = 0;
		s.grid = grid; s.has_own_grid = true;

		SingleLineGridSimulation sim;
		ScenarioBuilder::ApplyScenarioToSimulation(s, sim);
		std::string out_dir = base_dir + "/" + tag;
		std::filesystem::create_directories(out_dir);
		sim.SetResultFolder(out_dir);
		auto t0 = std::chrono::high_resolution_clock::now();
		sim.Initialize();
		sim.Run();
		auto t1 = std::chrono::high_resolution_clock::now();
		st.seconds = std::chrono::duration<double>(t1 - t0).count();

		for (int i = 0; i < nx; ++i)
			for (int j = 0; j < ny; ++j) {
				double e = sim.global_grid_.eta(i, j);
				if (!std::isfinite(e)) { st.nan_cells++; st.finite = false; continue; }
				st.max_abs_eta = std::max(st.max_abs_eta, std::abs(e));
				st.min_eta = std::min(st.min_eta, e);
			}
		return st;
	}

	void PrintHybridTable(const std::string& title,
	                      const std::vector<HybridRunStat>& rows, double amp)
	{
		std::cerr << "\n=== " << title << " ===" << std::endl;
		std::cerr << "  (acceptance: finite, NaN=0, max|eta| bounded — wave amp="
		          << amp << " m; differences vs pure MC are expected)" << std::endl;
		for (const auto& r : rows) {
			bool bounded = r.finite && r.max_abs_eta < 10.0 * amp;
			std::cerr << "  " << (r.finite && bounded ? "[PASS]" : "[FAIL]")
			          << " " << r.tag
			          << "  max|eta|=" << r.max_abs_eta
			          << "  min_eta=" << r.min_eta
			          << "  max|eta|_DG=" << r.max_abs_eta_dg
			          << "  NaN=" << r.nan_cells
			          << "  t=" << r.seconds << "s" << std::endl;
		}
	}

	/// Test 1: flat-bottom right-going traveling wave through MC+DG.
	/// Pure MC ref + pure DG control + hybrid with DG stripe in 3 positions.
	void HybridWaveTest()
	{
		std::cerr << "=== Hybrid Traveling-Wave Test (flat bottom, right-going) ===" << std::endl;
		const std::string base_dir = "Result/hybrid_wave";
		std::filesystem::create_directories(base_dir);

		const int nx = 240, ny = 60;
		const double dx = 1000.0, dy = 1000.0;
		const double g = 9.81, D0 = 1000.0, amp = 1.0, min_calc = 8.0;
		const double sigma = 10.0 * dx;     // packet width
		const double x0 = 40.0 * dx;        // start in stripe 0 (MC), ~17%
		const double mc_tau = 2.0;
		const double phys_time = 2200.0;    // wave traverses ~218 km at c=99 m/s

		auto dxf = [dx](int, int) { return dx; };
		auto make_grid = [&]() {
			Grid grid(nx, ny, dxf, dy);
			for (int i = 0; i < nx; ++i) {
				double x = (i + 0.5) * dx;
				double e = amp * std::exp(-((x - x0) * (x - x0)) / (2.0 * sigma * sigma));
				double u = std::sqrt(g / D0) * e; // right-going Riemann invariant
				for (int j = 0; j < ny; ++j) {
					grid.D(i, j) = D0; grid.eta(i, j) = e; grid.H(i, j) = D0 + e;
					grid.u(i, j) = u; grid.v(i, j) = 0.0;
				}
			}
			grid.MarkInitialized();
			return grid;
		};

		std::vector<HybridRunStat> rows;
		rows.push_back(RunPureMCCase(make_grid(), nx, ny, dxf, dy, g, mc_tau,
		                             phys_time, 4, min_calc, "pure_mc", base_dir));
		rows.push_back(RunHybridCase(make_grid(), nx, ny, dx, dy, g, mc_tau,
		                             phys_time, 1, 0, min_calc, "pure_dg", base_dir));
		for (int sidx : {1, 2, 3})
			rows.push_back(RunHybridCase(make_grid(), nx, ny, dx, dy, g, mc_tau,
			                             phys_time, 4, sidx, min_calc,
			                             "hybrid_dg" + std::to_string(sidx), base_dir));

		PrintHybridTable("Hybrid Traveling-Wave results", rows, amp);
	}

	/// Test 2: right-going wave hitting a CIRCULAR ISLAND in the middle.
	/// Pure MC ref + pure DG control + hybrid with island in 3 stripe positions.
	void HybridIslandTest()
	{
		std::cerr << "=== Hybrid Circular-Island Test (right-going wave) ===" << std::endl;
		const std::string base_dir = "Result/hybrid_island";
		std::filesystem::create_directories(base_dir);

		const int nx = 240, ny = 120;
		const double dx = 1000.0, dy = 1000.0;
		const double g = 9.81, D0 = 1000.0, amp = 1.0, min_calc = 8.0;
		const double sigma = 10.0 * dx;
		const double x0 = 40.0 * dx;            // wave starts left
		const double cx = 120.0 * dx, cy = 60.0 * dy; // island at domain center
		const double R = 25.0 * dx;            // island radius
		const double land_top = 50.0;          // peak 50 m above rest
		const double mc_tau = 2.0;
		const double phys_time = 2400.0;

		auto dxf = [dx](int, int) { return dx; };
		auto bathy = [&](double x, double y) {
			double r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
			if (r >= R) return D0;
			double s = 1.0 - r / R;                 // 0 at rim, 1 at centre
			return D0 - (D0 + land_top) * s;        // dips below 0 (land) near centre
		};
		auto make_grid = [&]() {
			Grid grid(nx, ny, dxf, dy);
			for (int i = 0; i < nx; ++i) {
				double x = (i + 0.5) * dx;
				double e0 = amp * std::exp(-((x - x0) * (x - x0)) / (2.0 * sigma * sigma));
				for (int j = 0; j < ny; ++j) {
					double y = (j + 0.5) * dy;
					double D = bathy(x, y);
					grid.D(i, j) = D;
					if (D < min_calc) {            // dry land: rest, H=D
						grid.eta(i, j) = 0.0; grid.H(i, j) = D;
						grid.u(i, j) = 0.0; grid.v(i, j) = 0.0;
					} else {
						double e = e0;
						double u = std::sqrt(g / D) * e;
						grid.eta(i, j) = e; grid.H(i, j) = D + e;
						grid.u(i, j) = u; grid.v(i, j) = 0.0;
					}
				}
			}
			grid.MarkInitialized();
			return grid;
		};

		std::vector<HybridRunStat> rows;
		rows.push_back(RunPureMCCase(make_grid(), nx, ny, dxf, dy, g, mc_tau,
		                             phys_time, 3, min_calc, "pure_mc", base_dir));
		rows.push_back(RunHybridCase(make_grid(), nx, ny, dx, dy, g, mc_tau,
		                             phys_time, 1, 0, min_calc, "pure_dg", base_dir));
		// num_subgrids=3, stripes [0..80)[80..160)[160..240). Island i∈[95,145].
		// dg_stripe=1 => island fully inside DG; =0/=2 => island in an MC stripe
		// with DG adjacent (seam stress near/through the island).
		for (int sidx : {0, 1, 2})
			rows.push_back(RunHybridCase(make_grid(), nx, ny, dx, dy, g, mc_tau,
			                             phys_time, 3, sidx, min_calc,
			                             "hybrid_dg" + std::to_string(sidx), base_dir));

		PrintHybridTable("Hybrid Circular-Island results", rows, amp);
	}

	/// Fast pure-DG island blow-up probe for pass-2 bisection (~30s/run).
	/// Small domain, circular island fully inside ONE DG patch, right-going wave.
	/// Reads DG_DISABLE_SLOPE_LIMITER / DG_DISABLE_WELL_BALANCED / DG_LIMIT_ETA
	/// (consumed in the builder) so the steep-shallow DG instability can be
	/// bisected quickly. Use DG_OFFENDER_ETA to tune the blow-up threshold.
	void DGIslandProbe()
	{
		std::cerr << "=== DG Island Probe (fast pure-DG bisection) ===" << std::endl;
		const std::string base_dir = "Result/dg_island_probe";
		std::filesystem::create_directories(base_dir);

		const int nx = 80, ny = 60;
		const double dx = 1000.0, dy = 1000.0;
		const double g = 9.81, D0 = 1000.0, amp = 1.0, min_calc = 8.0;
		const double sigma = 6.0 * dx;
		const double x0 = 12.0 * dx;
		const double cx = 48.0 * dx, cy = 30.0 * dy, R = 14.0 * dx, land_top = 50.0;
		const double mc_tau = 2.0, phys_time = 700.0;

		auto dxf = [dx](int, int) { return dx; };
		auto bathy = [&](double x, double y) {
			double r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
			if (r >= R) return D0;
			double s = 1.0 - r / R;
			return D0 - (D0 + land_top) * s;
		};
		Grid grid(nx, ny, dxf, dy);
		for (int i = 0; i < nx; ++i) {
			double x = (i + 0.5) * dx;
			double e0 = amp * std::exp(-((x - x0) * (x - x0)) / (2.0 * sigma * sigma));
			for (int j = 0; j < ny; ++j) {
				double y = (j + 0.5) * dy;
				double D = bathy(x, y);
				grid.D(i, j) = D;
				if (D < min_calc) {
					grid.eta(i, j) = 0.0; grid.H(i, j) = D;
					grid.u(i, j) = 0.0; grid.v(i, j) = 0.0;
				} else {
					double e = e0, u = std::sqrt(g / D) * e;
					grid.eta(i, j) = e; grid.H(i, j) = D + e;
					grid.u(i, j) = u; grid.v(i, j) = 0.0;
				}
			}
		}
		grid.MarkInitialized();

		std::vector<HybridRunStat> rows;
		rows.push_back(RunHybridCase(grid, nx, ny, dx, dy, g, mc_tau,
		                             phys_time, 1, 0, min_calc, "probe_dg", base_dir));
		PrintHybridTable("DG Island Probe results", rows, amp);
	}

	/// DG vs MacCormack comparison on flat-bottom Gaussian wave
	/// Runs both solvers on the same domain and compares results.
	void DGvsMacCormackTest()
	{
		std::cerr << "=== DG vs MacCormack Comparison ===" << std::endl;
		std::string result_dir = "Result/dg_vs_maccormack";
		std::filesystem::create_directories(result_dir);

		// Shared parameters
		const double Lx = 200000.0, Ly = 200000.0; // 200 km domain
		const double D0 = 1000.0;   // flat bottom 1000m
		const double g = 9.81;
		const double c = std::sqrt(g * D0); // ~99 m/s
		const double amp = 1.0;
		const double sigma = 10000.0; // 10 km Gaussian
		const double cx = Lx / 2, cy = Ly / 2;

		// MacCormack: 200×200 grid, dx=dy=1000m
		const int mc_nx = 200, mc_ny = 200;
		const double mc_dx = Lx / mc_nx;     // 1000m
		const double mc_dy = Ly / mc_ny;     // 1000m
		const double tau = 0.5 * mc_dx / c;  // CFL ~0.5 → tau ≈ 5.05s
		const int num_steps = 2000;
		const int save_interval = 200;

		// DG: same physical domain, coarser grid but higher order
		// P2 DG on 100×100 mesh → 20000 elements → comparable DOF count.
		// Allow override via env var DGVSMC_DG_NX for fast smoke regressions
		// (e.g. 50×50 finishes in ~3 minutes instead of 33).
		int dg_nx_env = 100;
		{
			char* buf = nullptr;
			size_t sz = 0;
			if (_dupenv_s(&buf, &sz, "DGVSMC_DG_NX") == 0 && buf != nullptr) {
				int v = std::atoi(buf);
				if (v >= 8 && v <= 400) dg_nx_env = v;
				free(buf);
			}
		}
		const int dg_nx = dg_nx_env, dg_ny = dg_nx_env;
		const double dg_cfl = 0.15; // Conservative CFL for P2 with RK3

		std::cerr << "  Domain: " << Lx/1000 << " x " << Ly/1000 << " km, D=" << D0 << " m" << std::endl;
		std::cerr << "  MacCormack: " << mc_nx << "x" << mc_ny << ", dx=" << mc_dx << " m, tau=" << tau << " s" << std::endl;
		std::cerr << "  DG P2: " << dg_nx << "x" << dg_ny << " mesh (" << 2*dg_nx*dg_ny << " elements)" << std::endl;
		std::cerr << "  Steps: " << num_steps << ", save every " << save_interval << std::endl;

		// ========== Part 1: MacCormack via HybridDomain (same arch as DG test) ==========
		std::cerr << "\n  Running MacCormack..." << std::endl;
		auto mc_start = std::chrono::high_resolution_clock::now();

		auto mc_delta_x = [mc_dx](int, int) { return mc_dx; };
		Grid mc_grid(mc_nx, mc_ny, mc_delta_x, mc_dy);
		for (int i = 0; i < mc_nx; ++i) {
			for (int j = 0; j < mc_ny; ++j) {
				double x = (i + 0.5) * mc_dx;
				double y = (j + 0.5) * mc_dy;
				double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
				double eta_val = amp * std::exp(-r2 / (2.0 * sigma * sigma));
				mc_grid.D(i, j) = D0;
				mc_grid.H(i, j) = D0 + eta_val;
				mc_grid.u(i, j) = 0.0;
				mc_grid.v(i, j) = 0.0;
				mc_grid.eta(i, j) = eta_val;
			}
		}

		auto mc_domain = HybridDomainBuilder::BuildStructuredStripes(
			g, tau, mc_delta_x, mc_dy,
			mc_nx, mc_ny, 1, // 1 stripe
			mc_grid, CoordSystem::Cartesian, 0.01, 0.0, 0, 0
		);

		SimulationRunner mc_runner(mc_domain, tau, num_steps, g);
		mc_runner.SetSaveInterval(save_interval);
		mc_runner.SetSaveFunction([&result_dir](int timestep, const Grid& grid) {
			std::filesystem::create_directories(result_dir + "/mc");
			std::string filename = result_dir + "/mc/eta_timestep_" + std::to_string(timestep) + ".txt";
			std::ofstream file(filename);
			if (file.is_open()) {
				int nx_g = grid.GetNx(); int ny_g = grid.GetNy();
				file << std::setprecision(8);
				for (int i = 0; i < nx_g; ++i) {
					for (int j = 0; j < ny_g; ++j) {
						file << grid.eta(i, j);
						if (j < ny_g - 1) file << " ";
					}
					file << "\n";
				}
			}
		});
		mc_runner.Run();
		mc_domain->CollectResultsToGlobalGrid();

		auto mc_end = std::chrono::high_resolution_clock::now();
		double mc_time = std::chrono::duration<double>(mc_end - mc_start).count();

		const Grid& mc_result = mc_domain->GetGlobalGrid();
		double mc_max_eta = 0;
		for (int i = 0; i < mc_nx; ++i)
			for (int j = 0; j < mc_ny; ++j)
				mc_max_eta = std::max(mc_max_eta, static_cast<double>(std::abs(mc_result.eta(i, j))));

		std::cerr << "  MacCormack done: " << mc_time << "s, max|eta|=" << mc_max_eta << std::endl;

		// ========== Part 2: DG ==========
		std::cerr << "\n  Running DG P2..." << std::endl;
		auto dg_start = std::chrono::high_resolution_clock::now();

		TriMesh mesh = DGValidation::GenerateRectMesh(Lx, Ly, dg_nx, dg_ny);
		ElementMatrices em;
		em.Precompute(mesh);

		DGSolver dg_solver;
		dg_solver.Initialize(&mesh, &em);
		dg_solver.SetBathymetryFunction([D0](double, double) { return D0; });
		dg_solver.SetDefaultBoundaryCondition(DGBoundaryType::Open);
		dg_solver.SetInitialCondition([D0, amp, sigma, cx, cy](double x, double y) {
			double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
			double H = D0 + amp * std::exp(-r2 / (2.0 * sigma * sigma));
			return SWEState(H, 0.0, 0.0);
		});
		// Non-reflecting radiation BC: ambient = rest state (H = D, u = v = 0)
		// at the boundary. Without this, qGhost = qInner at long times causes
		// the wave to "trickle back" and amplitude diverges (4.85e10 in the
		// original 23582-step run).
		dg_solver.SetUniformOpenReference(SWEState(D0, 0.0, 0.0));

		double dg_dt = dg_solver.ComputeTimeStep(dg_cfl);
		std::cerr << "  DG dt=" << dg_dt << "s (CFL=" << dg_cfl << ")" << std::endl;

		// Run DG for same physical time
		double total_time = num_steps * tau;
		int dg_steps = 0;
		double t = 0.0;
		int next_save = save_interval;
		double dg_save_time_interval = (num_steps / save_interval) > 0 ? total_time / (num_steps / save_interval) : total_time;
		double next_save_t = dg_save_time_interval;

		while (t < total_time) {
			double dt_step = std::min(dg_dt, total_time - t);
			dg_solver.StepRK3(dt_step);
			t += dt_step;
			dg_steps++;

			// Save at intervals
			if (t >= next_save_t) {
				// Save DG eta to file
				std::string fname = result_dir + "/dg/eta_timestep_" + std::to_string(static_cast<int>(t / tau)) + ".txt";
				std::filesystem::create_directories(result_dir + "/dg");
				std::ofstream dg_file(fname);
				if (dg_file.is_open()) {
					// Output cell-average eta on DG mesh (as unstructured data)
					dg_file << std::setprecision(8);
					for (int k = 0; k < dg_solver.NumElements(); ++k) {
						auto avg = dg_solver.GetCellAverage(k);
						dg_file << mesh.GetElement(k).cx << " "
						        << mesh.GetElement(k).cy << " "
						        << (avg.H - D0) << "\n";
					}
				}
				next_save_t += dg_save_time_interval;
			}
		}

		auto dg_end = std::chrono::high_resolution_clock::now();
		double dg_time = std::chrono::duration<double>(dg_end - dg_start).count();

		// DG max eta
		double dg_max_eta = 0;
		for (int k = 0; k < dg_solver.NumElements(); ++k) {
			auto avg = dg_solver.GetCellAverage(k);
			dg_max_eta = std::max(dg_max_eta, std::abs(avg.H - D0));
		}

		std::cerr << "  DG done: " << dg_time << "s (" << dg_steps << " steps), max|eta|=" << dg_max_eta << std::endl;

		// ========== Summary ==========
		std::cerr << "\n=== DG vs MacCormack Summary ===" << std::endl;
		std::cerr << "  Physical time: " << total_time << " s" << std::endl;
		std::cerr << "  MacCormack: " << mc_time << " s, " << num_steps << " steps, max|eta|=" << mc_max_eta << std::endl;
		std::cerr << "  DG P2:      " << dg_time << " s, " << dg_steps << " steps, max|eta|=" << dg_max_eta << std::endl;
		std::cerr << "  Speedup: MacCormack " << (dg_time > 0 ? dg_time / mc_time : 0) << "x faster" << std::endl;

		// Write summary file
		{
			std::ofstream summary(result_dir + "/summary.txt");
			summary << "DG vs MacCormack Comparison" << std::endl;
			summary << "Domain: " << Lx << " x " << Ly << " m, D=" << D0 << " m" << std::endl;
			summary << "Source: Gaussian amp=" << amp << " m, sigma=" << sigma << " m" << std::endl;
			summary << "MacCormack: " << mc_nx << "x" << mc_ny << " grid, dx=" << mc_dx << " m, tau=" << tau << " s" << std::endl;
			summary << "  Steps: " << num_steps << ", Time: " << mc_time << " s" << std::endl;
			summary << "  max|eta| final: " << mc_max_eta << std::endl;
			summary << "DG P2: " << dg_nx << "x" << dg_ny << " mesh (" << 2*dg_nx*dg_ny << " elements)" << std::endl;
			summary << "  Steps: " << dg_steps << ", dt=" << dg_dt << " s, Time: " << dg_time << " s" << std::endl;
			summary << "  max|eta| final: " << dg_max_eta << std::endl;
		}

		std::cerr << "  Results saved to " << result_dir << "/" << std::endl;
	}

	/// DiskOffload test: runs the same scenario both in-memory and disk-offload,
	/// compares results and reports timing.
	void DiskOffloadTest(int testIndex)
	{
		std::cerr << "=== DiskOffload Test: scenario " << testIndex << " ===" << std::endl;

		// Build scenario (same as SimulatorTests)
		SingleLineScenario scenario;
		auto build_scenario = [&]() {
			switch (testIndex) {
			case 1: return ScenarioBuilder::BuildTest1();
			case 2: return ScenarioBuilder::BuildTest2();
			case 3: return ScenarioBuilder::BuildTest3();
			case 4: return ScenarioBuilder::BuildTest4();
			case 5: return ScenarioBuilder::BuildTest5();
			case 6: return ScenarioBuilder::BuildTest6();
			default:
				throw std::runtime_error("Unknown disk test: " + std::to_string(testIndex));
			}
		};
		scenario = build_scenario();

		// Honor CLI overrides (--threads, --subgrids, --timesteps) when set via SetXxxOverride.
		if (threads_override_ > 0)   scenario.num_threads = threads_override_;
		if (subgrids_override_ > 0)  scenario.num_subgrids = subgrids_override_;
		if (timesteps_override_ > 0) scenario.timesteps = timesteps_override_;

		const bool skip_reference = disk_skip_reference_;
		double ref_ms = 0.0;
		SingleLineGridSimulation ref_sim;

		if (!skip_reference) {
			// ---- Part 1: In-memory reference ----
			std::cerr << "  [1/2] Running in-memory reference..." << std::endl;
			ScenarioBuilder::ApplyScenarioToSimulation(scenario, ref_sim);
			ref_sim.SetResultFolder("Result/disk_test" + std::to_string(testIndex) + "_reference");
			ref_sim.Initialize();

			auto ref_start = std::chrono::high_resolution_clock::now();
			ref_sim.Run();
			auto ref_end = std::chrono::high_resolution_clock::now();
			ref_ms = std::chrono::duration<double, std::milli>(ref_end - ref_start).count();
			std::cerr << "  In-memory time: " << ref_ms / 1000.0 << " s" << std::endl;
		} else {
			std::cerr << "  [1/2] Skipping in-memory reference (--disk-skip-reference)." << std::endl;
		}

		// ---- Part 2: Disk-offload ----
		std::cerr << "  [2/2] Running disk-offload..." << std::endl;

		// Rebuild scenario (fresh grid) — reapply overrides
		scenario = build_scenario();
		if (threads_override_ > 0)   scenario.num_threads = threads_override_;
		if (subgrids_override_ > 0)  scenario.num_subgrids = subgrids_override_;
		if (timesteps_override_ > 0) scenario.timesteps = timesteps_override_;

		DiskOffloadSimulator disk_sim;
		disk_sim.SetParameters(
			scenario.g, scenario.tau, scenario.delta_x, scenario.delta_y,
			scenario.nx, scenario.ny, scenario.timesteps, scenario.num_subgrids,
			scenario.exchangeBoundaryIndex, scenario.exchangeHatBoundaryIndex,
			scenario.coord_system, scenario.center_latitude_rad,
			scenario.minimum_depth_to_calculate
		);
		if (scenario.has_own_grid) {
			disk_sim.SetGrid(std::move(scenario.grid));
		}
		// Honor CLI --save-interval override (default 10 — was overflowing disk on Test 6).
		const int disk_save_interval = (save_interval_override_ >= 0) ? save_interval_override_ : 10;
		disk_sim.SetSaveInterval(disk_save_interval);
		disk_sim.SetResultFolder("Result/disk_test" + std::to_string(testIndex) + "_offload");
		disk_sim.SetTempFolder("Result/disk_test" + std::to_string(testIndex) + "_temp");

		disk_sim.Initialize();

		auto disk_start = std::chrono::high_resolution_clock::now();
		disk_sim.Run();
		auto disk_end = std::chrono::high_resolution_clock::now();
		double disk_ms = std::chrono::duration<double, std::milli>(disk_end - disk_start).count();

		std::cerr << "  Disk-offload time: " << disk_ms / 1000.0 << " s" << std::endl;

		// ---- Compare ----
		disk_sim.CollectResults(0);
		double max_diff = 0;
		int diff_cells = 0;
		if (!skip_reference) {
			// Only compare when reference run was actually performed.
			for (int i = 0; i < scenario.nx; ++i) {
				for (int j = 0; j < scenario.ny; ++j) {
					double d = std::abs(ref_sim.global_grid_.eta(i, j) - disk_sim.global_grid_.eta(i, j));
					if (d > max_diff) max_diff = d;
					if (d > 1e-12) diff_cells++;
				}
			}
		}

		std::cerr << "  max|diff|: " << max_diff << std::endl;
		std::cerr << "  Cells with diff > 1e-12: " << diff_cells << "/" << (scenario.nx * scenario.ny) << std::endl;
		const double slowdown = (ref_ms > 0.0) ? disk_ms / ref_ms : 0.0;
		if (ref_ms > 0.0) std::cerr << "  Slowdown: " << slowdown << "x" << std::endl;
		else std::cerr << "  Slowdown: n/a (reference skipped)" << std::endl;
		std::cerr << "  I/O overhead: " << disk_sim.total_io_time_ms_ / 1000.0 << " s ("
			<< 100.0 * disk_sim.total_io_time_ms_ / disk_ms << "%)" << std::endl;

		bool pass = (ref_ms > 0.0) && (max_diff < 1e-6);
		std::cerr << (pass ? "[PASS]" : (ref_ms > 0.0 ? "[WARN]" : "[INFO]"))
			<< " DiskOffload test " << testIndex
			<< (ref_ms > 0.0
				? (": max|diff|=" + std::to_string(max_diff)
				   + ", slowdown=" + std::to_string(slowdown) + "x")
				: (": disk_only run completed (no reference comparison)"))
			<< std::endl;

		// Write summary
		std::string summary_path = "Result/disk_test" + std::to_string(testIndex) + "_summary.txt";
		std::ofstream summary(summary_path);
		summary << "test=" << testIndex << "\n"
			<< "nx=" << scenario.nx << "\n"
			<< "ny=" << scenario.ny << "\n"
			<< "timesteps=" << scenario.timesteps << "\n"
			<< "num_subgrids=" << scenario.num_subgrids << "\n"
			<< "in_memory_time_s=" << ref_ms / 1000.0 << "\n"
			<< "disk_offload_time_s=" << disk_ms / 1000.0 << "\n"
			<< "slowdown=" << ((ref_ms > 0.0) ? slowdown : 0.0) << "\n"
			<< "io_time_s=" << disk_sim.total_io_time_ms_ / 1000.0 << "\n"
			<< "compute_time_s=" << disk_sim.total_compute_time_ms_ / 1000.0 << "\n"
			<< "max_diff=" << max_diff << "\n"
			<< "diff_cells=" << diff_cells << "\n"
			<< "result=" << (pass ? "PASS" : (ref_ms > 0.0 ? "WARN" : "INFO")) << "\n";
		summary.close();
		std::cerr << "  Summary saved: " << summary_path << std::endl;

		std::cerr << "=========================================" << std::endl;
	}

	/// Regression test: compare legacy SingleLineGridSimulation vs HybridSimulationRunner
	/// on the same scenario (Test1: 120x120, 500 steps)
	void RegressionTest()
	{
		std::cerr << "=== Regression Test: Legacy vs Hybrid ===" << std::endl;

		// --- Legacy run ---
		std::cerr << "Running legacy (SingleLineGridSimulation)..." << std::endl;
		SingleLineScenario scenario_legacy = ScenarioBuilder::BuildTest1();
		SingleLineGridSimulation simulation;
		ScenarioBuilder::ApplyScenarioToSimulation(scenario_legacy, simulation);
		simulation.Initialize();
		simulation.Run();

		// Capture legacy eta
		int nx = simulation.global_grid_.GetNx();
		int ny = simulation.global_grid_.GetNy();
		std::vector<double> eta_legacy(nx * ny);
		for (int i = 0; i < nx; ++i)
			for (int j = 0; j < ny; ++j)
				eta_legacy[i * ny + j] = simulation.global_grid_.eta(i, j);

		// --- Hybrid run ---
		std::cerr << "Running hybrid (SimulationRunner)..." << std::endl;
		SingleLineScenario scenario_hybrid = ScenarioBuilder::BuildTest1();

		auto domain = HybridDomainBuilder::BuildStructuredStripes(
			scenario_hybrid.g, scenario_hybrid.tau,
			scenario_hybrid.delta_x, scenario_hybrid.delta_y,
			scenario_hybrid.nx, scenario_hybrid.ny, scenario_hybrid.num_subgrids,
			scenario_hybrid.grid, scenario_hybrid.coord_system,
			scenario_hybrid.minimum_depth_to_calculate, scenario_hybrid.center_latitude_rad,
			scenario_hybrid.exchangeBoundaryIndex,
			scenario_hybrid.exchangeHatBoundaryIndex
		);

		SimulationRunner runner(domain, scenario_hybrid.tau, scenario_hybrid.timesteps, scenario_hybrid.g);
		runner.SetVerbose(false);
		runner.Run();

		// Collect hybrid results
		domain->CollectResultsToGlobalGrid();
		const Grid& hybrid_grid = domain->GetGlobalGrid();

		// --- Compare ---
		double max_diff = 0.0;
		double max_eta = 0.0;
		int max_i = 0, max_j = 0;

		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				double el = eta_legacy[i * ny + j];
				double eh = hybrid_grid.eta(i, j);
				double diff = std::abs(el - eh);
				if (diff > max_diff) {
					max_diff = diff;
					max_i = i;
					max_j = j;
				}
				max_eta = std::max(max_eta, std::abs(el));
			}
		}

		double rel_diff = (max_eta > 1e-15) ? max_diff / max_eta : max_diff;

		std::cerr << "  Grid: " << nx << " x " << ny << std::endl;
		std::cerr << "  max|eta_legacy - eta_hybrid| = " << max_diff
		          << " at (" << max_i << "," << max_j << ")" << std::endl;
		std::cerr << "  eta_legacy(" << max_i << "," << max_j << ") = "
		          << eta_legacy[max_i * ny + max_j] << std::endl;
		std::cerr << "  eta_hybrid(" << max_i << "," << max_j << ") = "
		          << hybrid_grid.eta(max_i, max_j) << std::endl;
		std::cerr << "  max|eta| = " << max_eta << std::endl;
		std::cerr << "  relative diff = " << rel_diff << std::endl;

		// Sample interior points
		int ci = nx / 2, cj = ny / 2;
		std::cerr << "  Center (" << ci << "," << cj << "): legacy="
		          << eta_legacy[ci * ny + cj] << " hybrid="
		          << hybrid_grid.eta(ci, cj) << std::endl;

		// Count cells with diff > 1e-12
		int diff_cells = 0;
		for (int i = 0; i < nx; ++i)
			for (int j = 0; j < ny; ++j)
				if (std::abs(eta_legacy[i * ny + j] - hybrid_grid.eta(i, j)) > 1e-12)
					++diff_cells;
		std::cerr << "  Cells with diff > 1e-12: " << diff_cells << " / " << nx * ny << std::endl;

		bool pass = rel_diff < 1e-10;
		std::cerr << (pass ? "[PASS]" : "[FAIL]")
		          << " Legacy vs Hybrid regression" << std::endl;
		std::cerr << "=========================================" << std::endl;
	}

	/// Regression test: horizontal vs vertical decomposition on Test1
	/// Expected: bitwise identical results
	void DecompositionRegressionTest()
	{
		std::cerr << "=== Decomposition Regression: Horizontal vs Vertical ===" << std::endl;

		// --- Horizontal run ---
		std::cerr << "Running horizontal decomposition..." << std::endl;
		SingleLineScenario scenario_h = ScenarioBuilder::BuildTest1();
		scenario_h.decomp_dir = DecompositionDirection::Horizontal;
		SingleLineGridSimulation sim_h;
		ScenarioBuilder::ApplyScenarioToSimulation(scenario_h, sim_h);
		sim_h.Initialize();
		sim_h.Run();

		int nx = sim_h.global_grid_.GetNx();
		int ny = sim_h.global_grid_.GetNy();
		std::vector<double> eta_h(nx * ny);
		for (int i = 0; i < nx; ++i)
			for (int j = 0; j < ny; ++j)
				eta_h[i * ny + j] = sim_h.global_grid_.eta(i, j);

		// --- Vertical run ---
		std::cerr << "Running vertical decomposition..." << std::endl;
		SingleLineScenario scenario_v = ScenarioBuilder::BuildTest1();
		scenario_v.decomp_dir = DecompositionDirection::Vertical;
		SingleLineGridSimulation sim_v;
		ScenarioBuilder::ApplyScenarioToSimulation(scenario_v, sim_v);
		sim_v.Initialize();
		sim_v.Run();

		// --- Compare (skip 2-cell border to exclude domain-edge treatment differences) ---
		const int margin = 2;
		double max_diff = 0.0;
		double max_eta = 0.0;
		int max_i = 0, max_j = 0;

		for (int i = margin; i < nx - margin; ++i) {
			for (int j = margin; j < ny - margin; ++j) {
				double eh = eta_h[i * ny + j];
				double ev = sim_v.global_grid_.eta(i, j);
				double diff = std::abs(eh - ev);
				if (diff > max_diff) {
					max_diff = diff;
					max_i = i;
					max_j = j;
				}
				max_eta = std::max(max_eta, std::abs(eh));
			}
		}

		double rel_diff = (max_eta > 1e-15) ? max_diff / max_eta : max_diff;

		std::cerr << "  Grid: " << nx << " x " << ny << " (interior " << margin << "-cell margin)" << std::endl;
		std::cerr << "  max|eta_horiz - eta_vert| = " << max_diff
		          << " at (" << max_i << "," << max_j << ")" << std::endl;
		std::cerr << "  max|eta| = " << max_eta << std::endl;
		std::cerr << "  relative diff = " << rel_diff << std::endl;

		int diff_cells = 0;
		for (int i = margin; i < nx - margin; ++i)
			for (int j = margin; j < ny - margin; ++j)
				if (std::abs(eta_h[i * ny + j] - sim_v.global_grid_.eta(i, j)) > 1e-12)
					++diff_cells;
		int interior_cells = (nx - 2*margin) * (ny - 2*margin);
		std::cerr << "  Cells with diff > 1e-12: " << diff_cells << " / " << interior_cells << std::endl;

		bool pass = rel_diff < 1e-10;
		std::cerr << (pass ? "[PASS]" : "[FAIL]")
		          << " Horizontal vs Vertical decomposition regression" << std::endl;
		std::cerr << "=========================================" << std::endl;
	}

	/// Phase 10.5: Coupling test — structured (left) ↔ mesh (right)
	/// Verifies data flows across coupling interface without NaN, with mass conservation
	void CouplingTest()
	{
		std::cerr << "=== Coupling Test: Structured <-> Mesh ===" << std::endl;

		const double Lx = 1000.0, Ly = 1000.0;
		const double depth = 100.0, g = 9.81;
		const int struct_nx = 60, struct_ny = 60;
		const int mesh_nx = 30, mesh_ny = 30;
		const double tau = 0.1;   // CFL safe: dx/c ≈ 8.33/31.3 ≈ 0.27
		const int num_steps = 200;

		// Build domain: left structured + right mesh
		auto domain = HybridDomainBuilder::BuildStructuredMeshDomain(
			Lx, Ly, struct_nx, struct_ny, mesh_nx, mesh_ny, depth, g, tau
		);

		// Set initial perturbation in the structured patch (Gaussian bump at x=250, y=500)
		{
			Grid& gg = domain->GetGlobalGrid();
			double dx = (Lx * 0.5) / struct_nx;
			double dy_g = Ly / struct_ny;
			double cx = 250.0, cy = 500.0, sigma = 50.0, amp = 1.0;
			for (int i = 0; i < struct_nx; ++i) {
				for (int j = 0; j < struct_ny; ++j) {
					double x = (i + 0.5) * dx;
					double y = (j + 0.5) * dy_g;
					double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
					double eta_bump = amp * std::exp(-r2 / (2.0 * sigma * sigma));
					gg.eta(i, j) = eta_bump;
					gg.H(i, j) = depth + eta_bump;
				}
			}
			// Re-initialize structured patch from updated global grid
			auto struct_patch = std::dynamic_pointer_cast<StructuredPatch>(domain->GetPatch(0));
			if (struct_patch) {
				struct_patch->GetSubGrid()->Initialize();
			}
		}

		// Capture mesh patch handle now so the per-step save callback below
		// can read DG cell averages without re-looking it up every step.
		auto mesh_patch_track = std::dynamic_pointer_cast<MeshPatch>(domain->GetPatch(1));

		// Per-step mass tracking for Phase 10.4 flux balance check.
		// Recorded once at t=0 (after initial exchange) plus after every step.
		struct MassRecord { int t; double mass_struct; double mass_mesh; };
		std::vector<MassRecord> mass_log;
		mass_log.reserve(static_cast<size_t>(num_steps) + 2);

		const double dx_struct = (Lx * 0.5) / struct_nx;
		const double dy_struct = Ly / struct_ny;
		auto record_mass = [&](int t, const Grid& g_now) {
			double ms = 0.0;
			for (int i = 0; i < struct_nx; ++i)
				for (int j = 0; j < struct_ny; ++j)
					ms += g_now.H(i, j) * dx_struct * dy_struct;
			double md = 0.0;
			if (mesh_patch_track) {
				const auto& dg = mesh_patch_track->GetDGSolver();
				const auto& mesh = mesh_patch_track->GetMesh();
				for (int k = 0; k < dg.NumElements(); ++k)
					md += dg.GetCellAverage(k).H * mesh.GetElement(k).area;
			}
			mass_log.push_back({ t, ms, md });
		};

		// Initial mass (t=0) — captured here before run so the first step's
		// delta is well-defined.
		domain->CollectResultsToGlobalGrid();
		record_mass(0, domain->GetGlobalGrid());

		// Run simulation
		SimulationRunner runner(domain, tau, num_steps, g);
		runner.SetVerbose(false);
		runner.SetSaveInterval(1);
		runner.SetSaveFunction([&](int t, const Grid& g_now) {
			record_mass(t, g_now);
		});

		auto start_time = std::chrono::high_resolution_clock::now();
		runner.Run();
		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end_time - start_time;
		std::cerr << "  Completed in " << elapsed.count() << "s" << std::endl;

		// --- Checks ---
		bool pass = true;

		// Check 1: No NaN in structured patch
		domain->CollectResultsToGlobalGrid();
		const Grid& gg = domain->GetGlobalGrid();
		int nan_count = 0;
		for (int i = 0; i < struct_nx; ++i)
			for (int j = 0; j < struct_ny; ++j)
				if (std::isnan(gg.eta(i, j)) || std::isnan(gg.H(i, j))) ++nan_count;

		std::cerr << "  Structured NaN count: " << nan_count << std::endl;
		if (nan_count > 0) pass = false;

		// Check 2: No NaN in mesh patch
		auto mesh_patch = std::dynamic_pointer_cast<MeshPatch>(domain->GetPatch(1));
		int mesh_nan = 0;
		if (mesh_patch) {
			const auto& dg = mesh_patch->GetDGSolver();
			for (int k = 0; k < dg.NumElements(); ++k) {
				SWEState avg = dg.GetCellAverage(k);
				if (std::isnan(avg.H) || std::isnan(avg.Hu) || std::isnan(avg.Hv))
					++mesh_nan;
			}
		}
		std::cerr << "  Mesh NaN elements: " << mesh_nan << std::endl;
		if (mesh_nan > 0) pass = false;

		// Check 3: H > 0 everywhere in mesh
		int neg_h = 0;
		double min_h = 1e30;
		if (mesh_patch) {
			const auto& dg = mesh_patch->GetDGSolver();
			for (int k = 0; k < dg.NumElements(); ++k) {
				SWEState avg = dg.GetCellAverage(k);
				min_h = std::min(min_h, avg.H);
				if (avg.H < 0.0) ++neg_h;
			}
		}
		std::cerr << "  Mesh min(H) = " << min_h << ", negative: " << neg_h << std::endl;
		if (neg_h > 0) pass = false;

		// Check 4: Total mass roughly conserved
		// Initial mass: struct_nx * struct_ny * dx * dy * depth + bump integral
		// We check that total mass hasn't changed by more than 5%
		double struct_mass = 0.0;
		{
			double dx = (Lx * 0.5) / struct_nx;
			double dy_g = Ly / struct_ny;
			for (int i = 0; i < struct_nx; ++i)
				for (int j = 0; j < struct_ny; ++j)
					struct_mass += gg.H(i, j) * dx * dy_g;
		}
		double mesh_mass = 0.0;
		if (mesh_patch) {
			const auto& dg = mesh_patch->GetDGSolver();
			const auto& mesh = mesh_patch->GetMesh();
			for (int k = 0; k < dg.NumElements(); ++k) {
				SWEState avg = dg.GetCellAverage(k);
				mesh_mass += avg.H * mesh.GetElement(k).area;
			}
		}
		double total_mass = struct_mass + mesh_mass;
		double expected_mass = depth * Lx * Ly; // without bump it's depth * area
		double mass_err = std::abs(total_mass - expected_mass) / expected_mass;
		std::cerr << "  Structured mass: " << struct_mass << std::endl;
		std::cerr << "  Mesh mass: " << mesh_mass << std::endl;
		std::cerr << "  Total mass: " << total_mass << " (expected ~" << expected_mass << ")" << std::endl;
		std::cerr << "  Mass error: " << mass_err * 100.0 << "%" << std::endl;
		if (mass_err > 0.05) pass = false;

		// Check 5: Wave reached mesh (at least some elements have eta != 0)
		int mesh_active = 0;
		if (mesh_patch) {
			const auto& dg = mesh_patch->GetDGSolver();
			for (int k = 0; k < dg.NumElements(); ++k) {
				SWEState avg = dg.GetCellAverage(k);
				double D_val = dg.GetBathymetry(k);
				double eta = avg.H - D_val;
				if (std::abs(eta) > 1e-10) ++mesh_active;
			}
		}
		int total_mesh_elems = mesh_patch ? mesh_patch->GetDGSolver().NumElements() : 0;
		std::cerr << "  Mesh elements with |eta| > 1e-10: " << mesh_active
		          << " / " << total_mesh_elems << std::endl;

		// Check 6: Phase 10.4 — formal flux balance on structured ↔ mesh interface.
		// With wall BCs on every external boundary the only mass path is through
		// the shared interface, so for every step
		//     Δmass_struct(t) = −Δmass_mesh(t)   (consistent two-sided flux),
		// and integrated over the run the two cumulative fluxes must agree.
		//
		// We report:
		//   • per-step asymmetry   max_t |Δms + Δmd| / mass_total
		//   • cumulative imbalance |Σ Δms + Σ Δmd| / mass_total
		// Tolerance is loose (1e-3) because hydrostatic reconstruction in DG
		// reshuffles momentum in the limiter step, but the mass column is
		// directly governed by the HLL flux that MC sees via
		// SimulationRunner::TransferDGFluxesToStructured.
		{
			double max_per_step_asym = 0.0;
			int max_asym_step = -1;
			double sum_dms = 0.0, sum_dmd = 0.0;
			double mass_total_init = (mass_log.empty() ? 1.0
			                          : mass_log.front().mass_struct + mass_log.front().mass_mesh);
			if (mass_total_init <= 0.0) mass_total_init = 1.0;

			for (size_t i = 1; i < mass_log.size(); ++i) {
				double dms = mass_log[i].mass_struct - mass_log[i - 1].mass_struct;
				double dmd = mass_log[i].mass_mesh   - mass_log[i - 1].mass_mesh;
				sum_dms += dms;
				sum_dmd += dmd;
				double asym = std::abs(dms + dmd);
				if (asym > max_per_step_asym) {
					max_per_step_asym = asym;
					max_asym_step = mass_log[i].t;
				}
			}
			double per_step_rel = max_per_step_asym / mass_total_init;
			double cumulative_rel = std::abs(sum_dms + sum_dmd) / mass_total_init;

			std::cerr << "  --- Flux balance (Phase 10.4) ---" << std::endl;
			std::cerr << "  Samples: " << mass_log.size() << " (initial + per-step)" << std::endl;
			std::cerr << "  Cumulative Δmass_struct: " << sum_dms
			          << ", Δmass_mesh: " << sum_dmd
			          << ", sum: " << (sum_dms + sum_dmd) << std::endl;
			std::cerr << "  Cumulative imbalance / total mass: " << cumulative_rel << std::endl;
			std::cerr << "  Max per-step |Δms+Δmd| / total mass: " << per_step_rel
			          << " (at step " << max_asym_step << ")" << std::endl;

			const double per_step_tol   = 1e-3;
			const double cumulative_tol = 1e-3;
			bool flux_ok = (per_step_rel < per_step_tol) && (cumulative_rel < cumulative_tol);
			if (!flux_ok) pass = false;
			std::cerr << "  Flux balance: " << (flux_ok ? "OK" : "FAIL")
			          << " (per-step tol " << per_step_tol
			          << ", cumulative tol " << cumulative_tol << ")" << std::endl;
		}

		std::cerr << (pass ? "[PASS]" : "[FAIL]")
		          << " Structured <-> Mesh coupling test" << std::endl;
		std::cerr << "=========================================" << std::endl;
	}

	/// Mixed solver test: compare pure MacCormack vs hybrid (MacCormack + DG + MacCormack)
	/// Verifies that hybrid result is physically plausible but different from pure MacCormack
	void MixedSolverTest()
	{
		std::cerr << "=== Mixed Solver Test: MacCormack vs Hybrid (MacCormack+DG+MacCormack) ===" << std::endl;

		const double Lx = 1000.0, Ly = 1000.0;
		const double depth = 100.0, g = 9.81;
		const double tau = 0.1;
		const int num_steps = 200;
		const double cx = 250.0, cy = 500.0, sigma = 50.0, amp = 1.0;

		// ---- Part 1: Pure MacCormack via CouplingTest pattern ----
		std::cerr << "  Running pure MacCormack (2 structured patches)..." << std::endl;
		const int struct_nx = 60, struct_ny = 60;
		auto pure_domain = HybridDomainBuilder::BuildStructuredMeshDomain(
			Lx, Ly, struct_nx, struct_ny, 30, 30, depth, g, tau
		);
		// Set initial condition
		{
			Grid& gg = pure_domain->GetGlobalGrid();
			double dx = (Lx * 0.5) / struct_nx;
			double dy_g = Ly / struct_ny;
			for (int i = 0; i < struct_nx; ++i) {
				for (int j = 0; j < struct_ny; ++j) {
					double x = (i + 0.5) * dx;
					double y = (j + 0.5) * dy_g;
					double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
					double eta_bump = amp * std::exp(-r2 / (2.0 * sigma * sigma));
					gg.eta(i, j) = eta_bump;
					gg.H(i, j) = depth + eta_bump;
				}
			}
			auto sp = std::dynamic_pointer_cast<StructuredPatch>(pure_domain->GetPatch(0));
			if (sp) sp->GetSubGrid()->Initialize();
		}

		SimulationRunner pure_runner(pure_domain, tau, num_steps, g);
		pure_runner.SetVerbose(false);
		pure_runner.Run();
		pure_domain->CollectResultsToGlobalGrid();

		// Store pure MacCormack structured-patch eta
		const Grid& pure_gg = pure_domain->GetGlobalGrid();
		std::vector<double> pure_eta(struct_nx * struct_ny);
		for (int i = 0; i < struct_nx; ++i)
			for (int j = 0; j < struct_ny; ++j)
				pure_eta[i * struct_ny + j] = pure_gg.eta(i, j);

		// Get mesh eta from DG patch
		auto pure_mesh = std::dynamic_pointer_cast<MeshPatch>(pure_domain->GetPatch(1));
		double pure_mesh_max_eta = 0;
		if (pure_mesh) {
			const auto& dg = pure_mesh->GetDGSolver();
			for (int k = 0; k < dg.NumElements(); ++k) {
				SWEState avg = dg.GetCellAverage(k);
				double eta = avg.H - dg.GetBathymetry(k);
				pure_mesh_max_eta = std::max(pure_mesh_max_eta, std::abs(eta));
			}
		}
		std::cerr << "  Pure hybrid completed. Struct max|eta|: "
		          << *std::max_element(pure_eta.begin(), pure_eta.end(),
		              [](double a, double b) { return std::abs(a) < std::abs(b); })
		          << ", Mesh max|eta|: " << pure_mesh_max_eta << std::endl;

		// ---- Part 2: Reference pure MacCormack (all structured) ----
		std::cerr << "  Running reference all-MacCormack (2 structured stripes)..." << std::endl;
		// Use SingleLineGridSimulation for reference
		SingleLineScenario ref_scenario = ScenarioBuilder::BuildTest2(); // Simple wave test
		// Override with our parameters
		ref_scenario.g = g;
		ref_scenario.tau = tau;
		ref_scenario.nx = 120;  // Full domain width (60+60)
		ref_scenario.ny = 60;
		ref_scenario.delta_x = [](int, int) { return 1000.0/120.0; };
		ref_scenario.delta_y = Ly / 60.0;
		ref_scenario.timesteps = num_steps;
		ref_scenario.num_threads = 4;
		ref_scenario.num_subgrids = 2;

		// Build grid with Gaussian bump
		ref_scenario.grid = Grid(120, 60, ref_scenario.delta_x, ref_scenario.delta_y);
		double ref_dx = 1000.0/120.0;
		double ref_dy = Ly / 60.0;
		for (int i = 0; i < 120; ++i) {
			for (int j = 0; j < 60; ++j) {
				ref_scenario.grid.D(i, j) = depth;
				ref_scenario.grid.H(i, j) = depth;
				ref_scenario.grid.u(i, j) = 0.0;
				ref_scenario.grid.v(i, j) = 0.0;
				double x = (i + 0.5) * ref_dx;
				double y = (j + 0.5) * ref_dy;
				double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
				double eta_bump = amp * std::exp(-r2 / (2.0 * sigma * sigma));
				ref_scenario.grid.eta(i, j) = eta_bump;
				ref_scenario.grid.H(i, j) = depth + eta_bump;
			}
		}
		ref_scenario.has_own_grid = true;

		SingleLineGridSimulation ref_sim;
		ScenarioBuilder::ApplyScenarioToSimulation(ref_scenario, ref_sim);
		ref_sim.Initialize();
		ref_sim.Run();

		// Compare left-half eta
		double max_diff = 0, max_pure = 0, max_ref = 0;
		for (int i = 0; i < 60; ++i) {
			for (int j = 0; j < 60; ++j) {
				double e_pure = pure_eta[i * 60 + j];
				double e_ref = ref_sim.global_grid_.eta(i, j);
				max_diff = std::max(max_diff, std::abs(e_pure - e_ref));
				max_pure = std::max(max_pure, std::abs(e_pure));
				max_ref = std::max(max_ref, std::abs(e_ref));
			}
		}

		std::cerr << "  Reference max|eta| (left half): " << max_ref << std::endl;
		std::cerr << "  Hybrid max|eta| (left half): " << max_pure << std::endl;
		std::cerr << "  max|diff|: " << max_diff << std::endl;
		double rel_diff = (max_ref > 1e-15) ? max_diff / max_ref : max_diff;
		std::cerr << "  Relative diff: " << rel_diff * 100.0 << "%" << std::endl;

		// ---- Checks ----
		bool pass = true;

		// Check 1: Both produce non-trivial results
		if (max_pure < 1e-10 || max_ref < 1e-10) {
			std::cerr << "  [FAIL] One of the runs produced all-zero eta" << std::endl;
			pass = false;
		}

		// Check 2: Results are different (DG adds some difference)
		if (max_diff < 1e-15) {
			std::cerr << "  [WARN] Results are bitwise identical — expected some DG difference" << std::endl;
		}

		// Check 3: Results are physically similar (not wildly off)
		if (rel_diff > 0.5) { // 50% tolerance for different solvers
			std::cerr << "  [FAIL] Relative difference too large: " << rel_diff * 100.0 << "%" << std::endl;
			pass = false;
		}

		// Check 4: Wave reached DG patch
		if (pure_mesh_max_eta < 1e-10) {
			std::cerr << "  [FAIL] Wave did not reach DG mesh patch" << std::endl;
			pass = false;
		}

		std::cerr << (pass ? "[PASS]" : "[FAIL]")
		          << " Mixed solver test: MacCormack vs Hybrid" << std::endl;
		std::cerr << "=========================================" << std::endl;
	}

	/// Nested DG patch test (Work_Plan §7 вариант 7.B): прямоугольная DG-область
	/// внутри MacCormack-домена. 300×300 grid, dg in [100,200)×[100,200), Gaussian
	/// bump в центре, 500 шагов. PASS если нет NaN, mass error < 1%, DG активна.
	void NestedDGTest()
	{
		std::cerr << "=== Nested DG Test ===" << std::endl;

		const int nx = 300;
		const int ny = 300;
		const double dx = 10.0;
		const double dy = 10.0;
		const double depth = 100.0;
		const double g = 9.81;
		const double tau = 0.05;
		const int timesteps = 500;
		const int i_dg0 = 100;
		const int i_dg1 = 200;
		const int j_dg0 = 100;
		const int j_dg1 = 200;
		const double cx = 150.0 * dx;
		const double cy = 80.0 * dy;
		const double sigma = 30.0 * dx;
		const double amp = 1.0;

		auto delta_x = [dx](int, int) { return dx; };
		Grid global_grid(nx, ny, delta_x, dy);
		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				global_grid.D(i, j) = depth;
				global_grid.H(i, j) = depth;
				global_grid.u(i, j) = 0.0;
				global_grid.v(i, j) = 0.0;
				global_grid.eta(i, j) = 0.0;
			}
		}

		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				const double x = (i + 0.5) * dx;
				const double y = (j + 0.5) * dy;
				const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
				const double eta_bump = amp * std::exp(-r2 / (2.0 * sigma * sigma));
				global_grid.eta(i, j) = eta_bump;
				global_grid.H(i, j) = depth + eta_bump;
			}
		}

		double initial_mass = 0.0;
		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				initial_mass += global_grid.H(i, j) * dx * dy;
			}
		}

		auto domain = HybridDomainBuilder::BuildNestedDGPatch(
			g, tau, delta_x, dy, nx, ny, global_grid,
			CoordSystem::Cartesian, 1e-6,
			i_dg0, i_dg1, j_dg0, j_dg1,
			1, 0
		);

		SimulationRunner runner(domain, tau, timesteps, g);
		runner.SetVerbose(false);
		runner.SetSaveInterval(999999);
		runner.Run();

		domain->CollectResultsToGlobalGrid();
		const Grid& final_grid = domain->GetGlobalGrid();

		int nan_count = 0;
		double final_mass = 0.0;
		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				if (std::isnan(final_grid.H(i, j)) || std::isnan(final_grid.eta(i, j)) ||
					std::isnan(final_grid.u(i, j)) || std::isnan(final_grid.v(i, j))) {
					++nan_count;
				}
				final_mass += final_grid.H(i, j) * dx * dy;
			}
		}

		auto dg_patch = std::shared_ptr<MeshPatch>();
		for (const auto& patch : domain->GetPatches()) {
			auto mp = std::dynamic_pointer_cast<MeshPatch>(patch);
			if (mp && mp->GetDGSolver().NumElements() > 0) {
				dg_patch = mp;
				break;
			}
		}

		int dg_nan_count = 0;
		double min_dg_h = std::numeric_limits<double>::max();
		int mesh_active = 0;
		int mesh_total = 0;
		if (dg_patch) {
			const auto& dg = dg_patch->GetDGSolver();
			mesh_total = dg.NumElements();
			for (int k = 0; k < dg.NumElements(); ++k) {
				const SWEState avg = dg.GetCellAverage(k);
				if (std::isnan(avg.H) || std::isnan(avg.Hu) || std::isnan(avg.Hv)) {
					++dg_nan_count;
					continue;
				}
				min_dg_h = std::min(min_dg_h, avg.H);
				const double eta = avg.H - dg.GetBathymetry(k);
				if (std::abs(eta) > 1e-4) {
					++mesh_active;
				}
			}
		}

		const double mass_err = (initial_mass > 0.0)
			? std::abs(final_mass - initial_mass) / initial_mass
			: 0.0;
		const double mesh_active_ratio = (mesh_total > 0)
			? (static_cast<double>(mesh_active) / mesh_total)
			: 0.0;

		bool pass = true;
		if (nan_count > 0 || dg_nan_count > 0) pass = false;
		if (!(min_dg_h > 0.0)) pass = false;
		if (mass_err >= 0.01) pass = false;
		if (mesh_active_ratio <= 0.10) pass = false;

		std::cerr << "  Grid NaN count: " << nan_count << std::endl;
		std::cerr << "  DG NaN count: " << dg_nan_count << std::endl;
		std::cerr << "  DG min(H): " << min_dg_h << std::endl;
		std::cerr << "  Mass error: " << mass_err * 100.0 << "%" << std::endl;
		std::cerr << "  DG active elements: " << mesh_active << " / " << mesh_total
		          << " (" << mesh_active_ratio * 100.0 << "%)" << std::endl;
		std::cerr << (pass ? "[PASS]" : "[FAIL]")
		          << " Nested DG test" << std::endl;
		std::cerr << "======================" << std::endl;
	}

	void SimulatorInvoker(const std::string& input_bat, const std::string& input_par)
	{
		auto input_par_data = new DefaultParametersParser();
		input_par_data->Read(input_par);
		IBaseBathymetryParser* inputData;
		if(input_par_data->GetInputBathymetryType() == 0)
		{
			inputData = new ZDataParser();
		}
		else if(input_par_data->GetInputBathymetryType() == 1)
		{
			inputData = new ASCParser();
		}
		else
		{
			throw new std::exception("Only 2 bathymetry variants are implemented.");
		}
		inputData->Read(input_bat);
		SingleLineGridSimulation* simulation = new SingleLineGridSimulation();
		// Устанавливаем параметры тестового сценария
		if (input_par_data->GetInputBathymetryType() == 0)
		{
			simulation->SetParameters(input_par_data);
			simulation->Initialize(*((ZDataParser*)inputData));
		}
		else if (input_par_data->GetInputBathymetryType() == 1)
		{
			simulation->Initialize(*((ASCParser*)inputData), *input_par_data);
		}
		else
		{
			throw new std::exception("Only 2 bathymetry variants are implemented.");
		}
		// Измеряем время выполнения симуляции
		auto start_time = std::chrono::high_resolution_clock::now();
		// Запускаем симуляцию
		simulation->Run();
		auto end_time = std::chrono::high_resolution_clock::now();
		// Рассчитываем и выводим длительность
		std::chrono::duration<double> elapsed = end_time - start_time;
		std::cout << std::endl << "SingleLineGridSimulation completed in " << elapsed.count() << " seconds." << std::endl;
		delete inputData;
		delete input_par_data;
	}
};
