#pragma once

#include <functional>
#include <memory>

#include "CoordSystem.h"
#include "Grid.h"
#include "SubGrid.h"
#include "SingleLineGridSimulation.h"
#include "IBaseParametersParser.h"

// Описание сценария полосовой модели
struct SingleLineScenario
{
	double g = 9.81;
	double tau = 0.0;
	std::function<double(int, int)> delta_x;
	double delta_y = 0.0;
	int nx = -1;
	int ny = -1;
	int timesteps = 0;
	int num_threads = 1;   // OpenMP threads
	int num_subgrids = 1;  // number of stripes (subgrids)
	double minimum_depth_to_calculate = 8.0; // Минимальная глубина для расчёта
	CoordSystem coord_system = CoordSystem::Cartesian; // Система координат
	DecompositionDirection decomp_dir = DecompositionDirection::Horizontal; // Направление разбивки

	// НОВОЕ: Параметры для сферической системы координат
	double center_latitude_rad = 0.785398163;  // Центральная широта (радианы), по умолчанию 45° (0.785398163)

	// Индексы в массивах ExchangeBoundaryDataFunctions / ExchangeHatBoundaryDataFunctions
	int exchangeBoundaryIndex = -1;
	int exchangeHatBoundaryIndex = -1;

	// Sponge layer на границах домена (Codex plan 2026-04-10).
	// Устраняет Kelvin wave у стенок в Coriolis-тестах.
	// Для каждой стороны ширина в ячейках задаётся отдельно; 0 = выключено.
	// Damping: q <- q * exp(-sigma_max * (r/W)^2 * dt), r — расстояние внутрь sponge.
	//
	// Важно: источник, лежащий внутри sponge-зоны, будет демпфироваться. Для
	// тестов с источником у южной границы (41/42 — plane wave на j=[0,10)) НЕ
	// включать sponge_south_cells, иначе волна не сформируется.
	int sponge_west_cells = 0;
	int sponge_east_cells = 0;
	int sponge_south_cells = 0;
	int sponge_north_cells = 0;
	double sponge_sigma_max = 0.0;  // s^-1, 0.002 recommended

	// Диагностика: при записи summary выводить отдельно max_eta в interior-окне
	// (исключая sponge-зоны). interior = [i_min..i_max) × [j_min..j_max) cells.
	// -1 = disabled (использовать всю сетку).
	int interior_i_min = -1;
	int interior_i_max = -1;
	int interior_j_min = -1;
	int interior_j_max = -1;

	// Кориолис: при Spherical по умолчанию включён. При false — f=0 даже на Spherical
	// сетке (чистое сравнение "с/без Кориолиса" на идентичной сетке).
	bool coriolis_enabled = true;

	// Сохранять ли u, v поля рядом с η в снапшотах (для Coriolis-валидации).
	// Учитывается в SingleLineGridSimulation::Run.
	bool save_velocities = false;

	// Есть ли у сценария уже подготовленная глобальная сетка
	bool has_own_grid = false;
	Grid grid;

	SingleLineScenario()
		: delta_x([](int, int) { return 0.0; })
	{
	}
};

class ScenarioBuilder
{
public:
	/**
	* \brief Проверка работоспособности реализованного алгоритма:
	* Массив высот волны в центре сетки задаётся прямоугольной волной размером 10 на 10 узлов высотой 1м в центре сетки
	*/
	static SingleLineScenario BuildTest1();
	/**
	 * \brief Проверка работоспособности реализованного алгоритма:
	 * Массив высот волны в центре сетки задаётся следующей формулой: 0.5 * (1 + cos(M_PI * dist_to_node / radius))
	 */
	static SingleLineScenario BuildTest2();
	/**
	 * \brief Проверка работоспособности реализованного алгоритма:
	 * Глубина дна увеличивается квадратично от 0 до k * pow(delta_y_ * (ny_ - j), 2) + 10 и волна задаётся следующим образом: 1.0 / 2.0 * (1.0 - cos(n * tau * 2 * M_PI / T))
	 */
	static SingleLineScenario BuildTest3();
	/**
	 * \brief Проверка работоспособности реализованного алгоритма:
	 * Глубина дна постоянная, не считая нижнего правого треугольника от центра низа до центра права где расположен берег
	 */
	static SingleLineScenario BuildTest4();
	/**
	 * \brief Проверка работоспособности реализованного алгоритма:
	 * Глубина дна постоянная, не считая нижнего правого треугольника от центра низа до центра права где расположен берег
	 */
	static SingleLineScenario BuildTest5();
	/**
	 * \brief Проверка работоспособности реализованного алгоритма:
	 * Японское море
	 */
	static SingleLineScenario BuildTest6();
	/**
	 * \brief Проверка работоспособности реализованного алгоритма:
	 * Линза Люнеберга с двумя радиусами
	 */
	static SingleLineScenario BuildTest7();

	// ========== Spherical Coordinate Versions (with Coriolis force) ==========

	/**
	 * \brief Test1 в сферической системе координат с учётом силы Кориолиса
	 * Массив высот волны в центре сетки задаётся прямоугольной волной размером 10 на 10 узлов высотой 1м в центре сетки
	 */
	static SingleLineScenario BuildTest1Spherical();

	/**
	 * \brief Test2 в сферической системе координат с учётом силы Кориолиса
	 * Массив высот волны в центре сетки задаётся следующей формулой: 0.5 * (1 + cos(M_PI * dist_to_node / radius))
	 */
	static SingleLineScenario BuildTest2Spherical();

	/**
	 * \brief Test3 в сферической системе координат с учётом силы Кориолиса
	 * Глубина дна увеличивается квадратично от 0 до k * pow(delta_y_ * (ny_ - j), 2) + 10 и волна задаётся следующим образом: 1.0 / 2.0 * (1.0 - cos(n * tau * 2 * M_PI / T))
	 */
	static SingleLineScenario BuildTest3Spherical();

	/**
	 * \brief Test4 в сферической системе координат с учётом силы Кориолиса
	 * Глубина дна постоянная, не считая нижнего правого треугольника от центра низа до центра права где расположен берег
	 */
	static SingleLineScenario BuildTest4Spherical();

	/**
	 * \brief Test5 в сферической системе координат с учётом силы Кориолиса
	 * Глубина дна постоянная, не считая нижнего правого треугольника от центра низа до центра права где расположен берег
	 */
	static SingleLineScenario BuildTest5Spherical();

	/**
	 * \brief Test6 в сферической системе координат с учётом силы Кориолиса
	 * Японское море
	 */
	static SingleLineScenario BuildTest6Spherical();

	/**
	 * \brief Test7 в сферической системе координат с учётом силы Кориолиса
	 * Линза Люнеберга с двумя радиусами
	 */
	static SingleLineScenario BuildTest7Spherical();

	// ========== Coriolis Test Cases ==========

	/**
	 * \brief Coriolis test 1: plane wave south→north through equator
	 * Flat bottom D=1000m, dx=dy=1000m, center at equator, 20° latitude span
	 * Returns Cartesian scenario — call with coord_system=Spherical for Coriolis
	 */
	static SingleLineScenario BuildTestCoriolis1(CoordSystem cs = CoordSystem::Cartesian);

	/**
	 * \brief Coriolis test 3: ellipsoidal source (N-S elongated, 50x200 km) at equator
	 * Finite width source to observe actual Coriolis wavefront deflection
	 */
	static SingleLineScenario BuildTestCoriolisEllipsoidal1(CoordSystem cs = CoordSystem::Cartesian);

	/**
	 * \brief Coriolis test 4: ellipsoidal source (W-E elongated, 200x50 km) at equator
	 */
	static SingleLineScenario BuildTestCoriolisEllipsoidal2(CoordSystem cs = CoordSystem::Cartesian);

	/**
	 * \brief Coriolis test 2: plane wave west→east through equator
	 * Same domain as test 1 but wave source on left edge, vertical decomposition
	 */
	static SingleLineScenario BuildTestCoriolis2(CoordSystem cs = CoordSystem::Cartesian);

	/**
	 * \brief Japan Sea with S7 source in Cartesian coordinates
	 * Source S7 legacy ellipse: lon=132.00, lat=36.00, angle=-80,
	 * r1=0.148 deg, r2=0.449 deg
	 */
	static SingleLineScenario BuildTest6CartesianS7();

	/**
	 * \brief Japan Sea with S7 source in spherical coordinates
	 * Source S7 legacy ellipse: lon=132.00, lat=36.00, angle=-80,
	 * r1=0.148 deg, r2=0.449 deg
	 */
	static SingleLineScenario BuildTest6SphericalS7();

	/**
	 * \brief Japan Sea with multi-source scenario (JapanSeaThreeSources.par)
	 * Used for Active Region Tracking validation: multiple disjoint
	 * connected components seeded into different stripes.
	 */
	static SingleLineScenario BuildTest6JapanSeaThreeSources();

	/**
	 * \brief NTHMP Okushiri benchmark — 1993 Hokkaido Nansei-Oki tsunami,
	 * 450 m coarse single-grid. Uses NOAA NCTR data files:
	 *   Data/Okushiri/D379-687-450m.txt   (bathymetry, 10F7.1)
	 *   Data/Okushiri/I379-687-450m-17a.txt (initial η perturbation)
	 *
	 * Purpose in this branch: ACTIVE REGION EQUIVALENCE TEST on a
	 * benchmark-derived realistic tsunami case. NOT NTHMP certification.
	 */
	static SingleLineScenario BuildOkushiri450m();

	/**
	 * \brief NTHMP Monai Valley benchmark (BP6/7 — laboratory 1/400 tank).
	 *   Data/Monai/MonaiValley_Bathymetry.txt   (XYZ pointwise grid)
	 *   Data/Monai/MonaiValley_InputWave.txt    (wavemaker time series, 22.5s)
	 *
	 * 393x244 cells, dx=dy=0.014m, dt=0.01s, wavemaker on LEFT boundary,
	 * zero-gradient walls on other 3 sides. Requires new BC index 6.
	 */
	static SingleLineScenario BuildMonaiValley();

	/**
	 * \brief Synthetic lake-at-rest test for FV well-balancedness check.
	 * Flat bathymetry D=100m, η=0 everywhere, u=v=0. 200×200 cells, dx=dy=1km.
	 * Expected: η stays at 0 (or machine epsilon) indefinitely.
	 * Used to validate Phase A of FV Godunov solver.
	 */
	static SingleLineScenario BuildLakeAtRestFlat();

	/**
	 * \brief Synthetic lake-at-rest with step bathymetry (Audusse test).
	 * D varies: 100m on west half, 50m on east half. η=0 everywhere.
	 * Tests Audusse well-balancedness on bathymetry step (Codex pitfall #2).
	 * Expected: η stays at 0 with proper HR; without it tiny waves appear.
	 */
	static SingleLineScenario BuildLakeAtRestStep();

	/**
	 * \brief 1D dam-break wet/wet (Stoker problem).
	 * Flat D=10m. H_left=15m (η=+5m water column above rest), H_right=10m
	 * (still water). At t=0 dam breaks: rarefaction wave moves left, shock
	 * moves right. Validates FV Riemann solver and conservation.
	 */
	static SingleLineScenario BuildDamBreakWetWet();

	/**
	 * \brief 1D dam-break wet/dry (Ritter problem).
	 * Flat D=10m. H_left=10m (still water), H_right=0m (dry).
	 * At t=0 water collapses onto dry bed. Validates wet/dry front handling.
	 */
	static SingleLineScenario BuildDamBreakWetDry();

	/**
	 * \brief Reflective wall box (mass conservation test).
	 * D=100m flat. Initial Gaussian η bump in center. All walls reflective.
	 * Mass = ∫h dV stays constant; wave bounces off walls indefinitely.
	 */
	static SingleLineScenario BuildReflectiveBox();

	/**
	 * \brief Truly-closed reflective basin (FV mass conservation test).
	 * D=100m interior with 2-cell-thick D=0 perimeter so FV wall detection
	 * triggers (D<=0 && h<=dry_tol). Gaussian bump in center. Sum(η·dx·dy)
	 * over interior must stay constant to FP precision. Phase E pre-paper
	 * diagnostic (case 85).
	 */
	static SingleLineScenario BuildClosedBasinFV();

	/**
	 * \brief Coriolis v2: S→N plane wave at 45° latitude, 4000x4000 grid, 8000 steps
	 * Large domain to observe actual Coriolis deflection (f ≈ 1.03e-4 at 45°)
	 */
	static SingleLineScenario BuildTestCoriolisV2_SN(CoordSystem cs = CoordSystem::Cartesian);

	/**
	 * \brief Coriolis v2: W→E plane wave at 45° latitude, 4000x4000 grid, 8000 steps
	 */
	static SingleLineScenario BuildTestCoriolisV2_WE(CoordSystem cs = CoordSystem::Cartesian);

	/**
	 * \brief Coriolis v2: Ellipsoidal N-S (50x200km) at 45° latitude, 4000x4000, 8000 steps
	 */
	static SingleLineScenario BuildTestCoriolisV2_EllipNS(CoordSystem cs = CoordSystem::Cartesian);

	/**
	 * \brief Coriolis v2: Ellipsoidal W-E (200x50km) at 45° latitude, 4000x4000, 8000 steps
	 */
	static SingleLineScenario BuildTestCoriolisV2_EllipWE(CoordSystem cs = CoordSystem::Cartesian);

	/**
	 * \brief Coriolis v7: S→N plane wave with finite horizontal extent on a small
	 *        800x800 grid (validation). Sommerfeld BC on all 4 sides, no sponge.
	 *        Source: plane wave strip j in [5, 15), i in [200, 500). 1700 timesteps.
	 */
	static SingleLineScenario BuildTestCoriolisV7_SmallSN(CoordSystem cs = CoordSystem::Spherical);

	/**
	 * \brief Coriolis v7: S→N plane wave with finite horizontal extent on the
	 *        production 1200x1200 grid. Same pattern as SmallSN but scaled up.
	 *        Source: j in [5, 15), i in [300, 800). 2500 timesteps.
	 */
	static SingleLineScenario BuildTestCoriolisV7_ProductionSN(CoordSystem cs = CoordSystem::Spherical);

	// ==================================================================
	// Verification сценарии для Sommerfeld radiation BC
	// ==================================================================

	/**
	 * \brief Lake-at-rest для проверки Sommerfeld BC: η=0, u=v=0, flat bottom.
	 *        500 шагов. Критерий: max|η|, max|u|, max|v| < 1e-12.
	 */
	static SingleLineScenario BuildRadiationLakeAtRest();

	/**
	 * \brief Gaussian pulse в центре: эллипсоидальная перегородка, flat bottom,
	 *        Sommerfeld BC. 1000 шагов. Критерий: после ухода волны через
	 *        границы max|η| в центральной области падает значительно (R<5%).
	 */
	static SingleLineScenario BuildRadiationGaussianPulse();

	/**
	 * \brief §5 Coriolis sweep по широтам (Work_Plan_2026-05-12).
	 *        Квадратный домен ny ≈ (lat_max-lat_min)·111.3 ячеек,
	 *        dx=dy=1000m, плоское дно 1000m. S→N плоская волна
	 *        конечной ширины (≈37.5% nx), 2500 шагов × τ=5s.
	 *        Используется для построения «отклонение vs широта».
	 */
	static SingleLineScenario BuildTestCoriolisLatBand(
		double lat_min_deg, double lat_max_deg,
		CoordSystem cs = CoordSystem::Spherical);

	/**
	 * \brief §1 Test L: benchmark scaffold для num_subgrids sweep.
	 *        5000×5000 Cartesian, объём данных ≈ 2 GB ≫ L3=33 MB.
	 *        S→N плоская волна i ∈ [1000,4000), 1000 шагов.
	 */
	static SingleLineScenario BuildTestL_Benchmark();

	/**
	 * \brief Тихий океан, Курило-Камчатский источник 2025.
	 *        Bath: Data/pacific_1min/pacific_1min.asc (9601×6781, 1', xll=129.99°E, yll=-53°N).
	 *        Источник: Data/pacific_sour2025_1min/sour2025_1min.asc (843×541, raw matrix).
	 *        Lower-left угол источника на (col=1231, row_from_south=6000) — Курильские острова / Камчатка.
	 *        sub_size > 0 — берёт квадрат subregion вокруг центра источника (для отладки).
	 *        sub_size = 0 — полный домен (требует ~13 GB памяти).
	 */
	static SingleLineScenario BuildPacificKamchatka(bool coriolis_enabled = true,
	                                                 int sub_size = 3000);

	// Перенос логики из SingleLineGridSimulation::SetParams(IBaseParametersParser*)
	static SingleLineScenario BuildFromParameters(IBaseParametersParser* parameters);

	static void ApplyScenarioToSimulation(SingleLineScenario& scenario,
		SingleLineGridSimulation& simulation);
};
