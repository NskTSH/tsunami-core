#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <sstream>
#include <functional>

#include "ScenarioBuilder.h"
#include "BoundaryCondition.h"  // WavemakerData::Load for Monai scenario

namespace
{
    // Индексы 0..6: 0..3 → ExchangeBoundaryData1..4, 4 → Sommerfeld radiation,
    // 5 → Flather, 6 → Monai wavemaker (Phase 8 NTHMP)
    int ClampBoundaryIndex(int idx)
    {
        if (idx < 0) return 0;
        if (idx > 6) return 6;
        return idx;
    }
}

// === Тест 1 ===============================================================
// Прямоугольная волна 10x10 в центре (SetParamsTest1)
SingleLineScenario ScenarioBuilder::BuildTest1()
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 120;
    s.ny = 120;
    s.timesteps = 500;
    s.num_threads = 8;
    s.coord_system = CoordSystem::Cartesian;

    s.exchangeBoundaryIndex = 0; // -> SubGrid::ExchangeBoundaryData1
    s.exchangeHatBoundaryIndex = 0; // -> SubGrid::ExchangeHatBoundaryData1

    // Переинициализация глобальной сетки как в SetParamsTest1
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsTest1();
    s.has_own_grid = true;

    return s;
}

// === Тест 2 ===============================================================
// Круговой фронт из центра (SetParamsTest3)
SingleLineScenario ScenarioBuilder::BuildTest2()
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 1000;
    s.ny = 1000;
    s.timesteps = 4500;
    s.num_threads = 8;
    s.coord_system = CoordSystem::Cartesian;

    // В исходном коде: ExchangeBoundaryData1 / ExchangeHatBoundaryData1
    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;

    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsTest2();
    s.has_own_grid = true;

    return s;
}

// === Тест 3 ===============================================================
// Плоский фронт от левого края поворачивает на 90 градусов. Дно квадратично углубляется
SingleLineScenario ScenarioBuilder::BuildTest3()
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 2.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 1000;
    s.ny = 1000;
    s.timesteps = 8000;
    s.num_threads = 8;
    s.coord_system = CoordSystem::Cartesian;

    // В исходном коде: ExchangeBoundaryData3 / ExchangeHatBoundaryData3
    s.exchangeBoundaryIndex = 2; // 0→1, 1→2, 2→3
    s.exchangeHatBoundaryIndex = 2;

    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsTest3(s.tau);
    s.has_own_grid = true;

    return s;
}

// === Тест 4 ===============================================================
// Линза Люнеберга (SetParamsTest4)
SingleLineScenario ScenarioBuilder::BuildTest4()
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 2.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 1000;
    s.ny = 1000;
    s.timesteps = 9000;
    s.num_threads = 8;
    s.coord_system = CoordSystem::Cartesian;

    // В исходном коде: ExchangeBoundaryData4 / ExchangeHatBoundaryData4
    s.exchangeBoundaryIndex = 3;
    s.exchangeHatBoundaryIndex = 3;

    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsTest4(s.tau);
    s.has_own_grid = true;

    return s;
}

// === Тест 5 ===============================================================
// Круглый остров посередине
SingleLineScenario ScenarioBuilder::BuildTest5()
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 2.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 1000;
    s.ny = 1000;
    s.timesteps = 6000;
    s.num_threads = 8;
    s.coord_system = CoordSystem::Cartesian;

    // В исходном коде: ExchangeBoundaryData4 / ExchangeHatBoundaryData4
    s.exchangeBoundaryIndex = 3;
    s.exchangeHatBoundaryIndex = 3;

    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsTest5(s.tau);
    s.has_own_grid = true;

    return s;
}

// === Helper: Japan Sea scenario with configurable par file ==================
static SingleLineScenario BuildJapanSeaScenario(const std::string& parFile,
                                                 const std::string& ascFile = "Data/JapanSea.asc")
{
    SingleLineScenario s;

    // Читаем файл параметров
    DefaultParametersParser parParser;
    parParser.Read(parFile);

    // Читаем файл батиметрии
    ASCParser ascParser;
    ascParser.Read(ascFile);

    s.g = 9.81;
    s.tau = parParser.GetDeltaT();
    s.timesteps = parParser.GetTimesteps();
    s.num_threads = parParser.GetNumThreads();
    s.num_subgrids = parParser.GetNumSubgrids();
    s.coord_system = parParser.GetCoordSystem();

    s.nx = ascParser.GetCols();
    s.ny = ascParser.GetRows();

    // delta_x зависит от широты (долготный шаг в метрах сужается к полюсам)
    double yllcorner = ascParser.GetYllCorner();
    double cellsize = ascParser.GetCellSize();
    int ny = s.ny;
    s.delta_x = [yllcorner, cellsize, ny](int i, int j) {
        double latitude = yllcorner + (ny - j) * cellsize;
        return 111300.0 * cellsize * std::cos(latitude * M_PI / 180.0);
    };
    s.delta_y = 462.5;

    int bType = parParser.GetBoundaryExchangeType();
    int hatType = parParser.GetHatBoundaryExchangeType();
    s.exchangeBoundaryIndex = ClampBoundaryIndex(bType == -1 ? 0 : bType);
    s.exchangeHatBoundaryIndex = ClampBoundaryIndex(hatType == -1 ? 0 : hatType);

    s.minimum_depth_to_calculate = parParser.GetMinDepthToCalc();

    // Центральная широта для сферической системы координат (из данных ASC)
    double center_lat_deg = yllcorner + (s.ny / 2.0) * cellsize;
    s.center_latitude_rad = center_lat_deg * M_PI / 180.0;

    // Строим сетку с батиметрией и источниками
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsTest6(s.tau, ascParser, parParser);
    s.has_own_grid = true;

    return s;
}

SingleLineScenario ScenarioBuilder::BuildTest6()
{
    return BuildJapanSeaScenario("Data/JapanSeaOneSourceS2.par");
}

// === Тест 7 ===============================================================
// "Линза Люнеберга с двумя радиусами"
// В текущем коде SetParamsTest7 полностью совпадает с SetParamsTest6.
SingleLineScenario ScenarioBuilder::BuildTest7()
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 0.5;
    s.timesteps = 18000;
    s.num_threads = 4;
    s.coord_system = CoordSystem::Cartesian;

    s.exchangeBoundaryIndex = 0; // ExchangeBoundaryData1
    s.exchangeHatBoundaryIndex = 0; // ExchangeHatBoundaryData1

    s.has_own_grid = false;
    s.nx = -1;
    s.ny = -1;
    s.delta_x = [](int, int) { return 0.0; };
    s.delta_y = 0.0;

    return s;
}

// === Сценарий по файлу параметров ========================================
// Это перенос логики из SingleLineGridSimulation::SetParams(IBaseParametersParser*)
SingleLineScenario ScenarioBuilder::BuildFromParameters(IBaseParametersParser* parameters)
{
    SingleLineScenario s;

    if (!parameters)
        return s;

    s.g = 9.81;
    s.timesteps = parameters->GetTimesteps();
    s.num_threads = parameters->GetNumThreads();
	s.num_subgrids = parameters->GetNumSubgrids();
    s.tau = parameters->GetDeltaT();
    s.coord_system = parameters->GetCoordSystem();

    const double deltaX = parameters->GetDeltaXStep();
    s.delta_x = [deltaX](int, int)
        {
            // границы долготы здесь от i,j не зависят
            return deltaX;
        };
    s.delta_y = parameters->GetDeltaYStep();

    int bType = parameters->GetBoundaryExchangeType();
    int hatType = parameters->GetHatBoundaryExchangeType();

    s.exchangeBoundaryIndex = ClampBoundaryIndex(bType == -1 ? 0 : bType);
    s.exchangeHatBoundaryIndex = ClampBoundaryIndex(hatType == -1 ? 0 : hatType);

    s.minimum_depth_to_calculate = parameters->GetMinDepthToCalc();

    // Nx, Ny и сетка задаются потом в Initialize(...),
    // поэтому здесь их не трогаем
    s.nx = -1;
    s.ny = -1;
    s.has_own_grid = false;

    return s;
}

// === Применение сценария к симулятору ====================================
void ScenarioBuilder::ApplyScenarioToSimulation(SingleLineScenario& scenario,
    SingleLineGridSimulation& simulation)
{
    simulation.SetParameters(
        scenario.g,
        scenario.tau,
        scenario.delta_x,
        scenario.delta_y,
        scenario.nx,
        scenario.ny,
        scenario.timesteps,
		scenario.num_threads,
		scenario.num_subgrids,
        scenario.exchangeBoundaryIndex,
        scenario.exchangeHatBoundaryIndex,
        scenario.coord_system,
        scenario.center_latitude_rad,
        scenario.minimum_depth_to_calculate
    );
    simulation.SetDecompDirection(scenario.decomp_dir);
    simulation.SetSpongeLayer(
        scenario.sponge_west_cells,
        scenario.sponge_east_cells,
        scenario.sponge_south_cells,
        scenario.sponge_north_cells,
        scenario.sponge_sigma_max);
    simulation.SetInteriorWindow(
        scenario.interior_i_min, scenario.interior_i_max,
        scenario.interior_j_min, scenario.interior_j_max);
    simulation.SetCoriolisEnabled(scenario.coriolis_enabled);
    simulation.SetSaveVelocities(scenario.save_velocities);

    if (scenario.has_own_grid)
    {
        // здесь мы двигаем только Grid, а не весь Scenario
        simulation.SetGrid(std::move(scenario.grid));
    }
}

// ============================================================================
// === SPHERICAL COORDINATE VERSIONS (с учётом силы Кориолиса) ================
// ============================================================================

// === Тест 1 (Сферический) ===================================================
// Прямоугольная волна 10x10 в центре, сферическая система координат
SingleLineScenario ScenarioBuilder::BuildTest1Spherical()
{
    // Базируемся на декартовом Test1, чтобы координаты и начальные условия
    // оставались 1-в-1 идентичными сценарию 1.
    SingleLineScenario s = BuildTest1();

    // Отличие только в системе координат и учете Кориолиса в solver-ветке.
    s.coord_system = CoordSystem::Spherical;
    s.center_latitude_rad = 0.785398163; // 45°

    return s;
}

// === Тест 2 (Сферический) ===================================================
// Круговой фронт из центра, сферическая система координат
SingleLineScenario ScenarioBuilder::BuildTest2Spherical()
{
    // Базируемся на декартовом Test2, чтобы координаты и начальные условия
    // оставались 1-в-1 идентичными сценарию 2.
    SingleLineScenario s = BuildTest2();

    // Отличие только в системе координат и учете Кориолиса в solver-ветке.
    s.coord_system = CoordSystem::Spherical;
    s.center_latitude_rad = 0.785398163; // 45°

    return s;
}

// === Тест 3 (Сферический) ===================================================
// Плоский фронт от левого края поворачивает на 90 градусов, сферическая система координат
SingleLineScenario ScenarioBuilder::BuildTest3Spherical()
{
    // Базируемся на декартовом Test3, чтобы координаты и начальные условия
    // оставались 1-в-1 идентичными сценарию 3.
    SingleLineScenario s = BuildTest3();

    // Отличие только в системе координат и учете Кориолиса в solver-ветке.
    s.coord_system = CoordSystem::Spherical;
    s.center_latitude_rad = 0.785398163; // 45°

    return s;
}

// === Тест 4 (Сферический) ===================================================
// Линза Люнеберга, сферическая система координат
SingleLineScenario ScenarioBuilder::BuildTest4Spherical()
{
    // Базируемся на декартовом Test4, чтобы координаты и начальные условия
    // оставались 1-в-1 идентичными сценарию 4.
    SingleLineScenario s = BuildTest4();

    // Отличие только в системе координат и учете Кориолиса в solver-ветке.
    s.coord_system = CoordSystem::Spherical;
    s.center_latitude_rad = 0.785398163; // 45°

    return s;
}

// === Тест 5 (Сферический) ===================================================
// Круглый остров посередине, сферическая система координат
SingleLineScenario ScenarioBuilder::BuildTest5Spherical()
{
    // Базируемся на декартовом Test5, чтобы координаты и начальные условия
    // оставались 1-в-1 идентичными сценарию 5.
    SingleLineScenario s = BuildTest5();

    // Отличие только в системе координат и учете Кориолиса в solver-ветке.
    s.coord_system = CoordSystem::Spherical;
    s.center_latitude_rad = 0.785398163; // 45°

    return s;
}

// === Тест 6 (Сферический) ===================================================
// "Японское море", сферическая система координат
SingleLineScenario ScenarioBuilder::BuildTest6Spherical()
{
    // Базируемся на декартовом Test6, чтобы координаты и начальные условия
    // оставались 1-в-1 идентичными сценарию 6.
    SingleLineScenario s = BuildTest6();

    // Отличие только в системе координат и учете Кориолиса в solver-ветке.
    // center_latitude_rad уже вычислен в BuildTest6() из данных ASC-файла.
    s.coord_system = CoordSystem::Spherical;

    return s;
}

// === Тест 7 (Сферический) ===================================================
// "Линза Люнеберга с двумя радиусами", сферическая система координат
SingleLineScenario ScenarioBuilder::BuildTest7Spherical()
{
    // Базируемся на декартовом Test7, чтобы координаты и начальные условия
    // оставались 1-в-1 идентичными сценарию 7.
    SingleLineScenario s = BuildTest7();

    // Отличие только в системе координат и учете Кориолиса в solver-ветке.
    s.coord_system = CoordSystem::Spherical;
    s.center_latitude_rad = 0.785398163; // 45°

    return s;
}

// === Кориолис тест 1 =========================================================
// Плоский фронт юг→север через экватор
SingleLineScenario ScenarioBuilder::BuildTestCoriolis1(CoordSystem cs)
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 5.0;          // CFL ≈ 0.5 при c=99 m/s, dx=1000m
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;

    // 20° latitude span ≈ 2226 km → 2226 cells at dy=1000m
    // Use 2200 x 2200 for clean division
    s.nx = 2200;
    s.ny = 2200;
    s.timesteps = 4000;    // ~20000 s = 5.5 hours
    s.num_threads = 12;
    s.num_subgrids = 12;
    s.coord_system = cs;
    s.center_latitude_rad = 0.0;  // Equator
    s.minimum_depth_to_calculate = 8.0;

    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;

    // Plane wave: 10-cell-wide strip at bottom with amplitude 1m
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsCoriolis(1000.0, 10, 1.0);
    s.has_own_grid = true;

    return s;
}

// === Кориолис тест 3 (эллипсоидный, N-S) =====================================
// Эллипсоидный источник вытянутый по широте: 50x200 км
SingleLineScenario ScenarioBuilder::BuildTestCoriolisEllipsoidal1(CoordSystem cs)
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;

    s.nx = 2200;
    s.ny = 2200;
    s.timesteps = 4000;
    s.num_threads = 12;
    s.num_subgrids = 12;
    s.coord_system = cs;
    s.center_latitude_rad = 0.0;
    s.minimum_depth_to_calculate = 8.0;

    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;

    // Ellipsoidal source: r1=50 cells (50km), r2=200 cells (200km), angle=0°, amplitude=1m
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsCoriolisEllipsoidal(1000.0, 50.0, 200.0, 0.0, 1.0);
    s.has_own_grid = true;

    return s;
}

// === Кориолис тест 4 (эллипсоидный, W-E) =====================================
// Эллипсоидный источник вытянутый по долготе: 200x50 км
SingleLineScenario ScenarioBuilder::BuildTestCoriolisEllipsoidal2(CoordSystem cs)
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;

    s.nx = 2200;
    s.ny = 2200;
    s.timesteps = 4000;
    s.num_threads = 12;
    s.num_subgrids = 12;
    s.coord_system = cs;
    s.center_latitude_rad = 0.0;
    s.minimum_depth_to_calculate = 8.0;

    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;

    // Ellipsoidal source: r1=200 cells (200km), r2=50 cells (50km), angle=0°, amplitude=1m
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsCoriolisEllipsoidal(1000.0, 200.0, 50.0, 0.0, 1.0);
    s.has_own_grid = true;

    return s;
}

// === Кориолис тест 2 =========================================================
// Плоский фронт запад→восток через экватор
SingleLineScenario ScenarioBuilder::BuildTestCoriolis2(CoordSystem cs)
{
    SingleLineScenario s;

    s.g = 9.81;
    s.tau = 5.0;          // CFL ≈ 0.5 при c=99 m/s, dx=1000m
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;

    s.nx = 2200;
    s.ny = 2200;
    s.timesteps = 4000;    // ~20000 s = 5.5 hours
    s.num_threads = 12;
    s.num_subgrids = 12;
    s.coord_system = cs;
    s.center_latitude_rad = 0.0;  // Equator
    s.minimum_depth_to_calculate = 8.0;
    s.decomp_dir = DecompositionDirection::Vertical;  // Vertical decomposition for W→E wave

    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;

    // Plane wave: 10-cell-wide strip at left edge with amplitude 1m
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsCoriolisHorizontal(1000.0, 10, 1.0);
    s.has_own_grid = true;

    return s;
}

// === Кориолис v2: увеличенная сетка, φ₀=45° ================================
// Цель: наблюдение реального Кориолис-эффекта (f ≈ 1.03e-4 на 45°)

static SingleLineScenario BuildCoriolisV2Base(CoordSystem cs) {
    SingleLineScenario s;
    s.g = 9.81;
    s.tau = 5.0;          // CFL ≈ 0.5 при c=99 m/s, dx=1000m
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 4000;
    s.ny = 4000;
    // v6: сокращено до 6000 шагов (2026-04-22 revision) — источник локализован
    // (см. BuildTestCoriolisV2_SN / _WE), до ближайшей стенки ~2000 км ≈ 4000
    // шагов. 6000 шагов = 30000 с ≈ 8.3 ч ≈ 0.5 инерционного периода на 45°:
    // пульс успевает отклониться силой Кориолиса, но Kelvin-мода у стенки
    // не успевает сформироваться.
    s.timesteps = 6000;
    s.num_threads = 12;
    s.num_subgrids = 12;
    s.coord_system = cs;
    s.center_latitude_rad = 0.785398163;  // 45° — f = 2Ω sin(45°) ≈ 1.03e-4
    s.minimum_depth_to_calculate = 8.0;
    // Sommerfeld radiation на всех 4 границах (Codex plan 2026-04-10).
    s.exchangeBoundaryIndex = 4;
    s.exchangeHatBoundaryIndex = 4;
    // Sponge layer параметры задаются отдельно в каждом конкретном
    // Build* сценарии, потому что направление волны и положение источника
    // определяют, на каких сторонах sponge применим без повреждения источника.
    s.sponge_sigma_max = 0.002;  // рекомендация Codex для c=99, dx=1000
    // Interior окно по умолчанию исключает 300-ячеечные полоски по всем сторонам
    // (300 > max sponge_width=200 + запас). Конкретные тесты могут переопределить.
    s.interior_i_min = 300;
    s.interior_i_max = 3700;
    s.interior_j_min = 300;
    s.interior_j_max = 3700;
    return s;
}

SingleLineScenario ScenarioBuilder::BuildTestCoriolisV2_SN(CoordSystem cs)
{
    SingleLineScenario s = BuildCoriolisV2Base(cs);
    // v6 (2026-04-22): локализованный эллиптический источник вместо сплошной
    // плоской волны — устраняет немедленное касание восточной стенки и
    // формирование трапированной Kelvin-моды у правого края домена.
    // Центр (i=2000, j=1000): по центру по долготе, на 1/4 высоты по широте
    // (южнее центра, расстояние до восточной стенки ~2000 км ≈ 4000 шагов).
    // Радиус 150 км × 150 км, амплитуда 1 м.
    // Все границы — Sommerfeld radiation, sponge отключены.
    s.sponge_west_cells = 0;
    s.sponge_east_cells = 0;
    s.sponge_south_cells = 0;
    s.sponge_north_cells = 0;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsCoriolisEllipsoidal(1000.0, 150.0, 150.0, 0.0, 1.0,
                                         2000.0, 1000.0);
    s.has_own_grid = true;
    return s;
}

SingleLineScenario ScenarioBuilder::BuildTestCoriolisV2_WE(CoordSystem cs)
{
    SingleLineScenario s = BuildCoriolisV2Base(cs);
    s.decomp_dir = DecompositionDirection::Vertical;
    // v6 (2026-04-22): локализованный эллиптический источник симметрично
    // SN-кейсу. Центр (i=1000, j=2000): по центру по широте, на 1/4 ширины
    // от запада. Радиус 150 км × 150 км, амплитуда 1 м.
    s.sponge_west_cells = 0;
    s.sponge_east_cells = 0;
    s.sponge_south_cells = 0;
    s.sponge_north_cells = 0;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsCoriolisEllipsoidal(1000.0, 150.0, 150.0, 0.0, 1.0,
                                         1000.0, 2000.0);
    s.has_own_grid = true;
    return s;
}

SingleLineScenario ScenarioBuilder::BuildTestCoriolisV2_EllipNS(CoordSystem cs)
{
    SingleLineScenario s = BuildCoriolisV2Base(cs);
    // Эллипсоидный источник 50x200km в центре домена — далеко от всех стенок.
    // Sponge на всех 4 сторонах для устранения Kelvin waves у любого борта.
    s.sponge_west_cells = 200;
    s.sponge_east_cells = 200;
    s.sponge_south_cells = 200;
    s.sponge_north_cells = 200;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsCoriolisEllipsoidal(1000.0, 50.0, 200.0, 0.0, 1.0);  // N-S 50x200km
    s.has_own_grid = true;
    return s;
}

SingleLineScenario ScenarioBuilder::BuildTestCoriolisV2_EllipWE(CoordSystem cs)
{
    SingleLineScenario s = BuildCoriolisV2Base(cs);
    s.sponge_west_cells = 200;
    s.sponge_east_cells = 200;
    s.sponge_south_cells = 200;
    s.sponge_north_cells = 200;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsCoriolisEllipsoidal(1000.0, 200.0, 50.0, 0.0, 1.0);  // W-E 200x50km
    s.has_own_grid = true;
    return s;
}

// ==========================================================================
//  Coriolis v7: S→N plane wave, finite horizontal extent (2026-04-23)
//
//  Design (после Codex review 2026-04-23):
//  - Источник — плоская волна шириной 10 ячеек (j ∈ [5, 15)) с конечной
//    горизонтальной протяжённостью (i ∈ [wave_i_min, wave_i_max)).
//  - Бока источника с большими отступами от западной и восточной границ.
//  - Только Sommerfeld BC, без sponge (запрет автора).
//  - Время расчёта: ровно чтобы фронт прошёл сетку и успел уйти через N-границу.
//  - Физика: за ~0.13 инерционного периода на 45° отклонение частиц
//    Кориолисом ~ 0.5·f·c·t². На валидационной сетке 800×800 это ~135 км,
//    на production 1200×1200 за 2424 шага ~300 км. East-margin не меньше
//    отклонения.
// ==========================================================================

SingleLineScenario ScenarioBuilder::BuildTestCoriolisV7_SmallSN(CoordSystem cs)
{
    SingleLineScenario s;
    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 800;
    s.ny = 800;
    s.timesteps = 1700;
    s.num_threads = 12;
    s.num_subgrids = 12;
    s.coord_system = cs;
    s.center_latitude_rad = 0.785398163;  // 45°
    s.minimum_depth_to_calculate = 8.0;
    // Zero-gradient BC (ExchangeBoundaryData1) на всех 4 сторонах — то же
    // open-boundary условие, что использовалось в Test1/Test2 и работало
    // чисто. Sommerfeld (index=4) не применяем: Codex показал, что для волн
    // с наклонным фронтом он частично отражает.
    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;
    // Sponge строго 0 на всех сторонах (feedback 2026-04-23)
    s.sponge_west_cells = 0;
    s.sponge_east_cells = 0;
    s.sponge_south_cells = 0;
    s.sponge_north_cells = 0;
    s.sponge_sigma_max = 0.0;
    s.interior_i_min = 0;
    s.interior_i_max = s.nx;
    s.interior_j_min = 0;
    s.interior_j_max = s.ny;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    // Плоская волна: j ∈ [5, 15), i ∈ [200, 500). 300 км ширина, 200 км
    // запад-отступ, 300 км восток-отступ.
    s.grid.SetParamsCoriolisFinitePlane(1000.0, 5, 15, 200, 500, 1.0);
    s.has_own_grid = true;
    return s;
}

SingleLineScenario ScenarioBuilder::BuildTestCoriolisV7_ProductionSN(CoordSystem cs)
{
    SingleLineScenario s;
    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    // v13 (2026-04-28): 4500 × 3000 (i × j). Расширенная по i сетка для
    // вытеснения угловых артефактов открытой границы за пределы зоны
    // отображения; в постпроцессинге показываем средние 2500 столбцов
    // i ∈ [1000, 3500) (тот же физический сектор, что был в v11).
    // Источник занимает всю ширину домена по i (Б2-вариант — однородная
    // плоская волна S→N), длина 15 км по j. Начальная скорость
    // v0 = +c·η/D задаёт чистую бегущую волну на север; u0 = 0.
    s.nx = 4500;
    s.ny = 3000;
    s.timesteps = 10000;  // ~10000 шагов × τ=5s = 50000s ≈ 14 часов
    s.num_threads = 12;
    s.num_subgrids = 12;
    s.coord_system = cs;
    s.center_latitude_rad = 0.785398163;
    s.minimum_depth_to_calculate = 8.0;
    // Zero-gradient BC — то же условие, что в Test1/Test2.
    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;
    s.sponge_west_cells = 0;
    s.sponge_east_cells = 0;
    s.sponge_south_cells = 0;
    s.sponge_north_cells = 0;
    s.sponge_sigma_max = 0.0;
    s.interior_i_min = 0;
    s.interior_i_max = s.nx;
    s.interior_j_min = 0;
    s.interior_j_max = s.ny;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    // v13 (2026-04-29): источник однороден по всей ширине сетки
    // i ∈ [0, 4500) — Б2-вариант (плоская волна без поперечных кромок).
    // Сетка 4500 при отображении центральных 2500 ячеек выводит угловые
    // артефакты двумерной экстраполяции открытой границы за пределы
    // зоны отображения. Длина волны по j = 15 ячеек (j ∈ [5, 20)).
    s.grid.SetParamsCoriolisFinitePlane(1000.0, 5, 20, 0, s.nx, 1.0,
        /*u0*/ 0.0, /*v0*/ 0.0990195135927845);
    s.has_own_grid = true;
    return s;
}

// ==========================================================================
//  Verification сценарии для Sommerfeld radiation BC (2026-04-10)
// ==========================================================================

// Тест 101: Lake-at-rest. Flat bottom D=1000 м, η=0, u=v=0.
// Sommerfeld BC + sponge layer не должны сами генерировать волну.
// Критерий: max|η|, max|u|, max|v| < 1e-12 на всех 500 шагах.
SingleLineScenario ScenarioBuilder::BuildRadiationLakeAtRest()
{
    SingleLineScenario s;
    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 200;
    s.ny = 200;
    s.timesteps = 500;
    s.num_threads = 4;
    s.num_subgrids = 4;
    s.coord_system = CoordSystem::Cartesian;
    s.center_latitude_rad = 0.0;
    s.minimum_depth_to_calculate = 8.0;
    s.exchangeBoundaryIndex = 4;       // Radiation на всех сторонах
    s.exchangeHatBoundaryIndex = 4;
    // Sponge тоже проверяем: не должен генерировать шум из нулевого состояния
    s.sponge_west_cells = 20;
    s.sponge_east_cells = 20;
    s.sponge_south_cells = 20;
    s.sponge_north_cells = 20;
    s.sponge_sigma_max = 0.002;
    s.interior_i_min = 30;
    s.interior_i_max = 170;
    s.interior_j_min = 30;
    s.interior_j_max = 170;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    // wave_width=0 → цикл не выполнится, останется только flat bottom
    s.grid.SetParamsCoriolis(1000.0, 0, 0.0);
    s.has_own_grid = true;
    return s;
}

// Тест 102: Gaussian-like эллипсоидальный импульс в центре, Sommerfeld + sponge.
// Волна должна уйти через границы с малым отражением.
// Критерий: после достижения границы (~step 100-200) энергия в центре
//           падает монотонно; к step 1000 max|η| в центральной области < 5%
//           от начального 1.0 m. max_eta_interior должен упасть до ~0.
SingleLineScenario ScenarioBuilder::BuildRadiationGaussianPulse()
{
    SingleLineScenario s;
    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 400;
    s.ny = 400;
    s.timesteps = 1000;
    s.num_threads = 4;
    s.num_subgrids = 4;
    s.coord_system = CoordSystem::Cartesian;
    s.center_latitude_rad = 0.0;
    s.minimum_depth_to_calculate = 8.0;
    s.exchangeBoundaryIndex = 4;       // Radiation на всех сторонах
    s.exchangeHatBoundaryIndex = 4;
    s.sponge_west_cells = 40;
    s.sponge_east_cells = 40;
    s.sponge_south_cells = 40;
    s.sponge_north_cells = 40;
    s.sponge_sigma_max = 0.002;
    s.interior_i_min = 60;
    s.interior_i_max = 340;
    s.interior_j_min = 60;
    s.interior_j_max = 340;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    // Круглый эллипсоидальный импульс радиуса 30 ячеек в центре, amp=1.0
    s.grid.SetParamsCoriolisEllipsoidal(1000.0, 30.0, 30.0, 0.0, 1.0);
    s.has_own_grid = true;
    return s;
}

// === Тест 6 (Декартовый, S7) ================================================
// "Японское море" с источником S7, декартова система координат
SingleLineScenario ScenarioBuilder::BuildTest6CartesianS7()
{
    return BuildJapanSeaScenario("Data/JapanSeaOneSourceS7.par");
}

// === Тест 6 (Сферический, S7) ===============================================
// "Японское море" с источником S7, сферическая система координат
SingleLineScenario ScenarioBuilder::BuildTest6SphericalS7()
{
    SingleLineScenario s = BuildJapanSeaScenario("Data/JapanSeaOneSourceS7.par");
    s.coord_system = CoordSystem::Spherical;
    return s;
}

// === Multi-source Japan Sea (для Active Region validation) ==================
// Использует существующий Data/JapanSeaThreeSources.par; sphererical координаты.
// На этом scenarrio Phase 6 проверяет independent activation components:
// 3 источника пространственно разнесены → 3 ComponentBbox после flood fill.
SingleLineScenario ScenarioBuilder::BuildTest6JapanSeaThreeSources()
{
    SingleLineScenario s = BuildJapanSeaScenario("Data/JapanSeaThreeSources.par");
    s.coord_system = CoordSystem::Spherical;
    return s;
}

// ===========================================================================
//  Okushiri (NTHMP BP9 — Field benchmark, 1993 Hokkaido Nansei-Oki tsunami)
// ---------------------------------------------------------------------------
//  Coarse single-grid (450 m) configuration used here for ACTIVE REGION
//  EQUIVALENCE TEST: baseline (full-domain) vs ACTIVE_REGION_MODE=geometric
//  on a tsunami benchmark-derived geometry. Source: NOAA NCTR Okushiri page,
//  files D379-687-450m.txt (bathymetry, 10F7.1, positive=water depth) and
//  I379-687-450m-17a.txt (initial η perturbation in metres).
//
//  Domain: 379 × 687 cells at 450 m → 170.55 km × 309.15 km.
//  No friction, no wet/dry tweaks here — minimum_depth_to_calculate uses
//  default 8 m so coastal cells are treated as land when D < 8.
//  Boundary: Sommerfeld radiation on all four open ocean sides.
//
//  NOT NTHMP certification — only equivalence-to-baseline validation.

static void LoadOkushiri10F7(const std::string& path, Grid& grid,
                             int nx, int ny, char target /*'D' or 'E'*/)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Okushiri: cannot open " + path);
    }
    // Format 10F7.1 — 10 values per line; each grid row of `nx` values occupies
    // ceil(nx / 10) consecutive lines. File row index 0 corresponds to the
    // West-South corner (j = 0). i runs across columns within each row.
    const int lines_per_row = (nx + 9) / 10;
    for (int j = 0; j < ny; ++j) {
        int col_written = 0;
        for (int line = 0; line < lines_per_row && col_written < nx; ++line) {
            std::string text;
            if (!std::getline(in, text)) {
                throw std::runtime_error("Okushiri: premature EOF in " + path +
                    " at row " + std::to_string(j) + " line " + std::to_string(line));
            }
            std::istringstream iss(text);
            double v;
            while (iss >> v && col_written < nx) {
                if (target == 'D') {
                    grid.D(col_written, j) = static_cast<Real>(v);
                } else if (target == 'E') {  // eta
                    grid.eta(col_written, j) = static_cast<Real>(v);
                }
                ++col_written;
            }
        }
        if (col_written != nx) {
            throw std::runtime_error("Okushiri: row " + std::to_string(j) +
                " produced only " + std::to_string(col_written) + " of " +
                std::to_string(nx) + " values");
        }
    }
}

// ===========================================================================
//  Monai Valley (NTHMP BP6/7 — Laboratory benchmark, 1/400 scale tank)
// ---------------------------------------------------------------------------
//  Coarse 1.4cm grid, 393x244 cells, dt small for CFL stability.
//  Source: NOAA NCTR Monai Valley page.
//    Data/Monai/MonaiValley_Bathymetry.txt   (XYZ pointwise, header line +
//      lines "x y z" with x in [0, 5.488], y in [0, 3.402], z positive=depth)
//    Data/Monai/MonaiValley_InputWave.txt    (Time(s) Water_surface(m), 451
//      samples at 0.05s intervals → 22.5 s total)
//
//  Numerical step tau=0.01s (5x finer than wave samples) so CFL≈0.82 for
//  c=sqrt(g·0.135)≈1.15 m/s and dx=0.014 m. Wavemaker BC samples wave file
//  via linear interpolation in physical time.
//
//  Boundary: WAVEMAKER on LEFT (index 6), zero-gradient walls on other 3 sides.
//  Purpose: active region equivalence test on a wavemaker-driven benchmark.

static void LoadMonaiBathymetryXYZ(const std::string& path, Grid& grid,
                                   int nx, int ny, double dx, double dy)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Monai: cannot open " + path);
    }
    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("Monai: empty bathymetry file");
    }
    std::string line;
    int count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) continue;
        const int i = static_cast<int>(std::round(x / dx));
        const int j = static_cast<int>(std::round(y / dy));
        if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
        grid.D(i, j) = static_cast<Real>(z);
        ++count;
    }
    if (count < nx * ny / 2) {
        throw std::runtime_error("Monai: only " + std::to_string(count) +
            " points loaded, expected ~" + std::to_string(nx * ny));
    }
}

SingleLineScenario ScenarioBuilder::BuildMonaiValley()
{
    SingleLineScenario s;
    s.g = 9.81;
    // CFL_2D = c·τ·(1/dx + 1/dy). For c=sqrt(g·0.135)≈1.15 m/s, dx=dy=0.014m:
    // τ_max ≈ dx / (2·c) ≈ 0.006 s. Use 0.001 for stability margin near runup
    // (near-dry cells can have transient |u|>>sqrt(gH), so we want CFL well<1).
    s.tau = 0.001;
    s.delta_x = [](int, int) { return 0.014; };
    s.delta_y = 0.014;
    s.nx = 393;
    s.ny = 244;
    // Phase E pre-paper (Codex 2026-05-28): MONAI_GRID_HALF=1 switches to
    // dx/2 = 0.007 resampled bathymetry (785x487) for grid-convergence study.
    // Wavemaker file is unchanged; CFL doubles (0.16 -> 0.33), still below
    // FV_CFL_MAX=0.45.
    {
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "MONAI_GRID_HALF");
        const bool half = (buf && std::string(buf) == "1");
        if (buf) free(buf);
        if (half) {
            s.delta_x = [](int, int) { return 0.007; };
            s.delta_y = 0.007;
            s.nx = 785;
            s.ny = 487;
            std::cerr << "[Monai] MONAI_GRID_HALF=1 -> 785x487 at dx=0.007 m" << std::endl;
        }
    }
    s.timesteps = 22500;                  // 22.5 s physical (covers full wave series)
    s.num_threads = 4;
    s.num_subgrids = 4;
    s.coord_system = CoordSystem::Cartesian;
    s.exchangeBoundaryIndex = 6;          // wavemaker left + radiation walls (Codex Phase 8)
    s.exchangeHatBoundaryIndex = 6;
    // Monai max water depth ≈ 0.135 m. Default min_calc_value=8 m would mark
    // EVERY cell as dry, skipping all compute. Use 1 mm so only true land cells
    // (where D ≤ 0) are skipped via the D > 0 check upstream.
    s.minimum_depth_to_calculate = 0.001;

    // Codex Phase 8: enable positivity floor + Audusse hydrostatic reconstruction.
    // Without HR Monai blows up around t=8s when wave hits Monai island
    // (well-balancedness fails at bathymetry step).
    // ENV is read once on first SubGrid construction, set here unless overridden.
    auto enable_env_if_unset = [](const char* name, const char* val) {
        char* existing = nullptr; size_t sz = 0;
        _dupenv_s(&existing, &sz, name);
        if (existing == nullptr) {
            _putenv_s(name, val);
        } else {
            free(existing);
        }
    };
    enable_env_if_unset("MC_WET_DRY_POSITIVITY", "1");
    enable_env_if_unset("MC_HYDROSTATIC_RECONSTRUCTION", "1");
    // Tuned wet/dry damping for Monai stability (sensitivity test 2026-05-28).
    // Fr_max=0.5 (subcritical) is CRITICAL — Fr_max=0.95 (default) blows up.
    // Manning n in {0.025..0.05} both stable; 0.025 is closer to smooth-tank.
    // shallow_depth=0.1 covers most of Monai water column (max depth 0.135m).
    enable_env_if_unset("MC_WET_DRY_FROUDE_MAX", "0.5");
    enable_env_if_unset("MC_WET_DRY_MANNING_N", "0.025");
    enable_env_if_unset("MC_WET_DRY_SHALLOW_DEPTH", "0.1");
    // Phase E P1 (Codex): Monai is wavemaker-driven on LEFT. Active Region
    // seed must explicitly mark the leftmost stripe active or AR will leave
    // everything inactive (initial state is at rest).
    enable_env_if_unset("ACTIVE_REGION_SEED_LEFT", "1");

    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);

    // Bathymetry (positive = water depth in metres; negative = land elevation).
    // Phase E pre-paper: select half-grid file if MONAI_GRID_HALF=1.
    const char* bathy_path = (s.nx == 785)
        ? "Data/Monai/MonaiValley_Bathymetry_half.txt"
        : "Data/Monai/MonaiValley_Bathymetry.txt";
    // Pass the correct dx to the XYZ -> grid mapper so indices align.
    const double bathy_dx = (s.nx == 785) ? 0.007 : 0.014;
    LoadMonaiBathymetryXYZ(bathy_path,
                           s.grid, s.nx, s.ny, bathy_dx, bathy_dx);

    // Initial state: at rest, H = D (water depth).
    for (int i = 0; i < s.nx; ++i) {
        for (int j = 0; j < s.ny; ++j) {
            const Real d = s.grid.D(i, j);
            s.grid.H(i, j) = (d > Real(0)) ? d : Real(0);
            s.grid.u(i, j) = Real(0);
            s.grid.v(i, j) = Real(0);
            s.grid.eta(i, j) = Real(0);
        }
    }

    // Load wavemaker time series ONCE (BC handlers consult global store).
    // Phase E pre-paper (Codex 2026-05-28): env MONAI_NO_WAVEMAKER=1 skips
    // the load so the wavemaker BC falls back to DefaultLeft (zero-gradient).
    // Used to verify well-balancedness on Monai bathymetry without any forcing.
    {
        char* nw = nullptr; size_t nws = 0;
        _dupenv_s(&nw, &nws, "MONAI_NO_WAVEMAKER");
        const bool skip_wm = (nw && std::string(nw) == "1");
        if (nw) free(nw);
        if (!skip_wm) {
            WavemakerData::Load("Data/Monai/MonaiValley_InputWave.txt", s.tau);
        } else {
            std::cerr << "[Monai] MONAI_NO_WAVEMAKER=1 — wavemaker disabled, lake-at-rest test" << std::endl;
        }
    }

    s.grid.MarkInitialized();
    s.has_own_grid = true;
    return s;
}

// ===========================================================================
//  FV Phase A validation scenarios (Codex Audusse design)
// ---------------------------------------------------------------------------
//  BuildLakeAtRestFlat: flat D=100m, η=0, u=v=0. Tests pure FV stability.
//  BuildLakeAtRestStep: D=100m west, D=50m east. Tests Audusse HR well-balanced.
//  Both: Cartesian, 200x200, dx=dy=1000m, tau=5s, reflective boundary.

SingleLineScenario ScenarioBuilder::BuildLakeAtRestFlat()
{
    SingleLineScenario s;
    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 200;
    s.ny = 200;
    s.timesteps = 1000;
    s.num_threads = 4;
    s.num_subgrids = 4;
    s.coord_system = CoordSystem::Cartesian;
    s.exchangeBoundaryIndex = 0;   // Default zero-gradient (Phase A — boundary handling deferred)
    s.exchangeHatBoundaryIndex = 0;
    s.minimum_depth_to_calculate = 1.0;

    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    for (int i = 0; i < s.nx; ++i) {
        for (int j = 0; j < s.ny; ++j) {
            s.grid.D(i, j)   = Real(100);
            s.grid.H(i, j)   = Real(100);
            s.grid.u(i, j)   = Real(0);
            s.grid.v(i, j)   = Real(0);
            s.grid.eta(i, j) = Real(0);
        }
    }
    s.grid.MarkInitialized();
    s.has_own_grid = true;
    return s;
}

SingleLineScenario ScenarioBuilder::BuildLakeAtRestStep()
{
    // Phase E pre-paper (Codex 2026-05-28): MC_STEP_DELTA_D env overrides
    // the east-half depth so the Audusse ablation scaling study (ΔD = 10,
    // 20, 30, 50 m) reuses a single scenario binary.
    Real D_west = Real(100);
    Real D_east = Real(50);
    {
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "MC_STEP_DELTA_D");
        if (buf) {
            // Strict full-string parse with end-pointer validation, so e.g.
            // MC_STEP_DELTA_D="50garbage" is rejected rather than silently
            // accepted as 50. Reject NaN/Inf and out-of-range values with an
            // explicit warning, and log the effective value when accepted so
            // the reproducibility log is unambiguous.
            const char* p = buf;
            char* end = nullptr;
            const double dd = std::strtod(p, &end);
            const bool parse_ok =
                (end != nullptr) && (end != p) && (*end == '\0') &&
                std::isfinite(dd);
            if (!parse_ok) {
                std::cerr << "[Scenario81] WARNING: MC_STEP_DELTA_D='" << buf
                          << "' could not be fully parsed as a finite number; "
                          << "ignoring and using default D_east=" << D_east << " m"
                          << std::endl;
            } else if (dd <= 0.0 || dd >= static_cast<double>(D_west)) {
                std::cerr << "[Scenario81] WARNING: MC_STEP_DELTA_D=" << dd
                          << " is out of range (0, " << D_west
                          << "); ignoring and using default D_east=" << D_east << " m"
                          << std::endl;
            } else {
                D_east = static_cast<Real>(D_west - Real(dd));
                std::cerr << "[Scenario81] MC_STEP_DELTA_D=" << dd
                          << " accepted; D_east=" << D_east << " m, effective ΔD="
                          << (D_west - D_east) << " m" << std::endl;
            }
            free(buf);
        }
    }
    SingleLineScenario s = BuildLakeAtRestFlat();
    const int half = s.nx / 2;
    for (int i = 0; i < s.nx; ++i) {
        const Real D_val = (i < half) ? D_west : D_east;
        for (int j = 0; j < s.ny; ++j) {
            s.grid.D(i, j) = D_val;
            s.grid.H(i, j) = D_val;       // H = D at rest
            s.grid.eta(i, j) = Real(0);   // eta = H - D = 0
        }
    }
    return s;
}

// ===========================================================================
//  FV Phase B validation scenarios
// ---------------------------------------------------------------------------

SingleLineScenario ScenarioBuilder::BuildDamBreakWetWet()
{
    SingleLineScenario s = BuildLakeAtRestFlat();
    // Reset D for shallow-water dam-break. D=10, H_left=15 (η=+5), H_right=10 (η=0).
    const int half = s.nx / 2;
    for (int i = 0; i < s.nx; ++i) {
        for (int j = 0; j < s.ny; ++j) {
            s.grid.D(i, j) = Real(10);
            const Real H_val = (i < half) ? Real(15) : Real(10);
            s.grid.H(i, j)   = H_val;
            s.grid.eta(i, j) = H_val - Real(10);  // +5 or 0
        }
    }
    s.timesteps = 200;
    return s;
}

SingleLineScenario ScenarioBuilder::BuildDamBreakWetDry()
{
    SingleLineScenario s = BuildLakeAtRestFlat();
    // D=10 base. Left wet H=10, right dry H=0 (Ritter problem).
    const int half = s.nx / 2;
    for (int i = 0; i < s.nx; ++i) {
        for (int j = 0; j < s.ny; ++j) {
            s.grid.D(i, j) = Real(10);
            if (i < half) {
                s.grid.H(i, j) = Real(10);
                s.grid.eta(i, j) = Real(0);
            } else {
                s.grid.H(i, j) = Real(0);
                s.grid.eta(i, j) = Real(-10);  // surface at bed
            }
        }
    }
    s.timesteps = 200;
    return s;
}

SingleLineScenario ScenarioBuilder::BuildReflectiveBox()
{
    SingleLineScenario s = BuildLakeAtRestFlat();
    // Gaussian bump in center: η = 5 * exp(-r²/(2σ²)), σ = 10 cells.
    const double cx = s.nx * 0.5;
    const double cy = s.ny * 0.5;
    const double sigma = 10.0;
    const double amp = 5.0;
    for (int i = 0; i < s.nx; ++i) {
        for (int j = 0; j < s.ny; ++j) {
            const double dx = i - cx;
            const double dy = j - cy;
            const double r2 = dx * dx + dy * dy;
            const double bump = amp * std::exp(-r2 / (2.0 * sigma * sigma));
            s.grid.eta(i, j) = static_cast<Real>(bump);
            s.grid.H(i, j) = s.grid.D(i, j) + static_cast<Real>(bump);
        }
    }
    s.timesteps = 5000;
    return s;
}

SingleLineScenario ScenarioBuilder::BuildClosedBasinFV()
{
    // Phase E pre-paper mass-conservation diagnostic (Codex 2026-05-28).
    // Closed reflective basin for FV Godunov: 2-cell D=0 perimeter around
    // a D=100m interior so wall detection (D<=0 && h<=dry_tol) triggers
    // and reflects the wave. Initial Gaussian eta bump in centre; mass
    // sum(eta) over interior cells must stay constant to FP precision.
    SingleLineScenario s = BuildLakeAtRestFlat();
    s.timesteps = 5000;

    const int wall = 2;       // perimeter thickness (D=0 cells, 2 ≥ FV stencil)
    const double cx = s.nx * 0.5;
    const double cy = s.ny * 0.5;
    const double sigma = 10.0;
    const double amp = 5.0;
    for (int i = 0; i < s.nx; ++i) {
        for (int j = 0; j < s.ny; ++j) {
            const bool in_wall = (i < wall || i >= s.nx - wall ||
                                  j < wall || j >= s.ny - wall);
            if (in_wall) {
                s.grid.D(i, j)   = Real(0);
                s.grid.H(i, j)   = Real(0);
                s.grid.u(i, j)   = Real(0);
                s.grid.v(i, j)   = Real(0);
                s.grid.eta(i, j) = Real(0);
            } else {
                const double dx = i - cx;
                const double dy = j - cy;
                const double r2 = dx * dx + dy * dy;
                const double bump = amp * std::exp(-r2 / (2.0 * sigma * sigma));
                s.grid.D(i, j)   = Real(100);
                s.grid.eta(i, j) = static_cast<Real>(bump);
                s.grid.H(i, j)   = Real(100) + static_cast<Real>(bump);
                s.grid.u(i, j)   = Real(0);
                s.grid.v(i, j)   = Real(0);
            }
        }
    }
    s.grid.MarkInitialized();
    return s;
}

SingleLineScenario ScenarioBuilder::BuildOkushiri450m()
{
    SingleLineScenario s;
    s.g = 9.81;
    s.tau = 1.0;                         // CFL ≈ sqrt(g·3500) * 1 / 450 ≈ 0.41
    s.delta_x = [](int, int) { return 450.0; };
    s.delta_y = 450.0;
    s.nx = 379;
    s.ny = 687;
    s.timesteps = 2000;                  // ~33 min physical
    s.num_threads = 4;
    s.num_subgrids = 8;
    s.coord_system = CoordSystem::Cartesian;
    s.exchangeBoundaryIndex = 4;         // Sommerfeld radiation (all 4 sides)
    s.exchangeHatBoundaryIndex = 4;

    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);

    // Bathymetry (positive = water depth in metres).
    LoadOkushiri10F7("Data/Okushiri/D379-687-450m.txt", s.grid, s.nx, s.ny, 'D');
    // Initial wave (η in metres; small ~-0.05..+positive).
    LoadOkushiri10F7("Data/Okushiri/I379-687-450m-17a.txt", s.grid, s.nx, s.ny, 'E');

    // H = D + η; u = v = 0 (rest with initial elevation).
    for (int i = 0; i < s.nx; ++i) {
        for (int j = 0; j < s.ny; ++j) {
            const Real d = s.grid.D(i, j);
            const Real e = s.grid.eta(i, j);
            s.grid.H(i, j) = (d > Real(0)) ? (d + e) : Real(0);
            s.grid.u(i, j) = Real(0);
            s.grid.v(i, j) = Real(0);
        }
    }
    s.grid.MarkInitialized();
    s.has_own_grid = true;
    return s;
}

// ==========================================================================
//  §5 Coriolis sweep по широтам (Work_Plan_2026-05-12)
//
//  Цель: зависимость отклонения волны от широты. Серия запусков на полосах
//  [lat_min, lat_max] (по плану: 0-20, 10-30, 20-40, ..., 60-80).
//
//  Дизайн:
//  - Квадратный домен. ny = round((lat_max - lat_min) × 111.3) → ≈ 2226 для
//    20° span при dy=1000m. nx = ny.
//  - dx = dy = 1000 m, плоское дно D = 1000 m, амплитуда 1 m.
//  - S→N плоская волна КОНЕЧНОЙ ширины (ширина по горизонтали < домена):
//    j ∈ [5, 20) длина 15 ячеек, i ∈ [imin, imax) ширина ≈ 37.5% nx.
//    Margins ≈ 31% от каждой боковой стенки — достаточно, чтобы Coriolis-
//    отклонение (на 60° за 3.5 ч ≈ 200 км) не достигало стенок.
//  - v0 = c · η / D = √(g D) · 1 / 1000 ≈ 0.099 m/s — чистая бегущая волна на N.
//  - τ = 5 s, 2500 шагов = 12500 s ≈ 3.5 h. На 60°/70° f·t ≈ 1.5-1.7 rad
//    (значительное отклонение). За это время фронт проходит ≈ 1235 / 2226
//    ≈ 55% домена — успеет показать deflection до встречи с N-стенкой.
//  - BC zero-gradient (index=0), без sponge (как в V7_SmallSN).
//  - center_latitude_rad = середина полосы.
//
//  Использование в Sandbox: пары (Coriolis ON / OFF) на одной геометрии.
// ==========================================================================
SingleLineScenario ScenarioBuilder::BuildTestCoriolisLatBand(
    double lat_min_deg, double lat_max_deg, CoordSystem cs)
{
    SingleLineScenario s;
    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    const double lat_span_deg = lat_max_deg - lat_min_deg;
    const int ny = static_cast<int>(std::round(lat_span_deg * 111.3));
    s.nx = ny;
    s.ny = ny;
    s.timesteps = 2500;
    s.num_threads = 20;
    s.num_subgrids = 1;
    s.coord_system = cs;
    const double center_lat_deg = 0.5 * (lat_min_deg + lat_max_deg);
    s.center_latitude_rad = center_lat_deg * M_PI / 180.0;
    s.minimum_depth_to_calculate = 8.0;
    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;
    s.sponge_west_cells = 0;
    s.sponge_east_cells = 0;
    s.sponge_south_cells = 0;
    s.sponge_north_cells = 0;
    s.sponge_sigma_max = 0.0;
    s.interior_i_min = 0;
    s.interior_i_max = s.nx;
    s.interior_j_min = 0;
    s.interior_j_max = s.ny;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    const int wave_width = static_cast<int>(std::round(0.375 * s.nx));
    const int wave_i_min = (s.nx - wave_width) / 2;
    const int wave_i_max = wave_i_min + wave_width;
    s.grid.SetParamsCoriolisFinitePlane(1000.0, 5, 20,
        wave_i_min, wave_i_max, 1.0,
        /*u0*/ 0.0, /*v0*/ 0.0990195135927845);
    s.has_own_grid = true;
    return s;
}

// ==========================================================================
//  §1 Test L: benchmark scaffold для num_subgrids sweep на сетке, где
//  объём данных значительно превышает L3 (33 MB на i7-14700K).
//
//  5000×5000 × 5 fields × 8 bytes ≈ 1 GB (с hat — ≈ 2 GB) ≫ L3.
//  S→N плоская волна конечной ширины (i ∈ [1000, 4000) = 60% nx).
//  1000 шагов × τ=5s = 5000 s — короткий бенчмарк, не полная физика.
// ==========================================================================
SingleLineScenario ScenarioBuilder::BuildTestL_Benchmark()
{
    SingleLineScenario s;
    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = [](int, int) { return 1000.0; };
    s.delta_y = 1000.0;
    s.nx = 5000;
    s.ny = 5000;
    s.timesteps = 1000;
    s.num_threads = 20;
    s.num_subgrids = 1;
    s.coord_system = CoordSystem::Cartesian;
    s.center_latitude_rad = 0.785398163;
    s.minimum_depth_to_calculate = 8.0;
    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;
    s.sponge_west_cells = 0;
    s.sponge_east_cells = 0;
    s.sponge_south_cells = 0;
    s.sponge_north_cells = 0;
    s.sponge_sigma_max = 0.0;
    s.interior_i_min = 0;
    s.interior_i_max = s.nx;
    s.interior_j_min = 0;
    s.interior_j_max = s.ny;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    s.grid.SetParamsCoriolisFinitePlane(1000.0, 5, 20, 1000, 4000, 1.0,
        /*u0*/ 0.0, /*v0*/ 0.0990195135927845);
    s.has_own_grid = true;
    return s;
}

// ==========================================================================
//  Тихий океан + Курило-Камчатский источник 2025.
//
//  Bath:    Data/pacific_1min/pacific_1min.asc
//           9601 cols × 6781 rows, cellsize = 1 arc-min ≈ 1855 m,
//           xll = 129.9917°E, yll = -53.0083°N (южная граница).
//  Source:  Data/pacific_sour2025_1min/sour2025_1min.asc
//           843 cols × 541 rows, голая матрица η₀ без заголовка.
//           Lower-left угол источника в pacific-grid на узле
//           (col=1231, row_from_south=6000), что соответствует
//           lon ≈ 150.5°E, lat ≈ +47°N — Курило-Камчатская зона.
//  ASCParser convention: row=0 — north (top), row=nrows-1 — south (bottom).
//
//  При sub_size > 0 выбирается квадратное окно `sub_size × sub_size`
//  вокруг центра источника — для отладки/быстрого probe.
// ==========================================================================
SingleLineScenario ScenarioBuilder::BuildPacificKamchatka(bool coriolis_enabled, int sub_size)
{
    std::cerr << "[Pacific] enter, coriolis=" << coriolis_enabled
              << " sub_size=" << sub_size << std::endl;
    SingleLineScenario s;

    // 1) Read full Pacific bathymetry.
    std::cerr << "[Pacific] reading Data/pacific_1min/pacific_1min.asc ..." << std::endl;
    ASCParser asc;
    asc.Read("Data/pacific_1min/pacific_1min.asc");
    const int ncols = asc.GetCols();
    const int nrows = asc.GetRows();
    std::cerr << "[Pacific] ASC ok: " << ncols << "x" << nrows << std::endl;
    const double yll = asc.GetYllCorner();
    const double cellsize = asc.GetCellSize();

    // 2) Source position (lower-left in pacific grid, counting from south edge).
    const int src_col0 = 1231;
    int src_row_from_south0 = 6000;
    const int src_ncols = 843;
    const int src_nrows = 541;
    // PACIFIC_SRC_ROW_OFFSET env override (diagnostic D1: move source south).
    // Positive offset moves source NORTH (higher row_from_south), negative moves SOUTH.
    {
        char* off_buf = nullptr; size_t off_sz = 0;
        _dupenv_s(&off_buf, &off_sz, "PACIFIC_SRC_ROW_OFFSET");
        if (off_buf) {
            const int off = std::atoi(off_buf);
            src_row_from_south0 += off;
            std::cerr << "[Pacific] source row offset override: row_from_south0=" << src_row_from_south0 << std::endl;
            free(off_buf);
        }
    }
    // Convert to ASCParser row index (row=0 is north).
    // row_from_top = (nrows-1) - row_from_south
    const int src_row_top_bottom = (nrows - 1) - src_row_from_south0;            // = 780
    const int src_row_top_top    = src_row_top_bottom - (src_nrows - 1);         // = 240

    // 3) Choose sub-region around source centre, or full domain.
    int i0 = 0, i1 = ncols;
    int j0 = 0, j1 = nrows;
    if (sub_size > 0) {
        const int src_col_c = src_col0 + src_ncols / 2;
        const int src_row_c = (src_row_top_top + src_row_top_bottom) / 2;
        const int half = sub_size / 2;
        i0 = std::max(0, src_col_c - half);
        i1 = std::min(ncols, src_col_c + half);
        j0 = std::max(0, src_row_c - half);
        j1 = std::min(nrows, src_row_c + half);
    }
    const int nx = i1 - i0;
    const int ny = j1 - j0;
    std::cerr << "[Pacific] sub-region: i=[" << i0 << "," << i1 << ") j=[" << j0 << "," << j1
              << ") -> " << nx << "x" << ny << std::endl;

    // 4) Fill scenario.
    s.g = 9.81;
    s.tau = 1.0;
    s.timesteps = 5000;
    s.num_threads = 12;
    s.num_subgrids = 12;
    s.coord_system = CoordSystem::Spherical;
    s.coriolis_enabled = coriolis_enabled;
    s.minimum_depth_to_calculate = 8.0;
    // Sub-region cuts the open ocean — waves that reach the artificial edges
    // must NOT reflect. Use Sommerfeld radiation BC (exchange index 4) on the
    // boundary instead of sponge layer. Sponge is disallowed for production
    // MacCormack runs (feedback_no_sponge_maccormack).
    // PACIFIC_BC_INDEX env override for diagnostic experiments D3 (zero-gradient=1, Sommerfeld=4)
    int bc_index = 4;
    {
        char* bc_buf = nullptr; size_t bc_sz = 0;
        _dupenv_s(&bc_buf, &bc_sz, "PACIFIC_BC_INDEX");
        if (bc_buf) {
            bc_index = std::atoi(bc_buf);
            std::cerr << "[Pacific] BC override from env: exchangeBoundaryIndex=" << bc_index << std::endl;
            free(bc_buf);
        }
    }
    s.exchangeBoundaryIndex    = bc_index;
    s.exchangeHatBoundaryIndex = bc_index;
    s.nx = nx;
    s.ny = ny;
    s.sponge_west_cells  = 0;
    s.sponge_east_cells  = 0;
    s.sponge_south_cells = 0;
    s.sponge_north_cells = 0;
    s.sponge_sigma_max = 0.0;
    s.interior_i_min = 0;
    s.interior_i_max = nx;
    s.interior_j_min = 0;
    s.interior_j_max = ny;

    // 5) δx(i,j) depends on latitude; sub-region's south corner is at
    //    yll_sub = yll + (nrows - j1) * cellsize.
    const double yll_sub = yll + (nrows - j1) * cellsize;
    s.delta_x = [yll_sub, cellsize, ny](int /*i*/, int j) {
        const double latitude = yll_sub + (ny - j) * cellsize;
        return 111300.0 * cellsize * std::cos(latitude * M_PI / 180.0);
    };
    s.delta_y = 111300.0 * cellsize;  // ≈ 1855 m

    const double center_lat_deg = yll_sub + (ny / 2.0) * cellsize;
    s.center_latitude_rad = center_lat_deg * M_PI / 180.0;

    std::cerr << "[Pacific] constructing Grid " << s.nx << "x" << s.ny << " ..." << std::endl;
    s.grid = Grid(s.nx, s.ny, s.delta_x, s.delta_y);
    std::cerr << "[Pacific] Grid ok" << std::endl;

    // 6) Copy bathymetry from ASCParser into sub-region grid.
    //    ASCParser already negates values so positive D means water depth.
    //    Pacific data has nodata_value = -99999 which becomes +99999 after the
    //    negation → without clamping that gives c = √(g·99999) ≈ 990 m/s and
    //    CFL > 1 (blow-up). Cap dry/nodata cells at the wet/dry threshold so
    //    EnforceDryCells treats them as solid. Cap deep ocean at 11 km (Mariana).
    const double D_NODATA_NEG = -asc.GetNoDataValue();   // = +99999 after parser negation
    int n_nodata = 0;
    int n_land   = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int gi = i0 + i;
            const int gj = j0 + j;
            double D = asc.GetValue(gj, gi);
            if (std::abs(D - D_NODATA_NEG) < 1.0 || D > 11000.0) {
                D = 0.0;            // nodata or absurd → mark as land (dry)
                ++n_nodata;
            } else if (D < 8.0) {
                ++n_land;
            }
            const double D_clamp = std::max(D, 8.0);
            s.grid.D(i, j) = D_clamp;
            s.grid.H(i, j) = D_clamp;
            s.grid.eta(i, j) = 0.0;
            s.grid.u(i, j) = 0.0;
            s.grid.v(i, j) = 0.0;
        }
    }
    std::cerr << "[Pacific] bathymetry stats: nodata/oversize=" << n_nodata
              << " shallow(<8m)=" << n_land << " of " << (nx * ny)
              << std::flush << std::endl;

    std::cerr << "[Pacific] bathymetry copied; reading source file ..." << std::flush << std::endl;
    // 7) Read source η₀ matrix (843 × 541, raw whitespace-separated reals).
    //    Source row 0 is the TOP of the source rectangle (ASC convention).
    //    Apply only over cells whose ORIGINAL ASC value indicates ocean
    //    (positive depth ≥ wet/dry threshold). Land/nodata cells got clamped
    //    to D=8 above; placing η₀=11 m on top of that creates a phantom
    //    shock right at the coastline (manifests as nonlinear blow-up
    //    at t≈4000 in the full-domain run).
    {
        std::ifstream src("Data/pacific_sour2025_1min/sour2025_1min.asc");
        if (!src.is_open()) {
            std::cerr << "BuildPacificKamchatka: failed to open Data/pacific_sour2025_1min/sour2025_1min.asc\n";
        } else {
            std::cerr << "[Pacific] source file open" << std::flush << std::endl;
            int written = 0;
            int skipped_land = 0;
            for (int r = 0; r < src_nrows; ++r) {
                for (int c = 0; c < src_ncols; ++c) {
                    double value = 0.0;
                    if (!(src >> value)) { value = 0.0; }
                    const int g_col = src_col0 + c;
                    const int g_row = src_row_top_top + r;
                    const int li = g_col - i0;
                    const int lj = g_row - j0;
                    if (li < 0 || li >= nx || lj < 0 || lj >= ny) continue;
                    // Original ASC value at this cell — positive = water depth,
                    // negative = land elevation (post-negation in parser).
                    const double D_orig = asc.GetValue(g_row, g_col);
                    if (D_orig <= 0.0 || D_orig > 11000.0) {
                        // Land or nodata: do NOT apply source here.
                        ++skipped_land;
                        continue;
                    }
                    if (D_orig < s.minimum_depth_to_calculate) {
                        // Very shallow (< 8 m): also skip, would amplify nonlinearly.
                        ++skipped_land;
                        continue;
                    }
                    // 2026-05-21: source scale factor for diagnostic. Set via
                    // env PACIFIC_SRC_SCALE=0.5 (etc). If lowering the
                    // amplitude cures the blow-up at t≈3000, the original
                    // source is over-energetic for the MacCormack/wet-dry
                    // tolerance of this implementation.
                    static double src_scale = -1.0;
                    if (src_scale < 0.0) {
                        char* buf = nullptr; size_t sz = 0;
                        _dupenv_s(&buf, &sz, "PACIFIC_SRC_SCALE");
                        src_scale = (buf && *buf) ? std::atof(buf) : 1.0;
                        if (buf) free(buf);
                        if (std::abs(src_scale - 1.0) > 1e-9) {
                            std::cerr << "[Pacific] source scaled by "
                                      << src_scale << std::endl;
                        }
                    }
                    const double v_scaled = value * src_scale;
                    s.grid.eta(li, lj) = static_cast<Real>(v_scaled);
                    s.grid.H(li, lj) = static_cast<Real>(D_orig + v_scaled);
                    ++written;
                }
            }
            std::cerr << "[Pacific] source applied: " << written << " cells, "
                      << "skipped (land/shallow): " << skipped_land
                      << std::flush << std::endl;
        }
    }

    // ========================================================================
    // CHECKPOINT/RESUME: if PACIFIC_RESUME_DIR + PACIFIC_RESUME_STEP env vars
    // are set, override the freshly-built initial condition with state loaded
    // from a previous run's snapshot files (eta_timestep_N.txt, u_timestep_N.txt,
    // v_timestep_N.txt). Used for progressive extension runs without restarting
    // from t=0. Source contribution is overwritten by the loaded snapshot, so
    // the source effectively becomes "ignored" in resume mode.
    // ========================================================================
    {
        char* ckpt_dir_buf = nullptr; size_t ckpt_sz = 0;
        _dupenv_s(&ckpt_dir_buf, &ckpt_sz, "PACIFIC_RESUME_DIR");
        if (ckpt_dir_buf) {
            char* step_buf = nullptr; size_t step_sz = 0;
            _dupenv_s(&step_buf, &step_sz, "PACIFIC_RESUME_STEP");
            int resume_step = step_buf ? std::atoi(step_buf) : 0;
            if (step_buf) free(step_buf);

            std::string dir(ckpt_dir_buf);
            free(ckpt_dir_buf);

            std::cerr << "[Pacific] CHECKPOINT RESUME: loading state from "
                      << dir << " step=" << resume_step << std::endl;

            auto field_path = [&](const char* var) {
                return dir + "/" + var + "_timestep_" + std::to_string(resume_step) + ".txt";
            };

            auto load_field = [&](const std::string& fname,
                                  std::function<void(int, int, double)> setter) -> bool {
                std::ifstream f(fname);
                if (!f) {
                    std::cerr << "[Pacific] FAILED to open " << fname << std::endl;
                    return false;
                }
                std::string line;
                int i = 0;
                long total_values = 0;
                int short_lines = 0;
                while (std::getline(f, line) && i < nx) {
                    std::istringstream iss(line);
                    int j = 0;
                    double v;
                    while (iss >> v && j < ny) {
                        setter(i, j, v);
                        ++j;
                        ++total_values;
                    }
                    if (j != ny) {
                        ++short_lines;
                        if (short_lines <= 3) {
                            std::cerr << "[Pacific] WARN: short line " << i << " (j=" << j << " < " << ny << ") in " << fname << std::endl;
                        }
                    }
                    ++i;
                }
                const long expected = (long)nx * (long)ny;
                if (i != nx || total_values != expected) {
                    std::cerr << "[Pacific] FAILED to load " << fname
                              << ": got lines=" << i << "/" << nx
                              << " values=" << total_values << "/" << expected << std::endl;
                    return false;
                }
                std::cerr << "[Pacific] loaded " << fname << " (" << total_values << " values)" << std::endl;
                return true;
            };

            // Load eta, u, v from snapshot files.
            bool ok_eta = load_field(field_path("eta"),
                [&](int i, int j, double v) { s.grid.eta(i, j) = (Real)v; });
            bool ok_u = load_field(field_path("u"),
                [&](int i, int j, double v) { s.grid.u(i, j) = (Real)v; });
            bool ok_v = load_field(field_path("v"),
                [&](int i, int j, double v) { s.grid.v(i, j) = (Real)v; });

            if (!(ok_eta && ok_u && ok_v)) {
                std::cerr << "[Pacific] CHECKPOINT load failed, aborting." << std::endl;
                std::exit(2);
            }

            // Recompute H = D + eta for each cell.
            // Dry cells (D < grnd) keep H = D (matches EnforceDryCells convention).
            const double grnd = s.minimum_depth_to_calculate;
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const double D = s.grid.D(i, j);
                    if (D < grnd) {
                        s.grid.H(i, j) = (Real)D;
                        s.grid.eta(i, j) = (Real)0.0;
                        s.grid.u(i, j) = (Real)0.0;
                        s.grid.v(i, j) = (Real)0.0;
                    } else {
                        s.grid.H(i, j) = (Real)(D + (double)s.grid.eta(i, j));
                    }
                }
            }

            std::cerr << "[Pacific] CHECKPOINT loaded step " << resume_step
                      << ", recomputed H = D + eta." << std::endl;

            // Also load eta_max, u_max, v_max if present (otherwise leave as initialized).
            std::string emax_path = dir + "/eta_max_timestep_0.txt";
            std::ifstream emax_f(emax_path);
            if (emax_f) {
                emax_f.close();
                load_field(emax_path,
                    [&](int i, int j, double v) { s.grid.eta_max(i, j) = (Real)v; });
                std::cerr << "[Pacific] loaded eta_max from checkpoint." << std::endl;
            }
        }
    }

    s.grid.MarkInitialized();
    s.has_own_grid = true;
    std::cerr << "[Pacific] BuildPacificKamchatka complete" << std::flush << std::endl;
    return s;
}
