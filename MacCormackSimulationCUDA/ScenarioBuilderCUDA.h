#pragma once

#include <functional>
#include <memory>

#include "SingleLineGridSimulationCUDA.cuh"

// Описание сценария для CUDA версии
struct SingleLineScenarioCUDA
{
    double g = 9.81;
    double tau = 0.0;
    double delta_x = 0.0;
    // Optional: variable delta_x function. If set, takes precedence over delta_x scalar
    // (used by Test 6 / Japan Sea where dx depends on latitude).
    std::function<double(int, int)> delta_x_func;
    double delta_y = 0.0;
    int nx = -1;
    int ny = -1;
    int timesteps = 0;
    
    // Количество подсеток для экономии видеопамяти
    // Сетка делится на num_subgrids частей по оси X и обрабатывается последовательно
    int num_subgrids = 1;

    // Индексы граничных условий
    int exchangeBoundaryIndex = -1;
    int exchangeHatBoundaryIndex = -1;

    // Номер теста (1-5: Cartesian; 11-15: Spherical with Coriolis) — для авто-вызова SetupTestX
    int testNumber = 0;

    // Sphere/Coriolis (используется для тестов 11-15, повторяющих CPU Build*Spherical)
    int coord_system_id = 0;       // 0=Cartesian, 1=Spherical
    bool coriolis_enabled = false;
    double center_latitude_rad = 0.785398163;  // 45° default

    // Минимальная глубина для расчёта (из .par). CPU читает её, CUDA по умолчанию 8.0.
    double min_depth = 8.0;

    // Также сохранять покадровые поля скоростей u/v (Pacific cases 900-903).
    bool save_velocities = false;

    // Есть ли у сценария уже подготовленная глобальная сетка
    bool has_own_grid = false;
    GridCUDA grid;

    SingleLineScenarioCUDA() = default;
};

class ScenarioBuilderCUDA
{
public:
    /**
     * \brief Универсальный метод для построения сценария по номеру теста
     * \param testNumber Номер теста (1-5)
     * \return Сценарий с соответствующими параметрами
     * 
     * Использование:
     *   auto scenario = ScenarioBuilderCUDA::BuildScenario(3);
     *   SingleLineGridSimulationCUDA simulation;
     *   ScenarioBuilderCUDA::ApplyScenarioToSimulation(scenario, simulation);
     *   // SetupTest вызывается автоматически!
     *   simulation.Initialize();
     *   simulation.Run();
     */
    static SingleLineScenarioCUDA BuildScenario(int testNumber);

    /**
    * \brief Тест 1: Прямоугольная волна 10x10 в центре (SetParamsTest1)
    * nx=120, ny=120, tau=5.0, timesteps=500
    */
    static SingleLineScenarioCUDA BuildTest1();
    
    /**
     * \brief Тест 2: Круговой фронт из центра (SetParamsTest2)
     * nx=1000, ny=1000, tau=5.0, timesteps=4500
     */
    static SingleLineScenarioCUDA BuildTest2();
    
    /**
     * \brief Тест 3: Плоский фронт с квадратичным углублением дна (SetParamsTest3)
     * nx=1000, ny=1000, tau=2.0, timesteps=8000
     * ВАЖНО: Использует динамические граничные условия с движущейся волной!
     */
    static SingleLineScenarioCUDA BuildTest3();
    
    /**
     * \brief Тест 4: Линза Люнеберга - треугольный остров (SetParamsTest4)
     * nx=1000, ny=1000, tau=2.0, timesteps=9000
     * ВАЖНО: Использует динамические граничные условия с движущейся волной!
     */
    static SingleLineScenarioCUDA BuildTest4();
    
    /**
     * \brief Тест 5: Круглый остров посередине (SetParamsTest5)
     * nx=1000, ny=1000, tau=2.0, timesteps=6000
     * ВАЖНО: Использует динамические граничные условия с движущейся волной!
     */
    static SingleLineScenarioCUDA BuildTest5();

    // === Spherical / Coriolis variants (matching CPU Test 11-15) =============
    // Identical configuration to Test1-Test5 but with coord_system=Spherical and
    // center_latitude_rad=45°. Coriolis force is enabled by default.
    static SingleLineScenarioCUDA BuildTest1Spherical();  // testNumber=11
    static SingleLineScenarioCUDA BuildTest2Spherical();  // testNumber=12
    static SingleLineScenarioCUDA BuildTest3Spherical();  // testNumber=13
    static SingleLineScenarioCUDA BuildTest4Spherical();  // testNumber=14
    static SingleLineScenarioCUDA BuildTest5Spherical();  // testNumber=15

    /**
     * \brief Тест 6: Japan Sea (S2 ellipsoidal source). Reads Data/JapanSea.asc.
     * Variable delta_x (latitude-dependent), delta_y=462.5, tau=0.5,
     * timesteps=25000, Spherical coords. Matches CPU BuildTest6 + BuildTest6Spherical.
     */
    static SingleLineScenarioCUDA BuildTest6();  // testNumber=6 (Cartesian)
    static SingleLineScenarioCUDA BuildTest6Spherical();  // testNumber=16

    // S7 (Japan Sea) variants — mirror CPU BuildTest6CartesianS7 / BuildTest6SphericalS7.
    // Source params from Data/JapanSeaOneSourceS7.par: center=(132.00,36.00), angle=-80,
    // r1=0.148, r2=0.449 (matches hardcoded). min_depth=7.0.
    static SingleLineScenarioCUDA BuildTest6CartesianS7();  // testNumber=19
    static SingleLineScenarioCUDA BuildTest6SphericalS7();  // testNumber=18

    /**
     * \brief Тихий океан + Курило-Камчатский источник 2025 (testNumber 900-903).
     * Reads Data/pacific_1min/pacific_1min.asc (9601x6781) + headerless source matrix
     * Data/pacific_sour2025_1min/sour2025_1min.asc. Spherical coords, latitude-dependent
     * delta_x, Sommerfeld radiation BC (index 4), num_subgrids=1 (monolithic GPU path).
     * \param coriolis_enabled  включить силу Кориолиса
     * \param sub_size  >0 — квадратное окно sub_size×sub_size вокруг источника (probe);
     *                  0 — полный домен.
     */
    static SingleLineScenarioCUDA BuildPacificKamchatka(bool coriolis_enabled, int sub_size);

    /**
     * \brief Синтетический smoke-тест Тихоокеанского pipeline (testNumber 904/905).
     * Плоский глубокий океан n×n, Spherical+Coriolis, радиационный BC, центральный
     * косинусный источник. НЕ требует файлов данных — быстрая проверка, что GPU-путь
     * (radiation BC + spherical + coriolis + save) собирается и считает без NaN.
     */
    static SingleLineScenarioCUDA BuildPacificSmoke(bool coriolis_enabled, int n);

    /**
     * \brief Применение сценария к симулятору
     * Автоматически вызывает соответствующий SetupTestX на основе testNumber
     */
    static void ApplyScenarioToSimulation(SingleLineScenarioCUDA& scenario,
        SingleLineGridSimulationCUDA& simulation);
};
