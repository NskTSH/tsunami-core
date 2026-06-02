#include "ScenarioBuilderCUDA.h"
#include "ASCParser.h"
#include "Grid.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// === Универсальный метод BuildScenario ========================================
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildScenario(int testNumber)
{
    switch (testNumber) {
        case 1:  return BuildTest1();
        case 2:  return BuildTest2();
        case 3:  return BuildTest3();
        case 4:  return BuildTest4();
        case 5:  return BuildTest5();
        case 6:  return BuildTest6();
        case 11: return BuildTest1Spherical();
        case 12: return BuildTest2Spherical();
        case 13: return BuildTest3Spherical();
        case 14: return BuildTest4Spherical();
        case 15: return BuildTest5Spherical();
        case 16: return BuildTest6Spherical();
        // S7 Japan Sea (CPU testIndex 18=Spherical, 19=Cartesian)
        case 18: return BuildTest6SphericalS7();
        case 19: return BuildTest6CartesianS7();
        // Pacific / Курило-Камчатский источник 2025 (mirror CPU Sandbox 900-903).
        // 900=Coriolis full domain, 901=no-Coriolis full domain,
        // 902/903 = 3000-крон вокруг источника (probe). 904/905 = синтетический smoke.
        case 900: { auto s = BuildPacificKamchatka(true,  0);    s.testNumber = 900; return s; }
        case 901: { auto s = BuildPacificKamchatka(false, 0);    s.testNumber = 901; return s; }
        case 902: { auto s = BuildPacificKamchatka(true,  3000); s.testNumber = 902; return s; }
        case 903: { auto s = BuildPacificKamchatka(false, 3000); s.testNumber = 903; return s; }
        case 904: { auto s = BuildPacificSmoke(true,  300);      s.testNumber = 904; return s; }
        case 905: { auto s = BuildPacificSmoke(false, 300);      s.testNumber = 905; return s; }
        default:
            throw std::invalid_argument("Invalid test number: " + std::to_string(testNumber) +
                                       ". Valid: 1-6/11-16 (S2), 18/19 (S7), 900-905 (Pacific).");
    }
}

// === Тест 1 ===============================================================
// Прямоугольная волна 10x10 в центре (SetParamsTest1)
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest1()
{
    SingleLineScenarioCUDA s;

    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = 1000.0;
    s.delta_y = 1000.0;
    s.nx = 120;
    s.ny = 120;
    s.timesteps = 500;
    s.num_subgrids = 1;  // Маленькая сетка - 1 подсетка достаточно

    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;
    
    s.testNumber = 1;  // Статические граничные условия
    s.has_own_grid = false;

    return s;
}

// === Тест 2 ===============================================================
// Круговой фронт из центра (SetParamsTest2)
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest2()
{
    SingleLineScenarioCUDA s;

    s.g = 9.81;
    s.tau = 5.0;
    s.delta_x = 1000.0;
    s.delta_y = 1000.0;
    s.nx = 1000;
    s.ny = 1000;
    s.timesteps = 4500;
    s.num_subgrids = 1;  // Для больших сеток можно увеличить

    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;

    s.testNumber = 2;  // Статические граничные условия
    s.has_own_grid = false;

    return s;
}

// === Тест 3 ===============================================================
// Плоский фронт от левого края поворачивает на 90 градусов. Дно квадратично углубляется
// ВАЖНО: Использует динамические граничные условия (exchangeBoundaryIndex = 2)
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest3()
{
    SingleLineScenarioCUDA s;

    s.g = 9.81;
    s.tau = 2.0;
    s.delta_x = 1000.0;
    s.delta_y = 1000.0;
    s.nx = 1000;
    s.ny = 1000;
    s.timesteps = 8000;
    s.num_subgrids = 1;

    // ВАЖНО: Тест 3 использует ExchangeBoundaryData3 с динамической волной!
    s.exchangeBoundaryIndex = 2;
    s.exchangeHatBoundaryIndex = 2;

    s.testNumber = 3;  // Динамические граничные условия
    s.has_own_grid = false;

    return s;
}

// === Тест 4 ===============================================================
// Линза Люнеберга (SetParamsTest4)
// ВАЖНО: Использует динамические граничные условия (exchangeBoundaryIndex = 3)
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest4()
{
    SingleLineScenarioCUDA s;

    s.g = 9.81;
    s.tau = 2.0;
    s.delta_x = 1000.0;
    s.delta_y = 1000.0;
    s.nx = 1000;
    s.ny = 1000;
    s.timesteps = 9000;
    s.num_subgrids = 1;

    // ВАЖНО: Тест 4 использует ExchangeBoundaryData4 с динамической волной!
    s.exchangeBoundaryIndex = 3;
    s.exchangeHatBoundaryIndex = 3;

    s.testNumber = 4;  // Динамические граничные условия
    s.has_own_grid = false;

    return s;
}

// === Тест 5 ===============================================================
// Круглый остров посередине
// ВАЖНО: Использует динамические граничные условия (exchangeBoundaryIndex = 3)
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest5()
{
    SingleLineScenarioCUDA s;

    s.g = 9.81;
    s.tau = 2.0;
    s.delta_x = 1000.0;
    s.delta_y = 1000.0;
    s.nx = 1000;
    s.ny = 1000;
    s.timesteps = 6000;
    s.num_subgrids = 1;

    // ВАЖНО: Тест 5 использует ExchangeBoundaryData5 с динамической волной!
    s.exchangeBoundaryIndex = 3;
    s.exchangeHatBoundaryIndex = 3;

    s.testNumber = 5;  // Динамические граничные условия
    s.has_own_grid = false;

    return s;
}

// === Spherical variants (matching CPU BuildTest{N}Spherical) ================
// All follow CPU pattern: base on Cartesian scenario, then set coord_system=Spherical,
// center_latitude_rad=45°, and enable Coriolis. CPU enables Coriolis via
// SingleLineScenario.coriolis_enabled (defaulted false in struct but turned on by Coriolis tests).
// For tests 11-15 we follow CPU SingleLineScenario default — Coriolis enabled.
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest1Spherical() {
    SingleLineScenarioCUDA s = BuildTest1();
    s.testNumber = 11;
    s.coord_system_id = 1;
    s.coriolis_enabled = true;
    s.center_latitude_rad = 0.785398163;  // 45°
    return s;
}

SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest2Spherical() {
    SingleLineScenarioCUDA s = BuildTest2();
    s.testNumber = 12;
    s.coord_system_id = 1;
    s.coriolis_enabled = true;
    s.center_latitude_rad = 0.785398163;
    return s;
}

SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest3Spherical() {
    SingleLineScenarioCUDA s = BuildTest3();
    s.testNumber = 13;
    s.coord_system_id = 1;
    s.coriolis_enabled = true;
    s.center_latitude_rad = 0.785398163;
    return s;
}

SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest4Spherical() {
    SingleLineScenarioCUDA s = BuildTest4();
    s.testNumber = 14;
    s.coord_system_id = 1;
    s.coriolis_enabled = true;
    s.center_latitude_rad = 0.785398163;
    return s;
}

SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest5Spherical() {
    SingleLineScenarioCUDA s = BuildTest5();
    s.testNumber = 15;
    s.coord_system_id = 1;
    s.coriolis_enabled = true;
    s.center_latitude_rad = 0.785398163;
    return s;
}

// === Japan Sea helper ======================================================
// Builds a Japan Sea (bathymetry from Data/JapanSea.asc) scenario with given
// ellipsoidal source params (center_x, center_y, al_deg) and min_depth from
// the corresponding .par file. Mirrors CPU's BuildJapanSeaScenario but with
// source params passed explicitly instead of read from a parser.
static SingleLineScenarioCUDA BuildJapanSeaCUDA(int testNumber,
                                                double source_x, double source_y,
                                                double source_al_deg, double min_depth) {
    SingleLineScenarioCUDA s;
    s.g = 9.81;
    s.tau = 0.5;
    s.delta_y = 462.5;
    s.timesteps = 25000;
    s.num_subgrids = 1;
    s.exchangeBoundaryIndex = 0;
    s.exchangeHatBoundaryIndex = 0;
    s.testNumber = testNumber;
    s.coord_system_id = 0;
    s.coriolis_enabled = false;
    s.min_depth = min_depth;

    ASCParser ascParser;
    ascParser.Read("Data/JapanSea.asc");
    const int nx = ascParser.GetCols();
    const int ny = ascParser.GetRows();
    s.nx = nx;
    s.ny = ny;

    const double yllcorner = ascParser.GetYllCorner();
    const double cellsize = ascParser.GetCellSize();

    s.delta_x_func = [yllcorner, cellsize, ny](int /*i*/, int j) {
        const double latitude = yllcorner + (ny - j) * cellsize;
        return 111300.0 * cellsize * std::cos(latitude * M_PI / 180.0);
    };
    s.delta_x = s.delta_x_func(0, ny / 2);

    const double center_lat_deg = yllcorner + (ny / 2.0) * cellsize;
    s.center_latitude_rad = center_lat_deg * M_PI / 180.0;

    Grid cpu_grid(nx, ny, s.delta_x_func, s.delta_y);
    cpu_grid.SetParamsTest6(s.tau, ascParser, source_x, source_y, source_al_deg);

    s.grid = GridCUDA(nx, ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            s.grid.H(i, j) = cpu_grid.H(i, j);
            s.grid.u(i, j) = cpu_grid.u(i, j);
            s.grid.v(i, j) = cpu_grid.v(i, j);
            s.grid.eta(i, j) = cpu_grid.eta(i, j);
            s.grid.D(i, j) = cpu_grid.D(i, j);
        }
    }
    s.grid.SetInitialized(true);
    s.has_own_grid = true;

    std::cout << "Japan Sea test " << testNumber << ": " << nx << "x" << ny
              << ", center_lat=" << center_lat_deg << " deg, tau=" << s.tau
              << ", source=(" << source_x << "," << source_y << "), al=" << source_al_deg
              << ", min_depth=" << min_depth << std::endl;
    return s;
}

// === Тест 6: Japan Sea (S2 source) ===========================================
// Data/JapanSeaOneSourceS2.par: source=(139.60, 40.28), angle=-10 (→ al_deg=10),
// min_depth=8.0. Source-rotation matches CPU SetParamsTest6.
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest6() {
    return BuildJapanSeaCUDA(6, 139.60, 40.28, 10.0, 8.0);
}

SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest6Spherical() {
    SingleLineScenarioCUDA s = BuildTest6();
    s.testNumber = 16;
    s.coord_system_id = 1;
    s.coriolis_enabled = false;
    return s;
}

// === S7 Japan Sea (CPU testIndex 19=Cartesian, 18=Spherical) =================
// Data/JapanSeaOneSourceS7.par: source=(132.00, 36.00), angle=-80 (→ al_deg=80),
// min_depth=7.0.
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest6CartesianS7() {
    return BuildJapanSeaCUDA(19, 132.00, 36.00, 80.0, 7.0);
}

SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildTest6SphericalS7() {
    SingleLineScenarioCUDA s = BuildTest6CartesianS7();
    s.testNumber = 18;
    s.coord_system_id = 1;
    s.coriolis_enabled = false;  // CPU BuildTest6SphericalS7 doesn't enable Coriolis
    return s;
}

// === Тихий океан + Курило-Камчатский источник 2025 (GPU monolithic) ==========
// Порт CPU ScenarioBuilder::BuildPacificKamchatka (ScenarioBuilder.cpp:1438).
// Отличия от CPU: num_subgrids=1 (монолитный GPU-путь умещается в 24 GB VRAM и имеет
// spherical/Coriolis ядра, в отличие от CUDA subgrid-пути); состояние заливается прямо
// в GridCUDA. Граничное условие — Sommerfeld radiation (index 4).
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildPacificKamchatka(bool coriolis_enabled, int sub_size)
{
    std::cerr << "[PacificCUDA] enter, coriolis=" << coriolis_enabled
              << " sub_size=" << sub_size << std::endl;
    SingleLineScenarioCUDA s;

    // 1) Полная батиметрия Тихого океана.
    std::cerr << "[PacificCUDA] reading Data/pacific_1min/pacific_1min.asc ..." << std::endl;
    ASCParser asc;
    asc.Read("Data/pacific_1min/pacific_1min.asc");  // бросит, если файла нет
    const int ncols = asc.GetCols();
    const int nrows = asc.GetRows();
    std::cerr << "[PacificCUDA] ASC ok: " << ncols << "x" << nrows << std::endl;
    const double yll = asc.GetYllCorner();
    const double cellsize = asc.GetCellSize();

    // 2) Положение источника (нижний-левый угол, отсчёт от южной границы).
    const int src_col0 = 1231;
    int src_row_from_south0 = 6000;
    const int src_ncols = 843;
    const int src_nrows = 541;
    // PACIFIC_SRC_ROW_OFFSET env override (diagnostic D1: move source N/S), as in CPU
    // ScenarioBuilder::BuildPacificKamchatka. Positive offset moves source NORTH.
    {
        char* off_buf = nullptr; size_t off_sz = 0;
        _dupenv_s(&off_buf, &off_sz, "PACIFIC_SRC_ROW_OFFSET");
        if (off_buf) {
            src_row_from_south0 += std::atoi(off_buf);
            std::cerr << "[PacificCUDA] source row offset override: row_from_south0="
                      << src_row_from_south0 << std::endl;
            free(off_buf);
        }
    }
    // row=0 — север (top). row_from_top = (nrows-1) - row_from_south.
    const int src_row_top_bottom = (nrows - 1) - src_row_from_south0;       // = 780
    const int src_row_top_top    = src_row_top_bottom - (src_nrows - 1);    // = 240

    // 3) Подрегион вокруг источника (probe) или полный домен.
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
    std::cerr << "[PacificCUDA] sub-region: i=[" << i0 << "," << i1 << ") j=[" << j0 << "," << j1
              << ") -> " << nx << "x" << ny << std::endl;

    // 4) Параметры сценария.
    s.g = 9.81;
    s.tau = 1.0;
    s.timesteps = 5000;
    s.num_subgrids = 1;                 // монолитный GPU-путь
    s.coord_system_id = 1;              // Spherical
    s.coriolis_enabled = coriolis_enabled;
    s.min_depth = 8.0;
    s.save_velocities = true;
    s.nx = nx;
    s.ny = ny;

    // Sommerfeld radiation BC (open boundary) — подрегион кроит открытый океан, отражения
    // от искусственных краёв недопустимы. PACIFIC_BC_INDEX env override (1=zero-gradient,
    // 4=Sommerfeld) для диагностики, как в CPU.
    int bc_index = 4;
    {
        char* bc_buf = nullptr; size_t bc_sz = 0;
        _dupenv_s(&bc_buf, &bc_sz, "PACIFIC_BC_INDEX");
        if (bc_buf) {
            bc_index = std::atoi(bc_buf);
            std::cerr << "[PacificCUDA] BC override from env: exchangeBoundaryIndex=" << bc_index << std::endl;
            free(bc_buf);
        }
    }
    s.exchangeBoundaryIndex = bc_index;
    s.exchangeHatBoundaryIndex = bc_index;

    // 5) δx(i,j) зависит от широты; южный угол подрегиона на yll_sub.
    const double yll_sub = yll + (nrows - j1) * cellsize;
    s.delta_x_func = [yll_sub, cellsize, ny](int /*i*/, int j) {
        const double latitude = yll_sub + (ny - j) * cellsize;
        return 111300.0 * cellsize * std::cos(latitude * M_PI / 180.0);
    };
    s.delta_x = s.delta_x_func(0, ny / 2);
    s.delta_y = 111300.0 * cellsize;   // ≈ 1855 м

    const double center_lat_deg = yll_sub + (ny / 2.0) * cellsize;
    s.center_latitude_rad = center_lat_deg * M_PI / 180.0;

    std::cerr << "[PacificCUDA] constructing GridCUDA " << nx << "x" << ny << " ..." << std::endl;
    s.grid = GridCUDA(nx, ny);

    // 6) Копируем батиметрию (ASCParser уже инвертировал знак: положит. = глубина воды).
    //    nodata = -99999 → после инверсии +99999. Без клэмпа c=√(g·99999)≈990 м/с,
    //    CFL>1 (blow-up). Клэмпим nodata/абсурд в сушу (D=0), глубокий океан в 11 км.
    const double D_NODATA_NEG = -asc.GetNoDataValue();   // +99999
    long long n_nodata = 0, n_land = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int gi = i0 + i;
            const int gj = j0 + j;
            double D = asc.GetValue(gj, gi);
            if (std::abs(D - D_NODATA_NEG) < 1.0 || D > 11000.0) {
                D = 0.0;            // nodata / абсурд → суша
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
    std::cerr << "[PacificCUDA] bathymetry stats: nodata/oversize=" << n_nodata
              << " shallow(<8m)=" << n_land << " of " << ((long long)nx * ny) << std::endl;

    // 7) Источник η₀ (843×541, голая матрица без заголовка). Применяется только на
    //    океанских ячейках (исходная глубина D_orig ≥ wet/dry), иначе фантомный шок
    //    на береговой линии → нелинейный blow-up. PACIFIC_SRC_SCALE env — масштаб амплитуды.
    {
        std::ifstream src("Data/pacific_sour2025_1min/sour2025_1min.asc");
        if (!src.is_open()) {
            std::cerr << "[PacificCUDA] FAILED to open Data/pacific_sour2025_1min/sour2025_1min.asc\n";
        } else {
            double src_scale = 1.0;
            char* buf = nullptr; size_t sz = 0;
            _dupenv_s(&buf, &sz, "PACIFIC_SRC_SCALE");
            if (buf && *buf) src_scale = std::atof(buf);
            if (buf) free(buf);
            if (std::abs(src_scale - 1.0) > 1e-9)
                std::cerr << "[PacificCUDA] source scaled by " << src_scale << std::endl;

            int written = 0, skipped_land = 0;
            for (int r = 0; r < src_nrows; ++r) {
                for (int c = 0; c < src_ncols; ++c) {
                    double value = 0.0;
                    if (!(src >> value)) value = 0.0;
                    const int g_col = src_col0 + c;
                    const int g_row = src_row_top_top + r;
                    const int li = g_col - i0;
                    const int lj = g_row - j0;
                    if (li < 0 || li >= nx || lj < 0 || lj >= ny) continue;
                    const double D_orig = asc.GetValue(g_row, g_col);
                    if (D_orig <= 0.0 || D_orig > 11000.0) { ++skipped_land; continue; }
                    if (D_orig < s.min_depth)              { ++skipped_land; continue; }
                    const double v_scaled = value * src_scale;
                    s.grid.eta(li, lj) = v_scaled;
                    s.grid.H(li, lj) = D_orig + v_scaled;
                    ++written;
                }
            }
            std::cerr << "[PacificCUDA] source applied: " << written
                      << " cells, skipped (land/shallow): " << skipped_land << std::endl;
        }
    }

    // ========================================================================
    // CHECKPOINT/RESUME (port of CPU): PACIFIC_RESUME_DIR + PACIFIC_RESUME_STEP
    // override the freshly-built IC with a previous chunk's snapshot
    // (eta/u/v_timestep_N.txt), recompute H=D+eta. Enables 5000-step chunked runs.
    // eta_max is NOT carried on-device (combine chunk eta_max files in post-processing).
    {
        char* ckpt_dir_buf = nullptr; size_t ckpt_sz = 0;
        _dupenv_s(&ckpt_dir_buf, &ckpt_sz, "PACIFIC_RESUME_DIR");
        if (ckpt_dir_buf) {
            char* step_buf = nullptr; size_t step_sz = 0;
            _dupenv_s(&step_buf, &step_sz, "PACIFIC_RESUME_STEP");
            const int resume_step = step_buf ? std::atoi(step_buf) : 0;
            if (step_buf) free(step_buf);
            std::string dir(ckpt_dir_buf); free(ckpt_dir_buf);
            std::cerr << "[PacificCUDA] RESUME from " << dir << " step=" << resume_step << std::endl;

            auto field_path = [&](const char* var) {
                return dir + "/" + var + "_timestep_" + std::to_string(resume_step) + ".txt";
            };
            auto load_field = [&](const std::string& fname,
                                  std::function<void(int,int,double)> setter) -> bool {
                std::ifstream f(fname);
                if (!f) { std::cerr << "[PacificCUDA] FAILED open " << fname << std::endl; return false; }
                std::string line; int i = 0; long total = 0;
                while (std::getline(f, line) && i < nx) {
                    std::istringstream iss(line); int j = 0; double val;
                    while (iss >> val && j < ny) { setter(i, j, val); ++j; ++total; }
                    ++i;
                }
                const long expected = (long)nx * (long)ny;
                if (i != nx || total != expected) {
                    std::cerr << "[PacificCUDA] FAILED load " << fname << ": lines=" << i << "/" << nx
                              << " values=" << total << "/" << expected << std::endl;
                    return false;
                }
                std::cerr << "[PacificCUDA] loaded " << fname << " (" << total << " values)" << std::endl;
                return true;
            };

            bool ok = load_field(field_path("eta"), [&](int i,int j,double v){ s.grid.eta(i,j)=v; })
                   && load_field(field_path("u"),   [&](int i,int j,double v){ s.grid.u(i,j)=v; })
                   && load_field(field_path("v"),   [&](int i,int j,double v){ s.grid.v(i,j)=v; });
            if (!ok) { std::cerr << "[PacificCUDA] RESUME failed, aborting." << std::endl; std::exit(2); }

            const double grnd = s.min_depth;
            for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
                const double D = s.grid.D(i, j);
                if (D < grnd) { s.grid.H(i,j)=D; s.grid.eta(i,j)=0.0; s.grid.u(i,j)=0.0; s.grid.v(i,j)=0.0; }
                else          { s.grid.H(i,j) = D + s.grid.eta(i,j); }
            }
            std::cerr << "[PacificCUDA] RESUME loaded step " << resume_step
                      << ", recomputed H=D+eta." << std::endl;
        }
    }

    s.grid.SetInitialized(true);
    s.has_own_grid = true;
    std::cout << "PacificCUDA: " << nx << "x" << ny << ", center_lat=" << center_lat_deg
              << " deg, tau=" << s.tau << ", BC=" << bc_index
              << ", coriolis=" << (coriolis_enabled ? "on" : "off") << std::endl;
    std::cerr << "[PacificCUDA] BuildPacificKamchatka complete" << std::endl;
    return s;
}

// === Синтетический smoke-тест Тихоокеанского pipeline (no data files) =========
// Плоский глубокий океан n×n, Spherical+Coriolis, радиационный BC index 4,
// центральный косинусный источник. Быстрая проверка GPU-пути на стабильность.
SingleLineScenarioCUDA ScenarioBuilderCUDA::BuildPacificSmoke(bool coriolis_enabled, int n)
{
    SingleLineScenarioCUDA s;
    const double cellsize = 1.0 / 60.0;          // 1 arc-min, как у Pacific
    const double yll_sub = 40.0;                 // широта южного края ≈ 40°N
    const double depth = 4000.0;

    s.g = 9.81;
    s.tau = 1.0;
    s.timesteps = 400;
    s.num_subgrids = 1;
    s.coord_system_id = 1;                        // Spherical
    s.coriolis_enabled = coriolis_enabled;
    s.min_depth = 8.0;
    s.save_velocities = false;
    s.nx = n;
    s.ny = n;
    s.exchangeBoundaryIndex = 4;                  // Sommerfeld radiation
    s.exchangeHatBoundaryIndex = 4;

    s.delta_x_func = [yll_sub, cellsize, n](int /*i*/, int j) {
        const double latitude = yll_sub + (n - j) * cellsize;
        return 111300.0 * cellsize * std::cos(latitude * M_PI / 180.0);
    };
    s.delta_x = s.delta_x_func(0, n / 2);
    s.delta_y = 111300.0 * cellsize;
    s.center_latitude_rad = (yll_sub + (n / 2.0) * cellsize) * M_PI / 180.0;

    s.grid = GridCUDA(n, n);
    const int cx = n / 2, cy = n / 2, rad = std::max(4, n / 10);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            s.grid.D(i, j) = depth;
            const double dx = i - cx, dy = j - cy;
            const double dist = std::sqrt(dx * dx + dy * dy);
            const double e = (dist > rad) ? 0.0 : 0.5 * (1.0 + std::cos(M_PI * dist / rad));
            s.grid.eta(i, j) = e;
            s.grid.H(i, j) = depth + e;
            s.grid.u(i, j) = 0.0;
            s.grid.v(i, j) = 0.0;
        }
    }
    s.grid.SetInitialized(true);
    s.has_own_grid = true;
    std::cout << "PacificSmoke: " << n << "x" << n << ", coriolis="
              << (coriolis_enabled ? "on" : "off") << ", radiation BC" << std::endl;
    return s;
}

// === Применение сценария к симулятору ====================================
void ScenarioBuilderCUDA::ApplyScenarioToSimulation(SingleLineScenarioCUDA& scenario,
    SingleLineGridSimulationCUDA& simulation)
{
    // Устанавливаем параметры симуляции (3-arg signature with num_subgrids / BC indices)
    simulation.SetParameters(
        scenario.g,
        scenario.tau,
        scenario.delta_x,
        scenario.delta_y,
        scenario.nx,
        scenario.ny,
        scenario.timesteps,
        scenario.num_subgrids,
        scenario.exchangeBoundaryIndex,
        scenario.exchangeHatBoundaryIndex
    );

    // If scenario provides a variable delta_x function (Test 6 Japan Sea),
    // register it so Initialize() builds the delta_x[j] device array.
    if (scenario.delta_x_func) {
        simulation.SetDeltaXFunction(scenario.delta_x_func);
    }

    // Sphere/Coriolis parameters (for testNumber 11-15, 16, 18)
    simulation.SetCoordSystemId(scenario.coord_system_id);
    simulation.SetCoriolisEnabled(scenario.coriolis_enabled);
    simulation.SetCenterLatitudeRad(scenario.center_latitude_rad);

    // min_depth from .par (S2=8.0 default, S7=7.0). Must be set BEFORE Initialize().
    simulation.SetMinCalcValue(scenario.min_depth);

    // Save per-interval u/v snapshots and u_max/v_max maps (Pacific cases 900-903).
    simulation.SetSaveVelocities(scenario.save_velocities);

    // Автоматически вызываем соответствующий SetupTest на основе testNumber.
    // Spherical tests (11-15) reuse Cartesian setups (the only difference is solver-side).
    std::cout << "Setting up Test " << scenario.testNumber
              << " (" << (scenario.coord_system_id == 1 ? "Spherical" : "Cartesian")
              << ", coriolis=" << (scenario.coriolis_enabled ? "on" : "off")
              << ")..." << std::endl;

    switch (scenario.testNumber) {
        case 1:
        case 11: simulation.SetupTest1(); break;
        case 2:
        case 12: simulation.SetupTest2(); break;
        case 3:
        case 13: simulation.SetupTest3(scenario.tau); break;
        case 4:
        case 14: simulation.SetupTest4(scenario.tau); break;
        case 5:
        case 15: simulation.SetupTest5(scenario.tau); break;
        case 6:
        case 16:
        case 18:  // S7 Japan Sea (Spherical)
        case 19:  // S7 Japan Sea (Cartesian)
        case 900: case 901: case 902: case 903:  // Pacific / Kamchatka (pre-built grid)
        case 904: case 905:                      // Pacific synthetic smoke
            // Grid built in the scenario builder and copied into scenario.grid below.
            // No SetupTestX needed — initial conditions come from the pre-built grid.
            break;
        default:
            throw std::invalid_argument("Unknown test number in scenario: " +
                                       std::to_string(scenario.testNumber));
    }

    // Если сценарий имеет собственную сетку, используем её
    if (scenario.has_own_grid)
    {
        simulation.SetGrid(std::move(scenario.grid));
    }
}
