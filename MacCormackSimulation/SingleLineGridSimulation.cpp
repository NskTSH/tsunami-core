// SingleLineGridSimulation_optimized.cpp
// Оптимизированная версия: последовательная обработка подсеток, OpenMP внутри каждой подсетки

#include "SingleLineGridSimulation.h"

#include <corecrt_math_defines.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <omp.h>

#include "ASCParser.h"
#include "CoordSystem.h"
#include "DefaultParametersParser.h"
#include "BoundaryCondition.h"
#include "FVSolverConfig.h"  // Phase E P2: FV_CFL_MAX enforcement
#include "FVSolver.h"        // Phase E: g_positivity_clamps counter

SingleLineGridSimulation::SingleLineGridSimulation()
    : g_(-1), tau_(-1), delta_x_([](int i, int j) { return 1.0; }), delta_y_(1.0),
    nx_(100), ny_(-1), timesteps_(-1), num_threads_(-1), num_subgrids_(-1),
    coord_system_(CoordSystem::Cartesian), // Значение по умолчанию только в конструкторе
    center_latitude_rad_(0.785398163),    // 45° по умолчанию для spherical-режима
    global_grid_(1, 1),
    save_interval_(10)
{
    ExchangeHatBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeHatBoundaryData1(s_grid, t); });
    ExchangeHatBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeHatBoundaryData2(s_grid, t); });
    ExchangeHatBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeHatBoundaryData3(s_grid, t); });
    ExchangeHatBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeHatBoundaryData4(s_grid, t); });
    // index 4: Sommerfeld radiation на всех 4 границах домена
    ExchangeHatBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeHatBoundaryDataRadiation(s_grid, t); });
    // index 5: Flather 1976 characteristic open boundary (production for Pacific)
    ExchangeHatBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeHatBoundaryDataFlather(s_grid, t); });
    // index 6: NTHMP Monai wavemaker on left, Sommerfeld radiation on R/T/B
    // (Codex Phase 8 — reflective walls caused standing-wave amplification with HR)
    ExchangeHatBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeHatBoundaryDataWavemaker(s_grid, t); });

    ExchangeBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeBoundaryData1(s_grid, t); });
    ExchangeBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeBoundaryData2(s_grid, t); });
    ExchangeBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeBoundaryData3(s_grid, t); });
    ExchangeBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeBoundaryData4(s_grid, t); });
    // index 4: Sommerfeld radiation на всех 4 границах домена
    ExchangeBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeBoundaryDataRadiation(s_grid, t); });
    // index 5: Flather 1976 characteristic open boundary (production for Pacific)
    ExchangeBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeBoundaryDataFlather(s_grid, t); });
    // index 6: NTHMP Monai wavemaker on left, Sommerfeld radiation on R/T/B
    ExchangeBoundaryDataFunctions.push_back([](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeBoundaryDataWavemaker(s_grid, t); });
}

void SingleLineGridSimulation::SetParameters(double g, double tau, std::function<double(int, int)> delta_x, double delta_y,
    int nx, int ny, int timesteps, int num_threads, int num_subgrids,
    int exchangeBoundaryIndex, int exchangeHatBoundaryIndex,
    CoordSystem coord_system, double center_latitude_rad, double min_depth_to_calc)
{
    g_ = g;
    tau_ = tau;
    delta_x_ = delta_x;
    delta_y_ = delta_y;
    nx_ = nx;
    ny_ = ny;
    timesteps_ = timesteps;
    num_threads_ = num_threads;
	num_subgrids_ = (num_subgrids > 0) ? num_subgrids : num_threads;
    coord_system_ = coord_system;
    center_latitude_rad_ = center_latitude_rad;
    minimum_depth_to_calculate = min_depth_to_calc;

    // Устанавливаем количество потоков OpenMP
    if (num_threads_ > 0) {
        omp_set_num_threads(num_threads_);
    }

    if (exchangeBoundaryIndex == -1) {
        ExchangeBoundaryData = ExchangeBoundaryDataFunctions[0];
    }
    else {
        ExchangeBoundaryData = ExchangeBoundaryDataFunctions[exchangeBoundaryIndex];
    }
    if (exchangeHatBoundaryIndex == -1) {
        ExchangeHatBoundaryData = ExchangeHatBoundaryDataFunctions[0];
    }
    else {
        ExchangeHatBoundaryData = ExchangeHatBoundaryDataFunctions[exchangeHatBoundaryIndex];
    }

    global_grid_ = Grid(nx_, ny_);
}

void SingleLineGridSimulation::SetParameters(DefaultParametersParser* parameters)
{
    g_ = 9.81;
    timesteps_ = parameters->GetTimesteps();
    num_threads_ = parameters->GetNumThreads();
	num_subgrids_ = parameters->GetNumSubgrids();
    tau_ = parameters->GetDeltaT();
    coord_system_ = parameters->GetCoordSystem();
    center_latitude_rad_ = 0.785398163; // TODO: вынести в файл параметров при необходимости
	if (num_subgrids_ <= 0) {
		num_subgrids_ = num_threads_;
	}
    auto deltaX = parameters->GetDeltaXStep();
    delta_x_ = [this, deltaX](int i, int j) {
        return deltaX;
        };
    delta_y_ = parameters->GetDeltaYStep();
    minimum_depth_to_calculate = parameters->GetMinDepthToCalc();

    // Устанавливаем количество потоков OpenMP
    if (num_threads_ > 0) {
        omp_set_num_threads(num_threads_);
    }
	if (num_subgrids_ <= 0) {
		num_subgrids_ = num_threads_;
	}

    global_grid_ = Grid(nx_, ny_, delta_x_, delta_y_);

    if (parameters->GetBoundaryExchangeType() == -1) {
        ExchangeBoundaryData = ExchangeBoundaryDataFunctions[0];
    }
    else {
        ExchangeBoundaryData = ExchangeBoundaryDataFunctions[parameters->GetBoundaryExchangeType()];
    }
    if (parameters->GetHatBoundaryExchangeType() == -1) {
        ExchangeHatBoundaryData = ExchangeHatBoundaryDataFunctions[0];
    }
    else {
        ExchangeHatBoundaryData = ExchangeHatBoundaryDataFunctions[parameters->GetHatBoundaryExchangeType()];
    }

    global_grid_ = Grid(nx_, ny_);
}

void SingleLineGridSimulation::ApplyParameters(
    double g,
    double tau,
    std::function<double(int, int)> delta_x,
    double delta_y,
    int nx,
    int ny,
	int timesteps,
	int num_threads,
	int num_subgrids,
    std::function<void(std::shared_ptr<SubGrid>, double)> exchangeBoundary,
    std::function<void(std::shared_ptr<SubGrid>, double)> exchangeHatBoundary,
    CoordSystem coord_system)
{
    g_ = g;
    tau_ = tau;
    delta_x_ = std::move(delta_x);
    delta_y_ = delta_y;
    nx_ = nx;
    ny_ = ny;
    timesteps_ = timesteps;
    num_threads_ = num_threads;
	num_subgrids_ = (num_subgrids > 0) ? num_subgrids : num_threads;
    coord_system_ = coord_system;
    center_latitude_rad_ = 0.785398163;

    // Устанавливаем количество потоков OpenMP
    if (num_threads_ > 0) {
        omp_set_num_threads(num_threads_);
    }

    ExchangeBoundaryData = std::move(exchangeBoundary);
    ExchangeHatBoundaryData = std::move(exchangeHatBoundary);
}

void SingleLineGridSimulation::SetGrid(Grid&& grid)
{
    global_grid_ = std::move(grid);
}

// Sponge layer на границах домена (Codex plan 2026-04-10).
//
// Применяется ПОСЛЕ ExchangeBoundaryData каждого шага к реальным ячейкам
// в sponge-зонах. Затухание вида:
//     q(t+dt) = q(t) * exp(-sigma(d) * dt)
// где d = расстояние (в ячейках) от стенки до ячейки, а sigma зависит от
// обеих расстояний (east и south могут перекрываться в углу). Для каждой
// ячейки выбирается максимальное sigma из четырёх возможных источников.
//
// Ширина каждой стороны задаётся независимо (sponge_*_cells). 0 = сторона
// не затухает. Применяется к eta, u, v; H восстанавливается как D + eta.
void SingleLineGridSimulation::ApplySpongeLayer() {
    const bool any_sponge =
        sponge_west_cells_ > 0 || sponge_east_cells_ > 0 ||
        sponge_south_cells_ > 0 || sponge_north_cells_ > 0;
    if (!any_sponge || sponge_sigma_max_ <= 0.0) return;

    static int once = 0;
    if (once == 0) {
        once = 1;
        std::cerr << "[ApplySpongeLayer] active: west=" << sponge_west_cells_
                  << " east=" << sponge_east_cells_
                  << " south=" << sponge_south_cells_
                  << " north=" << sponge_north_cells_
                  << " sigma_max=" << sponge_sigma_max_ << std::endl;
    }

    const double sigma_max_dt = sponge_sigma_max_ * tau_;
    const int num_subgrids = static_cast<int>(subgrids_.size());

    // Лямбда: вычисляет максимальное нормализованное quadratic ramp для
    // данной пары (i_global, j_global). Возвращает 0 если вне всех sponge-зон.
    auto sponge_ramp_sq = [&](int i_global, int j_global) -> double {
        double max_ramp_sq = 0.0;
        if (sponge_west_cells_ > 0 && i_global < sponge_west_cells_) {
            const int r = sponge_west_cells_ - i_global;
            const double ramp = static_cast<double>(r) / sponge_west_cells_;
            max_ramp_sq = std::max(max_ramp_sq, ramp * ramp);
        }
        if (sponge_east_cells_ > 0 && i_global >= nx_ - sponge_east_cells_) {
            const int r = i_global - (nx_ - sponge_east_cells_) + 1;
            const double ramp = static_cast<double>(r) / sponge_east_cells_;
            max_ramp_sq = std::max(max_ramp_sq, ramp * ramp);
        }
        if (sponge_south_cells_ > 0 && j_global < sponge_south_cells_) {
            const int r = sponge_south_cells_ - j_global;
            const double ramp = static_cast<double>(r) / sponge_south_cells_;
            max_ramp_sq = std::max(max_ramp_sq, ramp * ramp);
        }
        if (sponge_north_cells_ > 0 && j_global >= ny_ - sponge_north_cells_) {
            const int r = j_global - (ny_ - sponge_north_cells_) + 1;
            const double ramp = static_cast<double>(r) / sponge_north_cells_;
            max_ramp_sq = std::max(max_ramp_sq, ramp * ramp);
        }
        return max_ramp_sq;
    };

    for (int sg_idx = 0; sg_idx < num_subgrids; ++sg_idx) {
        // Phase 5: never-active subgrid stays at lake-at-rest (η=0, u=v=0);
        // sponge multiplies by (1-α·ramp²) which is a no-op on zeros.
        // Skip the whole inner double loop for those.
        if (active_region_ && !active_region_->IsActive(sg_idx)) continue;
        auto& sg = subgrids_[sg_idx];
        const int local_nx = sg->local_nx_;
        const int local_ny = sg->local_ny_;
        const DecompositionDirection dir = sg->GetDecompDir();

        // Мапирование local → global зависит от направления декомпозиции.
        // Horizontal: X разбивается по подсеткам, Y полный.
        //   global_i = global_i_start_ + (i_local - 1)
        //   global_j = j_local
        // Vertical: Y разбивается, X полный с 2 ghost.
        //   global_i = i_local - 1
        //   global_j = global_i_start_ + (j_local - 1)  (в этом режиме global_i_start_ хранит j-диапазон)

        for (int i_local = 1; i_local < local_nx - 1; ++i_local) {
            int i_global;
            if (dir == DecompositionDirection::Horizontal) {
                i_global = sg->global_i_start_ + (i_local - 1);
            } else {
                i_global = i_local - 1;
            }
            if (i_global < 0 || i_global >= nx_) continue;

            for (int j_local = 1; j_local < local_ny - 1; ++j_local) {
                int j_global;
                if (dir == DecompositionDirection::Horizontal) {
                    j_global = j_local;
                } else {
                    j_global = sg->global_i_start_ + (j_local - 1);
                }
                if (j_global < 0 || j_global >= ny_) continue;

                const double ramp_sq = sponge_ramp_sq(i_global, j_global);
                if (ramp_sq <= 0.0) continue;

                const double sigma_local_dt = sigma_max_dt * ramp_sq;
                const double damp = std::exp(-sigma_local_dt);

                const double eta_new = sg->local_grid_.eta(i_local, j_local) * damp;
                const double u_new   = sg->local_grid_.u(i_local, j_local)   * damp;
                const double v_new   = sg->local_grid_.v(i_local, j_local)   * damp;
                sg->local_grid_.eta(i_local, j_local) = eta_new;
                sg->local_grid_.u(i_local, j_local)   = u_new;
                sg->local_grid_.v(i_local, j_local)   = v_new;
                sg->local_grid_.H(i_local, j_local)   =
                    sg->local_grid_.D(i_local, j_local) + eta_new;
            }
        }
    }
}

void SingleLineGridSimulation::SetSaveInterval(int interval) {
    save_interval_ = interval;
}

void SingleLineGridSimulation::Initialize() {
    std::cerr << "[SLGS] Initialize enter; nx=" << nx_ << " ny=" << ny_
              << " num_subgrids_=" << num_subgrids_ << " num_threads=" << num_threads_
              << std::flush << std::endl;
    if (!global_grid_.IsInitialized()) {
        std::cerr << "[SLGS] global grid not initialized!" << std::flush << std::endl;
        throw std::exception("Global grid is not initialized!");
    }

    int num_subgrids = (num_subgrids_ > 0) ? num_subgrids_ : num_threads_;

    // Определяем размер разбивки по направлению
    int split_size = (decomp_dir_ == DecompositionDirection::Horizontal) ? nx_ : ny_;
    int m = split_size / num_subgrids;
    std::cerr << "[SLGS] num_subgrids=" << num_subgrids << " split_size=" << split_size
              << " m=" << m << std::flush << std::endl;

    global_grid_.subgrid_count = num_subgrids;

    for (int process_id = 0; process_id < num_subgrids; ++process_id) {
        int range_start = process_id * m;
        int range_end = (process_id == num_subgrids - 1) ? split_size : (process_id + 1) * m;

        std::cerr << "[SLGS] building subgrid " << process_id
                  << " range=[" << range_start << "," << range_end << ")"
                  << std::flush << std::endl;

        auto dummy_barrier = std::make_shared<std::barrier<>>(1);

        auto subgrid = std::make_shared<SubGrid>(
            process_id,
            range_start,
            range_end,
            global_grid_,
            g_, tau_, delta_x_, delta_y_, minimum_depth_to_calculate,
            dummy_barrier,
            dummy_barrier,
            1,
            coord_system_,
            center_latitude_rad_,
            decomp_dir_
        );
        subgrid->SetCoriolisEnabled(coriolis_enabled_);
        subgrid->ExchangeBoundaryData = ExchangeBoundaryData;
        subgrid->ExchangeHatBoundaryData = ExchangeHatBoundaryData;
        subgrid->Initialize();
        subgrids_.push_back(subgrid);
    }

    // Устанавливаем соседей для подрешёток
    for (int i = 0; i < num_subgrids; ++i) {
        SubGrid* neighbor_up = (i > 0) ? subgrids_[i - 1].get() : nullptr;
        SubGrid* neighbor_down = (i < num_subgrids - 1) ? subgrids_[i + 1].get() : nullptr;
        subgrids_[i]->SetNeighbors(neighbor_up, neighbor_down);
    }

    InitializeActiveRegionManagerIfRequested();
}

// Helper: создаёт active_region_ если ACTIVE_REGION_MODE задан.
// Вызывается из всех Initialize() overloads после SetNeighbors().
// Идемпотентен — повторный вызов no-op.
//
// На Phase 1-2 manager созданный здесь Run()-ом НЕ используется
// (skip-логики ещё нет). Phase 3+ начнёт его использовать.
void SingleLineGridSimulation::InitializeActiveRegionManagerIfRequested(
    const IBaseParametersParser* parser_or_null) {
    if (active_region_) return;  // already created
    auto cfg = active_region::Config::FromEnv();
    if (cfg.mode == active_region::Mode::Disabled) return;

    active_region_ = std::make_unique<active_region::ActiveRegionManager>();
    active_region_->Configure(cfg);

    // Build SubgridSpan list from real subgrid primary intervals
    // (GetPrimaryStart/End added in SubGrid.h — Phase 3).
    std::vector<active_region::SubgridSpan> spans;
    spans.reserve(subgrids_.size());
    for (int i = 0; i < static_cast<int>(subgrids_.size()); ++i) {
        active_region::SubgridSpan sp;
        sp.id = i;
        sp.primary_start = subgrids_[i]->GetPrimaryStart();
        sp.primary_end   = subgrids_[i]->GetPrimaryEnd();
        sp.decomp_dir    = subgrids_[i]->GetDecompDir();
        spans.push_back(sp);
    }

    // Conservative dx_min for activation cell-radius computations:
    //   * Cartesian: delta_x is constant
    //   * Spherical: delta_x = R*cos(lat)*dlam, smallest at highest latitude
    // Codex Phase 3 M2: start from +inf, take min only of positive samples,
    // then explicitly validate to avoid silent fallback-to-1.
    double dx_min = std::numeric_limits<double>::infinity();
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            const double dxij = delta_x_(i, j);
            if (dxij > 0.0 && dxij < dx_min) dx_min = dxij;
        }
        // Cartesian: dx is i-independent, one row is enough.
        if (coord_system_ != CoordSystem::Spherical) break;
    }
    if (!std::isfinite(dx_min) || dx_min <= 0.0 || delta_y_ <= 0.0) {
        std::cerr << "[ActiveRegion] WARNING: cannot derive valid dx_min/dy "
                     "(dx_min=" << dx_min << ", dy=" << delta_y_
                  << "); active_region manager NOT created." << std::endl;
        active_region_.reset();
        return;
    }

    const std::vector<std::unique_ptr<ISource>>* sources = nullptr;
    if (parser_or_null != nullptr) {
        sources = &parser_or_null->GetSources();
    }

    // Scan global D for max depth — needed when ACTIVE_REGION_CMAX is auto.
    double max_depth = 0.0;
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            const double d = static_cast<double>(global_grid_.D(i, j));
            if (d > max_depth) max_depth = d;
        }
    }

    active_region_->InitializeFromGridAndSources(
        &global_grid_, sources, spans, nx_, ny_, dx_min, delta_y_, g_, max_depth);
}

void SingleLineGridSimulation::Initialize(ZDataParser& parser)
{
    nx_ = parser.GetCols();
    ny_ = parser.GetRows();

    global_grid_ = Grid(nx_, ny_, delta_x_, delta_y_);
    global_grid_.SetParameters(tau_, parser);

    if (!global_grid_.IsInitialized()) {
        throw std::exception("Global grid is not initialized!");
    }

	int num_subgrids = (num_subgrids_ > 0) ? num_subgrids_ : num_threads_;
    int m = nx_ / num_subgrids;

    global_grid_.subgrid_count = num_subgrids;

    for (int process_id = 0; process_id < num_subgrids; ++process_id) {
        int global_i_start = process_id * m;
        int global_i_end = (process_id == num_subgrids - 1) ? nx_ : (process_id + 1) * m;

        auto dummy_barrier = std::make_shared<std::barrier<>>(1);

        auto subgrid = std::make_shared<SubGrid>(
            process_id,
            global_i_start,
            global_i_end,
            global_grid_,
            g_, tau_,
            delta_x_, delta_y_, minimum_depth_to_calculate,
            dummy_barrier,
            dummy_barrier,
            1,
            coord_system_,
            center_latitude_rad_
        );
        subgrid->SetCoriolisEnabled(coriolis_enabled_);
        subgrid->ExchangeBoundaryData = ExchangeBoundaryData;
        subgrid->ExchangeHatBoundaryData = ExchangeHatBoundaryData;
        subgrid->Initialize();
        subgrids_.push_back(subgrid);
    }

    for (int i = 0; i < num_subgrids; ++i) {
        SubGrid* neighbor_up = (i > 0) ? subgrids_[i - 1].get() : nullptr;
        SubGrid* neighbor_down = (i < num_subgrids - 1) ? subgrids_[i + 1].get() : nullptr;
        subgrids_[i]->SetNeighbors(neighbor_up, neighbor_down);
    }
    InitializeActiveRegionManagerIfRequested();
}

constexpr double EARTH_RADIUS = 6371000.0;
constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double METERS_PER_DEGREE = 111300;

void SingleLineGridSimulation::Initialize(ASCParser& parser, DefaultParametersParser& simulator_params)
{
    nx_ = parser.GetCols();
    ny_ = parser.GetRows();

    global_grid_ = Grid(nx_, ny_, delta_x_, delta_y_);
    global_grid_.SetParamsTest6(tau_, parser, simulator_params);

    if (!global_grid_.IsInitialized()) {
        throw std::exception("Global grid is not initialized!");
    }

    ExchangeBoundaryData = [](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeBoundaryData1(s_grid, t);
        };
    ExchangeHatBoundaryData = [](std::shared_ptr<SubGrid> s_grid, double t) {
        BoundaryCondition::ExchangeHatBoundaryData1(s_grid, t);
        };

	int num_subgrids = (num_subgrids_ > 0) ? num_subgrids_ : num_threads_;
    int m = nx_ / num_subgrids;

    global_grid_.subgrid_count = num_subgrids;

    for (int process_id = 0; process_id < num_subgrids; ++process_id) {
        int global_i_start = process_id * m;
        int global_i_end = (process_id == num_subgrids - 1) ? nx_ : (process_id + 1) * m;

        auto dummy_barrier = std::make_shared<std::barrier<>>(1);

        auto subgrid = std::make_shared<SubGrid>(
            process_id,
            global_i_start,
            global_i_end,
            global_grid_,
            g_, tau_,
            delta_x_, delta_y_, 8.0,
            dummy_barrier,
            dummy_barrier,
            1,
            coord_system_,
            center_latitude_rad_
        );
        subgrid->SetCoriolisEnabled(coriolis_enabled_);
        subgrid->ExchangeBoundaryData = ExchangeBoundaryData;
        subgrid->ExchangeHatBoundaryData = ExchangeHatBoundaryData;
        subgrid->Initialize();
        subgrids_.push_back(subgrid);
    }

    for (int i = 0; i < num_subgrids; ++i) {
        SubGrid* neighbor_up = (i > 0) ? subgrids_[i - 1].get() : nullptr;
        SubGrid* neighbor_down = (i < num_subgrids - 1) ? subgrids_[i + 1].get() : nullptr;
        subgrids_[i]->SetNeighbors(neighbor_up, neighbor_down);
    }
    InitializeActiveRegionManagerIfRequested(&simulator_params);
}

void SingleLineGridSimulation::Run() {
    const int num_subgrids = static_cast<int>(subgrids_.size());

    // Checkpoint resume: read PACIFIC_RESUME_STEP env to offset snapshot filenames.
    // Loop runs t=0..timesteps-1 as usual, but saved snapshots use t+1+step_offset_,
    // so resumed runs produce filenames continuing from previous segment.
    {
        char* off_buf = nullptr; size_t off_sz = 0;
        _dupenv_s(&off_buf, &off_sz, "PACIFIC_RESUME_STEP");
        if (off_buf) {
            step_offset_ = std::atoi(off_buf);
            free(off_buf);
            std::cout << "[RESUME] step offset = " << step_offset_
                      << " (snapshot filenames will start from "
                      << (step_offset_ + save_interval_) << ")" << std::endl;
        }
    }

    // ========== CFL CHECK ==========
    {
        double max_depth = 0.0;
        double min_dx = std::numeric_limits<double>::max();
        for (int i = 0; i < nx_; ++i) {
            for (int j = 0; j < ny_; ++j) {
                double d = global_grid_.D(i, j);
                if (d > max_depth) max_depth = d;
                double dx = delta_x_(i, j);
                if (dx > 0.0 && dx < min_dx) min_dx = dx;
            }
        }
        double c = std::sqrt(g_ * max_depth);
        double cfl = tau_ * (c / min_dx + c / delta_y_);

        std::cout << "=== Simulation Parameters ===" << std::endl;
        std::cout << "  Coord system: " << CoordSystemToString(coord_system_) << std::endl;
        std::cout << "  Grid: " << nx_ << " x " << ny_ << std::endl;
        std::cout << "  Subgrids: " << num_subgrids << std::endl;
        std::cout << "  tau: " << tau_ << " s" << std::endl;
        std::cout << "  dx_min: " << min_dx << " m, dy: " << delta_y_ << " m" << std::endl;
        std::cout << "  Max depth: " << max_depth << " m" << std::endl;
        std::cout << "  Wave speed (sqrt(gH)): " << c << " m/s" << std::endl;
        std::cout << "  CFL: " << cfl << std::endl;
        std::cout << "  Timesteps: " << timesteps_ << std::endl;
        if (cfl > 1.0) {
            std::cerr << "WARNING: CFL = " << cfl << " > 1.0 — simulation may be unstable!" << std::endl;
        }
        // Codex Phase E pre-paper check: enforce FV_CFL_MAX as a hard limit
        // when FV solver is active. MacCormack legacy keeps just-warning.
        if (fv::config().fv_enabled()) {
            const double fv_cfl_max = fv::config().cfl_max;
            if (cfl > fv_cfl_max) {
                throw std::runtime_error(
                    "FV Godunov CFL " + std::to_string(cfl) +
                    " exceeds FV_CFL_MAX=" + std::to_string(fv_cfl_max) +
                    ". Reduce tau or refine grid.");
            }
            std::cout << "  FV_CFL_MAX: " << fv_cfl_max << " (enforced)" << std::endl;
        }
        std::cout << "=============================" << std::endl;
    }

    // Phase 8 NTHMP: optional point-probe (gauge) output. Env-controlled.
    //   GAUGE_OUTPUT_FILE  — destination CSV (e.g. Result/monai/gauges.csv)
    //   GAUGE_POINTS       — "i1,j1;i2,j2;...;iN,jN" in global cell indices
    //   GAUGE_SAMPLE_EVERY — sample every N timesteps (default 1)
    // Writes columns: t_phys, eta(p1), eta(p2), ..., eta(pN).
    std::vector<std::pair<int,int>> gauge_points;
    std::ofstream gauge_out;
    int gauge_sample_every = 1;
    {
        char* fbuf = nullptr; size_t fsz = 0;
        _dupenv_s(&fbuf, &fsz, "GAUGE_OUTPUT_FILE");
        char* pbuf = nullptr; size_t psz = 0;
        _dupenv_s(&pbuf, &psz, "GAUGE_POINTS");
        char* nbuf = nullptr; size_t nsz = 0;
        _dupenv_s(&nbuf, &nsz, "GAUGE_SAMPLE_EVERY");
        if (fbuf && pbuf) {
            // Parse "i1,j1;i2,j2;..."
            std::string pts(pbuf);
            size_t pos = 0;
            while (pos < pts.size()) {
                size_t semi = pts.find(';', pos);
                std::string token = pts.substr(pos, (semi == std::string::npos) ? std::string::npos : semi - pos);
                size_t comma = token.find(',');
                if (comma != std::string::npos) {
                    int gi = std::atoi(token.substr(0, comma).c_str());
                    int gj = std::atoi(token.substr(comma + 1).c_str());
                    if (gi >= 0 && gi < nx_ && gj >= 0 && gj < ny_) {
                        gauge_points.emplace_back(gi, gj);
                    }
                }
                if (semi == std::string::npos) break;
                pos = semi + 1;
            }
            if (nbuf) gauge_sample_every = std::max(1, std::atoi(nbuf));
            std::filesystem::create_directories(std::filesystem::path(fbuf).parent_path());
            gauge_out.open(fbuf);
            if (gauge_out.is_open()) {
                gauge_out << "t_phys";
                for (size_t k = 0; k < gauge_points.size(); ++k) {
                    gauge_out << ",eta_g" << (k + 1)
                              << "(i=" << gauge_points[k].first
                              << "_j=" << gauge_points[k].second << ")";
                }
                gauge_out << std::endl;
                std::cerr << "[Gauge] writing " << gauge_points.size() << " gauges to "
                          << fbuf << " every " << gauge_sample_every << " step(s)"
                          << std::endl;
                // Phase E pre-paper: dump bathymetry D at each gauge so we can
                // verify all gauges are in WET cells (D > 0). Critical NTHMP
                // sanity check before claiming gauge comparison validity.
                for (size_t k = 0; k < gauge_points.size(); ++k) {
                    const int gi = gauge_points[k].first;
                    const int gj = gauge_points[k].second;
                    std::cerr << "[Gauge] g" << (k + 1) << " i=" << gi << " j=" << gj
                              << " D=" << global_grid_.D(gi, gj)
                              << " H=" << global_grid_.H(gi, gj)
                              << " eta=" << global_grid_.eta(gi, gj)
                              << ((global_grid_.D(gi, gj) > 0) ? " WET" : " DRY")
                              << std::endl;
                }
            }
        }
        if (fbuf) free(fbuf);
        if (pbuf) free(pbuf);
        if (nbuf) free(nbuf);
    }

    // Helper to read η at a global (i, j) point from the appropriate subgrid.
    auto sample_eta_at = [&](int gi, int gj) -> double {
        // Find subgrid owning this point (Horizontal decomp: range on i).
        for (int sg_idx = 0; sg_idx < static_cast<int>(subgrids_.size()); ++sg_idx) {
            auto& sg = subgrids_[sg_idx];
            const int start = sg->GetPrimaryStart();
            const int end   = sg->GetPrimaryEnd();
            if (sg->GetDecompDir() == DecompositionDirection::Horizontal) {
                if (gi >= start && gi < end) {
                    const int i_local = (gi - start) + 1;  // +1 for ghost
                    const int j_local = gj;
                    if (i_local >= 0 && i_local < sg->local_nx_ &&
                        j_local >= 0 && j_local < sg->local_ny_) {
                        return static_cast<double>(sg->local_grid_.eta(i_local, j_local));
                    }
                }
            } else {
                if (gj >= start && gj < end) {
                    const int j_local = (gj - start) + 1;
                    const int i_local = gi;
                    if (i_local >= 0 && i_local < sg->local_nx_ &&
                        j_local >= 0 && j_local < sg->local_ny_) {
                        return static_cast<double>(sg->local_grid_.eta(i_local, j_local));
                    }
                }
            }
        }
        return 0.0;
    };

    // Phase 4: precompute dx_min once for activation expansion.
    // Recomputed here (independent of CFL scan above) to avoid coupling.
    double active_dx_min = std::numeric_limits<double>::infinity();
    if (active_region_) {
        for (int i = 0; i < nx_; ++i) {
            for (int j = 0; j < ny_; ++j) {
                const double dxij = delta_x_(i, j);
                if (dxij > 0.0 && dxij < active_dx_min) active_dx_min = dxij;
            }
            if (coord_system_ != CoordSystem::Spherical) break;
        }
    }

    // Lambda: skip exchange iff THIS subgrid and BOTH neighbors are inactive
    // (Codex Phase 4 spec — active↔inactive must still copy ghosts).
    auto skip_exchange = [&](int i) -> bool {
        if (!active_region_) return false;
        const int left  = i - 1;
        const int right = i + 1;
        return active_region_->ShouldSkipExchange(i, left)
            && active_region_->ShouldSkipExchange(i, right);
    };

    for (int t = 0; t < timesteps_; ++t) {
        // ========== ACTIVATION UPDATE (BEFORE half-step — Codex invariant) ==========
        if (active_region_) {
            active_region_->UpdateActivation(subgrids_, t, tau_, active_dx_min);
        }

        // ========== ПОЛОВИННЫЙ ШАГ ==========
        // Последовательно обрабатываем все подсетки
        // OpenMP параллелизация происходит ВНУТРИ ComputeHalfStep()
        for (int i = 0; i < num_subgrids; ++i) {
            if (active_region_ && !active_region_->IsActive(i)) continue;
            subgrids_[i]->ComputeHalfStep();
        }

        // Обмен граничными данными для hat-переменных
        for (int i = 0; i < num_subgrids; ++i) {
            if (skip_exchange(i)) continue;
            subgrids_[i]->ExchangeHatBoundaryData(subgrids_[i], t);
        }

        // ========== ПОЛНЫЙ ШАГ ==========
        // For active subgrids in active-region mode pass global_t + global TVD
        // schedule flag, so late-activated stripes hit the same TVD phase as
        // baseline (Phase 0 TVD fix Variant A).
        const bool use_global_tvd = active_region_ && active_region_->UseGlobalTvdSchedule();
        for (int i = 0; i < num_subgrids; ++i) {
            if (active_region_ && !active_region_->IsActive(i)) continue;
            if (use_global_tvd) {
                subgrids_[i]->ComputeFullStep(t, true);
            } else {
                subgrids_[i]->ComputeFullStep();
            }
        }

        // Применение граничных условий
        for (int i = 0; i < num_subgrids; ++i) {
            if (skip_exchange(i)) continue;
            subgrids_[i]->ExchangeBoundaryData(subgrids_[i], t);
        }

        // Phase 6b: post-step signal diagnostic (Hybrid mode only). Reads actual
        // post-step eta; logs late-risk and optionally wakes up neighbors for t+1.
        if (active_region_) {
            active_region_->RunSignalDiagnostic(subgrids_, t);
        }

        // Phase 8 NTHMP: gauge sampling (env-controlled). Writes one CSV row
        // per sampled timestep with eta at each requested (i, j) point.
        if (gauge_out.is_open() && !gauge_points.empty() &&
            (t % gauge_sample_every == 0)) {
            const double t_phys = (static_cast<double>(t) + 1.0) * tau_;
            gauge_out << t_phys;
            for (const auto& pt : gauge_points) {
                gauge_out << "," << sample_eta_at(pt.first, pt.second);
            }
            gauge_out << "\n";
        }

        // Sponge layer на границах — устраняет Kelvin wave у стенок
        // (Codex plan 2026-04-10). Вызывается ПОСЛЕ exchange, чтобы damping
        // применялся к актуальному состоянию ghost+interior.
        const bool any_sponge_enabled =
            (sponge_west_cells_ > 0 || sponge_east_cells_ > 0 ||
             sponge_south_cells_ > 0 || sponge_north_cells_ > 0) &&
            sponge_sigma_max_ > 0.0;
        if (any_sponge_enabled) {
            ApplySpongeLayer();
        }

        // C1 optimization: UpdateGlobalGrid removed from hot loop.
        // Subgrids exchange boundary data directly between local grids.
        // Global grid is only synced when saving snapshots (below).

#ifdef SAVE_ETA_MAX_MAP
        // Track eta_max per-subgrid (uses local_grid_ data, no global sync needed).
        // Phase 5: never-active subgrid stays at lake-at-rest -> max(|eta|) over its
        // cells is constant 0 (assuming initial eta=0 there). Final flush below
        // handles the case if eta_max accumulator was already non-zero.
        for (int i = 0; i < num_subgrids; ++i) {
            if (active_region_ && !active_region_->IsActive(i)) continue;
            subgrids_[i]->UpdateEtaMaxIfNeeded(t);
        }
#endif

        // Save snapshots at configurable interval — sync to global grid on demand.
        // Phase 5: never-active subgrid still holds its baseline lake-at-rest state
        // in global_grid_ (it was set during Initialize and never touched), so skip
        // the redundant copy for unchanged regions.
        if (save_interval_ > 0 && (t + 1) % save_interval_ == 0) {
            for (int i = 0; i < num_subgrids; ++i) {
                if (active_region_ && !active_region_->IsActive(i)) continue;
                subgrids_[i]->UpdateGlobalGrid(i);
            }
            const int save_step = (t + 1) + step_offset_;
            SaveDataToFile("eta", global_grid_, save_step);
            if (save_velocities_) {
                SaveDataToFile("u", global_grid_, save_step);
                SaveDataToFile("v", global_grid_, save_step);
            }
        }

        // Progress: print only when percentage changes (every timesteps/100 steps).
        // Previous code flushed stdout EVERY step — a syscall per step that
        // dominated runtime for short tests (e.g., ~70% of test1, ~5% of test2).
        const int progress_stride = (timesteps_ >= 100) ? (timesteps_ / 100) : 1;
        if ((t + 1) % progress_stride == 0 || (t + 1) == timesteps_) {
            int percentage = static_cast<int>(100.0 * (t + 1) / timesteps_);
            std::cout << "\rSimulation progress: " << percentage << "%";
            std::cout.flush();
        }
    }

    std::cout << std::endl;

    // Final sync: subgrids → global grid (for output and summary)
    for (int i = 0; i < num_subgrids; ++i) {
        subgrids_[i]->UpdateGlobalGrid(i);
    }

    // Save final eta snapshot (offset for checkpoint resume)
    SaveDataToFile("eta", global_grid_, timesteps_ + step_offset_);

#ifdef SAVE_ETA_MAX_MAP
    // Flush per-subgrid eta_max/u_max/v_max to global grid
    for (int i = 0; i < num_subgrids; ++i) {
        subgrids_[i]->FlushEtaMaxToGlobal();
        subgrids_[i]->FlushUMaxToGlobal();
        subgrids_[i]->FlushVMaxToGlobal();
    }
    SaveDataToFile("eta_max", global_grid_, 0);
    SaveDataToFile("u_max", global_grid_, 0);
    SaveDataToFile("v_max", global_grid_, 0);
#endif

    // Write summary: max wave height, grid dimensions, parameters
    {
        std::filesystem::create_directories(result_folder_);
        std::string summary_file = result_folder_ + "/summary.txt";
        std::ofstream summary(summary_file);
        if (summary.is_open()) {
            double max_eta = 0.0, min_eta = 0.0;
            for (int i = 0; i < nx_; ++i) {
                for (int j = 0; j < ny_; ++j) {
                    double e = global_grid_.eta(i, j);
                    if (e > max_eta) max_eta = e;
                    if (e < min_eta) min_eta = e;
                }
            }
            double max_eta_max = 0.0;
#ifdef SAVE_ETA_MAX_MAP
            for (int i = 0; i < nx_; ++i) {
                for (int j = 0; j < ny_; ++j) {
                    double em = global_grid_.eta_max(i, j);
                    if (em > max_eta_max) max_eta_max = em;
                }
            }
#endif
            // Interior metric: max_eta в окне [i_min..i_max) × [j_min..j_max)
            // Исключает sponge-зоны и Kelvin wave у стенок.
            double max_eta_interior = 0.0;
            double min_eta_interior = 0.0;
            const bool interior_enabled =
                interior_i_min_ >= 0 && interior_i_max_ > interior_i_min_ &&
                interior_j_min_ >= 0 && interior_j_max_ > interior_j_min_;
            if (interior_enabled) {
                const int imin = std::max(0, interior_i_min_);
                const int imax = std::min(nx_, interior_i_max_);
                const int jmin = std::max(0, interior_j_min_);
                const int jmax = std::min(ny_, interior_j_max_);
                for (int i = imin; i < imax; ++i) {
                    for (int j = jmin; j < jmax; ++j) {
                        double e = global_grid_.eta(i, j);
                        if (e > max_eta_interior) max_eta_interior = e;
                        if (e < min_eta_interior) min_eta_interior = e;
                    }
                }
            }

            summary << "nx=" << nx_ << std::endl;
            summary << "ny=" << ny_ << std::endl;
            summary << "timesteps=" << timesteps_ << std::endl;
            summary << "tau=" << tau_ << std::endl;
            summary << "save_interval=" << save_interval_ << std::endl;
            summary << "coord_system=" << CoordSystemToString(coord_system_) << std::endl;
            summary << "coriolis_enabled=" << (coriolis_enabled_ ? 1 : 0) << std::endl;
            summary << "save_velocities=" << (save_velocities_ ? 1 : 0) << std::endl;
            summary << "sponge_west=" << sponge_west_cells_ << std::endl;
            summary << "sponge_east=" << sponge_east_cells_ << std::endl;
            summary << "sponge_south=" << sponge_south_cells_ << std::endl;
            summary << "sponge_north=" << sponge_north_cells_ << std::endl;
            summary << "sponge_sigma_max=" << sponge_sigma_max_ << std::endl;
            summary << "max_eta_final=" << std::setprecision(std::numeric_limits<double>::max_digits10) << max_eta << std::endl;
            summary << "min_eta_final=" << min_eta << std::endl;
            summary << "max_eta_max=" << max_eta_max << std::endl;
            if (interior_enabled) {
                summary << "interior_i_min=" << interior_i_min_ << std::endl;
                summary << "interior_i_max=" << interior_i_max_ << std::endl;
                summary << "interior_j_min=" << interior_j_min_ << std::endl;
                summary << "interior_j_max=" << interior_j_max_ << std::endl;
                summary << "max_eta_interior=" << max_eta_interior << std::endl;
                summary << "min_eta_interior=" << min_eta_interior << std::endl;
            }
            // Active region tracking provenance (only when manager exists).
            // Disabled-mode summary is unchanged — bitwise gate preserved.
            if (active_region_) {
                active_region_->WriteSummary(summary);
            }
            // Phase E pre-paper: FV positivity-clamp counter.
            if (fv::config().fv_enabled()) {
                const auto clamps = fv::g_positivity_clamps.load(std::memory_order_relaxed);
                summary << "fv_positivity_clamps=" << clamps << std::endl;
            }
            std::cerr << "Summary written to " << summary_file << std::endl;
            std::cerr << "  max_eta_final    = " << max_eta << " m" << std::endl;
            std::cerr << "  max_eta_max      = " << max_eta_max << " m" << std::endl;
            if (fv::config().fv_enabled()) {
                std::cerr << "  fv_positivity_clamps = "
                          << fv::g_positivity_clamps.load(std::memory_order_relaxed) << std::endl;
            }
            if (interior_enabled) {
                std::cerr << "  max_eta_interior = " << max_eta_interior
                    << " m  (i in [" << interior_i_min_ << ", " << interior_i_max_
                    << "), j in [" << interior_j_min_ << ", " << interior_j_max_ << "))" << std::endl;
            }
        }
    }
}

// ThreadFunction больше не используется - удаляем или оставляем пустым для совместимости
void SingleLineGridSimulation::ThreadFunction(int process_id) {
    // Не используется в оптимизированной версии
    (void)process_id;
}

void SingleLineGridSimulation::SaveSubGridArray(const std::string& filename, const double* arr, int ny, int nx) {
    std::fstream file(filename, std::ios::binary | std::ios::in | std::ios::out);

    if (!file) {
        std::cerr << "File does not exist. Creating a new one: " << filename << std::endl;
        std::ofstream new_file(filename, std::ios::binary);
        if (!new_file) {
            std::cerr << "Error: Cannot create file " << filename << std::endl;
            return;
        }
        file.open(filename, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) {
            std::cerr << "Error: Cannot reopen file " << filename << std::endl;
            return;
        }
    }

    if (!file) {
        std::cerr << "Error: seekp() failed." << std::endl;
        file.close();
        return;
    }

    file.write(reinterpret_cast<const char*>(arr), nx * ny * sizeof(Real));

    if (!file) {
        std::cerr << "Error: write() failed." << std::endl;
    }

    file.close();
}

void SingleLineGridSimulation::CollectResults(int timestep) {
    SaveDataToFile("eta", global_grid_, timestep);
}

void SingleLineGridSimulation::LoadSubGridArray(const std::string& filename, double* arr, int ny, int nx) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for reading." << std::endl;
        return;
    }

    std::unique_ptr<Real[]> buffer = std::make_unique<Real[]>(nx * ny);
    file.read(reinterpret_cast<char*>(buffer.get()), nx * ny * sizeof(Real));
    std::memcpy(arr, buffer.get(), nx * ny * sizeof(Real));
    file.close();
}

void SingleLineGridSimulation::SaveDataToFile(const std::string& variable_name, const Grid& grid, int timestep) {
    const std::string& folder_name = result_folder_;
    const std::string filename = folder_name + "/" + variable_name + "_timestep_" + std::to_string(timestep) + ".txt";

    std::filesystem::create_directories(folder_name);

    std::function<double(int, int)> getValue;
    if (variable_name == "H") {
        getValue = [&grid](int i, int j) { return grid.H(i, j); };
    }
    else if (variable_name == "D") {
        getValue = [&grid](int i, int j) { return grid.D(i, j); };
    }
    else if (variable_name == "u") {
        getValue = [&grid](int i, int j) { return grid.u(i, j); };
    }
    else if (variable_name == "v") {
        getValue = [&grid](int i, int j) { return grid.v(i, j); };
    }
    else if (variable_name == "eta") {
        getValue = [&grid](int i, int j) { return grid.eta(i, j); };
    }
    else if (variable_name == "eta_max") {
        getValue = [&grid](int i, int j) { return grid.eta_max(i, j); };
    }
    else if (variable_name == "u_max") {
        getValue = [&grid](int i, int j) { return grid.u_max(i, j); };
    }
    else if (variable_name == "v_max") {
        getValue = [&grid](int i, int j) { return grid.v_max(i, j); };
    }
    else {
        std::cerr << "Error: Unknown variable name " << variable_name << "." << std::endl;
        return;
    }

    std::ostringstream output_buffer;
    int nx = grid.GetNx();
    int ny = grid.GetNy();
    output_buffer << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            auto value = getValue(i, j);
            output_buffer << value << " ";
        }
        output_buffer << "\n";
    }

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }
    file << output_buffer.str();
}

int SingleLineGridSimulation::GetTimesteps() const {
    return timesteps_;
}

void SingleLineGridSimulation::SetCoordSystem(CoordSystem coord_system) {
    coord_system_ = coord_system;
}

CoordSystem SingleLineGridSimulation::GetCoordSystem() const {
    return coord_system_;
}
