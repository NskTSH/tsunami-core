// SingleLineGridSimulationCUDA.cu
// ПОЛНОСТЬЮ ПЕРЕРАБОТАННАЯ версия с корректным subgrid pipeline
// Следует CPU референсу:
// 1. Predictor для ВСЕХ подсеток
// 2. Обмен hat-переменных
// 3. Corrector для ВСЕХ подсеток
// 4. Обмен base-переменных

#include "SingleLineGridSimulationCUDA.cuh"
#include "MacCormackCUDA.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <string>
#include <limits>
#include <vector>
#if __has_include(<filesystem>)
#  include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#  include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#  error "No filesystem header available"
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IDX(i, j, ny) ((i) * (ny) + (j))

// ================== GridCUDA Implementation ==================

GridCUDA::GridCUDA(int nx, int ny) : nx_(nx), ny_(ny), is_initialized_(false) {
    if (nx > 0 && ny > 0) {
        size_t size = static_cast<size_t>(nx) * ny;
        H_ = std::make_unique<double[]>(size);
        u_ = std::make_unique<double[]>(size);
        v_ = std::make_unique<double[]>(size);
        eta_ = std::make_unique<double[]>(size);
        D_ = std::make_unique<double[]>(size);
        eta_max_ = std::make_unique<double[]>(size);
        u_max_ = std::make_unique<double[]>(size);
        v_max_ = std::make_unique<double[]>(size);
        std::fill(H_.get(), H_.get() + size, 0.0);
        std::fill(u_.get(), u_.get() + size, 0.0);
        std::fill(v_.get(), v_.get() + size, 0.0);
        std::fill(eta_.get(), eta_.get() + size, 0.0);
        std::fill(D_.get(), D_.get() + size, 0.0);
        std::fill(eta_max_.get(), eta_max_.get() + size, 0.0);
        std::fill(u_max_.get(), u_max_.get() + size, 0.0);
        std::fill(v_max_.get(), v_max_.get() + size, 0.0);
    }
}

GridCUDA::~GridCUDA() = default;

GridCUDA::GridCUDA(GridCUDA&& other) noexcept
    : nx_(other.nx_), ny_(other.ny_),
      H_(std::move(other.H_)), u_(std::move(other.u_)),
      v_(std::move(other.v_)), eta_(std::move(other.eta_)),
      D_(std::move(other.D_)), eta_max_(std::move(other.eta_max_)),
      u_max_(std::move(other.u_max_)), v_max_(std::move(other.v_max_)),
      is_initialized_(other.is_initialized_), subgrid_count(other.subgrid_count) {
    other.nx_ = 0; other.ny_ = 0; other.is_initialized_ = false;
}

GridCUDA& GridCUDA::operator=(GridCUDA&& other) noexcept {
    if (this != &other) {
        nx_ = other.nx_; ny_ = other.ny_;
        H_ = std::move(other.H_); u_ = std::move(other.u_);
        v_ = std::move(other.v_); eta_ = std::move(other.eta_);
        D_ = std::move(other.D_); eta_max_ = std::move(other.eta_max_);
        u_max_ = std::move(other.u_max_); v_max_ = std::move(other.v_max_);
        is_initialized_ = other.is_initialized_; subgrid_count = other.subgrid_count;
        other.nx_ = 0; other.ny_ = 0; other.is_initialized_ = false;
    }
    return *this;
}

GridCUDA::GridCUDA(const GridCUDA& other)
    : nx_(other.nx_), ny_(other.ny_), is_initialized_(other.is_initialized_), subgrid_count(other.subgrid_count) {
    if (nx_ > 0 && ny_ > 0) {
        size_t size = static_cast<size_t>(nx_) * ny_;
        H_ = std::make_unique<double[]>(size); u_ = std::make_unique<double[]>(size);
        v_ = std::make_unique<double[]>(size); eta_ = std::make_unique<double[]>(size);
        D_ = std::make_unique<double[]>(size); eta_max_ = std::make_unique<double[]>(size);
        u_max_ = std::make_unique<double[]>(size); v_max_ = std::make_unique<double[]>(size);
        std::copy(other.H_.get(), other.H_.get() + size, H_.get());
        std::copy(other.u_.get(), other.u_.get() + size, u_.get());
        std::copy(other.v_.get(), other.v_.get() + size, v_.get());
        std::copy(other.eta_.get(), other.eta_.get() + size, eta_.get());
        std::copy(other.D_.get(), other.D_.get() + size, D_.get());
        std::copy(other.eta_max_.get(), other.eta_max_.get() + size, eta_max_.get());
        std::copy(other.u_max_.get(), other.u_max_.get() + size, u_max_.get());
        std::copy(other.v_max_.get(), other.v_max_.get() + size, v_max_.get());
    }
}

GridCUDA& GridCUDA::operator=(const GridCUDA& other) {
    if (this != &other) {
        nx_ = other.nx_; ny_ = other.ny_;
        is_initialized_ = other.is_initialized_; subgrid_count = other.subgrid_count;
        if (nx_ > 0 && ny_ > 0) {
            size_t size = static_cast<size_t>(nx_) * ny_;
            H_ = std::make_unique<double[]>(size); u_ = std::make_unique<double[]>(size);
            v_ = std::make_unique<double[]>(size); eta_ = std::make_unique<double[]>(size);
            D_ = std::make_unique<double[]>(size); eta_max_ = std::make_unique<double[]>(size);
            u_max_ = std::make_unique<double[]>(size); v_max_ = std::make_unique<double[]>(size);
            std::copy(other.H_.get(), other.H_.get() + size, H_.get());
            std::copy(other.u_.get(), other.u_.get() + size, u_.get());
            std::copy(other.v_.get(), other.v_.get() + size, v_.get());
            std::copy(other.eta_.get(), other.eta_.get() + size, eta_.get());
            std::copy(other.D_.get(), other.D_.get() + size, D_.get());
            std::copy(other.eta_max_.get(), other.eta_max_.get() + size, eta_max_.get());
            std::copy(other.u_max_.get(), other.u_max_.get() + size, u_max_.get());
            std::copy(other.v_max_.get(), other.v_max_.get() + size, v_max_.get());
        } else {
            H_.reset(); u_.reset(); v_.reset(); eta_.reset(); D_.reset(); eta_max_.reset();
            u_max_.reset(); v_max_.reset();
        }
    }
    return *this;
}

double& GridCUDA::H(int i, int j) { return H_[IDX(i, j, ny_)]; }
double& GridCUDA::u(int i, int j) { return u_[IDX(i, j, ny_)]; }
double& GridCUDA::v(int i, int j) { return v_[IDX(i, j, ny_)]; }
double& GridCUDA::eta(int i, int j) { return eta_[IDX(i, j, ny_)]; }
double& GridCUDA::D(int i, int j) { return D_[IDX(i, j, ny_)]; }
double& GridCUDA::eta_max(int i, int j) { return eta_max_[IDX(i, j, ny_)]; }
double& GridCUDA::u_max(int i, int j) { return u_max_[IDX(i, j, ny_)]; }
double& GridCUDA::v_max(int i, int j) { return v_max_[IDX(i, j, ny_)]; }
const double& GridCUDA::H(int i, int j) const { return H_[IDX(i, j, ny_)]; }
const double& GridCUDA::u(int i, int j) const { return u_[IDX(i, j, ny_)]; }
const double& GridCUDA::v(int i, int j) const { return v_[IDX(i, j, ny_)]; }
const double& GridCUDA::eta(int i, int j) const { return eta_[IDX(i, j, ny_)]; }
const double& GridCUDA::D(int i, int j) const { return D_[IDX(i, j, ny_)]; }
const double& GridCUDA::eta_max(int i, int j) const { return eta_max_[IDX(i, j, ny_)]; }
const double& GridCUDA::u_max(int i, int j) const { return u_max_[IDX(i, j, ny_)]; }
const double& GridCUDA::v_max(int i, int j) const { return v_max_[IDX(i, j, ny_)]; }

// ================== Subgrid Descriptor ==================

struct SubgridDescriptor {
    int global_i_start;  // First column in global grid
    int global_i_end;    // One past last column
    int local_nx;        // local_nx = (global_i_end - global_i_start) + 2
    bool has_left_neighbor;
    bool has_right_neighbor;
};

// ================== CUDA Implementation (pimpl) ==================

struct SingleLineGridSimulationCUDA::CudaImpl {
    std::unique_ptr<IMacCormackCUDA> simulator;
    bool is_allocated = false;
    
    // Буферы для подсеток (base переменные)
    std::vector<double> H_buffer, u_buffer, v_buffer, eta_buffer, D_buffer, eta_max_buffer;
    
    // Буферы для hat переменных
    std::vector<double> H_hat_buffer, u_hat_buffer, v_hat_buffer, eta_hat_buffer;
    
    // Глобальные буферы для hat данных всего домена (для корректного обмена)
    std::vector<double> H_hat_global, u_hat_global, v_hat_global, eta_hat_global;
    
    // Сохранённые base данные (D не меняется, храним для corrector)
    std::vector<double> D_global_saved;
    
    int subgrid_gpu_nx = 0, subgrid_gpu_ny = 0;
    std::vector<SubgridDescriptor> subgrid_descs;
    
    void AllocateSubgridBuffers(int gpu_nx, int gpu_ny, int global_nx, int global_ny) {
        size_t local_size = static_cast<size_t>(gpu_nx) * gpu_ny;
        size_t global_size = static_cast<size_t>(global_nx) * global_ny;
        
        H_buffer.resize(local_size, 0.0); u_buffer.resize(local_size, 0.0);
        v_buffer.resize(local_size, 0.0); eta_buffer.resize(local_size, 0.0);
        D_buffer.resize(local_size, 0.0); eta_max_buffer.resize(local_size, 0.0);
        H_hat_buffer.resize(local_size, 0.0); u_hat_buffer.resize(local_size, 0.0);
        v_hat_buffer.resize(local_size, 0.0); eta_hat_buffer.resize(local_size, 0.0);
        
        // Глобальные hat буферы
        H_hat_global.resize(global_size, 0.0); u_hat_global.resize(global_size, 0.0);
        v_hat_global.resize(global_size, 0.0); eta_hat_global.resize(global_size, 0.0);
        D_global_saved.resize(global_size, 0.0);
        
        subgrid_gpu_nx = gpu_nx; subgrid_gpu_ny = gpu_ny;
    }
    
    void ClearLocalBuffers() {
        std::fill(H_buffer.begin(), H_buffer.end(), 0.0);
        std::fill(u_buffer.begin(), u_buffer.end(), 0.0);
        std::fill(v_buffer.begin(), v_buffer.end(), 0.0);
        std::fill(eta_buffer.begin(), eta_buffer.end(), 0.0);
        std::fill(D_buffer.begin(), D_buffer.end(), 0.0);
        std::fill(eta_max_buffer.begin(), eta_max_buffer.end(), 0.0);
        std::fill(H_hat_buffer.begin(), H_hat_buffer.end(), 0.0);
        std::fill(u_hat_buffer.begin(), u_hat_buffer.end(), 0.0);
        std::fill(v_hat_buffer.begin(), v_hat_buffer.end(), 0.0);
        std::fill(eta_hat_buffer.begin(), eta_hat_buffer.end(), 0.0);
    }
};

// ================== Constructor/Destructor ==================

SingleLineGridSimulationCUDA::SingleLineGridSimulationCUDA()
    : g_(9.81), tau_(0.1), delta_x_const_(1000.0), delta_y_(1000.0),
      nx_(0), ny_(0), timesteps_(1000), min_calc_value_(8.0),
      num_subgrids_(1), boundary_condition_type_(0), hat_boundary_condition_type_(0),
      current_timestep_(0),
      use_variable_dx_(false), save_interval_(0), output_folder_("Result"),
      cuda_impl_(std::make_unique<CudaImpl>()) {
    delta_x_func_ = [this](int, int) { return delta_x_const_; };
}

SingleLineGridSimulationCUDA::~SingleLineGridSimulationCUDA() { FreeGPU(); }

// ================== SetParameters ==================

void SingleLineGridSimulationCUDA::SetParameters(double g, double tau,
    std::function<double(int, int)> delta_x, double delta_y, int nx, int ny, int timesteps) {
    g_ = g; tau_ = tau; delta_x_func_ = delta_x; delta_y_ = delta_y;
    nx_ = nx; ny_ = ny; timesteps_ = timesteps;
    delta_x_const_ = delta_x(0, 0); use_variable_dx_ = false;
    global_grid_ = GridCUDA(nx, ny);
}

void SingleLineGridSimulationCUDA::SetParameters(double g, double tau,
    double delta_x_const, double delta_y, int nx, int ny, int timesteps, double min_depth) {
    g_ = g; tau_ = tau; delta_x_const_ = delta_x_const; delta_y_ = delta_y;
    nx_ = nx; ny_ = ny; timesteps_ = timesteps; min_calc_value_ = min_depth;
    use_variable_dx_ = false;
    delta_x_func_ = [delta_x_const](int, int) { return delta_x_const; };
    global_grid_ = GridCUDA(nx, ny);
}

void SingleLineGridSimulationCUDA::SetParameters(double g, double tau,
    double delta_x_const, double delta_y, int nx, int ny, int timesteps,
    int num_subgrids, int exchangeBoundaryIndex, int exchangeHatBoundaryIndex) {
    g_ = g; tau_ = tau; delta_x_const_ = delta_x_const; delta_y_ = delta_y;
    nx_ = nx; ny_ = ny; timesteps_ = timesteps; min_calc_value_ = 8.0; use_variable_dx_ = false;
    num_subgrids_ = (num_subgrids < 1) ? 1 : num_subgrids;
    boundary_condition_type_ = exchangeBoundaryIndex;
    hat_boundary_condition_type_ = exchangeHatBoundaryIndex;
    current_timestep_ = 0;
    delta_x_func_ = [delta_x_const](int, int) { return delta_x_const; };
    global_grid_ = GridCUDA(nx, ny);
}

void SingleLineGridSimulationCUDA::SetGrid(GridCUDA&& grid) {
    global_grid_ = std::move(grid);
    nx_ = global_grid_.GetNx(); ny_ = global_grid_.GetNy();
}

void SingleLineGridSimulationCUDA::SetGrid(const GridCUDA& grid) {
    nx_ = grid.GetNx(); ny_ = grid.GetNy();
    global_grid_ = GridCUDA(nx_, ny_);
    size_t size = static_cast<size_t>(nx_) * ny_;
    std::copy(grid.GetH(), grid.GetH() + size, global_grid_.GetH());
    std::copy(grid.GetU(), grid.GetU() + size, global_grid_.GetU());
    std::copy(grid.GetV(), grid.GetV() + size, global_grid_.GetV());
    std::copy(grid.GetEta(), grid.GetEta() + size, global_grid_.GetEta());
    std::copy(grid.GetD(), grid.GetD() + size, global_grid_.GetD());
    std::copy(grid.GetEtaMax(), grid.GetEtaMax() + size, global_grid_.GetEtaMax());
}

// ================== Initialize ==================

void SingleLineGridSimulationCUDA::Initialize() {
    if (nx_ <= 0 || ny_ <= 0) throw std::runtime_error("Grid dimensions not set!");
    if (!global_grid_.IsInitialized()) throw std::runtime_error("Global grid not initialized!");
    
    // Создаём дескрипторы подсеток
    cuda_impl_->subgrid_descs.clear();
    int subgrid_work_nx = nx_ / num_subgrids_;
    int max_local_nx = 0;
    
    for (int s = 0; s < num_subgrids_; ++s) {
        SubgridDescriptor desc;
        desc.global_i_start = s * subgrid_work_nx;
        desc.global_i_end = (s == num_subgrids_ - 1) ? nx_ : (s + 1) * subgrid_work_nx;
        desc.local_nx = (desc.global_i_end - desc.global_i_start) + 2;
        desc.has_left_neighbor = (s > 0);
        desc.has_right_neighbor = (s < num_subgrids_ - 1);
        cuda_impl_->subgrid_descs.push_back(desc);
        if (desc.local_nx > max_local_nx) max_local_nx = desc.local_nx;
    }
    
    // X-padded layout (matches CPU SubGrid::local_nx = global_nx + 2):
    // device storage uses i in [0, nx+1], where i=0 and i=nx+1 are ghost columns
    // outside the physical domain, and global eta(i_g, j) ↔ device eta(i_g+1, j).
    int gpu_nx = max_local_nx;
    
    SimulationParams params;
    params.g = g_; params.tau = tau_; params.delta_x_const = delta_x_const_;
    params.delta_y = delta_y_; params.min_calc_value = min_calc_value_;
    params.use_variable_dx = use_variable_dx_;
    params.boundaryConditionType = boundary_condition_type_;
    params.hatBoundaryConditionType = hat_boundary_condition_type_;
    params.coord_system = (coord_system_id_ == 1)
        ? CudaCoordSystem::Spherical : CudaCoordSystem::Cartesian;
    params.coriolis_enabled = coriolis_enabled_;
    params.center_latitude_rad = center_latitude_rad_;

    // === Stabilizer env toggles (mirror CPU SubGrid tvd_config/absorb_config) ===
    // Default OFF → bitwise-identical to existing runs unless explicitly enabled.
    auto env_str = [](const char* name) -> std::string {
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, name);
        std::string v = (buf && *buf) ? std::string(buf) : std::string();
        if (buf) free(buf);
        return v;
    };
    {
        std::string s;
        if ((s = env_str("TVD_ENABLED")) == "1") {
            params.tvd_enabled = true;
            params.tvd_force_minmod = (env_str("TVD_LIMITER") == "minmod");
            if (!(s = env_str("TVD_SHALLOW_DEPTH")).empty()) params.tvd_shallow_depth = std::atof(s.c_str());
            if (!(s = env_str("TVD_BOUNDARY_SKIP")).empty()) params.tvd_boundary_skip = std::atoi(s.c_str());
            if (!(s = env_str("TVD_APPLY_EVERY")).empty()) { int n = std::atoi(s.c_str()); params.tvd_apply_every = (n > 0) ? n : 1; }
            std::cout << "[TVD] post-corrector ENABLED, limiter="
                      << (params.tvd_force_minmod ? "minmod" : "vanLeer+minmod-near-shallow")
                      << ", shallow=" << params.tvd_shallow_depth
                      << ", boundary_skip=" << params.tvd_boundary_skip
                      << ", apply_every=" << params.tvd_apply_every << std::endl;
        }
        if (env_str("ABSORB_ENABLED") == "1") {
            params.absorb_enabled = true;
            if (!(s = env_str("ABSORB_LAYER_WIDTH")).empty()) params.absorb_layer_width = std::atoi(s.c_str());
            if (!(s = env_str("ABSORB_ALPHA_MAX")).empty()) params.absorb_alpha_max = std::atof(s.c_str());
            std::cout << "[ABSORB] north layer ENABLED: width=" << params.absorb_layer_width
                      << ", alpha_max=" << params.absorb_alpha_max << std::endl;
        }
        // Active Region (GPU): only geometric mode for the monolithic path — a
        // single south-growing row band anchored at j=0 (source is near north).
        // ACTIVE_REGION_MODE=geometric|geo enables it; margin via ACTIVE_REGION_MARGIN.
        s = env_str("ACTIVE_REGION_MODE");
        for (char& ch : s) ch = (char)std::tolower((unsigned char)ch);
        if (s == "geometric" || s == "geo" || s == "hybrid" || s == "all-active" || s == "all") {
            // Seed: southernmost source row (max j where |eta|>0) + max depth for c_max.
            const double* eta = global_grid_.GetEta();
            const double* Dg  = global_grid_.GetD();
            int seed_j_max = 0; double max_depth = 0.0;
            const size_t tot = (size_t)nx_ * ny_;
            for (size_t k = 0; k < tot; ++k) if (Dg[k] > max_depth) max_depth = Dg[k];
            for (int i = 0; i < nx_; ++i)
                for (int j = 0; j < ny_; ++j)
                    if (std::abs(eta[(size_t)i*ny_+j]) > 0.0 && j > seed_j_max) seed_j_max = j;
            params.ar_enabled = true;
            params.ar_c_max = std::sqrt(g_ * max_depth);
            params.ar_seed_j_max = seed_j_max;
            std::string m = env_str("ACTIVE_REGION_MARGIN");
            params.ar_margin_cells = m.empty() ? 32 : std::atoi(m.c_str());
            std::cout << "[ActiveRegion] GPU south-band ENABLED: seed_j_max=" << seed_j_max
                      << " (of ny=" << ny_ << "), c_max=" << params.ar_c_max
                      << " m/s, margin=" << params.ar_margin_cells << std::endl;
        }
    }

    std::cout << "\n=== Simulation Parameters ===" << std::endl;
    std::cout << "Grid: " << nx_ << "x" << ny_ << ", timesteps=" << timesteps_ << std::endl;
    std::cout << "g=" << g_ << ", tau=" << tau_ << ", dx=" << delta_x_const_ << ", dy=" << delta_y_ << std::endl;
    std::cout << "BC type=" << boundary_condition_type_ << ", num_subgrids=" << num_subgrids_ << std::endl;
    
    if (num_subgrids_ > 1) {
        std::cout << "CORRECT GLOBAL PIPELINE enabled:" << std::endl;
        std::cout << "  GPU allocation: " << gpu_nx << "x" << ny_ << std::endl;
        for (int s = 0; s < num_subgrids_; ++s) {
            const auto& d = cuda_impl_->subgrid_descs[s];
            std::cout << "  Subgrid " << s << ": i=[" << d.global_i_start << "," << d.global_i_end 
                      << "), local_nx=" << d.local_nx << std::endl;
        }
        cuda_impl_->AllocateSubgridBuffers(gpu_nx, ny_, nx_, ny_);
        
        // Сохраняем D (батиметрию) - она не меняется
        std::copy(global_grid_.GetD(), global_grid_.GetD() + static_cast<size_t>(nx_) * ny_,
                  cuda_impl_->D_global_saved.data());
    }
    
    cuda_impl_->simulator = CreateMacCormackCUDA(gpu_nx, ny_, params, use_fp32_);
    cuda_impl_->is_allocated = true;
    std::cout << "Precision: " << cuda_impl_->simulator->GetPrecisionName() << std::endl;

    // Auto-detect variable delta_x (when delta_x_func varies along j).
    // CPU uses delta_x_(i, j) — for Test 6 (Japan Sea) it depends on latitude.
    // Build dx[j] array indexed by global j (Y not padded).
    if (delta_x_func_) {
        const double dx0 = delta_x_func_(0, 0);
        bool varies = false;
        for (int j = 1; j < ny_; ++j) {
            if (std::abs(delta_x_func_(0, j) - dx0) > 1e-9) { varies = true; break; }
        }
        if (varies) {
            std::vector<double> dx_array(ny_);
            for (int j = 0; j < ny_; ++j) {
                dx_array[j] = delta_x_func_(0, j);
            }
            cuda_impl_->simulator->LoadDeltaXArray(dx_array.data(), ny_);
            std::cout << "Variable delta_x detected: dx[0]=" << dx_array[0]
                      << ", dx[ny/2]=" << dx_array[ny_/2]
                      << ", dx[ny-1]=" << dx_array[ny_-1] << std::endl;
        }
    }
    
    if (num_subgrids_ == 1) CopyToGPU();
    
    std::cout << "SingleLineGridSimulationCUDA initialized" << std::endl;
}

void SingleLineGridSimulationCUDA::AllocateGPU() {}
void SingleLineGridSimulationCUDA::FreeGPU() {
    if (cuda_impl_ && cuda_impl_->simulator) {
        cuda_impl_->simulator.reset();
        cuda_impl_->is_allocated = false;
    }
}

void SingleLineGridSimulationCUDA::CopyToGPU() {
    if (!cuda_impl_->simulator) return;
    // Build padded host buffers (gpu_nx, ny_), then upload.
    // Device storage: i_dev = i_global + 1; i_dev=0 and i_dev=gpu_nx-1 are ghost.
    const int gpu_nx = cuda_impl_->simulator->GetNx();
    const int ny = ny_;
    const size_t pad_size = static_cast<size_t>(gpu_nx) * ny;
    std::vector<double> H_pad(pad_size, 0.0);
    std::vector<double> u_pad(pad_size, 0.0);
    std::vector<double> v_pad(pad_size, 0.0);
    std::vector<double> eta_pad(pad_size, 0.0);
    std::vector<double> D_pad(pad_size, 0.0);

    const double* H_g = global_grid_.GetH();
    const double* u_g = global_grid_.GetU();
    const double* v_g = global_grid_.GetV();
    const double* eta_g = global_grid_.GetEta();
    const double* D_g = global_grid_.GetD();

    for (int i_g = 0; i_g < nx_; ++i_g) {
        const int i_dev = i_g + 1;
        for (int j = 0; j < ny; ++j) {
            const int idx_g = i_g * ny + j;
            const int idx_d = i_dev * ny + j;
            H_pad[idx_d] = H_g[idx_g];
            u_pad[idx_d] = u_g[idx_g];
            v_pad[idx_d] = v_g[idx_g];
            eta_pad[idx_d] = eta_g[idx_g];
            D_pad[idx_d] = D_g[idx_g];
        }
    }
    // Initial ghost columns: ZERO (matches CPU SubGrid::Initialize semantics —
    // out-of-range global_i → zeros for H,u,v,eta,D). The zero D in ghost
    // creates a wet/dry interface that zeroes u on column i=1 (= global 0)
    // throughout the simulation, which is the CPU reference behavior.
    // Vectors are already zero-initialized by std::vector<double>(size, 0.0),
    // so no extra work needed here.

    cuda_impl_->simulator->LoadInitialData(
        H_pad.data(), u_pad.data(), v_pad.data(), eta_pad.data(), D_pad.data());
}

// Minimal readback: only eta_max (de-padded), for the production etamax-only benchmark.
void SingleLineGridSimulationCUDA::CopyEtaMaxFromGPU() {
    if (!cuda_impl_->simulator) return;
    cudaDeviceSynchronize();
    const int gpu_nx = cuda_impl_->simulator->GetNx();
    const int ny = ny_;
    const size_t pad_size = static_cast<size_t>(gpu_nx) * ny;
    std::vector<double> eta_max_pad(pad_size);
    cuda_impl_->simulator->CopyEtaMaxToHost(eta_max_pad.data());
    double* eta_max_g = global_grid_.GetEtaMax();
    for (int i_g = 0; i_g < nx_; ++i_g) {
        const int i_dev = i_g + 1;
        for (int j = 0; j < ny; ++j) {
            eta_max_g[i_g * ny + j] = eta_max_pad[i_dev * ny + j];
        }
    }
}

void SingleLineGridSimulationCUDA::CopyFromGPU() {
    if (!cuda_impl_->simulator) return;
    cudaDeviceSynchronize();
    const int gpu_nx = cuda_impl_->simulator->GetNx();
    const int ny = ny_;
    const size_t pad_size = static_cast<size_t>(gpu_nx) * ny;
    std::vector<double> H_pad(pad_size);
    std::vector<double> u_pad(pad_size);
    std::vector<double> v_pad(pad_size);
    std::vector<double> eta_pad(pad_size);
    std::vector<double> eta_max_pad(pad_size);
    cuda_impl_->simulator->CopyToHost(H_pad.data(), u_pad.data(), v_pad.data(), eta_pad.data());
    cuda_impl_->simulator->CopyEtaMaxToHost(eta_max_pad.data());

    std::vector<double> u_max_pad, v_max_pad;
    if (save_velocities_) {
        u_max_pad.resize(pad_size);
        v_max_pad.resize(pad_size);
        cuda_impl_->simulator->CopyUMaxToHost(u_max_pad.data());
        cuda_impl_->simulator->CopyVMaxToHost(v_max_pad.data());
    }

    double* H_g = global_grid_.GetH();
    double* u_g = global_grid_.GetU();
    double* v_g = global_grid_.GetV();
    double* eta_g = global_grid_.GetEta();
    double* eta_max_g = global_grid_.GetEtaMax();
    double* u_max_g = global_grid_.GetUMax();
    double* v_max_g = global_grid_.GetVMax();

    for (int i_g = 0; i_g < nx_; ++i_g) {
        const int i_dev = i_g + 1;
        for (int j = 0; j < ny; ++j) {
            const int idx_g = i_g * ny + j;
            const int idx_d = i_dev * ny + j;
            H_g[idx_g] = H_pad[idx_d];
            u_g[idx_g] = u_pad[idx_d];
            v_g[idx_g] = v_pad[idx_d];
            eta_g[idx_g] = eta_pad[idx_d];
            eta_max_g[idx_g] = eta_max_pad[idx_d];
            if (save_velocities_) {
                u_max_g[idx_g] = u_max_pad[idx_d];
                v_max_g[idx_g] = v_max_pad[idx_d];
            }
        }
    }
}

// ================== SUBGRID PIPELINE HELPERS ==================

void SingleLineGridSimulationCUDA::LoadSubgridFromGlobal(int s, const double* H_src, const double* u_src,
    const double* v_src, const double* eta_src, const double* D_src) {
    const auto& desc = cuda_impl_->subgrid_descs[s];
    int gpu_nx = cuda_impl_->subgrid_gpu_nx;
    int gpu_ny = cuda_impl_->subgrid_gpu_ny;
    
    cuda_impl_->ClearLocalBuffers();
    
    double* H_buf = cuda_impl_->H_buffer.data();
    double* u_buf = cuda_impl_->u_buffer.data();
    double* v_buf = cuda_impl_->v_buffer.data();
    double* eta_buf = cuda_impl_->eta_buffer.data();
    double* D_buf = cuda_impl_->D_buffer.data();
    
    for (int local_i = 0; local_i < desc.local_nx && local_i < gpu_nx; ++local_i) {
        int global_i = desc.global_i_start - 1 + local_i;
        if (global_i < 0 || global_i >= nx_) continue;
        for (int j = 0; j < ny_; ++j) {
            int local_idx = local_i * gpu_ny + j;
            int global_idx = global_i * ny_ + j;
            H_buf[local_idx] = H_src[global_idx];
            u_buf[local_idx] = u_src[global_idx];
            v_buf[local_idx] = v_src[global_idx];
            eta_buf[local_idx] = eta_src[global_idx];
            D_buf[local_idx] = D_src[global_idx];
        }
    }
    
    SubgridInfo info;
    info.has_left_neighbor = desc.has_left_neighbor;
    info.has_right_neighbor = desc.has_right_neighbor;
    info.subgrid_index = s;
    info.num_subgrids = num_subgrids_;
    cuda_impl_->simulator->SetSubgridInfo(info);
    
    cuda_impl_->simulator->LoadInitialData(H_buf, u_buf, v_buf, eta_buf, D_buf);
}

void SingleLineGridSimulationCUDA::SaveSubgridToGlobal(int s, double* H_dst, double* u_dst,
    double* v_dst, double* eta_dst, double* eta_max_dst) {
    const auto& desc = cuda_impl_->subgrid_descs[s];
    int gpu_nx = cuda_impl_->subgrid_gpu_nx;
    int gpu_ny = cuda_impl_->subgrid_gpu_ny;
    
    double* H_buf = cuda_impl_->H_buffer.data();
    double* u_buf = cuda_impl_->u_buffer.data();
    double* v_buf = cuda_impl_->v_buffer.data();
    double* eta_buf = cuda_impl_->eta_buffer.data();
    double* eta_max_buf = cuda_impl_->eta_max_buffer.data();
    
    cuda_impl_->simulator->CopyToHost(H_buf, u_buf, v_buf, eta_buf);
    if (eta_max_dst) cuda_impl_->simulator->CopyEtaMaxToHost(eta_max_buf);
    
    // Копируем только рабочую область
    int local_i_start = desc.has_left_neighbor ? 1 : 0;
    int local_i_end = desc.has_right_neighbor ? (desc.local_nx - 1) : desc.local_nx;
    
    for (int local_i = local_i_start; local_i < local_i_end && local_i < gpu_nx; ++local_i) {
        int global_i = desc.global_i_start - 1 + local_i;
        if (global_i < 0 || global_i >= nx_) continue;
        for (int j = 0; j < ny_; ++j) {
            int local_idx = local_i * gpu_ny + j;
            int global_idx = global_i * ny_ + j;
            H_dst[global_idx] = H_buf[local_idx];
            u_dst[global_idx] = u_buf[local_idx];
            v_dst[global_idx] = v_buf[local_idx];
            eta_dst[global_idx] = eta_buf[local_idx];
            if (eta_max_dst && eta_max_buf[local_idx] > eta_max_dst[global_idx]) {
                eta_max_dst[global_idx] = eta_max_buf[local_idx];
            }
        }
    }
}

void SingleLineGridSimulationCUDA::CopyHatFromGPU(int s) {
    const auto& desc = cuda_impl_->subgrid_descs[s];
    int gpu_nx = cuda_impl_->subgrid_gpu_nx;
    int gpu_ny = cuda_impl_->subgrid_gpu_ny;

    cuda_impl_->simulator->DownloadHatToHost(
        cuda_impl_->H_hat_buffer.data(), cuda_impl_->u_hat_buffer.data(),
        cuda_impl_->v_hat_buffer.data(), cuda_impl_->eta_hat_buffer.data());
    
    // Сохраняем в глобальный hat буфер только рабочую область (без ghost-колонок)
    double* H_hat_g = cuda_impl_->H_hat_global.data();
    double* u_hat_g = cuda_impl_->u_hat_global.data();
    double* v_hat_g = cuda_impl_->v_hat_global.data();
    double* eta_hat_g = cuda_impl_->eta_hat_global.data();
    
    // Копируем только рабочую область, исключая ghost-колонки
    int local_i_start = desc.has_left_neighbor ? 1 : 0;
    int local_i_end = desc.has_right_neighbor ? (desc.local_nx - 1) : desc.local_nx;
    
    for (int local_i = local_i_start; local_i < local_i_end && local_i < gpu_nx; ++local_i) {
        int global_i = desc.global_i_start - 1 + local_i;
        if (global_i < 0 || global_i >= nx_) continue;
        for (int j = 0; j < ny_; ++j) {
            int local_idx = local_i * gpu_ny + j;
            int global_idx = global_i * ny_ + j;
            H_hat_g[global_idx] = cuda_impl_->H_hat_buffer[local_idx];
            u_hat_g[global_idx] = cuda_impl_->u_hat_buffer[local_idx];
            v_hat_g[global_idx] = cuda_impl_->v_hat_buffer[local_idx];
            eta_hat_g[global_idx] = cuda_impl_->eta_hat_buffer[local_idx];
        }
    }
}

void SingleLineGridSimulationCUDA::LoadHatToGPU(int s) {
    const auto& desc = cuda_impl_->subgrid_descs[s];
    int gpu_nx = cuda_impl_->subgrid_gpu_nx;
    int gpu_ny = cuda_impl_->subgrid_gpu_ny;
    
    // Загружаем из глобального hat буфера в локальный, включая ghost-колонки.
    // Ghost-колонки подтягиваются как "внутренние" колонки соседей из глобального буфера:
    // - local_i=0 (левый ghost) -> читает внутреннюю колонку левого соседа
    // - local_i=local_nx-1 (правый ghost) -> читает внутреннюю колонку правого соседа
    const double* H_hat_g = cuda_impl_->H_hat_global.data();
    const double* u_hat_g = cuda_impl_->u_hat_global.data();
    const double* v_hat_g = cuda_impl_->v_hat_global.data();
    const double* eta_hat_g = cuda_impl_->eta_hat_global.data();
    
    for (int local_i = 0; local_i < desc.local_nx && local_i < gpu_nx; ++local_i) {
        int global_i = desc.global_i_start - 1 + local_i;
        if (global_i < 0 || global_i >= nx_) continue;
        for (int j = 0; j < ny_; ++j) {
            int local_idx = local_i * gpu_ny + j;
            int global_idx = global_i * ny_ + j;
            cuda_impl_->H_hat_buffer[local_idx] = H_hat_g[global_idx];
            cuda_impl_->u_hat_buffer[local_idx] = u_hat_g[global_idx];
            cuda_impl_->v_hat_buffer[local_idx] = v_hat_g[global_idx];
            cuda_impl_->eta_hat_buffer[local_idx] = eta_hat_g[global_idx];
        }
    }
    
    cuda_impl_->simulator->UploadHatFromHost(
        cuda_impl_->H_hat_buffer.data(), cuda_impl_->u_hat_buffer.data(),
        cuda_impl_->v_hat_buffer.data(), cuda_impl_->eta_hat_buffer.data());
}

// ================== CORRECT GLOBAL PIPELINE ==================

void SingleLineGridSimulationCUDA::Step() {
    if (!cuda_impl_->simulator) throw std::runtime_error("Simulator not initialized!");
    
    if (num_subgrids_ <= 1) {
        // ===== Режим без подсеток (максимальная производительность) =====
        cuda_impl_->simulator->Step(current_timestep_);
        cuda_impl_->simulator->UpdateEtaMax();
        if (save_velocities_) cuda_impl_->simulator->UpdateUVMax();
    } else {
        // ===== КОРРЕКТНЫЙ ГЛОБАЛЬНЫЙ MacCormack PIPELINE =====
        // Следует CPU референсу (SingleLineGridSimulation.cpp):
        // 1. ComputeHalfStep() для ВСЕХ подсеток
        // 2. ExchangeHatBoundaryData() - обмен hat-переменных
        // 3. ComputeFullStep() для ВСЕХ подсеток
        // 4. ExchangeBoundaryData() - обмен base-переменных + BC
        
        const double* D_saved = cuda_impl_->D_global_saved.data();
        
        // ========== PHASE 1: PREDICTOR для всех подсеток ==========
        for (int s = 0; s < num_subgrids_; ++s) {
            // Загружаем base данные (H, u, v, eta, D) из global_grid_
            LoadSubgridFromGlobal(s, global_grid_.GetH(), global_grid_.GetU(),
                                  global_grid_.GetV(), global_grid_.GetEta(), D_saved);
            
            // Выполняем ТОЛЬКО predictor (half step)
            cuda_impl_->simulator->ComputePredictorStep();
            
            cudaDeviceSynchronize();
            
            // Выгружаем hat данные в глобальный hat буфер
            CopyHatFromGPU(s);
        }
        
        // ========== PHASE 2: Hat boundary exchange ==========
        // Происходит автоматически: данные сохранены в H_hat_global включая ghost регионы
        // Каждая подсетка при Phase 1 записала свои данные, ghost cells читаются из соседей
        
        // ========== PHASE 3: CORRECTOR для всех подсеток ==========
        for (int s = 0; s < num_subgrids_; ++s) {
            // Загружаем base данные (нужны для corrector: D, начальные H,u,v,eta)
            LoadSubgridFromGlobal(s, global_grid_.GetH(), global_grid_.GetU(),
                                  global_grid_.GetV(), global_grid_.GetEta(), D_saved);
            
            // Загружаем обновлённые hat данные (с обменянными ghost cells)
            LoadHatToGPU(s);
            
            // Применяем BC к hat переменным (только на внешних границах домена!)
            cuda_impl_->simulator->ApplyHatBoundaryConditions(current_timestep_);
            
            // Выполняем ТОЛЬКО corrector (full step)
            cuda_impl_->simulator->ComputeCorrectorStep();
            
            // Применяем BC к base переменным
            cuda_impl_->simulator->ApplyBaseBoundaryConditions(current_timestep_);
            
            // Обновляем eta_max на GPU
            cuda_impl_->simulator->UpdateEtaMax();
            
            cudaDeviceSynchronize();
            
            // Выгружаем результаты в global_grid_
            SaveSubgridToGlobal(s, global_grid_.GetH(), global_grid_.GetU(),
                               global_grid_.GetV(), global_grid_.GetEta(), global_grid_.GetEtaMax());
        }
        
        // ========== PHASE 4: Base boundary exchange ==========
        // Происходит автоматически при следующем шаге через LoadSubgridFromGlobal
    }
    
    current_timestep_++;
}

// Устаревшие методы для совместимости
void SingleLineGridSimulationCUDA::CopySubgridToGPU(int s) {
    LoadSubgridFromGlobal(s, global_grid_.GetH(), global_grid_.GetU(),
                          global_grid_.GetV(), global_grid_.GetEta(), global_grid_.GetD());
}

void SingleLineGridSimulationCUDA::CopySubgridFromGPU(int s) {
    SaveSubgridToGlobal(s, global_grid_.GetH(), global_grid_.GetU(),
                        global_grid_.GetV(), global_grid_.GetEta(), global_grid_.GetEtaMax());
}

void SingleLineGridSimulationCUDA::StepSubgrid(int s) {
    // Устаревший некорректный метод - НЕ ИСПОЛЬЗОВАТЬ!
    CopySubgridToGPU(s);
    cuda_impl_->simulator->Step(current_timestep_);
    cuda_impl_->simulator->UpdateEtaMax();
    cudaDeviceSynchronize();
    CopySubgridFromGPU(s);
}

// ================== Run ==================

void SingleLineGridSimulationCUDA::Run() {
    if (!cuda_impl_->simulator) throw std::runtime_error("Simulator not initialized!");
    
    current_timestep_ = 0;
    
    std::cout << "\n=== Starting CUDA Simulation ===" << std::endl;
    std::cout << "Grid: " << nx_ << "x" << ny_ << ", Timesteps: " << timesteps_ << std::endl;
    std::cout << "BC type: " << boundary_condition_type_;
    if (boundary_condition_type_ == 4) std::cout << " (Sommerfeld radiation)";
    else if (boundary_condition_type_ >= 2) std::cout << " (dynamic wave)";
    std::cout << std::endl;
    std::cout << "Subgrids: " << num_subgrids_;
    if (num_subgrids_ > 1) std::cout << " (CORRECT GLOBAL PIPELINE)";
    std::cout << std::endl;
    
    // Optional speedup-benchmark trace: every BENCH_TRACE_EVERY steps, log the
    // cumulative compute time and active-row fraction. Enables the AR speedup-decay
    // curve (cumulative ms + active fraction vs step) from a SINGLE run.
    int bench_every = 0;
    {
        char* b = nullptr; size_t bs = 0;
        _dupenv_s(&b, &bs, "BENCH_TRACE_EVERY");
        if (b && *b) bench_every = std::atoi(b);
        if (b) free(b);
    }
    std::ofstream bench;
    if (bench_every > 0) {
        fs::create_directories(output_folder_);
        bench.open(output_folder_ + "/bench_trace.csv");
        bench << "step,cumulative_compute_ms,active_rows,active_fraction\n";
        bench << std::setprecision(10);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < timesteps_; ++t) {
        Step();

        if ((t + 1) % 100 == 0 || t == timesteps_ - 1) {
            int pct = static_cast<int>(100.0 * (t + 1) / timesteps_);
            std::cout << "\rProgress: " << pct << "% (" << t + 1 << "/" << timesteps_ << ")";
            std::cout.flush();
        }
        if (bench_every > 0 && ((t + 1) % bench_every == 0)) {
            cuda_impl_->simulator->Synchronize();   // make the cumulative time meaningful
            auto now = std::chrono::high_resolution_clock::now();
            double cum_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            int arows = cuda_impl_->simulator->GetActiveRows();
            bench << (t + 1) << "," << cum_ms << "," << arows << ","
                  << (double)arows / (double)ny_ << "\n";
            bench.flush();
        }
        if (!save_eta_max_only_ && save_interval_ > 0 && (t + 1) % save_interval_ == 0) {
            CollectResults(t + 1);
        }
    }
    if (bench.is_open()) bench.close();

    // === Pure compute time: GPU integration loop only (no host transfer / disk) ===
    cuda_impl_->simulator->Synchronize();
    auto t_compute_end = std::chrono::high_resolution_clock::now();
    const double compute_ms =
        (double)std::chrono::duration_cast<std::chrono::milliseconds>(t_compute_end - start_time).count();

    // === Save phase: device→host readback + file write, timed separately for the paper ===
    auto t_save_start = std::chrono::high_resolution_clock::now();
    if (save_eta_max_only_) {
        // Production benchmark: only the final eta_max map is needed.
        CopyEtaMaxFromGPU();
        SaveDataToFile("eta_max", global_grid_, 0);
    } else {
        if (num_subgrids_ == 1) CopyFromGPU();
        SaveDataToFile("eta", global_grid_, timesteps_);
        if (save_velocities_) {
            SaveDataToFile("u", global_grid_, timesteps_);
            SaveDataToFile("v", global_grid_, timesteps_);
        }
#ifdef SAVE_ETA_MAX_MAP
        SaveDataToFile("eta_max", global_grid_, 0);
#endif
        if (save_velocities_) {
            SaveDataToFile("u_max", global_grid_, 0);
            SaveDataToFile("v_max", global_grid_, 0);
        }
    }
    auto t_save_end = std::chrono::high_resolution_clock::now();
    const double save_ms =
        (double)std::chrono::duration_cast<std::chrono::milliseconds>(t_save_end - t_save_start).count();
    const double total_ms = compute_ms + save_ms;

    std::cout << "\n\n=== Performance ===" << std::endl;
    std::cout << "Compute time (integration only): " << compute_ms << " ms ("
              << compute_ms / 1000.0 << " s)" << std::endl;
    std::cout << "Save time (eta_max readback + write): " << save_ms << " ms ("
              << save_ms / 1000.0 << " s)" << std::endl;
    std::cout << "Total time (compute + save): " << total_ms << " ms ("
              << total_ms / 1000.0 << " s)" << std::endl;
    std::cout << "Time per step (compute): " << compute_ms / timesteps_ << " ms" << std::endl;
    std::cout << "Cells/second (compute): "
              << (static_cast<double>(nx_) * ny_ * timesteps_) / (compute_ms / 1000.0) / 1e6 << " M" << std::endl;

    // summary.txt — метаданные прогона (формат совместим с CPU read_summary;
    // Python-пайплайн терпит его отсутствие, но для паритета артефактов пишем).
    {
        const size_t total = static_cast<size_t>(nx_) * ny_;
        const double* emax = global_grid_.GetEtaMax();
        double max_eta_max = -1e300;
        for (size_t k = 0; k < total; ++k) if (emax[k] > max_eta_max) max_eta_max = emax[k];

        // eta (final) stats only meaningful when the final eta field was read back.
        double max_eta_final = 0.0, min_eta_final = 0.0;
        if (!save_eta_max_only_) {
            const double* eta = global_grid_.GetEta();
            max_eta_final = -1e300; min_eta_final = 1e300;
            for (size_t k = 0; k < total; ++k) {
                if (eta[k] > max_eta_final) max_eta_final = eta[k];
                if (eta[k] < min_eta_final) min_eta_final = eta[k];
            }
        }
        fs::create_directories(output_folder_);
        std::ofstream sf(output_folder_ + "/summary.txt");
        if (sf) {
            sf << std::setprecision(std::numeric_limits<double>::max_digits10);
            sf << "nx " << nx_ << "\n";
            sf << "ny " << ny_ << "\n";
            sf << "timesteps " << timesteps_ << "\n";
            sf << "tau " << tau_ << "\n";
            sf << "save_interval " << save_interval_ << "\n";
            sf << "coord_system " << (coord_system_id_ == 1 ? "Spherical" : "Cartesian") << "\n";
            sf << "coriolis_enabled " << (coriolis_enabled_ ? 1 : 0) << "\n";
            sf << "save_velocities " << (save_velocities_ ? 1 : 0) << "\n";
            sf << "save_eta_max_only " << (save_eta_max_only_ ? 1 : 0) << "\n";
            sf << "precision " << (use_fp32_ ? "fp32" : "fp64") << "\n";
            if (!save_eta_max_only_) {
                sf << "max_eta_final " << max_eta_final << "\n";
                sf << "min_eta_final " << min_eta_final << "\n";
            }
            sf << "max_eta_max " << max_eta_max << "\n";
            // Timing (for the paper): pure GPU integration vs integration + save.
            sf << "compute_ms " << compute_ms << "\n";
            sf << "save_ms " << save_ms << "\n";
            sf << "total_ms " << total_ms << "\n";
            sf << "compute_ms_per_step " << (compute_ms / timesteps_) << "\n";
            sf << "cells_per_second_M " << ((static_cast<double>(nx_) * ny_ * timesteps_) / (compute_ms / 1000.0) / 1e6) << "\n";
        }
        std::cout << "Summary: max_eta_max=" << max_eta_max << std::endl;
    }
}

void SingleLineGridSimulationCUDA::SyncToHost() {
    if (num_subgrids_ == 1) CopyFromGPU();
}

void SingleLineGridSimulationCUDA::CollectResults(int timestep) {
    if (num_subgrids_ == 1) CopyFromGPU();
    SaveDataToFile("eta", global_grid_, timestep);
    if (save_velocities_) {
        SaveDataToFile("u", global_grid_, timestep);
        SaveDataToFile("v", global_grid_, timestep);
    }
}

void SingleLineGridSimulationCUDA::SetSaveInterval(int interval) { save_interval_ = interval; }
int SingleLineGridSimulationCUDA::GetTimesteps() const { return timesteps_; }

void SingleLineGridSimulationCUDA::SaveDataToFile(const std::string& var, const GridCUDA& grid, int timestep) {
    fs::create_directories(output_folder_);
    std::string filename = output_folder_ + "/" + var + "_timestep_" + std::to_string(timestep) + ".txt";
    std::ofstream file(filename);
    if (!file) { std::cerr << "Cannot open: " << filename << std::endl; return; }
    
    const double* data = nullptr;
    if (var == "H") data = grid.GetH();
    else if (var == "u") data = grid.GetU();
    else if (var == "v") data = grid.GetV();
    else if (var == "eta") data = grid.GetEta();
    else if (var == "D") data = grid.GetD();
    else if (var == "eta_max") data = grid.GetEtaMax();
    else if (var == "u_max") data = grid.GetUMax();
    else if (var == "v_max") data = grid.GetVMax();
    else { std::cerr << "Unknown: " << var << std::endl; return; }
    
    int nx = grid.GetNx(), ny = grid.GetNy();
    file << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) file << data[IDX(i, j, ny)] << " ";
        file << "\n";
    }
}

// ================== Test Scenarios ==================

void SingleLineGridSimulationCUDA::SetupTest1() {
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            global_grid_.D(i, j) = 1000.0;
            global_grid_.H(i, j) = global_grid_.D(i, j);
            global_grid_.eta(i, j) = 0.0;
            global_grid_.u(i, j) = 0.0;
            global_grid_.v(i, j) = 0.0;
        }
    }
    for (int i = 45; i < 55 && i < nx_; ++i) {
        for (int j = 45; j < 55 && j < ny_; ++j) {
            global_grid_.eta(i, j) = 1.0;
            global_grid_.H(i, j) = global_grid_.D(i, j) + global_grid_.eta(i, j);
        }
    }
    global_grid_.SetInitialized(true);
    std::cout << "Test 1: Square pulse at (45-55, 45-55)" << std::endl;
}

void SingleLineGridSimulationCUDA::SetupTest2() {
    double x0 = 500000.0, y0 = 500000.0;
    double radius = delta_x_const_ * 50.0;
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            global_grid_.D(i, j) = 1000.0;
            double x = i * delta_x_const_, y = j * delta_y_;
            double dist = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));
            global_grid_.eta(i, j) = (dist > radius) ? 0.0 : 0.5 * (1.0 + std::cos(M_PI * dist / radius));
            global_grid_.H(i, j) = global_grid_.D(i, j) + global_grid_.eta(i, j);
            global_grid_.u(i, j) = 0.0; global_grid_.v(i, j) = 0.0;
        }
    }
    global_grid_.SetInitialized(true);
    std::cout << "Test 2: Cosine pulse" << std::endl;
}

void SingleLineGridSimulationCUDA::SetupTest3(double tau) {
    double T = 300.0, k = 1e-8;
    double waveHeight = 0.5 * (1.0 - std::cos(0.0 * tau * 2.0 * M_PI / T));
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            global_grid_.D(i, j) = k * std::pow(delta_y_ * (ny_ - j), 2) + 10.0;
            global_grid_.eta(i, j) = waveHeight;
            global_grid_.H(i, j) = global_grid_.D(i, j) + global_grid_.eta(i, j);
            global_grid_.u(i, j) = 0.0; global_grid_.v(i, j) = 0.0;
        }
    }
    global_grid_.SetInitialized(true);
    std::cout << "Test 3: Variable depth" << std::endl;
}

void SingleLineGridSimulationCUDA::SetupTest4(double tau) {
    double T = 300.0;
    double waveHeight = 0.5 * (1.0 - std::cos(0.0 * tau * 2.0 * M_PI / T));
    int ax = nx_ / 2, ay = ny_ - 1, bx = nx_ - 1, by = ny_ / 2, cx = nx_ - 1, cy = ny_ - 1;
    auto sign = [](double x1, double y1, double x2, double y2, double x3, double y3) {
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
    };
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            double d1 = sign(i, j, ax, ay, bx, by);
            double d2 = sign(i, j, bx, by, cx, cy);
            double d3 = sign(i, j, cx, cy, ax, ay);
            bool inside = !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
            if (inside) {
                global_grid_.D(i, j) = 0.0; global_grid_.H(i, j) = 0.0;
                global_grid_.u(i, j) = 0.0; global_grid_.v(i, j) = 0.0; global_grid_.eta(i, j) = 0.0;
            } else {
                global_grid_.D(i, j) = 1000.0; global_grid_.eta(i, j) = waveHeight;
                global_grid_.H(i, j) = global_grid_.D(i, j) + global_grid_.eta(i, j);
                global_grid_.u(i, j) = 0.0; global_grid_.v(i, j) = 0.0;
            }
        }
    }
    global_grid_.SetInitialized(true);
    std::cout << "Test 4: Triangle island" << std::endl;
}

void SingleLineGridSimulationCUDA::SetupTest5(double tau) {
    double T = 300.0;
    double waveHeight = 0.5 * (1.0 - std::cos(0.0 * tau * 2.0 * M_PI / T));
    int centerX = nx_ / 2, centerY = ny_ / 2, radius = 100;
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            int dx = i - centerX, dy = j - centerY;
            global_grid_.D(i, j) = (dx*dx + dy*dy <= radius*radius) ? 0.0 : 1000.0;
            global_grid_.eta(i, j) = waveHeight;
            global_grid_.H(i, j) = global_grid_.D(i, j) + global_grid_.eta(i, j);
            global_grid_.u(i, j) = 0.0; global_grid_.v(i, j) = 0.0;
        }
    }
    global_grid_.SetInitialized(true);
    std::cout << "Test 5: Circular island" << std::endl;
}

#undef IDX
