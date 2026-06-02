// SubGrid.cpp - Оптимизированная версия с OpenMP + SIMD
// Требует /openmp:experimental в MSVC для simd
// Индексация: i * ny + j (совместимо с Grid)

#include "SubGrid.h"
#include "FVSolver.h"          // Phase A FV Godunov backend
#include "FVSolverConfig.h"    // env-driven dispatcher
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Макрос для индексации (i * ny + j) - совместим с Grid
#define IDX(i, j, ny) ((i) * (ny) + (j))

SubGrid::SubGrid(int process_id, int global_i_start, int global_i_end, Grid& global_grid,
    double g, double tau, std::function<double(int, int)> delta_x, double delta_y, double min_calc_value,
    std::shared_ptr<std::barrier<>>& barrier_half_step,
    std::shared_ptr<std::barrier<>>& barrier_full_step,
    int spec_step,
    CoordSystem coord_system,
    double center_latitude_rad,
    DecompositionDirection decomp_dir)
    : process_id_(process_id),
    global_i_start_(global_i_start),
    global_i_end_(global_i_end),
    global_grid_(global_grid),
    g_(g),
    tau_(tau),
    delta_x_(delta_x),
    delta_y_(delta_y),
    min_calc_value(min_calc_value),
    coord_system_(coord_system),
    center_latitude_rad_(center_latitude_rad),
    decomp_dir_(decomp_dir),
    neighbor_left_(nullptr),
    neighbor_right_(nullptr),
    barrier_half_step_(barrier_half_step),
    barrier_full_step_(barrier_full_step),
    local_grid_(0, 0),
    spec_step_(spec_step)
{
    // Размеры локальной подрешётки с граничными слоями
    if (decomp_dir_ == DecompositionDirection::Horizontal) {
        // Разбивка по X: ghost cells по X, полный диапазон Y
        local_nx_ = (global_i_end_ - global_i_start_) + 2;
        local_ny_ = global_grid_.GetNy();
    } else {
        // Разбивка по Y: ghost cells по ОБЕИМ осям
        // По X: +2 ghost columns (domain edges), по Y: +2 ghost rows (strip boundaries)
        // Это обеспечивает одинаковый stencil с горизонтальной разбивкой
        local_nx_ = global_grid_.GetNx() + 2;
        local_ny_ = (global_i_end_ - global_i_start_) + 2;
    }

    inv_dy_ = (delta_y_ > 0.0) ? 1.0 / delta_y_ : 0.0;

    // Precompute inv_dx array per j to avoid std::function calls in inner SIMD loop
    // delta_x depends only on j (latitude), not on i
    inv_dx_arr_ = std::make_unique<Real[]>(local_ny_);
    for (int j = 0; j < local_ny_; ++j) {
        double dx = delta_x_(0, j);
        inv_dx_arr_[j] = (dx > 0.0) ? 1.0 / dx : 0.0;
    }

    // Инициализация локальной се��ки
    local_grid_ = Grid(local_nx_, local_ny_);

    // Выделение памяти для hat-массивов
    const size_t size = static_cast<size_t>(local_nx_) * local_ny_;
    H_hat_ = std::make_unique<Real[]>(size);
    u_hat_ = std::make_unique<Real[]>(size);
    v_hat_ = std::make_unique<Real[]>(size);
    eta_hat_ = std::make_unique<Real[]>(size);

#ifdef SAVE_ETA_MAX_MAP
    eta_max_ = std::make_unique<Real[]>(size);
    u_max_ = std::make_unique<Real[]>(size);
    v_max_ = std::make_unique<Real[]>(size);
    std::memset(eta_max_.get(), 0, size * sizeof(Real));
    std::memset(u_max_.get(), 0, size * sizeof(Real));
    std::memset(v_max_.get(), 0, size * sizeof(Real));
#endif
}

void SubGrid::Initialize() {
    if (decomp_dir_ == DecompositionDirection::Horizontal) {
        // Horizontal: ghost cells по X, полный Y
        for (int i = 0; i < local_nx_; ++i) {
            int global_i = global_i_start_ - 1 + i;
            for (int j = 0; j < local_ny_; ++j) {
                if (global_i >= 0 && global_i < global_grid_.GetNx()) {
                    local_grid_.H(i, j) = global_grid_.H(global_i, j);
                    local_grid_.u(i, j) = global_grid_.u(global_i, j);
                    local_grid_.v(i, j) = global_grid_.v(global_i, j);
                    local_grid_.eta(i, j) = global_grid_.eta(global_i, j);
                    local_grid_.D(i, j) = global_grid_.D(global_i, j);
                } else {
                    local_grid_.H(i, j) = 0.0;
                    local_grid_.u(i, j) = 0.0;
                    local_grid_.v(i, j) = 0.0;
                    local_grid_.eta(i, j) = 0.0;
                    local_grid_.D(i, j) = 0.0;
                }
            }
        }
#ifdef SAVE_ETA_MAX_MAP
        // Checkpoint resume: also sync eta_max/u_max/v_max accumulators from global.
        // Otherwise local buffers stay zero (constructor memset) and after resume
        // the accumulator restarts from 0 instead of continuing from previous segment.
        // Codex robustness note: explicit zero for ghost cells in case Initialize
        // is ever called twice (matches H/u/v/eta/D pattern).
        for (int i = 0; i < local_nx_; ++i) {
            int global_i = global_i_start_ - 1 + i;
            for (int j = 0; j < local_ny_; ++j) {
                if (global_i >= 0 && global_i < global_grid_.GetNx()) {
                    eta_max_[IDX(i, j, local_ny_)] = global_grid_.eta_max(global_i, j);
                    u_max_[IDX(i, j, local_ny_)]   = global_grid_.u_max(global_i, j);
                    v_max_[IDX(i, j, local_ny_)]   = global_grid_.v_max(global_i, j);
                } else {
                    eta_max_[IDX(i, j, local_ny_)] = 0;
                    u_max_[IDX(i, j, local_ny_)]   = 0;
                    v_max_[IDX(i, j, local_ny_)]   = 0;
                }
            }
        }
#endif
    } else {
        // Vertical: ghost cells по обеим осям
        // local i=0 и i=local_nx-1 → X-ghost (domain edges)
        // local j=0 и j=local_ny-1 → Y-ghost (strip boundaries)
        int global_nx = global_grid_.GetNx();
        int global_ny = global_grid_.GetNy();
        for (int i = 0; i < local_nx_; ++i) {
            int global_i = i - 1; // offset: local i=1 → global i=0
            for (int j = 0; j < local_ny_; ++j) {
                int global_j = global_i_start_ - 1 + j;
                if (global_i >= 0 && global_i < global_nx &&
                    global_j >= 0 && global_j < global_ny) {
                    local_grid_.H(i, j) = global_grid_.H(global_i, global_j);
                    local_grid_.u(i, j) = global_grid_.u(global_i, global_j);
                    local_grid_.v(i, j) = global_grid_.v(global_i, global_j);
                    local_grid_.eta(i, j) = global_grid_.eta(global_i, global_j);
                    local_grid_.D(i, j) = global_grid_.D(global_i, global_j);
                } else {
                    local_grid_.H(i, j) = 0.0;
                    local_grid_.u(i, j) = 0.0;
                    local_grid_.v(i, j) = 0.0;
                    local_grid_.eta(i, j) = 0.0;
                    local_grid_.D(i, j) = 0.0;
                }
            }
        }
#ifdef SAVE_ETA_MAX_MAP
        // Checkpoint resume: sync eta_max/u_max/v_max from global (vertical decomp).
        // Codex robustness: explicit zero for ghost cells.
        for (int i = 0; i < local_nx_; ++i) {
            int global_i = i - 1;
            for (int j = 0; j < local_ny_; ++j) {
                int global_j = global_i_start_ - 1 + j;
                if (global_i >= 0 && global_i < global_nx &&
                    global_j >= 0 && global_j < global_ny) {
                    eta_max_[IDX(i, j, local_ny_)] = global_grid_.eta_max(global_i, global_j);
                    u_max_[IDX(i, j, local_ny_)]   = global_grid_.u_max(global_i, global_j);
                    v_max_[IDX(i, j, local_ny_)]   = global_grid_.v_max(global_i, global_j);
                } else {
                    eta_max_[IDX(i, j, local_ny_)] = 0;
                    u_max_[IDX(i, j, local_ny_)]   = 0;
                    v_max_[IDX(i, j, local_ny_)]   = 0;
                }
            }
        }
#endif
    }

    // Phase E P1 (Codex 2026-05-28): mirror main -> hat at init so an inactive
    // subgrid (in Active Region Geometric/Hybrid mode) that never runs a
    // predictor stage still exposes a CORRECT hat state to neighbors copying
    // ghost rows from it. Otherwise the zero-initialised hat (H_hat=0 instead
    // of H=D) reads as a spurious dry wall on the neighbor side and breaks
    // wave propagation between active and inactive stripes (×2.3 amplitude
    // divergence observed on Monai wavemaker scenario).
    {
        const size_t N = static_cast<size_t>(local_nx_) * local_ny_;
        const Real* H_main   = local_grid_.GetH();
        const Real* u_main   = local_grid_.GetU();
        const Real* v_main   = local_grid_.GetV();
        const Real* eta_main = local_grid_.GetEta();
        std::memcpy(H_hat_.get(),   H_main,   N * sizeof(Real));
        std::memcpy(u_hat_.get(),   u_main,   N * sizeof(Real));
        std::memcpy(v_hat_.get(),   v_main,   N * sizeof(Real));
        std::memcpy(eta_hat_.get(), eta_main, N * sizeof(Real));
    }
}

// Destructor defaulted here (where fv::FVSolver is complete) so the
// forward-declared unique_ptr<FVSolver> member can be destroyed.
SubGrid::~SubGrid() = default;

void SubGrid::SetNeighbors(SubGrid* neighbor_up, SubGrid* neighbor_down) {
    neighbor_left_ = neighbor_up;
    neighbor_right_ = neighbor_down;
}

void SubGrid::SetSpecStep(int spec_step) {
    spec_step_ = spec_step;
}

// ============================================================================
// Диспетчерские методы - выбор реализации по системе координат
// ============================================================================

static bool spherical_logged_ = false;
// (ComputeHalfStep moved below the anonymous namespace so it can reference absorb_config().)

// ============================================================================
// Davis 1987 / Liang 2007 TVD post-corrector helpers.
//
// Applied to conservative variables U = (H, Hu, Hv)^T after standard MacCormack
// corrector to suppress Gibbs oscillations near sharp gradients (coastal
// shoaling, wet/dry edges). Conditional via env TVD_ENABLED=1.
//
// Math reference: project memory davis-tvd-formula-may24.
//   B(nu, r) = 0.5 * nu * (1 - nu) * (1 - phi(r))      [single-face form]
//   U^{n+1} = U^{MC} + (B_{+1/2} * Delta_{+1/2} - B_{-1/2} * Delta_{-1/2})
//   Default limiter: van Leer.  Near wet/dry: minmod (more diffusive).
// ============================================================================
namespace {

inline double tvd_phi_van_leer(double r) {
    if (r <= 0.0) return 0.0;
    return 2.0 * r / (1.0 + r);
}

inline double tvd_phi_minmod(double r) {
    if (r <= 0.0) return 0.0;
    return std::min(1.0, r);
}

// Safe ratio: returns 0 if denominator near zero (treat as extremum -> no high-order correction)
inline double tvd_ratio(double d_up, double d_down) {
    constexpr double eps = 1e-12;
    if (std::abs(d_down) < eps) return 0.0;
    return d_up / d_down;
}

// Coefficient B for post-correction term. nu = local Courant number, r = gradient ratio.
//
// CRITICAL: (1 - phi) formulation requires phi in [0, 1] otherwise the sign of
// the correction flips and the scheme becomes anti-diffusive (blow-up).
//   minmod:  phi in [0, 1]                       — safe
//   van Leer: phi in [0, 2] (2r/(1+r) for r>1)   — must clamp to 1
//   superbee: phi in [0, 2]                      — must clamp to 1
// We clamp phi to [0, 1] so the post-correction is always dissipative
// (never anti-diffusive).
inline double tvd_B(double nu, double r, bool use_minmod) {
    double phi = use_minmod ? tvd_phi_minmod(r) : tvd_phi_van_leer(r);
    if (phi < 0.0) phi = 0.0;
    if (phi > 1.0) phi = 1.0;      // critical clamp: prevent (1-phi) sign flip
    // Clamp nu to [0, 1] so the (1-nu) factor never becomes negative.
    double nuc = nu;
    if (nuc < 0.0) nuc = 0.0;
    if (nuc > 1.0) nuc = 1.0;
    return 0.5 * nuc * (1.0 - nuc) * (1.0 - phi);
}

// Cache TVD config from env on first access.
struct TvdConfig {
    bool enabled = false;
    // If true, use minmod everywhere; otherwise van Leer (with auto-switch to minmod near shallow cells).
    bool force_minmod = false;
    // Shallow-cell threshold (m) for auto-switching to minmod near wet/dry.
    double shallow_depth = 100.0;
    // Skip TVD in cells within this many indices from physical boundaries (north/south).
    // Codex Phase 2b diagnostic: 0 = TVD everywhere, 20 = guard zone of 20 cells near j=0/ny-1.
    int boundary_skip = 0;
    // Apply TVD every Nth step (1 = every step). Codex Phase 2c: reduce cumulative
    // damping by lowering TVD frequency without changing magnitude per call.
    int apply_every = 1;
};

// North absorbing layer config (Codex spec 2026-05-27).
struct AbsorbConfig {
    bool enabled = false;
    int  layer_width = 96;       // cells from j=0 northern boundary
    double alpha_max = 0.05;     // = sigma_max * dt (dimensionless)
};

inline const AbsorbConfig& absorb_config() {
    static AbsorbConfig cfg = [](){
        AbsorbConfig c;
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "ABSORB_ENABLED");
        c.enabled = (buf && *buf == '1');
        if (buf) free(buf);
        char* buf2 = nullptr; size_t sz2 = 0;
        _dupenv_s(&buf2, &sz2, "ABSORB_LAYER_WIDTH");
        if (buf2) { c.layer_width = std::atoi(buf2); free(buf2); }
        char* buf3 = nullptr; size_t sz3 = 0;
        _dupenv_s(&buf3, &sz3, "ABSORB_ALPHA_MAX");
        if (buf3) { c.alpha_max = std::atof(buf3); free(buf3); }
        if (c.enabled) {
            std::cerr << "[ABSORB] North layer ENABLED: width=" << c.layer_width
                      << " cells, alpha_max=" << c.alpha_max
                      << " (= sigma_max*dt; ~" << (c.alpha_max * 100) << "% damping/step at j=0)" << std::endl;
        }
        return c;
    }();
    return cfg;
}

// damping factor (1 - alpha(j)) where alpha(j) = alpha_max * cos²(pi/2 * j/(W-1))
inline double absorb_factor_north(int j, const AbsorbConfig& cfg) {
    if (!cfg.enabled || j >= cfg.layer_width || j < 0) return 1.0;
    const int W = cfg.layer_width;
    const double x = (W > 1) ? double(j) / double(W - 1) : 0.0;
    const double c = std::cos(0.5 * 3.14159265358979323846 * x);
    const double alpha = cfg.alpha_max * c * c;
    return 1.0 - alpha;
}

inline const TvdConfig& tvd_config() {
    static TvdConfig cfg = [](){
        TvdConfig c;
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "TVD_ENABLED");
        c.enabled = (buf && *buf == '1');
        if (buf) free(buf);
        char* buf2 = nullptr; size_t sz2 = 0;
        _dupenv_s(&buf2, &sz2, "TVD_LIMITER");
        // "minmod" => force_minmod=true; default ("vanleer" or unset) => false
        c.force_minmod = (buf2 && (std::strcmp(buf2, "minmod") == 0));
        if (buf2) free(buf2);
        char* buf3 = nullptr; size_t sz3 = 0;
        _dupenv_s(&buf3, &sz3, "TVD_SHALLOW_DEPTH");
        if (buf3) { c.shallow_depth = std::atof(buf3); free(buf3); }
        char* buf4 = nullptr; size_t sz4 = 0;
        _dupenv_s(&buf4, &sz4, "TVD_BOUNDARY_SKIP");
        if (buf4) { c.boundary_skip = std::atoi(buf4); free(buf4); }
        char* buf5 = nullptr; size_t sz5 = 0;
        _dupenv_s(&buf5, &sz5, "TVD_APPLY_EVERY");
        if (buf5) {
            int n = std::atoi(buf5);
            c.apply_every = (n > 0) ? n : 1;
            free(buf5);
        }
        if (c.enabled) {
            std::cerr << "[TVD] post-corrector ENABLED, limiter="
                      << (c.force_minmod ? "minmod (forced)" : "van Leer + minmod-near-shallow")
                      << ", shallow_depth=" << c.shallow_depth << " m"
                      << ", boundary_skip=" << c.boundary_skip << " cells"
                      << ", apply_every=" << c.apply_every << " step(s)" << std::endl;
        }
        return c;
    }();
    return cfg;
}

// Wet/dry positivity floor (Codex Phase 8 Monai fix). Default OFF — keeps
// existing scenarios bitwise identical. ON for shallow-water benchmarks where
// MacCormack wet/dry interface produces NaN/inf without positivity clamping.
//
// MC_WET_DRY_POSITIVITY=1 changes:
//   * dry-cell H_hat = 0  (instead of D, which is negative for land)
//   * predictor/corrector H_new: if NaN/inf or < grnd → clamp to grnd,
//     eta = grnd - D (not 0); preserves bathymetry awareness without unphysical
//     return-to-still-water impulse.
//   * velocity Froude clamp: |u|,|v| <= cfl_factor * sqrt(g*H_local). Prevents
//     supercritical runaway at near-dry cells where g·∂η/∂x blows up.
//   * Manning friction for shallow cells: tau_friction = g·n²·|u|·u / H^(4/3),
//     applied after compute. n=0.025 default (smooth lab tank).
struct WetDryConfig {
    bool enabled = false;
    double froude_max = 0.95;      // |u|/sqrt(gH) clamp (subcritical safety)
    double manning_n  = 0.025;     // Manning friction coeff (smooth tank)
    double shallow_for_friction = 0.05;  // apply friction only if H < this (m)
};

// Audusse 2004 hydrostatic reconstruction (Codex Phase 8 design).
// DOI: 10.1137/S1064827503431090. Well-balanced + positivity-preserving.
//
// For an edge between cells L and R with depths D_L, D_R (D > 0 = water depth)
// and total heights H_L, H_R:
//   D_e   = min(D_L, D_R)              // raised shared depth
//   h_L_e = max(0, H_L - max(0, D_L - D_e))
//   h_R_e = max(0, H_R - max(0, D_R - D_e))
//   eta_L_e = h_L_e - D_e   // CRITICAL: -D_e, not -D_L (well-balancedness)
//   eta_R_e = h_R_e - D_e
//
// Lake-at-rest invariant: eta_L = eta_R = eta0 → eta_L_e = eta_R_e = eta0 →
// pressure gradient = 0 regardless of bathymetry step.
struct HydrostaticEdge { Real hL; Real hR; Real etaL; Real etaR; };

inline HydrostaticEdge HydroEdge(Real HL, Real DL, Real HR, Real DR) {
    const Real De = std::min(DL, DR);
    const Real hL = std::max(Real(0), HL - std::max(Real(0), DL - De));
    const Real hR = std::max(Real(0), HR - std::max(Real(0), DR - De));
    return {hL, hR, hL - De, hR - De};
}

inline bool hydrostatic_reconstruction_enabled() {
    static bool enabled = [](){
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "MC_HYDROSTATIC_RECONSTRUCTION");
        const bool v = (buf && *buf == '1');
        if (buf) free(buf);
        if (v) std::cerr << "[WetDry] hydrostatic reconstruction (Audusse) ENABLED"
                         << std::endl;
        return v;
    }();
    return enabled;
}

inline const WetDryConfig& wet_dry_config() {
    static WetDryConfig cfg = [](){
        WetDryConfig c;
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "MC_WET_DRY_POSITIVITY");
        c.enabled = (buf && *buf == '1');
        if (buf) free(buf);

        char* buf2 = nullptr; size_t sz2 = 0;
        _dupenv_s(&buf2, &sz2, "MC_WET_DRY_FROUDE_MAX");
        if (buf2) { c.froude_max = std::atof(buf2); free(buf2); }

        char* buf3 = nullptr; size_t sz3 = 0;
        _dupenv_s(&buf3, &sz3, "MC_WET_DRY_MANNING_N");
        if (buf3) { c.manning_n = std::atof(buf3); free(buf3); }

        char* buf4 = nullptr; size_t sz4 = 0;
        _dupenv_s(&buf4, &sz4, "MC_WET_DRY_SHALLOW_DEPTH");
        if (buf4) { c.shallow_for_friction = std::atof(buf4); free(buf4); }

        if (c.enabled) {
            std::cerr << "[WetDry] positivity floor ENABLED"
                      << " froude_max=" << c.froude_max
                      << " manning_n=" << c.manning_n
                      << " shallow=" << c.shallow_for_friction << "m"
                      << std::endl;
        }
        return c;
    }();
    return cfg;
}

} // namespace

void SubGrid::ComputeHalfStep() {
    // Phase A: FV Godunov dispatcher hook. When MC_SOLVER=fv_godunov env is
    // set, route to FVSolver and skip all legacy MacCormack / TVD / Froude /
    // Manning / absorbing layers (those are MacCormack-specific stabilizers).
    if (fv::config().fv_enabled()) {
        if (coord_system_ == CoordSystem::Spherical) {
            throw std::runtime_error(
                "FV Godunov backend does not support Spherical coords yet "
                "(Phase F). Use MC_SOLVER=mc or Cartesian scenario.");
        }
        if (!fv_solver_) {
            fv_solver_ = std::make_unique<fv::FVSolver>(this);
        }
        fv_solver_->Stage1Predictor();
        return;
    }

    // Legacy MacCormack path (default).
    if (coord_system_ == CoordSystem::Spherical) {
        if (!spherical_logged_) {
            std::cerr << "[DEBUG] Using SPHERICAL coordinate system (center_lat="
                      << center_latitude_rad_ << " rad)" << std::endl;
            spherical_logged_ = true;
        }
        ComputeHalfStepSpherical();
    } else {
        ComputeHalfStepCartesian();
    }
    // North absorbing layer applied to predicted state (Codex spec, Rayleigh damping).
    if (absorb_config().enabled) {
        ApplyNorthAbsorbingHat();
    }
}

void SubGrid::ComputeFullStep(int global_t, bool use_global_tvd_schedule) {
    // Phase A: FV Godunov dispatcher hook.
    if (fv::config().fv_enabled()) {
        // FVSolver lazily-constructed in Stage1Predictor; safe to use here.
        if (fv_solver_) {
            fv_solver_->Stage2Corrector();
        }
        return;
    }

    // Legacy MacCormack path (default).
    if (coord_system_ == CoordSystem::Spherical) {
        ComputeFullStepSpherical();
    } else {
        ComputeFullStepCartesian();
    }
    // North absorbing layer (Rayleigh damping on velocity only) — Codex spec 2026-05-27.
    if (absorb_config().enabled) {
        ApplyNorthAbsorbingMain();
    }
    // Davis 1987 / Liang 2007 TVD post-correction. Conditional via env TVD_ENABLED=1.
    if (tvd_config().enabled) {
        ApplyTvdPostCorrection(global_t, use_global_tvd_schedule);
    }
}

// Apply Rayleigh damping in the northern absorbing layer to main (corrector) state.
// j local == j global for horizontal decomposition (Pacific case).
void SubGrid::ApplyNorthAbsorbingMain() {
    const AbsorbConfig& cfg = absorb_config();
    if (!cfg.enabled) return;
    const int nx = local_nx_;
    const int W = cfg.layer_width;
    Real* __restrict u_data = local_grid_.GetU();
    Real* __restrict v_data = local_grid_.GetV();

    const int j_end = (W < local_ny_) ? W : local_ny_;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < j_end; ++j) {
        const double factor = absorb_factor_north(j, cfg);
        for (int i = 0; i < nx; ++i) {
            const int ij = IDX(i, j, local_ny_);
            u_data[ij] = (Real)(double(u_data[ij]) * factor);
            v_data[ij] = (Real)(double(v_data[ij]) * factor);
        }
    }
}

void SubGrid::ApplyNorthAbsorbingHat() {
    const AbsorbConfig& cfg = absorb_config();
    if (!cfg.enabled) return;
    const int nx = local_nx_;
    const int W = cfg.layer_width;
    Real* __restrict u_hat_data = u_hat_.get();
    Real* __restrict v_hat_data = v_hat_.get();

    const int j_end = (W < local_ny_) ? W : local_ny_;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < j_end; ++j) {
        const double factor = absorb_factor_north(j, cfg);
        for (int i = 0; i < nx; ++i) {
            const int ij = IDX(i, j, local_ny_);
            u_hat_data[ij] = (Real)(double(u_hat_data[ij]) * factor);
            v_hat_data[ij] = (Real)(double(v_hat_data[ij]) * factor);
        }
    }
}

// ============================================================================
// Реализация для СФЕРИЧЕСКОЙ системы координат с силой Кориолиса
// КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: обрабатываются границы j=0 и j=ny-1!
// ============================================================================

void SubGrid::ComputeHalfStepSpherical() {
    const int nx = local_nx_;
    const int ny = local_ny_;
    const double tau = tau_;
    const double g = g_;
    const double inv_dy = inv_dy_;
    const Real grnd = min_calc_value;

    // Константы для сферической системы
    constexpr double R_EARTH = 6371000.0;
    constexpr double OMEGA = 7.2921e-5;

    const double PHI_0 = center_latitude_rad_;
    const double delta_y = 1.0 / inv_dy;
    const double delta_phi = delta_y / R_EARTH;

    Real* __restrict H_data = local_grid_.GetH();
    Real* __restrict u_data = local_grid_.GetU();
    Real* __restrict v_data = local_grid_.GetV();
    Real* __restrict eta_data = local_grid_.GetEta();
    Real* __restrict D_data = local_grid_.GetD();
    Real* __restrict H_hat_data = H_hat_.get();
    Real* __restrict u_hat_data = u_hat_.get();
    Real* __restrict v_hat_data = v_hat_.get();
    Real* __restrict eta_hat_data = eta_hat_.get();

    // ОСНОВНОЙ ЦИКЛ: внутренние точки (1 < j < ny-1)
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < nx - 1; ++i) {
        #pragma omp simd
        for (int j = 1; j < ny - 1; ++j) {
            const int ij = IDX(i, j, ny);
            const int imj = IDX(i - 1, j, ny);
            const int ijm = IDX(i, j - 1, ny);
            const int ipj = IDX(i + 1, j, ny);
            const int ijp = IDX(i, j + 1, ny);

            const Real Dij = D_data[ij];

            if (Dij < grnd) {
                H_hat_data[ij] = Dij;
                u_hat_data[ij] = 0.0;
                v_hat_data[ij] = 0.0;
                eta_hat_data[ij] = 0.0;
                continue;
            }

            // Широта и Кориолис
            const double phi = PHI_0 + (j - ny / 2.0) * delta_phi;
            const double f = coriolis_enabled_ ? (2.0 * OMEGA * std::sin(phi)) : 0.0;

            const double dx = delta_x_(i, j);
            const Real inv_dx = 1.0 / dx;

            const Real eij = eta_data[ij];
            const Real eimj = eta_data[imj];
            const Real eijm = eta_data[ijm];

            const Real uij = u_data[ij];
            const Real uimj = u_data[imj];
            const Real uijm = u_data[ijm];

            const Real vij = v_data[ij];
            const Real vimj = v_data[imj];
            const Real vijm = v_data[ijm];

            const Real Hij = H_data[ij];
            const Real Himj = H_data[imj];
            const Real Hijm = H_data[ijm];

            const Real Dimj = D_data[imj];
            const Real Dijm = D_data[ijm];

            const Real Hu_ij = Hij * uij;
            const Real Hu_imj = Himj * uimj;

            const Real Hv_ij = Hij * vij;
            const Real Hv_ijm = Hijm * vijm;

            const Real flux_H_x = (Hu_ij - Hu_imj) * inv_dx;
            const Real flux_H_y = (Hv_ij - Hv_ijm) * inv_dy;

            H_hat_data[ij] = Hij - tau * (flux_H_x + flux_H_y);

            const Real du_dx = (uij - uimj) * inv_dx;
            const Real du_dy = (uij - uijm) * inv_dy;

            const Real dv_dx = (vij - vimj) * inv_dx;
            const Real dv_dy = (vij - vijm) * inv_dy;

            const Real deta_dx = (eij - eimj) * inv_dx;
            const Real deta_dy = (eij - eijm) * inv_dy;

            // Уравнения с Кориолисом
            Real u_hat_new = uij - tau * (
                uij * du_dx +
                vij * du_dy +
                g * deta_dx -
                f * vij
            );

            Real v_hat_new = vij - tau * (
                uij * dv_dx +
                vij * dv_dy +
                g * deta_dy +
                f * uij
            );

            if (u_hat_new != 0.0) {
                if ((i + 1 < nx - 1 && D_data[ipj] <= grnd) ||
                    (i - 1 > 0 && Dimj <= grnd)) {
                    u_hat_new = 0.0;
                }
            }

            if (v_hat_new != 0.0) {
                if ((j + 1 < ny - 1 && D_data[ijp] <= grnd) ||
                    (j - 1 > 0 && Dijm <= grnd)) {
                    v_hat_new = 0.0;
                }
            }

            u_hat_data[ij] = u_hat_new;
            v_hat_data[ij] = v_hat_new;
            eta_hat_data[ij] = H_hat_data[ij] - Dij;
        }
    }

    // КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обработка нижней границы j=0 с ОДНОСТОРОННИМИ разностями
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < nx - 1; ++i) {
        const int j = 0;
        const int ij = IDX(i, j, ny);
        const int imj = IDX(i - 1, j, ny);
        const int ijp = IDX(i, j + 1, ny);  // Односторонняя разность
        const int ipj = IDX(i + 1, j, ny);

        const Real Dij = D_data[ij];

        if (Dij < grnd) {
            H_hat_data[ij] = Dij;
            u_hat_data[ij] = 0.0;
            v_hat_data[ij] = 0.0;
            eta_hat_data[ij] = 0.0;
        } else {
            const double phi = PHI_0 + (j - ny / 2.0) * delta_phi;
            const double f = coriolis_enabled_ ? (2.0 * OMEGA * std::sin(phi)) : 0.0;

            const double dx = delta_x_(i, j);
            const Real inv_dx = 1.0 / dx;

            const Real eij = eta_data[ij];
            const Real eimj = eta_data[imj];
            const double eijp = eta_data[ijp];  // Вперед вместо назад

            const Real uij = u_data[ij];
            const Real uimj = u_data[imj];
            const double uijp = u_data[ijp];

            const Real vij = v_data[ij];
            const Real vimj = v_data[imj];
            const double vijp = v_data[ijp];

            const Real Hij = H_data[ij];
            const Real Himj = H_data[imj];
            const double Hijp = H_data[ijp];

            const Real Dimj = D_data[imj];

            const Real Hu_ij = Hij * uij;
            const Real Hu_imj = Himj * uimj;

            const Real Hv_ij = Hij * vij;
            const double Hv_ijp = Hijp * vijp;

            const Real flux_H_x = (Hu_ij - Hu_imj) * inv_dx;
            const Real flux_H_y = (Hv_ijp - Hv_ij) * inv_dy;  // Односторонняя

            H_hat_data[ij] = Hij - tau * (flux_H_x + flux_H_y);

            const Real du_dx = (uij - uimj) * inv_dx;
            const Real du_dy = (uijp - uij) * inv_dy;  // Односторонняя вперед

            const Real dv_dx = (vij - vimj) * inv_dx;
            const Real dv_dy = (vijp - vij) * inv_dy;  // Односторонняя вперед

            const Real deta_dx = (eij - eimj) * inv_dx;
            const Real deta_dy = (eijp - eij) * inv_dy;  // Односторонняя вперед

            u_hat_data[ij] = uij - tau * (uij * du_dx + vij * du_dy + g * deta_dx - f * vij);
            v_hat_data[ij] = vij - tau * (uij * dv_dx + vij * dv_dy + g * deta_dy + f * uij);
            eta_hat_data[ij] = H_hat_data[ij] - Dij;
        }
    }

    // КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обработка верхней границы j=ny-1 с ОДНОСТОРОННИМИ разностями
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < nx - 1; ++i) {
        const int j = ny - 1;
        const int ij = IDX(i, j, ny);
        const int imj = IDX(i - 1, j, ny);
        const int ijm = IDX(i, j - 1, ny);  // Односторонняя разность
        const int ipj = IDX(i + 1, j, ny);

        const Real Dij = D_data[ij];

        if (Dij < grnd) {
            H_hat_data[ij] = Dij;
            u_hat_data[ij] = 0.0;
            v_hat_data[ij] = 0.0;
            eta_hat_data[ij] = 0.0;
        } else {
            const double phi = PHI_0 + (j - ny / 2.0) * delta_phi;
            const double f = coriolis_enabled_ ? (2.0 * OMEGA * std::sin(phi)) : 0.0;

            const double dx = delta_x_(i, j);
            const Real inv_dx = 1.0 / dx;

            const Real eij = eta_data[ij];
            const Real eimj = eta_data[imj];
            const Real eijm = eta_data[ijm];  // Назад

            const Real uij = u_data[ij];
            const Real uimj = u_data[imj];
            const Real uijm = u_data[ijm];

            const Real vij = v_data[ij];
            const Real vimj = v_data[imj];
            const Real vijm = v_data[ijm];

            const Real Hij = H_data[ij];
            const Real Himj = H_data[imj];
            const Real Hijm = H_data[ijm];

            const Real Dimj = D_data[imj];

            const Real Hu_ij = Hij * uij;
            const Real Hu_imj = Himj * uimj;

            const Real Hv_ij = Hij * vij;
            const Real Hv_ijm = Hijm * vijm;

            const Real flux_H_x = (Hu_ij - Hu_imj) * inv_dx;
            const Real flux_H_y = (Hv_ij - Hv_ijm) * inv_dy;  // Односторонняя назад

            H_hat_data[ij] = Hij - tau * (flux_H_x + flux_H_y);

            const Real du_dx = (uij - uimj) * inv_dx;
            const Real du_dy = (uij - uijm) * inv_dy;  // Односторонняя назад

            const Real dv_dx = (vij - vimj) * inv_dx;
            const Real dv_dy = (vij - vijm) * inv_dy;  // Односторонняя назад

            const Real deta_dx = (eij - eimj) * inv_dx;
            const Real deta_dy = (eij - eijm) * inv_dy;  // Односторонняя назад

            u_hat_data[ij] = uij - tau * (uij * du_dx + vij * du_dy + g * deta_dx - f * vij);
            v_hat_data[ij] = vij - tau * (uij * dv_dx + vij * dv_dy + g * deta_dy + f * uij);
            eta_hat_data[ij] = H_hat_data[ij] - Dij;
        }
    }

    // Инициализация границ i=0 и i=nx-1
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < ny; ++j) {
        {
            const int ij = IDX(0, j, ny);
            const Real Dij = D_data[ij];
            if (Dij < grnd) {
                H_hat_data[ij] = Dij;
                u_hat_data[ij] = 0.0;
                v_hat_data[ij] = 0.0;
                eta_hat_data[ij] = 0.0;
            } else {
                H_hat_data[ij] = H_data[ij];
                u_hat_data[ij] = u_data[ij];
                v_hat_data[ij] = v_data[ij];
                eta_hat_data[ij] = eta_data[ij];
            }
        }
        {
            const int ij = IDX(nx - 1, j, ny);
            const Real Dij = D_data[ij];
            if (Dij < grnd) {
                H_hat_data[ij] = Dij;
                u_hat_data[ij] = 0.0;
                v_hat_data[ij] = 0.0;
                eta_hat_data[ij] = 0.0;
            } else {
                H_hat_data[ij] = H_data[ij];
                u_hat_data[ij] = u_data[ij];
                v_hat_data[ij] = v_data[ij];
                eta_hat_data[ij] = eta_data[ij];
            }
        }
    }
}

// ============================================================================
// Реализация для ДЕКАРТОВОЙ системы координат (БЕЗ Кориолиса)
// ============================================================================

void SubGrid::ComputeHalfStepCartesian() {
    const int nx = local_nx_;
    const int ny = local_ny_;
    const double tau = tau_;
    const double g = g_;
    const double inv_dy = inv_dy_;
    const Real grnd = min_calc_value;
    const bool is_vertical = (decomp_dir_ == DecompositionDirection::Vertical);
    const bool wet_dry = wet_dry_config().enabled;  // Phase 8 Codex Monai fix
    const bool hydro = wet_dry && hydrostatic_reconstruction_enabled();

    Real* __restrict H_data = local_grid_.GetH();
    Real* __restrict u_data = local_grid_.GetU();
    Real* __restrict v_data = local_grid_.GetV();
    Real* __restrict eta_data = local_grid_.GetEta();
    Real* __restrict D_data = local_grid_.GetD();
    Real* __restrict H_hat_data = H_hat_.get();
    Real* __restrict u_hat_data = u_hat_.get();
    Real* __restrict v_hat_data = v_hat_.get();
    Real* __restrict eta_hat_data = eta_hat_.get();

    // Interior loop: always [1, nx-2] x [1, ny-2]
    // Ghost/domain edges are handled by ExchangeBoundaryData (pre-filled with zero-gradient)
    const Real* __restrict inv_dx_data = inv_dx_arr_.get();

    #pragma omp parallel for schedule(static)
    for (int i = 1; i < nx - 1; ++i) {
        #pragma omp simd
        for (int j = 1; j < ny - 1; ++j) {
            const int ij = IDX(i, j, ny);
            const int imj = IDX(i - 1, j, ny);
            const int ijm = IDX(i, j - 1, ny);
            const int ipj = IDX(i + 1, j, ny);
            const int ijp = IDX(i, j + 1, ny);

            const Real Dij = D_data[ij];

            // Early exit for dry cells (preserves original short-circuit to avoid NaN from 0/0)
            if (Dij < grnd) {
                // Codex Phase 8: in wet_dry mode store H=0 instead of negative D
                // to avoid nonfinite propagation through MacCormack stencil.
                H_hat_data[ij] = wet_dry ? Real(0) : Dij;
                u_hat_data[ij] = 0.0;
                v_hat_data[ij] = 0.0;
                eta_hat_data[ij] = 0.0;
                continue;
            }

            const Real inv_dx = inv_dx_data[j];  // Precomputed, no std::function call

            const Real eij = eta_data[ij];
            const Real uij = u_data[ij];
            const Real vij = v_data[ij];
            const Real Hij = H_data[ij];

            const Real Himj = H_data[imj];
            const Real Hijm = H_data[ijm];

            const Real Dimj = D_data[imj];
            const Real Dijm = D_data[ijm];

            // Codex Phase 8 Audusse: hydrostatic reconstruction on backward edges
            // (i-1,i) for x and (j-1,j) for y. Both mass flux H*u and pressure
            // gradient g*∂η/∂x use reconstructed h, η values w.r.t. shared bottom.
            Real flux_H_x, flux_H_y, deta_dx, deta_dy;
            if (hydro) {
                const HydrostaticEdge ex = HydroEdge(Himj, Dimj, Hij, Dij);
                const HydrostaticEdge ey = HydroEdge(Hijm, Dijm, Hij, Dij);
                flux_H_x = (ex.hR * uij - ex.hL * u_data[imj]) * inv_dx;
                flux_H_y = (ey.hR * vij - ey.hL * v_data[ijm]) * inv_dy;
                deta_dx  = (ex.etaR - ex.etaL) * inv_dx;
                deta_dy  = (ey.etaR - ey.etaL) * inv_dy;
            } else {
                flux_H_x = (Hij * uij - Himj * u_data[imj]) * inv_dx;
                flux_H_y = (Hij * vij - Hijm * v_data[ijm]) * inv_dy;
                deta_dx  = (eij - eta_data[imj]) * inv_dx;
                deta_dy  = (eij - eta_data[ijm]) * inv_dy;
            }

            const Real H_hat_new = Hij - tau * (flux_H_x + flux_H_y);
            // Codex Phase 8 wet_dry: clamp H_hat to grnd if nonfinite or below
            // floor. eta_hat = grnd - Dij (not zero) preserves bathymetry awareness.
            if (wet_dry && (!std::isfinite(static_cast<double>(H_hat_new)) || H_hat_new < grnd)) {
                H_hat_data[ij]   = grnd;
                u_hat_data[ij]   = Real(0);
                v_hat_data[ij]   = Real(0);
                eta_hat_data[ij] = grnd - Dij;
                continue;
            }
            H_hat_data[ij] = H_hat_new;

            // Wet/dry mode: Froude clamp + Manning friction on shallow cells.
            // Applied to u/v BEFORE writing them (computed later in this block).

            const Real du_dx = (uij - u_data[imj]) * inv_dx;
            const Real du_dy = (uij - u_data[ijm]) * inv_dy;
            const Real dv_dx = (vij - v_data[imj]) * inv_dx;
            const Real dv_dy = (vij - v_data[ijm]) * inv_dy;
            // Note: deta_dx, deta_dy computed above (HR-aware if hydro==true)

            Real u_hat_new = uij - tau * (uij * du_dx + vij * du_dy + g * deta_dx);
            Real v_hat_new = vij - tau * (uij * dv_dx + vij * dv_dy + g * deta_dy);

            // Land boundary: zero velocity at water-land interface
            if (u_hat_new != 0.0) {
                if ((i + 1 < nx - 1 && D_data[ipj] <= grnd) ||
                    (i - 1 > 0 && Dimj <= grnd)) {
                    u_hat_new = 0.0;
                }
            }
            if (v_hat_new != 0.0) {
                if ((j + 1 < ny - 1 && D_data[ijp] <= grnd) ||
                    (j - 1 > 0 && Dijm <= grnd)) {
                    v_hat_new = 0.0;
                }
            }

            // Codex Phase 8 wet_dry: Froude clamp + Manning friction (after compute).
            if (wet_dry) {
                const WetDryConfig& wd = wet_dry_config();
                const Real H_loc = H_hat_data[ij];
                if (H_loc > grnd) {
                    const Real c_loc = std::sqrt(static_cast<Real>(g) * H_loc);
                    const Real umax = static_cast<Real>(wd.froude_max) * c_loc;
                    if (std::fabs(u_hat_new) > umax) {
                        u_hat_new = (u_hat_new > 0 ? umax : -umax);
                    }
                    if (std::fabs(v_hat_new) > umax) {
                        v_hat_new = (v_hat_new > 0 ? umax : -umax);
                    }
                    // Manning friction on shallow cells: f = g*n²*|U|*U / H^(4/3).
                    if (H_loc < static_cast<Real>(wd.shallow_for_friction)) {
                        const Real Cf = static_cast<Real>(g * wd.manning_n * wd.manning_n)
                                      / std::pow(H_loc, static_cast<Real>(4.0/3.0));
                        const Real U = std::sqrt(u_hat_new * u_hat_new + v_hat_new * v_hat_new);
                        const Real damp = Real(1) / (Real(1) + static_cast<Real>(tau) * Cf * U);
                        u_hat_new *= damp;
                        v_hat_new *= damp;
                    }
                }
            }

            u_hat_data[ij] = u_hat_new;
            v_hat_data[ij] = v_hat_new;
            eta_hat_data[ij] = H_hat_data[ij] - Dij;
        }
    }

    // Инициализация hat-переменных на границах (ghost/domain edges)
    // Для обоих режимов: инициализируем i=0, i=nx-1, j=0, j=ny-1
    // (те из них что являются ghost будут перезаписаны ExchangeHatBoundaryData)
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < ny; ++j) {
        for (int edge_i : {0, nx - 1}) {
            const int ij = IDX(edge_i, j, ny);
            const Real Dij = D_data[ij];
            if (Dij < grnd) {
                H_hat_data[ij] = wet_dry ? Real(0) : Dij;  // Codex Phase 8
                u_hat_data[ij] = 0.0;
                v_hat_data[ij] = 0.0;
                eta_hat_data[ij] = 0.0;
            } else {
                H_hat_data[ij] = H_data[ij];
                u_hat_data[ij] = u_data[ij];
                v_hat_data[ij] = v_data[ij];
                eta_hat_data[ij] = eta_data[ij];
            }
        }
    }
    if (is_vertical) {
        // Vertical: also init ghost rows j=0, j=ny-1
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < nx - 1; ++i) {
            for (int edge_j : {0, ny - 1}) {
                const int ij = IDX(i, edge_j, ny);
                const Real Dij = D_data[ij];
                if (Dij < grnd) {
                    H_hat_data[ij] = wet_dry ? Real(0) : Dij;  // Codex Phase 8
                    u_hat_data[ij] = 0.0;
                    v_hat_data[ij] = 0.0;
                    eta_hat_data[ij] = 0.0;
                } else {
                    H_hat_data[ij] = H_data[ij];
                    u_hat_data[ij] = u_data[ij];
                    v_hat_data[ij] = v_data[ij];
                    eta_hat_data[ij] = eta_data[ij];
                }
            }
        }
    }

    // Flux-based correction for DG-coupled boundaries.
    // Re-compute H_hat for i=1 (left) using the DG mass flux instead of the
    // ghost-cell-derived flux. Mass conservation only — momentum is handled
    // through state-coupled ghost values (u[0], v[0]) inside the standard
    // predictor loop above. The previous attempt to also drive momentum from
    // F_Hu/F_H is unsafe: at lake-at-rest the boundary mass flux F_H ≈ 0
    // while the pressure part of F_Hu = ∫ g*H²/2 dy is O(1), so F_Hu/F_H
    // blows up and seeds catastrophic velocities on the next predictor step.
    if (use_flux_left_ && !flux_left_.empty()) {
        const int i = 1;
        for (int j = 1; j < ny - 1; ++j) {
            const int ij = IDX(i, j, ny);
            if (D_data[ij] < grnd) continue;

            double Hij = H_data[ij];
            double uij = u_data[ij];
            double vij = v_data[ij];

            double F_H_dg = flux_left_[j * 3 + 0];
            double flux_H_x = (Hij * uij - F_H_dg) * inv_dx_data[j];
            double flux_H_y = (Hij * vij - H_data[IDX(i, j-1, ny)] * v_data[IDX(i, j-1, ny)]) * inv_dy_;

            H_hat_data[ij] = Hij - tau_ * (flux_H_x + flux_H_y);
            eta_hat_data[ij] = H_hat_data[ij] - D_data[ij];
            // u_hat[1,j], v_hat[1,j] keep values produced by the standard
            // predictor (with state-coupled ghost u[0],v[0]).
        }
    }

    // Predictor at i=nx-2 already reads H[nx-1] = state-coupled ghost from
    // the structured side of the right DG interface, so no flux override is
    // needed here. The right-side correction lives in ComputeFullStep.
}

void SubGrid::ComputeFullStepCartesian() {
    const int nx = local_nx_;
    const int ny = local_ny_;
    const double tau = tau_;
    const double g = g_;
    const double inv_dy = inv_dy_;
    const Real grnd = min_calc_value;
    const bool is_vertical = (decomp_dir_ == DecompositionDirection::Vertical);
    const bool wet_dry = wet_dry_config().enabled;  // Phase 8 Codex Monai fix
    const bool hydro = wet_dry && hydrostatic_reconstruction_enabled();

    Real* __restrict H_data = local_grid_.GetH();
    Real* __restrict u_data = local_grid_.GetU();
    Real* __restrict v_data = local_grid_.GetV();
    Real* __restrict eta_data = local_grid_.GetEta();
    Real* __restrict D_data = local_grid_.GetD();
    Real* __restrict H_hat_data = H_hat_.get();
    Real* __restrict u_hat_data = u_hat_.get();
    Real* __restrict v_hat_data = v_hat_.get();
    Real* __restrict eta_hat_data = eta_hat_.get();

    const Real* __restrict inv_dx_data = inv_dx_arr_.get();

    #pragma omp parallel for schedule(static)
    for (int i = 1; i < nx - 1; ++i) {
        #pragma omp simd
        for (int j = 1; j < ny - 1; ++j) {
            const int ij = IDX(i, j, ny);
            const int ipj = IDX(i + 1, j, ny);
            const int ijp = IDX(i, j + 1, ny);
            const int imj = IDX(i - 1, j, ny);
            const int ijm = IDX(i, j - 1, ny);

            const Real Dij = D_data[ij];

            if (Dij < grnd) {
                H_data[ij] = wet_dry ? Real(0) : Dij;  // Codex Phase 8
                u_data[ij] = 0.0;
                v_data[ij] = 0.0;
                eta_data[ij] = 0.0;
                continue;
            }

            const Real inv_dx = inv_dx_data[j];  // Precomputed, no std::function call

            const bool at_x_interface =
                (D_data[imj] < grnd) || (D_data[ipj] < grnd);
            const bool at_y_interface =
                (D_data[ijm] < grnd) || (D_data[ijp] < grnd);

            const Real H_hat_ij = H_hat_data[ij];
            const Real H_hat_ipj = H_hat_data[ipj];
            const Real H_hat_ijp = H_hat_data[ijp];

            const Real u_hat_ij = u_hat_data[ij];
            const Real u_hat_ipj = u_hat_data[ipj];
            const Real u_hat_ijp = u_hat_data[ijp];

            const Real v_hat_ij = v_hat_data[ij];
            const Real v_hat_ipj = v_hat_data[ipj];
            const Real v_hat_ijp = v_hat_data[ijp];

            const Real eta_hat_ij = eta_hat_data[ij];
            const Real eta_hat_ipj = eta_hat_data[ipj];
            const Real eta_hat_ijp = eta_hat_data[ijp];

            // Codex Phase 8 Audusse: HR on forward edges (i,i+1) and (j,j+1).
            Real flux_H_hat_x, flux_H_hat_y, deta_hat_dx, deta_hat_dy;
            if (hydro) {
                const HydrostaticEdge ex = HydroEdge(H_hat_ij, Dij, H_hat_ipj, D_data[ipj]);
                const HydrostaticEdge ey = HydroEdge(H_hat_ij, Dij, H_hat_ijp, D_data[ijp]);
                flux_H_hat_x = (ex.hR * u_hat_ipj - ex.hL * u_hat_ij) * inv_dx;
                flux_H_hat_y = (ey.hR * v_hat_ijp - ey.hL * v_hat_ij) * inv_dy;
                deta_hat_dx  = (ex.etaR - ex.etaL) * inv_dx;
                deta_hat_dy  = (ey.etaR - ey.etaL) * inv_dy;
            } else {
                flux_H_hat_x = (H_hat_ipj * u_hat_ipj - H_hat_ij * u_hat_ij) * inv_dx;
                flux_H_hat_y = (H_hat_ijp * v_hat_ijp - H_hat_ij * v_hat_ij) * inv_dy;
                deta_hat_dx  = (eta_hat_ipj - eta_hat_ij) * inv_dx;
                deta_hat_dy  = (eta_hat_ijp - eta_hat_ij) * inv_dy;
            }

            Real H_new = 0.5 * (H_data[ij] + H_hat_ij) - 0.5 * tau * (flux_H_hat_x + flux_H_hat_y);

            const Real du_hat_dx = (u_hat_ipj - u_hat_ij) * inv_dx;
            const Real du_hat_dy = (u_hat_ijp - u_hat_ij) * inv_dy;
            const Real dv_hat_dx = (v_hat_ipj - v_hat_ij) * inv_dx;
            const Real dv_hat_dy = (v_hat_ijp - v_hat_ij) * inv_dy;
            // Note: deta_hat_dx/dy computed above (HR-aware if hydro==true)

            Real u_new = 0.5 * (u_data[ij] + u_hat_ij) - 0.5 * tau * (
                u_hat_ij * du_hat_dx + v_hat_ij * du_hat_dy + g * deta_hat_dx);
            Real v_new = 0.5 * (v_data[ij] + v_hat_ij) - 0.5 * tau * (
                u_hat_ij * dv_hat_dx + v_hat_ij * dv_hat_dy + g * deta_hat_dy);

            if (at_x_interface) u_new = 0.0;
            if (at_y_interface) v_new = 0.0;

            // Enforce positivity (Codex Phase 8 wet_dry: clamp to grnd + NaN guard,
            // eta = grnd - Dij preserves bathymetry awareness; legacy: H = Dij, eta=0).
            const bool bad = wet_dry
                ? (!std::isfinite(static_cast<double>(H_new)) ||
                   !std::isfinite(static_cast<double>(u_new)) ||
                   !std::isfinite(static_cast<double>(v_new)) ||
                   H_new < grnd)
                : (H_new < grnd);
            if (bad) {
                H_data[ij] = wet_dry ? grnd : Dij;
                u_data[ij] = 0.0;
                v_data[ij] = 0.0;
                eta_data[ij] = wet_dry ? (grnd - Dij) : Real(0);
            } else {
                // Codex Phase 8 wet_dry: Froude clamp + Manning friction.
                if (wet_dry) {
                    const WetDryConfig& wd = wet_dry_config();
                    const Real c_loc = std::sqrt(static_cast<Real>(g) * H_new);
                    const Real umax = static_cast<Real>(wd.froude_max) * c_loc;
                    if (std::fabs(u_new) > umax) u_new = (u_new > 0 ? umax : -umax);
                    if (std::fabs(v_new) > umax) v_new = (v_new > 0 ? umax : -umax);
                    if (H_new < static_cast<Real>(wd.shallow_for_friction)) {
                        const Real Cf = static_cast<Real>(g * wd.manning_n * wd.manning_n)
                                      / std::pow(H_new, static_cast<Real>(4.0/3.0));
                        const Real U = std::sqrt(u_new * u_new + v_new * v_new);
                        const Real damp = Real(1) / (Real(1) + static_cast<Real>(tau) * Cf * U);
                        u_new *= damp;
                        v_new *= damp;
                    }
                }
                H_data[ij] = H_new;
                u_data[ij] = u_new;
                v_data[ij] = v_new;
                eta_data[ij] = H_new - Dij;
            }
        }
    }

    // Flux-based correction for FullStep (corrector) at DG-coupled boundaries.
    // Replace ONLY the H corrector at i=nx-2 with the DG mass flux. Momentum
    // (u, v) keeps the value produced by the standard corrector above (which
    // reads u_hat[nx-1], v_hat[nx-1] from the state-coupled ghost row). See
    // the matching comment in ComputeHalfStep for why we do not derive
    // u_dg = F_Hu / F_H here.
    if (use_flux_right_ && !flux_right_.empty()) {
        const int i = nx - 2;
        for (int j = 1; j < ny - 1; ++j) {
            const int ij = IDX(i, j, ny);
            if (D_data[ij] < grnd) continue;

            const Real inv_dx = inv_dx_data[j];
            const Real H_hat_ij = H_hat_data[ij];
            const Real u_hat_ij = u_hat_data[ij];
            const Real v_hat_ij = v_hat_data[ij];

            double F_H_dg = flux_right_[j * 3 + 0];
            double flux_H_hat_x = (F_H_dg - H_hat_ij * u_hat_ij) * inv_dx;
            double flux_H_hat_y = (H_hat_data[IDX(i, j+1, ny)] * v_hat_data[IDX(i, j+1, ny)] - H_hat_ij * v_hat_ij) * inv_dy_;

            Real H_new = 0.5 * (H_data[ij] + H_hat_ij) - 0.5 * tau_ * (flux_H_hat_x + flux_H_hat_y);

            if (H_new < grnd) {
                H_data[ij] = D_data[ij];
                u_data[ij] = 0;
                v_data[ij] = 0;
                eta_data[ij] = 0;
            } else {
                H_data[ij] = H_new;
                eta_data[ij] = H_new - D_data[ij];
                // u_data[ij], v_data[ij] keep the values written by the
                // standard corrector loop above.
            }
        }
    }

    // FullStep for i=1 uses H_hat[2] (interior) — no correction needed for left boundary predictor.
    // But corrector at i=1 uses H_hat[i+1]=H_hat[2] which is interior — OK.
    // Left boundary correction only needed in HalfStep (predictor uses H[0] ghost).
}

void SubGrid::ComputeFullStepSpherical() {
    const int nx = local_nx_;
    const int ny = local_ny_;
    const double tau = tau_;
    const double g = g_;
    const double inv_dy = inv_dy_;
    const Real grnd = min_calc_value;

    // Константы для сферической системы
    constexpr double R_EARTH = 6371000.0;
    constexpr double OMEGA = 7.2921e-5;

    const double PHI_0 = center_latitude_rad_;
    const double delta_y = 1.0 / inv_dy;
    const double delta_phi = delta_y / R_EARTH;

    Real* __restrict H_data = local_grid_.GetH();
    Real* __restrict u_data = local_grid_.GetU();
    Real* __restrict v_data = local_grid_.GetV();
    Real* __restrict eta_data = local_grid_.GetEta();
    Real* __restrict D_data = local_grid_.GetD();
    Real* __restrict H_hat_data = H_hat_.get();
    Real* __restrict u_hat_data = u_hat_.get();
    Real* __restrict v_hat_data = v_hat_.get();
    Real* __restrict eta_hat_data = eta_hat_.get();

    #pragma omp parallel for schedule(static)
    for (int i = 1; i < nx - 1; ++i) {
        #pragma omp simd
        for (int j = 1; j < ny - 1; ++j) {
            const int ij = IDX(i, j, ny);
            const int ipj = IDX(i + 1, j, ny);
            const int ijp = IDX(i, j + 1, ny);
            const int imj = IDX(i - 1, j, ny);
            const int ijm = IDX(i, j - 1, ny);

            const Real Dij = D_data[ij];

            if (Dij < grnd) {
                H_data[ij] = Dij;
                u_data[ij] = 0.0;
                v_data[ij] = 0.0;
                eta_data[ij] = 0.0;
                continue;
            }

            // Широта и Кориолис
            const double phi = PHI_0 + (j - ny / 2.0) * delta_phi;
            const double f = coriolis_enabled_ ? (2.0 * OMEGA * std::sin(phi)) : 0.0;

            const double dx = delta_x_(i, j);
            const Real inv_dx = 1.0 / dx;

            const bool at_x_interface =
                (D_data[imj] < grnd) || (D_data[ipj] < grnd);
            const bool at_y_interface =
                (D_data[ijm] < grnd) || (D_data[ijp] < grnd);

            // Кэширование hat-значений
            const Real H_hat_ij = H_hat_data[ij];
            const Real H_hat_ipj = H_hat_data[ipj];
            const Real H_hat_ijp = H_hat_data[ijp];

            const Real u_hat_ij = u_hat_data[ij];
            const Real u_hat_ipj = u_hat_data[ipj];
            const Real u_hat_ijp = u_hat_data[ijp];

            const Real v_hat_ij = v_hat_data[ij];
            const Real v_hat_ipj = v_hat_data[ipj];
            const Real v_hat_ijp = v_hat_data[ijp];

            const Real eta_hat_ij = eta_hat_data[ij];
            const Real eta_hat_ipj = eta_hat_data[ipj];
            const Real eta_hat_ijp = eta_hat_data[ijp];

            const double Hu_hat_ij = H_hat_ij * u_hat_ij;
            const double Hu_hat_ipj = H_hat_ipj * u_hat_ipj;

            const double Hv_hat_ij = H_hat_ij * v_hat_ij;
            const double Hv_hat_ijp = H_hat_ijp * v_hat_ijp;

            const Real flux_H_hat_x = (Hu_hat_ipj - Hu_hat_ij) * inv_dx;
            const Real flux_H_hat_y = (Hv_hat_ijp - Hv_hat_ij) * inv_dy;

            Real H_new = 0.5 * (H_data[ij] + H_hat_ij) - 0.5 * tau * (flux_H_hat_x + flux_H_hat_y);

            const Real du_hat_dx = (u_hat_ipj - u_hat_ij) * inv_dx;
            const Real du_hat_dy = (u_hat_ijp - u_hat_ij) * inv_dy;

            const Real dv_hat_dx = (v_hat_ipj - v_hat_ij) * inv_dx;
            const Real dv_hat_dy = (v_hat_ijp - v_hat_ij) * inv_dy;

            const Real deta_hat_dx = (eta_hat_ipj - eta_hat_ij) * inv_dx;
            const Real deta_hat_dy = (eta_hat_ijp - eta_hat_ij) * inv_dy;

            // С Кориолисом
            Real u_new = 0.5 * (u_data[ij] + u_hat_ij) - 0.5 * tau * (
                u_hat_ij * du_hat_dx +
                v_hat_ij * du_hat_dy +
                g * deta_hat_dx -
                f * v_hat_ij
            );

            Real v_new = 0.5 * (v_data[ij] + v_hat_ij) - 0.5 * tau * (
                u_hat_ij * dv_hat_dx +
                v_hat_ij * dv_hat_dy +
                g * deta_hat_dy +
                f * u_hat_ij
            );

            if (at_x_interface) u_new = 0.0;
            if (at_y_interface) v_new = 0.0;

            // Enforce positivity: если полная глубина < порога — ячейка "сухая"
            if (H_new < grnd) {
                H_data[ij] = Dij;
                u_data[ij] = 0.0;
                v_data[ij] = 0.0;
                eta_data[ij] = 0.0;
            } else {
                H_data[ij] = H_new;
                u_data[ij] = u_new;
                v_data[ij] = v_new;
                eta_data[ij] = H_new - Dij;
            }
        }
    }
}

// ============================================================================
// Davis 1987 / Liang 2007 TVD post-correction.
//
// Applied after standard MacCormack corrector to (H, Hu, Hv) conservative
// variables. Suppresses Gibbs oscillations near sharp gradients (coastal
// shoaling). Operates on interior cells (i,j in [2..nx-3] x [2..ny-3]).
// Boundary cells keep the MC values to avoid stencil-out-of-range issues.
//
// Wet/dry safety: skip cell if center or any neighbor in 5-point stencil
// has D < grnd. Near shallow (D < shallow_depth) auto-switch limiter to minmod.
// ============================================================================
void SubGrid::ApplyTvdPostCorrection(int global_t, bool use_global_schedule) {
    const TvdConfig& cfg = tvd_config();
    if (!cfg.enabled) return;

    // Phase 2c: apply TVD only every Nth step (default 1).
    // Two scheduling paths:
    //   - LEGACY (use_global_schedule=false, default): per-subgrid tvd_call_count_.
    //     All subgrids stay in sync because they're called together each timestep.
    //   - ACTIVE-REGION (use_global_schedule=true): global timestep schedule.
    //     Late-активированные subgrids иначе попали бы в другую фазу TVD.
    //     Условие совпадает с legacy при apply_every==1 и первым вызовом на t=0.
    if (use_global_schedule) {
        // Caller (active-region path) must pass the current global timestep.
        // global_t < 0 is a programming error: legacy default would be silently
        // wrong (phase shift relative to apply_every). Fall back to legacy with
        // a loud warning so the bug surfaces in logs instead of corrupting TVD.
        if (global_t < 0) {
            std::cerr << "[SubGrid::ApplyTvdPostCorrection] ERROR: "
                         "use_global_schedule=true but global_t=" << global_t
                      << " (must be >= 0). Falling back to legacy per-subgrid counter."
                      << std::endl;
            tvd_call_count_ += 1;
            if (cfg.apply_every > 1 && ((tvd_call_count_ % cfg.apply_every) != 0)) {
                return;
            }
        } else {
            if (cfg.apply_every > 1 && ((global_t + 1) % cfg.apply_every) != 0) {
                return;
            }
        }
    } else {
        tvd_call_count_ += 1;
        if (cfg.apply_every > 1 && ((tvd_call_count_ % cfg.apply_every) != 0)) {
            return;
        }
    }

    const int nx = local_nx_;
    const int ny = local_ny_;
    const double tau = tau_;
    const double g = g_;
    const double inv_dy = inv_dy_;
    const Real grnd = min_calc_value;
    const double shallow = cfg.shallow_depth;
    const bool force_minmod = cfg.force_minmod;
    const int bskip = cfg.boundary_skip;

    // Effective loop bounds: skip TVD in boundary guard zone (Codex Phase 2b
    // diagnostic — isolates BC-TVD stencil/ghost interaction from interior).
    const int j_lo = std::max(2, bskip);
    const int j_hi = std::min(ny - 2, ny - 1 - bskip);
    // (i bounds also extended by bskip for consistency since horizontal stripes
    // also touch global east/west physical boundaries in full-domain runs.)
    const int i_lo = std::max(2, bskip);
    const int i_hi = std::min(nx - 2, nx - 1 - bskip);

    Real* __restrict H_data = local_grid_.GetH();
    Real* __restrict u_data = local_grid_.GetU();
    Real* __restrict v_data = local_grid_.GetV();
    Real* __restrict eta_data = local_grid_.GetEta();
    Real* __restrict D_data = local_grid_.GetD();

    // Avoid in-place race condition: accumulate TVD deltas into temp buffers,
    // then apply in a second pass. The 5-point stencil otherwise reads cells
    // partially updated by neighboring OMP threads.
    const size_t N = (size_t)nx * (size_t)ny;
    std::vector<double> deltaH (N, 0.0);
    std::vector<double> deltaHu(N, 0.0);
    std::vector<double> deltaHv(N, 0.0);
    std::vector<unsigned char> apply_mask(N, 0);  // 1 if this cell got TVD update

    // Stencil needs i-2..i+2 and j-2..j+2. Boundary-skip zone removes TVD
    // from cells near physical edges (e.g. j<20 / j>ny-21).
    #pragma omp parallel for schedule(static)
    for (int i = i_lo; i < i_hi; ++i) {
        for (int j = j_lo; j < j_hi; ++j) {
            const int ij = IDX(i, j, ny);
            const double D_ij = D_data[ij];
            if (D_ij < grnd) continue;

            // Wet/dry safety: skip if any of the 5-point stencil neighbors is dry.
            const int im2 = IDX(i - 2, j, ny);
            const int im1 = IDX(i - 1, j, ny);
            const int ip1 = IDX(i + 1, j, ny);
            const int ip2 = IDX(i + 2, j, ny);
            const int jm2 = IDX(i, j - 2, ny);
            const int jm1 = IDX(i, j - 1, ny);
            const int jp1 = IDX(i, j + 1, ny);
            const int jp2 = IDX(i, j + 2, ny);

            if (D_data[im2] < grnd || D_data[im1] < grnd || D_data[ip1] < grnd || D_data[ip2] < grnd) continue;
            if (D_data[jm2] < grnd || D_data[jm1] < grnd || D_data[jp1] < grnd || D_data[jp2] < grnd) continue;

            // Determine if any cell in stencil is shallow -> use minmod locally.
            bool use_minmod = force_minmod;
            if (!use_minmod) {
                if (D_ij < shallow ||
                    D_data[im2] < shallow || D_data[im1] < shallow || D_data[ip1] < shallow || D_data[ip2] < shallow ||
                    D_data[jm2] < shallow || D_data[jm1] < shallow || D_data[jp1] < shallow || D_data[jp2] < shallow) {
                    use_minmod = true;
                }
            }

            // Local dx (delta_x_ is std::function<double(int,int)>).
            const double dx = delta_x_(i, j);
            const double inv_dx = 1.0 / dx;

            // Local wave speed for Courant number (use max over stencil for conservativeness).
            const double c0 = std::sqrt(g * H_data[ij]);
            const double u0 = u_data[ij];
            const double v0 = v_data[ij];
            const double ax = std::abs(u0) + c0;
            const double ay = std::abs(v0) + c0;
            const double nu_x = ax * tau * inv_dx;
            const double nu_y = ay * tau * inv_dy;

            // WELL-BALANCED FIX (Phase 2a, 2026-05-24):
            // Davis TVD post-correction must operate on perturbation variables.
            // Naive (h, hu, hv) breaks well-balancedness because Δh = ΔD ≠ 0 at
            // lake-at-rest, so the limiter diffuses static bathymetry into a
            // spurious wave (η_new = h_new - D ≠ 0). Confirmed by spatial
            // analysis on Pacific cor (project_tvd_wb_violation_may24).
            //
            // Fix: TVD operates on (η, Hu, Hv) — at lake-at-rest all three are
            // zero, so all Δ = 0 and the correction vanishes (well-balanced).
            // Conservation preserved: ∂η/∂t = ∂h/∂t for static D, divergent
            // flux form remains a flux-difference.

            // η values at 5-point stencil (x-direction)
            const double eta_im2 = (double)eta_data[im2];
            const double eta_im1 = (double)eta_data[im1];
            const double eta_ij  = (double)eta_data[ij];
            const double eta_ip1 = (double)eta_data[ip1];
            const double eta_ip2 = (double)eta_data[ip2];

            // H = D + η (compute on the fly for Hu/Hv flux; conservative)
            const double H_im2 = D_data[im2] + eta_im2;
            const double H_im1 = D_data[im1] + eta_im1;
            const double H_ij  = D_ij        + eta_ij;
            const double H_ip1 = D_data[ip1] + eta_ip1;
            const double H_ip2 = D_data[ip2] + eta_ip2;

            const double Hu_im2 = H_im2 * u_data[im2];
            const double Hu_im1 = H_im1 * u_data[im1];
            const double Hu_ij  = H_ij  * u0;
            const double Hu_ip1 = H_ip1 * u_data[ip1];
            const double Hu_ip2 = H_ip2 * u_data[ip2];

            const double Hv_im2 = H_im2 * v_data[im2];
            const double Hv_im1 = H_im1 * v_data[im1];
            const double Hv_ij  = H_ij  * v0;
            const double Hv_ip1 = H_ip1 * v_data[ip1];
            const double Hv_ip2 = H_ip2 * v_data[ip2];

            // Deltas in x — apply TVD to η (NOT h) for well-balancedness.
            const double dE_xm  = eta_im1 - eta_im2;
            const double dE_x   = eta_ij  - eta_im1;
            const double dE_xp  = eta_ip1 - eta_ij;
            const double dE_xpp = eta_ip2 - eta_ip1;

            const double dHu_xm  = Hu_im1 - Hu_im2;
            const double dHu_x   = Hu_ij  - Hu_im1;
            const double dHu_xp  = Hu_ip1 - Hu_ij;
            const double dHu_xpp = Hu_ip2 - Hu_ip1;

            const double dHv_xm  = Hv_im1 - Hv_im2;
            const double dHv_x   = Hv_ij  - Hv_im1;
            const double dHv_xp  = Hv_ip1 - Hv_ij;
            const double dHv_xpp = Hv_ip2 - Hv_ip1;

            // Ratios r at face i+1/2 (using van Leer/minmod convention r = up/down)
            // For B at face i+1/2: r_left = dH_x / dH_xp,  r_right = dH_xpp / dH_xp
            // B_{i+1/2} = (1/2) [ (1-phi(r_left)) + (1-phi(r_right)) ] * nu * (1-nu) / 2
            // But we use the simpler single-face form: B = 0.5 * nu * (1-nu) * (1 - phi(r_upwind))
            // where r_upwind is r_left (upwind side for positive flow).
            // For symmetric form take average of left/right limiter contribution.

            auto B_at_face = [&](double d_up, double d_face, double d_down_right) {
                // r_upwind = d_up / d_face
                // r_downwind = d_down_right / d_face
                double r_up = tvd_ratio(d_up, d_face);
                double r_dn = tvd_ratio(d_down_right, d_face);
                // Symmetric (Davis) form: average two contributions
                double Bu = tvd_B(nu_x, r_up, use_minmod);
                double Bd = tvd_B(nu_x, r_dn, use_minmod);
                return 0.5 * (Bu + Bd);
            };
            auto By_at_face = [&](double d_up, double d_face, double d_down_right) {
                double r_up = tvd_ratio(d_up, d_face);
                double r_dn = tvd_ratio(d_down_right, d_face);
                double Bu = tvd_B(nu_y, r_up, use_minmod);
                double Bd = tvd_B(nu_y, r_dn, use_minmod);
                return 0.5 * (Bu + Bd);
            };

            // x-direction TVD term — limiter on η (well-balanced), Hu/Hv (conservative).
            const double Bx_E_p  = B_at_face(dE_x,  dE_xp, dE_xpp);
            const double Bx_E_m  = B_at_face(dE_xm, dE_x,  dE_xp );
            const double Bx_Hu_p = B_at_face(dHu_x,  dHu_xp, dHu_xpp);
            const double Bx_Hu_m = B_at_face(dHu_xm, dHu_x,  dHu_xp );
            const double Bx_Hv_p = B_at_face(dHv_x,  dHv_xp, dHv_xpp);
            const double Bx_Hv_m = B_at_face(dHv_xm, dHv_x,  dHv_xp );

            const double DxE  = Bx_E_p  * dE_xp  - Bx_E_m  * dE_x;
            const double DxHu = Bx_Hu_p * dHu_xp - Bx_Hu_m * dHu_x;
            const double DxHv = Bx_Hv_p * dHv_xp - Bx_Hv_m * dHv_x;

            // y-direction (same well-balanced approach: limiter on η, Hu/Hv conservative).
            const double eta_jm2 = (double)eta_data[jm2];
            const double eta_jm1 = (double)eta_data[jm1];
            const double eta_jp1 = (double)eta_data[jp1];
            const double eta_jp2 = (double)eta_data[jp2];

            const double H_jm2 = D_data[jm2] + eta_jm2;
            const double H_jm1 = D_data[jm1] + eta_jm1;
            const double H_jp1 = D_data[jp1] + eta_jp1;
            const double H_jp2 = D_data[jp2] + eta_jp2;
            const double Hu_jm2 = H_jm2 * u_data[jm2];
            const double Hu_jm1 = H_jm1 * u_data[jm1];
            const double Hu_jp1 = H_jp1 * u_data[jp1];
            const double Hu_jp2 = H_jp2 * u_data[jp2];
            const double Hv_jm2 = H_jm2 * v_data[jm2];
            const double Hv_jm1 = H_jm1 * v_data[jm1];
            const double Hv_jp1 = H_jp1 * v_data[jp1];
            const double Hv_jp2 = H_jp2 * v_data[jp2];

            const double dE_ym  = eta_jm1 - eta_jm2;
            const double dE_y   = eta_ij  - eta_jm1;
            const double dE_yp  = eta_jp1 - eta_ij;
            const double dE_ypp = eta_jp2 - eta_jp1;

            const double dHu_ym  = Hu_jm1 - Hu_jm2;
            const double dHu_y   = Hu_ij  - Hu_jm1;
            const double dHu_yp  = Hu_jp1 - Hu_ij;
            const double dHu_ypp = Hu_jp2 - Hu_jp1;

            const double dHv_ym  = Hv_jm1 - Hv_jm2;
            const double dHv_y   = Hv_ij  - Hv_jm1;
            const double dHv_yp  = Hv_jp1 - Hv_ij;
            const double dHv_ypp = Hv_jp2 - Hv_jp1;

            const double By_E_p  = By_at_face(dE_y,  dE_yp, dE_ypp);
            const double By_E_m  = By_at_face(dE_ym, dE_y,  dE_yp );
            const double By_Hu_p = By_at_face(dHu_y,  dHu_yp, dHu_ypp);
            const double By_Hu_m = By_at_face(dHu_ym, dHu_y,  dHu_yp );
            const double By_Hv_p = By_at_face(dHv_y,  dHv_yp, dHv_ypp);
            const double By_Hv_m = By_at_face(dHv_ym, dHv_y,  dHv_yp );

            const double DyE  = By_E_p  * dE_yp  - By_E_m  * dE_y;
            const double DyHu = By_Hu_p * dHu_yp - By_Hu_m * dHu_y;
            const double DyHv = By_Hv_p * dHv_yp - By_Hv_m * dHv_y;

            // Accumulate deltas (non-in-place: avoid OMP race with stencil reads)
            deltaH [ij] = DxE  + DyE;          // now stores delta-eta (NOT delta-h)
            deltaHu[ij] = DxHu + DyHu;
            deltaHv[ij] = DxHv + DyHv;
            apply_mask[ij] = 1;
        }
    }

    // Second pass: apply accumulated deltas. No race here (read fresh H_data,
    // write fresh H_data using only the delta computed in pass 1).
    #pragma omp parallel for schedule(static)
    for (int i = i_lo; i < i_hi; ++i) {
        for (int j = j_lo; j < j_hi; ++j) {
            const int ij = IDX(i, j, ny);
            if (!apply_mask[ij]) continue;

            const double D_ij = D_data[ij];
            const double eta_old = (double)eta_data[ij];
            const double H_old   = D_ij + eta_old;
            const double Hu_old  = H_old * (double)u_data[ij];
            const double Hv_old  = H_old * (double)v_data[ij];

            // deltaH actually stores delta-eta (well-balanced fix).
            const double eta_new = eta_old + deltaH [ij];
            const double H_new   = D_ij + eta_new;
            const double Hu_new  = Hu_old + deltaHu[ij];
            const double Hv_new  = Hv_old + deltaHv[ij];

            // Safety: enforce positivity. If H drops below grnd, mark dry.
            if (H_new < (double)grnd) {
                H_data[ij]   = (Real)D_ij;
                u_data[ij]   = (Real)0.0;
                v_data[ij]   = (Real)0.0;
                eta_data[ij] = (Real)0.0;
                continue;
            }

            const double u_new = Hu_new / H_new;
            const double v_new = Hv_new / H_new;
            H_data[ij]   = (Real)H_new;
            u_data[ij]   = (Real)u_new;
            v_data[ij]   = (Real)v_new;
            eta_data[ij] = (Real)eta_new;
        }
    }
}

void SubGrid::UpdateGlobalGrid(int pid) {
    const int nx = local_nx_;
    const int ny = local_ny_;

    if (decomp_dir_ == DecompositionDirection::Horizontal) {
        // Horizontal: пропускаем ghost columns (i=0, i=nx-1)
        int local_i_start = pid != 0 ? 1 : 0;
        int local_i_end = pid != global_grid_.subgrid_count - 1 ? nx - 1 : nx;

        for (int i = local_i_start; i < local_i_end; ++i) {
            int global_i = global_i_start_ + i - 1;

            if (global_i >= 0 && global_i < global_grid_.GetNx()) {
                Real* local_row_start = &local_grid_.H(i, 0);
                Real* global_row_start = &global_grid_.H(global_i, 0);
                std::copy(local_row_start, local_row_start + ny, global_row_start);

                local_row_start = &local_grid_.u(i, 0);
                global_row_start = &global_grid_.u(global_i, 0);
                std::copy(local_row_start, local_row_start + ny, global_row_start);

                local_row_start = &local_grid_.v(i, 0);
                global_row_start = &global_grid_.v(global_i, 0);
                std::copy(local_row_start, local_row_start + ny, global_row_start);

                local_row_start = &local_grid_.eta(i, 0);
                global_row_start = &global_grid_.eta(global_i, 0);
                std::copy(local_row_start, local_row_start + ny, global_row_start);
            }
        }
    } else {
        // Vertical: пропускаем X-ghost (i=0, i=nx-1) и Y-ghost (j=0, j=ny-1)
        int local_j_start = pid != 0 ? 1 : 0;
        int local_j_end = pid != global_grid_.subgrid_count - 1 ? ny - 1 : ny;

        for (int i = 1; i < nx - 1; ++i) { // skip X-ghost columns
            int global_i = i - 1;
            if (global_i < 0 || global_i >= global_grid_.GetNx()) continue;
            for (int j = local_j_start; j < local_j_end; ++j) {
                int global_j = global_i_start_ + j - 1;
                if (global_j >= 0 && global_j < global_grid_.GetNy()) {
                    global_grid_.H(global_i, global_j) = local_grid_.H(i, j);
                    global_grid_.u(global_i, global_j) = local_grid_.u(i, j);
                    global_grid_.v(global_i, global_j) = local_grid_.v(i, j);
                    global_grid_.eta(global_i, global_j) = local_grid_.eta(i, j);
                }
            }
        }
    }
}

void SubGrid::UpdateEtaMaxIfNeeded(int t) {
#ifdef SAVE_ETA_MAX_MAP
    if ((t + 1) % spec_step_ != 0) return;

    const int nx = local_nx_;
    const int ny = local_ny_;
    const Real grnd = min_calc_value;
    Real* __restrict eta_data = local_grid_.GetEta();
    Real* __restrict u_data = local_grid_.GetU();
    Real* __restrict v_data = local_grid_.GetV();
    Real* __restrict D_data = local_grid_.GetD();

    // ВАЖНО: используем eta напрямую, а НЕ H-D, потому что на граничных
    // ячейках H копируется с соседней строки (другая D), а eta копируется корректно.
    // eta_max хранит max положительный eta (согласуется с исторической семантикой).
    // u_max, v_max хранят max|u|, max|v| (физически осмысленно для Coriolis-анализа).
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        #pragma omp simd
        for (int j = 0; j < ny; ++j) {
            const int ij = IDX(i, j, ny);
            if (D_data[ij] < grnd) continue;
            const Real eta = eta_data[ij];
            Real& m = eta_max_[ij];
            if (eta > m) m = eta;
            const Real au = std::fabs(u_data[ij]);
            Real& mu = u_max_[ij];
            if (au > mu) mu = au;
            const Real av = std::fabs(v_data[ij]);
            Real& mv = v_max_[ij];
            if (av > mv) mv = av;
        }
    }
#else
    (void)t;
#endif
}

#ifdef SAVE_ETA_MAX_MAP
// Общий обход владеемых ячеек с учётом направления декомпозиции.
// Для каждой owned-ячейки вызывает body(local_i, local_j, global_i, global_j).
template <typename Body>
static inline void IterateOwnedCells(
    int local_nx, int local_ny,
    int global_nx, int global_ny,
    int global_i_start,
    DecompositionDirection decomp_dir,
    bool is_first, bool is_last,
    Body&& body)
{
    if (decomp_dir == DecompositionDirection::Horizontal) {
        // Пропускаем ghost columns: i=0 слева (если не первый), i=nx-1 справа (если не последний).
        int li_start = is_first ? 0 : 1;
        int li_end = is_last ? local_nx : local_nx - 1;
        for (int li = li_start; li < li_end; ++li) {
            int gi = global_i_start + li - 1;
            if (gi < 0 || gi >= global_nx) continue;
            for (int lj = 0; lj < local_ny; ++lj) {
                int gj = lj;
                if (gj < 0 || gj >= global_ny) continue;
                body(li, lj, gi, gj);
            }
        }
    } else {
        // Vertical: X-ghost по i=[0,nx-1], Y-ghost по j=0 (если не первый) и j=ny-1 (если не последний).
        int lj_start = is_first ? 0 : 1;
        int lj_end = is_last ? local_ny : local_ny - 1;
        for (int li = 1; li < local_nx - 1; ++li) {
            int gi = li - 1;
            if (gi < 0 || gi >= global_nx) continue;
            for (int lj = lj_start; lj < lj_end; ++lj) {
                int gj = global_i_start + lj - 1;
                if (gj < 0 || gj >= global_ny) continue;
                body(li, lj, gi, gj);
            }
        }
    }
}

void SubGrid::FlushEtaMaxToGlobal() {
    const int nx = local_nx_;
    const int ny = local_ny_;
    const int gnx = global_grid_.GetNx();
    const int gny = global_grid_.GetNy();
    const bool is_first = (decomp_dir_ == DecompositionDirection::Horizontal)
        ? (neighbor_left_ == nullptr)
        : (process_id_ == 0);
    const bool is_last = (decomp_dir_ == DecompositionDirection::Horizontal)
        ? (neighbor_right_ == nullptr)
        : (process_id_ == global_grid_.subgrid_count - 1);

    IterateOwnedCells(nx, ny, gnx, gny, global_i_start_, decomp_dir_,
        is_first, is_last,
        [&](int li, int lj, int gi, int gj) {
            const int ij = IDX(li, lj, ny);
            double v = eta_max_[ij];
            Real& g = global_grid_.eta_max(gi, gj);
            if (v > g) g = v;
        });
}

void SubGrid::FlushUMaxToGlobal() {
    const int nx = local_nx_;
    const int ny = local_ny_;
    const int gnx = global_grid_.GetNx();
    const int gny = global_grid_.GetNy();
    const bool is_first = (decomp_dir_ == DecompositionDirection::Horizontal)
        ? (neighbor_left_ == nullptr)
        : (process_id_ == 0);
    const bool is_last = (decomp_dir_ == DecompositionDirection::Horizontal)
        ? (neighbor_right_ == nullptr)
        : (process_id_ == global_grid_.subgrid_count - 1);

    IterateOwnedCells(nx, ny, gnx, gny, global_i_start_, decomp_dir_,
        is_first, is_last,
        [&](int li, int lj, int gi, int gj) {
            const int ij = IDX(li, lj, ny);
            double v = u_max_[ij];
            Real& g = global_grid_.u_max(gi, gj);
            if (v > g) g = v;
        });
}

void SubGrid::FlushVMaxToGlobal() {
    const int nx = local_nx_;
    const int ny = local_ny_;
    const int gnx = global_grid_.GetNx();
    const int gny = global_grid_.GetNy();
    const bool is_first = (decomp_dir_ == DecompositionDirection::Horizontal)
        ? (neighbor_left_ == nullptr)
        : (process_id_ == 0);
    const bool is_last = (decomp_dir_ == DecompositionDirection::Horizontal)
        ? (neighbor_right_ == nullptr)
        : (process_id_ == global_grid_.subgrid_count - 1);

    IterateOwnedCells(nx, ny, gnx, gny, global_i_start_, decomp_dir_,
        is_first, is_last,
        [&](int li, int lj, int gi, int gj) {
            const int ij = IDX(li, lj, ny);
            double v = v_max_[ij];
            Real& g = global_grid_.v_max(gi, gj);
            if (v > g) g = v;
        });
}

Real* SubGrid::EtaMaxRowPtr(int i) {
    return &eta_max_[IDX(i, 0, local_ny_)];
}
#endif

#undef IDX
