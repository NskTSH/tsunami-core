// SubGrid.h
#ifndef SUBGRID_H
#define SUBGRID_H

#include "Grid.h"
#include "CoordSystem.h"
#include <memory>
#include <barrier>
#include <functional>

// Forward declaration для hybrid архитектуры
class StructuredPatch;
namespace fv { class FVSolver; }  // Phase A FV Godunov backend

class SubGrid {
public:
    friend class BoundaryCondition;
    friend class StructuredPatch;  // Для доступа к приватным полям в hybrid архитектуре
    friend class DiskOffloadSimulator;  // Для доступа к hat-массивам при disk I/O
    friend class SingleLineGridSimulation;  // Для sponge layer и interior-метрики
    friend class fv::FVSolver;  // Phase A: FV backend reads g_, tau_, delta_x_/y_, hat arrays

    // Конструктор (coord_system обязателен)
    // global_i_start/global_i_end — диапазон по PRIMARY оси разбивки:
    //   Horizontal → X (i-индексы), Vertical → Y (j-индексы)
    SubGrid(int process_id, int global_i_start, int global_i_end, Grid& global_grid,
        double g, double tau, std::function<double(int, int)> delta_x, double delta_y, double min_calc_value,
        std::shared_ptr<std::barrier<>>& barrier_half_step,
        std::shared_ptr<std::barrier<>>& barrier_full_step,
        int spec_step,
        CoordSystem coord_system,
        double center_latitude_rad = 0.785398163,
        DecompositionDirection decomp_dir = DecompositionDirection::Horizontal);

    // Explicit destructor declared (defined in .cpp where fv::FVSolver
    // is complete) to allow forward-declared unique_ptr<FVSolver> member.
    ~SubGrid();

    void UpdateEtaMaxIfNeeded(int t);
    void Initialize();
    
    // Диспетчерские методы - выбирают реализацию по coord_system_
    void ComputeHalfStep();
    // global_t / use_global_tvd_schedule: active-region path передаёт
    // глобальный timestep, чтобы TVD apply_every использовал global schedule
    // вместо per-subgrid tvd_call_count_ (late-активированные subgrids
    // иначе попадают в другую фазу TVD). Default = legacy поведение.
    void ComputeFullStep(int global_t = -1, bool use_global_tvd_schedule = false);

    // Реализации для декартовой системы координат
    void ComputeHalfStepCartesian();
    void ComputeFullStepCartesian();

    // Реализации для сферической системы координат (пока заглушки)
    void ComputeHalfStepSpherical();
    void ComputeFullStepSpherical();

    // Davis 1987 / Liang 2007 TVD post-corrector for (η, Hu, Hv).
    // Conditional via env TVD_ENABLED=1. Applied inside ComputeFullStep dispatcher.
    // Skipped on every step except every Nth (env TVD_APPLY_EVERY, default 1).
    // global_t/use_global_schedule: active-region path; см. ComputeFullStep.
    void ApplyTvdPostCorrection(int global_t = -1, bool use_global_schedule = false);

    // North absorbing layer (Codex spec 2026-05-27):
    // Rayleigh damping on velocity only (η untouched). sigma(j) = (alpha/dt) * cos^2(pi*j/(2*(W-1)))
    // for j < W, zero otherwise. Default: alpha=0.05, W=96. NOT applied to eta → mass conserved.
    // Applied to (u, v) at end of full-step; (u_hat, v_hat) at end of half-step.
    // Env: ABSORB_ENABLED, ABSORB_LAYER_WIDTH, ABSORB_ALPHA_MAX (=sigma_max*dt).
    void ApplyNorthAbsorbingMain();
    void ApplyNorthAbsorbingHat();

    // Per-subgrid counter for TVD-every-N-steps logic. Incremented inside
    // ApplyTvdPostCorrection (called once per ComputeFullStep per subgrid).
    int tvd_call_count_ = 0;

    // Phase A: FV Godunov backend. Lazily constructed on first FV dispatch
    // (only when MC_SOLVER=fv_godunov). Default null → legacy MacCormack path
    // is untouched and existing tests remain bitwise.
    std::unique_ptr<fv::FVSolver> fv_solver_;

    std::function<void(std::shared_ptr<SubGrid>, double)> ExchangeHatBoundaryData;
    std::function<void(std::shared_ptr<SubGrid>, double)> ExchangeBoundaryData;

    void UpdateGlobalGrid(int pid);
    void SetNeighbors(SubGrid* neighbor_up, SubGrid* neighbor_down);
    void SetSpecStep(int spec_step);
    void SetTau(double tau) { tau_ = tau; }

    // Включить/выключить Кориолис в Spherical-ветке (f_eff = coriolis_enabled_ ? f : 0).
    void SetCoriolisEnabled(bool enabled) { coriolis_enabled_ = enabled; }
    bool GetCoriolisEnabled() const { return coriolis_enabled_; }

    // Flux-based coupling: DISABLED (unit mismatch between DG integrated flux and MC density).
    // Kept as placeholder for future implementation.
    std::vector<double> flux_left_;
    std::vector<double> flux_right_;
    bool use_flux_left_ = false;
    bool use_flux_right_ = false;

#ifdef SAVE_ETA_MAX_MAP
    void FlushEtaMaxToGlobal();
    void FlushUMaxToGlobal();
    void FlushVMaxToGlobal();
    Real* EtaMaxRowPtr(int i);
#endif

    // Публичные члены для BoundaryCondition
    Grid local_grid_;
    int local_nx_;
    int local_ny_;
    SubGrid* neighbor_left_;
    SubGrid* neighbor_right_;

    Real* GetHHat() { return H_hat_.get(); }
    Real* GetUHat() { return u_hat_.get(); }
    Real* GetVHat() { return v_hat_.get(); }
    Real* GetEtaHat() { return eta_hat_.get(); }
    CoordSystem GetCoordSystem() const { return coord_system_; }
    DecompositionDirection GetDecompDir() const { return decomp_dir_; }

    // Primary-axis interval (global cell indices) along the decomposition axis:
    //   Horizontal -> i-interval (x), Vertical -> j-interval (y).
    // Used by ActiveRegionManager (separate file) to avoid friend dependency.
    int GetPrimaryStart() const { return global_i_start_; }
    int GetPrimaryEnd()   const { return global_i_end_; }
    int GetProcessId()    const { return process_id_; }

private:
    int process_id_;
    int global_i_start_;
    int global_i_end_;
    Grid& global_grid_;

    double g_;
    double tau_;
    std::function<double(int, int)> delta_x_;
    double delta_y_;
    double min_calc_value = 0.0;
    CoordSystem coord_system_; // Система координат (обязательно задаётся через конструктор)
    double center_latitude_rad_; // Центральная широта (радианы) для сферической системы
    DecompositionDirection decomp_dir_; // Направление разбивки

    double inv_dy_;  // 1.0 / delta_y_
    std::unique_ptr<Real[]> inv_dx_arr_;  // Precomputed 1/delta_x per j (avoids std::function in inner loop)
    
    std::shared_ptr<std::barrier<>> barrier_half_step_;
    std::shared_ptr<std::barrier<>> barrier_full_step_;

    std::unique_ptr<Real[]> H_hat_;
    std::unique_ptr<Real[]> u_hat_;
    std::unique_ptr<Real[]> v_hat_;
    std::unique_ptr<Real[]> eta_hat_;

#ifdef SAVE_ETA_MAX_MAP
    std::unique_ptr<Real[]> eta_max_;
    std::unique_ptr<Real[]> u_max_;
    std::unique_ptr<Real[]> v_max_;
#endif

    int spec_step_ = 1;
    int step_counter_ = 0;

    // Кориолис-флаг (при false в Spherical-ветке f полагается 0).
    bool coriolis_enabled_ = true;

#ifdef SAVE_ETA_MAX_MAP
    void UpdateLocalEtaMaxIfDue();
#endif
};

#endif // SUBGRID_H