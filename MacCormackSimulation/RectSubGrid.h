#ifndef RECT_SUBGRID_H
#define RECT_SUBGRID_H

#include "Grid.h"
#include "CoordSystem.h"
#include <functional>
#include <memory>
#include <vector>

class NestedStructuredPatch;

class RectSubGrid {
public:
    friend class NestedStructuredPatch;

    RectSubGrid(int patch_id,
                int global_i_start,
                int global_i_end,
                int global_j_start,
                int global_j_end,
                Grid& global_grid,
                double g,
                double tau,
                std::function<double(int, int)> delta_x,
                double delta_y,
                double min_calc_value,
                CoordSystem coord_system);

    void Initialize();

    void ComputeHalfStep();
    void ComputeFullStep();

    void ComputeHalfStepCartesian();
    void ComputeFullStepCartesian();

    void UpdateGlobalGrid(int pid);

    void SetTau(double tau) { tau_ = tau; }
    void ClearFluxCoupling();

    int GetPatchId() const { return patch_id_; }
    int GetGlobalIStart() const { return global_i_start_; }
    int GetGlobalIEnd() const { return global_i_end_; }
    int GetGlobalJStart() const { return global_j_start_; }
    int GetGlobalJEnd() const { return global_j_end_; }

    Real* GetHHat() { return H_hat_.get(); }
    Real* GetUHat() { return u_hat_.get(); }
    Real* GetVHat() { return v_hat_.get(); }
    Real* GetEtaHat() { return eta_hat_.get(); }

    Grid local_grid_;
    int local_nx_ = 0;
    int local_ny_ = 0;

    std::vector<double> flux_left_;
    std::vector<double> flux_right_;
    std::vector<double> flux_top_;
    std::vector<double> flux_bottom_;
    bool use_flux_left_ = false;
    bool use_flux_right_ = false;
    bool use_flux_top_ = false;
    bool use_flux_bottom_ = false;

private:
    int patch_id_;
    int global_i_start_;
    int global_i_end_;
    int global_j_start_;
    int global_j_end_;
    Grid& global_grid_;

    double g_;
    double tau_;
    std::function<double(int, int)> delta_x_;
    double delta_y_;
    double min_calc_value_;
    CoordSystem coord_system_;

    double inv_dy_ = 0.0;
    std::unique_ptr<Real[]> inv_dx_arr_;

    std::unique_ptr<Real[]> H_hat_;
    std::unique_ptr<Real[]> u_hat_;
    std::unique_ptr<Real[]> v_hat_;
    std::unique_ptr<Real[]> eta_hat_;

    int ClampGlobalJ(int global_j) const;
};

#endif // RECT_SUBGRID_H
