#include "RectSubGrid.h"
#include <algorithm>
#include <stdexcept>

#define IDX(i, j, ny) ((i) * (ny) + (j))

RectSubGrid::RectSubGrid(int patch_id,
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
                         CoordSystem coord_system)
    : patch_id_(patch_id),
      global_i_start_(global_i_start),
      global_i_end_(global_i_end),
      global_j_start_(global_j_start),
      global_j_end_(global_j_end),
      global_grid_(global_grid),
      g_(g),
      tau_(tau),
      delta_x_(delta_x),
      delta_y_(delta_y),
      min_calc_value_(min_calc_value),
      coord_system_(coord_system),
      local_grid_(0, 0) {
    local_nx_ = (global_i_end_ - global_i_start_) + 2;
    local_ny_ = (global_j_end_ - global_j_start_) + 2;
    inv_dy_ = (delta_y_ > 0.0) ? (1.0 / delta_y_) : 0.0;

    inv_dx_arr_ = std::make_unique<Real[]>(local_ny_);
    for (int local_j = 0; local_j < local_ny_; ++local_j) {
        const int global_j = ClampGlobalJ(global_j_start_ - 1 + local_j);
        const double dx = delta_x_(0, global_j);
        inv_dx_arr_[local_j] = (dx > 0.0) ? static_cast<Real>(1.0 / dx) : 0.0;
    }

    local_grid_ = Grid(local_nx_, local_ny_);

    const size_t size = static_cast<size_t>(local_nx_) * static_cast<size_t>(local_ny_);
    H_hat_ = std::make_unique<Real[]>(size);
    u_hat_ = std::make_unique<Real[]>(size);
    v_hat_ = std::make_unique<Real[]>(size);
    eta_hat_ = std::make_unique<Real[]>(size);
}

int RectSubGrid::ClampGlobalJ(int global_j) const {
    if (global_grid_.GetNy() <= 0) {
        return 0;
    }
    return std::max(0, std::min(global_j, global_grid_.GetNy() - 1));
}

void RectSubGrid::ClearFluxCoupling() {
    use_flux_left_ = false;
    use_flux_right_ = false;
    use_flux_top_ = false;
    use_flux_bottom_ = false;
    flux_left_.clear();
    flux_right_.clear();
    flux_top_.clear();
    flux_bottom_.clear();
}

void RectSubGrid::Initialize() {
    const int global_nx = global_grid_.GetNx();
    const int global_ny = global_grid_.GetNy();

    for (int local_i = 0; local_i < local_nx_; ++local_i) {
        const int global_i = global_i_start_ - 1 + local_i;
        for (int local_j = 0; local_j < local_ny_; ++local_j) {
            const int global_j = global_j_start_ - 1 + local_j;
            if (global_i >= 0 && global_i < global_nx &&
                global_j >= 0 && global_j < global_ny) {
                local_grid_.H(local_i, local_j) = global_grid_.H(global_i, global_j);
                local_grid_.u(local_i, local_j) = global_grid_.u(global_i, global_j);
                local_grid_.v(local_i, local_j) = global_grid_.v(global_i, global_j);
                local_grid_.eta(local_i, local_j) = global_grid_.eta(global_i, global_j);
                local_grid_.D(local_i, local_j) = global_grid_.D(global_i, global_j);
            } else {
                local_grid_.H(local_i, local_j) = 0.0;
                local_grid_.u(local_i, local_j) = 0.0;
                local_grid_.v(local_i, local_j) = 0.0;
                local_grid_.eta(local_i, local_j) = 0.0;
                local_grid_.D(local_i, local_j) = 0.0;
            }
        }
    }
}

void RectSubGrid::ComputeHalfStep() {
    if (coord_system_ != CoordSystem::Cartesian) {
        throw std::runtime_error("RectSubGrid supports Cartesian coordinates only");
    }
    ComputeHalfStepCartesian();
}

void RectSubGrid::ComputeFullStep() {
    if (coord_system_ != CoordSystem::Cartesian) {
        throw std::runtime_error("RectSubGrid supports Cartesian coordinates only");
    }
    ComputeFullStepCartesian();
}

void RectSubGrid::ComputeHalfStepCartesian() {
    const int nx = local_nx_;
    const int ny = local_ny_;
    const double tau = tau_;
    const double g = g_;
    const double inv_dy = inv_dy_;
    const Real grnd = static_cast<Real>(min_calc_value_);

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

    for (int i = 1; i < nx - 1; ++i) {
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

            const Real inv_dx = inv_dx_data[j];
            const Real Hij = H_data[ij];
            const Real uij = u_data[ij];
            const Real vij = v_data[ij];
            const Real eij = eta_data[ij];
            const Real Dimj = D_data[imj];
            const Real Dijm = D_data[ijm];

            const Real flux_H_x = (Hij * uij - H_data[imj] * u_data[imj]) * inv_dx;
            const Real flux_H_y = (Hij * vij - H_data[ijm] * v_data[ijm]) * inv_dy;

            H_hat_data[ij] = Hij - tau * (flux_H_x + flux_H_y);

            const Real du_dx = (uij - u_data[imj]) * inv_dx;
            const Real du_dy = (uij - u_data[ijm]) * inv_dy;
            const Real dv_dx = (vij - v_data[imj]) * inv_dx;
            const Real dv_dy = (vij - v_data[ijm]) * inv_dy;
            const Real deta_dx = (eij - eta_data[imj]) * inv_dx;
            const Real deta_dy = (eij - eta_data[ijm]) * inv_dy;

            Real u_hat_new = uij - tau * (uij * du_dx + vij * du_dy + g * deta_dx);
            Real v_hat_new = vij - tau * (uij * dv_dx + vij * dv_dy + g * deta_dy);

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

    for (int j = 0; j < ny; ++j) {
        for (int edge_i : {0, nx - 1}) {
            const int ij = IDX(edge_i, j, ny);
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

    for (int i = 1; i < nx - 1; ++i) {
        for (int edge_j : {0, ny - 1}) {
            const int ij = IDX(i, edge_j, ny);
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

    if (use_flux_left_ && !flux_left_.empty()) {
        const int i = 1;
        for (int j = 1; j < ny - 1; ++j) {
            const int ij = IDX(i, j, ny);
            if (D_data[ij] < grnd) {
                continue;
            }

            const double F_H_dg = (static_cast<size_t>(j) * 3 < flux_left_.size())
                ? flux_left_[static_cast<size_t>(j) * 3]
                : 0.0;
            const double flux_H_x = (H_data[ij] * u_data[ij] - F_H_dg) * inv_dx_data[j];
            const double flux_H_y = (H_data[ij] * v_data[ij] - H_data[IDX(i, j - 1, ny)] * v_data[IDX(i, j - 1, ny)]) * inv_dy_;
            H_hat_data[ij] = static_cast<Real>(H_data[ij] - tau_ * (flux_H_x + flux_H_y));
            eta_hat_data[ij] = H_hat_data[ij] - D_data[ij];
        }
    }

    if (use_flux_bottom_ && !flux_bottom_.empty()) {
        const int j = 1;
        const Real inv_dx = inv_dx_data[j];
        for (int i = 1; i < nx - 1; ++i) {
            const int ij = IDX(i, j, ny);
            if (D_data[ij] < grnd) {
                continue;
            }

            const double F_H_dg = (static_cast<size_t>(i) * 3 < flux_bottom_.size())
                ? flux_bottom_[static_cast<size_t>(i) * 3]
                : 0.0;
            const double flux_H_x = (H_data[ij] * u_data[ij] - H_data[IDX(i - 1, j, ny)] * u_data[IDX(i - 1, j, ny)]) * inv_dx;
            const double flux_H_y = (H_data[ij] * v_data[ij] - F_H_dg) * inv_dy_;
            H_hat_data[ij] = static_cast<Real>(H_data[ij] - tau_ * (flux_H_x + flux_H_y));
            eta_hat_data[ij] = H_hat_data[ij] - D_data[ij];
        }
    }
}

void RectSubGrid::ComputeFullStepCartesian() {
    const int nx = local_nx_;
    const int ny = local_ny_;
    const double tau = tau_;
    const double g = g_;
    const double inv_dy = inv_dy_;
    const Real grnd = static_cast<Real>(min_calc_value_);

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

    for (int i = 1; i < nx - 1; ++i) {
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

            const Real inv_dx = inv_dx_data[j];
            const bool at_x_interface = (D_data[imj] < grnd) || (D_data[ipj] < grnd);
            const bool at_y_interface = (D_data[ijm] < grnd) || (D_data[ijp] < grnd);

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

            const Real flux_H_hat_x = (H_hat_ipj * u_hat_ipj - H_hat_ij * u_hat_ij) * inv_dx;
            const Real flux_H_hat_y = (H_hat_ijp * v_hat_ijp - H_hat_ij * v_hat_ij) * inv_dy;
            const Real H_new = static_cast<Real>(0.5 * (H_data[ij] + H_hat_ij) - 0.5 * tau * (flux_H_hat_x + flux_H_hat_y));

            const Real du_hat_dx = (u_hat_ipj - u_hat_ij) * inv_dx;
            const Real du_hat_dy = (u_hat_ijp - u_hat_ij) * inv_dy;
            const Real dv_hat_dx = (v_hat_ipj - v_hat_ij) * inv_dx;
            const Real dv_hat_dy = (v_hat_ijp - v_hat_ij) * inv_dy;
            const Real deta_hat_dx = (eta_hat_ipj - eta_hat_ij) * inv_dx;
            const Real deta_hat_dy = (eta_hat_ijp - eta_hat_ij) * inv_dy;

            Real u_new = static_cast<Real>(0.5 * (u_data[ij] + u_hat_ij) - 0.5 * tau * (u_hat_ij * du_hat_dx + v_hat_ij * du_hat_dy + g * deta_hat_dx));
            Real v_new = static_cast<Real>(0.5 * (v_data[ij] + v_hat_ij) - 0.5 * tau * (u_hat_ij * dv_hat_dx + v_hat_ij * dv_hat_dy + g * deta_hat_dy));

            if (at_x_interface) {
                u_new = 0.0;
            }
            if (at_y_interface) {
                v_new = 0.0;
            }

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

    if (use_flux_right_ && !flux_right_.empty()) {
        const int i = nx - 2;
        for (int j = 1; j < ny - 1; ++j) {
            const int ij = IDX(i, j, ny);
            if (D_data[ij] < grnd) {
                continue;
            }

            const double F_H_dg = (static_cast<size_t>(j) * 3 < flux_right_.size())
                ? flux_right_[static_cast<size_t>(j) * 3]
                : 0.0;
            const double flux_H_hat_x = (F_H_dg - H_hat_data[ij] * u_hat_data[ij]) * inv_dx_data[j];
            const double flux_H_hat_y = (H_hat_data[IDX(i, j + 1, ny)] * v_hat_data[IDX(i, j + 1, ny)] - H_hat_data[ij] * v_hat_data[ij]) * inv_dy_;
            const Real H_new = static_cast<Real>(0.5 * (H_data[ij] + H_hat_data[ij]) - 0.5 * tau_ * (flux_H_hat_x + flux_H_hat_y));

            if (H_new < grnd) {
                H_data[ij] = D_data[ij];
                u_data[ij] = 0.0;
                v_data[ij] = 0.0;
                eta_data[ij] = 0.0;
            } else {
                H_data[ij] = H_new;
                eta_data[ij] = H_new - D_data[ij];
            }
        }
    }

    if (use_flux_top_ && !flux_top_.empty()) {
        const int j = ny - 2;
        const Real inv_dx = inv_dx_data[j];
        for (int i = 1; i < nx - 1; ++i) {
            const int ij = IDX(i, j, ny);
            if (D_data[ij] < grnd) {
                continue;
            }

            const double F_H_dg = (static_cast<size_t>(i) * 3 < flux_top_.size())
                ? flux_top_[static_cast<size_t>(i) * 3]
                : 0.0;
            const double flux_H_hat_x = (H_hat_data[IDX(i + 1, j, ny)] * u_hat_data[IDX(i + 1, j, ny)] - H_hat_data[ij] * u_hat_data[ij]) * inv_dx;
            const double flux_H_hat_y = (F_H_dg - H_hat_data[ij] * v_hat_data[ij]) * inv_dy_;
            const Real H_new = static_cast<Real>(0.5 * (H_data[ij] + H_hat_data[ij]) - 0.5 * tau_ * (flux_H_hat_x + flux_H_hat_y));

            if (H_new < grnd) {
                H_data[ij] = D_data[ij];
                u_data[ij] = 0.0;
                v_data[ij] = 0.0;
                eta_data[ij] = 0.0;
            } else {
                H_data[ij] = H_new;
                eta_data[ij] = H_new - D_data[ij];
            }
        }
    }
}

void RectSubGrid::UpdateGlobalGrid(int pid) {
    (void)pid;
    for (int local_i = 1; local_i < local_nx_ - 1; ++local_i) {
        const int global_i = global_i_start_ + local_i - 1;
        for (int local_j = 1; local_j < local_ny_ - 1; ++local_j) {
            const int global_j = global_j_start_ + local_j - 1;
            global_grid_.H(global_i, global_j) = local_grid_.H(local_i, local_j);
            global_grid_.u(global_i, global_j) = local_grid_.u(local_i, local_j);
            global_grid_.v(global_i, global_j) = local_grid_.v(local_i, local_j);
            global_grid_.eta(global_i, global_j) = local_grid_.eta(local_i, local_j);
        }
    }
}
