// DGSolver.cpp - DG solver for shallow water equations

#include "DGSolver.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <omp.h>

namespace {
bool UseWellBalancedPositivityLimiter() {
    static int wb_limiter_cache = -1;
    if (wb_limiter_cache < 0) {
        wb_limiter_cache = 1; // default ON
        char* buf = nullptr;
        size_t sz = 0;
        _dupenv_s(&buf, &sz, "DG_WB_LIMITER");
        if (buf && *buf == '0') wb_limiter_cache = 0;
        if (buf) free(buf);
        std::cerr << "[DG] positivity limiter operates on "
                  << (wb_limiter_cache ? "eta=H-D (default)" : "raw H (DG_WB_LIMITER=0)")
                  << "\n";
    }
    return wb_limiter_cache == 1;
}

// Well-balanced PRESSURE reformulation (DG_WB_PRESSURE, default ON). Replaces the
// raw hydrostatic pressure 0.5*g*H^2 in the volume flux and the bed-slope source
// g*H*dD with the DEVIATION pressure 0.5*g*(H^2 - D^2) = g*eta*(D + 0.5*eta) and
// source g*eta*dD, using the SAME D trace (D_coeffs_) at each quad point. At rest
// (eta=H-D=0) every integrand is identically zero BEFORE quadrature, so the
// discrete C-property holds at ANY quadrature degree (measured rest-residual
// 199924 -> ~0). The pure-bed part 0.5*g*D^2 is cancelled by subtracting the same
// per-side trace from the surface flux (see SurfaceIntegral). Codex-reviewed.
bool UseWellBalancedPressure() {
    static int c = -1;
    if (c < 0) {
        c = 1; // default ON
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "DG_WB_PRESSURE");
        if (buf && *buf == '0') c = 0;
        if (buf) free(buf);
        std::cerr << "[DG] pressure formulation: "
                  << (c ? "well-balanced deviation 0.5g(H^2-D^2) (default)"
                        : "raw 0.5gH^2 (DG_WB_PRESSURE=0)") << "\n";
    }
    return c == 1;
}
}

void DGSolver::Initialize(const TriMesh* mesh, const ElementMatrices* elem_matrices) {
    mesh_ = mesh;
    elem_matrices_ = elem_matrices;
    num_elements_ = mesh->NumElements();

    int total = num_elements_ * DG_BLOCK_SIZE;
    q_.resize(total, 0.0);
    rhs_.resize(total, 0.0);
    q_tmp_.resize(total, 0.0);
    q_tmp2_.resize(total, 0.0);
    D_.resize(num_elements_, 0.0);
    D_coeffs_.resize(num_elements_ * DG_BASIS_SIZE, 0.0);

    BuildExpandedMassInverse();
}

void DGSolver::BuildExpandedMassInverse() {
    mass_inv_expanded_.resize(num_elements_);

    for (int k = 0; k < num_elements_; ++k) {
        auto& exp_inv = mass_inv_expanded_[k];
        exp_inv.fill(0.0);

        const auto& M_inv = elem_matrices_->GetElementData(k).M_inv;

        // Block-diagonal: for each variable (H, Hu, Hv), insert M_inv
        for (int var = 0; var < DG_NUM_VARS; ++var) {
            int offset = var * DG_BASIS_SIZE;
            for (int i = 0; i < DG_BASIS_SIZE; ++i) {
                for (int j = 0; j < DG_BASIS_SIZE; ++j) {
                    exp_inv[(offset + i) * DG_BLOCK_SIZE + (offset + j)]
                        = M_inv[i * DG_BASIS_SIZE + j];
                }
            }
        }
    }
}

void DGSolver::SetBathymetry(const std::vector<double>& bathymetry) {
    D_ = bathymetry;
    ProjectBathymetry();
}

void DGSolver::SetBathymetryFunction(std::function<double(double, double)> func) {
    for (int k = 0; k < num_elements_; ++k) {
        const auto& elem = mesh_->GetElement(k);
        D_[k] = func(elem.cx, elem.cy);
    }
    ProjectBathymetryFromFunction(func);
}

void DGSolver::ProjectBathymetryFromFunction(std::function<double(double, double)> func) {
    // L2 project bathymetry D(x,y) onto DG basis using Gauss quadrature
    // Same approach as SetInitialCondition: proj_l = M_inv * ∫ D * phi_l dA

    for (int k = 0; k < num_elements_; ++k) {
        const auto& ed = elem_matrices_->GetElementData(k);
        const auto& elem = mesh_->GetElement(k);

        const auto& p0 = mesh_->GetNode(elem.nodes[0]);
        const auto& p1 = mesh_->GetNode(elem.nodes[1]);
        const auto& p2 = mesh_->GetNode(elem.nodes[2]);

        double det_J = 2.0 * elem.area;
        int dbase = k * DG_BASIS_SIZE;

        double proj[DG_BASIS_SIZE] = {};
        auto quad = gauss::TriOrder4();
        for (const auto& qp : quad) {
            double x, y;
            gauss::MapToPhysical(qp.xi, qp.eta,
                                 p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                                 x, y);

            double D_val = func(x, y);
            double phi[DG_BASIS_SIZE];
            ed.basis.Evaluate(x, y, phi);

            double w = qp.weight * det_J;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                proj[l] += D_val * phi[l] * w;
            }
        }

        // Apply M_inv
        const auto& M_inv = ed.M_inv;
        for (int i = 0; i < DG_BASIS_SIZE; ++i) {
            double coeff = 0.0;
            for (int j = 0; j < DG_BASIS_SIZE; ++j) {
                coeff += M_inv[i * DG_BASIS_SIZE + j] * proj[j];
            }
            D_coeffs_[dbase + i] = coeff;
        }

        // Positivity-limit the projected bathymetry (Codex review 2026-05-31). The P2
        // L2-projection of a steep seafloor OVERSHOOTS to D_pt < 0 on deep cells
        // (measured min -160 m) — unphysical (depth >= 0 in a wet cell). At lake-at-rest
        // H_coeffs == D_coeffs, so H also goes negative there; the H-positivity limiter
        // then floors H >= 0 each step and ADDS mass, breaking the rest state by ~138 m
        // (mode-0 drift, insensitive to mass/momentum damping). Zhang-Shu scale the
        // deviation modes (1..5) by theta so D_pt >= 0 at the volume quad points and the
        // 3 vertices, preserving the cell mean D_0. No-op for flat/linear/gentle D
        // (theta=1) -> dgvalidation and accuracy unaffected. Env DG_BATHY_POS (default ON).
        static int bp_on = -1;
        if (bp_on < 0) { char* b = nullptr; size_t s = 0; _dupenv_s(&b, &s, "DG_BATHY_POS"); bp_on = (b && b[0] == '0') ? 0 : 1; if (b) free(b); }
        if (bp_on) {
            double D0 = D_coeffs_[dbase + 0];
            if (D0 > 0.0) {
                double Dmin = D0;
                for (const auto& qp : quad) {
                    double x, y;
                    gauss::MapToPhysical(qp.xi, qp.eta, p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, x, y);
                    double phi[DG_BASIS_SIZE]; ed.basis.Evaluate(x, y, phi);
                    double Dp = 0.0; for (int l = 0; l < DG_BASIS_SIZE; ++l) Dp += D_coeffs_[dbase + l] * phi[l];
                    if (Dp < Dmin) Dmin = Dp;
                }
                const MeshNode* verts[3] = { &p0, &p1, &p2 };
                for (int v = 0; v < 3; ++v) {
                    double phi[DG_BASIS_SIZE]; ed.basis.Evaluate(verts[v]->x, verts[v]->y, phi);
                    double Dp = 0.0; for (int l = 0; l < DG_BASIS_SIZE; ++l) Dp += D_coeffs_[dbase + l] * phi[l];
                    if (Dp < Dmin) Dmin = Dp;
                }
                // Also the EDGE quadrature points: the surface integral / Audusse and
                // the H-positivity limiter evaluate D there, so D must be >= 0 there too.
                for (int le = 0; le < 3; ++le) {
                    for (int eq = 0; eq < ElementData::EDGE_NPTS; ++eq) {
                        const double* phi = ed.edge_phi[le].data() + eq * DG_BASIS_SIZE;
                        double Dp = 0.0; for (int l = 0; l < DG_BASIS_SIZE; ++l) Dp += D_coeffs_[dbase + l] * phi[l];
                        if (Dp < Dmin) Dmin = Dp;
                    }
                }
                if (Dmin < 0.0) {
                    double theta = D0 / (D0 - Dmin); // in (0,1): pulls min up to 0, mean fixed
                    if (theta < 0.0) theta = 0.0; if (theta > 1.0) theta = 1.0;
                    for (int l = 1; l < DG_BASIS_SIZE; ++l) D_coeffs_[dbase + l] *= theta;
                }
            }
        }
        D_[k] = D_coeffs_[dbase];
    }
}

void DGSolver::ProjectBathymetry() {
    // L2 project bathymetry D(x,y) onto DG basis for each element
    // Uses centroid value D_[k] and Green-Gauss gradient from neighbors
    // For P2: quadratic modes set to zero (D is typically smooth, linear suffices)

    for (int k = 0; k < num_elements_; ++k) {
        const auto& ed = elem_matrices_->GetElementData(k);
        const auto& elem = mesh_->GetElement(k);
        int dbase = k * DG_BASIS_SIZE;

        // Cell average (basis 0 = 1)
        D_coeffs_[dbase] = D_[k];

        // Linear modes via least-squares gradient from neighbors
        double dD_dx = 0.0, dD_dy = 0.0;
        double w_sum = 0.0;
        for (int e = 0; e < 3; ++e) {
            int nbr = elem.neighbors[e];
            if (nbr < 0) continue;
            const auto& nbr_elem = mesh_->GetElement(nbr);
            double dx = nbr_elem.cx - elem.cx;
            double dy = nbr_elem.cy - elem.cy;
            double dist2 = dx * dx + dy * dy;
            if (dist2 < config_.zero_dist_guard) continue;
            double dD = D_[nbr] - D_[k];
            double inv_dist2 = 1.0 / dist2;
            dD_dx += dD * dx * inv_dist2;
            dD_dy += dD * dy * inv_dist2;
            w_sum += 1.0;
        }
        if (w_sum > 0) {
            dD_dx /= w_sum;
            dD_dy /= w_sum;
        }

        // Convert gradient to basis coefficients: coeff_1 = dD/dx * hx, coeff_2 = dD/dy * hy
        D_coeffs_[dbase + 1] = dD_dx * ed.basis.GetHx();
        D_coeffs_[dbase + 2] = dD_dy * ed.basis.GetHy();

        // Higher-order modes (P2) set to zero
        for (int l = 3; l < DG_BASIS_SIZE; ++l) {
            D_coeffs_[dbase + l] = 0.0;
        }
    }
}

void DGSolver::SetInitialCondition(std::function<SWEState(double, double)> func) {
    // Project initial condition onto DG basis using L2 projection
    // For cell averages (basis function 0 = 1): just evaluate at centroid
    // For higher modes: use quadrature

    for (int k = 0; k < num_elements_; ++k) {
        const auto& ed = elem_matrices_->GetElementData(k);
        const auto& elem = mesh_->GetElement(k);

        const auto& p0 = mesh_->GetNode(elem.nodes[0]);
        const auto& p1 = mesh_->GetNode(elem.nodes[1]);
        const auto& p2 = mesh_->GetNode(elem.nodes[2]);

        double det_J = 2.0 * elem.area;

        // Compute projection: q_l = M_inv * integral(q * phi_l)
        // First compute integral(q * phi_l) for each variable
        double proj_H[DG_BASIS_SIZE] = {};
        double proj_Hu[DG_BASIS_SIZE] = {};
        double proj_Hv[DG_BASIS_SIZE] = {};

        auto quad = gauss::TriOrder4();
        for (const auto& qp : quad) {
            double x, y;
            gauss::MapToPhysical(qp.xi, qp.eta,
                                 p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                                 x, y);

            SWEState q = func(x, y);

            double phi[DG_BASIS_SIZE];
            ed.basis.Evaluate(x, y, phi);

            double w = qp.weight * det_J;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                proj_H[l]  += q.H  * phi[l] * w;
                proj_Hu[l] += q.Hu * phi[l] * w;
                proj_Hv[l] += q.Hv * phi[l] * w;
            }
        }

        // Apply M_inv to get coefficients
        const auto& M_inv = ed.M_inv;
        int base = k * DG_BLOCK_SIZE;

        for (int i = 0; i < DG_BASIS_SIZE; ++i) {
            double coeff_H = 0.0, coeff_Hu = 0.0, coeff_Hv = 0.0;
            for (int j = 0; j < DG_BASIS_SIZE; ++j) {
                coeff_H  += M_inv[i * DG_BASIS_SIZE + j] * proj_H[j];
                coeff_Hu += M_inv[i * DG_BASIS_SIZE + j] * proj_Hu[j];
                coeff_Hv += M_inv[i * DG_BASIS_SIZE + j] * proj_Hv[j];
            }
            q_[base + 0 * DG_BASIS_SIZE + i] = coeff_H;
            q_[base + 1 * DG_BASIS_SIZE + i] = coeff_Hu;
            q_[base + 2 * DG_BASIS_SIZE + i] = coeff_Hv;
        }
    }
}

void DGSolver::SetInitialConditionWellBalanced(
    std::function<SWEState(double, double)> q_func,
    std::function<double(double, double)> d_func) {
    // Project eta = H - D and momentum onto the basis, then set
    //   H_coeffs[l] = D_coeffs_[l] + eta_coeffs[l]
    // so the reconstructed H matches the projected bathymetry exactly plus the
    // smooth free surface. Requires D_coeffs_ already filled (call after
    // SetBathymetryFunction). At lake-at-rest eta == 0 -> eta_coeffs == 0 ->
    // H_pt - D_pt == 0 at every quadrature point (well-balanced by construction).
    bool force_lake_rest = false;
    {
        char* lar = nullptr;
        size_t lar_sz = 0;
        _dupenv_s(&lar, &lar_sz, "LAKE_AT_REST");
        force_lake_rest = (lar && *lar == '1');
        if (lar) free(lar);
    }

    for (int k = 0; k < num_elements_; ++k) {
        const auto& ed = elem_matrices_->GetElementData(k);
        const auto& elem = mesh_->GetElement(k);

        const auto& p0 = mesh_->GetNode(elem.nodes[0]);
        const auto& p1 = mesh_->GetNode(elem.nodes[1]);
        const auto& p2 = mesh_->GetNode(elem.nodes[2]);

        double det_J = 2.0 * elem.area;

        double proj_eta[DG_BASIS_SIZE] = {};
        double proj_Hu[DG_BASIS_SIZE]  = {};
        double proj_Hv[DG_BASIS_SIZE]  = {};

        auto quad = gauss::TriOrder4();
        for (const auto& qp : quad) {
            double x, y;
            gauss::MapToPhysical(qp.xi, qp.eta,
                                 p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                                 x, y);

            SWEState q = q_func(x, y);
            double D = d_func(x, y);
            double eta = force_lake_rest ? 0.0 : q.H - D; // free surface relative to bed

            double phi[DG_BASIS_SIZE];
            ed.basis.Evaluate(x, y, phi);

            double w = qp.weight * det_J;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                proj_eta[l] += eta  * phi[l] * w;
                proj_Hu[l]  += q.Hu * phi[l] * w;
                proj_Hv[l]  += q.Hv * phi[l] * w;
            }
        }

        const auto& M_inv = ed.M_inv;
        int base = k * DG_BLOCK_SIZE;
        int dbase = k * DG_BASIS_SIZE;

        for (int i = 0; i < DG_BASIS_SIZE; ++i) {
            double c_eta = 0.0, c_Hu = 0.0, c_Hv = 0.0;
            for (int j = 0; j < DG_BASIS_SIZE; ++j) {
                c_eta += M_inv[i * DG_BASIS_SIZE + j] * proj_eta[j];
                c_Hu  += M_inv[i * DG_BASIS_SIZE + j] * proj_Hu[j];
                c_Hv  += M_inv[i * DG_BASIS_SIZE + j] * proj_Hv[j];
            }
            // H = D + eta in coefficient space -> H_pt - D_pt = eta_pt exactly.
            q_[base + 0 * DG_BASIS_SIZE + i] = D_coeffs_[dbase + i] + c_eta;
            q_[base + 1 * DG_BASIS_SIZE + i] = c_Hu;
            q_[base + 2 * DG_BASIS_SIZE + i] = c_Hv;
        }
    }
}

void DGSolver::SetBoundaryCondition(int boundary_tag, DGBoundaryType type) {
    bc_map_[boundary_tag] = type;
}

void DGSolver::SetDefaultBoundaryCondition(DGBoundaryType type) {
    default_bc_ = type;
}

void DGSolver::SetCoupledSurface(int edge_idx, double eta, double u, double v) {
    coupled_surface_[edge_idx] = CoupledSurface{eta, u, v};
}

void DGSolver::ClearCoupledStates() {
    coupled_surface_.clear();
}

void DGSolver::SetUniformOpenReference(const SWEState& q_far) {
    open_ref_uniform_ = q_far;
    open_ref_uniform_set_ = true;
    use_open_radiation_bc_ = true;
}

void DGSolver::SetOpenReferenceFunction(std::function<SWEState(double, double)> func) {
    if (!mesh_) return;
    open_ref_per_edge_.clear();
    const int num_edges = mesh_->NumEdges();
    for (int e = 0; e < num_edges; ++e) {
        const auto& edge = mesh_->GetEdge(e);
        if (edge.elem_right >= 0) continue; // skip interior edges
        if (GetBoundaryType(e) != DGBoundaryType::Open) continue;
        open_ref_per_edge_[e] = func(edge.mx, edge.my);
    }
    use_open_radiation_bc_ = true;
}

void DGSolver::ClearOpenReference() {
    open_ref_uniform_set_ = false;
    open_ref_per_edge_.clear();
    use_open_radiation_bc_ = false;
}

DGBoundaryType DGSolver::GetBoundaryType(int edge_idx) const {
    int tag = mesh_->GetEdge(edge_idx).boundary_tag;
    auto it = bc_map_.find(tag);
    return (it != bc_map_.end()) ? it->second : default_bc_;
}

double DGSolver::MaxRestImbalance() const {
    // For lake-at-rest (H == D, u = v = 0) the reconstructed values at
    // every volume quadrature point should satisfy H_pt == D_pt to roundoff.
    // Any larger value indicates a projection or basis mismatch between
    // the H modes (stored in q_) and the D modes (stored in D_coeffs_).
    double worst = 0.0;
    for (int k = 0; k < num_elements_; ++k) {
        const auto& ed = elem_matrices_->GetElementData(k);
        int base = k * DG_BLOCK_SIZE;
        int dbase = k * DG_BASIS_SIZE;
        for (int q = 0; q < ElementData::VOL_NPTS; ++q) {
            const double* phi = ed.vol_phi.data() + q * DG_BASIS_SIZE;
            double Hp = 0.0, Dp = 0.0;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                Hp += q_[base + 0 * DG_BASIS_SIZE + l] * phi[l];
                Dp += D_coeffs_[dbase + l] * phi[l];
            }
            double d = std::abs(Hp - Dp);
            if (d > worst) worst = d;
        }
    }
    return worst;
}

double DGSolver::RestResidual(double& mom_only, int& worst_elem,
                              double* worst_wet, int* worst_wet_elem) {
    // Discrete operator residual ||M^{-1} L(q)|| at the current state. At a true
    // discrete steady state (well-balanced lake-at-rest) this is ~roundoff; a
    // nonzero value reveals the scheme does NOT preserve the rest state even when
    // H==D pointwise (MaxRestImbalance==0). Mirrors the M^{-1}*rhs update in Step().
    // worst_wet/worst_wet_elem (optional) report the worst residual over WET
    // elements only (D_[k] >= wet_dry_threshold) — frozen dry-land cells carry a
    // residual that never evolves them, so the wet-only value is what matters.
    ComputeResidual(q_.data());
    double worst = 0.0;
    mom_only = 0.0;
    worst_elem = -1;
    double w_wet = 0.0; int w_wet_e = -1;
    for (int k = 0; k < num_elements_; ++k) {
        const auto& M_inv_exp = mass_inv_expanded_[k];
        int base = k * DG_BLOCK_SIZE;
        double upd[DG_BLOCK_SIZE];
        for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
            double sum = 0.0;
            for (int j = 0; j < DG_BLOCK_SIZE; ++j) {
                sum += M_inv_exp[i * DG_BLOCK_SIZE + j] * rhs_[base + j];
            }
            upd[i] = sum;
        }
        double n2 = 0.0, m2 = 0.0;
        for (int i = 0; i < DG_BLOCK_SIZE; ++i) n2 += upd[i] * upd[i];
        // momentum-only = vars 1,2 (Hu,Hv) blocks
        for (int i = DG_BASIS_SIZE; i < 3 * DG_BASIS_SIZE; ++i) m2 += upd[i] * upd[i];
        double n = std::sqrt(n2);
        if (n > worst) { worst = n; worst_elem = k; }
        if (D_[k] >= config_.wet_dry_threshold && n > w_wet) { w_wet = n; w_wet_e = k; }
        double m = std::sqrt(m2);
        if (m > mom_only) mom_only = m;
    }
    if (worst_wet) *worst_wet = w_wet;
    if (worst_wet_elem) *worst_wet_elem = w_wet_e;
    return worst;
}

double DGSolver::MaxModalRestDiff(int& worst_elem) const {
    // Direct modal comparison H_coeffs vs D_coeffs over WET elements only.
    // (Fix A => zero rest residual iff H_coeffs == D_coeffs.)
    double worst = 0.0;
    worst_elem = -1;
    for (int k = 0; k < num_elements_; ++k) {
        if (D_[k] < config_.wet_dry_threshold) continue; // frozen dry land
        int base = k * DG_BLOCK_SIZE;
        int dbase = k * DG_BASIS_SIZE;
        for (int l = 0; l < DG_BASIS_SIZE; ++l) {
            double d = std::abs(q_[base + 0 * DG_BASIS_SIZE + l] - D_coeffs_[dbase + l]);
            if (d > worst) { worst = d; worst_elem = k; }
        }
    }
    return worst;
}

void DGSolver::ModalImbalanceBreakdown(double* out_per_mode) const {
    // Diagnostic: for each basis mode l, the max over WET elements of
    // |H_coeff_l - D_coeff_l|. Reveals WHICH modes carry the lake-at-rest drift
    // (mode 0 mean, 1-2 linear, 3-5 quadratic) — distinguishes a top-mode aliasing
    // instability (3-5) from a linear-mode one (1-2, capped by the slope limiter).
    for (int l = 0; l < DG_BASIS_SIZE; ++l) out_per_mode[l] = 0.0;
    for (int k = 0; k < num_elements_; ++k) {
        if (D_[k] < config_.wet_dry_threshold) continue;
        int base = k * DG_BLOCK_SIZE;
        int dbase = k * DG_BASIS_SIZE;
        for (int l = 0; l < DG_BASIS_SIZE; ++l) {
            double d = std::abs(q_[base + 0 * DG_BASIS_SIZE + l] - D_coeffs_[dbase + l]);
            if (d > out_per_mode[l]) out_per_mode[l] = d;
        }
    }
}

double DGSolver::MinBathymetryTrace(int& worst_elem) const {
    // Diagnostic (Codex 2026-05-31): min over WET elements of the reconstructed
    // bathymetry D_pt at volume AND edge quadrature points. The P2 projection of a
    // steep real seafloor can OVERSHOOT to D_pt < 0 even when the cell-average D is
    // large & positive. Where D_pt < 0, at rest H_pt = D_pt < 0 so H_safe=max(H,0)=0
    // but eta = H_safe - D_pt = -D_pt > 0 -> spurious rest residual while
    // MaxRestImbalance (which compares raw H_pt-D_pt) still reads ~0. A negative
    // value here pinpoints the clipping mismatch as the residual source.
    double worst = 1e300;
    worst_elem = -1;
    for (int k = 0; k < num_elements_; ++k) {
        if (D_[k] < config_.wet_dry_threshold) continue; // frozen dry land
        const auto& ed = elem_matrices_->GetElementData(k);
        int dbase = k * DG_BASIS_SIZE;
        for (int q = 0; q < ElementData::VOL_NPTS; ++q) {
            const double* phi = ed.vol_phi.data() + q * DG_BASIS_SIZE;
            double Dp = 0.0;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) Dp += D_coeffs_[dbase + l] * phi[l];
            if (Dp < worst) { worst = Dp; worst_elem = k; }
        }
        for (int le = 0; le < 3; ++le) {
            for (int q = 0; q < ElementData::EDGE_NPTS; ++q) {
                const double* phi = ed.edge_phi[le].data() + q * DG_BASIS_SIZE;
                double Dp = 0.0;
                for (int l = 0; l < DG_BASIS_SIZE; ++l) Dp += D_coeffs_[dbase + l] * phi[l];
                if (Dp < worst) { worst = Dp; worst_elem = k; }
            }
        }
    }
    return worst;
}

SWEState DGSolver::EvaluateAt(int elem_idx, double x, double y) const {
    assert(elem_idx >= 0 && elem_idx < num_elements_ && "EvaluateAt: elem_idx out of range");
    const auto& ed = elem_matrices_->GetElementData(elem_idx);
    double phi[DG_BASIS_SIZE];
    ed.basis.Evaluate(x, y, phi);

    int base = elem_idx * DG_BLOCK_SIZE;
    SWEState result;
    result.H = result.Hu = result.Hv = 0.0;

    for (int l = 0; l < DG_BASIS_SIZE; ++l) {
        result.H  += q_[base + 0 * DG_BASIS_SIZE + l] * phi[l];
        result.Hu += q_[base + 1 * DG_BASIS_SIZE + l] * phi[l];
        result.Hv += q_[base + 2 * DG_BASIS_SIZE + l] * phi[l];
    }
    return result;
}

SWEState DGSolver::GetCellAverage(int elem_idx) const {
    assert(elem_idx >= 0 && elem_idx < num_elements_ && "GetCellAverage: elem_idx out of range");
    int base = elem_idx * DG_BLOCK_SIZE;
    return SWEState(
        q_[base + 0 * DG_BASIS_SIZE],   // H average (phi_0 = 1)
        q_[base + 1 * DG_BASIS_SIZE],   // Hu average
        q_[base + 2 * DG_BASIS_SIZE]    // Hv average
    );
}

// ========== TIME STEPPING ==========

void DGSolver::Step(double dt) {
    // Forward Euler: q^{n+1} = q^n + dt * M^{-1} * L(q^n)
    ComputeResidual(q_.data());

    // Apply M^{-1} to rhs and update
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        const auto& M_inv_exp = mass_inv_expanded_[k];
        int base = k * DG_BLOCK_SIZE;

        double update[DG_BLOCK_SIZE];
        for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
            double sum = 0.0;
            for (int j = 0; j < DG_BLOCK_SIZE; ++j) {
                sum += M_inv_exp[i * DG_BLOCK_SIZE + j] * rhs_[base + j];
            }
            update[i] = sum;
        }

        for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
            q_[base + i] += dt * update[i];
        }
    }

    ApplySlopeLimiter();
    ApplyPositivityLimiter();
}

void DGSolver::StepRK2(double dt) {
    int total = num_elements_ * DG_BLOCK_SIZE;

    // Save q^n for stage 2
    std::vector<double> q_n(q_.begin(), q_.end());

    // Stage 1: q* = q^n + dt * M^{-1} * L(q^n)
    ComputeResidual(q_.data());

    std::memcpy(q_tmp_.data(), q_.data(), total * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        const auto& M_inv_exp = mass_inv_expanded_[k];
        int base = k * DG_BLOCK_SIZE;

        for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
            double sum = 0.0;
            for (int j = 0; j < DG_BLOCK_SIZE; ++j) {
                sum += M_inv_exp[i * DG_BLOCK_SIZE + j] * rhs_[base + j];
            }
            q_tmp_[base + i] += dt * sum;
        }
    }

    // Apply limiters after stage 1
    std::swap(q_, q_tmp_);
    ApplySlopeLimiter();
    ApplyPositivityLimiter();
    std::swap(q_, q_tmp_);

    // Stage 2: q^{n+1} = 0.5 * (q^n + q* + dt * M^{-1} * L(q*))
    ComputeResidual(q_tmp_.data());

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        const auto& M_inv_exp = mass_inv_expanded_[k];
        int base = k * DG_BLOCK_SIZE;

        for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
            double sum = 0.0;
            for (int j = 0; j < DG_BLOCK_SIZE; ++j) {
                sum += M_inv_exp[i * DG_BLOCK_SIZE + j] * rhs_[base + j];
            }
            q_[base + i] = 0.5 * (q_n[base + i] + q_tmp_[base + i] + dt * sum);
        }
    }

    ApplySlopeLimiter();
    ApplyPositivityLimiter();
}

void DGSolver::StepRK3(double dt) {
    // SSP-RK3 (Shu-Osher):
    // q^(1) = q^n + dt * M^{-1} * L(q^n)
    // q^(2) = 3/4 * q^n + 1/4 * (q^(1) + dt * M^{-1} * L(q^(1)))
    // q^{n+1} = 1/3 * q^n + 2/3 * (q^(2) + dt * M^{-1} * L(q^(2)))

    int total = num_elements_ * DG_BLOCK_SIZE;

    // Save q^n
    std::vector<double> q_n(q_.begin(), q_.end());

    // --- Stage 1: q^(1) = q^n + dt * M^{-1} * L(q^n) ---
    ComputeResidual(q_.data());
    std::memcpy(q_tmp_.data(), q_.data(), total * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        const auto& M_inv_exp = mass_inv_expanded_[k];
        int base = k * DG_BLOCK_SIZE;
        for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
            double sum = 0.0;
            for (int j = 0; j < DG_BLOCK_SIZE; ++j) {
                sum += M_inv_exp[i * DG_BLOCK_SIZE + j] * rhs_[base + j];
            }
            q_tmp_[base + i] += dt * sum;
        }
    }

    // Apply limiters after stage 1 (q_tmp_ holds q^(1))
    std::swap(q_, q_tmp_);
    ApplySlopeLimiter();
    ApplyPositivityLimiter();
    std::swap(q_, q_tmp_);

    // --- Stage 2: q^(2) = 3/4 * q^n + 1/4 * (q^(1) + dt * M^{-1} * L(q^(1))) ---
    ComputeResidual(q_tmp_.data());

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        const auto& M_inv_exp = mass_inv_expanded_[k];
        int base = k * DG_BLOCK_SIZE;
        for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
            double sum = 0.0;
            for (int j = 0; j < DG_BLOCK_SIZE; ++j) {
                sum += M_inv_exp[i * DG_BLOCK_SIZE + j] * rhs_[base + j];
            }
            q_tmp2_[base + i] = 0.75 * q_n[base + i]
                               + 0.25 * (q_tmp_[base + i] + dt * sum);
        }
    }

    // Apply limiters after stage 2 (q_tmp2_ holds q^(2))
    std::swap(q_, q_tmp2_);
    ApplySlopeLimiter();
    ApplyPositivityLimiter();
    std::swap(q_, q_tmp2_);

    // --- Stage 3: q^{n+1} = 1/3 * q^n + 2/3 * (q^(2) + dt * M^{-1} * L(q^(2))) ---
    ComputeResidual(q_tmp2_.data());

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        const auto& M_inv_exp = mass_inv_expanded_[k];
        int base = k * DG_BLOCK_SIZE;
        for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
            double sum = 0.0;
            for (int j = 0; j < DG_BLOCK_SIZE; ++j) {
                sum += M_inv_exp[i * DG_BLOCK_SIZE + j] * rhs_[base + j];
            }
            q_[base + i] = (1.0 / 3.0) * q_n[base + i]
                         + (2.0 / 3.0) * (q_tmp2_[base + i] + dt * sum);
        }
    }

    ApplySlopeLimiter();
    ApplyPositivityLimiter();
    ApplyModalFilter();
    ApplySteepMomentumDamping(dt);
}

// ========== RESIDUAL COMPUTATION ==========

void DGSolver::ComputeResidual(const double* q_in) {
    int total = num_elements_ * DG_BLOCK_SIZE;
    std::fill(rhs_.begin(), rhs_.end(), 0.0);
    coupled_fluxes_.clear();

    // Volume integrals (parallelized by element — no shared writes)
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        VolumeIntegral(k, q_in, rhs_.data());
    }

    // Surface integrals: edge-based loop (D2 optimization)
    // Each edge processed ONCE — no redundant Riemann solves for interior edges
    SurfaceIntegralEdgeBased(q_in, rhs_.data());

    // Source terms (parallelized by element)
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        SourceTerm(k, q_in, rhs_.data());
    }
}

void DGSolver::VolumeIntegral(int elem_idx, const double* q_in, double* rhs_out) {
    // Skip dry elements entirely — no flux computation needed
    double H_avg = q_in[elem_idx * DG_BLOCK_SIZE + 0 * DG_BASIS_SIZE];
    if (H_avg < config_.wet_dry_threshold && D_[elem_idx] < config_.wet_dry_threshold) return;

    const auto& ed = elem_matrices_->GetElementData(elem_idx);
    int base = elem_idx * DG_BLOCK_SIZE;
    int dbase = elem_idx * DG_BASIS_SIZE;
    const bool wb_p = UseWellBalancedPressure();

    for (int q = 0; q < ElementData::VOL_NPTS; ++q) {
        const double* phi = ed.vol_phi.data() + q * DG_BASIS_SIZE;
        const double* dphi_dx = ed.vol_dphi_dx.data() + q * DG_BASIS_SIZE;
        const double* dphi_dy = ed.vol_dphi_dy.data() + q * DG_BASIS_SIZE;
        double w = ed.vol_weights[q];

        // Reconstruct q at this quadrature point using precomputed basis
        double H = 0.0, Hu = 0.0, Hv = 0.0, D_pt = 0.0;
        for (int l = 0; l < DG_BASIS_SIZE; ++l) {
            H  += q_in[base + 0 * DG_BASIS_SIZE + l] * phi[l];
            Hu += q_in[base + 1 * DG_BASIS_SIZE + l] * phi[l];
            Hv += q_in[base + 2 * DG_BASIS_SIZE + l] * phi[l];
            D_pt += D_coeffs_[dbase + l] * phi[l];
        }

        double H_safe = std::max(H, 0.0);
        double u_vel = SafeVelCapped(H_safe, Hu);
        double v_vel = SafeVelCapped(H_safe, Hv);

        // Hydrostatic pressure. Well-balanced (default): the DEVIATION pressure
        //   P = 0.5*g*(H^2 - D^2) = g*eta*(D + 0.5*eta),  eta = H - D
        // computed in the cancellation-safe product form (Codex Q4) to avoid
        // catastrophic cancellation when eta << D. The dropped 0.5*g*D^2 part is
        // restored as a pure surface term (per-side trace) in SurfaceIntegral.
        // Legacy (DG_WB_PRESSURE=0): raw 0.5*g*H^2.
        double pressure;
        if (wb_p) {
            // Clamp negative P2 bathymetry overshoot: the L2 projection of a steep
            // seafloor can give D_pt < 0 at a quad point even on a deep wet cell
            // (measured min D_pt = -160 m). At rest H = D_pt < 0 => H_safe = 0, so
            // eta = H_safe - D_pt would be +|D_pt| and inject spurious pressure.
            // Using D_c = max(D_pt,0) consistently with H_safe = max(H,0) keeps
            // eta == 0 at rest (Codex review 2026-05-31). No-op where D_pt >= 0.
            double D_c = std::max(D_pt, 0.0);
            double eta = H_safe - D_c;
            pressure = SWE_G * eta * (D_c + 0.5 * eta);
        } else {
            pressure = 0.5 * SWE_G * H_safe * H_safe;
        }

        // Physical fluxes F = (Hu, Hu^2 + P, Huv), G = (Hv, Huv, Hv^2 + P)
        double F_H  = H_safe * u_vel;
        double F_Hu = H_safe * u_vel * u_vel + pressure;
        double F_Hv = H_safe * u_vel * v_vel;

        double G_H  = H_safe * v_vel;
        double G_Hu = H_safe * v_vel * u_vel;
        double G_Hv = H_safe * v_vel * v_vel + pressure;

        // Volume integral: + integral(F * dphi/dx + G * dphi/dy) dA
        for (int l = 0; l < DG_BASIS_SIZE; ++l) {
            rhs_out[base + 0 * DG_BASIS_SIZE + l] += (F_H  * dphi_dx[l] + G_H  * dphi_dy[l]) * w;
            rhs_out[base + 1 * DG_BASIS_SIZE + l] += (F_Hu * dphi_dx[l] + G_Hu * dphi_dy[l]) * w;
            rhs_out[base + 2 * DG_BASIS_SIZE + l] += (F_Hv * dphi_dx[l] + G_Hv * dphi_dy[l]) * w;
        }
    }
}

// Helper: evaluate SWE state from precomputed basis values at a quadrature point
static SWEState EvalFromBasis(int base, const double* q_in, const double* phi) {
    SWEState result;
    result.H = result.Hu = result.Hv = 0.0;
    for (int l = 0; l < DG_BASIS_SIZE; ++l) {
        result.H  += q_in[base + 0 * DG_BASIS_SIZE + l] * phi[l];
        result.Hu += q_in[base + 1 * DG_BASIS_SIZE + l] * phi[l];
        result.Hv += q_in[base + 2 * DG_BASIS_SIZE + l] * phi[l];
    }
    return result;
}

void DGSolver::SurfaceIntegralElement(int elem_idx, const double* q_in, double* rhs_out) {
    // Element-based surface integral: process all 3 edges of this element.
    // Each element only writes to its own rhs block — no race conditions.
    // Interior edges are computed by both adjacent elements (redundant but parallel-safe).

    const auto& elem = mesh_->GetElement(elem_idx);
    const auto& ed = elem_matrices_->GetElementData(elem_idx);
    int base = elem_idx * DG_BLOCK_SIZE;

    for (int local_edge = 0; local_edge < 3; ++local_edge) {
        int edge_idx = elem.edges[local_edge];
        const auto& edge = mesh_->GetEdge(edge_idx);

        int eL = edge.elem_left;
        int eR = edge.elem_right;

        // Determine if this element is left or right of the edge
        bool is_left = (eL == elem_idx);
        int other = is_left ? eR : eL;

        // Normal always points left→right; flip sign if we are the right element
        double sign = is_left ? -1.0 : 1.0;
        double nx = edge.normal_x;
        double ny = edge.normal_y;

        for (int q = 0; q < ElementData::EDGE_NPTS; ++q) {
            double x = ed.edge_quad[local_edge][q * 3 + 0];
            double y = ed.edge_quad[local_edge][q * 3 + 1];
            double w = ed.edge_quad[local_edge][q * 3 + 2];

            // Evaluate this element's state at edge quadrature point
            SWEState q_this = EvalFromBasis(base, q_in,
                ed.edge_phi[local_edge].data() + q * DG_BASIS_SIZE);

            // Evaluate the other side's state
            SWEState q_other;
            double D_this_pt = 0.0, D_other_pt = 0.0;
            if (other >= 0) {
                // Check if neighbor is permanently dry (D < threshold)
                bool other_permanently_dry = (D_[other] < config_.wet_dry_threshold);
                bool this_permanently_dry = (D_[elem_idx] < config_.wet_dry_threshold);
                if (other_permanently_dry || this_permanently_dry) {
                    // Zero-gradient BC: exterior = interior with zero velocity
                    // Analogous to MacCormack at_x/y_interface → u_new = 0
                    q_other = q_this;
                    q_other.Hu = 0.0;
                    q_other.Hv = 0.0;
                    D_this_pt = D_[elem_idx];
                    D_other_pt = D_[other];
                } else {
                const auto& ed_other = elem_matrices_->GetElementData(other);
                double phi_other[DG_BASIS_SIZE];
                ed_other.basis.Evaluate(x, y, phi_other);
                q_other = EvalFromBasis(other * DG_BLOCK_SIZE, q_in, phi_other);

                // Evaluate bathymetry from both sides for hydrostatic reconstruction
                const double* phi_me_d = ed.edge_phi[local_edge].data() + q * DG_BASIS_SIZE;
                int dbase_this = elem_idx * DG_BASIS_SIZE;
                int dbase_other = other * DG_BASIS_SIZE;
                for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                    D_this_pt  += D_coeffs_[dbase_this + l] * phi_me_d[l];
                    D_other_pt += D_coeffs_[dbase_other + l] * phi_other[l];
                }
                } // end else (wet neighbor)
            } else {
                // Boundary edge (only the left element owns boundary edges)
                DGBoundaryType bc = GetBoundaryType(edge_idx);
                if (bc == DGBoundaryType::Coupled) {
                    auto it = coupled_surface_.find(edge_idx);
                    if (it != coupled_surface_.end()) {
                        // Well-balanced coupled ghost: reconstruct H at THIS edge
                        // quadrature point from the exchanged free surface eta and the
                        // DG element's OWN bathymetry. eta is continuous across the seam;
                        // depth is local, so at lake-at-rest (eta=0) q_other == q_this and
                        // the HLL flux cancels the volume integral exactly.
                        const double* phi_d = ed.edge_phi[local_edge].data() + q * DG_BASIS_SIZE;
                        int dbase = elem_idx * DG_BASIS_SIZE;
                        double D_q = 0.0;
                        for (int l = 0; l < DG_BASIS_SIZE; ++l) D_q += D_coeffs_[dbase + l] * phi_d[l];
                        double H_g = std::max(0.0, it->second.eta + D_q);
                        q_other.H  = H_g;
                        q_other.Hu = H_g * it->second.u;
                        q_other.Hv = H_g * it->second.v;
                    } else {
                        RiemannSolver::OpenBC(q_this, nx, ny, q_other);
                    }
                } else if (bc == DGBoundaryType::Wall) {
                    RiemannSolver::WallBC(q_this, nx, ny, q_other);
                } else {
                    // Open BC: use characteristic / Flather radiation when a
                    // far-field reference is configured, else legacy copy.
                    if (use_open_radiation_bc_) {
                        SWEState q_far = open_ref_uniform_;
                        auto it = open_ref_per_edge_.find(edge_idx);
                        if (it != open_ref_per_edge_.end()) {
                            q_far = it->second;
                        } else if (!open_ref_uniform_set_) {
                            q_far = q_this; // safe fallback
                        }
                        RiemannSolver::OpenBCRadiation(q_this, q_far, nx, ny, q_other);
                    } else {
                        RiemannSolver::OpenBC(q_this, nx, ny, q_other);
                    }
                }
                D_this_pt = D_[elem_idx];
                D_other_pt = D_this_pt; // boundary: same bathymetry
            }

            // --- Hydrostatic reconstruction (Audusse et al. 2004) ---
            // POSITIVE-DEPTH convention used in this codebase: D >= 0 is water depth,
            // free surface eta = H - D (lake-at-rest <=> H == D).
            // For this convention, the reconstructed depths must use D_min, not D_max:
            //   D_min  = min(D_L, D_R)
            //   H_L*   = max(0, H_L - D_L + D_min)
            //   H_R*   = max(0, H_R - D_R + D_min)
            // At rest (H==D), H_L*==H_R*==D_min, so the HLL flux is exactly zero.
            // Using D_star=max(D_L,D_R) with H+D-D_star would correspond to bed-elevation
            // convention (eta = H + D) and produces a fictitious dam-break across every
            // bathymetry jump on lake-at-rest with steep slopes (Japan Sea S7 blow-up).
            SWEState q_this_flux = q_this;
            SWEState q_other_flux = q_other;

            if (config_.well_balanced && other >= 0) {
                double D_min = std::min(D_this_pt, D_other_pt);

                double H_this_star = std::max(0.0, q_this.H - D_this_pt + D_min);
                double H_other_star = std::max(0.0, q_other.H - D_other_pt + D_min);

                // Scale momentum to preserve velocity: Hu_star = H_star/H * Hu
                double r_this = (q_this.H > SWE_H_DRY) ? H_this_star / q_this.H : 0.0;
                double r_other = (q_other.H > SWE_H_DRY) ? H_other_star / q_other.H : 0.0;

                q_this_flux.H = H_this_star;
                q_this_flux.Hu = r_this * q_this.Hu;
                q_this_flux.Hv = r_this * q_this.Hv;

                q_other_flux.H = H_other_star;
                q_other_flux.Hu = r_other * q_other.Hu;
                q_other_flux.Hv = r_other * q_other.Hv;
            }

            // Compute numerical flux with consistent left/right ordering
            SWEState qL_flux = is_left ? q_this_flux : q_other_flux;
            SWEState qR_flux = is_left ? q_other_flux : q_this_flux;

            SWEState num_flux;
            RiemannSolver::ComputeFlux(qL_flux, qR_flux, nx, ny, num_flux);

            // Hydrostatic pressure correction (Audusse et al. 2004):
            // Each element K adds g/2*(H_K² - H_K*²) to its edge flux (before sign).
            // This ensures that for lake-at-rest (η=const, u=v=0), the surface integral
            // contribution equals ∮ gH_K²/2 * n ds, matching the volume integral exactly.
            if (config_.well_balanced && other >= 0) {
                double H_orig = q_this.H;
                double H_star = q_this_flux.H;
                double dP = 0.5 * SWE_G * (H_orig * H_orig - H_star * H_star);
                num_flux.Hu += dP * nx;
                num_flux.Hv += dP * ny;
            }

            // Store coupled boundary flux for flux-based coupling
            if (other < 0 && GetBoundaryType(edge_idx) == DGBoundaryType::Coupled) {
                // Accumulate flux * weight for this quadrature point
                // Total flux = sum over quad points of (num_flux * edge_weight)
                auto& cf = coupled_fluxes_[edge_idx];
                if (q == 0) { cf.H = 0; cf.Hu = 0; cf.Hv = 0; }
                cf.H  += num_flux.H  * w;
                cf.Hu += num_flux.Hu * w;
                cf.Hv += num_flux.Hv * w;
            }

            // Accumulate: left gets -flux*phi*w, right gets +flux*phi*w
            const double* phi_me = ed.edge_phi[local_edge].data() + q * DG_BASIS_SIZE;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                rhs_out[base + 0 * DG_BASIS_SIZE + l] += sign * num_flux.H  * phi_me[l] * w;
                rhs_out[base + 1 * DG_BASIS_SIZE + l] += sign * num_flux.Hu * phi_me[l] * w;
                rhs_out[base + 2 * DG_BASIS_SIZE + l] += sign * num_flux.Hv * phi_me[l] * w;
            }
        }
    }
}

// Helper: find which local edge index (0,1,2) of an element corresponds to global edge_idx
static int FindLocalEdge(const TriElement& elem, int edge_idx) {
    for (int le = 0; le < 3; ++le) {
        if (elem.edges[le] == edge_idx) return le;
    }
    return -1; // should never happen for valid meshes
}

void DGSolver::SurfaceIntegralEdgeBased(const double* q_in, double* rhs_out) {
    // D2 optimization: iterate over each edge ONCE.
    // Interior edges: compute Riemann flux once, accumulate into both adjacent elements.
    // Boundary edges: compute flux and accumulate into the owning element only.
    // This eliminates ~50% of Riemann solves compared to the element-based approach.

    const int num_edges = mesh_->NumEdges();

    for (int edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
        const auto& edge = mesh_->GetEdge(edge_idx);
        const int eL = edge.elem_left;
        const int eR = edge.elem_right;

        // Find local edge indices within each element
        const auto& elem_L = mesh_->GetElement(eL);
        const int local_edge_L = FindLocalEdge(elem_L, edge_idx);
        const auto& ed_L = elem_matrices_->GetElementData(eL);
        const int base_L = eL * DG_BLOCK_SIZE;

        const double nx = edge.normal_x;
        const double ny = edge.normal_y;

        // Preload right element data if interior edge
        const ElementData* ed_R_ptr = nullptr;
        int base_R = 0;
        int local_edge_R = -1;
        if (eR >= 0) {
            ed_R_ptr = &elem_matrices_->GetElementData(eR);
            base_R = eR * DG_BLOCK_SIZE;
            local_edge_R = FindLocalEdge(mesh_->GetElement(eR), edge_idx);
        }

        for (int q = 0; q < ElementData::EDGE_NPTS; ++q) {
            const double x = ed_L.edge_quad[local_edge_L][q * 3 + 0];
            const double y = ed_L.edge_quad[local_edge_L][q * 3 + 1];
            const double w = ed_L.edge_quad[local_edge_L][q * 3 + 2];

            // Evaluate left state using precomputed basis
            const double* phi_L = ed_L.edge_phi[local_edge_L].data() + q * DG_BASIS_SIZE;
            SWEState qL = EvalFromBasis(base_L, q_in, phi_L);

            // Evaluate right state
            SWEState qR;
            double D_L_pt = 0.0, D_R_pt = 0.0;
            double phi_R_buf[DG_BASIS_SIZE] = {};

            if (eR >= 0) {
                // Edge-trace bathymetry from D_coeffs_ for BOTH sides, evaluated at
                // the SAME quadrature point as the state. q.H is always an edge trace,
                // so the matching depth MUST be the edge trace too — never the cell
                // average D_[e]. Mixing edge-trace H with cell-average D makes
                // eta = H - D nonzero at rest and breaks the deviation-pressure
                // C-property at the coastline (Codex review 2026-05-31).
                ed_R_ptr->basis.Evaluate(x, y, phi_R_buf);
                const int dbase_L = eL * DG_BASIS_SIZE;
                const int dbase_R = eR * DG_BASIS_SIZE;
                double D_L_trace = 0.0, D_R_trace = 0.0;
                for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                    D_L_trace += D_coeffs_[dbase_L + l] * phi_L[l];
                    D_R_trace += D_coeffs_[dbase_R + l] * phi_R_buf[l];
                }

                // Check permanently dry neighbors
                bool L_perm_dry = (D_[eL] < config_.wet_dry_threshold);
                bool R_perm_dry = (D_[eR] < config_.wet_dry_threshold);
                if (L_perm_dry || R_perm_dry) {
                    // Zero-gradient on the FREE SURFACE (eta), zero velocity at the
                    // dry interface. The ghost copies the wet side's H, so its depth
                    // MUST equal the wet side's edge-trace depth (Codex rule: if
                    // q_G.H = q_K.H then D_G = D_K) — otherwise eta_ghost != 0 at rest.
                    if (!L_perm_dry) {                 // L wet, R dry
                        qR = qL; qR.Hu = 0.0; qR.Hv = 0.0;
                        D_L_pt = D_L_trace; D_R_pt = D_L_trace;
                    } else if (!R_perm_dry) {          // L dry, R wet
                        qR = EvalFromBasis(base_R, q_in, phi_R_buf);
                        qL = qR; qL.Hu = 0.0; qL.Hv = 0.0;
                        D_L_pt = D_R_trace; D_R_pt = D_R_trace;
                    } else {                           // both dry
                        qL = {SWE_H_DRY, 0, 0}; qR = {SWE_H_DRY, 0, 0};
                        D_L_pt = D_L_trace; D_R_pt = D_R_trace;
                    }
                } else {
                    // Interior wet/wet edge: right state + its own edge-trace depth.
                    qR = EvalFromBasis(base_R, q_in, phi_R_buf);
                    D_L_pt = D_L_trace; D_R_pt = D_R_trace;
                }
            } else {
                // Boundary edge: apply BC
                DGBoundaryType bc = GetBoundaryType(edge_idx);
                if (bc == DGBoundaryType::Coupled) {
                    auto it = coupled_surface_.find(edge_idx);
                    if (it != coupled_surface_.end()) {
                        // Well-balanced coupled ghost: reconstruct H at THIS edge
                        // quadrature point from the exchanged free surface eta and the DG
                        // element's OWN bathymetry (eta continuous, depth local). At rest
                        // (eta=0) qR == qL so the HLL flux cancels the volume integral.
                        const int dbase_L = eL * DG_BASIS_SIZE;
                        double D_q = 0.0;
                        for (int l = 0; l < DG_BASIS_SIZE; ++l) D_q += D_coeffs_[dbase_L + l] * phi_L[l];
                        double H_g = std::max(0.0, it->second.eta + D_q);
                        qR.H  = H_g;
                        qR.Hu = H_g * it->second.u;
                        qR.Hv = H_g * it->second.v;
                    } else {
                        RiemannSolver::OpenBC(qL, nx, ny, qR);
                    }
                } else if (bc == DGBoundaryType::Wall) {
                    RiemannSolver::WallBC(qL, nx, ny, qR);
                } else {
                    // Open BC: characteristic / Flather radiation when ref set
                    if (use_open_radiation_bc_) {
                        SWEState q_far = open_ref_uniform_;
                        auto it = open_ref_per_edge_.find(edge_idx);
                        if (it != open_ref_per_edge_.end()) {
                            q_far = it->second;
                        } else if (!open_ref_uniform_set_) {
                            q_far = qL;
                        }
                        RiemannSolver::OpenBCRadiation(qL, q_far, nx, ny, qR);
                    } else {
                        RiemannSolver::OpenBC(qL, nx, ny, qR);
                    }
                }
                // Use the DG edge TRACE of D (not the centroid D_[eL]) so the
                // boundary edge is consistent with the deviation-pressure volume
                // term — required for the discrete C-property at boundary cells.
                {
                    double Db = 0.0;
                    const int dbase_L = eL * DG_BASIS_SIZE;
                    for (int l = 0; l < DG_BASIS_SIZE; ++l) Db += D_coeffs_[dbase_L + l] * phi_L[l];
                    D_L_pt = Db;
                    D_R_pt = Db;
                }
            }

            // Clamp negative P2 bathymetry overshoot (and the matching H trace) so
            // eta = H - D is exactly 0 at rest even where the steep-seafloor P2
            // projection gives an edge-trace D_pt < 0 (measured min -160 m).
            // Consistent with the VolumeIntegral/SourceTerm D_c clamp; no-op where
            // D_pt >= 0 (Codex review 2026-05-31). Keeps the deviation surface flux
            // (dP) and Audusse H* from injecting spurious momentum at steep wet cells.
            D_L_pt = std::max(D_L_pt, 0.0);
            D_R_pt = std::max(D_R_pt, 0.0);
            qL.H = std::max(qL.H, 0.0);
            qR.H = std::max(qR.H, 0.0);

            // --- Hydrostatic reconstruction (Audusse et al. 2004) ---
            // Positive-depth convention: see longer comment in SurfaceIntegralElement.
            SWEState qL_flux = qL;
            SWEState qR_flux = qR;

            if (config_.well_balanced && eR >= 0) {
                double D_min = std::min(D_L_pt, D_R_pt);

                double H_L_star = std::max(0.0, qL.H - D_L_pt + D_min);
                double H_R_star = std::max(0.0, qR.H - D_R_pt + D_min);

                double r_L = (qL.H > SWE_H_DRY) ? H_L_star / qL.H : 0.0;
                double r_R = (qR.H > SWE_H_DRY) ? H_R_star / qR.H : 0.0;

                qL_flux.H  = H_L_star;
                qL_flux.Hu = r_L * qL.Hu;
                qL_flux.Hv = r_L * qL.Hv;

                qR_flux.H  = H_R_star;
                qR_flux.Hu = r_R * qR.Hu;
                qR_flux.Hv = r_R * qR.Hv;
            }

            // Compute Riemann flux ONCE (same result both elements would get)
            SWEState num_flux;
            RiemannSolver::ComputeFlux(qL_flux, qR_flux, nx, ny, num_flux);

            // Conservative well-balanced MASS-FLUX VISCOSITY (Codex review 2026-05-31).
            // The lake-at-rest fixed point is EXACT (residual 1.5e-8) but linearly
            // UNSTABLE on steep deep bathymetry: the mode-0 loop eta_0 -> bed source
            // g*eta_0*gradD -> Hu_0 -> mass flux -> eta_0 has gain ~|gradD|*h/H that
            // exceeds the HLL eta-jump damping, so the cell-average (mass) mode grows
            // ~8%/step (drift to 138 m, capped by the limiter). Add a conservative
            // diffusive flux on the FREE-SURFACE jump [eta]=eta_L-eta_R applied to the
            // SHARED mass flux (equal-and-opposite -> mass-conservative). At rest /
            // any constant free surface eta_L==eta_R => strict NO-OP, so Fix A
            // well-balancedness and dgvalidation are preserved. Damps the eta_0 loop.
            static int mv_on = -1; static double mv_theta = 0.5;
            if (mv_on < 0) {
                char* b = nullptr; size_t s = 0;
                _dupenv_s(&b, &s, "DG_MASS_VISC"); mv_on = (b && b[0] == '1') ? 1 : 0; if (b) free(b); // default OFF: theta=0.5 over-diffused (dgval NaN) & worsened S7; needs proper scaling
                char* bt = nullptr; size_t st = 0;
                _dupenv_s(&bt, &st, "DG_MASS_VISC_THETA");
                if (bt && bt[0]) { double v = atof(bt); if (v >= 0.0) mv_theta = v; } if (bt) free(bt);
                std::cerr << "[DG] mass-flux viscosity " << (mv_on ? "ON" : "OFF")
                          << " theta=" << mv_theta << " (free-surface jump damping)\n";
            }
            if (mv_on && eR >= 0) {
                double etaL_t = qL.H - D_L_pt;   // clamped edge traces; eta = H - D
                double etaR_t = qR.H - D_R_pt;
                double H_bar = 0.5 * (qL.H + qR.H);
                double c_e = std::sqrt(SWE_G * std::max(H_bar, SWE_H_DRY));
                num_flux.H += (mv_theta * c_e) * (etaL_t - etaR_t);
            }

            // Well-balanced DEVIATION surface (Codex Q3, matches the deviation
            // pressure used in VolumeIntegral/SourceTerm). The raw HLL momentum
            // flux carries the bed hydrostatic 0.5*g*D_min^2 at rest; subtract it
            // so the surface delivers only the deviation pressure. Then each side
            // adds its own deviation pressure correction dP_dev,K computed in the
            // cancellation-safe product form 0.5*g*(eta_K*(H_K+D_K) -
            // eta*_K*(H*_K+D_min)). At rest eta=0 => HLL part 0.5gD_min^2 is
            // removed and dP_dev=0 => zero surface momentum (exact C-property).
            const bool wb_p = UseWellBalancedPressure();
            // Bed baseline removed from the raw HLL momentum flux so the surface
            // carries only the deviation pressure. Interior edges: at rest both
            // star states equal D_min, HLL pressure = 0.5 g D_min^2. Boundary edges:
            // no Audusse rebuild, so the raw flux pressure = 0.5 g D_L_pt^2 (Wall/
            // Open reflect/copy keep H=D_L_pt at rest). Subtract the matching value.
            double D_min_e = (eR >= 0) ? std::min(D_L_pt, D_R_pt) : D_L_pt;
            if (wb_p) {
                double pbed = 0.5 * SWE_G * D_min_e * D_min_e;
                num_flux.Hu -= pbed * nx;
                num_flux.Hv -= pbed * ny;
            }

            // Store coupled boundary flux for flux-based coupling (edge-based path)
            if (eR < 0 && GetBoundaryType(edge_idx) == DGBoundaryType::Coupled) {
                auto& cf = coupled_fluxes_[edge_idx];
                if (q == 0) { cf.H = 0; cf.Hu = 0; cf.Hv = 0; }
                cf.H  += num_flux.H  * w;
                cf.Hu += num_flux.Hu * w;
                cf.Hv += num_flux.Hv * w;
            }

            // --- Accumulate into LEFT element (sign = -1) ---
            {
                // Hydrostatic pressure correction for left element
                SWEState flux_L = num_flux;
                if (config_.well_balanced && eR >= 0) {
                    double dP;
                    if (wb_p) {
                        // dP_dev,L = 0.5 g [ (H_L^2 - D_L^2) - (H_L*^2 - D_min^2) ]
                        // in cancellation-safe form: 0.5 g [ etaL*(H_L+D_L) - etaL*_*(H_L*+D_min) ]
                        double etaL = qL.H - D_L_pt;
                        double etaL_s = qL_flux.H - D_min_e;
                        dP = 0.5 * SWE_G * (etaL * (qL.H + D_L_pt) - etaL_s * (qL_flux.H + D_min_e));
                    } else {
                        dP = 0.5 * SWE_G * (qL.H * qL.H - qL_flux.H * qL_flux.H);
                    }
                    flux_L.Hu += dP * nx;
                    flux_L.Hv += dP * ny;
                }
                for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                    rhs_out[base_L + 0 * DG_BASIS_SIZE + l] -= flux_L.H  * phi_L[l] * w;
                    rhs_out[base_L + 1 * DG_BASIS_SIZE + l] -= flux_L.Hu * phi_L[l] * w;
                    rhs_out[base_L + 2 * DG_BASIS_SIZE + l] -= flux_L.Hv * phi_L[l] * w;
                }
            }

            // --- Accumulate into RIGHT element (sign = +1) ---
            if (eR >= 0) {
                // Reuse phi_R_buf from state evaluation above (same point, same basis)
                // Hydrostatic pressure correction for right element
                SWEState flux_R = num_flux;
                if (config_.well_balanced) {
                    double dP;
                    if (wb_p) {
                        double etaR = qR.H - D_R_pt;
                        double etaR_s = qR_flux.H - D_min_e;
                        dP = 0.5 * SWE_G * (etaR * (qR.H + D_R_pt) - etaR_s * (qR_flux.H + D_min_e));
                    } else {
                        dP = 0.5 * SWE_G * (qR.H * qR.H - qR_flux.H * qR_flux.H);
                    }
                    flux_R.Hu += dP * nx;
                    flux_R.Hv += dP * ny;
                }
                for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                    rhs_out[base_R + 0 * DG_BASIS_SIZE + l] += flux_R.H  * phi_R_buf[l] * w;
                    rhs_out[base_R + 1 * DG_BASIS_SIZE + l] += flux_R.Hu * phi_R_buf[l] * w;
                    rhs_out[base_R + 2 * DG_BASIS_SIZE + l] += flux_R.Hv * phi_R_buf[l] * w;
                }
            }
        }
    }
}

void DGSolver::SourceTerm(int elem_idx, const double* q_in, double* rhs_out) {
    // Skip dry elements — no source term needed
    double H_avg = q_in[elem_idx * DG_BLOCK_SIZE + 0 * DG_BASIS_SIZE];
    if (H_avg < config_.wet_dry_threshold && D_[elem_idx] < config_.wet_dry_threshold) return;

    // Bathymetry source term, POSITIVE-DEPTH convention (D >= 0 is water depth,
    // bed elevation z = -D, eta = H - D):
    //   S = (0, +g*H*dD/dx, +g*H*dD/dy)
    // Derivation: classic SWE source is -g*H*dz/dx where z is bed elevation; here
    // z = -D, so dz/dx = -dD/dx and the source becomes +g*H*dD/dx.
    // Using -g*H*dD/dx (bed-elevation convention) flips the sign and prevents the
    // pressure-gradient term g*d(H^2/2)/dx from cancelling at lake-at-rest.
    // Well-balanced: evaluate at quadrature points using DG-projected D(x,y).

    const auto& ed = elem_matrices_->GetElementData(elem_idx);
    int base = elem_idx * DG_BLOCK_SIZE;
    int dbase = elem_idx * DG_BASIS_SIZE;
    const bool wb_p = UseWellBalancedPressure();

    for (int q = 0; q < ElementData::VOL_NPTS; ++q) {
        const double* phi = ed.vol_phi.data() + q * DG_BASIS_SIZE;
        const double* dphi_dx = ed.vol_dphi_dx.data() + q * DG_BASIS_SIZE;
        const double* dphi_dy = ed.vol_dphi_dy.data() + q * DG_BASIS_SIZE;
        double w = ed.vol_weights[q];

        // Reconstruct H and D at quadrature point
        double H = 0.0, D_pt = 0.0;
        for (int l = 0; l < DG_BASIS_SIZE; ++l) {
            H    += q_in[base + 0 * DG_BASIS_SIZE + l] * phi[l];
            D_pt += D_coeffs_[dbase + l] * phi[l];
        }
        double H_safe = std::max(H, 0.0);

        // Reconstruct D gradient at quadrature point from D coefficients
        double dD_dx = 0.0, dD_dy = 0.0;
        for (int l = 0; l < DG_BASIS_SIZE; ++l) {
            dD_dx += D_coeffs_[dbase + l] * dphi_dx[l];
            dD_dy += D_coeffs_[dbase + l] * dphi_dy[l];
        }

        // Bed-slope source. Well-balanced (default): use the DEVIATION g*eta*dD
        // (eta = H - D) so the pure-bed part g*D*dD — which together with the
        // dropped 0.5*g*D^2 volume pressure forms the surface bed term — is NOT
        // computed by (aliased) volume quadrature. At rest eta=0 => source=0
        // identically. Legacy (DG_WB_PRESSURE=0): full g*H*dD.
        // eta or H; clamp negative P2 bathymetry overshoot (D_c = max(D_pt,0)) so
        // src_coef = H_safe - D_c == 0 at rest even where D_pt < 0 (see VolumeIntegral).
        double src_coef = wb_p ? (H_safe - std::max(D_pt, 0.0)) : H_safe;

        // Source: integral(+g*src_coef*dD/dx * phi_l) for momentum equations
        // Positive-depth convention; see comment above.
        for (int l = 0; l < DG_BASIS_SIZE; ++l) {
            rhs_out[base + 1 * DG_BASIS_SIZE + l] += SWE_G * src_coef * dD_dx * phi[l] * w;
            rhs_out[base + 2 * DG_BASIS_SIZE + l] += SWE_G * src_coef * dD_dy * phi[l] * w;
        }
    }
}

// ========== SLOPE LIMITER ==========

void DGSolver::ApplyModalFilter() {
    // Well-balanced spectral viscosity (Codex review 2026-05-31). The eta-deviation
    // pressure/source cancellation -gD*grad(eta) - g*eta*grad(eta) is exact at rest
    // (eta=0) but, for eta!=0 on steep/coarse bathymetry, the degree-5 integrands
    // Dη∇φ and η∇D·φ are under-integrated by the degree-4 volume rule, leaving a
    // phase-wrong, weakly UNSTABLE feedback into the top P2 (quadratic) modes. That
    // 2Δx mode is ~edge-continuous so the HLL flux gives it no dissipation and it
    // grows ~8%/step (lake-at-rest drift 0 -> 138 m). Cure: damp the QUADRATIC modes
    // (l=3,4,5) of the DEVIATION eta = H - D (and of momentum). At rest eta_l = 0 and
    // Hu_l = Hv_l = 0, so this is a STRICT NO-OP — preserves Fix A well-balancedness
    // and dgvalidation 9/9. Per-mode retain factor sigma in (0,1); env DG_MODAL_FILTER
    // (default ON) and DG_FILTER_SIGMA (default 0.5).
    static int on_cache = -1;
    static double sigma_cache = 0.5;
    if (on_cache < 0) {
        char* b = nullptr; size_t s = 0;
        _dupenv_s(&b, &s, "DG_MODAL_FILTER");
        on_cache = (b && b[0] == '1') ? 1 : 0; // default OFF (filter on modes 3-5 did
        // not reduce the saturated drift — it lives in the LINEAR modes — and it broke
        // P2 convergence (dgval 12.2). Kept for experiments; root fix is higher quadrature.
        if (b) free(b);
        char* bs = nullptr; size_t ss = 0;
        _dupenv_s(&bs, &ss, "DG_FILTER_SIGMA");
        if (bs && bs[0]) { double v = atof(bs); if (v >= 0.0 && v <= 1.0) sigma_cache = v; }
        if (bs) free(bs);
        std::cerr << "[DG] modal filter " << (on_cache ? "ON" : "OFF")
                  << " sigma=" << sigma_cache << " (eta quadratic modes 3-5)\n";
    }
    if (!on_cache || DG_BASIS_SIZE <= 3) return;
    const double sigma = sigma_cache;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        if (D_[k] < config_.wet_dry_threshold) continue; // frozen dry land
        int base = k * DG_BLOCK_SIZE;
        int dbase = k * DG_BASIS_SIZE;
        for (int l = 3; l < DG_BASIS_SIZE; ++l) { // quadratic modes
            // H as deviation eta = H - D (well-balanced: no-op when H_l == D_l)
            double eta_l = q_[base + 0 * DG_BASIS_SIZE + l] - D_coeffs_[dbase + l];
            q_[base + 0 * DG_BASIS_SIZE + l] = D_coeffs_[dbase + l] + sigma * eta_l;
            // momentum (raw; no-op at rest where Hu=Hv=0)
            q_[base + 1 * DG_BASIS_SIZE + l] *= sigma;
            q_[base + 2 * DG_BASIS_SIZE + l] *= sigma;
        }
    }
}

void DGSolver::ApplySteepMomentumDamping(double dt) {
    // Codex review 2026-05-31: the lake-at-rest fixed point is EXACT but linearly
    // UNSTABLE on steep deep bathymetry via the MOMENTUM leg of the mode-0 loop
    //   eta_0 -> bed source g*eta_0*gradD -> Hu_0 -> mass flux -> eta_0.
    // Linearized 2x2: stability needs mu*nu > A*B with B ~ g*|gradD|. HLL gives the
    // mass damping mu; raising mu (mass viscosity) was ineffective + NaN. The correct
    // lever is to raise the MOMENTUM damping nu, and ONLY where the steepness sensor
    // S_K = h_K*|gradD|_K / H_K is large. P0 (cell-average) Rayleigh damping of
    // (Hu_0, Hv_0). NO-OP at lake-at-rest (Hu=Hv=0) and on flat bathymetry
    // (gradD=0 => chi=0), so Fix A well-balancedness, dgvalidation and smooth-flow
    // accuracy are preserved; it acts only on the steep cells that drive the instability.
    static int on_cache = -1;
    static double S0 = 0.15, S1 = 0.35;
    static int diag_done = 0;
    if (on_cache < 0) {
        char* b = nullptr; size_t s = 0;
        _dupenv_s(&b, &s, "DG_STEEP_DAMP"); on_cache = (b && b[0] == '1') ? 1 : 0; if (b) free(b); // default OFF: ineffective (mode-0 momentum not the lever); kept for experiments
        char* b0 = nullptr; size_t s0 = 0; _dupenv_s(&b0, &s0, "DG_STEEP_S0"); if (b0 && b0[0]) S0 = atof(b0); if (b0) free(b0);
        char* b1 = nullptr; size_t s1 = 0; _dupenv_s(&b1, &s1, "DG_STEEP_S1"); if (b1 && b1[0]) S1 = atof(b1); if (b1) free(b1);
        std::cerr << "[DG] steep-cell P0 momentum damping " << (on_cache ? "ON" : "OFF")
                  << " (gradD sensor S0=" << S0 << " S1=" << S1 << ")\n";
    }
    if (!on_cache) return;
    // One-time calibration print: the actual steepness-sensor distribution over wet cells.
    if (!diag_done) {
        diag_done = 1;
        double maxS = 0.0; int fired = 0;
        for (int k = 0; k < num_elements_; ++k) {
            if (D_[k] < config_.wet_dry_threshold) continue;
            const auto& ed = elem_matrices_->GetElementData(k);
            const auto& elem = mesh_->GetElement(k);
            int dbase = k * DG_BASIS_SIZE;
            double Hx = ed.basis.GetHx(), Hy = ed.basis.GetHy();
            double dDdx = (Hx > 0.0) ? D_coeffs_[dbase + 1] / Hx : 0.0;
            double dDdy = (Hy > 0.0) ? D_coeffs_[dbase + 2] / Hy : 0.0;
            double gradD = std::sqrt(dDdx * dDdx + dDdy * dDdy);
            const auto& p0 = mesh_->GetNode(elem.nodes[0]);
            const auto& p1 = mesh_->GetNode(elem.nodes[1]);
            const auto& p2 = mesh_->GetNode(elem.nodes[2]);
            double perim = std::hypot(p1.x - p0.x, p1.y - p0.y) + std::hypot(p2.x - p1.x, p2.y - p1.y) + std::hypot(p0.x - p2.x, p0.y - p2.y);
            double h_K = (perim > 0.0) ? 2.0 * elem.area / perim : std::sqrt(std::max(elem.area, 1e-30));
            double H_K = std::max(q_[k * DG_BLOCK_SIZE + 0], (double)SWE_H_DRY);
            double Sk = h_K * gradD / H_K;
            if (Sk > maxS) maxS = Sk;
            if (Sk >= S0) fired++;
        }
        std::cerr << "[DG steep-sensor] max S_K = " << maxS << "  cells with S_K>=S0(" << S0
                  << "): " << fired << "\n";
    }
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        if (D_[k] < config_.wet_dry_threshold) continue; // frozen dry land
        int base = k * DG_BLOCK_SIZE;
        int dbase = k * DG_BASIS_SIZE;
        const auto& ed = elem_matrices_->GetElementData(k);
        const auto& elem = mesh_->GetElement(k);
        double Hx = ed.basis.GetHx(), Hy = ed.basis.GetHy();
        double dDdx = (Hx > 0.0) ? D_coeffs_[dbase + 1] / Hx : 0.0;
        double dDdy = (Hy > 0.0) ? D_coeffs_[dbase + 2] / Hy : 0.0;
        double gradD = std::sqrt(dDdx * dDdx + dDdy * dDdy);
        const auto& p0 = mesh_->GetNode(elem.nodes[0]);
        const auto& p1 = mesh_->GetNode(elem.nodes[1]);
        const auto& p2 = mesh_->GetNode(elem.nodes[2]);
        double perim = std::hypot(p1.x - p0.x, p1.y - p0.y)
                     + std::hypot(p2.x - p1.x, p2.y - p1.y)
                     + std::hypot(p0.x - p2.x, p0.y - p2.y);
        double h_K = (perim > 0.0) ? 2.0 * elem.area / perim
                                   : std::sqrt(std::max(elem.area, 1e-30));
        double H_K = std::max(q_[base + 0 * DG_BASIS_SIZE], (double)SWE_H_DRY);
        double S_K = h_K * gradD / H_K;
        double chi = (S_K - S0) / (S1 - S0);
        chi = std::max(0.0, std::min(1.0, chi));
        if (chi <= 0.0) continue;
        double c_K = std::sqrt(SWE_G * H_K);
        double nudt = chi * std::min(0.25, std::max(0.12, 2.5 * c_K * dt / std::max(h_K, 1e-30)));
        double fac = 1.0 - nudt;
        q_[base + 1 * DG_BASIS_SIZE + 0] *= fac; // Hu cell-average (mode 0)
        q_[base + 2 * DG_BASIS_SIZE + 0] *= fac; // Hv cell-average (mode 0)
    }
}

void DGSolver::ApplySlopeLimiter() {
    if (!use_slope_limiter_) return;

    // 2026-05-20 evening: Well-balanced limiter (Codex Q4 / Bunya-Westerink-style).
    // Default: limit raw H modes (legacy).
    // With DG_LIMIT_ETA=1: limit η = H − D modes instead. For lake-at-rest
    // η ≡ 0 in every basis coefficient, so the limiter is a no-op (no spurious
    // free-surface generation). Diagnosed root cause: limiting H without D
    // makes η_pt nonzero across element interiors, which the bed-slope source
    // and Audusse HR then amplify (drift 19 m / 50 steps → blow-up).
    // 2026-05-30: eta-mode is now DEFAULT ON. Measured on dgislandprobe: limiting
    // raw H modes gives max|eta|=20.78 (FAIL) on a steep shallow island, while
    // limiting eta=H-D gives 5.86 (PASS); dgvalidation stays 9/9 and hybridwave
    // 5/5 with eta-mode on. Opt out for A/B with env DG_LIMIT_ETA=0.
    static int eta_mode_cache = -1;
    if (eta_mode_cache < 0) {
        eta_mode_cache = 1; // default ON
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "DG_LIMIT_ETA");
        if (buf && *buf == '0') eta_mode_cache = 0; // explicit opt-out
        if (buf) free(buf);
        std::cerr << "[DG] slope limiter operates on "
                  << (eta_mode_cache ? "eta=H-D (default)" : "raw H (DG_LIMIT_ETA=0)") << "\n";
    }
    const bool limit_eta = (eta_mode_cache == 1);

    if (limit_eta) {
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < num_elements_; ++k) {
            int base = k * DG_BLOCK_SIZE;
            int dbase = k * DG_BASIS_SIZE;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                q_[base + 0 * DG_BASIS_SIZE + l] -= D_coeffs_[dbase + l];
            }
        }
    }

    // Two-pass slope limiter for DG P2:
    // Pass 1: Detect troubled cells using jump indicator (Krivodonova-style)
    // Pass 2: In troubled cells, reduce to P0 + apply BJ limiter on reconstructed P1

    // Mark troubled cells (per-element, not per-variable — if any var is troubled, limit all)
    // Note: use vector<char> instead of vector<bool> for thread-safe parallel writes
    // (vector<bool> uses bit-packing, causing race conditions on adjacent elements)
    std::vector<char> troubled(num_elements_, 0);

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        const auto& elem = mesh_->GetElement(k);
        int base = k * DG_BLOCK_SIZE;
        double h = std::sqrt(elem.area);
        // In eta-mode q_[var=0] is η; the wet/dry test on cell-average H
        // becomes a test on the static bathymetry D_[k] (frozen).
        double H_avg = limit_eta ? D_[k] : q_[base + 0 * DG_BASIS_SIZE];

        // Force troubled if this cell or any neighbor is semi-dry (wet/dry front)
        if (H_avg < config_.wet_dry_threshold) {
            troubled[k] = 1;
        } else {
            for (int e = 0; e < 3; ++e) {
                int nbr = elem.neighbors[e];
                if (nbr < 0) continue; // Physical boundary — don't force troubled
                double H_nbr = limit_eta ? D_[nbr]
                                          : q_[nbr * DG_BLOCK_SIZE + 0 * DG_BASIS_SIZE];
                if (H_nbr < config_.wet_dry_threshold) {
                    troubled[k] = 1;
                    break;
                }
            }
        }

        for (int var = 0; var < DG_NUM_VARS; ++var) {
            if (troubled[k] != 0) break;
            int vbase = base + var * DG_BASIS_SIZE;
            double q_avg = q_[vbase];

            // Compute max face jump relative to cell value
            for (int e = 0; e < 3; ++e) {
                int nbr = elem.neighbors[e];
                if (nbr < 0) continue;
                double q_nbr = q_[nbr * DG_BLOCK_SIZE + var * DG_BASIS_SIZE];
                double jump = std::abs(q_nbr - q_avg);
                double scale = std::max(std::abs(q_avg), 1.0);

                // Troubled if face jump is large relative to cell value
                if (jump > tvb_M_ * elem.area && jump > config_.limiter_jump_threshold * scale) {
                    troubled[k] = 1;
                    break;
                }
            }

            // Also check if higher-order modes are large (oscillation indicator)
            if (troubled[k] == 0 && DG_ORDER >= 2) {
                double mode_sum = 0.0;
                for (int l = 1; l < DG_BASIS_SIZE; ++l) {
                    mode_sum += q_[vbase + l] * q_[vbase + l];
                }
                double avg2 = q_avg * q_avg + config_.zero_area_guard;
                if (mode_sum > config_.limiter_oscillation_threshold * avg2) {
                    troubled[k] = 1;
                }
            }
        }
    }

    // Pass 2: Limit troubled cells
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < num_elements_; ++k) {
        if (troubled[k] == 0) continue;

        const auto& elem = mesh_->GetElement(k);
        const auto& ed = elem_matrices_->GetElementData(k);
        int base = k * DG_BLOCK_SIZE;

        for (int var = 0; var < DG_NUM_VARS; ++var) {
            int vbase = base + var * DG_BASIS_SIZE;
            double q_avg = q_[vbase];

            // Get neighbor cell averages
            double q_min = q_avg, q_max = q_avg;
            double nbr_avgs[3];
            int nbr_count = 0;
            for (int e = 0; e < 3; ++e) {
                int nbr = elem.neighbors[e];
                if (nbr >= 0) {
                    double q_nbr = q_[nbr * DG_BLOCK_SIZE + var * DG_BASIS_SIZE];
                    q_min = std::min(q_min, q_nbr);
                    q_max = std::max(q_max, q_nbr);
                    nbr_avgs[nbr_count++] = q_nbr;
                }
            }

            // Zero all higher-order modes (reduce to P0)
            for (int l = 1; l < DG_BASIS_SIZE; ++l) {
                q_[vbase + l] = 0.0;
            }

            // Reconstruct limited P1 gradient using minmod of neighbor differences
            // For each direction (xi, eta), compute candidate gradients from neighbors
            // and apply componentwise minmod
            if (nbr_count >= 2) {
                // Compute gradient from centroid-to-centroid differences
                double dq_dx = 0.0, dq_dy = 0.0;
                double w_sum = 0.0;
                for (int e = 0; e < 3; ++e) {
                    int nbr = elem.neighbors[e];
                    if (nbr < 0) continue;
                    double q_nbr = q_[nbr * DG_BLOCK_SIZE + var * DG_BASIS_SIZE];
                    const auto& nbr_elem = mesh_->GetElement(nbr);
                    double dx = nbr_elem.cx - elem.cx;
                    double dy = nbr_elem.cy - elem.cy;
                    double dist2 = dx * dx + dy * dy;
                    if (dist2 < config_.zero_dist_guard) continue;
                    double dq = q_nbr - q_avg;
                    double inv_dist2 = 1.0 / dist2;
                    dq_dx += dq * dx * inv_dist2;
                    dq_dy += dq * dy * inv_dist2;
                    w_sum += 1.0;
                }
                if (w_sum > 0) {
                    dq_dx /= w_sum;
                    dq_dy /= w_sum;
                }

                // Convert gradient to basis coefficients: q_1 = dq/dx * hx, q_2 = dq/dy * hy
                double hx = ed.basis.GetHx();
                double hy = ed.basis.GetHy();
                q_[vbase + 1] = dq_dx * hx;
                q_[vbase + 2] = dq_dy * hy;

                // BJ limiter: scale to ensure vertex values stay in [q_min, q_max]
                double theta = 1.0;
                for (int n = 0; n < 3; ++n) {
                    const auto& node = mesh_->GetNode(elem.nodes[n]);
                    double phi[DG_BASIS_SIZE];
                    ed.basis.Evaluate(node.x, node.y, phi);

                    double q_pt = q_avg + q_[vbase + 1] * phi[1] + q_[vbase + 2] * phi[2];
                    if (q_pt > q_max + config_.limiter_bj_tol && q_pt > q_avg + config_.limiter_bj_tol) {
                        theta = std::min(theta, (q_max - q_avg) / (q_pt - q_avg));
                    } else if (q_pt < q_min - config_.limiter_bj_tol && q_pt < q_avg - config_.limiter_bj_tol) {
                        theta = std::min(theta, (q_min - q_avg) / (q_pt - q_avg));
                    }
                }
                theta = std::max(theta, 0.0);
                q_[vbase + 1] *= theta;
                q_[vbase + 2] *= theta;
            }
        }
    }

    // Restore H = D + η after eta-mode limiting.
    if (limit_eta) {
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < num_elements_; ++k) {
            int base = k * DG_BLOCK_SIZE;
            int dbase = k * DG_BASIS_SIZE;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                q_[base + 0 * DG_BASIS_SIZE + l] += D_coeffs_[dbase + l];
            }
        }
    }
}

// ========== POSITIVITY LIMITER ==========

void DGSolver::ApplyPositivityLimiter() {
    const bool wb_limiter = UseWellBalancedPositivityLimiter();

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        int base = k * DG_BLOCK_SIZE;
        double H_avg = q_[base + 0 * DG_BASIS_SIZE];

        if (!wb_limiter) {
            if (H_avg < SWE_H_DRY) {
                for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                    q_[base + 0 * DG_BASIS_SIZE + l] = (l == 0) ? SWE_H_DRY : 0.0;
                    q_[base + 1 * DG_BASIS_SIZE + l] = 0.0;
                    q_[base + 2 * DG_BASIS_SIZE + l] = 0.0;
                }
                continue;
            }

            const auto& ed = elem_matrices_->GetElementData(k);
            const auto& elem = mesh_->GetElement(k);
            double H_min = H_avg;

            for (int n = 0; n < 3; ++n) {
                const auto& node = mesh_->GetNode(elem.nodes[n]);
                double phi[DG_BASIS_SIZE];
                ed.basis.Evaluate(node.x, node.y, phi);

                double H_pt = 0.0;
                for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                    H_pt += q_[base + 0 * DG_BASIS_SIZE + l] * phi[l];
                }
                H_min = std::min(H_min, H_pt);
            }

            for (int e = 0; e < 3; ++e) {
                for (int qp = 0; qp < ElementData::EDGE_NPTS; ++qp) {
                    const double* phi_edge = ed.edge_phi[e].data() + qp * DG_BASIS_SIZE;
                    double H_pt = 0.0;
                    for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                        H_pt += q_[base + 0 * DG_BASIS_SIZE + l] * phi_edge[l];
                    }
                    H_min = std::min(H_min, H_pt);
                }
            }

            if (H_min < 0.0) {
                double theta = std::min(H_avg / (H_avg - H_min), 1.0);
                for (int var = 0; var < DG_NUM_VARS; ++var) {
                    for (int l = 1; l < DG_BASIS_SIZE; ++l) {
                        q_[base + var * DG_BASIS_SIZE + l] *= theta;
                    }
                }
            }

            if (H_avg < config_.dry_element_factor * SWE_H_DRY) {
                for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                    q_[base + 1 * DG_BASIS_SIZE + l] = 0.0;
                    q_[base + 2 * DG_BASIS_SIZE + l] = 0.0;
                }
                continue;
            }

            if (H_avg < config_.wet_dry_threshold) {
                for (int var = 0; var < DG_NUM_VARS; ++var) {
                    for (int l = 1; l < DG_BASIS_SIZE; ++l) {
                        q_[base + var * DG_BASIS_SIZE + l] = 0.0;
                    }
                }

                double Hu = q_[base + 1 * DG_BASIS_SIZE];
                double Hv = q_[base + 2 * DG_BASIS_SIZE];
                double u = Hu / std::max(H_avg, SWE_H_DRY);
                double v = Hv / std::max(H_avg, SWE_H_DRY);
                double speed = std::sqrt(u * u + v * v);
                if (speed > config_.max_velocity) {
                    double scale = config_.max_velocity / speed;
                    q_[base + 1 * DG_BASIS_SIZE] = H_avg * u * scale;
                    q_[base + 2 * DG_BASIS_SIZE] = H_avg * v * scale;
                }
                continue;
            }

            double Hu = q_[base + 1 * DG_BASIS_SIZE];
            double Hv = q_[base + 2 * DG_BASIS_SIZE];
            double u = Hu / H_avg;
            double v = Hv / H_avg;
            double speed = std::sqrt(u * u + v * v);
            if (speed > config_.max_velocity) {
                double scale = config_.max_velocity / speed;
                for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                    q_[base + 1 * DG_BASIS_SIZE + l] *= scale;
                    q_[base + 2 * DG_BASIS_SIZE + l] *= scale;
                }
            }
            continue;
        }

        int dbase = k * DG_BASIS_SIZE;
        double eta[DG_BASIS_SIZE];
        for (int l = 0; l < DG_BASIS_SIZE; ++l) {
            eta[l] = q_[base + 0 * DG_BASIS_SIZE + l] - D_coeffs_[dbase + l];
        }

        auto restore_h = [&]() {
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                q_[base + 0 * DG_BASIS_SIZE + l] = D_coeffs_[dbase + l] + eta[l];
            }
        };

        auto zero_momentum = [&]() {
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                q_[base + 1 * DG_BASIS_SIZE + l] = 0.0;
                q_[base + 2 * DG_BASIS_SIZE + l] = 0.0;
            }
        };

        const auto& ed = elem_matrices_->GetElementData(k);
        const auto& elem = mesh_->GetElement(k);

        auto eval_d_eta = [&](const double* phi, double& D_pt, double& eta_pt) {
            D_pt = 0.0;
            eta_pt = 0.0;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                D_pt += D_coeffs_[dbase + l] * phi[l];
                eta_pt += eta[l] * phi[l];
            }
        };

        auto sample_points = [&](auto&& fn) {
            for (int n = 0; n < 3; ++n) {
                const auto& node = mesh_->GetNode(elem.nodes[n]);
                double phi[DG_BASIS_SIZE];
                ed.basis.Evaluate(node.x, node.y, phi);
                fn(phi);
            }
            for (int e = 0; e < 3; ++e) {
                for (int qp = 0; qp < ElementData::EDGE_NPTS; ++qp) {
                    const double* phi_edge = ed.edge_phi[e].data() + qp * DG_BASIS_SIZE;
                    fn(phi_edge);
                }
            }
        };

        auto min_sampled_h = [&]() {
            double H_min = D_coeffs_[dbase] + eta[0];
            sample_points([&](const double* phi) {
                double D_pt, eta_pt;
                eval_d_eta(phi, D_pt, eta_pt);
                H_min = std::min(H_min, D_pt + eta_pt);
            });
            return H_min;
        };

        auto min_sampled_d = [&]() {
            double D_min = D_coeffs_[dbase];
            sample_points([&](const double* phi) {
                double D_pt, eta_pt;
                eval_d_eta(phi, D_pt, eta_pt);
                D_min = std::min(D_min, D_pt);
            });
            return D_min;
        };

        auto eta_has_content = [&]() {
            double eta_norm = 0.0;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                eta_norm += std::abs(eta[l]);
            }
            return eta_norm > 1e-12;
        };

        if (H_avg < SWE_H_DRY) {
            for (int l = 1; l < DG_BASIS_SIZE; ++l) {
                eta[l] = 0.0;
            }
            double min_H_p0_eta = min_sampled_d() + eta[0];
            if (min_H_p0_eta < 0.0 && eta_has_content()) {
                eta[0] -= min_H_p0_eta;
                H_avg = D_coeffs_[dbase] + eta[0];
            }
            zero_momentum();
            restore_h();
            continue;
        }

        double H_min = min_sampled_h();
        if (H_min < 0.0) {
            const double eta_avg = eta[0];
            double theta = 1.0;
            sample_points([&](const double* phi) {
                double D_pt, eta_pt;
                eval_d_eta(phi, D_pt, eta_pt);
                double H_pt = D_pt + eta_pt;
                if (H_pt >= 0.0) return;

                double eta_high_pt = eta_pt - eta_avg;
                double H_p0_eta = D_pt + eta_avg;
                if (eta_high_pt < 0.0 && H_p0_eta > 0.0) {
                    theta = std::min(theta, H_p0_eta / (H_p0_eta - H_pt));
                } else {
                    theta = 0.0;
                }
            });
            theta = std::clamp(theta, 0.0, 1.0);
            for (int l = 1; l < DG_BASIS_SIZE; ++l) {
                eta[l] *= theta;
                q_[base + 1 * DG_BASIS_SIZE + l] *= theta;
                q_[base + 2 * DG_BASIS_SIZE + l] *= theta;
            }

            double limited_min = min_sampled_h();
            if (limited_min < 0.0 && eta_has_content()) {
                eta[0] -= limited_min;
                H_avg = D_coeffs_[dbase] + eta[0];
            }
        }

        if (H_avg < config_.dry_element_factor * SWE_H_DRY) {
            zero_momentum();
            restore_h();
            continue;
        }

        if (H_avg < config_.wet_dry_threshold) {
            for (int l = 1; l < DG_BASIS_SIZE; ++l) {
                eta[l] = 0.0;
                q_[base + 1 * DG_BASIS_SIZE + l] = 0.0;
                q_[base + 2 * DG_BASIS_SIZE + l] = 0.0;
            }

            double Hu = q_[base + 1 * DG_BASIS_SIZE];
            double Hv = q_[base + 2 * DG_BASIS_SIZE];
            double u = Hu / std::max(H_avg, SWE_H_DRY);
            double v = Hv / std::max(H_avg, SWE_H_DRY);
            double speed = std::sqrt(u * u + v * v);
            if (speed > config_.max_velocity) {
                double scale = config_.max_velocity / speed;
                q_[base + 1 * DG_BASIS_SIZE] = H_avg * u * scale;
                q_[base + 2 * DG_BASIS_SIZE] = H_avg * v * scale;
            }
            restore_h();
            continue;
        }

        double Hu = q_[base + 1 * DG_BASIS_SIZE];
        double Hv = q_[base + 2 * DG_BASIS_SIZE];
        double u = Hu / H_avg;
        double v = Hv / H_avg;
        double speed = std::sqrt(u * u + v * v);
        if (speed > config_.max_velocity) {
            double scale = config_.max_velocity / speed;
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                q_[base + 1 * DG_BASIS_SIZE + l] *= scale;
                q_[base + 2 * DG_BASIS_SIZE + l] *= scale;
            }
        }
        restore_h();
    }

    EnforceDryCells();
}

void DGSolver::EnforceDryCells() {
    const bool wb_limiter = UseWellBalancedPositivityLimiter();

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_elements_; ++k) {
        if (D_[k] < config_.wet_dry_threshold) {
            int base = k * DG_BLOCK_SIZE;
            if (wb_limiter) {
                int dbase = k * DG_BASIS_SIZE;
                for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                    q_[base + 0 * DG_BASIS_SIZE + l] = D_coeffs_[dbase + l];
                }
            } else {
                q_[base + 0 * DG_BASIS_SIZE] = D_[k];
                for (int l = 1; l < DG_BASIS_SIZE; ++l) {
                    q_[base + 0 * DG_BASIS_SIZE + l] = 0.0;
                }
            }
            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                q_[base + 1 * DG_BASIS_SIZE + l] = 0.0;
                q_[base + 2 * DG_BASIS_SIZE + l] = 0.0;
            }
        }
    }
}

// ========== CFL ==========

double DGSolver::ComputeTimeStep(double cfl_target) const {
    double dt_min = config_.dt_upper_bound;

    #pragma omp parallel
    {
        double dt_local_min = config_.dt_upper_bound;
        #pragma omp for schedule(static) nowait
        for (int k = 0; k < num_elements_; ++k) {
            SWEState q_avg = GetCellAverage(k);
            double wave_speed = RiemannSolver::MaxWaveSpeed(q_avg);

            if (wave_speed < config_.min_wave_speed) continue;

            double h_elem = std::sqrt(mesh_->GetElement(k).area);

            // DG CFL: dt <= CFL * h / ((2*p+1) * max_wave_speed)
            // Valid for SSP-RK1/2/3 (SSP coefficient c_eff = 1)
            double factor = 2.0 * DG_ORDER + 1.0;
            double dt_local = cfl_target * h_elem / (factor * wave_speed);
            dt_local_min = std::min(dt_local_min, dt_local);
        }
        #pragma omp critical
        {
            dt_min = std::min(dt_min, dt_local_min);
        }
    }

    return dt_min;
}
