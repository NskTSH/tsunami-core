// FVSolver.cpp — Phase A first-order HR-HLL implementation
//
// SSP-RK2 (Heun) scheme:
//   U1 = U0 + dt * L(U0)
//   U_new = 0.5 * U0 + 0.5 * (U1 + dt * L(U1))
//
// We map this onto the project's existing predictor-corrector seam:
//   Stage1Predictor: build U0 from primitive, compute U1 = U0 + dt*L(U0),
//                    write U1 as primitive into hat arrays. External exchange
//                    fills hat ghost rows.
//   Stage2Corrector: build U1 from primitive hat, compute U_new with Heun
//                    formula, write U_new as primitive into main arrays.

#include "FVSolver.h"

#include <algorithm>
#include <cmath>
#include <omp.h>

#include "FVFlux.h"
#include "FVHydrostaticReconstruction.h"
#include "FVSolverConfig.h"
#include "Grid.h"
#include "SubGrid.h"

namespace fv {

std::atomic<std::uint64_t> g_positivity_clamps{0};

FVSolver::FVSolver(SubGrid* owner) : owner_(owner) {}

void FVSolver::EnsureBuffers() {
    const int nx = owner_->local_nx_;
    const int ny = owner_->local_ny_;
    if (nx == nx_cached_ && ny == ny_cached_) return;
    const size_t N = static_cast<size_t>(nx) * ny;
    U0_.assign(N, ZeroState());
    U1_.assign(N, ZeroState());
    L_.assign (N, ZeroState());
    nx_cached_ = nx;
    ny_cached_ = ny;
}

void FVSolver::LoadConservativeFromPrimitive() {
    EnsureBuffers();
    const int nx = owner_->local_nx_;
    const int ny = owner_->local_ny_;
    Grid& g = owner_->local_grid_;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const size_t idx = static_cast<size_t>(i) * ny + j;
            const Real H = g.H(i, j);
            const Real u = g.u(i, j);
            const Real v = g.v(i, j);
            U0_[idx] = FromPrimitive(H, u, v);
        }
    }
}

void FVSolver::WriteConservativeToHat(const std::vector<FVState>& U) {
    const int nx = owner_->local_nx_;
    const int ny = owner_->local_ny_;
    Grid& g = owner_->local_grid_;
    Real* H_hat   = owner_->H_hat_.get();
    Real* u_hat   = owner_->u_hat_.get();
    Real* v_hat   = owner_->v_hat_.get();
    Real* eta_hat = owner_->eta_hat_.get();
    const Real dry_tol = static_cast<Real>(config().dry_tol);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const size_t idx = static_cast<size_t>(i) * ny + j;
            const Real D = g.D(i, j);
            Real H, u, v, eta;
            ToPrimitive(U[idx], D, dry_tol, H, u, v, eta);
            H_hat[idx]   = H;
            u_hat[idx]   = u;
            v_hat[idx]   = v;
            eta_hat[idx] = eta;
        }
    }
}

void FVSolver::WriteConservativeToMain(const std::vector<FVState>& U) {
    const int nx = owner_->local_nx_;
    const int ny = owner_->local_ny_;
    Grid& g = owner_->local_grid_;
    const Real dry_tol = static_cast<Real>(config().dry_tol);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const size_t idx = static_cast<size_t>(i) * ny + j;
            const Real D = g.D(i, j);
            Real H, u, v, eta;
            ToPrimitive(U[idx], D, dry_tol, H, u, v, eta);
            g.H(i, j)   = H;
            g.u(i, j)   = u;
            g.v(i, j)   = v;
            g.eta(i, j) = eta;
        }
    }
}

// Compute Euler RHS L(U) = -1/Vol * (sum of face fluxes) for each interior
// cell. Phase A: Cartesian uniform dx, dy. Faces handled with HR + HLL +
// Audusse pressure correction. Boundary faces — Phase B (reflective walls).
// Phase A interim: skip boundary faces (i=0, i=nx-1, j=0, j=ny-1 cells get
// zero RHS), interior cells [1..nx-2, 1..ny-2] are updated using face fluxes
// at (i-1/2, i+1/2, j-1/2, j+1/2) with HR.
void FVSolver::ComputeRHS(const std::vector<FVState>& U,
                          std::vector<FVState>& L_out)
{
    const int nx = owner_->local_nx_;
    const int ny = owner_->local_ny_;
    const Grid& g = owner_->local_grid_;
    const Real grav = static_cast<Real>(owner_->g_);
    const Real dry_tol = static_cast<Real>(config().dry_tol);
    // dx, dy are uniform in Cartesian Phase A. owner_->inv_dy_ is set;
    // delta_x(0,0) sample.
    const Real dx = static_cast<Real>(owner_->delta_x_(0, 0));
    const Real inv_dx = (dx > 0) ? Real(1) / dx : Real(0);
    const Real inv_dy = static_cast<Real>(owner_->inv_dy_);

    // Zero RHS for all cells including ghosts (boundary cells get zero on
    // this Phase A pass — Phase B replaces with reflective wall flux).
    std::fill(L_out.begin(), L_out.end(), ZeroState());

    // Detect domain-edge: a neighbor cell with D<=0 is treated as REFLECTIVE
    // WALL (Codex pitfall #1: zero-gradient ≠ reflective). Helper builds a
    // wall-ghost state by flipping normal momentum at the face, then runs HR
    // on the (cell, wall-ghost) pair.
    // Wall detection rule (Phase B):
    //   * D <= 0 AND h <= dry_tol  → genuine wall ghost (domain edge with no
    //     wavemaker forcing): substitute reflected state.
    //   * D <= 0 BUT h > dry_tol   → ghost is set by external BC (wavemaker
    //     etc.) — use as-is with D_e = D_other_side fallback (HR cannot use
    //     invalid D=0, so we degrade to plain HLL between cell and forced ghost).
    auto x_face_or_wall = [&](Real h_L, Real hu_L, Real hv_L, Real D_L,
                              Real h_R, Real hu_R, Real hv_R, Real D_R)
        -> std::pair<HydroFaceState, FluxResult>
    {
        const bool wall_L = (D_L <= Real(0)) && (h_L <= dry_tol);
        const bool wall_R = (D_R <= Real(0)) && (h_R <= dry_tol);
        if (wall_L && !wall_R) {
            FVState gst = WallGhostX({h_R, hu_R, hv_R});
            HydroFaceState f = ReconstructFace(gst.h, gst.hu, gst.hv, D_R,
                                               h_R, hu_R, hv_R, D_R, dry_tol);
            return {f, HLL_X(f.UL, f.UR, grav, dry_tol)};
        }
        if (wall_R && !wall_L) {
            FVState gst = WallGhostX({h_L, hu_L, hv_L});
            HydroFaceState f = ReconstructFace(h_L, hu_L, hv_L, D_L,
                                               gst.h, gst.hu, gst.hv, D_L, dry_tol);
            return {f, HLL_X(f.UL, f.UR, grav, dry_tol)};
        }
        if (wall_L && wall_R) {
            return {{}, {Real(0), Real(0), Real(0)}};
        }
        // Mixed case: BC-driven ghost (h>0, D<=0) — use the other side's D
        // for HR fallback so the formula doesn't degenerate.
        const Real D_L_use = (D_L <= Real(0)) ? D_R : D_L;
        const Real D_R_use = (D_R <= Real(0)) ? D_L : D_R;
        HydroFaceState f = ReconstructFace(h_L, hu_L, hv_L, D_L_use,
                                           h_R, hu_R, hv_R, D_R_use, dry_tol);
        return {f, HLL_X(f.UL, f.UR, grav, dry_tol)};
    };
    auto y_face_or_wall = [&](Real h_L, Real hu_L, Real hv_L, Real D_L,
                              Real h_R, Real hu_R, Real hv_R, Real D_R)
        -> std::pair<HydroFaceState, FluxResult>
    {
        const bool wall_L = (D_L <= Real(0)) && (h_L <= dry_tol);
        const bool wall_R = (D_R <= Real(0)) && (h_R <= dry_tol);
        if (wall_L && !wall_R) {
            FVState gst = WallGhostY({h_R, hu_R, hv_R});
            HydroFaceState f = ReconstructFace(gst.h, gst.hu, gst.hv, D_R,
                                               h_R, hu_R, hv_R, D_R, dry_tol);
            return {f, HLL_Y(f.UL, f.UR, grav, dry_tol)};
        }
        if (wall_R && !wall_L) {
            FVState gst = WallGhostY({h_L, hu_L, hv_L});
            HydroFaceState f = ReconstructFace(h_L, hu_L, hv_L, D_L,
                                               gst.h, gst.hu, gst.hv, D_L, dry_tol);
            return {f, HLL_Y(f.UL, f.UR, grav, dry_tol)};
        }
        if (wall_L && wall_R) {
            return {{}, {Real(0), Real(0), Real(0)}};
        }
        const Real D_L_use = (D_L <= Real(0)) ? D_R : D_L;
        const Real D_R_use = (D_R <= Real(0)) ? D_L : D_R;
        HydroFaceState f = ReconstructFace(h_L, hu_L, hv_L, D_L_use,
                                           h_R, hu_R, hv_R, D_R_use, dry_tol);
        return {f, HLL_Y(f.UL, f.UR, grav, dry_tol)};
    };

    #pragma omp parallel for schedule(static)
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            const size_t cij = static_cast<size_t>(i) * ny + j;

            const Real h_ij = U[cij].h;
            const Real D_ij = g.D(i, j);
            // Skip permanently dry land cells (D < 0); Phase A.
            if (D_ij <= Real(0)) {
                continue;
            }

            // West face (i-1/2): between cell (i-1, j) and cell (i, j).
            const size_t cWm = static_cast<size_t>(i - 1) * ny + j;
            const Real D_Wm = g.D(i - 1, j);
            auto [W, FW] = x_face_or_wall(
                U[cWm].h, U[cWm].hu, U[cWm].hv, D_Wm,
                U[cij].h, U[cij].hu, U[cij].hv, D_ij);

            // East face (i+1/2): between cell (i, j) and cell (i+1, j).
            const size_t cEp = static_cast<size_t>(i + 1) * ny + j;
            const Real D_Ep = g.D(i + 1, j);
            auto [E, FE] = x_face_or_wall(
                U[cij].h, U[cij].hu, U[cij].hv, D_ij,
                U[cEp].h, U[cEp].hu, U[cEp].hv, D_Ep);

            // South face (j-1/2): between (i, j-1) and (i, j).
            const size_t cSm = static_cast<size_t>(i) * ny + (j - 1);
            const Real D_Sm = g.D(i, j - 1);
            auto [S, FS] = y_face_or_wall(
                U[cSm].h, U[cSm].hu, U[cSm].hv, D_Sm,
                U[cij].h, U[cij].hu, U[cij].hv, D_ij);

            // North face (j+1/2): between (i, j) and (i, j+1).
            const size_t cNp = static_cast<size_t>(i) * ny + (j + 1);
            const Real D_Np = g.D(i, j + 1);
            auto [N, FN] = y_face_or_wall(
                U[cij].h, U[cij].hu, U[cij].hv, D_ij,
                U[cNp].h, U[cNp].hu, U[cNp].hv, D_Np);

            // Audusse per-cell L/R pressure corrections (well-balanced source).
            // For cell (i,j):
            //   West face contributes to its EAST-facing momentum:
            //     hu adjustment = +g/2 * (h_ij^2 - h_W.R^2)  (R = this cell's view)
            //   East face contributes to its WEST-facing momentum:
            //     hu adjustment = -g/2 * (h_ij^2 - h_E.L^2)  (L = this cell's view)
            // The sign convention: cell sees flux IN through west face, OUT through east.
            // FV_AUDUSSE_PRESSURE=0 zeroes these → bare HLL flux divergence,
            // not well-balanced (pre-paper ablation diagnostic).
            const bool use_audusse = config().audusse_pressure;
            const Real pW_R = use_audusse ? AudussePressureCorrection(h_ij, W.UR.h, grav) : Real(0);
            const Real pE_L = use_audusse ? AudussePressureCorrection(h_ij, E.UL.h, grav) : Real(0);
            const Real pS_R = use_audusse ? AudussePressureCorrection(h_ij, S.UR.h, grav) : Real(0);
            const Real pN_L = use_audusse ? AudussePressureCorrection(h_ij, N.UL.h, grav) : Real(0);

            // L(U) = -(F_E - F_W)/dx - (F_N - F_S)/dy  for each conserved var,
            // with Audusse pressure correction on momentum components:
            //   hu corrected flux at W: (FW.F_hu + 0)        (we add p to W.F_hu only inside cell view)
            //   hu corrected flux at E: (FE.F_hu + 0)
            // Audusse formal form: replace F_HLL momentum with F_HLL + p_correction
            // where p is computed using THIS cell's h vs reconstructed h*.
            // For x-momentum:
            const Real FW_hu_corr = FW.F_hu + pW_R;  // west face seen by cell (R side)
            const Real FE_hu_corr = FE.F_hu + pE_L;  // east face seen by cell (L side); note: sign handled in flux divergence
            // For y-momentum at y-faces:
            const Real FS_hv_corr = FS.F_hv + pS_R;
            const Real FN_hv_corr = FN.F_hv + pN_L;

            // Conservative flux divergence (outward-positive at east/north,
            // inward-positive at west/south of this cell):
            //   L_h  = -(F_E.h  - F_W.h)/dx  - (F_N.h  - F_S.h)/dy
            //   L_hu = -(F_E.hu - F_W.hu)/dx - (F_N.hu - F_S.hu)/dy
            //   L_hv = -(F_E.hv - F_W.hv)/dx - (F_N.hv - F_S.hv)/dy
            FVState L;
            L.h  = -(FE.F_h  - FW.F_h ) * inv_dx - (FN.F_h  - FS.F_h ) * inv_dy;
            // Use Audusse-corrected hu/hv at the corresponding face direction:
            L.hu = -(FE_hu_corr - FW_hu_corr) * inv_dx
                   - (FN.F_hu   - FS.F_hu  ) * inv_dy;
            L.hv = -(FE.F_hv   - FW.F_hv  ) * inv_dx
                   - (FN_hv_corr - FS_hv_corr) * inv_dy;
            L_out[cij] = L;
        }
    }
}

void FVSolver::Stage1Predictor() {
    LoadConservativeFromPrimitive();
    ComputeRHS(U0_, L_);
    const Real tau = static_cast<Real>(owner_->tau_);
    const int nx = owner_->local_nx_;
    const int ny = owner_->local_ny_;
    const Real dry_tol = static_cast<Real>(config().dry_tol);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const size_t idx = static_cast<size_t>(i) * ny + j;
            FVState Un;
            Un.h  = U0_[idx].h  + tau * L_[idx].h;
            Un.hu = U0_[idx].hu + tau * L_[idx].hu;
            Un.hv = U0_[idx].hv + tau * L_[idx].hv;
            // Drying: enforce h >= 0; momentum zero if dry.
            // Phase E pre-paper: only count GENUINE positivity events (cell was
            // wet before, would have gone dry). Already-dry wall cells skipped.
            if (Un.h <= dry_tol) {
                if (U0_[idx].h > dry_tol) {
                    g_positivity_clamps.fetch_add(1, std::memory_order_relaxed);
                }
                Un.h = Real(0); Un.hu = Real(0); Un.hv = Real(0);
            }
            U1_[idx] = Un;
        }
    }
    WriteConservativeToHat(U1_);
}

void FVSolver::Stage2Corrector() {
    // U0_ already loaded from previous Stage1Predictor (held in member).
    // Build U1 from exchanged primitive hat (so ghost rows reflect neighbor data).
    EnsureBuffers();
    const int nx = owner_->local_nx_;
    const int ny = owner_->local_ny_;
    Grid& g = owner_->local_grid_;
    const Real* H_hat   = owner_->H_hat_.get();
    const Real* u_hat   = owner_->u_hat_.get();
    const Real* v_hat   = owner_->v_hat_.get();
    (void)g;  // for now
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const size_t idx = static_cast<size_t>(i) * ny + j;
            U1_[idx] = FromPrimitive(H_hat[idx], u_hat[idx], v_hat[idx]);
        }
    }

    // Compute L(U1).
    ComputeRHS(U1_, L_);

    const Real tau = static_cast<Real>(owner_->tau_);
    const Real dry_tol = static_cast<Real>(config().dry_tol);
    std::vector<FVState> U_new(static_cast<size_t>(nx) * ny);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const size_t idx = static_cast<size_t>(i) * ny + j;
            FVState Un;
            Un.h  = Real(0.5) * (U0_[idx].h  + U1_[idx].h  + tau * L_[idx].h );
            Un.hu = Real(0.5) * (U0_[idx].hu + U1_[idx].hu + tau * L_[idx].hu);
            Un.hv = Real(0.5) * (U0_[idx].hv + U1_[idx].hv + tau * L_[idx].hv);
            if (Un.h <= dry_tol) {
                if (U0_[idx].h > dry_tol) {
                    g_positivity_clamps.fetch_add(1, std::memory_order_relaxed);
                }
                Un.h = Real(0); Un.hu = Real(0); Un.hv = Real(0);
            }
            U_new[idx] = Un;
        }
    }
    WriteConservativeToMain(U_new);
}

}  // namespace fv
