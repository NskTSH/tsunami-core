// FVHydrostaticReconstruction.h — Audusse 2004 face reconstruction
//
// Phase A (Codex Audusse FV design 2026-05-28).
// Reference: Audusse, Bouchut, Bristeau, Klein, Perthame (2004),
//            "A fast and stable well-balanced scheme with hydrostatic
//            reconstruction for shallow water flows", SIAM J. Sci. Comput.
//            25(6), 2050-2065. DOI 10.1137/S1064827503431090.
//
// Project convention: D > 0 is water depth at rest, so bottom elevation
// z_b = -D. For a face between cells L and R:
//   z*    = max(z_b_L, z_b_R) = -min(D_L, D_R)      // raised shared bottom
//   h_L*  = max(0, h_L + z_b_L - z*) = max(0, h_L - max(0, D_L - D*))   D* = min(D_L,D_R)
//   h_R*  = max(0, h_R + z_b_R - z*) = max(0, h_R - max(0, D_R - D*))
//
// Lake-at-rest: if eta_L = eta_R = eta0 over a step (D_L != D_R), then
// h_L* + z_b_L_at_face = h_R* + z_b_R_at_face = eta0, so reconstructed
// surface elevation is identical on both sides; HLL flux + Audusse pressure
// correction cancels exactly. This is the well-balanced property.

#pragma once

#include <algorithm>
#include <cmath>

#include "FVTypes.h"

namespace fv {

struct HydroFaceState {
    FVState UL;   // reconstructed left state at face (h_L*, hu_L scaled, hv_L scaled)
    FVState UR;   // reconstructed right state at face
    Real    D_star;  // shared face depth = min(D_L, D_R)
    Real    h_L_cell;  // original cell depth left side (for pressure correction)
    Real    h_R_cell;  // original cell depth right side
};

// Compute hydrostatic reconstruction at one face between cells L and R.
// h_L, h_R     — full water heights at cell centers (= D + eta, must be >= 0)
// hu_L, hu_R   — conservative momenta along face normal direction
// hv_L, hv_R   — conservative momenta tangential to face
// D_L, D_R     — still-water depths at cell centers (positive = water)
// dry_tol      — wet/dry threshold
//
// Returns reconstructed (h*, hu*, hv*) on each side with momentum scaled to
// preserve velocity: u* = hu / h_cell. If reconstructed h* = 0 (dry face),
// momentum is zero.
inline HydroFaceState ReconstructFace(
    Real h_L, Real hu_L, Real hv_L, Real D_L,
    Real h_R, Real hu_R, Real hv_R, Real D_R,
    Real dry_tol)
{
    HydroFaceState f;
    f.D_star = std::min(D_L, D_R);
    f.h_L_cell = h_L;
    f.h_R_cell = h_R;

    // Reconstructed depths at the face (positivity-preserving).
    const Real hL_star = std::max(Real(0), h_L - std::max(Real(0), D_L - f.D_star));
    const Real hR_star = std::max(Real(0), h_R - std::max(Real(0), D_R - f.D_star));

    // Scale momentum so velocity is preserved: u* = u_cell when wet.
    Real uL = Real(0), vL = Real(0);
    if (h_L > dry_tol) {
        const Real inv_h = Real(1) / h_L;
        uL = hu_L * inv_h;
        vL = hv_L * inv_h;
    }
    Real uR = Real(0), vR = Real(0);
    if (h_R > dry_tol) {
        const Real inv_h = Real(1) / h_R;
        uR = hu_R * inv_h;
        vR = hv_R * inv_h;
    }

    f.UL.h  = hL_star;
    f.UL.hu = hL_star * uL;
    f.UL.hv = hL_star * vL;

    f.UR.h  = hR_star;
    f.UR.hu = hR_star * uR;
    f.UR.hv = hR_star * vR;

    return f;
}

// Audusse pressure correction added to the momentum flux seen by each cell,
// to balance the cell-centered pressure gradient `g/2 * h^2`. For the
// face shared by cells L and R, the L cell sees flux F + (0, g/2*(h_L^2 - h_L*^2), 0)
// and the R cell sees flux F + (0, g/2*(h_R^2 - h_R*^2), 0) but with opposite sign.
//
// Returns the magnitude `g/2 * (h_cell^2 - h_star^2)` for the L or R side.
inline Real AudussePressureCorrection(Real h_cell, Real h_star, Real g) {
    return Real(0.5) * g * (h_cell * h_cell - h_star * h_star);
}

}  // namespace fv
