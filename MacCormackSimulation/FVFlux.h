// FVFlux.h — HLL Riemann flux for SWE conservative form
//
// Phase A (Codex Audusse FV design 2026-05-28).
// HLL (Harten-Lax-van Leer) numerical flux, used per-face on
// hydrostatically-reconstructed states.
//
// Physical flux for SWE conservative form U = (h, hu, hv):
//   F_x(U) = (hu, hu^2/h + g h^2/2, hu hv / h)
//   F_y(U) = (hv, hu hv / h,        hv^2/h + g h^2/2)
//
// HLL: F_HLL = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)
// with wave speeds S_L = min(u_L - c_L, u_R - c_R), S_R = max(u_L + c_L, u_R + c_R)
// where c = sqrt(g h). Davis 1988 / Toro 2001.

#pragma once

#include <algorithm>
#include <cmath>

#include "FVTypes.h"

namespace fv {

// Output: numerical flux at a face. Direction-specific (X face uses x-momentum
// as normal; Y face uses y-momentum).
struct FluxResult {
    Real F_h;    // mass flux (h * u_n)
    Real F_hu;   // x-momentum flux
    Real F_hv;   // y-momentum flux
};

inline Real WaveSpeed(Real h, Real g) {
    return (h > 0) ? std::sqrt(g * h) : Real(0);
}

// HLL flux at an X-face (normal direction = x). UL, UR are the (already
// hydrostatically-reconstructed) states on the left and right of the face.
// `g` is gravity.
inline FluxResult HLL_X(const FVState& UL, const FVState& UR, Real g, Real dry_tol) {
    // Recover velocities (safe).
    Real uL, vL, uR, vR;
    SafeVel(UL, dry_tol, uL, vL);
    SafeVel(UR, dry_tol, uR, vR);
    const Real cL = WaveSpeed(UL.h, g);
    const Real cR = WaveSpeed(UR.h, g);

    // Davis wave-speed estimates (simple, robust).
    const Real SL = std::min(uL - cL, uR - cR);
    const Real SR = std::max(uL + cL, uR + cR);

    // Physical fluxes on each side.
    const Real FL_h  = UL.hu;
    const Real FL_hu = UL.hu * uL + Real(0.5) * g * UL.h * UL.h;
    const Real FL_hv = UL.hu * vL;

    const Real FR_h  = UR.hu;
    const Real FR_hu = UR.hu * uR + Real(0.5) * g * UR.h * UR.h;
    const Real FR_hv = UR.hu * vR;

    FluxResult F;
    if (SL >= 0) {
        F.F_h  = FL_h;  F.F_hu = FL_hu; F.F_hv = FL_hv;
    } else if (SR <= 0) {
        F.F_h  = FR_h;  F.F_hu = FR_hu; F.F_hv = FR_hv;
    } else {
        const Real inv = Real(1) / (SR - SL);
        F.F_h  = (SR * FL_h  - SL * FR_h  + SL * SR * (UR.h  - UL.h )) * inv;
        F.F_hu = (SR * FL_hu - SL * FR_hu + SL * SR * (UR.hu - UL.hu)) * inv;
        F.F_hv = (SR * FL_hv - SL * FR_hv + SL * SR * (UR.hv - UL.hv)) * inv;
    }
    return F;
}

// HLL flux at a Y-face (normal direction = y). Same formulation with x↔y
// momentum roles swapped.
inline FluxResult HLL_Y(const FVState& UL, const FVState& UR, Real g, Real dry_tol) {
    Real uL, vL, uR, vR;
    SafeVel(UL, dry_tol, uL, vL);
    SafeVel(UR, dry_tol, uR, vR);
    const Real cL = WaveSpeed(UL.h, g);
    const Real cR = WaveSpeed(UR.h, g);

    const Real SL = std::min(vL - cL, vR - cR);
    const Real SR = std::max(vL + cL, vR + cR);

    const Real FL_h  = UL.hv;
    const Real FL_hu = UL.hv * uL;
    const Real FL_hv = UL.hv * vL + Real(0.5) * g * UL.h * UL.h;

    const Real FR_h  = UR.hv;
    const Real FR_hu = UR.hv * uR;
    const Real FR_hv = UR.hv * vR + Real(0.5) * g * UR.h * UR.h;

    FluxResult F;
    if (SL >= 0) {
        F.F_h  = FL_h;  F.F_hu = FL_hu; F.F_hv = FL_hv;
    } else if (SR <= 0) {
        F.F_h  = FR_h;  F.F_hu = FR_hu; F.F_hv = FR_hv;
    } else {
        const Real inv = Real(1) / (SR - SL);
        F.F_h  = (SR * FL_h  - SL * FR_h  + SL * SR * (UR.h  - UL.h )) * inv;
        F.F_hu = (SR * FL_hu - SL * FR_hu + SL * SR * (UR.hu - UL.hu)) * inv;
        F.F_hv = (SR * FL_hv - SL * FR_hv + SL * SR * (UR.hv - UL.hv)) * inv;
    }
    return F;
}

// Reflective wall ghost state for a face with outward normal in +X direction.
// Codex pitfall #1: zero-gradient is NOT reflective. True reflective wall
// requires normal-velocity sign flip; tangential velocity copied; h unchanged.
inline FVState WallGhostX(const FVState& U_interior) {
    FVState g;
    g.h  =  U_interior.h;
    g.hu = -U_interior.hu;  // flip normal momentum
    g.hv =  U_interior.hv;  // tangential preserved
    return g;
}

inline FVState WallGhostY(const FVState& U_interior) {
    FVState g;
    g.h  =  U_interior.h;
    g.hu =  U_interior.hu;
    g.hv = -U_interior.hv;
    return g;
}

}  // namespace fv
