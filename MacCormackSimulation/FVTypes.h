// FVTypes.h — conservative-variable types for FV Godunov backend
//
// Phase A (Codex Audusse FV design 2026-05-28). Conservative SWE state for
// per-cell working storage inside the FV solver. Grid public API stays
// primitive (H, u, v, eta); FV converts at stage seams only.

#pragma once

#include <cmath>

#include "Precision.h"

namespace fv {

// Conservative shallow-water state. h = water column depth (>= 0),
// hu = h * u, hv = h * v. Velocity is recovered safely via Vel() to avoid
// 0/0 on dry cells.
struct FVState {
    Real h;
    Real hu;
    Real hv;
};

inline FVState ZeroState() {
    return { Real(0), Real(0), Real(0) };
}

// Safe velocity recovery: returns (u, v) avoiding div-by-zero on dry cells.
// Uses regularized division when h is at or below `dry_tol`.
inline void SafeVel(const FVState& U, Real dry_tol, Real& u, Real& v) {
    if (U.h <= dry_tol) {
        u = Real(0);
        v = Real(0);
    } else {
        const Real inv_h = Real(1) / U.h;
        u = U.hu * inv_h;
        v = U.hv * inv_h;
    }
}

// Convert from project's primitive (H, u, v) to conservative (h, hu, hv).
// Caller guarantees H >= 0 in wet cells; dry cells should pass H = 0.
inline FVState FromPrimitive(Real H, Real u, Real v) {
    FVState U;
    U.h  = H;
    U.hu = H * u;
    U.hv = H * v;
    return U;
}

// Convert conservative back to primitive. eta = h - D where D = -z_b (project
// convention: positive D = water depth at rest). Caller passes still-water
// depth D (so dry land cells have D <= 0).
inline void ToPrimitive(const FVState& U, Real D, Real dry_tol,
                        Real& H, Real& u, Real& v, Real& eta) {
    H = U.h;
    SafeVel(U, dry_tol, u, v);
    eta = U.h - D;
}

}  // namespace fv
