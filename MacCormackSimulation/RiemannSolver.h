// RiemannSolver.h - Approximate Riemann solver for shallow water equations
// Computes numerical flux across element edges for DG solver

#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include <cmath>
#include <algorithm>
#include <cstdlib>

/// Constants for SWE
constexpr double SWE_G = 9.81;       // Gravitational acceleration
constexpr double SWE_H_DRY = 1e-6;   // Minimum depth threshold (wet/dry)

// Pass-2 (2026-05-30): point-value wet/dry velocity cap. At steep shallow
// run-up a DG point depth H can be tiny while the cell average is healthy, so
// u=mom/H reaches hundreds of m/s and pumps the momentum flux and HLL speeds.
// Toggle OFF with env DG_DISABLE_VCAP=1; set ceiling with env DG_VMAX
// (default 20).
inline double DGVelCapVmax() {
    static double v = -1.0;
    if (v < 0.0) {
        char* b = nullptr;
        size_t s = 0;
        _dupenv_s(&b, &s, "DG_VMAX");
        v = (b && *b) ? std::atof(b) : 20.0;
        if (b) std::free(b);
        if (v <= 0.0) v = 20.0;
    }
    return v;
}

inline bool DGVelCapEnabled() {
    static int e = -1;
    if (e < 0) {
        char* b = nullptr;
        size_t s = 0;
        _dupenv_s(&b, &s, "DG_DISABLE_VCAP");
        e = (b && *b == '1') ? 0 : 1;
        if (b) std::free(b);
    }
    return e == 1;
}

// Capped velocity recovery from total depth H and a momentum component.
inline double SafeVelCapped(double H, double mom) {
    if (H <= SWE_H_DRY) return 0.0;
    double u = mom / H;
    if (DGVelCapEnabled()) {
        const double vmax = DGVelCapVmax();
        if (u >  vmax) u =  vmax;
        if (u < -vmax) u = -vmax;
    }
    return u;
}

/// SWE state vector: (H, Hu, Hv) where H = total depth
struct SWEState {
    double H;   // Total water depth
    double Hu;  // x-momentum
    double Hv;  // y-momentum

    SWEState() : H(0.0), Hu(0.0), Hv(0.0) {}
    SWEState(double h, double hu, double hv) : H(h), Hu(hu), Hv(hv) {}

    double GetU() const { return SafeVelCapped(H, Hu); }
    double GetV() const { return SafeVelCapped(H, Hv); }
};

/// Riemann solver for SWE at an edge
///
/// Solves the 1D Riemann problem in the normal direction to the edge:
///   q_t + F(q)_n = 0
/// where F_n = F * n_x + G * n_y is the normal flux
///
/// Uses HLL (Harten-Lax-van Leer) approximate Riemann solver
class RiemannSolver {
public:
    /// Compute numerical flux across an edge
    /// @param qL Left state (from element owning the edge)
    /// @param qR Right state (from neighbor element)
    /// @param nx, ny Unit outward normal (from left to right)
    /// @param flux Output: numerical flux (H-flux, Hu-flux, Hv-flux)
    static void ComputeFlux(const SWEState& qL, const SWEState& qR,
                            double nx, double ny,
                            SWEState& flux);

    /// Boundary condition: solid wall (free-slip)
    /// Reflects normal velocity, preserves tangential
    /// @param qInner Interior state
    /// @param nx, ny Outward normal
    /// @param qGhost Output: ghost state for Riemann solver
    static void WallBC(const SWEState& qInner, double nx, double ny,
                       SWEState& qGhost);

    /// Boundary condition: open (radiation / Sommerfeld)
    /// Legacy zero-gradient form: qGhost = qInner. Kept for backward compatibility
    /// with tests that don't supply a far-field reference. Prefer OpenBCRadiation.
    /// @param qInner Interior state
    /// @param nx, ny Outward normal
    /// @param qGhost Output: ghost state for Riemann solver
    static void OpenBC(const SWEState& qInner, double nx, double ny,
                       SWEState& qGhost);

    /// Characteristic-based non-reflecting radiation BC for subcritical flow
    /// (Flather-style). Outgoing Riemann invariant R+ = u_n + 2c taken from
    /// interior; incoming R- = u_n - 2c taken from the supplied far-field
    /// reference state. Tangential velocity comes from the upwind side.
    ///
    /// For lake-at-rest (qInner == qFar), the ghost equals interior exactly,
    /// so the BC is well-balanced.
    ///
    /// @param qInner Interior state at the boundary edge
    /// @param qFar Far-field reference (ambient state at rest)
    /// @param nx, ny Outward unit normal
    /// @param qGhost Output: ghost state for the Riemann solver
    static void OpenBCRadiation(const SWEState& qInner, const SWEState& qFar,
                                double nx, double ny, SWEState& qGhost);

    /// Maximum wave speed for CFL computation
    /// @param q State vector
    /// @return max(|u| + c, |v| + c) where c = sqrt(g*H)
    static double MaxWaveSpeed(const SWEState& q);

private:
    /// Rotate velocity into edge-normal coordinates
    /// U_n = u * nx + v * ny (normal velocity)
    /// U_t = -u * ny + v * nx (tangential velocity)
    static void RotateToNormal(double u, double v, double nx, double ny,
                               double& un, double& ut);

    /// Rotate back from edge-normal to physical coordinates
    static void RotateFromNormal(double un, double ut, double nx, double ny,
                                 double& u, double& v);
};

#endif // RIEMANN_SOLVER_H
