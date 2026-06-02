// RiemannSolver.cpp - HLL Riemann solver for shallow water equations
// Wave speed bounds: two-rarefaction star-state estimate (Toro 2001)

#include "RiemannSolver.h"
#include <cmath>
#include <algorithm>

void RiemannSolver::RotateToNormal(double u, double v, double nx, double ny,
                                    double& un, double& ut) {
    un =  u * nx + v * ny;
    ut = -u * ny + v * nx;
}

void RiemannSolver::RotateFromNormal(double un, double ut, double nx, double ny,
                                      double& u, double& v) {
    u = un * nx - ut * ny;
    v = un * ny + ut * nx;
}

void RiemannSolver::ComputeFlux(const SWEState& qL, const SWEState& qR,
                                 double nx, double ny,
                                 SWEState& flux) {
    // Handle dry states
    double HL = std::max(qL.H, 0.0);
    double HR = std::max(qR.H, 0.0);

    if (HL < SWE_H_DRY && HR < SWE_H_DRY) {
        flux = SWEState(0.0, 0.0, 0.0);
        return;
    }

    // Velocities (safe division with point-value cap)
    double uL = SafeVelCapped(HL, qL.Hu);
    double vL = SafeVelCapped(HL, qL.Hv);
    double uR = SafeVelCapped(HR, qR.Hu);
    double vR = SafeVelCapped(HR, qR.Hv);

    // Rotate to normal coordinates
    double unL, utL, unR, utR;
    RotateToNormal(uL, vL, nx, ny, unL, utL);
    RotateToNormal(uR, vR, nx, ny, unR, utR);

    // Wave speeds
    double cL = std::sqrt(SWE_G * HL);
    double cR = std::sqrt(SWE_G * HR);

    // HLL wave speed estimates (two-rarefaction / star-state approximation)
    // See Toro (2001) "Shock-Capturing Methods for Free-Surface Shallow Flows"
    double uStar = 0.5 * (unL + unR) + cL - cR;
    double cStar = 0.5 * (cL + cR) + 0.25 * (unL - unR);
    cStar = std::max(cStar, 0.0);

    double sL = std::min(unL - cL, uStar - cStar);
    double sR = std::max(unR + cR, uStar + cStar);

    // Normal fluxes in rotated coordinates: F(q) = (H*un, H*un^2 + g*H^2/2, H*un*ut)
    // Left flux
    double fL_H  = HL * unL;
    double fL_Hu = HL * unL * unL + 0.5 * SWE_G * HL * HL;
    double fL_Hv = HL * unL * utL;

    // Right flux
    double fR_H  = HR * unR;
    double fR_Hu = HR * unR * unR + 0.5 * SWE_G * HR * HR;
    double fR_Hv = HR * unR * utR;

    // HLL flux in normal coordinates
    double fn_H, fn_Hu, fn_Hv;

    if (sL >= 0.0) {
        // All waves move right -> use left flux
        fn_H  = fL_H;
        fn_Hu = fL_Hu;
        fn_Hv = fL_Hv;
    }
    else if (sR <= 0.0) {
        // All waves move left -> use right flux
        fn_H  = fR_H;
        fn_Hu = fR_Hu;
        fn_Hv = fR_Hv;
    }
    else {
        // Intermediate state
        double inv_ds = 1.0 / (sR - sL);
        fn_H  = (sR * fL_H  - sL * fR_H  + sL * sR * (HR - HL)) * inv_ds;
        fn_Hu = (sR * fL_Hu - sL * fR_Hu + sL * sR * (HR * unR - HL * unL)) * inv_ds;
        fn_Hv = (sR * fL_Hv - sL * fR_Hv + sL * sR * (HR * utR - HL * utL)) * inv_ds;
    }

    // Rotate flux back to physical coordinates
    // fn_Hu is normal-direction momentum flux, fn_Hv is tangential
    double flux_Hu, flux_Hv;
    RotateFromNormal(fn_Hu, fn_Hv, nx, ny, flux_Hu, flux_Hv);

    flux.H  = fn_H;
    flux.Hu = flux_Hu;
    flux.Hv = flux_Hv;
}

void RiemannSolver::WallBC(const SWEState& qInner, double nx, double ny,
                            SWEState& qGhost) {
    // Reflect normal velocity: U_n -> -U_n, U_t -> U_t
    double u = qInner.GetU();
    double v = qInner.GetV();

    double un, ut;
    RotateToNormal(u, v, nx, ny, un, ut);

    // Flip normal velocity
    un = -un;

    double u_ghost, v_ghost;
    RotateFromNormal(un, ut, nx, ny, u_ghost, v_ghost);

    qGhost.H  = qInner.H;
    qGhost.Hu = qInner.H * u_ghost;
    qGhost.Hv = qInner.H * v_ghost;
}

void RiemannSolver::OpenBC(const SWEState& qInner, double nx, double ny,
                            SWEState& qGhost) {
    // Legacy zero-gradient: ghost = interior. Diffusive but not strictly
    // non-reflecting. Prefer OpenBCRadiation when a far-field state is known.
    qGhost = qInner;
}

void RiemannSolver::OpenBCRadiation(const SWEState& qInner, const SWEState& qFar,
                                     double nx, double ny, SWEState& qGhost) {
    // Subcritical characteristic / Flather radiation BC.
    //
    // Decompose velocity into normal/tangential at the boundary, then take the
    // outgoing Riemann invariant R+ = u_n + 2c from the interior and the
    // incoming R- = u_n - 2c from the far-field reference. Reconstruct the
    // ghost state from R+, R-:
    //     u_n,g = 0.5*(R+ + R-)
    //     c_g   = 0.25*(R+ - R-)
    //     H_g   = c_g^2 / g
    // Tangential velocity is upwinded: outflow → interior, inflow → far-field.
    //
    // When qInner == qFar (lake at rest), R+ and R- collapse so that
    // (u_n,g, H_g, u_t,g) == interior, and the BC is exactly well-balanced.
    double H_in = std::max(qInner.H, 0.0);
    double H_far = std::max(qFar.H, 0.0);
    if (H_in < SWE_H_DRY && H_far < SWE_H_DRY) {
        qGhost = SWEState(0.0, 0.0, 0.0);
        return;
    }

    double u_in = SafeVelCapped(H_in, qInner.Hu);
    double v_in = SafeVelCapped(H_in, qInner.Hv);
    double u_fr = SafeVelCapped(H_far, qFar.Hu);
    double v_fr = SafeVelCapped(H_far, qFar.Hv);

    double un_in, ut_in, un_fr, ut_fr;
    RotateToNormal(u_in, v_in, nx, ny, un_in, ut_in);
    RotateToNormal(u_fr, v_fr, nx, ny, un_fr, ut_fr);

    double c_in = std::sqrt(SWE_G * H_in);
    double c_fr = std::sqrt(SWE_G * H_far);

    double Rp = un_in + 2.0 * c_in;  // outgoing from interior
    double Rm = un_fr - 2.0 * c_fr;  // incoming from far field

    double un_g = 0.5 * (Rp + Rm);
    double c_g = 0.25 * (Rp - Rm);
    c_g = std::max(c_g, 0.0);
    double H_g = c_g * c_g / SWE_G;

    // Upwind tangential velocity by sign of normal velocity at the ghost state
    double ut_g = (un_g >= 0.0) ? ut_in : ut_fr;

    double u_g, v_g;
    RotateFromNormal(un_g, ut_g, nx, ny, u_g, v_g);

    qGhost.H = H_g;
    qGhost.Hu = H_g * u_g;
    qGhost.Hv = H_g * v_g;
}

double RiemannSolver::MaxWaveSpeed(const SWEState& q) {
    if (q.H < SWE_H_DRY) return 0.0;

    double u = std::abs(SafeVelCapped(q.H, q.Hu));
    double v = std::abs(SafeVelCapped(q.H, q.Hv));
    double c = std::sqrt(SWE_G * q.H);

    return std::max(u + c, v + c);
}
