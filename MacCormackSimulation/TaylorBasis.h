// TaylorBasis.h - Taylor polynomial basis for DG solver on triangles
// Basis functions centered at element centroid: 1, (x-xc), (y-yc), ...
// Supports orders R=1 (3 DOF) and R=2 (6 DOF)

#ifndef TAYLOR_BASIS_H
#define TAYLOR_BASIS_H

#include "TriMesh.h"
#include "GaussQuadrature.h"
#include <array>
#include <vector>
#include <cmath>

// ========== COMPILE-TIME DG ORDER CONFIGURATION ==========

constexpr int DG_ORDER = 2;  // 1 or 2
constexpr int DG_BASIS_SIZE = (DG_ORDER == 1) ? 3 : 6;
constexpr int DG_NUM_VARS = 3;  // H, Hu, Hv
constexpr int DG_BLOCK_SIZE = DG_NUM_VARS * DG_BASIS_SIZE;  // 9 or 18

/// Taylor basis evaluator for a single triangle element
///
/// R=1 basis: phi_0 = 1, phi_1 = (x-xc)/hx, phi_2 = (y-yc)/hy
/// R=2 basis: + phi_3 = ((x-xc)/hx)^2 - <..>_k, phi_4 = (x-xc)(y-yc)/(hx*hy), phi_5 = ((y-yc)/hy)^2 - <..>_k
///
/// hx, hy are characteristic element lengths for scaling (= sqrt(area))
class TaylorBasis {
public:
    TaylorBasis() = default;

    /// Initialize basis for a given triangle element
    /// @param cx, cy Centroid coordinates
    /// @param area Element area
    void Initialize(double cx, double cy, double area);

    /// Evaluate all basis functions at point (x, y)
    /// @param x, y Physical coordinates
    /// @param phi Output array of size DG_BASIS_SIZE
    void Evaluate(double x, double y, double* phi) const;

    /// Evaluate gradients of all basis functions at point (x, y)
    /// @param x, y Physical coordinates
    /// @param dphi_dx, dphi_dy Output arrays of size DG_BASIS_SIZE
    void EvaluateGradient(double x, double y, double* dphi_dx, double* dphi_dy) const;

    double GetCx() const { return cx_; }
    double GetCy() const { return cy_; }
    double GetHx() const { return hx_; }
    double GetHy() const { return hy_; }
    double GetArea() const { return area_; }

    // Orthogonalization constants for R=2 (mean values of xi^2 and eta^2 over element)
    double GetMeanXi2() const { return mean_xi2_; }
    double GetMeanEta2() const { return mean_eta2_; }

    /// Set mean values for orthogonalization (computed by ElementMatrices)
    void SetOrthogonalizationConstants(double mean_xi2, double mean_eta2, double mean_xieta);

private:
    double cx_ = 0.0, cy_ = 0.0;  // Centroid
    double area_ = 0.0;
    double hx_ = 1.0, hy_ = 1.0;  // Characteristic lengths
    double inv_hx_ = 1.0, inv_hy_ = 1.0;

    // R=2 orthogonalization: phi_3 = xi^2 - mean_xi2, phi_4 = xi*eta - mean_xieta, phi_5 = eta^2 - mean_eta2
    double mean_xi2_ = 0.0;
    double mean_eta2_ = 0.0;
    double mean_xieta_ = 0.0;
};

#endif // TAYLOR_BASIS_H
