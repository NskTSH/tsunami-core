// TaylorBasis.cpp - Taylor polynomial basis implementation

#include "TaylorBasis.h"
#include <cmath>
#include <algorithm>

void TaylorBasis::Initialize(double cx, double cy, double area) {
    cx_ = cx;
    cy_ = cy;
    area_ = area;

    // Characteristic length: sqrt(area) is a natural scale for a triangle
    double h = std::sqrt(std::max(area, 1e-30));
    hx_ = h;
    hy_ = h;
    inv_hx_ = 1.0 / hx_;
    inv_hy_ = 1.0 / hy_;

    // Default orthogonalization constants (will be overridden by ElementMatrices)
    mean_xi2_ = 1.0 / 6.0;
    mean_eta2_ = 1.0 / 6.0;
    mean_xieta_ = 0.0;
}

void TaylorBasis::SetOrthogonalizationConstants(double mean_xi2, double mean_eta2, double mean_xieta) {
    mean_xi2_ = mean_xi2;
    mean_eta2_ = mean_eta2;
    mean_xieta_ = mean_xieta;
}

void TaylorBasis::Evaluate(double x, double y, double* phi) const {
    double xi = (x - cx_) * inv_hx_;
    double eta = (y - cy_) * inv_hy_;

    // R=1 basis
    phi[0] = 1.0;
    phi[1] = xi;
    phi[2] = eta;

    if constexpr (DG_ORDER >= 2) {
        // R=2 basis (orthogonalized against phi_0: <phi_k, phi_0> = 0)
        phi[3] = xi * xi - mean_xi2_;
        phi[4] = xi * eta - mean_xieta_;
        phi[5] = eta * eta - mean_eta2_;
    }
}

void TaylorBasis::EvaluateGradient(double x, double y, double* dphi_dx, double* dphi_dy) const {
    double xi = (x - cx_) * inv_hx_;
    double eta = (y - cy_) * inv_hy_;

    // dphi/dx = dphi/dxi * dxi/dx = dphi/dxi * inv_hx
    // dphi/dy = dphi/deta * deta/dy = dphi/deta * inv_hy

    // phi_0 = 1
    dphi_dx[0] = 0.0;
    dphi_dy[0] = 0.0;

    // phi_1 = xi
    dphi_dx[1] = inv_hx_;
    dphi_dy[1] = 0.0;

    // phi_2 = eta
    dphi_dx[2] = 0.0;
    dphi_dy[2] = inv_hy_;

    if constexpr (DG_ORDER >= 2) {
        // phi_3 = xi^2 - mean_xi2
        dphi_dx[3] = 2.0 * xi * inv_hx_;
        dphi_dy[3] = 0.0;

        // phi_4 = xi * eta
        dphi_dx[4] = eta * inv_hx_;
        dphi_dy[4] = xi * inv_hy_;

        // phi_5 = eta^2 - mean_eta2
        dphi_dx[5] = 0.0;
        dphi_dy[5] = 2.0 * eta * inv_hy_;
    }
}
