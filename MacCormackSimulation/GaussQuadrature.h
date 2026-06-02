// GaussQuadrature.h - Gauss quadrature rules for triangles and edges
// Used by DG solver for numerical integration

#ifndef GAUSS_QUADRATURE_H
#define GAUSS_QUADRATURE_H

#include <array>
#include <cmath>

namespace gauss {

// ========== QUADRATURE POINT ==========

struct QuadPoint2D {
    double xi;     // Barycentric / reference coordinate 1
    double eta;    // Barycentric / reference coordinate 2
    double weight; // Quadrature weight
};

struct QuadPoint1D {
    double xi;     // Reference coordinate on [0, 1]
    double weight; // Quadrature weight
};

// ========== TRIANGLE QUADRATURES (reference triangle: (0,0)-(1,0)-(0,1)) ==========

// Order 1, 1 point (exact for polynomials up to degree 1)
constexpr int TRI_ORDER1_NPTS = 1;
constexpr std::array<QuadPoint2D, TRI_ORDER1_NPTS> TriOrder1() {
    return {{ {1.0/3.0, 1.0/3.0, 0.5} }};
}

// Order 2, 3 points (exact for polynomials up to degree 2)
constexpr int TRI_ORDER2_NPTS = 3;
constexpr std::array<QuadPoint2D, TRI_ORDER2_NPTS> TriOrder2() {
    return {{
        {1.0/6.0, 1.0/6.0, 1.0/6.0},
        {2.0/3.0, 1.0/6.0, 1.0/6.0},
        {1.0/6.0, 2.0/3.0, 1.0/6.0}
    }};
}

// Order 4, 6 points (exact for polynomials up to degree 4)
// Dunavant quadrature
constexpr int TRI_ORDER4_NPTS = 6;
inline std::array<QuadPoint2D, TRI_ORDER4_NPTS> TriOrder4() {
    const double a1 = 0.445948490915965;
    const double b1 = 0.108103018168070;
    const double w1 = 0.111690794839006; // weights already include ref triangle area (1/2)
    const double a2 = 0.091576213509771;
    const double b2 = 0.816847572980459;
    const double w2 = 0.054975871827661;

    return {{
        {a1, a1, w1}, {a1, b1, w1}, {b1, a1, w1},
        {a2, a2, w2}, {a2, b2, w2}, {b2, a2, w2}
    }};
}

// ========== EDGE (1D) QUADRATURES on [0, 1] ==========

// Order 2, 2 points (Gauss-Legendre mapped to [0,1])
constexpr int EDGE_ORDER2_NPTS = 2;
inline std::array<QuadPoint1D, EDGE_ORDER2_NPTS> EdgeOrder2() {
    const double a = 0.5 - 0.5 / std::sqrt(3.0);
    const double b = 0.5 + 0.5 / std::sqrt(3.0);
    return {{ {a, 0.5}, {b, 0.5} }};
}

// Order 3, 3 points (Gauss-Legendre mapped to [0,1])
constexpr int EDGE_ORDER3_NPTS = 3;
inline std::array<QuadPoint1D, EDGE_ORDER3_NPTS> EdgeOrder3() {
    const double a = 0.5 - 0.5 * std::sqrt(3.0/5.0);
    const double b = 0.5;
    const double c = 0.5 + 0.5 * std::sqrt(3.0/5.0);
    return {{ {a, 5.0/18.0}, {b, 8.0/18.0}, {c, 5.0/18.0} }};
}

// ========== UTILITY ==========

/// Map reference triangle point (xi, eta) to physical triangle (x0,y0)-(x1,y1)-(x2,y2)
/// x = x0 + (x1-x0)*xi + (x2-x0)*eta
/// y = y0 + (y1-y0)*xi + (y2-y0)*eta
inline void MapToPhysical(double xi, double eta,
                          double x0, double y0,
                          double x1, double y1,
                          double x2, double y2,
                          double& x, double& y) {
    x = x0 + (x1 - x0) * xi + (x2 - x0) * eta;
    y = y0 + (y1 - y0) * xi + (y2 - y0) * eta;
}

/// Map reference edge parameter t in [0,1] to physical edge (xa,ya)-(xb,yb)
inline void MapEdgeToPhysical(double t,
                              double xa, double ya,
                              double xb, double yb,
                              double& x, double& y) {
    x = xa + (xb - xa) * t;
    y = ya + (yb - ya) * t;
}

} // namespace gauss

#endif // GAUSS_QUADRATURE_H
