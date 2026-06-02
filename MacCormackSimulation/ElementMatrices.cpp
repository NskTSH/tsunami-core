// ElementMatrices.cpp - Local DG matrix computation

#include "ElementMatrices.h"
#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <iostream>

void ElementMatrices::Precompute(const TriMesh& mesh) {
    int num_elem = mesh.NumElements();
    data_.resize(num_elem);

    for (int i = 0; i < num_elem; ++i) {
        const auto& elem = mesh.GetElement(i);

        // Initialize Taylor basis
        data_[i].basis.Initialize(elem.cx, elem.cy, elem.area);

        // Compute orthogonalization constants first (needed for R=2 basis)
        if constexpr (DG_ORDER >= 2) {
            ComputeOrthogonalization(i, mesh);
        }

        ComputeMassMatrix(i, mesh);
        ComputeStiffnessMatrices(i, mesh);
        ComputeVolumeQuadrature(i, mesh);
        ComputeEdgeIntegrals(i, mesh);
    }
}

void ElementMatrices::ComputeOrthogonalization(int elem_idx, const TriMesh& mesh) {
    const auto& elem = mesh.GetElement(elem_idx);
    auto& ed = data_[elem_idx];

    const auto& p0 = mesh.GetNode(elem.nodes[0]);
    const auto& p1 = mesh.GetNode(elem.nodes[1]);
    const auto& p2 = mesh.GetNode(elem.nodes[2]);

    double cx = ed.basis.GetCx();
    double cy = ed.basis.GetCy();
    double inv_hx = 1.0 / ed.basis.GetHx();
    double inv_hy = 1.0 / ed.basis.GetHy();

    // Integrate xi^2, eta^2, xi*eta over element using quadrature
    auto quad = gauss::TriOrder4();
    double sum_xi2 = 0.0, sum_eta2 = 0.0, sum_xieta = 0.0, sum_1 = 0.0;

    double det_J = 2.0 * elem.area; // Jacobian determinant for reference -> physical

    for (const auto& qp : quad) {
        double x, y;
        gauss::MapToPhysical(qp.xi, qp.eta,
                             p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                             x, y);

        double xi = (x - cx) * inv_hx;
        double eta = (y - cy) * inv_hy;

        double w = qp.weight * det_J;
        sum_xi2 += xi * xi * w;
        sum_eta2 += eta * eta * w;
        sum_xieta += xi * eta * w;
        sum_1 += w;
    }

    double mean_xi2 = sum_xi2 / std::max(sum_1, 1e-30);
    double mean_eta2 = sum_eta2 / std::max(sum_1, 1e-30);
    double mean_xieta = sum_xieta / std::max(sum_1, 1e-30);

    ed.basis.SetOrthogonalizationConstants(mean_xi2, mean_eta2, mean_xieta);
}

void ElementMatrices::ComputeMassMatrix(int elem_idx, const TriMesh& mesh) {
    const auto& elem = mesh.GetElement(elem_idx);
    auto& ed = data_[elem_idx];

    const auto& p0 = mesh.GetNode(elem.nodes[0]);
    const auto& p1 = mesh.GetNode(elem.nodes[1]);
    const auto& p2 = mesh.GetNode(elem.nodes[2]);

    double det_J = 2.0 * elem.area;

    std::fill(ed.M.begin(), ed.M.end(), 0.0);

    auto quad = gauss::TriOrder4();
    double phi[DG_BASIS_SIZE];

    for (const auto& qp : quad) {
        double x, y;
        gauss::MapToPhysical(qp.xi, qp.eta,
                             p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                             x, y);

        ed.basis.Evaluate(x, y, phi);

        double w = qp.weight * det_J;
        for (int i = 0; i < DG_BASIS_SIZE; ++i) {
            for (int j = 0; j < DG_BASIS_SIZE; ++j) {
                ed.M[i * DG_BASIS_SIZE + j] += phi[i] * phi[j] * w;
            }
        }
    }

    // Compute inverse
    InvertMatrix(ed.M.data(), ed.M_inv.data(), DG_BASIS_SIZE);
}

void ElementMatrices::ComputeStiffnessMatrices(int elem_idx, const TriMesh& mesh) {
    const auto& elem = mesh.GetElement(elem_idx);
    auto& ed = data_[elem_idx];

    const auto& p0 = mesh.GetNode(elem.nodes[0]);
    const auto& p1 = mesh.GetNode(elem.nodes[1]);
    const auto& p2 = mesh.GetNode(elem.nodes[2]);

    double det_J = 2.0 * elem.area;

    std::fill(ed.Ax.begin(), ed.Ax.end(), 0.0);
    std::fill(ed.Ay.begin(), ed.Ay.end(), 0.0);

    auto quad = gauss::TriOrder4();
    double phi[DG_BASIS_SIZE];
    double dphi_dx[DG_BASIS_SIZE], dphi_dy[DG_BASIS_SIZE];

    for (const auto& qp : quad) {
        double x, y;
        gauss::MapToPhysical(qp.xi, qp.eta,
                             p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                             x, y);

        ed.basis.Evaluate(x, y, phi);
        ed.basis.EvaluateGradient(x, y, dphi_dx, dphi_dy);

        double w = qp.weight * det_J;
        for (int i = 0; i < DG_BASIS_SIZE; ++i) {
            for (int j = 0; j < DG_BASIS_SIZE; ++j) {
                // Ax_{ij} = integral phi_i * dphi_j/dx
                ed.Ax[i * DG_BASIS_SIZE + j] += phi[i] * dphi_dx[j] * w;
                // Ay_{ij} = integral phi_i * dphi_j/dy
                ed.Ay[i * DG_BASIS_SIZE + j] += phi[i] * dphi_dy[j] * w;
            }
        }
    }
}

void ElementMatrices::ComputeVolumeQuadrature(int elem_idx, const TriMesh& mesh) {
    const auto& elem = mesh.GetElement(elem_idx);
    auto& ed = data_[elem_idx];

    const auto& p0 = mesh.GetNode(elem.nodes[0]);
    const auto& p1 = mesh.GetNode(elem.nodes[1]);
    const auto& p2 = mesh.GetNode(elem.nodes[2]);

    double det_J = 2.0 * elem.area;
    auto quad = gauss::TriOrder4();

    for (int q = 0; q < gauss::TRI_ORDER4_NPTS; ++q) {
        double x, y;
        gauss::MapToPhysical(quad[q].xi, quad[q].eta,
                             p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                             x, y);

        ed.basis.Evaluate(x, y, ed.vol_phi.data() + q * DG_BASIS_SIZE);
        ed.basis.EvaluateGradient(x, y,
                                  ed.vol_dphi_dx.data() + q * DG_BASIS_SIZE,
                                  ed.vol_dphi_dy.data() + q * DG_BASIS_SIZE);
        ed.vol_weights[q] = quad[q].weight * det_J;
    }
}

void ElementMatrices::ComputeEdgeIntegrals(int elem_idx, const TriMesh& mesh) {
    const auto& elem = mesh.GetElement(elem_idx);
    auto& ed = data_[elem_idx];

    auto edge_quad_pts = gauss::EdgeOrder3();

    for (int e = 0; e < 3; ++e) {
        int n0 = elem.nodes[e];
        int n1 = elem.nodes[(e + 1) % 3];

        const auto& pa = mesh.GetNode(n0);
        const auto& pb = mesh.GetNode(n1);

        double edge_len = std::sqrt((pb.x - pa.x) * (pb.x - pa.x)
                                  + (pb.y - pa.y) * (pb.y - pa.y));

        for (int q = 0; q < gauss::EDGE_ORDER3_NPTS; ++q) {
            double x, y;
            gauss::MapEdgeToPhysical(edge_quad_pts[q].xi,
                                     pa.x, pa.y, pb.x, pb.y,
                                     x, y);

            // Store quadrature point info
            ed.edge_quad[e][q * 3 + 0] = x;
            ed.edge_quad[e][q * 3 + 1] = y;
            ed.edge_quad[e][q * 3 + 2] = edge_quad_pts[q].weight * edge_len;

            // Evaluate basis at edge quadrature point
            double phi[DG_BASIS_SIZE];
            ed.basis.Evaluate(x, y, phi);

            for (int l = 0; l < DG_BASIS_SIZE; ++l) {
                ed.edge_phi[e][q * DG_BASIS_SIZE + l] = phi[l];
            }
        }
    }
}

// ========== MATRIX INVERSION (small dense, Gauss-Jordan) ==========

void ElementMatrices::InvertMatrix(const double* M, double* M_inv, int n) {
    // Augmented matrix [M | I]
    std::vector<double> aug(n * 2 * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i * 2 * n + j] = M[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Gauss-Jordan elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        // Find pivot
        int pivot_row = col;
        double pivot_val = std::abs(aug[col * 2 * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double val = std::abs(aug[row * 2 * n + col]);
            if (val > pivot_val) {
                pivot_val = val;
                pivot_row = row;
            }
        }

        // Swap rows
        if (pivot_row != col) {
            for (int j = 0; j < 2 * n; ++j) {
                std::swap(aug[col * 2 * n + j], aug[pivot_row * 2 * n + j]);
            }
        }

        double diag = aug[col * 2 * n + col];
        if (std::abs(diag) < 1e-30) {
            std::cerr << "ElementMatrices: Singular mass matrix!" << std::endl;
            std::fill(M_inv, M_inv + n * n, 0.0);
            return;
        }

        // Scale pivot row
        double inv_diag = 1.0 / diag;
        for (int j = 0; j < 2 * n; ++j) {
            aug[col * 2 * n + j] *= inv_diag;
        }

        // Eliminate column in other rows
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double factor = aug[row * 2 * n + col];
            for (int j = 0; j < 2 * n; ++j) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }

    // Extract inverse
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            M_inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
}
