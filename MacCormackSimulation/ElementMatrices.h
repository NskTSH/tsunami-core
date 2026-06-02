// ElementMatrices.h - Local DG matrices for a single triangle element
// Mass matrix, stiffness matrices, edge integrals

#ifndef ELEMENT_MATRICES_H
#define ELEMENT_MATRICES_H

#include "TaylorBasis.h"
#include "TriMesh.h"
#include "GaussQuadrature.h"
#include <array>
#include <vector>

/// Pre-computed matrices for a single DG element
struct ElementData {
    // Mass matrix M_k (DG_BASIS_SIZE x DG_BASIS_SIZE)
    // M_{ij} = integral over K of phi_i * phi_j dA
    std::array<double, DG_BASIS_SIZE * DG_BASIS_SIZE> M;

    // Inverse mass matrix M_k^{-1}
    std::array<double, DG_BASIS_SIZE * DG_BASIS_SIZE> M_inv;

    // Stiffness matrix A^x: integral over K of phi_i * dphi_j/dx dA
    std::array<double, DG_BASIS_SIZE * DG_BASIS_SIZE> Ax;

    // Stiffness matrix A^y: integral over K of phi_i * dphi_j/dy dA
    std::array<double, DG_BASIS_SIZE * DG_BASIS_SIZE> Ay;

    // Volume quadrature: precomputed basis values and gradients at each quad point
    // Avoids calling TaylorBasis::Evaluate/EvaluateGradient at runtime
    static constexpr int VOL_NPTS = gauss::TRI_ORDER4_NPTS;  // 6
    std::array<double, VOL_NPTS * DG_BASIS_SIZE> vol_phi;       // phi[q * DG_BASIS_SIZE + l]
    std::array<double, VOL_NPTS * DG_BASIS_SIZE> vol_dphi_dx;   // dphi_dx[q * DG_BASIS_SIZE + l]
    std::array<double, VOL_NPTS * DG_BASIS_SIZE> vol_dphi_dy;   // dphi_dy[q * DG_BASIS_SIZE + l]
    std::array<double, VOL_NPTS> vol_weights;                    // qp.weight * det_J

    // Edge integrals for each of the 3 edges
    // edge_phi[e][q][l] = phi_l evaluated at quadrature point q of edge e
    // Stored flat: edge_phi[e] has EDGE_NPTS * DG_BASIS_SIZE entries
    static constexpr int EDGE_NPTS = gauss::EDGE_ORDER3_NPTS;
    std::array<std::array<double, EDGE_NPTS * DG_BASIS_SIZE>, 3> edge_phi;

    // Edge quadrature physical coordinates and weights (scaled by edge length)
    // For each edge e, quad point q: x, y, weight * edge_length
    std::array<std::array<double, EDGE_NPTS * 3>, 3> edge_quad; // {x, y, w*len}

    // Taylor basis for this element (stored for runtime evaluation)
    TaylorBasis basis;
};

/// Compute all local matrices for a given mesh
class ElementMatrices {
public:
    /// Precompute matrices for all elements in the mesh
    /// @param mesh Input triangular mesh (must have geometry computed)
    void Precompute(const TriMesh& mesh);

    /// Access element data
    const ElementData& GetElementData(int elem_idx) const { return data_[elem_idx]; }
    ElementData& GetElementData(int elem_idx) { return data_[elem_idx]; }

    int NumElements() const { return static_cast<int>(data_.size()); }

private:
    std::vector<ElementData> data_;

    /// Compute mass matrix for one element using Gauss quadrature
    void ComputeMassMatrix(int elem_idx, const TriMesh& mesh);

    /// Compute stiffness matrices Ax, Ay for one element
    void ComputeStiffnessMatrices(int elem_idx, const TriMesh& mesh);

    /// Precompute basis values and gradients at volume quadrature points
    void ComputeVolumeQuadrature(int elem_idx, const TriMesh& mesh);

    /// Compute edge quadrature data for one element
    void ComputeEdgeIntegrals(int elem_idx, const TriMesh& mesh);

    /// Compute orthogonalization constants (mean of xi^2, eta^2 over element)
    void ComputeOrthogonalization(int elem_idx, const TriMesh& mesh);

    /// Invert a small dense matrix (DG_BASIS_SIZE x DG_BASIS_SIZE)
    static void InvertMatrix(const double* M, double* M_inv, int n);
};

#endif // ELEMENT_MATRICES_H
