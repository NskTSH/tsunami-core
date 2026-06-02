// BlockSparseMatrix.h - Block-sparse matrix for DG system
// Each block is DG_BLOCK_SIZE x DG_BLOCK_SIZE, stored per-element with neighbor contributions

#ifndef BLOCK_SPARSE_MATRIX_H
#define BLOCK_SPARSE_MATRIX_H

#include "TaylorBasis.h"
#include <vector>
#include <array>
#include <cstring>

/// Block-sparse matrix for DG linear system B * q^{n+1} = rhs
///
/// Structure: for each element k, store:
/// - Diagonal block B_kk (DG_BLOCK_SIZE x DG_BLOCK_SIZE)
/// - Up to 3 off-diagonal blocks B_kj for neighbor elements
///
/// For explicit schemes, only diagonal blocks are needed (B = M).
/// For semi-implicit (Crank-Nicolson sources), off-diagonal blocks may appear.
class BlockSparseMatrix {
public:
    struct ElementBlock {
        // Diagonal block
        std::array<double, DG_BLOCK_SIZE * DG_BLOCK_SIZE> diag;

        // Off-diagonal blocks for up to 3 neighbors
        // neighbor_idx[i] = global element index, -1 if no neighbor
        std::array<int, 3> neighbor_idx;
        std::array<std::array<double, DG_BLOCK_SIZE * DG_BLOCK_SIZE>, 3> off_diag;

        ElementBlock() {
            diag.fill(0.0);
            neighbor_idx.fill(-1);
            for (auto& b : off_diag) b.fill(0.0);
        }
    };

    /// Allocate for given number of elements
    void Allocate(int num_elements);

    /// Access element block
    ElementBlock& GetBlock(int elem_idx) { return blocks_[elem_idx]; }
    const ElementBlock& GetBlock(int elem_idx) const { return blocks_[elem_idx]; }

    int NumElements() const { return static_cast<int>(blocks_.size()); }

    /// Zero all blocks
    void Zero();

    /// Solve B * x = rhs using block-Jacobi iteration
    /// @param rhs Right-hand side (num_elements * DG_BLOCK_SIZE)
    /// @param x Solution (num_elements * DG_BLOCK_SIZE), used as initial guess
    /// @param M_inv Inverse mass matrices for preconditioner (from ElementMatrices)
    /// @param max_iter Maximum iterations
    /// @param tol Convergence tolerance (L2 residual)
    /// @return Number of iterations performed
    int SolveBlockJacobi(const double* rhs, double* x,
                         const std::vector<std::array<double, DG_BLOCK_SIZE * DG_BLOCK_SIZE>>& diag_inv,
                         int max_iter = 100, double tol = 1e-10) const;

    /// For explicit scheme: just apply M_inv * rhs (no off-diagonal)
    /// This is the fast path when B = M (diagonal mass matrix)
    void ApplyDiagonalInverse(const double* rhs, double* x,
                              const std::vector<std::array<double, DG_BLOCK_SIZE * DG_BLOCK_SIZE>>& diag_inv) const;

private:
    std::vector<ElementBlock> blocks_;

    /// Multiply a single DG_BLOCK_SIZE x DG_BLOCK_SIZE block by a vector
    static void BlockMulVec(const double* block, const double* vec, double* result);

    /// Add block * vec to result
    static void BlockMulVecAdd(const double* block, const double* vec, double* result);
};

#endif // BLOCK_SPARSE_MATRIX_H
