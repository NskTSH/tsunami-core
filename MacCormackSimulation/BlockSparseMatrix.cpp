// BlockSparseMatrix.cpp - Block-sparse matrix operations

#include "BlockSparseMatrix.h"
#include <cmath>
#include <algorithm>
#include <cstring>

void BlockSparseMatrix::Allocate(int num_elements) {
    blocks_.resize(num_elements);
}

void BlockSparseMatrix::Zero() {
    for (auto& b : blocks_) {
        b.diag.fill(0.0);
        for (auto& od : b.off_diag) od.fill(0.0);
    }
}

void BlockSparseMatrix::BlockMulVec(const double* block, const double* vec, double* result) {
    for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
        double sum = 0.0;
        for (int j = 0; j < DG_BLOCK_SIZE; ++j) {
            sum += block[i * DG_BLOCK_SIZE + j] * vec[j];
        }
        result[i] = sum;
    }
}

void BlockSparseMatrix::BlockMulVecAdd(const double* block, const double* vec, double* result) {
    for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
        double sum = 0.0;
        for (int j = 0; j < DG_BLOCK_SIZE; ++j) {
            sum += block[i * DG_BLOCK_SIZE + j] * vec[j];
        }
        result[i] += sum;
    }
}

void BlockSparseMatrix::ApplyDiagonalInverse(
    const double* rhs, double* x,
    const std::vector<std::array<double, DG_BLOCK_SIZE * DG_BLOCK_SIZE>>& diag_inv) const
{
    int n = NumElements();
    for (int k = 0; k < n; ++k) {
        BlockMulVec(diag_inv[k].data(),
                    rhs + k * DG_BLOCK_SIZE,
                    x + k * DG_BLOCK_SIZE);
    }
}

int BlockSparseMatrix::SolveBlockJacobi(
    const double* rhs, double* x,
    const std::vector<std::array<double, DG_BLOCK_SIZE * DG_BLOCK_SIZE>>& diag_inv,
    int max_iter, double tol) const
{
    int n = NumElements();
    int total_size = n * DG_BLOCK_SIZE;

    std::vector<double> x_new(total_size);
    std::vector<double> residual(DG_BLOCK_SIZE);

    for (int iter = 0; iter < max_iter; ++iter) {
        double norm2 = 0.0;

        for (int k = 0; k < n; ++k) {
            const auto& block = blocks_[k];

            // residual_k = rhs_k - sum(B_kj * x_j) for j != k
            std::memcpy(residual.data(), rhs + k * DG_BLOCK_SIZE, DG_BLOCK_SIZE * sizeof(double));

            // Subtract off-diagonal contributions
            for (int e = 0; e < 3; ++e) {
                int j = block.neighbor_idx[e];
                if (j < 0) continue;

                // residual -= B_kj * x_j
                for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
                    double sum = 0.0;
                    for (int jj = 0; jj < DG_BLOCK_SIZE; ++jj) {
                        sum += block.off_diag[e][i * DG_BLOCK_SIZE + jj]
                             * x[j * DG_BLOCK_SIZE + jj];
                    }
                    residual[i] -= sum;
                }
            }

            // x_new_k = D_k^{-1} * residual_k
            BlockMulVec(diag_inv[k].data(), residual.data(),
                        x_new.data() + k * DG_BLOCK_SIZE);

            // Accumulate convergence check
            for (int i = 0; i < DG_BLOCK_SIZE; ++i) {
                double diff = x_new[k * DG_BLOCK_SIZE + i] - x[k * DG_BLOCK_SIZE + i];
                norm2 += diff * diff;
            }
        }

        // Update
        std::memcpy(x, x_new.data(), total_size * sizeof(double));

        if (std::sqrt(norm2) < tol) {
            return iter + 1;
        }
    }

    return max_iter;
}
