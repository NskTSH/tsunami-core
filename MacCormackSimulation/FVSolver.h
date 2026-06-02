// FVSolver.h — per-SubGrid FV Godunov backend
//
// Phase A (Codex Audusse FV design 2026-05-28). One FVSolver instance per
// SubGrid; owns conservative working buffers and implements SSP-RK2 stages.
//
// Integration model:
//   * SubGrid::ComputeHalfStep() — if FV mode → fv_solver_->Stage1Predictor()
//     reads primitive main state, builds U0 (conservative), does Euler step,
//     writes primitive hat arrays back so existing exchange/output paths work.
//   * Exchange hat-boundary data (existing primitive copy + reflective handling
//     for FV walls — Phase B).
//   * SubGrid::ComputeFullStep() — if FV mode → fv_solver_->Stage2Corrector()
//     reads primitive main + hat (exchanged), does Heun corrector, writes
//     primitive main back.
//
// Phase A scope: Cartesian only, flat or step bathymetry, first-order HR-HLL,
// reflective walls. NO TVD/Froude/Manning legacy features.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "FVTypes.h"
#include "Precision.h"

class SubGrid;  // forward

namespace fv {

// Phase E pre-paper (Codex 2026-05-28): global positivity-clamp counter.
// Incremented once for every cell where the SSP-RK2 stage produces h <= dry_tol
// and the code resets (h, hu, hv) -> 0. For closed wet scenarios this should
// stay at 0 (no drying). For wet/dry scenarios (Monai, dam-break-wet-dry) it
// quantifies how aggressive the floor activates. Logged at end of Run.
extern std::atomic<std::uint64_t> g_positivity_clamps;

class FVSolver {
public:
    explicit FVSolver(SubGrid* owner);
    ~FVSolver() = default;

    FVSolver(const FVSolver&) = delete;
    FVSolver& operator=(const FVSolver&) = delete;

    // Phase A: SSP-RK2 Heun stage 1 (Euler predictor in primitive output).
    void Stage1Predictor();

    // SSP-RK2 Heun stage 2 (corrector → main state).
    void Stage2Corrector();

private:
    // Convert owner's primitive main state to conservative U0; cached at
    // start of each timestep (Stage1).
    void LoadConservativeFromPrimitive();

    // Write a conservative state buffer to owner's primitive hat arrays
    // (used after Stage1).
    void WriteConservativeToHat(const std::vector<FVState>& U_stage);

    // Write a conservative state buffer to owner's primitive main arrays
    // (used after Stage2).
    void WriteConservativeToMain(const std::vector<FVState>& U_stage);

    // Compute Euler RHS: dU/dt = -dF_x/dx - dF_y/dy + S(U, D).
    // Inputs: conservative state buffer, output: per-cell L(U).
    void ComputeRHS(const std::vector<FVState>& U,
                    std::vector<FVState>& L_out);

private:
    SubGrid* owner_;

    // Local copies sized to local_nx_ * local_ny_; allocated lazily.
    std::vector<FVState> U0_;   // state at start of timestep (frozen)
    std::vector<FVState> U1_;   // intermediate stage state (after stage1)
    std::vector<FVState> L_;    // scratch RHS buffer

    int nx_cached_ = 0;
    int ny_cached_ = 0;

    void EnsureBuffers();
};

}  // namespace fv
