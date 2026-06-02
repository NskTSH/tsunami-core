// DGSolverCUDA.cuh
// CUDA port of the unstructured P2 Discontinuous-Galerkin shallow-water solver.
//
// Strategy (Codex-reviewed): the entire HOST-side setup of the CPU DG solver
// (TriMesh + ElementMatrices + DGSolver: mesh, mass matrices, quadrature, the
// well-balanced initial condition AND the bathymetry-positivity fix) is REUSED by
// compiling the CPU DG host sources into this CUDA project. Only the per-step hot
// loop (ComputeResidual = volume + surface + source, SSP-RK3, slope + positivity
// limiters) is ported to CUDA kernels operating on flat per-element/per-edge SoA
// device arrays uploaded once after host setup. FP64 is used for the DG state /
// bathymetry / residual (FP32 is too fragile for the well-balanced cancellation on
// deep bathymetry). This does NOT touch the existing structured-MacCormack GPU
// code; the DG solver + DG/MC hybrid are added alongside it.
//
// Phase 0: validate that the host reuse compiles, links and runs inside the CUDA
// executable, reproducing the CPU lake-at-rest residual (so the bathy-positivity
// well-balancing holds in this build before any kernels are written).

#ifndef DG_SOLVER_CUDA_CUH
#define DG_SOLVER_CUDA_CUH

// Entry point for the DG-on-GPU self test / milestone driver. Returns process exit
// code (0 = ok). Dispatched from main.cpp via the --dgtest CLI flag.
int RunDGCudaTest();

// Runs the REAL S7 (Japan Sea) MacCormack+DG hybrid on the GPU: real ASC bathymetry
// + S7 ellipsoidal source, MC west / DG east, two-way seam coupling. --s7hybrid.
int RunS7HybridGPU();

// NATIVE-resolution S7: 8 vertical bands, band #3 = DG (holds source), rest = MC
// (two instances), MC|DG|MC two-seam coupling. Writes full eta_max map. --s7full.
int RunS7HybridFullGPU();

#endif // DG_SOLVER_CUDA_CUH
