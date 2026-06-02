#pragma once

#include "Grid.h" // For host grid interaction
#include "SimulationKernels.h" // For SubGridCudaData and kernel launchers
#include "CudaUtils.h" // For checkCudaErrors
#include <vector>
#include <memory> // For unique_ptr

// This class implements D2D-only boundary exchange mode.
// All ghost cell data is transferred directly between GPU memories via cudaMemcpyDeviceToDevice.
// Host-staging boundary exchange code has been permanently removed from this codebase.
// There are NO compile-time flags to re-enable host-based boundary exchange.

class SubGridCUDA {
public:
    int process_id_;        // ID of this subgrid
    int global_i_start_;    // Starting global column index (inclusive)
    int global_i_end_;      // Ending global column index (exclusive)
    int local_nx_;          // Local width (including ghost cells)
    int local_ny_;          // Local height (same as global ny)

    // Simulation parameters (copied from main simulation)
    double g_;
    double tau_;
    double delta_x_const_;
    double delta_y_;
    double min_calc_value_;
    bool use_delta_x_array_; // Does this subgrid use the shared array?
    const double* d_shared_delta_x_arr_ = nullptr; // Pointer to shared device array for delta_x

    // CUDA Data
    SubGridCudaData d_subgrid_data_; // Holds device pointers for this subgrid
    dim3 blockDim_;
    dim3 gridDim_;           // Grid dim for compute kernels (covers local_nx x local_ny)
    dim3 boundaryGridDim_;   // Grid dim for 1D internal boundary kernel
    dim3 boundaryBlockDim_;  // Block dim for 1D internal boundary kernel

    // NOTE: Host buffers for boundary exchange have been REMOVED.
    // All boundary exchange is now D2D (Device-to-Device) only.
    // This prevents regression to slower D2H/H2D transfers.

    // Pointers to neighbors (raw pointers okay if lifetime managed by SingleLineGridSimulation)
    SubGridCUDA* neighbor_left_ = nullptr;
    SubGridCUDA* neighbor_right_ = nullptr;

    // Private helpers
    void AllocateDeviceMemory();
    void FreeDeviceMemory();
    void CopyInitialDataFromHost(const Grid& host_grid);

    // D2D boundary exchange helpers - direct GPU-to-GPU copy without host staging
    void CopyBoundaryColumnD2D(double* d_src, int src_col, double* d_dest, int dest_col, int ny, bool is_hat_src, bool is_hat_dest);
    double* GetDevicePointer(const std::string& var_name, bool is_hat) const;
    int GetLocalNy() const { return local_ny_; }

    SubGridCUDA(int process_id, int global_i_start, int global_i_end, int global_ny,
        double g, double tau, double delta_x_const, double delta_y, double min_calc_value,
        bool use_delta_x_array, const double* d_shared_delta_x_arr, // Pass shared array pointer
        const Grid& initial_host_grid); // Pass initial host grid for setup

    ~SubGridCUDA();

    // Disable copy/move
    SubGridCUDA(const SubGridCUDA&) = delete;
    SubGridCUDA& operator=(const SubGridCUDA&) = delete;
    SubGridCUDA(SubGridCUDA&&) = delete;
    SubGridCUDA& operator=(SubGridCUDA&&) = delete;

    void SetNeighbors(SubGridCUDA* left, SubGridCUDA* right);

    // --- Main Computation Steps ---
    void ComputeHalfStep();
    void ComputeFullStep();

    // --- Boundary Exchange (D2D only) ---
    // Prepare data to be sent TO neighbors - no-op in D2D mode (data stays on GPU)
    void PrepareLeftBoundarySendBuffer(bool for_hat_step);
    void PrepareRightBoundarySendBuffer(bool for_hat_step);
    // Receive data FROM neighbors via direct D2D copy
    void ReceiveLeftBoundaryData(bool for_hat_step);
    void ReceiveRightBoundaryData(bool for_hat_step);
    // Apply internal (Top/Bottom) boundary conditions
    void ApplyInternalBoundaries(bool for_hat_step);

    // --- Result Handling ---
    void CopyResultToHostGrid(Grid& target_host_grid, const std::string& variable_name);

    int GetProcessId() const { return process_id_; }
};