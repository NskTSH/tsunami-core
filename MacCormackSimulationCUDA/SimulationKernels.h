#ifndef SIMULATION_KERNELS_H
#define SIMULATION_KERNELS_H

#include <cuda_runtime.h>

// Structure to hold pointers for a SINGLE SubGrid's data on the device
struct SubGridCudaData {
    double* H = nullptr;
    double* u = nullptr;
    double* v = nullptr;
    double* eta = nullptr;
    double* D = nullptr; // Bathymetry/Depth

    double* H_hat = nullptr;
    double* u_hat = nullptr;
    double* v_hat = nullptr;
    double* eta_hat = nullptr;

    // Dimensions of this subgrid's LOCAL data (including ghost cells)
    int local_nx = 0;
    int local_ny = 0;
};


// Function to launch the ComputeHalfStep kernel for a subgrid
void launchSubGridComputeHalfStep(const SubGridCudaData& d_subgrid,
    const double* d_delta_x_arr, // Pointer to SHARED delta_x array (or nullptr)
    int global_j_offset,       // Needed if delta_x depends on global j
    double g, double tau, double delta_x_const, double delta_y,
    double min_calc_value,
    dim3 gridDim, dim3 blockDim);

// Function to launch the ComputeFullStep kernel for a subgrid
void launchSubGridComputeFullStep(const SubGridCudaData& d_subgrid,
    const double* d_delta_x_arr, // Pointer to SHARED delta_x array (or nullptr)
    int global_j_offset,       // Needed if delta_x depends on global j
    double g, double tau, double delta_x_const, double delta_y,
    double min_calc_value,
    dim3 gridDim, dim3 blockDim);

// Kernel to handle internal boundary conditions (Top/Bottom edges within subgrid)
// This copies j=1 data to j=0 and j=ny-2 data to j=ny-1
void launchSubGridInternalBoundary(const SubGridCudaData& d_subgrid, bool apply_to_hat, dim3 gridDim, dim3 blockDim);

// Kernel to update eta_max within a subgrid (if keeping eta_max on GPU per subgrid)
// Note: Managing eta_max across subgrids on GPU is complex, easier on host after copy-back.
// void launchSubGridUpdateEtaMax(const SubGridCudaData& d_subgrid, double* d_subgrid_eta_max, dim3 gridDim, dim3 blockDim);


#endif // SIMULATION_KERNELS_H