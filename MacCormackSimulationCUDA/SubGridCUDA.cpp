#include "SubGridCUDA.h"
#include <stdexcept>
#include <cmath> // For ceil
#include <iostream> // For cerr

SubGridCUDA::SubGridCUDA(int process_id, int global_i_start, int global_i_end, int global_ny,
    double g, double tau, double delta_x_const, double delta_y, double min_calc_value,
    bool use_delta_x_array, const double* d_shared_delta_x_arr,
    const Grid& initial_host_grid)
    : process_id_(process_id),
    global_i_start_(global_i_start),
    global_i_end_(global_i_end),
    local_ny_(global_ny),
    g_(g),
    tau_(tau),
    delta_x_const_(delta_x_const),
    delta_y_(delta_y),
    min_calc_value_(min_calc_value),
    use_delta_x_array_(use_delta_x_array),
    d_shared_delta_x_arr_(d_shared_delta_x_arr),
    neighbor_left_(nullptr),
    neighbor_right_(nullptr),
    blockDim_(32, 8, 1) // 2026-05-17: 32×8 keeps warp inside single row (full coalescing)
{
    if (global_ny <= 0) {
        throw std::invalid_argument("SubGrid global_ny must be positive.");
    }
    if (global_i_end <= global_i_start) {
        throw std::invalid_argument("SubGrid global_i_end must be greater than global_i_start.");
    }

    // Calculate local width including 1 ghost cell on each side (left/right)
    local_nx_ = (global_i_end_ - global_i_start_) + 2;

    // Initialize dimensions in device data struct
    d_subgrid_data_.local_nx = local_nx_;
    d_subgrid_data_.local_ny = local_ny_;

    // NOTE: Host buffers for boundary exchange have been REMOVED (D2D-only mode).
    // All boundary data is now transferred directly between GPU memories.

    // Allocate memory on the GPU for this subgrid
    AllocateDeviceMemory();

    // Copy initial data slice from the host grid to device memory
    CopyInitialDataFromHost(initial_host_grid);

    // Calculate CUDA grid dimensions for kernels.
    // 2026-05-17: swapped — threadIdx.x → j (memory-fast), threadIdx.y → i.
    // gridDim.x must cover local_ny, gridDim.y must cover local_nx.
    gridDim_.x = static_cast<unsigned int>(std::ceil(static_cast<double>(local_ny_) / blockDim_.x));
    gridDim_.y = static_cast<unsigned int>(std::ceil(static_cast<double>(local_nx_) / blockDim_.y));
    gridDim_.z = 1;

    // Grid/Block dims for 1D internal boundary kernel (covers i-dimension)
    boundaryBlockDim_.x = 256; // Example 1D block size
    boundaryBlockDim_.y = 1;
    boundaryBlockDim_.z = 1;
    boundaryGridDim_.x = (local_nx_ + boundaryBlockDim_.x - 1) / boundaryBlockDim_.x;
    boundaryGridDim_.y = 1;
    boundaryGridDim_.z = 1;

}

SubGridCUDA::~SubGridCUDA() {
    FreeDeviceMemory();
}

void SubGridCUDA::AllocateDeviceMemory() {
    size_t localGridSize = static_cast<size_t>(local_nx_) * local_ny_;
    size_t gridSizeBytes = localGridSize * sizeof(double);

    if (localGridSize == 0) return; // Nothing to allocate

    try {
        checkCudaErrors(cudaMalloc(&d_subgrid_data_.H, gridSizeBytes));
        checkCudaErrors(cudaMalloc(&d_subgrid_data_.u, gridSizeBytes));
        checkCudaErrors(cudaMalloc(&d_subgrid_data_.v, gridSizeBytes));
        checkCudaErrors(cudaMalloc(&d_subgrid_data_.eta, gridSizeBytes));
        checkCudaErrors(cudaMalloc(&d_subgrid_data_.D, gridSizeBytes)); // Bathymetry

        checkCudaErrors(cudaMalloc(&d_subgrid_data_.H_hat, gridSizeBytes));
        checkCudaErrors(cudaMalloc(&d_subgrid_data_.u_hat, gridSizeBytes));
        checkCudaErrors(cudaMalloc(&d_subgrid_data_.v_hat, gridSizeBytes));
        checkCudaErrors(cudaMalloc(&d_subgrid_data_.eta_hat, gridSizeBytes));

        // Initialize hat variables to zero (optional, but good practice)
        checkCudaErrors(cudaMemset(d_subgrid_data_.H_hat, 0, gridSizeBytes));
        checkCudaErrors(cudaMemset(d_subgrid_data_.u_hat, 0, gridSizeBytes));
        checkCudaErrors(cudaMemset(d_subgrid_data_.v_hat, 0, gridSizeBytes));
        checkCudaErrors(cudaMemset(d_subgrid_data_.eta_hat, 0, gridSizeBytes));

    }
    catch (const std::runtime_error& e) {
        std::cerr << "CUDA Memory Allocation Failed for SubGrid " << process_id_ << ": " << e.what() << std::endl;
        FreeDeviceMemory(); // Clean up partial allocations
        throw;
    }
}

void SubGridCUDA::FreeDeviceMemory() {
    checkCudaErrors(cudaFree(d_subgrid_data_.H)); d_subgrid_data_.H = nullptr;
    checkCudaErrors(cudaFree(d_subgrid_data_.u)); d_subgrid_data_.u = nullptr;
    checkCudaErrors(cudaFree(d_subgrid_data_.v)); d_subgrid_data_.v = nullptr;
    checkCudaErrors(cudaFree(d_subgrid_data_.eta)); d_subgrid_data_.eta = nullptr;
    checkCudaErrors(cudaFree(d_subgrid_data_.D)); d_subgrid_data_.D = nullptr;

    checkCudaErrors(cudaFree(d_subgrid_data_.H_hat)); d_subgrid_data_.H_hat = nullptr;
    checkCudaErrors(cudaFree(d_subgrid_data_.u_hat)); d_subgrid_data_.u_hat = nullptr;
    checkCudaErrors(cudaFree(d_subgrid_data_.v_hat)); d_subgrid_data_.v_hat = nullptr;
    checkCudaErrors(cudaFree(d_subgrid_data_.eta_hat)); d_subgrid_data_.eta_hat = nullptr;
    // No need to free d_shared_delta_x_arr_, it's managed externally
}

void SubGridCUDA::CopyInitialDataFromHost(const Grid& host_grid) {
    if (!d_subgrid_data_.H) { // Check if memory is allocated
        throw std::runtime_error("Device memory not allocated before copying initial data.");
    }
    if (host_grid.GetNy() != local_ny_) {
        throw std::runtime_error("Host grid ny does not match subgrid local_ny.");
    }

    size_t rowSizeBytes = static_cast<size_t>(local_ny_) * sizeof(double);
    int num_compute_cols = global_i_end_ - global_i_start_;

    // Create temporary host buffers to hold the slice data for efficient copy
    std::vector<double> h_H(static_cast<size_t>(num_compute_cols) * local_ny_);
    std::vector<double> h_u(static_cast<size_t>(num_compute_cols) * local_ny_);
    std::vector<double> h_v(static_cast<size_t>(num_compute_cols) * local_ny_);
    std::vector<double> h_eta(static_cast<size_t>(num_compute_cols) * local_ny_);
    std::vector<double> h_D(static_cast<size_t>(num_compute_cols) * local_ny_);

    // Gather data from the host grid slice
    for (int i_local_comp = 0; i_local_comp < num_compute_cols; ++i_local_comp) {
        int global_i = global_i_start_ + i_local_comp;
        for (int j = 0; j < local_ny_; ++j) {
            size_t slice_idx = static_cast<size_t>(i_local_comp) * local_ny_ + j;
            h_H[slice_idx] = host_grid.H(global_i, j);
            h_u[slice_idx] = host_grid.u(global_i, j);
            h_v[slice_idx] = host_grid.v(global_i, j);
            h_eta[slice_idx] = host_grid.eta(global_i, j);
            h_D[slice_idx] = host_grid.D(global_i, j);
        }
    }

    // Copy the gathered slice to the device memory (computation part, i=1 to local_nx-2)
    // Use cudaMemcpy2D for potentially better performance if rows are large,
    // otherwise multiple cudaMemcpy calls work. Using cudaMemcpy for simplicity here.
    try {
        size_t sliceSizeBytes = static_cast<size_t>(num_compute_cols) * local_ny_ * sizeof(double);
        // Destination pointer starts at the first computational column (local i = 1)
        size_t offset = static_cast<size_t>(1) * local_ny_; // Offset by one column (left ghost cells)
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.H + offset, h_H.data(), sliceSizeBytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.u + offset, h_u.data(), sliceSizeBytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.v + offset, h_v.data(), sliceSizeBytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.eta + offset, h_eta.data(), sliceSizeBytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.D + offset, h_D.data(), sliceSizeBytes, cudaMemcpyHostToDevice));

        // Initialize ghost cells (i=0 and i=local_nx-1) - simple copy from neighbors initially
        // Note: This initial ghost cell state might be overwritten by the first boundary exchange.
        // Copy i=1 data to i=0
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.H, d_subgrid_data_.H + local_ny_, rowSizeBytes, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.u, d_subgrid_data_.u + local_ny_, rowSizeBytes, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.v, d_subgrid_data_.v + local_ny_, rowSizeBytes, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.eta, d_subgrid_data_.eta + local_ny_, rowSizeBytes, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.D, d_subgrid_data_.D + local_ny_, rowSizeBytes, cudaMemcpyDeviceToDevice)); // Copy depth too

        // Copy i=local_nx-2 data to i=local_nx-1
        size_t src_offset = static_cast<size_t>(local_nx_ - 2) * local_ny_;
        size_t dest_offset = static_cast<size_t>(local_nx_ - 1) * local_ny_;
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.H + dest_offset, d_subgrid_data_.H + src_offset, rowSizeBytes, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.u + dest_offset, d_subgrid_data_.u + src_offset, rowSizeBytes, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.v + dest_offset, d_subgrid_data_.v + src_offset, rowSizeBytes, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.eta + dest_offset, d_subgrid_data_.eta + src_offset, rowSizeBytes, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_subgrid_data_.D + dest_offset, d_subgrid_data_.D + src_offset, rowSizeBytes, cudaMemcpyDeviceToDevice)); // Copy depth too

    }
    catch (const std::runtime_error& e) {
        std::cerr << "CUDA Memcpy Host to Device Failed during initial data copy for SubGrid " << process_id_ << ": " << e.what() << std::endl;
        throw;
    }
}


void SubGridCUDA::SetNeighbors(SubGridCUDA* left, SubGridCUDA* right) {
    neighbor_left_ = left;
    neighbor_right_ = right;
}

void SubGridCUDA::ComputeHalfStep() {
    // Assumes ghost cells (i=0, i=nx-1, j=0, j=ny-1) are already updated from previous step/exchange
    // Kernels compute for i in [1, local_nx-2] and j in [1, local_ny-2]
    int global_j_offset = 0; // Assuming delta_x array uses global j indexing starting from 0
    // Adjust if the array has a different base or subgrid needs relative offset.

    launchSubGridComputeHalfStep(d_subgrid_data_, d_shared_delta_x_arr_, global_j_offset,
        g_, tau_, delta_x_const_, delta_y_, min_calc_value_,
        gridDim_, blockDim_);
    // No sync needed here if subsequent operations are in the same default stream
}

void SubGridCUDA::ComputeFullStep() {
    // Assumes HAT ghost cells are updated from boundary exchange
    int global_j_offset = 0; // Adjust as needed for delta_x array indexing

    launchSubGridComputeFullStep(d_subgrid_data_, d_shared_delta_x_arr_, global_j_offset,
        g_, tau_, delta_x_const_, delta_y_, min_calc_value_,
        gridDim_, blockDim_);
    // No sync needed here typically
}

// --- Boundary Exchange ---

// NOTE: Legacy host-staging functions CopyBoundaryColumnGPUToHost and CopyBoundaryColumnHostToGPU
// have been REMOVED to enforce D2D-only boundary exchange mode.
// This prevents accidental regression to slower D2H/H2D transfers.
// All boundary exchange is now performed via direct cudaMemcpyDeviceToDevice calls.

// --- D2D Boundary Exchange Methods ---

double* SubGridCUDA::GetDevicePointer(const std::string& var_name, bool is_hat) const {
    if (var_name == "H") return is_hat ? d_subgrid_data_.H_hat : d_subgrid_data_.H;
    if (var_name == "u") return is_hat ? d_subgrid_data_.u_hat : d_subgrid_data_.u;
    if (var_name == "v") return is_hat ? d_subgrid_data_.v_hat : d_subgrid_data_.v;
    if (var_name == "eta") return is_hat ? d_subgrid_data_.eta_hat : d_subgrid_data_.eta;
    return nullptr;
}

void SubGridCUDA::CopyBoundaryColumnD2D(double* d_src, int src_col, double* d_dest, int dest_col, int ny, bool is_hat_src, bool is_hat_dest) {
    // This method copies a column directly from one device array to another (D2D)
    // d_src and d_dest are base pointers (H, u, v, or eta), we select hat variant if needed
    
    size_t src_offset = static_cast<size_t>(src_col) * ny;
    size_t dest_offset = static_cast<size_t>(dest_col) * ny;
    size_t columnSizeBytes = static_cast<size_t>(ny) * sizeof(double);
    
    checkCudaErrors(cudaMemcpy(d_dest + dest_offset, d_src + src_offset, columnSizeBytes, cudaMemcpyDeviceToDevice));
}


// Prepare buffers FOR sending data TO neighbors
// With D2D exchange, this is now a no-op - data stays on GPU and is copied directly in Receive*
void SubGridCUDA::PrepareLeftBoundarySendBuffer(bool for_hat_step) {
    // D2D: No preparation needed - data will be copied directly from GPU to neighbor's GPU in ReceiveRightBoundaryData
    // Left send buffer (column i=1) will be read directly by the left neighbor
    (void)for_hat_step; // Suppress unused parameter warning
}

void SubGridCUDA::PrepareRightBoundarySendBuffer(bool for_hat_step) {
    // D2D: No preparation needed - data will be copied directly from GPU to neighbor's GPU in ReceiveLeftBoundaryData
    // Right send buffer (column i=local_nx-2) will be read directly by the right neighbor
    (void)for_hat_step; // Suppress unused parameter warning
}

// Receive data FROM neighbors into GPU ghost cells - using D2D copy
void SubGridCUDA::ReceiveLeftBoundaryData(bool for_hat_step) {
    if (!neighbor_left_) { // No left neighbor, apply reflective/copy boundary condition
        // Copy local column i=1 to ghost column i=0
        size_t rowSizeBytes = static_cast<size_t>(local_ny_) * sizeof(double);
        double* d_H = for_hat_step ? d_subgrid_data_.H_hat : d_subgrid_data_.H;
        double* d_u = for_hat_step ? d_subgrid_data_.u_hat : d_subgrid_data_.u;
        double* d_v = for_hat_step ? d_subgrid_data_.v_hat : d_subgrid_data_.v;
        double* d_eta = for_hat_step ? d_subgrid_data_.eta_hat : d_subgrid_data_.eta;
        if (d_H) checkCudaErrors(cudaMemcpy(d_H, d_H + local_ny_, rowSizeBytes, cudaMemcpyDeviceToDevice));
        if (d_u) checkCudaErrors(cudaMemcpy(d_u, d_u + local_ny_, rowSizeBytes, cudaMemcpyDeviceToDevice));
        if (d_v) checkCudaErrors(cudaMemcpy(d_v, d_v + local_ny_, rowSizeBytes, cudaMemcpyDeviceToDevice));
        if (d_eta) checkCudaErrors(cudaMemcpy(d_eta, d_eta + local_ny_, rowSizeBytes, cudaMemcpyDeviceToDevice));
    }
    else {
        // D2D: Copy directly from neighbor's GPU memory (their column local_nx-2) to our ghost column (i=0)
        // Neighbor's "right inner boundary" is at their (local_nx - 2), we receive into our ghost column 0
        int neighbor_src_col = neighbor_left_->local_nx_ - 2; // Neighbor's inner right boundary column
        int my_dest_col = 0; // Our left ghost column
        
        // Get pointers to neighbor's device arrays
        double* neighbor_H = neighbor_left_->GetDevicePointer("H", for_hat_step);
        double* neighbor_u = neighbor_left_->GetDevicePointer("u", for_hat_step);
        double* neighbor_v = neighbor_left_->GetDevicePointer("v", for_hat_step);
        double* neighbor_eta = neighbor_left_->GetDevicePointer("eta", for_hat_step);
        
        // Get pointers to my device arrays
        double* my_H = GetDevicePointer("H", for_hat_step);
        double* my_u = GetDevicePointer("u", for_hat_step);
        double* my_v = GetDevicePointer("v", for_hat_step);
        double* my_eta = GetDevicePointer("eta", for_hat_step);
        
        size_t columnSizeBytes = static_cast<size_t>(local_ny_) * sizeof(double);
        size_t src_offset = static_cast<size_t>(neighbor_src_col) * local_ny_;
        size_t dest_offset = static_cast<size_t>(my_dest_col) * local_ny_;
        
        // Direct D2D copy from neighbor to this subgrid
        if (neighbor_H && my_H) checkCudaErrors(cudaMemcpy(my_H + dest_offset, neighbor_H + src_offset, columnSizeBytes, cudaMemcpyDeviceToDevice));
        if (neighbor_u && my_u) checkCudaErrors(cudaMemcpy(my_u + dest_offset, neighbor_u + src_offset, columnSizeBytes, cudaMemcpyDeviceToDevice));
        if (neighbor_v && my_v) checkCudaErrors(cudaMemcpy(my_v + dest_offset, neighbor_v + src_offset, columnSizeBytes, cudaMemcpyDeviceToDevice));
        if (neighbor_eta && my_eta) checkCudaErrors(cudaMemcpy(my_eta + dest_offset, neighbor_eta + src_offset, columnSizeBytes, cudaMemcpyDeviceToDevice));
    }
}

void SubGridCUDA::ReceiveRightBoundaryData(bool for_hat_step) {
    if (!neighbor_right_) { // No right neighbor, apply reflective/copy boundary condition
        // Copy local column i=local_nx-2 to ghost column i=local_nx-1
        size_t rowSizeBytes = static_cast<size_t>(local_ny_) * sizeof(double);
        size_t src_offset = static_cast<size_t>(local_nx_ - 2) * local_ny_;
        size_t dest_offset = static_cast<size_t>(local_nx_ - 1) * local_ny_;
        double* d_H = for_hat_step ? d_subgrid_data_.H_hat : d_subgrid_data_.H;
        double* d_u = for_hat_step ? d_subgrid_data_.u_hat : d_subgrid_data_.u;
        double* d_v = for_hat_step ? d_subgrid_data_.v_hat : d_subgrid_data_.v;
        double* d_eta = for_hat_step ? d_subgrid_data_.eta_hat : d_subgrid_data_.eta;
        if (d_H) checkCudaErrors(cudaMemcpy(d_H + dest_offset, d_H + src_offset, rowSizeBytes, cudaMemcpyDeviceToDevice));
        if (d_u) checkCudaErrors(cudaMemcpy(d_u + dest_offset, d_u + src_offset, rowSizeBytes, cudaMemcpyDeviceToDevice));
        if (d_v) checkCudaErrors(cudaMemcpy(d_v + dest_offset, d_v + src_offset, rowSizeBytes, cudaMemcpyDeviceToDevice));
        if (d_eta) checkCudaErrors(cudaMemcpy(d_eta + dest_offset, d_eta + src_offset, rowSizeBytes, cudaMemcpyDeviceToDevice));
    }
    else {
        // D2D: Copy directly from neighbor's GPU memory (their column 1) to our ghost column (local_nx-1)
        // Neighbor's "left inner boundary" is at their column 1, we receive into our right ghost column
        int neighbor_src_col = 1; // Neighbor's inner left boundary column
        int my_dest_col = local_nx_ - 1; // Our right ghost column
        
        // Get pointers to neighbor's device arrays
        double* neighbor_H = neighbor_right_->GetDevicePointer("H", for_hat_step);
        double* neighbor_u = neighbor_right_->GetDevicePointer("u", for_hat_step);
        double* neighbor_v = neighbor_right_->GetDevicePointer("v", for_hat_step);
        double* neighbor_eta = neighbor_right_->GetDevicePointer("eta", for_hat_step);
        
        // Get pointers to my device arrays
        double* my_H = GetDevicePointer("H", for_hat_step);
        double* my_u = GetDevicePointer("u", for_hat_step);
        double* my_v = GetDevicePointer("v", for_hat_step);
        double* my_eta = GetDevicePointer("eta", for_hat_step);
        
        size_t columnSizeBytes = static_cast<size_t>(local_ny_) * sizeof(double);
        size_t src_offset = static_cast<size_t>(neighbor_src_col) * neighbor_right_->local_ny_;
        size_t dest_offset = static_cast<size_t>(my_dest_col) * local_ny_;
        
        // Direct D2D copy from neighbor to this subgrid
        if (neighbor_H && my_H) checkCudaErrors(cudaMemcpy(my_H + dest_offset, neighbor_H + src_offset, columnSizeBytes, cudaMemcpyDeviceToDevice));
        if (neighbor_u && my_u) checkCudaErrors(cudaMemcpy(my_u + dest_offset, neighbor_u + src_offset, columnSizeBytes, cudaMemcpyDeviceToDevice));
        if (neighbor_v && my_v) checkCudaErrors(cudaMemcpy(my_v + dest_offset, neighbor_v + src_offset, columnSizeBytes, cudaMemcpyDeviceToDevice));
        if (neighbor_eta && my_eta) checkCudaErrors(cudaMemcpy(my_eta + dest_offset, neighbor_eta + src_offset, columnSizeBytes, cudaMemcpyDeviceToDevice));
    }
}


void SubGridCUDA::ApplyInternalBoundaries(bool for_hat_step) {
    // Launch kernel to copy j=1 to j=0 and j=ny-2 to j=ny-1
    launchSubGridInternalBoundary(d_subgrid_data_, for_hat_step, boundaryGridDim_, boundaryBlockDim_);
}

// --- Result Handling ---
void SubGridCUDA::CopyResultToHostGrid(Grid& target_host_grid, const std::string& variable_name) {
    if (target_host_grid.GetNy() != local_ny_) {
        std::cerr << "Warning: Target host grid ny mismatch in CopyResultToHostGrid for subgrid " << process_id_ << std::endl;
        return;
    }
    if (global_i_end_ > target_host_grid.GetNx()) {
        std::cerr << "Warning: Subgrid global_i_end exceeds target host grid nx for subgrid " << process_id_ << std::endl;
        return;
    }

    double* d_source_ptr = nullptr;
    double* h_dest_base_ptr = nullptr;

    if (variable_name == "H") {
        d_source_ptr = d_subgrid_data_.H;
        h_dest_base_ptr = target_host_grid.GetHData();
    }
    else if (variable_name == "u") {
        d_source_ptr = d_subgrid_data_.u;
        h_dest_base_ptr = target_host_grid.GetUData();
    }
    else if (variable_name == "v") {
        d_source_ptr = d_subgrid_data_.v;
        h_dest_base_ptr = target_host_grid.GetVData();
    }
    else if (variable_name == "eta") {
        d_source_ptr = d_subgrid_data_.eta;
        h_dest_base_ptr = target_host_grid.GetEtaData();
    }
    // Add eta_max if managing it per-subgrid and copying back
    else {
        std::cerr << "Error: Unknown variable name '" << variable_name << "' for CopyResultToHostGrid." << std::endl;
        return;
    }

    if (!d_source_ptr || !h_dest_base_ptr) {
        std::cerr << "Error: Null pointers encountered in CopyResultToHostGrid for " << variable_name << std::endl;
        return;
    }

    // Copy the computational part (excluding ghost cells) back to the host grid slice
    int num_compute_cols = global_i_end_ - global_i_start_;
    size_t computeSliceSize = static_cast<size_t>(num_compute_cols) * local_ny_;
    size_t computeSliceBytes = computeSliceSize * sizeof(double);

    // Create a temporary host buffer to receive the data
    std::vector<double> h_temp_buffer(computeSliceSize);

    // Source pointer starts at the first computational column (local i = 1)
    size_t src_offset = static_cast<size_t>(1) * local_ny_;

    // Copy from device (computation part) to temporary host buffer
    checkCudaErrors(cudaMemcpy(h_temp_buffer.data(), d_source_ptr + src_offset, computeSliceBytes, cudaMemcpyDeviceToHost));

    // Copy from temporary host buffer to the correct slice in the target host grid
    for (int i_local_comp = 0; i_local_comp < num_compute_cols; ++i_local_comp) {
        int global_i = global_i_start_ + i_local_comp;
        double* h_dest_col_ptr = h_dest_base_ptr + static_cast<size_t>(global_i) * local_ny_;
        const double* h_src_col_ptr = h_temp_buffer.data() + static_cast<size_t>(i_local_comp) * local_ny_;
        std::copy(h_src_col_ptr, h_src_col_ptr + local_ny_, h_dest_col_ptr);
    }
}


// NOTE: Host buffer accessor functions (GetLeftSendBuffer, GetRightSendBuffer, 
// GetLeftRecvBuffer, GetRightRecvBuffer) have been REMOVED.
// All boundary exchange is now D2D-only.
// This compile-time removal prevents any code from accidentally using host-staging.