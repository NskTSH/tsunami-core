#include "SingleLineGridSimulationSubgridCUDA.h"
#include "CudaUtils.h" // Include CUDA utilities for error checking
#include <cmath>     // For std::cos, M_PI, ceil
#include <stdexcept>
#include <numeric> // For std::iota
#include <filesystem>

namespace fs = std::filesystem;

SingleLineGridSimulationSubgridCUDA::SingleLineGridSimulationSubgridCUDA()
	: g_(9.81), tau_(1.0), delta_y_(1.0), delta_x_const_(1.0),
	nx_(0), ny_(0), timesteps_(100), save_interval_(10), min_calc_value_(0.01),
	num_subgrids_(1), // Default to 1 subgrid
	host_grid_(0, 0),
	use_delta_x_array_(false),
	d_shared_delta_x_arr_(nullptr)
{
}

SingleLineGridSimulationSubgridCUDA::~SingleLineGridSimulationSubgridCUDA() {
	FreeSharedCudaMemory(); // subgrids_ unique_ptrs handle their own memory
}

void SingleLineGridSimulationSubgridCUDA::FreeSharedCudaMemory() {
	if (d_shared_delta_x_arr_) {
		checkCudaErrors(cudaFree(d_shared_delta_x_arr_));
		d_shared_delta_x_arr_ = nullptr;
	}
}

void SingleLineGridSimulationSubgridCUDA::PrecomputeDeltaX_GPU() {
	// Identical logic to the previous version's PrecomputeDeltaX_GPU
	// Ensures the shared d_shared_delta_x_arr_ is allocated and filled if needed.
	if (!use_delta_x_array_ || ny_ <= 0) return;
	if (d_shared_delta_x_arr_) {
		std::cout << "Shared delta_x array already computed." << std::endl;
		return; // Already done
	}

	size_t deltaXArraySize = static_cast<size_t>(ny_);
	size_t deltaXArrayBytes = deltaXArraySize * sizeof(double);
	std::vector<double> h_delta_x(deltaXArraySize);

	if (!host_grid_.IsInitialized()) {
		throw std::runtime_error("Host grid must be initialized before precomputing delta_x");
	}

	std::cout << "Precomputing shared delta_x array for GPU..." << std::endl;
	for (int j = 0; j < ny_; ++j) {
		h_delta_x[j] = host_grid_.GetDeltaX(0, j); // Use host grid's function
		if (h_delta_x[j] <= 0) {
			std::cerr << "Warning: Calculated delta_x <= 0 for j=" << j << ". Value: " << h_delta_x[j] << std::endl;
		}
	}

	checkCudaErrors(cudaMalloc(&d_shared_delta_x_arr_, deltaXArrayBytes));
	checkCudaErrors(cudaMemcpy(d_shared_delta_x_arr_, h_delta_x.data(), deltaXArrayBytes, cudaMemcpyHostToDevice));
	std::cout << "Shared delta_x array copied to GPU." << std::endl;
}


void SingleLineGridSimulationSubgridCUDA::SetParamsTest1(int num_subgrids) {
	g_ = 9.81;
	tau_ = 5.0;
	delta_x_const_ = 1000.0;
	delta_y_ = 1000.0;
	nx_ = 120; // Example size
	ny_ = 120;
	timesteps_ = 500;
	save_interval_ = 50;
	min_calc_value_ = 0.01;
	num_subgrids_ = std::max(1, num_subgrids); // Ensure at least 1 subgrid
	use_delta_x_array_ = false; // Constant delta_x for this test

	// Initialize host grid
	host_grid_.ResizeAndAllocate(nx_, ny_, [dx = delta_x_const_](int, int) { return dx; }, delta_y_);
	host_grid_.SetParamsTest1();
	if (!host_grid_.IsInitialized()) {
		throw std::runtime_error("Host grid initialization failed for Test1!");
	}

	// Clear previous subgrids if any
	subgrids_.clear();
	FreeSharedCudaMemory(); // Free old delta_x array if exists

	// --- Create Subgrids ---
	std::cout << "Creating " << num_subgrids_ << " subgrids for Test1..." << std::endl;
	int base_subgrid_width = nx_ / num_subgrids_;
	int remainder = nx_ % num_subgrids_;
	int current_i_start = 0;

	for (int pid = 0; pid < num_subgrids_; ++pid) {
		int subgrid_width = base_subgrid_width + (pid < remainder ? 1 : 0);
		int current_i_end = current_i_start + subgrid_width;

		if (subgrid_width <= 0) {
			std::cerr << "Warning: Subgrid " << pid << " has zero width. Skipping." << std::endl;
			continue;
		}

		subgrids_.push_back(std::make_unique<SubGridCUDA>(
			pid,
			current_i_start,
			current_i_end,
			ny_,
			g_, tau_, delta_x_const_, delta_y_, min_calc_value_,
			use_delta_x_array_, d_shared_delta_x_arr_, // Pass false and nullptr
			host_grid_ // Pass initial host grid
		));
		current_i_start = current_i_end;
		std::cout << "  Subgrid " << pid << ": i = [" << subgrids_.back()->global_i_start_
			<< ", " << subgrids_.back()->global_i_end_ << ")" << std::endl;
	}

	// --- Set Neighbors ---
	for (int i = 0; i < subgrids_.size(); ++i) {
		SubGridCUDA* neighbor_left = (i > 0) ? subgrids_[i - 1].get() : nullptr;
		SubGridCUDA* neighbor_right = (i < subgrids_.size() - 1) ? subgrids_[i + 1].get() : nullptr;
		subgrids_[i]->SetNeighbors(neighbor_left, neighbor_right);
	}
	std::cout << "Subgrids created and neighbors set." << std::endl;
}

void SingleLineGridSimulationSubgridCUDA::InitializeFromParser(IBaseBathymetryParser& parser, double g, double tau,
	int timesteps, int num_subgrids, int save_interval, double min_calc_value) {
	g_ = g;
	tau_ = tau;
	nx_ = parser.GetCols();
	ny_ = parser.GetRows();
	// Estimate delta_y based on latitude degrees - adjust if needed
	delta_y_ = parser.GetCellSize() * METERS_PER_DEGREE_LAT;
	// delta_y_ = 462.5; // Or use the fixed value from original code

	timesteps_ = timesteps;
	save_interval_ = save_interval;
	min_calc_value_ = min_calc_value;
	num_subgrids_ = std::max(1, num_subgrids);
	use_delta_x_array_ = true; // Delta_x depends on latitude (j)
	delta_x_const_ = 0.0; // Not used


	// Define the delta_x function for host grid initialization
	double yllcorner = parser.GetYllCorner();
	double cellsize = parser.GetCellSize();
	auto delta_x_func = [this, yllcorner, cellsize](int i, int j) -> double {
		double latitude = yllcorner + (ny_ - 1 - j + 0.5) * cellsize;
		return METERS_PER_DEGREE_LAT * cellsize * std::cos(latitude * DEG_TO_RAD);
		};

	// Initialize host grid
	host_grid_.ResizeAndAllocate(nx_, ny_, delta_x_func, delta_y_);
	host_grid_.SetParamsTest6(tau_, parser); // Initialize H, D, eta on host
	if (!host_grid_.IsInitialized()) {
		throw std::runtime_error("Host grid initialization failed for Parser setup!");
	}

	// Clear previous subgrids if any
	subgrids_.clear();
	FreeSharedCudaMemory(); // Free old delta_x array if exists

	// Precompute the shared delta_x array on GPU *before* creating subgrids
	PrecomputeDeltaX_GPU();

	// --- Create Subgrids ---
	std::cout << "Creating " << num_subgrids_ << " subgrids from Parser..." << std::endl;
	int base_subgrid_width = nx_ / num_subgrids_;
	int remainder = nx_ % num_subgrids_;
	int current_i_start = 0;

	for (int pid = 0; pid < num_subgrids_; ++pid) {
		int subgrid_width = base_subgrid_width + (pid < remainder ? 1 : 0);
		int current_i_end = current_i_start + subgrid_width;

		if (subgrid_width <= 0) {
			std::cerr << "Warning: Subgrid " << pid << " has zero width. Skipping." << std::endl;
			continue;
		}

		subgrids_.push_back(std::make_unique<SubGridCUDA>(
			pid,
			current_i_start,
			current_i_end,
			ny_,
			g_, tau_, delta_x_const_, delta_y_, min_calc_value_,
			use_delta_x_array_, d_shared_delta_x_arr_, // Pass true and the device pointer
			host_grid_ // Pass initial host grid
		));
		current_i_start = current_i_end;
		std::cout << "  Subgrid " << pid << ": i = [" << subgrids_.back()->global_i_start_
			<< ", " << subgrids_.back()->global_i_end_ << ")" << std::endl;
	}

	// --- Set Neighbors ---
	for (int i = 0; i < subgrids_.size(); ++i) {
		SubGridCUDA* neighbor_left = (i > 0) ? subgrids_[i - 1].get() : nullptr;
		SubGridCUDA* neighbor_right = (i < subgrids_.size() - 1) ? subgrids_[i + 1].get() : nullptr;
		subgrids_[i]->SetNeighbors(neighbor_left, neighbor_right);
	}
	std::cout << "Subgrids created and neighbors set." << std::endl;
}

void SingleLineGridSimulationSubgridCUDA::Run() {
	if (subgrids_.empty()) {
		throw std::runtime_error("Simulation must be initialized (SetParamsTest1 or InitializeFromParser) before running.");
	}
	if (timesteps_ <= 0) {
		std::cout << "No timesteps to run." << std::endl;
		return;
	}

	std::cout << "Starting Subgridded CUDA Simulation (" << num_subgrids_ << " subgrids)..." << std::endl;

	// Initial save at timestep 0
	// Data is already on host grid from initialization
	SaveDataToFile("eta", 0);
#ifdef SAVE_ETA_MAX_MAP
	// Initialize eta_max on host grid (0 for dry cells, eta for wet cells)
	for (int i = 0; i < nx_; ++i) for (int j = 0; j < ny_; ++j)
	{
		host_grid_.eta_max(i, j) = (host_grid_.D(i, j) >= min_calc_value_)
			? host_grid_.eta(i, j) : 0.0;
	}
#endif

	// Track whether last timestep was saved (to avoid redundant final D2H copy)
	bool last_step_saved = false;

	// --- Main Simulation Loop ---
	// NOTE: No cudaDeviceSynchronize() calls inside the loop except before D2H copy.
	// All GPU operations (kernel launches and D2D copies) use the default stream,
	// which guarantees in-order execution. Explicit synchronization is only needed
	// before accessing data on the host.
	if (out_of_core_mode_) {
		std::cout << "[OOC mode] Out-of-core enabled: per-step H2D + D2H roundtrip for all subgrids." << std::endl;
	}
	for (int t = 0; t < timesteps_; ++t) {

		// === Case 3 (out-of-core) hook ===
		// Refresh device subgrids from host_grid_ at the start of every step.
		// This simulates the cost of true out-of-core where canonical data lives
		// in host RAM and must be uploaded each step. VRAM usage stays the same
		// (hack) but timing reflects PCIe bandwidth limit.
		if (out_of_core_mode_) {
			for (auto& sg : subgrids_) {
				sg->CopyInitialDataFromHost(host_grid_);
			}
		}

		// --- Half Step Phase ---
		// 1. Compute Half Step for all subgrids (async kernel launch)
		for (auto& sg : subgrids_) {
			sg->ComputeHalfStep();
		}

		// 2. Prepare boundary data FOR sending (D2D: no-op, data stays on GPU)
		for (auto& sg : subgrids_) {
			sg->PrepareLeftBoundarySendBuffer(true); // For hat variables
			sg->PrepareRightBoundarySendBuffer(true);
		}

		// 3. Receive boundary data FROM neighbors (D2D copy: GPU -> GPU)
		// cudaMemcpy in default stream implicitly waits for previous operations
		for (auto& sg : subgrids_) {
			sg->ReceiveLeftBoundaryData(true); // For hat variables
			sg->ReceiveRightBoundaryData(true);
		}

		// 4. Apply internal Top/Bottom boundaries for hat variables
		for (auto& sg : subgrids_) {
			sg->ApplyInternalBoundaries(true);
		}

		// --- Full Step Phase ---
		// 5. Compute Full Step for all subgrids (async kernel launch)
		for (auto& sg : subgrids_) {
			sg->ComputeFullStep();
		}

		// 6. Prepare boundary data FOR sending (D2D: no-op, data stays on GPU)
		for (auto& sg : subgrids_) {
			sg->PrepareLeftBoundarySendBuffer(false); // For full step variables
			sg->PrepareRightBoundarySendBuffer(false);
		}

		// 7. Receive boundary data FROM neighbors (D2D copy: GPU -> GPU)
		for (auto& sg : subgrids_) {
			sg->ReceiveLeftBoundaryData(false); // For full step variables
			sg->ReceiveRightBoundaryData(false);
		}

		// 8. Apply internal Top/Bottom boundaries for full step variables
		for (auto& sg : subgrids_) {
			sg->ApplyInternalBoundaries(false);
		}

		// --- Output/Saving ---
		int percentage = static_cast<int>(100.0 * (t + 1) / timesteps_);
		std::cout << "\rSimulation Progress: " << percentage << "% (Timestep " << t + 1 << "/" << timesteps_ << ")";
		std::cout.flush();

		last_step_saved = false;

		// === Case 3 (out-of-core) hook ===
		// After full-step, drain results back to host_grid_ — emulates D2H download
		// of one subgrid at a time. Must happen every step (not on save_interval).
		if (out_of_core_mode_) {
			checkCudaErrors(cudaDeviceSynchronize());
			for (auto& sg : subgrids_) {
				sg->CopyResultToHostGrid(host_grid_, "H");
				sg->CopyResultToHostGrid(host_grid_, "u");
				sg->CopyResultToHostGrid(host_grid_, "v");
				sg->CopyResultToHostGrid(host_grid_, "eta");
			}
			last_step_saved = true;
		}

		// D2H copy only when save condition is met (not unconditionally at every timestep)
		if ((t + 1) % save_interval_ == 0) {
			// Synchronize ONLY before host data access (D2H copy)
			checkCudaErrors(cudaDeviceSynchronize());

			// Copy results from all subgrids back to the main host grid
			for (auto& sg : subgrids_) {
				sg->CopyResultToHostGrid(host_grid_, "H");
				sg->CopyResultToHostGrid(host_grid_, "u");
				sg->CopyResultToHostGrid(host_grid_, "v");
				sg->CopyResultToHostGrid(host_grid_, "eta");
			}

			last_step_saved = true;

			//SaveDataToFile("eta", t + 1);

#ifdef SAVE_ETA_MAX_MAP
			UpdateEtaMaxOnHost(); // Calculate eta_max on host using updated eta
#endif
		}

	} // End of timestep loop

	std::cout << std::endl << "Simulation Finished." << std::endl;

	// Conditional final D2H copy: only if the last timestep was NOT already saved by save_interval.
	// This is the ONLY allowed D2H outside of the save_interval condition.
	// If save_interval divides timesteps evenly, this block will not execute.
	if (!last_step_saved) {
		std::cout << "Collecting final state (last step not covered by save_interval)..." << std::endl;
		// Synchronize before final D2H copy (only sync allowed at end of simulation)
		checkCudaErrors(cudaDeviceSynchronize());
		for (auto& sg : subgrids_) {
			sg->CopyResultToHostGrid(host_grid_, "H");
			sg->CopyResultToHostGrid(host_grid_, "u");
			sg->CopyResultToHostGrid(host_grid_, "v");
			sg->CopyResultToHostGrid(host_grid_, "eta");
		}
#ifdef SAVE_ETA_MAX_MAP
		UpdateEtaMaxOnHost();
#endif
	}

#ifdef SAVE_ETA_MAX_MAP
	SaveDataToFile("eta_max", timesteps_);
#endif

}

void SingleLineGridSimulationSubgridCUDA::UpdateEtaMaxOnHost() {
	if (!host_grid_.IsInitialized()) return;
	std::cout << "\nUpdating eta_max on host..." << std::flush;
	for (int i = 0; i < nx_; ++i) {
		for (int j = 0; j < ny_; ++j) {
			if (host_grid_.D(i, j) < min_calc_value_) continue;
			// Используем eta() напрямую, а НЕ H()-D(), чтобы избежать артефактов
			// на граничных ячейках где H копируется с другой D
			double current_eta = host_grid_.eta(i, j);
			if (current_eta > host_grid_.eta_max(i, j)) {
				host_grid_.eta_max(i, j) = current_eta;
			}
		}
	}
	std::cout << " Done." << std::flush;
}


void SingleLineGridSimulationSubgridCUDA::SaveDataToFile(const std::string& variable_name, int timestep) {
	// Identical logic to the previous SaveDataToFile, using host_grid_
	const std::string folder_name = "Result_SubgridCUDA";
	const std::string filename = folder_name + "/" + variable_name + "_timestep_" + std::to_string(timestep) + ".txt";

	try {
		if (!fs::exists(folder_name)) {
			if (fs::create_directories(folder_name)) {
				std::cout << "\nCreated output directory: " << folder_name << std::endl;
			}
			else {
				std::cerr << "\nFailed to create directory: " << folder_name << std::endl;
			}
		}
	}
	catch (const fs::filesystem_error& e) {
		std::cerr << "\nError creating directory " << folder_name << ": " << e.what() << std::endl;
		return;
	}

	const double* data_ptr = nullptr;
	if (variable_name == "H")
	{
		data_ptr = host_grid_.GetHData();
	}
	else if (variable_name == "u")
	{
		data_ptr = host_grid_.GetUData();
	}
	else if (variable_name == "v")
	{
		data_ptr = host_grid_.GetVData();
	}
	else if (variable_name == "eta")
	{
		data_ptr = host_grid_.GetEtaData();
	}
	else if (variable_name == "eta_max")
	{
		data_ptr = host_grid_.GetEtaMaxData();
	}
	else if (variable_name == "D")
	{
		data_ptr = host_grid_.GetDData();
	}

	if (!data_ptr) {
		std::cerr << "\nError: Cannot save unknown variable '" << variable_name << "'." << std::endl;
		return;
	}

	std::ofstream file(filename);
	if (!file) {
		std::cerr << "\nError: Cannot open file " << filename << " for writing." << std::endl;
		return;
	}

	std::cout << "\nSaving " << variable_name << " to " << filename << "..." << std::flush;

	std::ostringstream output_buffer;
	//output_buffer << std::fixed << std::setprecision(6);

	for (int i = 0; i < nx_; ++i) {
		for (int j = 0; j < ny_; ++j) {
			output_buffer << data_ptr[static_cast<size_t>(i) * ny_ + j] << (j == ny_ - 1 ? "" : " ");
		}
		output_buffer << "\n";
	}

	file << output_buffer.str();
	file.close();
	std::cout << " Done." << std::flush;
}

int SingleLineGridSimulationSubgridCUDA::GetTimesteps() const {
	return timesteps_;
}

const Grid& SingleLineGridSimulationSubgridCUDA::GetHostGrid() const {
	// Final results should be in host_grid_ after Run() completes
	return host_grid_;
}