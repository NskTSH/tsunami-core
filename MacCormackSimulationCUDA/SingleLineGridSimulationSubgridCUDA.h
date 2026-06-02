#pragma once
#include <corecrt_math_defines.h>

#include "Grid.h"
#include "SubGridCUDA.h" // Include the CUDA-enabled SubGrid
#include "IBaseBathymetryParser.h"
#include <string>
#include <vector>
#include <memory> // For unique_ptr
#include <functional>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip> // For setprecision

class SingleLineGridSimulationSubgridCUDA {
private:
    double g_;
    double tau_;
    double delta_y_;
    double delta_x_const_; // Used if delta_x is constant
    int nx_; // Total grid size x
    int ny_; // Total grid size y
    int timesteps_;
    int save_interval_;
    double min_calc_value_;
    int num_subgrids_; // Number of subgrids to divide into

    Grid host_grid_; // Holds the full grid data on the host

    std::vector<std::unique_ptr<SubGridCUDA>> subgrids_;

    // Shared GPU resource for delta_x if it varies
    bool use_delta_x_array_ = false;
    double* d_shared_delta_x_arr_ = nullptr;

    // Out-of-core mode (Case 3 benchmark): per-step H2D + D2H roundtrip.
    bool out_of_core_mode_ = false;

    // Private Helper Methods
    void PrecomputeDeltaX_GPU(); // Allocate and compute shared delta_x array
    void FreeSharedCudaMemory(); // Free the shared delta_x array
    void SaveDataToFile(const std::string& variable_name, int timestep);
    void UpdateEtaMaxOnHost(); // Helper to calculate eta_max after copying H back

    // Constants
    static constexpr double EARTH_RADIUS = 6371000.0;
    static constexpr double DEG_TO_RAD = M_PI / 180.0;
    static constexpr double METERS_PER_DEGREE_LAT = 111300.0;


public:
    SingleLineGridSimulationSubgridCUDA();
    ~SingleLineGridSimulationSubgridCUDA(); // Destructor

    // Disable copy/move
    SingleLineGridSimulationSubgridCUDA(const SingleLineGridSimulationSubgridCUDA&) = delete;
    SingleLineGridSimulationSubgridCUDA& operator=(const SingleLineGridSimulationSubgridCUDA&) = delete;
    SingleLineGridSimulationSubgridCUDA(SingleLineGridSimulationSubgridCUDA&&) = delete;
    SingleLineGridSimulationSubgridCUDA& operator=(SingleLineGridSimulationSubgridCUDA&&) = delete;

    // Initialization presets
    void SetParamsTest1(int num_subgrids);
    void InitializeFromParser(IBaseBathymetryParser& parser, double g, double tau,
        int timesteps, int num_subgrids, int save_interval = 10, double min_calc_value = 8.0);

    // Main simulation execution
    void Run();

    // Out-of-core (Case 3, Work_Plan §1 4-cases benchmark): emulates the case
    // where subgrids canonical home is host RAM and they must be transferred
    // to GPU for each timestep (insufficient VRAM scenario). When enabled, Run()
    // performs H2D from host_grid_ → device subgrids before each step and
    // D2H device → host_grid_ after each step. VRAM is still allocated for all
    // N subgrids (hack), but transfer overhead per step matches true OOC.
    void SetOutOfCoreMode(bool v) { out_of_core_mode_ = v; }

    // Getters
    int GetTimesteps() const;
    const Grid& GetHostGrid() const; // Access final host grid
};