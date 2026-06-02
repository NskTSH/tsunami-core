#pragma once
// DiskOffloadSimulator.h
// Memory-efficient simulator: only ONE subgrid in RAM at a time.
// Other subgrids are stored on disk as binary files.

#ifndef DISK_OFFLOAD_SIMULATOR_H
#define DISK_OFFLOAD_SIMULATOR_H

#include "Grid.h"
#include "SubGrid.h"
#include "CoordSystem.h"
#include "BaseSimulator.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <chrono>

class DiskOffloadSimulator : public IBaseSimulator
{
public:
    DiskOffloadSimulator();

    void SetParameters(double g, double tau, std::function<double(int, int)> delta_x, double delta_y,
        int nx, int ny, int timesteps, int num_subgrids,
        int exchangeBoundaryIndex, int exchangeHatBoundaryIndex,
        CoordSystem coord_system, double center_latitude_rad = 0.785398163,
        double minimum_depth_to_calc = 8.0);

    void SetGrid(Grid&& grid);

    void Initialize() override;
    void Run() override;
    void CollectResults(int timestep) override;
    void SetSaveInterval(int interval) override;
    int GetTimesteps() const override;

    void SetResultFolder(const std::string& folder) { result_folder_ = folder; }
    void SetTempFolder(const std::string& folder) { temp_folder_ = folder; }

    void SaveDataToFile(const std::string& variable_name, const Grid& grid, int timestep) override;

    Grid global_grid_;

    // Timing statistics
    double total_io_time_ms_ = 0;
    double total_compute_time_ms_ = 0;

private:
    double g_;
    double tau_;
    std::function<double(int, int)> delta_x_;
    double delta_y_;
    int nx_;
    int ny_;
    int timesteps_;
    int num_subgrids_;
    int save_interval_ = 10;
    CoordSystem coord_system_;
    double center_latitude_rad_;
    double minimum_depth_to_calc_;
    std::string result_folder_ = "Result";
    std::string temp_folder_ = "Result/disk_offload_temp";

    std::function<void(std::shared_ptr<SubGrid>, double)> ExchangeBoundaryData_;
    std::function<void(std::shared_ptr<SubGrid>, double)> ExchangeHatBoundaryData_;

    // Subgrid metadata (constant through simulation)
    struct SubGridInfo {
        int process_id;
        int global_i_start;
        int global_i_end;
        int local_nx;
        int local_ny;
    };
    std::vector<SubGridInfo> subgrid_infos_;

    // File I/O for subgrid state
    // Binary format: raw double arrays, no header (metadata in SubGridInfo)
    void SaveSubGridState(int pid, const SubGrid& sg);
    void LoadSubGridState(int pid, SubGrid& sg);

    // Save/load just hat variables
    void SaveSubGridHat(int pid, const SubGrid& sg);
    void LoadSubGridHat(int pid, SubGrid& sg);

    // Load boundary column from neighbor's saved state
    // Loads the rightmost interior column (for left ghost) or leftmost interior column (for right ghost)
    void LoadBoundaryColumn(int neighbor_pid, bool right_edge,
        double* H_col, double* u_col, double* v_col, double* eta_col);

    // Load boundary column from neighbor's saved hat arrays
    void LoadHatBoundaryColumn(int neighbor_pid, bool right_edge,
        double* H_col, double* u_col, double* v_col, double* eta_col);

    // Apply boundary column to subgrid ghost cells
    void ApplyBoundaryToGhost(SubGrid& sg, bool left_ghost,
        const double* H_col, const double* u_col, const double* v_col, const double* eta_col);
    void ApplyHatBoundaryToGhost(SubGrid& sg, bool left_ghost,
        const double* H_col, const double* u_col, const double* v_col, const double* eta_col);

    // Create a temporary SubGrid for computation (allocates local memory)
    std::shared_ptr<SubGrid> CreateTempSubGrid(int pid);

    // Cleanup temp files
    void CleanupTempFiles();

    std::string StateFilePath(int pid) const;
    std::string HatFilePath(int pid) const;
};

#endif // DISK_OFFLOAD_SIMULATOR_H
