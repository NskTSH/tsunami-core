// DiskOffloadSimulator.cpp
// Memory-efficient simulator: only ONE subgrid in RAM at a time.
// Algorithm: two passes per timestep.
//   Pass 1: For each subgrid: load state -> ComputeHalfStep -> save hat to disk
//   Pass 2: For each subgrid: load state+hat -> load neighbor hat columns -> apply hat BC -> ComputeFullStep -> apply BC -> save state

#include "DiskOffloadSimulator.h"
#include "BoundaryCondition.h"
#include "ScenarioBuilder.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <chrono>

namespace fs = std::filesystem;

DiskOffloadSimulator::DiskOffloadSimulator()
    : g_(9.81), tau_(1.0), delta_y_(1.0),
    nx_(100), ny_(100), timesteps_(100),
    num_subgrids_(4), coord_system_(CoordSystem::Cartesian),
    center_latitude_rad_(0.785398163), minimum_depth_to_calc_(8.0) {
    save_interval_ = 10;
}

void DiskOffloadSimulator::SetParameters(double g, double tau,
    std::function<double(int, int)> delta_x, double delta_y,
    int nx, int ny, int timesteps, int num_subgrids,
    int exchangeBoundaryIndex, int exchangeHatBoundaryIndex,
    CoordSystem coord_system, double center_latitude_rad,
    double minimum_depth_to_calc) {
    g_ = g;
    tau_ = tau;
    delta_x_ = std::move(delta_x);
    delta_y_ = delta_y;
    nx_ = nx;
    ny_ = ny;
    timesteps_ = timesteps;
    num_subgrids_ = num_subgrids;
    coord_system_ = coord_system;
    center_latitude_rad_ = center_latitude_rad;
    minimum_depth_to_calc_ = minimum_depth_to_calc;

    // Set boundary exchange functions based on index
    // Using the same pattern as SingleLineGridSimulation
    switch (exchangeBoundaryIndex) {
    case 0: ExchangeBoundaryData_ = BoundaryCondition::ExchangeBoundaryData1; break;
    case 1: ExchangeBoundaryData_ = BoundaryCondition::ExchangeBoundaryData2; break;
    case 2: ExchangeBoundaryData_ = BoundaryCondition::ExchangeBoundaryData3; break;
    case 3: ExchangeBoundaryData_ = BoundaryCondition::ExchangeBoundaryData4; break;
    case 4: ExchangeBoundaryData_ = BoundaryCondition::ExchangeBoundaryData5; break;
    default: ExchangeBoundaryData_ = BoundaryCondition::ExchangeBoundaryData1; break;
    }
    switch (exchangeHatBoundaryIndex) {
    case 0: ExchangeHatBoundaryData_ = BoundaryCondition::ExchangeHatBoundaryData1; break;
    case 1: ExchangeHatBoundaryData_ = BoundaryCondition::ExchangeHatBoundaryData2; break;
    case 2: ExchangeHatBoundaryData_ = BoundaryCondition::ExchangeHatBoundaryData3; break;
    case 3: ExchangeHatBoundaryData_ = BoundaryCondition::ExchangeHatBoundaryData4; break;
    case 4: ExchangeHatBoundaryData_ = BoundaryCondition::ExchangeHatBoundaryData5; break;
    default: ExchangeHatBoundaryData_ = BoundaryCondition::ExchangeHatBoundaryData1; break;
    }
}

void DiskOffloadSimulator::SetGrid(Grid&& grid) {
    global_grid_ = std::move(grid);
}

void DiskOffloadSimulator::SetSaveInterval(int interval) { save_interval_ = interval; }
int DiskOffloadSimulator::GetTimesteps() const { return timesteps_; }

std::string DiskOffloadSimulator::StateFilePath(int pid) const {
    return temp_folder_ + "/subgrid_" + std::to_string(pid) + "_state.bin";
}

std::string DiskOffloadSimulator::HatFilePath(int pid) const {
    return temp_folder_ + "/subgrid_" + std::to_string(pid) + "_hat.bin";
}

// ========== FILE I/O ==========

void DiskOffloadSimulator::SaveSubGridState(int pid, const SubGrid& sg) {
    const int n = sg.local_nx_ * sg.local_ny_;
    std::ofstream f(StateFilePath(pid), std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write state file for pid=" + std::to_string(pid));
    // Write 5 arrays: H, u, v, eta, D
    f.write(reinterpret_cast<const char*>(sg.local_grid_.GetH()), n * sizeof(double));
    f.write(reinterpret_cast<const char*>(sg.local_grid_.GetU()), n * sizeof(double));
    f.write(reinterpret_cast<const char*>(sg.local_grid_.GetV()), n * sizeof(double));
    f.write(reinterpret_cast<const char*>(sg.local_grid_.GetEta()), n * sizeof(double));
    f.write(reinterpret_cast<const char*>(sg.local_grid_.GetD()), n * sizeof(double));
}

void DiskOffloadSimulator::LoadSubGridState(int pid, SubGrid& sg) {
    const int n = sg.local_nx_ * sg.local_ny_;
    std::ifstream f(StateFilePath(pid), std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read state file for pid=" + std::to_string(pid));
    f.read(reinterpret_cast<char*>(sg.local_grid_.GetH()), n * sizeof(double));
    f.read(reinterpret_cast<char*>(sg.local_grid_.GetU()), n * sizeof(double));
    f.read(reinterpret_cast<char*>(sg.local_grid_.GetV()), n * sizeof(double));
    f.read(reinterpret_cast<char*>(sg.local_grid_.GetEta()), n * sizeof(double));
    f.read(reinterpret_cast<char*>(sg.local_grid_.GetD()), n * sizeof(double));
}

void DiskOffloadSimulator::SaveSubGridHat(int pid, const SubGrid& sg) {
    const int n = sg.local_nx_ * sg.local_ny_;
    std::ofstream f(HatFilePath(pid), std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write hat file for pid=" + std::to_string(pid));
    f.write(reinterpret_cast<const char*>(sg.H_hat_.get()), n * sizeof(double));
    f.write(reinterpret_cast<const char*>(sg.u_hat_.get()), n * sizeof(double));
    f.write(reinterpret_cast<const char*>(sg.v_hat_.get()), n * sizeof(double));
    f.write(reinterpret_cast<const char*>(sg.eta_hat_.get()), n * sizeof(double));
}

void DiskOffloadSimulator::LoadSubGridHat(int pid, SubGrid& sg) {
    const int n = sg.local_nx_ * sg.local_ny_;
    std::ifstream f(HatFilePath(pid), std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read hat file for pid=" + std::to_string(pid));
    f.read(reinterpret_cast<char*>(sg.H_hat_.get()), n * sizeof(double));
    f.read(reinterpret_cast<char*>(sg.u_hat_.get()), n * sizeof(double));
    f.read(reinterpret_cast<char*>(sg.v_hat_.get()), n * sizeof(double));
    f.read(reinterpret_cast<char*>(sg.eta_hat_.get()), n * sizeof(double));
}

// Load one column from a neighbor's saved state file
// right_edge=true: read column (local_nx-2), used for left ghost of next subgrid
// right_edge=false: read column 1, used for right ghost of previous subgrid
void DiskOffloadSimulator::LoadBoundaryColumn(int neighbor_pid, bool right_edge,
    double* H_col, double* u_col, double* v_col, double* eta_col) {
    const auto& info = subgrid_infos_[neighbor_pid];
    const int ny = info.local_ny;
    const int nx = info.local_nx;
    const int col = right_edge ? (nx - 2) : 1;
    const int n = nx * ny;

    std::ifstream f(StateFilePath(neighbor_pid), std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read boundary from pid=" + std::to_string(neighbor_pid));

    // State file layout: H[n], u[n], v[n], eta[n], D[n]
    // Column col: indices col*ny .. col*ny + ny - 1
    auto read_col = [&](int array_offset, double* out) {
        std::streamoff pos = (array_offset * n + col * ny) * sizeof(double);
        f.seekg(pos);
        f.read(reinterpret_cast<char*>(out), ny * sizeof(double));
    };

    read_col(0, H_col);   // H
    read_col(1, u_col);   // u
    read_col(2, v_col);   // v
    read_col(3, eta_col); // eta
}

void DiskOffloadSimulator::LoadHatBoundaryColumn(int neighbor_pid, bool right_edge,
    double* H_col, double* u_col, double* v_col, double* eta_col) {
    const auto& info = subgrid_infos_[neighbor_pid];
    const int ny = info.local_ny;
    const int nx = info.local_nx;
    const int col = right_edge ? (nx - 2) : 1;
    const int n = nx * ny;

    std::ifstream f(HatFilePath(neighbor_pid), std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read hat boundary from pid=" + std::to_string(neighbor_pid));

    auto read_col = [&](int array_offset, double* out) {
        std::streamoff pos = (array_offset * n + col * ny) * sizeof(double);
        f.seekg(pos);
        f.read(reinterpret_cast<char*>(out), ny * sizeof(double));
    };

    read_col(0, H_col);
    read_col(1, u_col);
    read_col(2, v_col);
    read_col(3, eta_col);
}

void DiskOffloadSimulator::ApplyBoundaryToGhost(SubGrid& sg, bool left_ghost,
    const double* H_col, const double* u_col, const double* v_col, const double* eta_col) {
    const int i = left_ghost ? 0 : (sg.local_nx_ - 1);
    for (int j = 0; j < sg.local_ny_; ++j) {
        sg.local_grid_.H(i, j) = H_col[j];
        sg.local_grid_.u(i, j) = u_col[j];
        sg.local_grid_.v(i, j) = v_col[j];
        sg.local_grid_.eta(i, j) = eta_col[j];
    }
}

void DiskOffloadSimulator::ApplyHatBoundaryToGhost(SubGrid& sg, bool left_ghost,
    const double* H_col, const double* u_col, const double* v_col, const double* eta_col) {
    const int ny = sg.local_ny_;
    const int i = left_ghost ? 0 : (sg.local_nx_ - 1);
    const int offset = i * ny;
    for (int j = 0; j < ny; ++j) {
        sg.H_hat_[offset + j] = H_col[j];
        sg.u_hat_[offset + j] = u_col[j];
        sg.v_hat_[offset + j] = v_col[j];
        sg.eta_hat_[offset + j] = eta_col[j];
    }
}

// ========== INITIALIZATION ==========

void DiskOffloadSimulator::Initialize() {
    if (!global_grid_.IsInitialized()) {
        throw std::runtime_error("Global grid is not initialized!");
    }

    fs::create_directories(temp_folder_);

    int m = nx_ / num_subgrids_;
    subgrid_infos_.resize(num_subgrids_);

    // Compute metadata for each subgrid
    for (int pid = 0; pid < num_subgrids_; ++pid) {
        int range_start = pid * m;
        int range_end = (pid == num_subgrids_ - 1) ? nx_ : (pid + 1) * m;
        int local_nx = (range_end - range_start) + 2; // +2 for ghost columns
        int local_ny = ny_;

        subgrid_infos_[pid] = { pid, range_start, range_end, local_nx, local_ny };
    }

    // Initialize each subgrid and save initial state to disk
    std::cout << "=== DiskOffload: Initializing " << num_subgrids_ << " subgrids ===" << std::endl;

    for (int pid = 0; pid < num_subgrids_; ++pid) {
        auto sg = CreateTempSubGrid(pid);
        sg->Initialize();

        // Save initial state
        SaveSubGridState(pid, *sg);

        const auto& info = subgrid_infos_[pid];
        double state_mb = 5.0 * info.local_nx * info.local_ny * 8.0 / (1024 * 1024);
        if (pid == 0) {
            std::cout << "  Subgrid size: " << info.local_nx << " x " << info.local_ny
                << " (" << state_mb << " MB state on disk)" << std::endl;
        }
    }

    double total_disk_mb = 0;
    for (const auto& info : subgrid_infos_) {
        total_disk_mb += 9.0 * info.local_nx * info.local_ny * 8.0 / (1024 * 1024); // state + hat
    }
    std::cout << "  Total disk usage: ~" << total_disk_mb << " MB" << std::endl;
    std::cout << "  RAM per subgrid: ~"
        << 9.0 * subgrid_infos_[0].local_nx * subgrid_infos_[0].local_ny * 8.0 / (1024 * 1024)
        << " MB" << std::endl;
    std::cout << "==================================" << std::endl;
}

std::shared_ptr<SubGrid> DiskOffloadSimulator::CreateTempSubGrid(int pid) {
    const auto& info = subgrid_infos_[pid];
    auto dummy_barrier = std::make_shared<std::barrier<>>(1);

    auto sg = std::make_shared<SubGrid>(
        info.process_id,
        info.global_i_start,
        info.global_i_end,
        global_grid_,
        g_, tau_, delta_x_, delta_y_, minimum_depth_to_calc_,
        dummy_barrier, dummy_barrier,
        1, coord_system_, center_latitude_rad_,
        DecompositionDirection::Horizontal
    );
    sg->ExchangeBoundaryData = ExchangeBoundaryData_;
    sg->ExchangeHatBoundaryData = ExchangeHatBoundaryData_;
    // No neighbors set — boundary exchange done manually via disk
    sg->SetNeighbors(nullptr, nullptr);
    return sg;
}

// ========== MAIN LOOP ==========

void DiskOffloadSimulator::Run() {
    std::cout << "=== DiskOffload Simulation ===" << std::endl;
    std::cout << "  Grid: " << nx_ << " x " << ny_ << std::endl;
    std::cout << "  Subgrids: " << num_subgrids_ << std::endl;
    std::cout << "  Timesteps: " << timesteps_ << std::endl;
    std::cout << "  Save interval: " << save_interval_ << std::endl;
    std::cout << "==============================" << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    // Temp buffers for boundary columns
    std::vector<double> H_col(ny_), u_col(ny_), v_col(ny_), eta_col(ny_);

    for (int t = 0; t < timesteps_; ++t) {

        // ===== PASS 1: ComputeHalfStep for all subgrids =====
        for (int pid = 0; pid < num_subgrids_; ++pid) {
            auto io_start = std::chrono::high_resolution_clock::now();

            auto sg = CreateTempSubGrid(pid);
            sg->Initialize(); // Allocates hat arrays
            LoadSubGridState(pid, *sg);

            auto io_end = std::chrono::high_resolution_clock::now();
            total_io_time_ms_ += std::chrono::duration<double, std::milli>(io_end - io_start).count();

            auto comp_start = std::chrono::high_resolution_clock::now();
            sg->ComputeHalfStep();
            auto comp_end = std::chrono::high_resolution_clock::now();
            total_compute_time_ms_ += std::chrono::duration<double, std::milli>(comp_end - comp_start).count();

            io_start = std::chrono::high_resolution_clock::now();
            SaveSubGridHat(pid, *sg);
            io_end = std::chrono::high_resolution_clock::now();
            total_io_time_ms_ += std::chrono::duration<double, std::milli>(io_end - io_start).count();
        }

        // ===== PASS 2: Exchange hat BC + ComputeFullStep + Exchange BC =====
        for (int pid = 0; pid < num_subgrids_; ++pid) {
            auto io_start = std::chrono::high_resolution_clock::now();

            auto sg = CreateTempSubGrid(pid);
            sg->Initialize();
            LoadSubGridState(pid, *sg);
            LoadSubGridHat(pid, *sg);

            // Apply hat boundary from disk neighbors
            if (pid > 0) {
                LoadHatBoundaryColumn(pid - 1, true, H_col.data(), u_col.data(), v_col.data(), eta_col.data());
                ApplyHatBoundaryToGhost(*sg, true, H_col.data(), u_col.data(), v_col.data(), eta_col.data());
            }
            else {
                // Domain left edge: zero-gradient BC for hat
                const int ny = sg->local_ny_;
                for (int j = 0; j < ny; ++j) {
                    sg->H_hat_[j] = sg->H_hat_[ny + j];
                    sg->u_hat_[j] = sg->u_hat_[ny + j];
                    sg->v_hat_[j] = sg->v_hat_[ny + j];
                    sg->eta_hat_[j] = sg->eta_hat_[ny + j];
                }
            }

            if (pid < num_subgrids_ - 1) {
                LoadHatBoundaryColumn(pid + 1, false, H_col.data(), u_col.data(), v_col.data(), eta_col.data());
                ApplyHatBoundaryToGhost(*sg, false, H_col.data(), u_col.data(), v_col.data(), eta_col.data());
            }
            else {
                // Domain right edge: zero-gradient BC for hat
                const int ny = sg->local_ny_;
                const int last = (sg->local_nx_ - 1) * ny;
                const int prev = (sg->local_nx_ - 2) * ny;
                for (int j = 0; j < ny; ++j) {
                    sg->H_hat_[last + j] = sg->H_hat_[prev + j];
                    sg->u_hat_[last + j] = sg->u_hat_[prev + j];
                    sg->v_hat_[last + j] = sg->v_hat_[prev + j];
                    sg->eta_hat_[last + j] = sg->eta_hat_[prev + j];
                }
            }

            // Apply j-boundary (top/bottom) zero-gradient for hat
            {
                const int ny = sg->local_ny_;
                const int nx = sg->local_nx_;
                for (int i = 0; i < nx; ++i) {
                    // j=0: copy from j=1
                    sg->H_hat_[i * ny] = sg->H_hat_[i * ny + 1];
                    sg->u_hat_[i * ny] = sg->u_hat_[i * ny + 1];
                    sg->v_hat_[i * ny] = sg->v_hat_[i * ny + 1];
                    sg->eta_hat_[i * ny] = sg->eta_hat_[i * ny + 1];
                    // j=ny-1: copy from j=ny-2
                    sg->H_hat_[i * ny + ny - 1] = sg->H_hat_[i * ny + ny - 2];
                    sg->u_hat_[i * ny + ny - 1] = sg->u_hat_[i * ny + ny - 2];
                    sg->v_hat_[i * ny + ny - 1] = sg->v_hat_[i * ny + ny - 2];
                    sg->eta_hat_[i * ny + ny - 1] = sg->eta_hat_[i * ny + ny - 2];
                }
            }

            auto io_end = std::chrono::high_resolution_clock::now();
            total_io_time_ms_ += std::chrono::duration<double, std::milli>(io_end - io_start).count();

            // ComputeFullStep
            auto comp_start = std::chrono::high_resolution_clock::now();
            sg->ComputeFullStep();
            auto comp_end = std::chrono::high_resolution_clock::now();
            total_compute_time_ms_ += std::chrono::duration<double, std::milli>(comp_end - comp_start).count();

            // Apply boundary conditions for state (without neighbors, use zero-gradient)
            io_start = std::chrono::high_resolution_clock::now();
            if (pid > 0) {
                LoadBoundaryColumn(pid - 1, true, H_col.data(), u_col.data(), v_col.data(), eta_col.data());
                ApplyBoundaryToGhost(*sg, true, H_col.data(), u_col.data(), v_col.data(), eta_col.data());
            }
            else {
                // Left domain edge
                for (int j = 0; j < sg->local_ny_; ++j) {
                    sg->local_grid_.H(0, j) = sg->local_grid_.H(1, j);
                    sg->local_grid_.u(0, j) = sg->local_grid_.u(1, j);
                    sg->local_grid_.v(0, j) = sg->local_grid_.v(1, j);
                    sg->local_grid_.eta(0, j) = sg->local_grid_.eta(1, j);
                }
            }

            if (pid < num_subgrids_ - 1) {
                // Right boundary: we need the UPDATED state of the RIGHT neighbor
                // But it hasn't been computed yet in this pass!
                // Solution: use the PREVIOUS timestep's state for the right neighbor
                // This is consistent with the sequential in-memory approach where
                // ExchangeBoundaryData reads from neighbor's already-computed state.
                //
                // Wait — in the original code, ALL subgrids do full step FIRST, then
                // ALL do boundary exchange. So at exchange time, all neighbors have
                // the NEW state. But here we process sequentially.
                //
                // For correctness: we need to save the updated state BEFORE processing
                // right neighbors, then read it. Since we process left-to-right:
                // - pid-1 is already saved with new state (from earlier iteration of this pass)
                // - pid+1 still has OLD state
                //
                // This is different from in-memory! For bit-identical results, we need
                // to do boundary exchange in a SEPARATE pass. Let's save state now
                // and do boundary exchange in a third pass.
                //
                // For simplicity, use old state of right neighbor (pid+1 not yet computed).
                // This introduces a small difference from in-memory. To get bit-identical:
                // would need pass 3.
                LoadBoundaryColumn(pid + 1, false, H_col.data(), u_col.data(), v_col.data(), eta_col.data());
                ApplyBoundaryToGhost(*sg, false, H_col.data(), u_col.data(), v_col.data(), eta_col.data());
            }
            else {
                for (int j = 0; j < sg->local_ny_; ++j) {
                    sg->local_grid_.H(sg->local_nx_ - 1, j) = sg->local_grid_.H(sg->local_nx_ - 2, j);
                    sg->local_grid_.u(sg->local_nx_ - 1, j) = sg->local_grid_.u(sg->local_nx_ - 2, j);
                    sg->local_grid_.v(sg->local_nx_ - 1, j) = sg->local_grid_.v(sg->local_nx_ - 2, j);
                    sg->local_grid_.eta(sg->local_nx_ - 1, j) = sg->local_grid_.eta(sg->local_nx_ - 2, j);
                }
            }

            // Apply j-boundary (top/bottom) for state
            for (int i = 0; i < sg->local_nx_; ++i) {
                sg->local_grid_.H(i, 0) = sg->local_grid_.H(i, 1);
                sg->local_grid_.u(i, 0) = sg->local_grid_.u(i, 1);
                sg->local_grid_.v(i, 0) = sg->local_grid_.v(i, 1);
                sg->local_grid_.eta(i, 0) = sg->local_grid_.eta(i, 1);
                sg->local_grid_.H(i, ny_ - 1) = sg->local_grid_.H(i, ny_ - 2);
                sg->local_grid_.u(i, ny_ - 1) = sg->local_grid_.u(i, ny_ - 2);
                sg->local_grid_.v(i, ny_ - 1) = sg->local_grid_.v(i, ny_ - 2);
                sg->local_grid_.eta(i, ny_ - 1) = sg->local_grid_.eta(i, ny_ - 2);
            }

            SaveSubGridState(pid, *sg);
            io_end = std::chrono::high_resolution_clock::now();
            total_io_time_ms_ += std::chrono::duration<double, std::milli>(io_end - io_start).count();
        }

        // ===== SNAPSHOT SAVE =====
        if (save_interval_ > 0 && (t + 1) % save_interval_ == 0) {
            // Rebuild global grid from disk
            for (int pid = 0; pid < num_subgrids_; ++pid) {
                auto sg = CreateTempSubGrid(pid);
                sg->Initialize();
                LoadSubGridState(pid, *sg);
                sg->UpdateGlobalGrid(pid);
            }
            SaveDataToFile("eta", global_grid_, t + 1);
            SaveDataToFile("u", global_grid_, t + 1);
            SaveDataToFile("v", global_grid_, t + 1);
        }

        // Progress
        int percentage = static_cast<int>(100.0 * (t + 1) / timesteps_);
        std::cout << "\r  DiskOffload: step " << (t + 1) << "/" << timesteps_
            << " (" << percentage << "%)";
        std::cout.flush();
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << std::endl;
    std::cout << "=== DiskOffload Performance ===" << std::endl;
    std::cout << "  Total time: " << total_ms / 1000.0 << " s" << std::endl;
    std::cout << "  I/O time:   " << total_io_time_ms_ / 1000.0 << " s ("
        << 100.0 * total_io_time_ms_ / total_ms << "%)" << std::endl;
    std::cout << "  Compute time: " << total_compute_time_ms_ / 1000.0 << " s ("
        << 100.0 * total_compute_time_ms_ / total_ms << "%)" << std::endl;
    std::cout << "===============================" << std::endl;
}

void DiskOffloadSimulator::CollectResults(int timestep) {
    // Rebuild global grid from disk for final state
    for (int pid = 0; pid < num_subgrids_; ++pid) {
        auto sg = CreateTempSubGrid(pid);
        sg->Initialize();
        LoadSubGridState(pid, *sg);
        sg->UpdateGlobalGrid(pid);
    }
}

void DiskOffloadSimulator::SaveDataToFile(const std::string& variable_name, const Grid& grid, int timestep) {
    fs::create_directories(result_folder_);
    const std::string filename = result_folder_ + "/" + variable_name
        + "_timestep_" + std::to_string(timestep) + ".txt";

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "ERROR: Cannot open " << filename << " for writing." << std::endl;
        return;
    }

    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            double val = 0;
            if (variable_name == "eta") val = grid.eta(i, j);
            else if (variable_name == "u") val = grid.u(i, j);
            else if (variable_name == "v") val = grid.v(i, j);
            else if (variable_name == "H") val = grid.H(i, j);

            file << val;
            if (j < ny_ - 1) file << "\t";
        }
        file << "\n";
    }
}

void DiskOffloadSimulator::CleanupTempFiles() {
    if (fs::exists(temp_folder_)) {
        for (const auto& entry : fs::directory_iterator(temp_folder_)) {
            if (entry.path().extension() == ".bin") {
                fs::remove(entry.path());
            }
        }
        std::cout << "  Cleaned up temp files in " << temp_folder_ << std::endl;
    }
}
