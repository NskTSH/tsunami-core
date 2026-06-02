// PartionFileGridSimulator.cpp

#include "PartionFileGridSimulator.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

PartionFileGridSimulator::PartionFileGridSimulator(bool use_file_storage)
    : g_(9.81), tau_(1.0), delta_x_(1.0), delta_y_(1.0),
    nx_(100), ny_(100), timesteps_(100),
    global_grid_(1, 1), save_interval_(10),
    use_file_storage_(use_file_storage) {
}

void PartionFileGridSimulator::SetParameters(double g, double tau, double delta_x, double delta_y,
    int nx, int ny, int timesteps) {
    g_ = g;
    tau_ = tau;
    delta_x_ = delta_x;
    delta_y_ = delta_y;
    nx_ = nx;
    ny_ = ny;
    timesteps_ = timesteps;

    if (!use_file_storage_) {
        global_grid_ = Grid(nx_, ny_);
    }
}

void PartionFileGridSimulator::Initialize() {
    if (!use_file_storage_) {
        if (!global_grid_.IsInitialized()) {
            throw std::runtime_error("Global grid is not initialized!");
        }
    }
    else {
        GenerateAndSaveGrid();
    }
}

void PartionFileGridSimulator::Run() {
    for (int t = 0; t < timesteps_; ++t) {
        std::cout << "\rSimulation step: " << (t + 1) << "/" << timesteps_;
        std::cout.flush();

        if (use_file_storage_) {
            LoadSubGridAndCompute();
        }
        else {
            ComputeFullStep(global_grid_);
        }

        if ((t + 1) % save_interval_ == 0) {
            SaveDataToFile("eta", global_grid_, t + 1);
        }
    }
    std::cout << std::endl;
}

void PartionFileGridSimulator::GenerateAndSaveGrid() {
    const std::string filename = "Global_grid";
	GenerateAndSaveArray(filename + "_H.bin", global_grid_.GetH());
	GenerateAndSaveArray(filename + "_u.bin", global_grid_.GetU());
	GenerateAndSaveArray(filename + "_v.bin", global_grid_.GetV());
	GenerateAndSaveArray(filename + "_eta.bin", global_grid_.GetEta());
	GenerateAndSaveArray(filename + "_D.bin", global_grid_.GetD());
}

void PartionFileGridSimulator::GenerateAndSaveArray(const std::string& filename, 
    double* arr)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr
    	<< "Error: Cannot open file "
    	<< filename
    	<< " for writing."
    	<< std::endl;
        return;
    }

    int nx = global_grid_.GetNx();
    int ny = global_grid_.GetNy();
    file.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
    file.write(reinterpret_cast<const char*>(&ny), sizeof(ny));

    file.write(reinterpret_cast<const char*>(arr),
        nx * ny * sizeof(double));

    file.close();
}

void PartionFileGridSimulator::LoadSubGridFromFile(int start_x, int end_x, Grid& subgrid) {
    const std::string filename = "Global_grid";
    LoadSubGridArray(filename + "_H.bin", subgrid.GetH(), start_x, end_x);
    LoadSubGridArray(filename + "_u.bin", subgrid.GetU(), start_x, end_x);
    LoadSubGridArray(filename + "_v.bin", subgrid.GetV(), start_x, end_x);
    LoadSubGridArray(filename + "_eta.bin", subgrid.GetEta(), start_x, end_x);
    LoadSubGridArray(filename + "_D.bin", subgrid.GetD(), start_x, end_x);
}

void PartionFileGridSimulator::LoadSubGridArray(const std::string& filename, double* arr, int start_x, int end_x) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for reading." << std::endl;
        return;
    }

    int nx, ny;
    file.read(reinterpret_cast<char*>(&nx), sizeof(nx));
    file.read(reinterpret_cast<char*>(&ny), sizeof(ny));

    int width = end_x - start_x;
    std::unique_ptr<double[]> buffer = std::make_unique<double[]>(width * ny);

    file.seekg((start_x * ny) * sizeof(double), std::ios::cur);
    file.read(reinterpret_cast<char*>(buffer.get()), width * ny * sizeof(double));

    std::memcpy(arr, buffer.get(), width * ny * sizeof(double));

    file.close();
}

void PartionFileGridSimulator::SaveSubGridToFile(int start_x, int end_x, const Grid& subgrid) {
    const std::string filename = "Global_grid";
    SaveSubGridArray(filename + "_H.bin", subgrid.GetH(), start_x, end_x);
    SaveSubGridArray(filename + "_u.bin", subgrid.GetU(), start_x, end_x);
    SaveSubGridArray(filename + "_v.bin", subgrid.GetV(), start_x, end_x);
    SaveSubGridArray(filename + "_eta.bin", subgrid.GetEta(), start_x, end_x);
    SaveSubGridArray(filename + "_D.bin", subgrid.GetD(), start_x, end_x);
}

void PartionFileGridSimulator::SaveSubGridArray(const std::string& filename, const double* arr, int start_x, int end_x) {
    std::fstream file(filename, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }

    int nx, ny;
    file.read(reinterpret_cast<char*>(&nx), sizeof(nx));
    file.read(reinterpret_cast<char*>(&ny), sizeof(ny));

    int width = end_x - start_x;
    file.seekp((start_x * ny) * sizeof(double), std::ios::cur);
    file.write(reinterpret_cast<const char*>(arr), width * ny * sizeof(double));

    file.close();
}


void PartionFileGridSimulator::SaveDataToFile(const std::string& variable_name, const Grid& grid, int timestep) {
    const std::string folder_name = "Result";
    const std::string filename = folder_name + "/" + variable_name + "_timestep_" + std::to_string(timestep) + ".bin";

    std::filesystem::create_directories(folder_name);

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error: Cannot open file for writing.");
    }
    file.write(reinterpret_cast<const char*>(grid.Data()), sizeof(double) * nx_ * ny_);
}