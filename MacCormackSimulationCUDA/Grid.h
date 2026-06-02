#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "IBaseBathymetryParser.h"

class Grid {
private:
    int nx_;
    int ny_;
    std::unique_ptr<double[]> H_;
    std::unique_ptr<double[]> u_;
    std::unique_ptr<double[]> v_;
    std::unique_ptr<double[]> eta_;
    std::unique_ptr<double[]> D_;
    std::unique_ptr<double[]> eta_max_;

    std::function<double(int, int)> delta_x_func_;
    double delta_y_val_;

    bool is_initialized_ = false;

public:
    // Constructor allowing zero size for delayed initialization
    Grid(int nx = 0, int ny = 0);

    // Constructor with delta functions/values
    Grid(int nx, int ny, std::function<double(int, int)> delta_x, double delta_y);
    
    // Destructor
    ~Grid();
    
    // Move semantics
    Grid(Grid&& other) noexcept;
    Grid& operator=(Grid&& other) noexcept;
    
    // Copy semantics
    Grid(const Grid& other);
    Grid& operator=(const Grid& other);

    // Initialize or resize the grid
    void ResizeAndAllocate(int nx, int ny, std::function<double(int, int)> delta_x, double delta_y);

    // Accessors
    double& H(int i, int j);
    double& u(int i, int j);
    double& v(int i, int j);
    double& eta(int i, int j);
    double& D(int i, int j);
    double& eta_max(int i, int j);

    const double& H(int i, int j) const;
    const double& u(int i, int j) const;
    const double& v(int i, int j) const;
    const double& eta(int i, int j) const;
    const double& D(int i, int j) const;
    const double& eta_max(int i, int j) const;

    int GetNx() const;
    int GetNy() const;
    double GetDeltaY() const;
    double GetDeltaX(int i, int j) const;

    bool IsInitialized() const;

    // Initialization routines
    void SetParamsTest1();
    // S2 (default): center=(139.60, 40.28), al_deg=10.
    // S7 variant uses center=(132.00, 36.00), al_deg=80 (CPU passes -GetAngle() = -(-80) = 80).
    void SetParamsTest6(double tau, IBaseBathymetryParser& parser,
                        double center_x = 139.60, double center_y = 40.28, double al_deg = 10.0);

    // Get raw pointers (useful for CUDA memcpy)
    double* GetHData();
    double* GetUData();
    double* GetVData();
    double* GetEtaData();
    double* GetDData();
    double* GetEtaMaxData();

    const double* GetHData() const;
    const double* GetUData() const;
    const double* GetVData() const;
    const double* GetEtaData() const;
    const double* GetDData() const;
    const double* GetEtaMaxData() const;
};
