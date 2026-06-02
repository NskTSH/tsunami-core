#define _USE_MATH_DEFINES
#include "Grid.h"
#include <math.h>
#include <stdexcept>
#include <iostream> // For cerr

Grid::Grid(int nx, int ny) : nx_(0), ny_(0), delta_y_val_(0.0) {
    if (nx > 0 && ny > 0) {
        ResizeAndAllocate(nx, ny, [](int, int) { return 0.0; }, 0.0); // Dummy delta_x
    }
}


Grid::Grid(int nx, int ny, std::function<double(int, int)> delta_x, double delta_y)
    : nx_(0), ny_(0) // Initialize to 0 first
{
    ResizeAndAllocate(nx, ny, delta_x, delta_y);
}

Grid::~Grid() = default;

// Move constructor
Grid::Grid(Grid&& other) noexcept
    : nx_(other.nx_), ny_(other.ny_),
      H_(std::move(other.H_)), u_(std::move(other.u_)),
      v_(std::move(other.v_)), eta_(std::move(other.eta_)),
      D_(std::move(other.D_)), eta_max_(std::move(other.eta_max_)),
      delta_x_func_(std::move(other.delta_x_func_)),
      delta_y_val_(other.delta_y_val_),
      is_initialized_(other.is_initialized_)
{
    other.nx_ = 0;
    other.ny_ = 0;
    other.delta_y_val_ = 0.0;
    other.is_initialized_ = false;
}

// Move assignment
Grid& Grid::operator=(Grid&& other) noexcept {
    if (this != &other) {
        nx_ = other.nx_;
        ny_ = other.ny_;
        H_ = std::move(other.H_);
        u_ = std::move(other.u_);
        v_ = std::move(other.v_);
        eta_ = std::move(other.eta_);
        D_ = std::move(other.D_);
        eta_max_ = std::move(other.eta_max_);
        delta_x_func_ = std::move(other.delta_x_func_);
        delta_y_val_ = other.delta_y_val_;
        is_initialized_ = other.is_initialized_;
        
        other.nx_ = 0;
        other.ny_ = 0;
        other.delta_y_val_ = 0.0;
        other.is_initialized_ = false;
    }
    return *this;
}

// Copy constructor
Grid::Grid(const Grid& other)
    : nx_(other.nx_), ny_(other.ny_),
      delta_x_func_(other.delta_x_func_),
      delta_y_val_(other.delta_y_val_),
      is_initialized_(other.is_initialized_)
{
    if (nx_ > 0 && ny_ > 0) {
        size_t size = static_cast<size_t>(nx_) * ny_;
        H_ = std::make_unique<double[]>(size);
        u_ = std::make_unique<double[]>(size);
        v_ = std::make_unique<double[]>(size);
        eta_ = std::make_unique<double[]>(size);
        D_ = std::make_unique<double[]>(size);
        eta_max_ = std::make_unique<double[]>(size);
        
        std::copy(other.H_.get(), other.H_.get() + size, H_.get());
        std::copy(other.u_.get(), other.u_.get() + size, u_.get());
        std::copy(other.v_.get(), other.v_.get() + size, v_.get());
        std::copy(other.eta_.get(), other.eta_.get() + size, eta_.get());
        std::copy(other.D_.get(), other.D_.get() + size, D_.get());
        std::copy(other.eta_max_.get(), other.eta_max_.get() + size, eta_max_.get());
    }
}

// Copy assignment
Grid& Grid::operator=(const Grid& other) {
    if (this != &other) {
        nx_ = other.nx_;
        ny_ = other.ny_;
        delta_x_func_ = other.delta_x_func_;
        delta_y_val_ = other.delta_y_val_;
        is_initialized_ = other.is_initialized_;
        
        if (nx_ > 0 && ny_ > 0) {
            size_t size = static_cast<size_t>(nx_) * ny_;
            H_ = std::make_unique<double[]>(size);
            u_ = std::make_unique<double[]>(size);
            v_ = std::make_unique<double[]>(size);
            eta_ = std::make_unique<double[]>(size);
            D_ = std::make_unique<double[]>(size);
            eta_max_ = std::make_unique<double[]>(size);
            
            std::copy(other.H_.get(), other.H_.get() + size, H_.get());
            std::copy(other.u_.get(), other.u_.get() + size, u_.get());
            std::copy(other.v_.get(), other.v_.get() + size, v_.get());
            std::copy(other.eta_.get(), other.eta_.get() + size, eta_.get());
            std::copy(other.D_.get(), other.D_.get() + size, D_.get());
            std::copy(other.eta_max_.get(), other.eta_max_.get() + size, eta_max_.get());
        } else {
            H_.reset();
            u_.reset();
            v_.reset();
            eta_.reset();
            D_.reset();
            eta_max_.reset();
        }
    }
    return *this;
}


void Grid::ResizeAndAllocate(int nx, int ny, std::function<double(int, int)> delta_x, double delta_y) {
    if (nx <= 0 || ny <= 0) {
        // Allow resetting to zero size
        if (nx == 0 && ny == 0) {
            nx_ = 0;
            ny_ = 0;
            H_.reset();
            u_.reset();
            v_.reset();
            eta_.reset();
            D_.reset();
            eta_max_.reset();
            delta_x_func_ = nullptr;
            delta_y_val_ = 0.0;
            is_initialized_ = false;
            return;
        }
        else {
            throw std::invalid_argument("Grid dimensions must be positive or both zero.");
        }
    }

    nx_ = nx;
    ny_ = ny;
    delta_x_func_ = delta_x;
    delta_y_val_ = delta_y;

    try {
        size_t total_size = static_cast<size_t>(nx_) * ny_;
        H_ = std::make_unique<double[]>(total_size);
        u_ = std::make_unique<double[]>(total_size);
        v_ = std::make_unique<double[]>(total_size);
        eta_ = std::make_unique<double[]>(total_size);
        D_ = std::make_unique<double[]>(total_size);
        eta_max_ = std::make_unique<double[]>(total_size);

        std::fill(H_.get(), H_.get() + total_size, 0.0);
        std::fill(u_.get(), u_.get() + total_size, 0.0);
        std::fill(v_.get(), v_.get() + total_size, 0.0);
        std::fill(eta_.get(), eta_.get() + total_size, 0.0);
        std::fill(D_.get(), D_.get() + total_size, 0.0);
        std::fill(eta_max_.get(), eta_max_.get() + total_size, 0.0);
        // Note: is_initialized_ should be set by specific setup routines like SetParamsTest1
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate memory for Grid: " << e.what() << std::endl;
        nx_ = 0; // Reset dimensions on failure
        ny_ = 0;
        throw; // Re-throw the exception
    }
}

// --- Accessors (Implement bounds checking if desired) ---
double& Grid::H(int i, int j) { return H_[static_cast<size_t>(i) * ny_ + j]; }
double& Grid::u(int i, int j) { return u_[static_cast<size_t>(i) * ny_ + j]; }
double& Grid::v(int i, int j) { return v_[static_cast<size_t>(i) * ny_ + j]; }
double& Grid::eta(int i, int j) { return eta_[static_cast<size_t>(i) * ny_ + j]; }
double& Grid::D(int i, int j) { return D_[static_cast<size_t>(i) * ny_ + j]; }
double& Grid::eta_max(int i, int j) { return eta_max_[static_cast<size_t>(i) * ny_ + j]; }

const double& Grid::H(int i, int j) const { return H_[static_cast<size_t>(i) * ny_ + j]; }
const double& Grid::u(int i, int j) const { return u_[static_cast<size_t>(i) * ny_ + j]; }
const double& Grid::v(int i, int j) const { return v_[static_cast<size_t>(i) * ny_ + j]; }
const double& Grid::eta(int i, int j) const { return eta_[static_cast<size_t>(i) * ny_ + j]; }
const double& Grid::D(int i, int j) const { return D_[static_cast<size_t>(i) * ny_ + j]; }
const double& Grid::eta_max(int i, int j) const { return eta_max_[static_cast<size_t>(i) * ny_ + j]; }

int Grid::GetNx() const { return nx_; }
int Grid::GetNy() const { return ny_; }
double Grid::GetDeltaY() const { return delta_y_val_; }
double Grid::GetDeltaX(int i, int j) const {
    if (delta_x_func_) {
        // Basic bounds check for safety, although kernel index calculation should be correct
        i = std::max(0, std::min(i, nx_ - 1));
        j = std::max(0, std::min(j, ny_ - 1));
        return delta_x_func_(i, j);
    }
    return 0.0; // Or throw an error if function is expected
}


bool Grid::IsInitialized() const { return is_initialized_; }

// --- Raw Data Pointers ---
double* Grid::GetHData() { return H_.get(); }
double* Grid::GetUData() { return u_.get(); }
double* Grid::GetVData() { return v_.get(); }
double* Grid::GetEtaData() { return eta_.get(); }
double* Grid::GetDData() { return D_.get(); }
double* Grid::GetEtaMaxData() { return eta_max_.get(); }

const double* Grid::GetHData() const { return H_.get(); }
const double* Grid::GetUData() const { return u_.get(); }
const double* Grid::GetVData() const { return v_.get(); }
const double* Grid::GetEtaData() const { return eta_.get(); }
const double* Grid::GetDData() const { return D_.get(); }
const double* Grid::GetEtaMaxData() const { return eta_max_.get(); }


// --- Initialization Routines (Keep as they are) ---
void Grid::SetParamsTest1() {
    if (nx_ == 0 || ny_ == 0) throw std::runtime_error("Grid not allocated before SetParamsTest1");

    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            size_t idx = static_cast<size_t>(i) * ny_ + j;
            D_[idx] = 1000.0;
            H_[idx] = D_[idx]; // Initialize H = D
            eta_[idx] = 0.0;   // Initialize eta = 0
            u_[idx] = 0.0;
            v_[idx] = 0.0;
            eta_max_[idx] = 0.0; // Initialize eta_max
        }
    }

    // Initial disturbance
    int i_start = nx_ / 2 - 5; // Centered example
    int i_end = nx_ / 2 + 5;
    int j_start = ny_ / 2 - 5;
    int j_end = ny_ / 2 + 5;

    for (int i = i_start; i < i_end; ++i) {
        if (i >= 0 && i < nx_) {
            for (int j = j_start; j < j_end; ++j) {
                if (j >= 0 && j < ny_) {
                    size_t idx = static_cast<size_t>(i) * ny_ + j;
                    eta_[idx] = 1.0;
                    H_[idx] = D_[idx] + eta_[idx]; // H = D + eta
                }
            }
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsTest6(double tau, IBaseBathymetryParser& parser,
                          double center_x, double center_y, double al_deg) {
    if (nx_ == 0 || ny_ == 0) throw std::runtime_error("Grid not allocated before SetParamsTest6");
    if (nx_ != parser.GetCols() || ny_ != parser.GetRows()) {
        throw std::runtime_error("Grid dimensions do not match parser dimensions.");
    }

    // Mirrors CPU Grid::SetParamsTest6 exactly. r1, r2 hardcoded in degrees;
    // cell coords have no +0.5 offset; bathymetry not clamped (negative D = land,
    // handled by solver via min_depth). Caller supplies source center/angle from .par.
    const double r1 = 0.148;
    const double r2 = 0.449;
    const double al = al_deg * M_PI / 180.0;
    const double h0 = 2.0;

    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            size_t idx = static_cast<size_t>(i) * ny_ + j;
            D_[idx] = parser.GetValue(j, i);  // (row, col) → (j, i)

            double x = parser.GetXllCorner() + i * parser.GetCellSize();
            double y = parser.GetYllCorner() + (ny_ - 1 - j) * parser.GetCellSize();

            double ax = ((x - center_x) * std::cos(al) - (y - center_y) * std::sin(al)) / r1;
            double ay = ((x - center_x) * std::sin(al) + (y - center_y) * std::cos(al)) / r2;
            double arg = std::sqrt(ax * ax + ay * ay);

            if (arg <= 1.0) {
                eta_[idx] = (1.0 + std::cos(M_PI * arg)) * h0 / 2.0;
            } else {
                eta_[idx] = 0.0;
            }
            H_[idx] = D_[idx] + eta_[idx];
            u_[idx] = 0.0;
            v_[idx] = 0.0;
            eta_max_[idx] = 0.0;
        }
    }
    is_initialized_ = true;
}