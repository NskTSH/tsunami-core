// Grid.cpp - Оптимизированная версия
#define _USE_MATH_DEFINES
#include "Grid.h"

#include <iomanip>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

// ========== КОНСТРУКТОР ==========
Grid::Grid(int nx, int ny, std::function<double(int, int)> delta_x, double delta_y)
    : nx_(nx), ny_(ny), delta_x_(delta_x), delta_y_(delta_y),
      delta_x_cached_(0.0), inv_dx_(0.0), inv_dy_(0.0), is_initialized_(false)
{
    AllocateArrays();
    PrecomputeConstants();
}

void Grid::AllocateArrays() {
    if (nx_ > 0 && ny_ > 0) {
        const size_t size = static_cast<size_t>(nx_) * ny_;
        H_ = std::make_unique<Real[]>(size);
        u_ = std::make_unique<Real[]>(size);
        v_ = std::make_unique<Real[]>(size);
        eta_ = std::make_unique<Real[]>(size);
        D_ = std::make_unique<Real[]>(size);
        eta_max_ = std::make_unique<Real[]>(size);
        u_max_ = std::make_unique<Real[]>(size);
        v_max_ = std::make_unique<Real[]>(size);

        std::fill(H_.get(), H_.get() + size, 0.0);
        std::fill(u_.get(), u_.get() + size, 0.0);
        std::fill(v_.get(), v_.get() + size, 0.0);
        std::fill(eta_.get(), eta_.get() + size, 0.0);
        std::fill(D_.get(), D_.get() + size, 0.0);
        std::fill(eta_max_.get(), eta_max_.get() + size, 0.0);
        std::fill(u_max_.get(), u_max_.get() + size, 0.0);
        std::fill(v_max_.get(), v_max_.get() + size, 0.0);
    }
}

void Grid::PrecomputeConstants() {
    if (delta_x_) {
        delta_x_cached_ = delta_x_(0, 0);
        inv_dx_ = (delta_x_cached_ > 0.0) ? 1.0 / delta_x_cached_ : 0.0;
    }
    inv_dy_ = (delta_y_ > 0.0) ? 1.0 / delta_y_ : 0.0;
}

// ========== MOVE КОНСТРУКТОР ==========
Grid::Grid(Grid&& other) noexcept
    : subgrid_count(other.subgrid_count),
      nx_(other.nx_), ny_(other.ny_),
      delta_x_(std::move(other.delta_x_)), delta_y_(other.delta_y_),
      delta_x_cached_(other.delta_x_cached_),
      inv_dx_(other.inv_dx_), inv_dy_(other.inv_dy_),
      H_(std::move(other.H_)), u_(std::move(other.u_)),
      v_(std::move(other.v_)), eta_(std::move(other.eta_)),
      D_(std::move(other.D_)), eta_max_(std::move(other.eta_max_)),
      u_max_(std::move(other.u_max_)), v_max_(std::move(other.v_max_)),
      is_initialized_(other.is_initialized_)
{
    other.nx_ = 0;
    other.ny_ = 0;
    other.is_initialized_ = false;
}

// ========== MOVE ПРИСВАИВАНИЕ ==========
Grid& Grid::operator=(Grid&& other) noexcept {
    if (this != &other) {
        subgrid_count = other.subgrid_count;
        nx_ = other.nx_;
        ny_ = other.ny_;
        delta_x_ = std::move(other.delta_x_);
        delta_y_ = other.delta_y_;
        delta_x_cached_ = other.delta_x_cached_;
        inv_dx_ = other.inv_dx_;
        inv_dy_ = other.inv_dy_;
        H_ = std::move(other.H_);
        u_ = std::move(other.u_);
        v_ = std::move(other.v_);
        eta_ = std::move(other.eta_);
        D_ = std::move(other.D_);
        eta_max_ = std::move(other.eta_max_);
        u_max_ = std::move(other.u_max_);
        v_max_ = std::move(other.v_max_);
        is_initialized_ = other.is_initialized_;

        other.nx_ = 0;
        other.ny_ = 0;
        other.is_initialized_ = false;
    }
    return *this;
}

// ========== КОПИРУЮЩИЙ КОНСТРУКТОР ==========
Grid::Grid(const Grid& other)
    : subgrid_count(other.subgrid_count),
      nx_(other.nx_), ny_(other.ny_),
      delta_x_(other.delta_x_), delta_y_(other.delta_y_),
      delta_x_cached_(other.delta_x_cached_),
      inv_dx_(other.inv_dx_), inv_dy_(other.inv_dy_),
      is_initialized_(other.is_initialized_)
{
    if (nx_ > 0 && ny_ > 0) {
        const size_t size = static_cast<size_t>(nx_) * ny_;
        H_ = std::make_unique<Real[]>(size);
        u_ = std::make_unique<Real[]>(size);
        v_ = std::make_unique<Real[]>(size);
        eta_ = std::make_unique<Real[]>(size);
        D_ = std::make_unique<Real[]>(size);
        eta_max_ = std::make_unique<Real[]>(size);
        u_max_ = std::make_unique<Real[]>(size);
        v_max_ = std::make_unique<Real[]>(size);

        std::memcpy(H_.get(), other.H_.get(), size * sizeof(Real));
        std::memcpy(u_.get(), other.u_.get(), size * sizeof(Real));
        std::memcpy(v_.get(), other.v_.get(), size * sizeof(Real));
        std::memcpy(eta_.get(), other.eta_.get(), size * sizeof(Real));
        std::memcpy(D_.get(), other.D_.get(), size * sizeof(Real));
        std::memcpy(eta_max_.get(), other.eta_max_.get(), size * sizeof(Real));
        std::memcpy(u_max_.get(), other.u_max_.get(), size * sizeof(Real));
        std::memcpy(v_max_.get(), other.v_max_.get(), size * sizeof(Real));
    }
}

// ========== КОПИРУЮЩЕЕ ПРИСВАИВАНИЕ ==========
Grid& Grid::operator=(const Grid& other) {
    if (this != &other) {
        subgrid_count = other.subgrid_count;
        nx_ = other.nx_;
        ny_ = other.ny_;
        delta_x_ = other.delta_x_;
        delta_y_ = other.delta_y_;
        delta_x_cached_ = other.delta_x_cached_;
        inv_dx_ = other.inv_dx_;
        inv_dy_ = other.inv_dy_;
        is_initialized_ = other.is_initialized_;

        if (nx_ > 0 && ny_ > 0) {
            const size_t size = static_cast<size_t>(nx_) * ny_;
            H_ = std::make_unique<Real[]>(size);
            u_ = std::make_unique<Real[]>(size);
            v_ = std::make_unique<Real[]>(size);
            eta_ = std::make_unique<Real[]>(size);
            D_ = std::make_unique<Real[]>(size);
            eta_max_ = std::make_unique<Real[]>(size);
            u_max_ = std::make_unique<Real[]>(size);
            v_max_ = std::make_unique<Real[]>(size);

            std::memcpy(H_.get(), other.H_.get(), size * sizeof(Real));
            std::memcpy(u_.get(), other.u_.get(), size * sizeof(Real));
            std::memcpy(v_.get(), other.v_.get(), size * sizeof(Real));
            std::memcpy(eta_.get(), other.eta_.get(), size * sizeof(Real));
            std::memcpy(D_.get(), other.D_.get(), size * sizeof(Real));
            std::memcpy(eta_max_.get(), other.eta_max_.get(), size * sizeof(Real));
            std::memcpy(u_max_.get(), other.u_max_.get(), size * sizeof(Real));
            std::memcpy(v_max_.get(), other.v_max_.get(), size * sizeof(Real));
        } else {
            H_.reset();
            u_.reset();
            v_.reset();
            eta_.reset();
            D_.reset();
            eta_max_.reset();
            u_max_.reset();
            v_max_.reset();
        }
    }
    return *this;
}

// ========== INITIALIZE ==========
void Grid::Initialize() {
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            D_[i * ny_ + j] = 1000.0;
            H_[i * ny_ + j] = D_[i * ny_ + j];
        }
    }
    is_initialized_ = true;
}

// ========== ДОСТУП К ЭЛЕМЕНТАМ (оригинальная индексация) ==========
Real& Grid::H(int i, int j) { return H_[i * ny_ + j]; }
Real& Grid::eta_max(int i, int j) { return eta_max_[i * ny_ + j]; }
Real& Grid::u_max(int i, int j) { return u_max_[i * ny_ + j]; }
Real& Grid::v_max(int i, int j) { return v_max_[i * ny_ + j]; }
Real& Grid::u(int i, int j) { return u_[i * ny_ + j]; }
Real& Grid::v(int i, int j) { return v_[i * ny_ + j]; }
Real& Grid::eta(int i, int j) { return eta_[i * ny_ + j]; }
Real& Grid::D(int i, int j) { return D_[i * ny_ + j]; }

const Real& Grid::H(int i, int j) const { return H_[i * ny_ + j]; }
const Real& Grid::eta_max(int i, int j) const { return eta_max_[i * ny_ + j]; }
const Real& Grid::u_max(int i, int j) const { return u_max_[i * ny_ + j]; }
const Real& Grid::v_max(int i, int j) const { return v_max_[i * ny_ + j]; }
const Real& Grid::u(int i, int j) const { return u_[i * ny_ + j]; }
const Real& Grid::v(int i, int j) const { return v_[i * ny_ + j]; }
const Real& Grid::eta(int i, int j) const { return eta_[i * ny_ + j]; }
const Real& Grid::D(int i, int j) const { return D_[i * ny_ + j]; }

int Grid::GetNx() const { return nx_; }
int Grid::GetNy() const { return ny_; }

// ========== ТЕСТЫ ==========
void Grid::SetParamsTest1() {
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            D_[i * ny_ + j] = 1000.0;
            H_[i * ny_ + j] = D_[i * ny_ + j];
        }
    }

    for (int i = 45; i < 55; ++i) {
        if (i >= 0 && i < nx_) {
            for (int j = 45; j < 55; ++j) {
                if (j >= 0 && j < ny_) {
                    eta_[i * ny_ + j] = 1.0;
                    H_[i * ny_ + j] = D_[i * ny_ + j] + eta_[i * ny_ + j];
                }
            }
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsTest2() {
    double x0 = 500000;
    double y0 = 500000;
    double node_radius = 50;
    double radius = delta_x_(0, 0) * node_radius;
    
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            D_[i * ny_ + j] = 1000.0;
            H_[i * ny_ + j] = D_[i * ny_ + j];
            double x = i * delta_x_(0, 0);
            double y = j * delta_y_;
            double dist_to_node = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));
            if (dist_to_node > radius) {
                eta_[i * ny_ + j] = 0.0;
            } else {
                eta_[i * ny_ + j] = 0.5 * (1 + std::cos(M_PI * dist_to_node / radius));
            }
            H_[i * ny_ + j] = D_[i * ny_ + j] + eta_[i * ny_ + j];
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsTest3(double tau) {
    double n = 0.0;
    double T = 300.0;
    double waveHeight = 0.5 * (1.0 - std::cos(n * tau * 2 * M_PI / T));
    double k = 1e-8;
    
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            D_[i * ny_ + j] = k * std::pow(delta_y_ * (ny_ - j), 2) + 10;
            H_[i * ny_ + j] = D_[i * ny_ + j];
            eta_[i * ny_ + j] = waveHeight;
            H_[i * ny_ + j] = D_[i * ny_ + j] + eta_[i * ny_ + j];
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsTest4(double tau) {
    double n = 0.0;
    double T = 300.0;
    double waveHeight = 0.5 * (1.0 - std::cos(n * tau * 2 * M_PI / T));

    const int ax = nx_ / 2, ay = ny_ - 1;
    const int bx = nx_ - 1, by = ny_ / 2;
    const int cx = nx_ - 1, cy = ny_ - 1;

    auto sign = [](double x1, double y1, double x2, double y2, double x3, double y3) {
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
    };

    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            const double d1 = sign(i, j, ax, ay, bx, by);
            const double d2 = sign(i, j, bx, by, cx, cy);
            const double d3 = sign(i, j, cx, cy, ax, ay);

            const bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
            const bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
            const bool inside = !(has_neg && has_pos);

            if (inside) {
                D_[i * ny_ + j] = 0.0;
                H_[i * ny_ + j] = 0.0;
                u_[i * ny_ + j] = 0.0;
                v_[i * ny_ + j] = 0.0;
                eta_[i * ny_ + j] = 0.0;
            } else {
                D_[i * ny_ + j] = 1000.0;
                eta_[i * ny_ + j] = waveHeight;
                H_[i * ny_ + j] = D_[i * ny_ + j] + eta_[i * ny_ + j];
            }
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsTest5(double tau) {
    double n = 0.0;
    double T = 300.0;
    double waveHeight = 0.5 * (1.0 - std::cos(n * tau * 2 * M_PI / T));
    int centerX = nx_ / 2;
    int centerY = ny_ / 2;
    int radius = 100;
    
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            int dx = i - centerX;
            int dy = j - centerY;
            if (dx * dx + dy * dy <= radius * radius) {
                D_[i * ny_ + j] = 0;
            } else {
                D_[i * ny_ + j] = 1000;
            }
            H_[i * ny_ + j] = D_[i * ny_ + j];
            eta_[i * ny_ + j] = waveHeight;
            H_[i * ny_ + j] = D_[i * ny_ + j] + eta_[i * ny_ + j];
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsTest6(double tau, ASCParser& parser, DefaultParametersParser& simulator_params) {
    double r1 = 0.148;
    double r2 = 0.449;

    const auto& sources = simulator_params.GetSources();
    for (size_t s = 0; s < sources.size(); ++s) {
		ISource* src = sources[s].get();
		if (!src || src->GetSourceType() != SourceType::ELLIPSOIDAL) {
			continue;
		}
		auto* ellipsoidSource = static_cast<EllipsoidalSource*>(src);
        
        double center_x = ellipsoidSource->GetCenterX();
        double center_y = ellipsoidSource->GetCenterY();
        double al_deg = -ellipsoidSource->GetAngle();
        double al = al_deg * M_PI / 180.0;
        double h0 = 2.0;
        
        for (int i = 0; i < nx_; ++i) {
            for (int j = 0; j < ny_; ++j) {
                D_[i * ny_ + j] = parser.GetValue(j, i);
                H_[i * ny_ + j] = D_[i * ny_ + j];

                double x = parser.GetXllCorner() + i * parser.GetCellSize();
                double y = parser.GetYllCorner() + (ny_ - 1 - j) * parser.GetCellSize();
                
                double ax = ((x - center_x) * std::cos(al) - (y - center_y) * std::sin(al)) / r1;
                double ay = ((x - center_x) * std::sin(al) + (y - center_y) * std::cos(al)) / r2;
                double arg = std::sqrt(ax * ax + ay * ay);

                if (arg <= 1) {
                    eta_[i * ny_ + j] = (1 + std::cos(M_PI * arg)) * h0 / 2.0;
                }
                H_[i * ny_ + j] = D_[i * ny_ + j] + eta_[i * ny_ + j];
            }
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsCoriolis(double depth, int wave_width, double wave_amplitude) {
    // Flat bottom
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            D_[i * ny_ + j] = depth;
            H_[i * ny_ + j] = depth;
            u_[i * ny_ + j] = 0.0;
            v_[i * ny_ + j] = 0.0;
            eta_[i * ny_ + j] = 0.0;
        }
    }
    // Plane wave front along bottom edge: j ∈ [0, wave_width)
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < std::min(wave_width, ny_); ++j) {
            eta_[i * ny_ + j] = wave_amplitude;
            H_[i * ny_ + j] = depth + wave_amplitude;
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsCoriolisEllipsoidal(double depth, double r1_cells, double r2_cells,
    double angle_deg, double amplitude) {
    SetParamsCoriolisEllipsoidal(depth, r1_cells, r2_cells, angle_deg, amplitude,
        nx_ / 2.0, ny_ / 2.0);
}

void Grid::SetParamsCoriolisEllipsoidal(double depth, double r1_cells, double r2_cells,
    double angle_deg, double amplitude, double cx_cells, double cy_cells) {
    // Flat bottom
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            D_[i * ny_ + j] = depth;
            H_[i * ny_ + j] = depth;
            u_[i * ny_ + j] = 0.0;
            v_[i * ny_ + j] = 0.0;
            eta_[i * ny_ + j] = 0.0;
        }
    }
    double al = angle_deg * M_PI / 180.0;
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            double dx = i - cx_cells;
            double dy = j - cy_cells;
            double ax = (dx * std::cos(al) - dy * std::sin(al)) / r1_cells;
            double ay = (dx * std::sin(al) + dy * std::cos(al)) / r2_cells;
            double arg = std::sqrt(ax * ax + ay * ay);
            if (arg <= 1.0) {
                double eta_val = (1.0 + std::cos(M_PI * arg)) * amplitude / 2.0;
                eta_[i * ny_ + j] = eta_val;
                H_[i * ny_ + j] = depth + eta_val;
            }
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsCoriolisFinitePlane(double depth,
    int wave_j_min, int wave_j_max,
    int wave_i_min, int wave_i_max,
    double wave_amplitude,
    double u0,
    double v0) {
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            D_[i * ny_ + j] = depth;
            H_[i * ny_ + j] = depth;
            u_[i * ny_ + j] = 0.0;
            v_[i * ny_ + j] = 0.0;
            eta_[i * ny_ + j] = 0.0;
        }
    }
    int j0 = std::max(0, wave_j_min);
    int j1 = std::min(ny_, wave_j_max);
    int i0 = std::max(0, wave_i_min);
    int i1 = std::min(nx_, wave_i_max);
    for (int i = i0; i < i1; ++i) {
        for (int j = j0; j < j1; ++j) {
            eta_[i * ny_ + j] = wave_amplitude;
            H_[i * ny_ + j] = depth + wave_amplitude;
            u_[i * ny_ + j] = u0;
            v_[i * ny_ + j] = v0;
        }
    }
    is_initialized_ = true;
}

void Grid::SetParamsCoriolisHorizontal(double depth, int wave_width, double wave_amplitude) {
    // Flat bottom
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            D_[i * ny_ + j] = depth;
            H_[i * ny_ + j] = depth;
            u_[i * ny_ + j] = 0.0;
            v_[i * ny_ + j] = 0.0;
            eta_[i * ny_ + j] = 0.0;
        }
    }
    // Plane wave front along left edge: i ∈ [0, wave_width)
    for (int i = 0; i < std::min(wave_width, nx_); ++i) {
        for (int j = 0; j < ny_; ++j) {
            eta_[i * ny_ + j] = wave_amplitude;
            H_[i * ny_ + j] = depth + wave_amplitude;
        }
    }
    is_initialized_ = true;
}

void Grid::SetParameters(double tau, IBaseBathymetryParser& parser) {
    const int I_OFF = 300;
    const int J_OFF = 1200;
    const int ny_orig = ny_ - 2 * J_OFF;
    const int nx_orig = nx_ - 2 * I_OFF;

    double x0 = 400 * 1000;
    double y0 = 300 * 1000;
    double node_radius = 50;
    double radius = delta_x_(0, 0) * node_radius;

    for (int i = 0; i < nx_; ++i) {
        int ii = i - I_OFF;
        int ii_clamped = (ii < 0 ? 0 : (ii >= nx_orig ? nx_orig - 1 : ii));
        double x = ii * delta_x_(0, 0);

        for (int j = 0; j < ny_; ++j) {
            int jj = j - J_OFF;
            int jj_clamped = (jj < 0 ? 0 : (jj >= ny_orig ? ny_orig - 1 : jj));

            double Dval = parser.GetValue(jj_clamped, ii_clamped);
            Dval = 10000.0 * Dval * Dval / 9.81;
            D_[i * ny_ + j] = Dval;
            H_[i * ny_ + j] = Dval;

            double y = jj * delta_y_;
            double eta_val = 0.0;
            
            if (ii >= 0 && ii < nx_orig && jj >= 0 && jj < ny_orig) {
                double dx = x - x0;
                double dy = y - y0;
                double dist = std::sqrt(dx * dx + dy * dy);
                if (dist <= radius) {
                    eta_val = 0.5 * (1.0 + std::cos(M_PI * dist / radius)) * 5.0;
                }
            }
            eta_[i * ny_ + j] = eta_val;
            H_[i * ny_ + j] = Dval + eta_val;
        }
    }
    is_initialized_ = true;
}
