#include "BoundaryCondition.h"
#include <algorithm>
#include <cmath>
#include <corecrt_math_defines.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "SubGrid.h"

// ===== Phase 8 NTHMP Monai: wavemaker time-series store =====================
namespace WavemakerData {
namespace {
    static std::vector<std::pair<double, double>> g_samples;  // (time, eta)
    static double g_tau = 0.0;
    static bool   g_loaded = false;
}

void Load(const std::string& path, double tau) {
    g_samples.clear();
    g_tau = tau;
    g_loaded = false;
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Wavemaker: cannot open " + path);
    }
    std::string line;
    // Skip header line "Time(s)  Water surface(m)" — non-numeric.
    if (!std::getline(in, line)) {
        throw std::runtime_error("Wavemaker: empty file " + path);
    }
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        double t, eta;
        if (!(iss >> t >> eta)) continue;  // skip malformed lines silently
        g_samples.emplace_back(t, eta);
    }
    if (g_samples.empty()) {
        throw std::runtime_error("Wavemaker: no samples parsed from " + path);
    }
    g_loaded = true;
    std::cerr << "[Wavemaker] loaded " << g_samples.size() << " samples from "
              << path << " (range t=[" << g_samples.front().first << ", "
              << g_samples.back().first << "] s)" << std::endl;
}

double GetEtaAt(double t_phys) {
    if (!g_loaded || g_samples.empty()) return 0.0;
    if (t_phys <= g_samples.front().first) return g_samples.front().second;
    if (t_phys >= g_samples.back().first)  return g_samples.back().second;
    // Linear interp: binary search.
    auto it = std::lower_bound(g_samples.begin(), g_samples.end(), t_phys,
        [](const std::pair<double,double>& a, double v) { return a.first < v; });
    if (it == g_samples.begin()) return it->second;
    const auto& hi = *it;
    const auto& lo = *(it - 1);
    const double w = (t_phys - lo.first) / (hi.first - lo.first);
    return lo.second + w * (hi.second - lo.second);
}

bool IsLoaded() { return g_loaded; }
}  // namespace WavemakerData

// ===== DEFAULT-УСЛОВИЯ ДЛЯ БАЗОВЫХ ПЕРЕМЕННЫХ =====

void BoundaryCondition::DefaultLeftConditions(std::shared_ptr<SubGrid> s_grid)
{
    // Верхний граничный слой
    if (s_grid->neighbor_left_) {
        // Отправляем свои данные верхней сетке и получаем его данные
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            // Получаем данные от сетки
            s_grid->local_grid_.H(0, j) =
                s_grid->neighbor_left_->local_grid_.H(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.u(0, j) =
                s_grid->neighbor_left_->local_grid_.u(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.v(0, j) =
                s_grid->neighbor_left_->local_grid_.v(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.eta(0, j) =
                s_grid->neighbor_left_->local_grid_.eta(s_grid->neighbor_left_->local_nx_ - 2, j);
        }
    }
    else {
        //RadiationLeftConditions(s_grid);
        //return;
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
            s_grid->local_grid_.u(0, j) = s_grid->local_grid_.u(1, j);
            s_grid->local_grid_.v(0, j) = s_grid->local_grid_.v(1, j);
            s_grid->local_grid_.eta(0, j) = s_grid->local_grid_.eta(1, j);
        }
    }
}

void BoundaryCondition::DefaultRightConditions(std::shared_ptr<SubGrid> s_grid)
{
    // Правый граничный слой
    if (s_grid->neighbor_right_) {
        // Отправляем свои данные правой сетке и получаем его данные
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) =
                s_grid->neighbor_right_->local_grid_.H(1, j);
            s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) =
                s_grid->neighbor_right_->local_grid_.u(1, j);
            s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) =
                s_grid->neighbor_right_->local_grid_.v(1, j);
            s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) =
                s_grid->neighbor_right_->local_grid_.eta(1, j);
        }
    }
    else {
        //RadiationRightConditions(s_grid);
        //return;
        // Граничные условия на правой границе
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) =
                s_grid->local_grid_.H(s_grid->local_nx_ - 2, j);
            s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) =
                s_grid->local_grid_.u(s_grid->local_nx_ - 2, j);
            s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) =
                s_grid->local_grid_.v(s_grid->local_nx_ - 2, j);
            s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) =
                s_grid->local_grid_.eta(s_grid->local_nx_ - 2, j);
        }
    }
}

// ===== DEFAULT-УСЛОВИЯ ДЛЯ HAT-ПЕРЕМЕННЫХ =====

void BoundaryCondition::DefaultLeftHatConditions(std::shared_ptr<SubGrid> s_grid)
{
    // Верхний граничный слой
    if (s_grid->neighbor_left_) {
        // Отправляем свои данные верхней сетке и получаем его данные
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            // Получаем данные от соседа
            s_grid->H_hat_[j] =
                s_grid->neighbor_left_->H_hat_[(s_grid->neighbor_left_->local_nx_ - 2)
                * s_grid->local_ny_ + j];
            s_grid->u_hat_[j] =
                s_grid->neighbor_left_->u_hat_[(s_grid->neighbor_left_->local_nx_ - 2)
                * s_grid->local_ny_ + j];
            s_grid->v_hat_[j] =
                s_grid->neighbor_left_->v_hat_[(s_grid->neighbor_left_->local_nx_ - 2)
                * s_grid->local_ny_ + j];
            s_grid->eta_hat_[j] =
                s_grid->neighbor_left_->eta_hat_[(s_grid->neighbor_left_->local_nx_ - 2)
                * s_grid->local_ny_ + j];
        }
    }
    else {
        //RadiationLeftHatConditions(s_grid);
        //return;
        // Граничные условия на верхней границе
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->H_hat_[j] = s_grid->H_hat_[s_grid->local_ny_ + j];
            s_grid->u_hat_[j] = s_grid->u_hat_[s_grid->local_ny_ + j];
            s_grid->v_hat_[j] = s_grid->v_hat_[s_grid->local_ny_ + j];
            s_grid->eta_hat_[j] = s_grid->eta_hat_[s_grid->local_ny_ + j];
        }
    }
}

void BoundaryCondition::DefaultRightHatConditions(std::shared_ptr<SubGrid> s_grid)
{
    // Правый граничный слой
    if (s_grid->neighbor_right_) {
        // Получаем данные с правой сетке
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->H_hat_[(s_grid->local_nx_ - 1) * s_grid->local_ny_ + j] =
                s_grid->neighbor_right_->H_hat_[s_grid->local_ny_ + j];
            s_grid->u_hat_[(s_grid->local_nx_ - 1) * s_grid->local_ny_ + j] =
                s_grid->neighbor_right_->u_hat_[s_grid->local_ny_ + j];
            s_grid->v_hat_[(s_grid->local_nx_ - 1) * s_grid->local_ny_ + j] =
                s_grid->neighbor_right_->v_hat_[s_grid->local_ny_ + j];
            s_grid->eta_hat_[(s_grid->local_nx_ - 1) * s_grid->local_ny_ + j] =
                s_grid->neighbor_right_->eta_hat_[s_grid->local_ny_ + j];
        }
    }
    else {
        //RadiationRightHatConditions(s_grid);
        //return;
        // Граничные условия
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->H_hat_[(s_grid->local_nx_ - 1) * s_grid->local_ny_ + j] =
                s_grid->H_hat_[(s_grid->local_nx_ - 2) * s_grid->local_ny_ + j];
            s_grid->u_hat_[(s_grid->local_nx_ - 1) * s_grid->local_ny_ + j] =
                s_grid->u_hat_[(s_grid->local_nx_ - 2) * s_grid->local_ny_ + j];
            s_grid->v_hat_[(s_grid->local_nx_ - 1) * s_grid->local_ny_ + j] =
                s_grid->v_hat_[(s_grid->local_nx_ - 2) * s_grid->local_ny_ + j];
            s_grid->eta_hat_[(s_grid->local_nx_ - 1) * s_grid->local_ny_ + j] =
                s_grid->eta_hat_[(s_grid->local_nx_ - 2) * s_grid->local_ny_ + j];
        }
    }
}

// ===== DEFAULT-УСЛОВИЯ ДЛЯ ВЕРТИКАЛЬНОЙ РАЗБИВКИ (ghost rows по Y) =====

void BoundaryCondition::DefaultBottomConditions(std::shared_ptr<SubGrid> s_grid)
{
    // Bottom ghost row: j=0
    if (s_grid->neighbor_left_) {
        // neighbor_left_ = neighbor снизу (меньшие j)
        const int nbr_ny = s_grid->neighbor_left_->local_ny_;
        for (int i = 0; i < s_grid->local_nx_; ++i) {
            s_grid->local_grid_.H(i, 0) =
                s_grid->neighbor_left_->local_grid_.H(i, nbr_ny - 2);
            s_grid->local_grid_.u(i, 0) =
                s_grid->neighbor_left_->local_grid_.u(i, nbr_ny - 2);
            s_grid->local_grid_.v(i, 0) =
                s_grid->neighbor_left_->local_grid_.v(i, nbr_ny - 2);
            s_grid->local_grid_.eta(i, 0) =
                s_grid->neighbor_left_->local_grid_.eta(i, nbr_ny - 2);
        }
    } else {
        // Domain bottom edge: zero-gradient
        for (int i = 0; i < s_grid->local_nx_; ++i) {
            s_grid->local_grid_.H(i, 0) = s_grid->local_grid_.H(i, 1);
            s_grid->local_grid_.u(i, 0) = s_grid->local_grid_.u(i, 1);
            s_grid->local_grid_.v(i, 0) = s_grid->local_grid_.v(i, 1);
            s_grid->local_grid_.eta(i, 0) = s_grid->local_grid_.eta(i, 1);
        }
    }
}

void BoundaryCondition::DefaultTopConditions(std::shared_ptr<SubGrid> s_grid)
{
    // Top ghost row: j=ny-1
    const int ny = s_grid->local_ny_;
    if (s_grid->neighbor_right_) {
        // neighbor_right_ = neighbor сверху (большие j)
        for (int i = 0; i < s_grid->local_nx_; ++i) {
            s_grid->local_grid_.H(i, ny - 1) =
                s_grid->neighbor_right_->local_grid_.H(i, 1);
            s_grid->local_grid_.u(i, ny - 1) =
                s_grid->neighbor_right_->local_grid_.u(i, 1);
            s_grid->local_grid_.v(i, ny - 1) =
                s_grid->neighbor_right_->local_grid_.v(i, 1);
            s_grid->local_grid_.eta(i, ny - 1) =
                s_grid->neighbor_right_->local_grid_.eta(i, 1);
        }
    } else {
        // Domain top edge: zero-gradient
        for (int i = 0; i < s_grid->local_nx_; ++i) {
            s_grid->local_grid_.H(i, ny - 1) = s_grid->local_grid_.H(i, ny - 2);
            s_grid->local_grid_.u(i, ny - 1) = s_grid->local_grid_.u(i, ny - 2);
            s_grid->local_grid_.v(i, ny - 1) = s_grid->local_grid_.v(i, ny - 2);
            s_grid->local_grid_.eta(i, ny - 1) = s_grid->local_grid_.eta(i, ny - 2);
        }
    }
}

void BoundaryCondition::DefaultBottomHatConditions(std::shared_ptr<SubGrid> s_grid)
{
    const int ny = s_grid->local_ny_;
    if (s_grid->neighbor_left_) {
        const int nbr_ny = s_grid->neighbor_left_->local_ny_;
        for (int i = 0; i < s_grid->local_nx_; ++i) {
            // j=0 ghost row from neighbor's j=ny-2 interior row
            const int dst = i * ny + 0;
            const int src = i * nbr_ny + (nbr_ny - 2);
            s_grid->H_hat_[dst] = s_grid->neighbor_left_->H_hat_[src];
            s_grid->u_hat_[dst] = s_grid->neighbor_left_->u_hat_[src];
            s_grid->v_hat_[dst] = s_grid->neighbor_left_->v_hat_[src];
            s_grid->eta_hat_[dst] = s_grid->neighbor_left_->eta_hat_[src];
        }
    } else {
        for (int i = 0; i < s_grid->local_nx_; ++i) {
            const int dst = i * ny + 0;
            const int src = i * ny + 1;
            s_grid->H_hat_[dst] = s_grid->H_hat_[src];
            s_grid->u_hat_[dst] = s_grid->u_hat_[src];
            s_grid->v_hat_[dst] = s_grid->v_hat_[src];
            s_grid->eta_hat_[dst] = s_grid->eta_hat_[src];
        }
    }
}

void BoundaryCondition::DefaultTopHatConditions(std::shared_ptr<SubGrid> s_grid)
{
    const int ny = s_grid->local_ny_;
    if (s_grid->neighbor_right_) {
        const int nbr_ny = s_grid->neighbor_right_->local_ny_;
        for (int i = 0; i < s_grid->local_nx_; ++i) {
            const int dst = i * ny + (ny - 1);
            const int src = i * nbr_ny + 1;
            s_grid->H_hat_[dst] = s_grid->neighbor_right_->H_hat_[src];
            s_grid->u_hat_[dst] = s_grid->neighbor_right_->u_hat_[src];
            s_grid->v_hat_[dst] = s_grid->neighbor_right_->v_hat_[src];
            s_grid->eta_hat_[dst] = s_grid->neighbor_right_->eta_hat_[src];
        }
    } else {
        for (int i = 0; i < s_grid->local_nx_; ++i) {
            const int dst = i * ny + (ny - 1);
            const int src = i * ny + (ny - 2);
            s_grid->H_hat_[dst] = s_grid->H_hat_[src];
            s_grid->u_hat_[dst] = s_grid->u_hat_[src];
            s_grid->v_hat_[dst] = s_grid->v_hat_[src];
            s_grid->eta_hat_[dst] = s_grid->eta_hat_[src];
        }
    }
}

// ===== RADIATION (Sommerfeld 1D) ДЛЯ БАЗОВЫХ ПЕРЕМЕННЫХ =====
//
// Формула: eta_t + c * eta_n = 0, где c = sqrt(g * H), n — внешняя нормаль.
// Дискретизация (forward difference от внутренних точек):
//   ghost = interior1 + sigma * (interior1 - interior2),   sigma = c * dt / dn
// где dn — шаг в направлении нормали (dx для left/right, dy для top/bottom).
//
// Для сферических координат dx зависит от широты (j), поэтому для Left/Right
// используется предвычисленный inv_dx_arr_[j] = 1 / delta_x(j). Для Top/Bottom
// шаг dy постоянен (inv_dy_). Физическая скорость вычисляется через локальное
// H (с защитой от H<=0 в сухих ячейках).

namespace {
    inline double SafeWaveSpeed(double g, double H) {
        return (H > 0.0) ? std::sqrt(g * H) : 0.0;
    }
}

void BoundaryCondition::RadiationLeftConditions(std::shared_ptr<SubGrid> s)
{
    const double dt = s->tau_;
    for (int j = 0; j < s->local_ny_; ++j) {
        const double H1 = s->local_grid_.H(1, j);
        const double H2 = s->local_grid_.H(2, j);
        const double inv_dx = s->inv_dx_arr_[j];  // 1 / delta_x(lat(j))
        const double sigma = SafeWaveSpeed(s->g_, H1) * dt * inv_dx;

        s->local_grid_.H(0, j) = H1 + sigma * (H1 - H2);
        s->local_grid_.u(0, j) = s->local_grid_.u(1, j)
            + sigma * (s->local_grid_.u(1, j) - s->local_grid_.u(2, j));
        s->local_grid_.v(0, j) = s->local_grid_.v(1, j)
            + sigma * (s->local_grid_.v(1, j) - s->local_grid_.v(2, j));
        s->local_grid_.eta(0, j) = s->local_grid_.eta(1, j)
            + sigma * (s->local_grid_.eta(1, j) - s->local_grid_.eta(2, j));
    }
}

void BoundaryCondition::RadiationRightConditions(std::shared_ptr<SubGrid> s)
{
    const double dt = s->tau_;
    const int NX = s->local_nx_;

    for (int j = 0; j < s->local_ny_; ++j) {
        const double H1 = s->local_grid_.H(NX - 2, j);
        const double H2 = s->local_grid_.H(NX - 3, j);
        const double inv_dx = s->inv_dx_arr_[j];
        const double sigma = SafeWaveSpeed(s->g_, H1) * dt * inv_dx;

        s->local_grid_.H(NX - 1, j) = H1 + sigma * (H1 - H2);
        s->local_grid_.u(NX - 1, j) = s->local_grid_.u(NX - 2, j)
            + sigma * (s->local_grid_.u(NX - 2, j) - s->local_grid_.u(NX - 3, j));
        s->local_grid_.v(NX - 1, j) = s->local_grid_.v(NX - 2, j)
            + sigma * (s->local_grid_.v(NX - 2, j) - s->local_grid_.v(NX - 3, j));
        s->local_grid_.eta(NX - 1, j) = s->local_grid_.eta(NX - 2, j)
            + sigma * (s->local_grid_.eta(NX - 2, j) - s->local_grid_.eta(NX - 3, j));
    }
}

void BoundaryCondition::RadiationBottomConditions(std::shared_ptr<SubGrid> s)
{
    const double dt = s->tau_;
    const double inv_dy = s->inv_dy_;

    for (int i = 0; i < s->local_nx_; ++i) {
        const double H1 = s->local_grid_.H(i, 1);
        const double H2 = s->local_grid_.H(i, 2);
        const double sigma = SafeWaveSpeed(s->g_, H1) * dt * inv_dy;

        s->local_grid_.H(i, 0) = H1 + sigma * (H1 - H2);
        s->local_grid_.u(i, 0) = s->local_grid_.u(i, 1)
            + sigma * (s->local_grid_.u(i, 1) - s->local_grid_.u(i, 2));
        s->local_grid_.v(i, 0) = s->local_grid_.v(i, 1)
            + sigma * (s->local_grid_.v(i, 1) - s->local_grid_.v(i, 2));
        s->local_grid_.eta(i, 0) = s->local_grid_.eta(i, 1)
            + sigma * (s->local_grid_.eta(i, 1) - s->local_grid_.eta(i, 2));
    }
}

void BoundaryCondition::RadiationTopConditions(std::shared_ptr<SubGrid> s)
{
    const double dt = s->tau_;
    const double inv_dy = s->inv_dy_;
    const int NY = s->local_ny_;

    for (int i = 0; i < s->local_nx_; ++i) {
        const double H1 = s->local_grid_.H(i, NY - 2);
        const double H2 = s->local_grid_.H(i, NY - 3);
        const double sigma = SafeWaveSpeed(s->g_, H1) * dt * inv_dy;

        s->local_grid_.H(i, NY - 1) = H1 + sigma * (H1 - H2);
        s->local_grid_.u(i, NY - 1) = s->local_grid_.u(i, NY - 2)
            + sigma * (s->local_grid_.u(i, NY - 2) - s->local_grid_.u(i, NY - 3));
        s->local_grid_.v(i, NY - 1) = s->local_grid_.v(i, NY - 2)
            + sigma * (s->local_grid_.v(i, NY - 2) - s->local_grid_.v(i, NY - 3));
        s->local_grid_.eta(i, NY - 1) = s->local_grid_.eta(i, NY - 2)
            + sigma * (s->local_grid_.eta(i, NY - 2) - s->local_grid_.eta(i, NY - 3));
    }
}

// ===== RADIATION ДЛЯ HAT-ПЕРЕМЕННЫХ =====

void BoundaryCondition::RadiationLeftHatConditions(std::shared_ptr<SubGrid> s)
{
    const int NY = s->local_ny_;
    const double dt = s->tau_;

    for (int j = 0; j < NY; ++j) {
        const int idx1 = 1 * NY + j;
        const int idx2 = 2 * NY + j;

        const double H1 = s->H_hat_[idx1];
        const double H2 = s->H_hat_[idx2];
        const double u1 = s->u_hat_[idx1];
        const double u2 = s->u_hat_[idx2];
        const double v1 = s->v_hat_[idx1];
        const double v2 = s->v_hat_[idx2];
        const double eta1 = s->eta_hat_[idx1];
        const double eta2 = s->eta_hat_[idx2];

        const double sigma = SafeWaveSpeed(s->g_, H1) * dt * s->inv_dx_arr_[j];

        s->H_hat_[j] = H1 + sigma * (H1 - H2);
        s->u_hat_[j] = u1 + sigma * (u1 - u2);
        s->v_hat_[j] = v1 + sigma * (v1 - v2);
        s->eta_hat_[j] = eta1 + sigma * (eta1 - eta2);
    }
}

void BoundaryCondition::RadiationRightHatConditions(std::shared_ptr<SubGrid> s)
{
    const int NX = s->local_nx_;
    const int NY = s->local_ny_;
    const double dt = s->tau_;

    for (int j = 0; j < NY; ++j) {
        const int idx1 = (NX - 2) * NY + j;
        const int idx2 = (NX - 3) * NY + j;
        const int idxG = (NX - 1) * NY + j;

        const double H1 = s->H_hat_[idx1];
        const double H2 = s->H_hat_[idx2];
        const double u1 = s->u_hat_[idx1];
        const double u2 = s->u_hat_[idx2];
        const double v1 = s->v_hat_[idx1];
        const double v2 = s->v_hat_[idx2];
        const double eta1 = s->eta_hat_[idx1];
        const double eta2 = s->eta_hat_[idx2];

        const double sigma = SafeWaveSpeed(s->g_, H1) * dt * s->inv_dx_arr_[j];

        s->H_hat_[idxG] = H1 + sigma * (H1 - H2);
        s->u_hat_[idxG] = u1 + sigma * (u1 - u2);
        s->v_hat_[idxG] = v1 + sigma * (v1 - v2);
        s->eta_hat_[idxG] = eta1 + sigma * (eta1 - eta2);
    }
}

void BoundaryCondition::RadiationBottomHatConditions(std::shared_ptr<SubGrid> s)
{
    const int NY = s->local_ny_;
    const double dt = s->tau_;
    const double inv_dy = s->inv_dy_;

    for (int i = 0; i < s->local_nx_; ++i) {
        const int idx0 = i * NY + 0;
        const int idx1 = i * NY + 1;
        const int idx2 = i * NY + 2;

        const double H1 = s->H_hat_[idx1];
        const double H2 = s->H_hat_[idx2];
        const double u1 = s->u_hat_[idx1];
        const double u2 = s->u_hat_[idx2];
        const double v1 = s->v_hat_[idx1];
        const double v2 = s->v_hat_[idx2];
        const double eta1 = s->eta_hat_[idx1];
        const double eta2 = s->eta_hat_[idx2];

        const double sigma = SafeWaveSpeed(s->g_, H1) * dt * inv_dy;

        s->H_hat_[idx0] = H1 + sigma * (H1 - H2);
        s->u_hat_[idx0] = u1 + sigma * (u1 - u2);
        s->v_hat_[idx0] = v1 + sigma * (v1 - v2);
        s->eta_hat_[idx0] = eta1 + sigma * (eta1 - eta2);
    }
}

void BoundaryCondition::RadiationTopHatConditions(std::shared_ptr<SubGrid> s)
{
    const int NY = s->local_ny_;
    const double dt = s->tau_;
    const double inv_dy = s->inv_dy_;

    for (int i = 0; i < s->local_nx_; ++i) {
        const int idxG = i * NY + (NY - 1);
        const int idx1 = i * NY + (NY - 2);
        const int idx2 = i * NY + (NY - 3);

        const double H1 = s->H_hat_[idx1];
        const double H2 = s->H_hat_[idx2];
        const double u1 = s->u_hat_[idx1];
        const double u2 = s->u_hat_[idx2];
        const double v1 = s->v_hat_[idx1];
        const double v2 = s->v_hat_[idx2];
        const double eta1 = s->eta_hat_[idx1];
        const double eta2 = s->eta_hat_[idx2];

        const double sigma = SafeWaveSpeed(s->g_, H1) * dt * inv_dy;

        s->H_hat_[idxG] = H1 + sigma * (H1 - H2);
        s->u_hat_[idxG] = u1 + sigma * (u1 - u2);
        s->v_hat_[idxG] = v1 + sigma * (v1 - v2);
        s->eta_hat_[idxG] = eta1 + sigma * (eta1 - eta2);
    }
}

// ===== FLATHER 1976 OPEN BOUNDARY CONDITION (index 5) =====
//
// Characteristic open BC: u_n = u_ext + sqrt(g/H_eff) * (eta - eta_ext)
// with eta_ext = 0, u_ext = 0 (open ocean at rest). Coriolis NOT in BC (stays in interior).
//
// Per-side outward-normal sign:
//   left  (i=0):     u_n = -u → u_ghost = -c*eta
//   right (i=nx-1):  u_n = +u → u_ghost = +c*eta
//   bottom (j=0):    u_n = -v → v_ghost = -c*eta
//   top   (j=ny-1):  u_n = +v → v_ghost = +c*eta
//
// Codex-mandated patterns:
//   - eta_ghost = eta_interior (zero-gradient, NOT 0)
//   - tangential velocity = zero-gradient from interior
//   - H_eff = max(H_interior, H_min) safety (avoid sqrt of near-zero)
//   - dry boundary: hold ghost as dry (H = D, u=v=eta=0)

namespace {
    inline double FlatherWaveSpeed(double g, double H_interior, double H_min) {
        const double H_eff = (H_interior > H_min) ? H_interior : H_min;
        return std::sqrt(g / H_eff);
    }
}

void BoundaryCondition::FlatherLeftConditions(std::shared_ptr<SubGrid> s)
{
    const double g = s->g_;
    const double H_min = s->min_calc_value;
    for (int j = 0; j < s->local_ny_; ++j) {
        const double H1 = s->local_grid_.H(1, j);
        const double eta1 = s->local_grid_.eta(1, j);
        if (H1 < H_min) {
            s->local_grid_.H(0, j)   = s->local_grid_.D(0, j);
            s->local_grid_.u(0, j)   = 0.0;
            s->local_grid_.v(0, j)   = 0.0;
            s->local_grid_.eta(0, j) = 0.0;
            continue;
        }
        const double c = FlatherWaveSpeed(g, H1, H_min);
        const double D_ghost = s->local_grid_.D(0, j);
        s->local_grid_.eta(0, j) = eta1;                    // zero-gradient
        s->local_grid_.H(0, j)   = D_ghost + eta1;          // consistent H
        s->local_grid_.u(0, j)   = -c * eta1;               // outward = -x
        s->local_grid_.v(0, j)   = s->local_grid_.v(1, j);  // tangent zero-gradient
    }
}

void BoundaryCondition::FlatherRightConditions(std::shared_ptr<SubGrid> s)
{
    const double g = s->g_;
    const double H_min = s->min_calc_value;
    const int NX = s->local_nx_;
    for (int j = 0; j < s->local_ny_; ++j) {
        const double H1 = s->local_grid_.H(NX - 2, j);
        const double eta1 = s->local_grid_.eta(NX - 2, j);
        if (H1 < H_min) {
            s->local_grid_.H(NX - 1, j)   = s->local_grid_.D(NX - 1, j);
            s->local_grid_.u(NX - 1, j)   = 0.0;
            s->local_grid_.v(NX - 1, j)   = 0.0;
            s->local_grid_.eta(NX - 1, j) = 0.0;
            continue;
        }
        const double c = FlatherWaveSpeed(g, H1, H_min);
        const double D_ghost = s->local_grid_.D(NX - 1, j);
        s->local_grid_.eta(NX - 1, j) = eta1;
        s->local_grid_.H(NX - 1, j)   = D_ghost + eta1;
        s->local_grid_.u(NX - 1, j)   = +c * eta1;          // outward = +x
        s->local_grid_.v(NX - 1, j)   = s->local_grid_.v(NX - 2, j);
    }
}

void BoundaryCondition::FlatherBottomConditions(std::shared_ptr<SubGrid> s)
{
    const double g = s->g_;
    const double H_min = s->min_calc_value;
    for (int i = 0; i < s->local_nx_; ++i) {
        const double H1 = s->local_grid_.H(i, 1);
        const double eta1 = s->local_grid_.eta(i, 1);
        if (H1 < H_min) {
            s->local_grid_.H(i, 0)   = s->local_grid_.D(i, 0);
            s->local_grid_.u(i, 0)   = 0.0;
            s->local_grid_.v(i, 0)   = 0.0;
            s->local_grid_.eta(i, 0) = 0.0;
            continue;
        }
        const double c = FlatherWaveSpeed(g, H1, H_min);
        const double D_ghost = s->local_grid_.D(i, 0);
        s->local_grid_.eta(i, 0) = eta1;
        s->local_grid_.H(i, 0)   = D_ghost + eta1;
        s->local_grid_.v(i, 0)   = -c * eta1;               // outward = -y
        s->local_grid_.u(i, 0)   = s->local_grid_.u(i, 1);  // tangent zero-gradient
    }
}

void BoundaryCondition::FlatherTopConditions(std::shared_ptr<SubGrid> s)
{
    const double g = s->g_;
    const double H_min = s->min_calc_value;
    const int NY = s->local_ny_;
    for (int i = 0; i < s->local_nx_; ++i) {
        const double H1 = s->local_grid_.H(i, NY - 2);
        const double eta1 = s->local_grid_.eta(i, NY - 2);
        if (H1 < H_min) {
            s->local_grid_.H(i, NY - 1)   = s->local_grid_.D(i, NY - 1);
            s->local_grid_.u(i, NY - 1)   = 0.0;
            s->local_grid_.v(i, NY - 1)   = 0.0;
            s->local_grid_.eta(i, NY - 1) = 0.0;
            continue;
        }
        const double c = FlatherWaveSpeed(g, H1, H_min);
        const double D_ghost = s->local_grid_.D(i, NY - 1);
        s->local_grid_.eta(i, NY - 1) = eta1;
        s->local_grid_.H(i, NY - 1)   = D_ghost + eta1;
        s->local_grid_.v(i, NY - 1)   = +c * eta1;          // outward = +y
        s->local_grid_.u(i, NY - 1)   = s->local_grid_.u(i, NY - 2);
    }
}

// ===== FLATHER HAT VERSIONS (predictor step) =====

void BoundaryCondition::FlatherLeftHatConditions(std::shared_ptr<SubGrid> s)
{
    const double g = s->g_;
    const double H_min = s->min_calc_value;
    const int NY = s->local_ny_;
    for (int j = 0; j < NY; ++j) {
        const int idxG = 0 * NY + j;
        const int idx1 = 1 * NY + j;
        const double H1 = s->H_hat_[idx1];
        const double eta1 = s->eta_hat_[idx1];
        if (H1 < H_min) {
            s->H_hat_[idxG]   = s->local_grid_.D(0, j);
            s->u_hat_[idxG]   = 0.0;
            s->v_hat_[idxG]   = 0.0;
            s->eta_hat_[idxG] = 0.0;
            continue;
        }
        const double c = FlatherWaveSpeed(g, H1, H_min);
        const double D_ghost = s->local_grid_.D(0, j);
        s->eta_hat_[idxG] = eta1;
        s->H_hat_[idxG]   = D_ghost + eta1;
        s->u_hat_[idxG]   = -c * eta1;
        s->v_hat_[idxG]   = s->v_hat_[idx1];
    }
}

void BoundaryCondition::FlatherRightHatConditions(std::shared_ptr<SubGrid> s)
{
    const double g = s->g_;
    const double H_min = s->min_calc_value;
    const int NX = s->local_nx_;
    const int NY = s->local_ny_;
    for (int j = 0; j < NY; ++j) {
        const int idxG = (NX - 1) * NY + j;
        const int idx1 = (NX - 2) * NY + j;
        const double H1 = s->H_hat_[idx1];
        const double eta1 = s->eta_hat_[idx1];
        if (H1 < H_min) {
            s->H_hat_[idxG]   = s->local_grid_.D(NX - 1, j);
            s->u_hat_[idxG]   = 0.0;
            s->v_hat_[idxG]   = 0.0;
            s->eta_hat_[idxG] = 0.0;
            continue;
        }
        const double c = FlatherWaveSpeed(g, H1, H_min);
        const double D_ghost = s->local_grid_.D(NX - 1, j);
        s->eta_hat_[idxG] = eta1;
        s->H_hat_[idxG]   = D_ghost + eta1;
        s->u_hat_[idxG]   = +c * eta1;
        s->v_hat_[idxG]   = s->v_hat_[idx1];
    }
}

void BoundaryCondition::FlatherBottomHatConditions(std::shared_ptr<SubGrid> s)
{
    const double g = s->g_;
    const double H_min = s->min_calc_value;
    const int NY = s->local_ny_;
    for (int i = 0; i < s->local_nx_; ++i) {
        const int idxG = i * NY + 0;
        const int idx1 = i * NY + 1;
        const double H1 = s->H_hat_[idx1];
        const double eta1 = s->eta_hat_[idx1];
        if (H1 < H_min) {
            s->H_hat_[idxG]   = s->local_grid_.D(i, 0);
            s->u_hat_[idxG]   = 0.0;
            s->v_hat_[idxG]   = 0.0;
            s->eta_hat_[idxG] = 0.0;
            continue;
        }
        const double c = FlatherWaveSpeed(g, H1, H_min);
        const double D_ghost = s->local_grid_.D(i, 0);
        s->eta_hat_[idxG] = eta1;
        s->H_hat_[idxG]   = D_ghost + eta1;
        s->v_hat_[idxG]   = -c * eta1;
        s->u_hat_[idxG]   = s->u_hat_[idx1];
    }
}

void BoundaryCondition::FlatherTopHatConditions(std::shared_ptr<SubGrid> s)
{
    const double g = s->g_;
    const double H_min = s->min_calc_value;
    const int NY = s->local_ny_;
    for (int i = 0; i < s->local_nx_; ++i) {
        const int idxG = i * NY + (NY - 1);
        const int idx1 = i * NY + (NY - 2);
        const double H1 = s->H_hat_[idx1];
        const double eta1 = s->eta_hat_[idx1];
        if (H1 < H_min) {
            s->H_hat_[idxG]   = s->local_grid_.D(i, NY - 1);
            s->u_hat_[idxG]   = 0.0;
            s->v_hat_[idxG]   = 0.0;
            s->eta_hat_[idxG] = 0.0;
            continue;
        }
        const double c = FlatherWaveSpeed(g, H1, H_min);
        const double D_ghost = s->local_grid_.D(i, NY - 1);
        s->eta_hat_[idxG] = eta1;
        s->H_hat_[idxG]   = D_ghost + eta1;
        s->v_hat_[idxG]   = +c * eta1;
        s->u_hat_[idxG]   = s->u_hat_[idx1];
    }
}

// ===== ПОЛНЫЕ СЦЕНАРИИ ОБМЕНА ДАННЫМИ (H/u/v/eta) =====

void BoundaryCondition::ExchangeBoundaryData1(std::shared_ptr<SubGrid> s_grid, double /*t*/)
{
    if (s_grid->GetDecompDir() == DecompositionDirection::Horizontal) {
        // Horizontal: exchange ghost columns (X), free-surface on Y domain edges
        DefaultLeftConditions(s_grid);
        DefaultRightConditions(s_grid);

        for (int i = 0; i < s_grid->local_nx_; ++i) {
            s_grid->local_grid_.H(i, 0) = s_grid->local_grid_.H(i, 1);
            s_grid->local_grid_.u(i, 0) = s_grid->local_grid_.u(i, 1);
            s_grid->local_grid_.v(i, 0) = s_grid->local_grid_.v(i, 1);
            s_grid->local_grid_.eta(i, 0) = s_grid->local_grid_.eta(i, 1);

            s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.H(i, s_grid->local_ny_ - 2);
            s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.u(i, s_grid->local_ny_ - 2);
            s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.v(i, s_grid->local_ny_ - 2);
            s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.eta(i, s_grid->local_ny_ - 2);
        }
    } else {
        // Vertical: exchange Y-ghost rows (strip boundaries) + X-ghost columns (domain edges)
        DefaultBottomConditions(s_grid);
        DefaultTopConditions(s_grid);

        // Y-domain edge free-surface: if no bottom neighbor, j=1 is global j=0 (domain edge)
        // Force zero-gradient to match horizontal mode behavior
        if (!s_grid->neighbor_left_) {
            for (int i = 0; i < s_grid->local_nx_; ++i) {
                s_grid->local_grid_.H(i, 1) = s_grid->local_grid_.H(i, 2);
                s_grid->local_grid_.u(i, 1) = s_grid->local_grid_.u(i, 2);
                s_grid->local_grid_.v(i, 1) = s_grid->local_grid_.v(i, 2);
                s_grid->local_grid_.eta(i, 1) = s_grid->local_grid_.eta(i, 2);
            }
        }
        if (!s_grid->neighbor_right_) {
            const int jlast = s_grid->local_ny_ - 2; // last real row (global ny-1)
            for (int i = 0; i < s_grid->local_nx_; ++i) {
                s_grid->local_grid_.H(i, jlast) = s_grid->local_grid_.H(i, jlast - 1);
                s_grid->local_grid_.u(i, jlast) = s_grid->local_grid_.u(i, jlast - 1);
                s_grid->local_grid_.v(i, jlast) = s_grid->local_grid_.v(i, jlast - 1);
                s_grid->local_grid_.eta(i, jlast) = s_grid->local_grid_.eta(i, jlast - 1);
            }
        }

        // X-ghost columns (domain edges): zero-gradient
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
            s_grid->local_grid_.u(0, j) = s_grid->local_grid_.u(1, j);
            s_grid->local_grid_.v(0, j) = s_grid->local_grid_.v(1, j);
            s_grid->local_grid_.eta(0, j) = s_grid->local_grid_.eta(1, j);

            s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.H(s_grid->local_nx_ - 2, j);
            s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.u(s_grid->local_nx_ - 2, j);
            s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.v(s_grid->local_nx_ - 2, j);
            s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.eta(s_grid->local_nx_ - 2, j);
        }
    }
}

void BoundaryCondition::ExchangeBoundaryData2(std::shared_ptr<SubGrid> s_grid, double /*t*/)
{
    DefaultLeftConditions(s_grid);
    DefaultRightConditions(s_grid);

    for (int i = 0; i < s_grid->local_nx_; ++i) {
        s_grid->local_grid_.H(i, 0) = s_grid->local_grid_.H(i, 1);
        s_grid->local_grid_.u(i, 0) = s_grid->local_grid_.u(i, 1);
        s_grid->local_grid_.v(i, 0) = 0.0; 
        s_grid->local_grid_.eta(i, 0) = s_grid->local_grid_.eta(i, 1);

        s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.H(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.u(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) = 0.0;
        s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.eta(i, s_grid->local_ny_ - 2);
    }
}

void BoundaryCondition::ExchangeBoundaryData3(std::shared_ptr<SubGrid> s_grid, double t)
{
    // Верхний граничный слой
    if (s_grid->neighbor_left_) {
        // Отправляем свои данные верхней сетке и получаем его данные
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->local_grid_.H(0, j) = s_grid->neighbor_left_->local_grid_.H(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.u(0, j) = s_grid->neighbor_left_->local_grid_.u(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.v(0, j) = s_grid->neighbor_left_->local_grid_.v(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.eta(0, j) = s_grid->neighbor_left_->local_grid_.eta(s_grid->neighbor_left_->local_nx_ - 2, j);
        }
    }
    else {
        double T = 300.;
        double waveHeight = 2.0 / 2.0 * (1.0 - cos(t * s_grid->tau_ * 2 * M_PI / T));
        // Граничные условия на верхней границе
        if (t <= T / s_grid->tau_) {
            for (int j = 0; j < s_grid->local_ny_; ++j) {
                s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
                s_grid->local_grid_.v(0, j) = 0;
                s_grid->local_grid_.eta(0, j) = waveHeight;
                s_grid->local_grid_.u(0, j) = s_grid->local_grid_.D(0, j) > 0.0 ?
                    1.0 * sqrt(s_grid->g_ / s_grid->local_grid_.D(0, j)) :
                    0.0;
                s_grid->local_grid_.H(0, j) = s_grid->local_grid_.D(0, j) + s_grid->local_grid_.eta(0, j);
            }
        }
        else
        {
            for (int j = 0; j < s_grid->local_ny_; ++j) {
                s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
                s_grid->local_grid_.u(0, j) = s_grid->local_grid_.u(1, j);
                s_grid->local_grid_.v(0, j) = s_grid->local_grid_.v(1, j);
                s_grid->local_grid_.eta(0, j) = s_grid->local_grid_.eta(1, j);
            }
        }
    }

    DefaultRightConditions(s_grid);

    for (int i = 0; i < s_grid->local_nx_; ++i) {
        s_grid->local_grid_.H(i, 0) = s_grid->local_grid_.H(i, 1);
        s_grid->local_grid_.u(i, 0) = s_grid->local_grid_.u(i, 1);
        s_grid->local_grid_.v(i, 0) = s_grid->local_grid_.v(i, 1);
        s_grid->local_grid_.eta(i, 0) = s_grid->local_grid_.eta(i, 1);
        s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.H(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.u(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.v(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.eta(i, s_grid->local_ny_ - 2);
    }
}

void BoundaryCondition::ExchangeBoundaryData4(std::shared_ptr<SubGrid> s_grid, double t)
{
    // Верхний граничный слой
    if (s_grid->neighbor_left_) {
        // Отправляем свои данные верхней сетке и получаем его данные
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->local_grid_.H(0, j) = s_grid->neighbor_left_->local_grid_.H(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.u(0, j) = s_grid->neighbor_left_->local_grid_.u(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.v(0, j) = s_grid->neighbor_left_->local_grid_.v(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.eta(0, j) = s_grid->neighbor_left_->local_grid_.eta(s_grid->neighbor_left_->local_nx_ - 2, j);
        }
    }
    else {
        double T = 300.;
        double waveHeight = 1.0 / 2.0 * (1.0 - cos(t * s_grid->tau_ * 2 * M_PI / T));
        // Граничные условия на левой границе
        if (t <= T / s_grid->tau_) {
            for (int j = 0; j < s_grid->local_ny_; ++j) {
                s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
                s_grid->local_grid_.v(0, j) = 0;
                s_grid->local_grid_.eta(0, j) = waveHeight;
                s_grid->local_grid_.u(0, j) = s_grid->local_grid_.D(0, j) > 0.0 ?
                    1.0 * sqrt(s_grid->g_ / s_grid->local_grid_.D(0, j)) :
                    0.0;
                s_grid->local_grid_.H(0, j) = s_grid->local_grid_.D(0, j) + s_grid->local_grid_.eta(0, j);
            }
        }
        else
        {
            for (int j = 0; j < s_grid->local_ny_; ++j) {
                s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
                s_grid->local_grid_.u(0, j) = s_grid->local_grid_.u(1, j);
                s_grid->local_grid_.v(0, j) = s_grid->local_grid_.v(1, j);
                s_grid->local_grid_.eta(0, j) = s_grid->local_grid_.eta(1, j);
            }
        }
    }

    DefaultRightConditions(s_grid);

    for (int i = 0; i < s_grid->local_nx_; ++i) {
        // Top lines
        s_grid->local_grid_.H(i, 0) = s_grid->local_grid_.H(i, 1);
        s_grid->local_grid_.u(i, 0) = s_grid->local_grid_.u(i, 1);
        s_grid->local_grid_.v(i, 0) = s_grid->local_grid_.v(i, 1);
        s_grid->local_grid_.eta(i, 0) = s_grid->local_grid_.eta(i, 1);
        // Bottom lines
        s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.H(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.u(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.v(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.eta(i, s_grid->local_ny_ - 2);
    }
}

void BoundaryCondition::ExchangeBoundaryData5(std::shared_ptr<SubGrid> s_grid, double t)
{
    // Верхний граничный слой
    if (s_grid->neighbor_left_) {
        // Отправляем свои данные верхней сетке и получаем его данные
        for (int j = 0; j < s_grid->local_ny_; ++j) {
            s_grid->local_grid_.H(0, j) = s_grid->neighbor_left_->local_grid_.H(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.u(0, j) = s_grid->neighbor_left_->local_grid_.u(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.v(0, j) = s_grid->neighbor_left_->local_grid_.v(s_grid->neighbor_left_->local_nx_ - 2, j);
            s_grid->local_grid_.eta(0, j) = s_grid->neighbor_left_->local_grid_.eta(s_grid->neighbor_left_->local_nx_ - 2, j);
        }
    }
    else {
        double T = 300.;
        double waveHeight = 1.0 / 2.0 * (1.0 - cos(t * s_grid->tau_ * 2 * M_PI / T));
        // Граничные условия на верхней границе
        if (t <= T / s_grid->tau_) {
            for (int j = 0; j < s_grid->local_ny_; ++j) {
                s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
                s_grid->local_grid_.v(0, j) = 0;
                s_grid->local_grid_.eta(0, j) = waveHeight;
                s_grid->local_grid_.u(0, j) = s_grid->local_grid_.D(0, j) > 0.0 ?
                    1.0 * sqrt(s_grid->g_ / s_grid->local_grid_.D(0, j)) :
                    0.0;
                s_grid->local_grid_.H(0, j) = s_grid->local_grid_.D(0, j) + s_grid->local_grid_.eta(0, j);
            }
        }
        else
        {
            for (int j = 0; j < s_grid->local_ny_; ++j) {
                s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
                s_grid->local_grid_.u(0, j) = s_grid->local_grid_.u(1, j);
                s_grid->local_grid_.v(0, j) = s_grid->local_grid_.v(1, j);
                s_grid->local_grid_.eta(0, j) = s_grid->local_grid_.eta(1, j);
            }
        }
    }

    DefaultRightConditions(s_grid);

    for (int i = 0; i < s_grid->local_nx_; ++i) {
        s_grid->local_grid_.H(i, 0) = s_grid->local_grid_.H(i, 1);
        s_grid->local_grid_.u(i, 0) = s_grid->local_grid_.u(i, 1);
        s_grid->local_grid_.v(i, 0) = s_grid->local_grid_.v(i, 1);
        s_grid->local_grid_.eta(i, 0) = s_grid->local_grid_.eta(i, 1);
        s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.H(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.u(i, s_grid->local_ny_ - 2);
        s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) = 0;
        s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.eta(i, s_grid->local_ny_ - 2);
    }


}

// ===== HAT-СЦЕНАРИИ (аналогично) =====

void BoundaryCondition::ExchangeHatBoundaryData1(std::shared_ptr<SubGrid> s_grid, double /*t*/)
{
    if (s_grid->GetDecompDir() == DecompositionDirection::Horizontal) {
        DefaultLeftHatConditions(s_grid);
        DefaultRightHatConditions(s_grid);

        for (int i = 0; i < s_grid->local_nx_; ++i) {
            s_grid->H_hat_[i * s_grid->local_ny_] =
                s_grid->H_hat_[i * s_grid->local_ny_ + 1];
            s_grid->u_hat_[i * s_grid->local_ny_] =
                s_grid->u_hat_[i * s_grid->local_ny_ + 1];
            s_grid->v_hat_[i * s_grid->local_ny_] =
                s_grid->v_hat_[i * s_grid->local_ny_ + 1];
            s_grid->eta_hat_[i * s_grid->local_ny_] =
                s_grid->eta_hat_[i * s_grid->local_ny_ + 1];

            s_grid->H_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
                s_grid->H_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 2];
            s_grid->u_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
                s_grid->u_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 2];
            s_grid->v_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
                s_grid->v_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 2];
            s_grid->eta_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
                s_grid->eta_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 2];
        }
    } else {
        // Vertical: exchange Y-ghost rows + X-ghost columns (domain edges)
        DefaultBottomHatConditions(s_grid);
        DefaultTopHatConditions(s_grid);

        // Y-domain edge hat: force zero-gradient on first/last real row
        const int ny = s_grid->local_ny_;
        if (!s_grid->neighbor_left_) {
            for (int i = 0; i < s_grid->local_nx_; ++i) {
                const int dst = i * ny + 1;
                const int src = i * ny + 2;
                s_grid->H_hat_[dst] = s_grid->H_hat_[src];
                s_grid->u_hat_[dst] = s_grid->u_hat_[src];
                s_grid->v_hat_[dst] = s_grid->v_hat_[src];
                s_grid->eta_hat_[dst] = s_grid->eta_hat_[src];
            }
        }
        if (!s_grid->neighbor_right_) {
            const int jlast = ny - 2;
            for (int i = 0; i < s_grid->local_nx_; ++i) {
                const int dst = i * ny + jlast;
                const int src = i * ny + (jlast - 1);
                s_grid->H_hat_[dst] = s_grid->H_hat_[src];
                s_grid->u_hat_[dst] = s_grid->u_hat_[src];
                s_grid->v_hat_[dst] = s_grid->v_hat_[src];
                s_grid->eta_hat_[dst] = s_grid->eta_hat_[src];
            }
        }

        // X-ghost hat columns: zero-gradient
        for (int j = 0; j < ny; ++j) {
            s_grid->H_hat_[0 * ny + j] = s_grid->H_hat_[1 * ny + j];
            s_grid->u_hat_[0 * ny + j] = s_grid->u_hat_[1 * ny + j];
            s_grid->v_hat_[0 * ny + j] = s_grid->v_hat_[1 * ny + j];
            s_grid->eta_hat_[0 * ny + j] = s_grid->eta_hat_[1 * ny + j];

            const int last_i = s_grid->local_nx_ - 1;
            s_grid->H_hat_[last_i * ny + j] = s_grid->H_hat_[(last_i - 1) * ny + j];
            s_grid->u_hat_[last_i * ny + j] = s_grid->u_hat_[(last_i - 1) * ny + j];
            s_grid->v_hat_[last_i * ny + j] = s_grid->v_hat_[(last_i - 1) * ny + j];
            s_grid->eta_hat_[last_i * ny + j] = s_grid->eta_hat_[(last_i - 1) * ny + j];
        }
    }
}

void BoundaryCondition::ExchangeHatBoundaryData2(std::shared_ptr<SubGrid> s_grid, double /*t*/)
{
    DefaultLeftHatConditions(s_grid);
    DefaultRightHatConditions(s_grid);

    for (int i = 0; i < s_grid->local_nx_; ++i) {
        s_grid->H_hat_[i * s_grid->local_ny_] =
            s_grid->H_hat_[i * s_grid->local_ny_ + 1];
        s_grid->u_hat_[i * s_grid->local_ny_] =
            s_grid->u_hat_[i * s_grid->local_ny_ + 1];
        s_grid->v_hat_[i * s_grid->local_ny_] = 0.0; //v_hat_[i*local_ny_+1];
        s_grid->eta_hat_[i * s_grid->local_ny_] =
            s_grid->eta_hat_[i * s_grid->local_ny_ + 1];

        s_grid->H_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
            s_grid->H_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 2];
        s_grid->u_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
            s_grid->u_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 2];
        s_grid->v_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] = 0.0;
        s_grid->eta_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
            s_grid->eta_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 2];
    }
}

void BoundaryCondition::ExchangeHatBoundaryData3(std::shared_ptr<SubGrid> s_grid, double t)
{
    DefaultLeftHatConditions(s_grid);
    DefaultRightHatConditions(s_grid);

    for (int i = 0; i < s_grid->local_nx_; ++i) {
        s_grid->H_hat_[i * s_grid->local_ny_] =
            s_grid->H_hat_[i * s_grid->local_ny_ + 1];
        s_grid->u_hat_[i * s_grid->local_ny_] =
            s_grid->u_hat_[i * s_grid->local_ny_ + 1];
        s_grid->v_hat_[i * s_grid->local_ny_] =
            s_grid->v_hat_[i * s_grid->local_ny_ + 1];
        s_grid->eta_hat_[i * s_grid->local_ny_] =
            s_grid->eta_hat_[i * s_grid->local_ny_ + 1];
        s_grid->H_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
            s_grid->H_hat_[(i)*s_grid->local_ny_ + s_grid->local_ny_ - 2];
        s_grid->u_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
            s_grid->u_hat_[(i)*s_grid->local_ny_ + s_grid->local_ny_ - 2];
        s_grid->v_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] = 0;
        s_grid->eta_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
            s_grid->eta_hat_[(i)*s_grid->local_ny_ + s_grid->local_ny_ - 2];
    }
}

void BoundaryCondition::ExchangeHatBoundaryData4(std::shared_ptr<SubGrid> s_grid, double t)
{
    DefaultLeftHatConditions(s_grid);
    DefaultRightHatConditions(s_grid);
    int linePos = -1;
    for (int i = 0; i < s_grid->local_nx_; ++i) {
        linePos = i * s_grid->local_ny_;
        s_grid->H_hat_[linePos] =
            s_grid->H_hat_[linePos + 1];
        s_grid->u_hat_[linePos] =
            s_grid->u_hat_[linePos + 1];
        s_grid->v_hat_[linePos] =
            s_grid->v_hat_[linePos + 1];
        s_grid->eta_hat_[linePos] =
            s_grid->eta_hat_[linePos + 1];
        s_grid->H_hat_[linePos + s_grid->local_ny_ - 1] =
            s_grid->H_hat_[linePos + s_grid->local_ny_ - 2];
        s_grid->u_hat_[linePos + s_grid->local_ny_ - 1] =
            s_grid->u_hat_[linePos + s_grid->local_ny_ - 2];
        s_grid->v_hat_[linePos + s_grid->local_ny_ - 1] =
            s_grid->v_hat_[linePos + s_grid->local_ny_ - 2];
        s_grid->eta_hat_[linePos + s_grid->local_ny_ - 1] =
            s_grid->eta_hat_[linePos + s_grid->local_ny_ - 2];
    }
}

void BoundaryCondition::ExchangeHatBoundaryData5(std::shared_ptr<SubGrid> s_grid, double t)
{
    DefaultLeftHatConditions(s_grid);
    DefaultRightHatConditions(s_grid);
    int linePos = -1;
    for (int i = 0; i < s_grid->local_nx_; ++i) {
        linePos = i * s_grid->local_ny_;
        s_grid->H_hat_[linePos] =
            s_grid->H_hat_[linePos + 1];
        s_grid->u_hat_[linePos] =
            s_grid->u_hat_[linePos + 1];
        s_grid->v_hat_[linePos] =
            s_grid->v_hat_[linePos + 1];
        s_grid->eta_hat_[linePos] =
            s_grid->eta_hat_[linePos + 1];
        s_grid->H_hat_[linePos + s_grid->local_ny_ - 1] =
            s_grid->H_hat_[linePos + s_grid->local_ny_ - 2];
        s_grid->u_hat_[linePos + s_grid->local_ny_ - 1] =
            s_grid->u_hat_[linePos + s_grid->local_ny_ - 2];
        s_grid->v_hat_[linePos + s_grid->local_ny_ - 1] =
            s_grid->v_hat_[linePos + s_grid->local_ny_ - 2];
        s_grid->eta_hat_[linePos + s_grid->local_ny_ - 1] =
            s_grid->eta_hat_[linePos + s_grid->local_ny_ - 2];
    }
}

// ===== SOMMERFELD RADIATION НА ВСЕХ 4 ГРАНИЦАХ ДОМЕНА =====
//
// Внутренние границы между подсетками (есть neighbor_left/neighbor_right)
// обмениваются обычным способом через Default* условия. Только внешние
// границы всего домена (нет соседа) получают Sommerfeld radiation.
// Это позволяет использовать Radiation BC совместно с декомпозицией
// (горизонтальной или вертикальной).

void BoundaryCondition::ExchangeBoundaryDataRadiation(std::shared_ptr<SubGrid> s_grid, double /*t*/)
{
    if (s_grid->GetDecompDir() == DecompositionDirection::Horizontal) {
        // Horizontal decomposition: подсетки соседствуют по X.
        // Left/Right: либо сосед (DefaultLeft/Right copy ghosts), либо Radiation.
        // Top/Bottom: всегда domain edge -> Radiation.
        if (s_grid->neighbor_left_) {
            DefaultLeftConditions(s_grid);
        } else {
            RadiationLeftConditions(s_grid);
        }
        if (s_grid->neighbor_right_) {
            DefaultRightConditions(s_grid);
        } else {
            RadiationRightConditions(s_grid);
        }
        RadiationBottomConditions(s_grid);
        RadiationTopConditions(s_grid);
    } else {
        // Vertical decomposition: подсетки соседствуют по Y.
        // Bottom/Top: либо сосед, либо Radiation.
        // Left/Right: всегда domain edge -> Radiation.
        if (s_grid->neighbor_left_) {  // neighbor_left_ = snake bottom в vertical mode
            DefaultBottomConditions(s_grid);
        } else {
            RadiationBottomConditions(s_grid);
        }
        if (s_grid->neighbor_right_) {
            DefaultTopConditions(s_grid);
        } else {
            RadiationTopConditions(s_grid);
        }
        RadiationLeftConditions(s_grid);
        RadiationRightConditions(s_grid);
    }
}

void BoundaryCondition::ExchangeHatBoundaryDataRadiation(std::shared_ptr<SubGrid> s_grid, double /*t*/)
{
    if (s_grid->GetDecompDir() == DecompositionDirection::Horizontal) {
        if (s_grid->neighbor_left_) {
            DefaultLeftHatConditions(s_grid);
        } else {
            RadiationLeftHatConditions(s_grid);
        }
        if (s_grid->neighbor_right_) {
            DefaultRightHatConditions(s_grid);
        } else {
            RadiationRightHatConditions(s_grid);
        }
        RadiationBottomHatConditions(s_grid);
        RadiationTopHatConditions(s_grid);
    } else {
        if (s_grid->neighbor_left_) {
            DefaultBottomHatConditions(s_grid);
        } else {
            RadiationBottomHatConditions(s_grid);
        }
        if (s_grid->neighbor_right_) {
            DefaultTopHatConditions(s_grid);
        } else {
            RadiationTopHatConditions(s_grid);
        }
        RadiationLeftHatConditions(s_grid);
        RadiationRightHatConditions(s_grid);
    }
}

// Flather characteristic open BC on all domain edges (index 5).
// Inter-subgrid ghosts use Default*; domain edges use Flather*.
void BoundaryCondition::ExchangeBoundaryDataFlather(std::shared_ptr<SubGrid> s_grid, double /*t*/)
{
    if (s_grid->GetDecompDir() == DecompositionDirection::Horizontal) {
        if (s_grid->neighbor_left_) {
            DefaultLeftConditions(s_grid);
        } else {
            FlatherLeftConditions(s_grid);
        }
        if (s_grid->neighbor_right_) {
            DefaultRightConditions(s_grid);
        } else {
            FlatherRightConditions(s_grid);
        }
        FlatherBottomConditions(s_grid);
        FlatherTopConditions(s_grid);
    } else {
        if (s_grid->neighbor_left_) {
            DefaultBottomConditions(s_grid);
        } else {
            FlatherBottomConditions(s_grid);
        }
        if (s_grid->neighbor_right_) {
            DefaultTopConditions(s_grid);
        } else {
            FlatherTopConditions(s_grid);
        }
        FlatherLeftConditions(s_grid);
        FlatherRightConditions(s_grid);
    }
}

void BoundaryCondition::ExchangeHatBoundaryDataFlather(std::shared_ptr<SubGrid> s_grid, double /*t*/)
{
    if (s_grid->GetDecompDir() == DecompositionDirection::Horizontal) {
        if (s_grid->neighbor_left_) {
            DefaultLeftHatConditions(s_grid);
        } else {
            FlatherLeftHatConditions(s_grid);
        }
        if (s_grid->neighbor_right_) {
            DefaultRightHatConditions(s_grid);
        } else {
            FlatherRightHatConditions(s_grid);
        }
        FlatherBottomHatConditions(s_grid);
        FlatherTopHatConditions(s_grid);
    } else {
        if (s_grid->neighbor_left_) {
            DefaultBottomHatConditions(s_grid);
        } else {
            FlatherBottomHatConditions(s_grid);
        }
        if (s_grid->neighbor_right_) {
            DefaultTopHatConditions(s_grid);
        } else {
            FlatherTopHatConditions(s_grid);
        }
        FlatherLeftHatConditions(s_grid);
        FlatherRightHatConditions(s_grid);
    }
}

// ===== Phase 8 NTHMP Monai: wavemaker boundary =============================
// Left boundary (i=0) is forced to follow a loaded η time series. Right and
// top/bottom remain zero-gradient walls (Default*). Wavemaker derives u from
// shallow-water characteristic so the forced surface elevation generates an
// inward-propagating wave: u_inflow = sqrt(g/D) * eta_forced.

void BoundaryCondition::WavemakerLeftConditions(std::shared_ptr<SubGrid> s, double t)
{
    if (!WavemakerData::IsLoaded()) {
        // Fallback: zero-gradient (wall) — should not happen in production.
        DefaultLeftConditions(s);
        return;
    }
    const double phys_t = t * s->tau_;
    const double eta_f = WavemakerData::GetEtaAt(phys_t);
    // Codex Phase 8: align dry threshold with solver, use linear-wave c = sqrt(g*D)
    // (still-water depth), not sqrt(g*H). Avoids nonlinear amplification at boundary.
    const double dry = std::max(0.0, s->min_calc_value);
    for (int j = 0; j < s->local_ny_; ++j) {
        // Ghost row D(0,j) is NOT initialized (exchange copies H/u/v/eta but
        // not D). Use the first OWNED cell D(1,j) as the wet/dry test and
        // physical depth for the wavemaker characteristic.
        const double D_owned = s->local_grid_.D(1, j);
        if (D_owned <= dry) {
            // Land at left edge of owned: zero-gradient (effective reflective wall).
            s->local_grid_.H(0, j) = s->local_grid_.H(1, j);
            s->local_grid_.u(0, j) = s->local_grid_.u(1, j);
            s->local_grid_.v(0, j) = s->local_grid_.v(1, j);
            s->local_grid_.eta(0, j) = s->local_grid_.eta(1, j);
            continue;
        }
        const double H = std::max(D_owned + eta_f, dry);
        const double c = std::sqrt(s->g_ * D_owned);
        s->local_grid_.eta(0, j) = static_cast<Real>(eta_f);
        s->local_grid_.H(0, j)   = static_cast<Real>(H);
        s->local_grid_.u(0, j)   = static_cast<Real>(c * eta_f / D_owned);
        s->local_grid_.v(0, j)   = s->local_grid_.v(1, j);
    }
}

void BoundaryCondition::WavemakerLeftHatConditions(std::shared_ptr<SubGrid> s, double t)
{
    if (!WavemakerData::IsLoaded()) {
        DefaultLeftHatConditions(s);
        return;
    }
    // Hat (half-step) uses the same forced eta — wavemaker is a Dirichlet on η,
    // velocity from shallow-water characteristic. Half-step buffer is updated
    // at the same physical time as full-step (Run() passes the same t).
    const double phys_t = t * s->tau_;
    const double eta_f = WavemakerData::GetEtaAt(phys_t);
    // Codex Phase 8: same conventions as WavemakerLeftConditions.
    const double dry = std::max(0.0, s->min_calc_value);
    const int ny = s->local_ny_;
    for (int j = 0; j < ny; ++j) {
        const double D_owned = s->local_grid_.D(1, j);
        if (D_owned <= dry) {
            s->H_hat_[j]   = s->H_hat_[ny + j];
            s->u_hat_[j]   = s->u_hat_[ny + j];
            s->v_hat_[j]   = s->v_hat_[ny + j];
            s->eta_hat_[j] = s->eta_hat_[ny + j];
            continue;
        }
        const double H = std::max(D_owned + eta_f, dry);
        const double c = std::sqrt(s->g_ * D_owned);
        s->eta_hat_[j] = static_cast<Real>(eta_f);
        s->H_hat_[j]   = static_cast<Real>(H);
        s->u_hat_[j]   = static_cast<Real>(c * eta_f / D_owned);
        s->v_hat_[j]   = s->v_hat_[ny + j];
    }
}

// Env-controlled switch: MONAI_REFLECTIVE_WALLS=1 → use Default (zero-gradient)
// walls instead of Sommerfeld radiation on right/top/bottom. NTHMP spec for
// Monai Valley = reflective vertical sidewalls.
static bool monai_reflective_walls() {
    static bool enabled = [](){
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "MONAI_REFLECTIVE_WALLS");
        const bool v = (buf && *buf == '1');
        if (buf) free(buf);
        if (v) std::cerr << "[Wavemaker] MONAI_REFLECTIVE_WALLS=1 (zero-gradient walls)"
                         << std::endl;
        return v;
    }();
    return enabled;
}

void BoundaryCondition::ExchangeBoundaryDataWavemaker(std::shared_ptr<SubGrid> s_grid, double t)
{
    const bool refl = monai_reflective_walls();
    if (s_grid->GetDecompDir() == DecompositionDirection::Horizontal) {
        if (s_grid->neighbor_left_) {
            DefaultLeftConditions(s_grid);
        } else {
            WavemakerLeftConditions(s_grid, t);
        }
        if (s_grid->neighbor_right_) {
            DefaultRightConditions(s_grid);
        } else if (refl) {
            DefaultRightConditions(s_grid);
        } else {
            RadiationRightConditions(s_grid);
        }
        if (refl) { DefaultBottomConditions(s_grid); DefaultTopConditions(s_grid); }
        else      { RadiationBottomConditions(s_grid); RadiationTopConditions(s_grid); }
    } else {
        DefaultBottomConditions(s_grid);
        DefaultTopConditions(s_grid);
        WavemakerLeftConditions(s_grid, t);
        if (refl) DefaultRightConditions(s_grid);
        else      RadiationRightConditions(s_grid);
    }
}

void BoundaryCondition::ExchangeHatBoundaryDataWavemaker(std::shared_ptr<SubGrid> s_grid, double t)
{
    const bool refl = monai_reflective_walls();
    if (s_grid->GetDecompDir() == DecompositionDirection::Horizontal) {
        if (s_grid->neighbor_left_) {
            DefaultLeftHatConditions(s_grid);
        } else {
            WavemakerLeftHatConditions(s_grid, t);
        }
        if (s_grid->neighbor_right_) {
            DefaultRightHatConditions(s_grid);
        } else if (refl) {
            DefaultRightHatConditions(s_grid);
        } else {
            RadiationRightHatConditions(s_grid);
        }
        if (refl) { DefaultBottomHatConditions(s_grid); DefaultTopHatConditions(s_grid); }
        else      { RadiationBottomHatConditions(s_grid); RadiationTopHatConditions(s_grid); }
    } else {
        DefaultBottomHatConditions(s_grid);
        DefaultTopHatConditions(s_grid);
        WavemakerLeftHatConditions(s_grid, t);
        if (refl) DefaultRightHatConditions(s_grid);
        else      RadiationRightHatConditions(s_grid);
    }
}
