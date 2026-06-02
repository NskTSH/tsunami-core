// Grid.h - Оптимизированная версия с сохранением совместимости
// Добавлено: move-семантика, предрассчёт inv_dx/inv_dy, прямой доступ к массивам

#ifndef GRID_H
#define GRID_H

#include <functional>
#include <memory>
#include <vector>
#include <algorithm>

#include "Precision.h"  // Real (float/double) — selected at build via USE_FLOAT_PRECISION
#include "ASCParser.h"
#include "DefaultParametersParser.h"
#include "IBaseBathymetryParser.h"

class Grid {
public:
    int subgrid_count = -1;

    // Конструктор с заданием размеров сетки (совместимость с оригиналом)
    Grid(int nx = 0, int ny = 0, 
         std::function<double(int, int)> delta_x = [](int, int) { return 0.0; }, 
         double delta_y = 0.0);

    // ========== MOVE СЕМАНТИКА ==========
    Grid(Grid&& other) noexcept;
    Grid& operator=(Grid&& other) noexcept;

    // Копирование (для совместимости)
    Grid(const Grid& other);
    Grid& operator=(const Grid& other);

    // Метод инициализации сетки
    void Initialize();

    // ========== ДОСТУП К ЭЛЕМЕНТАМ (оригинальная индексация i * ny + j) ==========
    Real& H(int i, int j);
    Real& eta_max(int i, int j);
    Real& u_max(int i, int j);
    Real& v_max(int i, int j);
    Real& u(int i, int j);
    Real& v(int i, int j);
    Real& eta(int i, int j);
    Real& D(int i, int j);

    const Real& H(int i, int j) const;
    const Real& eta_max(int i, int j) const;
    const Real& u_max(int i, int j) const;
    const Real& v_max(int i, int j) const;
    const Real& u(int i, int j) const;
    const Real& v(int i, int j) const;
    const Real& eta(int i, int j) const;
    const Real& D(int i, int j) const;

    int GetNx() const;
    int GetNy() const;

    // ========== ПРЯМОЙ ДОСТУП К МАССИВАМ ==========
    Real* GetH() { return H_.get(); }
    Real* GetU() { return u_.get(); }
    Real* GetV() const { return v_.get(); }
    Real* GetEta() { return eta_.get(); }
    Real* GetD() { return D_.get(); }
    Real* GetEtaMax() { return eta_max_.get(); }
    Real* GetUMax() { return u_max_.get(); }
    Real* GetVMax() { return v_max_.get(); }

    const Real* GetH() const { return H_.get(); }
    const Real* GetU() const { return u_.get(); }
    const Real* GetEta() const { return eta_.get(); }
    const Real* GetD() const { return D_.get(); }
    const Real* GetEtaMax() const { return eta_max_.get(); }
    const Real* GetUMax() const { return u_max_.get(); }
    const Real* GetVMax() const { return v_max_.get(); }

    // ========== ПРЕДРАССЧИТАННЫЕ ЗНАЧЕНИЯ ==========
    double GetDeltaX() const { return delta_x_cached_; }
    double GetDeltaY() const { return delta_y_; }
    double GetInvDx() const { return inv_dx_; }
    double GetInvDy() const { return inv_dy_; }

    // Доступ к функции delta_x (для совместимости)
    std::function<double(int, int)>& GetDeltaXFunc() { return delta_x_; }
    const std::function<double(int, int)>& GetDeltaXFunc() const { return delta_x_; }

    // ========== УСТАНОВКА НАЧАЛЬНЫХ УСЛОВИЙ ==========
    void SetParamsTest1();
    void SetParamsTest2();
    void SetParamsTest3(double tau);
    void SetParamsTest4(double tau);
    void SetParamsTest5(double tau);
    void SetParamsTest6(double tau, ASCParser& parser, DefaultParametersParser& simulator_params);
    void SetParameters(double tau, IBaseBathymetryParser& parser);

    /// Coriolis test: flat bottom D=depth, plane wave front at j=[0, wave_width)
    void SetParamsCoriolis(double depth, int wave_width, double wave_amplitude);

    /// Coriolis test 2: flat bottom D=depth, plane wave front at i=[0, wave_width) (west→east)
    void SetParamsCoriolisHorizontal(double depth, int wave_width, double wave_amplitude);

    /// Coriolis test with ellipsoidal source: flat bottom, cosine-tapered ellipsoidal perturbation at center
    void SetParamsCoriolisEllipsoidal(double depth, double r1_cells, double r2_cells,
        double angle_deg, double amplitude);

    /// Same as above but with explicit source center in grid cells (cx, cy)
    void SetParamsCoriolisEllipsoidal(double depth, double r1_cells, double r2_cells,
        double angle_deg, double amplitude, double cx_cells, double cy_cells);

    /// Plane wave IC with finite horizontal extent.
    /// Fills eta=H=depth+amplitude in the cells j in [wave_j_min, wave_j_max), i in [wave_i_min, wave_i_max).
    /// Also sets u=u0, v=v0 inside the wave box (one-way traveling plane wave when v0 = +c*eta/D).
    /// Rest of domain: flat bottom, eta=0, u=v=0, H=depth.
    void SetParamsCoriolisFinitePlane(double depth,
        int wave_j_min, int wave_j_max,
        int wave_i_min, int wave_i_max,
        double wave_amplitude,
        double u0 = 0.0,
        double v0 = 0.0);
    
    bool IsInitialized() const { return is_initialized_; }

    /// Mark the grid as initialized when bathymetry / fields were filled
    /// externally (e.g. in a ScenarioBuilder that doesn't go through any
    /// `SetParams*` helper).
    void MarkInitialized() { is_initialized_ = true; }

private:
    int nx_;
    int ny_;
    std::function<double(int, int)> delta_x_;
    double delta_y_;
    
    // Предрассчитанные значения для оптимизации
    double delta_x_cached_;  // delta_x_(0, 0)
    double inv_dx_;          // 1.0 / delta_x_(0, 0)
    double inv_dy_;          // 1.0 / delta_y_

    // Матрицы для хранения физических полей
    std::unique_ptr<Real[]> H_;
    std::unique_ptr<Real[]> u_;
    std::unique_ptr<Real[]> v_;
    std::unique_ptr<Real[]> eta_;
    std::unique_ptr<Real[]> D_;
    std::unique_ptr<Real[]> eta_max_;
    std::unique_ptr<Real[]> u_max_;
    std::unique_ptr<Real[]> v_max_;

    bool is_initialized_ = false;

    void AllocateArrays();
    void PrecomputeConstants();
};

#endif // GRID_H
