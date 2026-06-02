// FourNeighborSubGrid.h

#ifndef FOURNEIGHBORSUBGRID_H
#define FOURNEIGHBORSUBGRID_H

#include "Grid.h"
#include "CoordSystem.h"
#include <memory>  // Для std::shared_ptr
#include <barrier> // Требуется C++20
#include <functional>

// TODO: There is some problem with this class
class FourNeighborSubGrid {
public:
    // Конструктор (coord_system обязателен)
    FourNeighborSubGrid(
        int process_id, int global_i_start, int global_i_end,
        int global_j_start, int global_j_end, Grid& global_grid,
        double g, double tau, std::function<double(int, int)> delta_x, double delta_y,
        std::shared_ptr<std::barrier<>>& barrier_half_step,
        std::shared_ptr<std::barrier<>>& barrier_full_step,
        CoordSystem coord_system,
        double center_latitude_rad = 0.785398163);

    // Метод инициализации подрешётки
    void Initialize();

    // Диспетчерские методы - выбирают реализацию по coord_system_
    void ComputeHalfStep();
    void ComputeFullStep();
    
    // Реализации для декартовой системы координат
    void ComputeHalfStepCartesian();
    void ComputeFullStepCartesian();
    
    // Реализации для сферической системы координат (пока заглушки)
    void ComputeHalfStepSpherical();
    void ComputeFullStepSpherical();

    // Обмен граничными данными
    std::function<void(std::shared_ptr<FourNeighborSubGrid>, double)> ExchangeHatBoundaryData;
    std::function<void(std::shared_ptr<FourNeighborSubGrid>, double)> ExchangeBoundaryData;
    // Fixed
    static void ExchangeHatBoundaryData1(std::shared_ptr<FourNeighborSubGrid> s_grid, double t);
    // Fixed
    static void ExchangeBoundaryData1(std::shared_ptr<FourNeighborSubGrid> s_grid, double t);

    static void ExchangeHatBoundaryData2(std::shared_ptr<FourNeighborSubGrid> s_grid, double t);
    static void ExchangeBoundaryData2(std::shared_ptr<FourNeighborSubGrid> s_grid, double t);

    static void ExchangeHatBoundaryData3(std::shared_ptr<FourNeighborSubGrid> s_grid, double t);
    static void ExchangeBoundaryData3(std::shared_ptr<FourNeighborSubGrid> s_grid, double t);

    // Обновление глобальной сетки
    // Fixed
    void UpdateGlobalGrid(int row, int col, int num_rows_, int num_cols_);

    // Соседи
    // Fixed
    void SetNeighbors(FourNeighborSubGrid* neighbor_up, 
					  FourNeighborSubGrid* neighbor_down, 
					  FourNeighborSubGrid* neighbor_left, 
					  FourNeighborSubGrid* neighbor_right);

    // Публичные члены для доступа из других классов (например, BoundaryCondition)
    Grid local_grid_;         // Локальная подрешётка для вычислений
    int local_nx_;            // Размер подрешётки по x с учётом граничных слоёв
    int local_ny_;            // Размер подрешётки по y
    FourNeighborSubGrid* neighbor_up_;    // Верхний сосед
    FourNeighborSubGrid* neighbor_down_;  // Нижний сосед
    FourNeighborSubGrid* neighbor_left_;  // Левый сосед
    FourNeighborSubGrid* neighbor_right_;  // Правый сосед
    inline static int id = 0;
    int local_id = -1;

    // Геттер системы координат
    CoordSystem GetCoordSystem() const { return coord_system_; }

private:

    int process_id_;          // Идентификатор процесса (потока)
    int global_i_start_;      // Глобальный индекс начала подрешётки по x
    int global_i_end_;        // Глобальный индекс конца подрешётки по x
    int global_j_start_;      // Глобальный индекс начала подрешётки по x
    int global_j_end_;        // Глобальный индекс конца подрешётки по x

    Grid& global_grid_;       // Ссылка на глобальную сетку

    // Параметры модели
    double g_;
    double tau_;
    std::function<double(int, int)> delta_x_;
    double delta_y_;
    CoordSystem coord_system_; // Система координат (обязательно задаётся через конструктор)
    double center_latitude_rad_; // Центральная широта (радианы) для сферической системы

    // Барьеры для синхронизации
    std::shared_ptr<std::barrier<>> barrier_half_step_;
    std::shared_ptr<std::barrier<>> barrier_full_step_;

    // Локальные массивы для сохранения промежуточных результатов
    std::unique_ptr<double[]> H_hat_;
    std::unique_ptr<double[]> u_hat_;
    std::unique_ptr<double[]> v_hat_;
    std::unique_ptr<double[]> eta_hat_;

    // Геттеры и сеттеры для доступа к элементам сетки
    double& H_hat(int i, int j);
    double& u_hat(int i, int j);
    double& v_hat(int i, int j);
    double& eta_hat(int i, int j);

    // Константные версии
    const double& H_hat(int i, int j) const;
    const double& u_hat(int i, int j) const;
    const double& v_hat(int i, int j) const;
    const double& eta_hat(int i, int j) const;
};

#endif // FOURNEIGHBORSUBGRID_H