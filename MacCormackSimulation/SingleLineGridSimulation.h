#pragma once
// SingleLineGridSimulation.h

#ifndef SIMULATION_H
#define SIMULATION_H

#include "Grid.h"
#include <vector>
#include <thread>
#include <functional> // Для std::ref
#include <barrier>    // Требуется C++20
#include <memory>     // Для smart pointers
#include <string_view>

#include "ActiveRegionManager.h"
#include "ASCParser.h"
#include "BaseSimulator.h"
#include "CoordSystem.h"
#include "DefaultParametersParser.h"
#include "IBaseBathymetryParser.h"
#include "IBaseParametersParser.h"
#include "SubGrid.h"
#include "ZDataParser.h"

class SingleLineGridSimulation : public IBaseSimulator
{
public:
    // Конструктор по умолчанию
    SingleLineGridSimulation();

    // Метод для установки параметров модели
    void SetParameters(double g, double tau, std::function<double(int, int)> delta_x, double delta_y,
        int nx, int ny, int timesteps, int num_threads, int num_subgrids,
        int exchangeBoundaryIndex, int exchangeHatBoundaryIndex,
        CoordSystem coord_system, double center_latitude_rad = 0.785398163,
        double min_depth_to_calc = 8.0);

    void SetParameters(DefaultParametersParser* parameters);

    //void SetParams(IBaseParametersParser* parameters);
	// 1) Просто сохранить параметры (без сценариев)
    void ApplyParameters(
        double g,
        double tau,
        std::function<double(int, int)> delta_x,
        double delta_y,
        int nx,
        int ny,
        int timesteps,
        int num_threads,
		int num_subgrids,
        std::function<void(std::shared_ptr<SubGrid>, double)> exchangeBoundary,
        std::function<void(std::shared_ptr<SubGrid>, double)> exchangeHatBoundary,
        CoordSystem coord_system
    );

    // Установка системы координат отдельно
    void SetCoordSystem(CoordSystem coord_system);

    // Установка направления разбивки
    void SetDecompDirection(DecompositionDirection dir) { decomp_dir_ = dir; }

    // Установка папки для результатов (по умолчанию "Result")
    void SetResultFolder(const std::string& folder) { result_folder_ = folder; }

    // Sponge layer на границах (Codex plan 2026-04-10).
    // Можно включить независимо для каждой стороны.
    void SetSpongeLayer(int west_cells, int east_cells,
                        int south_cells, int north_cells,
                        double sigma_max) {
        sponge_west_cells_ = west_cells;
        sponge_east_cells_ = east_cells;
        sponge_south_cells_ = south_cells;
        sponge_north_cells_ = north_cells;
        sponge_sigma_max_ = sigma_max;
    }

    // Включение/выключение Кориолиса на Spherical-сетке (для "с/без Кориолиса"
    // сравнений на идентичной геометрии).
    void SetCoriolisEnabled(bool enabled) { coriolis_enabled_ = enabled; }
    bool GetCoriolisEnabled() const { return coriolis_enabled_; }

    // Сохранять ли u, v в снапшотах (по тому же save_interval, что и η).
    void SetSaveVelocities(bool enabled) { save_velocities_ = enabled; }

    // Окно для interior-метрики max_eta (исключая sponge). -1 = вся сетка.
    void SetInteriorWindow(int i_min, int i_max, int j_min, int j_max) {
        interior_i_min_ = i_min;
        interior_i_max_ = i_max;
        interior_j_min_ = j_min;
        interior_j_max_ = j_max;
    }

    // 2) Передача уже подготовленной глобальной сетки
    void SetGrid(Grid&& grid);

    // Метод инициализации симуляции
    void Initialize() override;

    void Initialize(ZDataParser& parser);
    void Initialize(ASCParser& parser, DefaultParametersParser& simulator_params);

    // Метод запуска симуляции
    void Run();

    // Метод сбора результатов
    void CollectResults(int timestep) override;

    void SetSaveInterval(int interval) override;

    int GetTimesteps() const override;

    // Геттер для системы координат
    CoordSystem GetCoordSystem() const;

    Grid global_grid_; // Глобальная сетка модели

    void SaveDataToFile(const std::string& variable_name, const Grid& grid, int timestep) override;

private:
    int save_interval_; // Интервал сохранения результатов
    int step_offset_ = 0; // Checkpoint resume: snapshot file step = (t+1) + step_offset_
    std::vector<std::shared_ptr<SubGrid>> subgrids_;

    // Active region tracking (опционально). См. Docs/Plans/Active_Region_Tracking_Plan.md
    // При Mode::Disabled (default из env) — манежер либо null, либо ничего не делает;
    // существующий Run() работает бит-в-бит как прежде.
    std::unique_ptr<active_region::ActiveRegionManager> active_region_;

    // Helper: создать active_region_ если env требует, вызвать из ВСЕХ Initialize().
    // Идемпотентен (если уже создан — не пересоздаёт). Вызывать ПОСЛЕ SetNeighbors().
    // parser_or_null: если scenario получает sources из parser, передать его
    // для построения source bbox; иначе nullptr — manager fall-back на grid scan.
    void InitializeActiveRegionManagerIfRequested(
        const IBaseParametersParser* parser_or_null = nullptr);

    // Функция, выполняемая каждым потоком
    void ThreadFunction(int process_id);

    // Параметры модели
    double g_;         // Ускорение свободного падения (м / c^2)
    double tau_;       // Временной шаг (с)
    std::function<double(int, int)> delta_x_;   // Шаг по x (м)
    double delta_y_;   // Шаг по y (м)
    int nx_;           // Размер сетки по x (ед.)
    int ny_;           // Размер сетки по y (ед.)
    int timesteps_;    // Количество временных шагов
	int num_threads_;  // Количество OpenMP потоков (ед.)
	int num_subgrids_; // Количество подсеток/полос (ед.)
    CoordSystem coord_system_; // Система координат (обязательно задаётся явно)
    double center_latitude_rad_; // Центральная широта (радианы) для сферической системы
    DecompositionDirection decomp_dir_ = DecompositionDirection::Horizontal;
    double minimum_depth_to_calculate = 8.0;
    std::string result_folder_ = "Result";

    // Sponge layer на границах домена
    int sponge_west_cells_ = 0;
    int sponge_east_cells_ = 0;
    int sponge_south_cells_ = 0;
    int sponge_north_cells_ = 0;
    double sponge_sigma_max_ = 0.0;
    // Interior window для диагностики
    int interior_i_min_ = -1;
    int interior_i_max_ = -1;
    int interior_j_min_ = -1;
    int interior_j_max_ = -1;
    // Coriolis (работает только при coord_system_ == Spherical). default=true.
    bool coriolis_enabled_ = true;
    // Сохранять ли u, v вместе с η в снапшотах (для Coriolis-валидации).
    bool save_velocities_ = false;

    // Применить sponge damping ко всем реальным ячейкам в sponge-зонах.
    // Вызывается после ExchangeBoundaryData (на шаге t).
    void ApplySpongeLayer();

    // Потоки и барьеры
    std::vector<std::thread> threads_;                   // Вектор потоков
    std::shared_ptr<std::barrier<>> barrier_half_step_;  // Барьер половинного шага
    std::shared_ptr<std::barrier<>> barrier_full_step_;  // Барьер полного шага
    std::shared_ptr<std::barrier<>> barrier_output_;     // Барьер для синхронизации перед сохранением данных

    std::vector< std::function<void(std::shared_ptr<SubGrid>, double)>> ExchangeHatBoundaryDataFunctions;
    std::vector< std::function<void(std::shared_ptr<SubGrid>, double)>> ExchangeBoundaryDataFunctions;
    std::function<void(std::shared_ptr<SubGrid>, double)> ExchangeHatBoundaryData;
    std::function<void(std::shared_ptr<SubGrid>, double)> ExchangeBoundaryData;

    // Запись/чтение полос из подрешёток (поддержка бинарного вывода)
    void SaveSubGridArray(const std::string& filename, const double* arr, int ny, int nx);
    void LoadSubGridArray(const std::string& filename, double* arr, int ny, int nx);
};

#endif // SIMULATION_H
