#pragma once
// PartionFileGridSimulator.h

#ifndef SIMULATION_H
#define SIMULATION_H

#include "Grid.h"
#include <vector>
#include <thread>
#include <functional> // Для std::ref
#include <barrier>    // Требуется C++20
#include <memory>     // Для smart pointers
#include <string_view>

#include "BaseSimulator.h"
#include "SubGrid.h"

class PartionFileGridSimulator : BaseSimulator
{
public:
    // Конструктор по умолчанию
    PartionFileGridSimulator(bool use_file_storage);

    // Метод для установки параметров модели
    void SetParameters(double g, double tau, double delta_x, double delta_y,
        int nx, int ny, int timesteps);

    // Метод для установки параметров тестового сценария 
    /**
     * \brief Проверка работоспособности реализованного алгоритма:
     * Массив высот волны в центре сетки задаётся прямоугольной волной размером 10 на 10 узлов высотой 1м в центре сетки
     */
    void SetParamsTest1();

    /**
     * \brief Проверка работоспособности реализованного алгоритма:
     * Массив высот волны в центре сетки задаётся следующей формулой: 0.5 * (1 + cos(M_PI * dist_to_node / radius))
     */
    void SetParamsTest2();

    /**
     * \brief Проверка работоспособности реализованного алгоритма:
     * Глубина дна увеличивается квадратично от 0 до k * pow(delta_y_ * (ny_ - j), 2) + 10 и волна задаётся следующим образом: 1.0 / 2.0 * (1.0 - cos(n * tau * 2 * M_PI / T))
     */
    void SetParamsTest3();

    // Метод инициализации симуляции
    void Initialize() override;

    // Метод запуска симуляции
    void Run();

    // Метод сбора результатов
    void CollectResults(int timestep) override;

    void SetSaveInterval(int interval) override;

    int GetTimesteps() const override;
private:
    int save_interval_; // Интервал сохранения результатов
    std::vector<std::shared_ptr<SubGrid>> subgrids_;
    // Функция, выполняемая каждым потоком
    void ThreadFunction(int process_id);
    void SaveDataToFile(const std::string& variable_name, const Grid& grid, int timestep) override;
    // Параметры модели
    double g_;         // Ускорение свободного падения (м / c^2)
    double tau_;       // Временной шаг (с)
    double delta_x_;   // Шаг по x (м)
    double delta_y_;   // Шаг по y (м)
    int nx_;           // Размер сетки по x (ед.)
    int ny_;           // Размер сетки по y (ед.)
    int timesteps_;    // Количество временных шагов
    int num_threads_;  // Количество потоков (ед.)
	bool use_file_storage_; // Флаг использования файлового хранилища
    Grid global_grid_; // Глобальная сетка модели

    // Потоки и барьеры
    std::vector<std::thread> threads_;                   // Вектор потоков
    std::shared_ptr<std::barrier<>> barrier_half_step_;  // Барьер половинного шага
    std::shared_ptr<std::barrier<>> barrier_full_step_;  // Барьер полного шага
    std::shared_ptr<std::barrier<>> barrier_output_; // Барьер для синхронизации перед сохранением данных

    std::function<void(std::shared_ptr<SubGrid>, double)> ExchangeHatBoundaryData;
    std::function<void(std::shared_ptr<SubGrid>, double)> ExchangeBoundaryData;
    // Дополнительные данные, необходимые для подрешеток, могут быть определены здесь
    // Например, вектор подрешеток, но поскольку мы не реализуем SubGrid, этот аспект опустим

    void GenerateAndSaveGrid();
    void GenerateAndSaveArray(const std::string& filename, double* arr);
    void SaveSubGridArray(const std::string& filename, const double* arr, int start_x, int end_x);
    void LoadSubGridArray(const std::string& filename, double* arr, int start_x, int end_x);
    void LoadSubGridFromFile(int start_x, int end_x, Grid& subgrid);
    void SaveSubGridToFile(int start_x, int end_x, const Grid& subgrid);
};

#endif // SIMULATION_H