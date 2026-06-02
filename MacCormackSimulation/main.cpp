// main.cpp - Пример использования оптимизированного симулятора
// 
// Компиляция (MSVC):
//   cl /EHsc /O2 /openmp /std:c++20 main.cpp Grid.cpp SubGrid.cpp BoundaryCondition.cpp SingleLineGridSimulation.cpp
//
// Компиляция (GCC/Clang):
//   g++ -O3 -fopenmp -std=c++20 main.cpp Grid.cpp SubGrid.cpp BoundaryCondition.cpp SingleLineGridSimulation.cpp -o simulation
//
// Для использования float вместо double:
//   #define USE_FLOAT_PRECISION

#include <iostream>
#include <chrono>
#include "SingleLineGridSimulation.h"

// Раскомментируйте для использования float
// #define USE_FLOAT_PRECISION

#ifdef USE_FLOAT_PRECISION
    using Real = float;
    #define PRECISION_NAME "float"
#else
    using Real = double;
    #define PRECISION_NAME "double"
#endif

int main(int argc, char* argv[]) {
    std::cout << "=== MacCormack Simulation (Optimized) ===" << std::endl;
    std::cout << "Precision: " << PRECISION_NAME << std::endl;
    std::cout << std::endl;

    // Параметры симуляции
    const int nx = 500;           // Размер сетки по X
    const int ny = 500;           // Размер сетки по Y
    const int timesteps = 1000;   // Количество временных шагов
    const int num_threads = 8;    // Количество OpenMP потоков
	const int num_subgrids = 8;   // Количество подсеток (полос)
    
    const Real g = Real(9.81);           // Ускорение свободного падения
    const Real tau = Real(0.1);          // Временной шаг
    const Real delta_x = Real(1000.0);   // Шаг по X (метры)
    const Real delta_y = Real(1000.0);   // Шаг по Y (метры)

    // Создание симулятора
    SingleLineGridSimulation<Real> simulation;

    // Установка параметров
    simulation.SetParameters(
        g,
        tau,
        delta_x,
        delta_y,
        nx,
        ny,
		timesteps,
		num_threads,
		num_subgrids,
        0,  // exchangeBoundaryIndex (0 = default)
        0   // exchangeHatBoundaryIndex (0 = default)
    );

    // Установка начальных условий (тест 2 - круговое возмущение)
    simulation.SetupTest2();

    // Инициализация
    simulation.Initialize();

    // Установка интервала сохранения (0 = не сохранять)
    simulation.SetSaveInterval(100);

    // Запуск симуляции с замером времени
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.Run();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Статистика
    std::cout << std::endl;
    std::cout << "=== Performance ===" << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "Time per step: " << duration.count() / timesteps << " ms" << std::endl;
    std::cout << "Grid cells: " << nx * ny << std::endl;
    std::cout << "Cells/second: " << (static_cast<double>(nx) * ny * timesteps) / (duration.count() / 1000.0) / 1e6 << " M" << std::endl;

    return 0;
}
