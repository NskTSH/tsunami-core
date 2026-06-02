#pragma once
#include <string>
#include "Grid.h"


class IBaseSimulator
{
public:
	virtual ~IBaseSimulator() = default;

	// Метод инициализации симуляции
	virtual void Initialize() = 0;

	// Метод запуска симуляции
	virtual void Run() = 0;

	// Метод сбора результатов
	virtual void CollectResults(int timestep) = 0;

	virtual void SetSaveInterval(int interval) = 0;
	virtual int GetTimesteps() const = 0;

	// Функция, выполняемая каждым потоком //TODO: Вынести в отдельный класс
	virtual void SaveDataToFile(const std::string& variable_name, const Grid& grid, int timestep) = 0;

protected:
	int save_interval_; // Интервал сохранения результатов
	double g_;         // Ускорение свободного падения (м / c^2)


	Grid global_grid_; // Глобальная сетка модели
};
