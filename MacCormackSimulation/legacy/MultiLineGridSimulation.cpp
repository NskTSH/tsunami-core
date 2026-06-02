// MultiLineGridSimulation.cpp

#include "MultiLineGridSimulation.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


MultiLineGridSimulation::MultiLineGridSimulation()
	: g_(9.81), tau_(1.0), delta_x_([](int, int) { return 1.0; }), delta_y_(1.0),
	nx_(100), ny_(100), timesteps_(100), num_threads_(4),
	coord_system_(CoordSystem::Cartesian), // Значение по умолчанию только в конструкторе
	global_grid_(1, 1),
	save_interval_(10)  // Инициализируем с минимальным размером, переинициализируем позже
{
	// По умолчанию параметры могут быть переустановлены позже
}

void MultiLineGridSimulation::SetParameters(double g, double tau, std::function<double(int, int)> delta_x, double delta_y,
	int nx, int ny, int timesteps, int num_threads, CoordSystem coord_system)
{
	g_ = g;
	tau_ = tau;
	delta_x_ = delta_x;
	delta_y_ = delta_y;
	nx_ = nx;
	ny_ = ny;
	timesteps_ = timesteps;
	num_threads_ = num_threads;
	coord_system_ = coord_system;

	// Переинициализируем глобальную сетку с новыми размерами
	global_grid_ = Grid(nx_, ny_);
}

void MultiLineGridSimulation::SetCoordSystem(CoordSystem coord_system) {
	coord_system_ = coord_system;
}

CoordSystem MultiLineGridSimulation::GetCoordSystem() const {
	return coord_system_;
}

void MultiLineGridSimulation::SetParamsTest1() {
	// Устанавливаем параметры согласно вашему сценарию
	g_ = 9.81;
	tau_ = 5.0;
	delta_x_ = [](int, int) { return 1000.0; };
	delta_y_ = 1000.0;
	nx_ = 120;
	ny_ = 120;
	timesteps_ = 500;
	num_threads_ = 8;
	num_columns_ = 1;
	num_rows_ = 2;
	coord_system_ = CoordSystem::Cartesian; // Явная установка системы координат

	// Переинициализируем глобальную сетку с новыми размерами
	global_grid_ = Grid(nx_, ny_, delta_x_, delta_y_);
	global_grid_.SetParamsTest1();

	ExchangeBoundaryData = [](std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
		FourNeighborSubGrid::ExchangeBoundaryData1(s_grid, t);
		};
	ExchangeHatBoundaryData = [](std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
		FourNeighborSubGrid::ExchangeHatBoundaryData1(s_grid, t);
		};
}

void MultiLineGridSimulation::SetParamsTest2() {
	// Устанавливаем параметры согласно вашему сценарию
	g_ = 9.81;
	tau_ = 5.0;
	delta_x_= [](int, int) { return 1000.0; };
	delta_y_ = 1000.0;
	nx_ = 1000;
	ny_ = 1000;
	timesteps_ = 4500;
	num_threads_ = 8;
	num_columns_ = 4;
	num_rows_ = 2;
	coord_system_ = CoordSystem::Cartesian; // Явная установка системы координат

	// Переинициализируем глобальную сетку с новыми размерами
	global_grid_ = Grid(nx_, ny_, delta_x_, delta_y_);
	global_grid_.SetParamsTest2();

	ExchangeBoundaryData = [](std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
		FourNeighborSubGrid::ExchangeBoundaryData2(s_grid, t);
		};
	ExchangeHatBoundaryData = [](std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
		FourNeighborSubGrid::ExchangeHatBoundaryData2(s_grid, t);
		};
}

void MultiLineGridSimulation::SetParamsTest3() {
	// Устанавливаем параметры согласно вашему сценарию
	g_ = 9.81;
	tau_ = 2;
	delta_x_ = [](int, int) { return 1000.0; };
	delta_y_ = 1000.0;
	nx_ = 1000;
	ny_ = 1000;
	timesteps_ = 8000;
	num_threads_ = 8;
	num_columns_ = 4;
	num_rows_ = 2;
	coord_system_ = CoordSystem::Cartesian; // Явная установка системы координат

	// Переинициализируем глобальную сетку с новыми размерами
	global_grid_ = Grid(nx_, ny_, delta_x_, delta_y_);
	global_grid_.SetParamsTest3(tau_);

	ExchangeBoundaryData = [](std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
		FourNeighborSubGrid::ExchangeBoundaryData3(s_grid, t);
		};
	ExchangeHatBoundaryData = [](std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
		FourNeighborSubGrid::ExchangeHatBoundaryData3(s_grid, t);
		};
}

void MultiLineGridSimulation::SetSaveInterval(int interval) {
	save_interval_ = interval;
}

void MultiLineGridSimulation::Initialize() {
	// Проверяем, инициализирована ли глобальная сетка
	if (!global_grid_.IsInitialized()) {
		throw std::exception("Global grid is not initialized!");
	}

	// Рассчитываем размеры подрешёток
	int m_x = nx_ / num_columns_; // Размер подрешётки по x
	int m_y = ny_ / num_rows_;    // Размер подрешётки по y

	// Инициализируем барьеры синхронизации
	int total_threads = num_rows_ * num_columns_; // Всего потоков по количеству подсеток
	barrier_half_step_ = std::make_shared<std::barrier<>>(total_threads);
	barrier_full_step_ = std::make_shared<std::barrier<>>(total_threads);
	barrier_output_ = std::make_shared<std::barrier<>>(total_threads + 1); // +1 для основного потока

	// Резервируем место для подсеток и потоков
	subgrids_.resize(num_rows_);
	threads_.resize(num_rows_);
	for (int i = 0; i < num_rows_; ++i) {
		subgrids_[i].resize(num_columns_);
		threads_[i].resize(num_columns_);
	}

	// Создаём подрешётки
	for (int row = 0; row < num_rows_; ++row) {
		for (int col = 0; col < num_columns_; ++col) {
			int global_x_start = col * m_x;
			int global_x_end = (col == num_columns_ - 1) ? nx_ : (col + 1) * m_x;
			int global_y_start = row * m_y;
			int global_y_end = (row == num_rows_ - 1) ? ny_ : (row + 1) * m_y;

			// Создаём подрешётку
			subgrids_[row][col] = std::make_unique<FourNeighborSubGrid>(
				row * num_columns_ + col, // process_id
				global_x_start,
				global_x_end,
				global_y_start,
				global_y_end,
				global_grid_,
				g_, tau_, delta_x_, delta_y_,
				barrier_half_step_,
				barrier_full_step_,
				coord_system_
			);

			// Устанавливаем функции обмена данными
			subgrids_[row][col]->ExchangeBoundaryData = ExchangeBoundaryData;
			subgrids_[row][col]->ExchangeHatBoundaryData = ExchangeHatBoundaryData;

			// Инициализируем подрешётку
			subgrids_[row][col]->Initialize();
		}
	}

	// Устанавливаем соседей для подрешёток
	for (int row = 0; row < num_rows_; ++row) {
		for (int col = 0; col < num_columns_; ++col) {
			
			
			FourNeighborSubGrid* neighbor_up = (row > 0) ? subgrids_[row - 1][col].get() : nullptr;
			FourNeighborSubGrid* neighbor_down = (row < num_rows_ - 1) ? subgrids_[row + 1][col].get() : nullptr;
			FourNeighborSubGrid* neighbor_left  = (col > 0) ? subgrids_[row][col - 1].get() : nullptr;
			FourNeighborSubGrid* neighbor_right = (col < num_columns_ - 1) ? subgrids_[row][col + 1].get() : nullptr;

			subgrids_[row][col]->SetNeighbors(neighbor_left, neighbor_right,neighbor_up, neighbor_down);
		}
	}

}

void MultiLineGridSimulation::Run() {
	// Запускаем потоки
	CollectResults(0);
	for (int row = 0; row < num_rows_; ++row) {
		for (int col = 0; col < num_columns_; ++col) {
			threads_[row][col] = std::thread(&MultiLineGridSimulation::ThreadFunction, this, row, col);
		}
	}

	for (int t = 0; t < timesteps_; ++t) {
		// Ожидаем, пока все потоки завершат текущий временной шаг
		barrier_output_->arrive_and_wait(); // Основной поток ожидает

		// Обновляем глобальную сетку перед сохранением данных
		int counter = 0;
		for (int row = 0; row < num_rows_; ++row) {
			for (int col = 0; col < num_columns_; ++col) {
				// TODO: There is some problems with local grid sizes!!!!
				subgrids_[row][col]->UpdateGlobalGrid(row, col, num_rows_, num_columns_);
			}
		}

		int percentage = static_cast<int>(100.0 * (t + 1) / timesteps_);

		// Выводим процент выполнения в консоль, перезаписывая предыдущий вывод
		std::cout << "\rSimulation percent: " << percentage << "%";
		std::cout.flush();

		// Проверяем, нужно ли сохранять результаты
		if ((t + 1) % save_interval_ == 0) {
			CollectResults(t + 1); // Сохраняем данные, передаем текущий временной шаг
		}

		// Разрешаем потокам продолжить работу
		barrier_output_->arrive_and_wait();
	}

	// Ожидаем завершения всех потоков
	for (int row = 0; row < num_rows_; ++row) {
		for (int col = 0; col < num_columns_; ++col) {
			if (threads_[row][col].joinable()) {
				threads_[row][col].join();
			}
		}
	}
}

void MultiLineGridSimulation::ThreadFunction(int row, int col) {
	auto& subgrid = subgrids_[row][col];

	for (int t = 0; t < timesteps_; ++t) {
		// Вычисление на половинном шаге
		subgrid->ComputeHalfStep();

		// Синхронизация потоков на половинном шаге
		barrier_half_step_->arrive_and_wait();

		// Обмен граничными данными (шапка)
		subgrid->ExchangeHatBoundaryData(subgrid, t);

		// Синхронизация перед полным шагом
		barrier_half_step_->arrive_and_wait();

		// Вычисление на полном шаге
		subgrid->ComputeFullStep();
		
		// Обмен граничными данными
		subgrid->ExchangeBoundaryData(subgrid, t);

		// Синхронизация потоков перед обменом граничными данными
		barrier_full_step_->arrive_and_wait();

		// Синхронизация потоков на полном шаге
		barrier_full_step_->arrive_and_wait();

		// Синхронизация перед сохранением данных
		barrier_output_->arrive_and_wait(); // Поток сообщает о завершении шага
		barrier_output_->arrive_and_wait(); // Ожидает разрешения от основного потока
	}

	// Обновление глобальной сетки после завершения вычислений
	subgrid->UpdateGlobalGrid(row, col, num_rows_, num_columns_);
}

void MultiLineGridSimulation::CollectResults(int timestep) {
	// Сохраняем массивы H, u, v, eta в файлы
	//SaveDataToFile("H", global_grid_, timestep);
	//SaveDataToFile("u", global_grid_, timestep);
	//SaveDataToFile("v", global_grid_, timestep);
	SaveDataToFile("eta", global_grid_, timestep);
}

void MultiLineGridSimulation::SaveDataToFile(const std::string& variable_name, const Grid& grid, int timestep) {
	// Создаем имя файла
	const std::string folder_name = "Result";
	const std::string filename = folder_name + "/" + variable_name + "_timestep_" + std::to_string(timestep) + ".txt";

	// Проверяем и создаем папку при необходимости
	std::filesystem::create_directories(folder_name);

	// Определяем функцию для доступа к данным
	std::function<double(int, int)> getValue;
	if (variable_name == "H") {
		getValue = [&grid](int i, int j) { return grid.H(i, j); };
	}
	else if (variable_name == "u") {
		getValue = [&grid](int i, int j) { return grid.u(i, j); };
	}
	else if (variable_name == "v") {
		getValue = [&grid](int i, int j) { return grid.v(i, j); };
	}
	else if (variable_name == "eta") {
		getValue = [&grid](int i, int j) { return grid.eta(i, j); };
	}
	else {
		std::cerr << "Error: Unknown variable name " << variable_name << "." << std::endl;
		return;
	}

	// Предварительное формирование данных для записи
	std::ostringstream output_buffer;
	int nx = grid.GetNx();
	int ny = grid.GetNy();
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < ny; ++j) {
			auto result = getValue(i, j);
			if(result == -100.0)
			{
				std::cout << "Hello"<<std::endl;
			}
			output_buffer << result << " ";
		}
		output_buffer << "\n";
	}

	// Записываем в файл одним вызовом
	std::ofstream file(filename);
	if (!file) {
		std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
		return;
	}
	file << output_buffer.str();
}

int MultiLineGridSimulation::GetTimesteps() const {
	return timesteps_;
}