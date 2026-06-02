// FourNeighborSubGrid.cpp

#include "FourNeighborSubGrid.h"

#include <corecrt_math_defines.h>
#include <stdexcept>

#include "BoundaryCondition.h"

FourNeighborSubGrid::FourNeighborSubGrid(
	int process_id, int global_i_start, int global_i_end,
	int global_j_start, int global_j_end, Grid& global_grid,
	double g, double tau, std::function<double(int, int)> delta_x, double delta_y,
	std::shared_ptr<std::barrier<>>& barrier_half_step,
	std::shared_ptr<std::barrier<>>& barrier_full_step,
	CoordSystem coord_system,
	double center_latitude_rad)
	: process_id_(process_id),
	global_i_start_(global_i_start),
	global_i_end_(global_i_end),
	global_j_start_(global_j_start),
	global_j_end_(global_j_end),
	global_grid_(global_grid),
	g_(g),
	tau_(tau),
	delta_x_(delta_x),
	delta_y_(delta_y),
	coord_system_(coord_system),
	center_latitude_rad_(center_latitude_rad),
	neighbor_left_(nullptr),
	neighbor_right_(nullptr),
	neighbor_up_(nullptr),
	neighbor_down_(nullptr),
	barrier_half_step_(barrier_half_step),
	barrier_full_step_(barrier_full_step),
	local_grid_(0, 0) // Инициализируем local_grid_ позже
{
	// Устанавливаем размеры локальной подрешётки с учётом граничных слоёв
	local_nx_ = (global_i_end_ - global_i_start_) + 2; // Добавляем 2 для граничных слоёв
	local_ny_ = (global_j_end_ - global_j_start_) + 2;

	// Инициализируем локальную сетку
	local_grid_ = Grid(local_nx_, local_ny_);

	// Инициализируем локальные массивы для промежуточных результатов
	H_hat_ = std::make_unique<double[]>(local_nx_ * local_ny_);
	u_hat_ = std::make_unique<double[]>(local_nx_ * local_ny_);
	v_hat_ = std::make_unique<double[]>(local_nx_ * local_ny_);
	eta_hat_ = std::make_unique<double[]>(local_nx_ * local_ny_);
	local_id = id++;
}

void FourNeighborSubGrid::Initialize() {
	// Копируем данные из глобальной сетки в локальную подрешётку
	for (int i = 0; i < local_nx_; ++i) {
		int global_i = global_i_start_ - 1 + i; // Корректируем индекс для граничных слоёв
		for (int j = 0; j < local_ny_; ++j) {
			int global_j = global_j_start_ - 1 + j; // Для обработки вертикальных граничных слоёв
			if (global_i >= 0 && global_i < global_grid_.GetNx() &&
				global_j >= 0 && global_j < global_grid_.GetNy()) {
				// Если индексы внутри глобальной сетки, копируем данные
				local_grid_.H(i, j) = global_grid_.H(global_i, global_j);
				local_grid_.u(i, j) = global_grid_.u(global_i, global_j);
				local_grid_.v(i, j) = global_grid_.v(global_i, global_j);
				local_grid_.eta(i, j) = global_grid_.eta(global_i, global_j);
				local_grid_.D(i, j) = global_grid_.D(global_i, global_j);
			}
			else {
				// Инициализация граничных слоёв, если за пределами глобальной сетки
				if (global_i < 0) { // Верхняя граница
					local_grid_.H(i, j) = global_grid_.H(0, global_j);
					local_grid_.u(i, j) = global_grid_.u(0, global_j);
					local_grid_.v(i, j) = global_grid_.v(0, global_j);
					local_grid_.eta(i, j) = global_grid_.eta(0, global_j);
					local_grid_.D(i, j) = global_grid_.D(0, global_j);
				}
				else if (global_i >= global_grid_.GetNx()) { // Нижняя граница
					local_grid_.H(i, j) = global_grid_.H(global_grid_.GetNx() - 1, global_j);
					local_grid_.u(i, j) = global_grid_.u(global_grid_.GetNx() - 1, global_j);
					local_grid_.v(i, j) = global_grid_.v(global_grid_.GetNx() - 1, global_j);
					local_grid_.eta(i, j) = global_grid_.eta(global_grid_.GetNx() - 1, global_j);
					local_grid_.D(i, j) = global_grid_.D(global_grid_.GetNx() - 1, global_j);
				}
				else if (global_j < 0) { // Левая граница
					local_grid_.H(i, j) = global_grid_.H(global_i, 0);
					local_grid_.u(i, j) = global_grid_.u(global_i, 0);
					local_grid_.v(i, j) = global_grid_.v(global_i, 0);
					local_grid_.eta(i, j) = global_grid_.eta(global_i, 0);
					local_grid_.D(i, j) = global_grid_.D(global_i, 0);
				}
				else if (global_j >= global_grid_.GetNy()) { // Правая граница
					local_grid_.H(i, j) = global_grid_.H(global_i, global_grid_.GetNy() - 1);
					local_grid_.u(i, j) = global_grid_.u(global_i, global_grid_.GetNy() - 1);
					local_grid_.v(i, j) = global_grid_.v(global_i, global_grid_.GetNy() - 1);
					local_grid_.eta(i, j) = global_grid_.eta(global_i, global_grid_.GetNy() - 1);
					local_grid_.D(i, j) = global_grid_.D(global_i, global_grid_.GetNy() - 1);
				}
				else {
					// Если по каким-то причинам индексы не попадают в известный диапазон, инициализируем нулями
					local_grid_.H(i, j) = 0.0;
					local_grid_.u(i, j) = 0.0;
					local_grid_.v(i, j) = 0.0;
					local_grid_.eta(i, j) = 0.0;
					local_grid_.D(i, j) = 0.0;
				}
			}
		}
	}
}

void FourNeighborSubGrid::SetNeighbors(FourNeighborSubGrid* neighbor_up,
	FourNeighborSubGrid* neighbor_down,
	FourNeighborSubGrid* neighbor_left,
	FourNeighborSubGrid* neighbor_right)
{
	neighbor_left_ = neighbor_up;
	neighbor_right_ = neighbor_down;
	neighbor_up_ = neighbor_left;
	neighbor_down_ = neighbor_right;
}

// ============================================================================
// Диспетчерские методы - выбор реализации по системе координат
// ============================================================================

void FourNeighborSubGrid::ComputeHalfStep() {
	if (coord_system_ == CoordSystem::Spherical) {
		ComputeHalfStepSpherical();
	} else {
		ComputeHalfStepCartesian();
	}
}

void FourNeighborSubGrid::ComputeFullStep() {
	if (coord_system_ == CoordSystem::Spherical) {
		ComputeFullStepSpherical();
	} else {
		ComputeFullStepCartesian();
	}
}

// ============================================================================
// Реализация для сферической системы координат с силой Кориолиса
// ============================================================================

void FourNeighborSubGrid::ComputeHalfStepSpherical() {
	// Константы для сферической системы
	constexpr double R_EARTH = 6371000.0;           // Радиус Земли (м)
	constexpr double OMEGA = 7.2921e-5;             // Угловая скорость вращения Земли (рад/с)

	// FIXED: Используем center_latitude_rad_ из параметров сценария
	const double PHI_0 = center_latitude_rad_;

	// FIXED: Правильная формула для Δφ (delta_y_ в метрах)
	if (delta_y_ <= 0.0) {
		throw std::runtime_error("FourNeighborSubGrid::ComputeHalfStepSpherical: delta_y <= 0");
	}
	const double delta_phi = delta_y_ / R_EARTH;

	// Получаем размер глобальной сетки для правильного вычисления центра
	const int global_ny = global_grid_.GetNy();

	// Вычисления на половинном временном шаге
	for (int i = 1; i < local_nx_ - 1; ++i) {
		for (int j = 1; j < local_ny_ - 1; ++j) {
			// FIXED: Вычисляем широту относительно ГЛОБАЛЬНОГО индекса
			const int global_j = global_j_start_ + (j - 1);  // -1 т.к. j=1 соответствует первой внутренней ячейке
			const double phi = PHI_0 + (global_j - global_ny / 2.0) * delta_phi;
			const double sin_phi = std::sin(phi);

			// Параметр Кориолиса (единственное отличие от картезианской системы!)
			const double f = 2.0 * OMEGA * sin_phi;

			// FIXED: Для метрической сетки не нужны дополнительные коэффициенты
			const double dx = delta_x_(i, j);
			const double inv_dx = 1.0 / dx;
			const double inv_dy = 1.0 / delta_y_;

			// FIXED: Обычные потоки без лишних метрических коэффициентов
			double flux_H_x = (
				local_grid_.H(i, j) * local_grid_.u(i, j) -
				local_grid_.H(i - 1, j) * local_grid_.u(i - 1, j)) * inv_dx;
			double flux_H_y = (
				local_grid_.H(i, j) * local_grid_.v(i, j) -
				local_grid_.H(i, j - 1) * local_grid_.v(i, j - 1)) * inv_dy;

			H_hat_[i * local_ny_ + j] = local_grid_.H(i, j) - tau_ * (flux_H_x + flux_H_y);

			// FIXED: Обычные производные без лишних метрических коэффициентов
			double du_dx = (local_grid_.u(i, j) - local_grid_.u(i - 1, j)) * inv_dx;
			double du_dy = (local_grid_.u(i, j) - local_grid_.u(i, j - 1)) * inv_dy;

			double dv_dx = (local_grid_.v(i, j) - local_grid_.v(i - 1, j)) * inv_dx;
			double dv_dy = (local_grid_.v(i, j) - local_grid_.v(i, j - 1)) * inv_dy;

			double deta_dx = (local_grid_.eta(i, j) - local_grid_.eta(i - 1, j)) * inv_dx;
			double deta_dy = (local_grid_.eta(i, j) - local_grid_.eta(i, j - 1)) * inv_dy;

			// Уравнения с силой Кориолиса
			u_hat_[i * local_ny_ + j] = local_grid_.u(i, j) - tau_ * (
				local_grid_.u(i, j) * du_dx +
				local_grid_.v(i, j) * du_dy +
				g_ * deta_dx -
				f * local_grid_.v(i, j)  // Сила Кориолиса
			);

			v_hat_[i * local_ny_ + j] = local_grid_.v(i, j) - tau_ * (
				local_grid_.u(i, j) * dv_dx +
				local_grid_.v(i, j) * dv_dy +
				g_ * deta_dy +
				f * local_grid_.u(i, j)  // Сила Кориолиса
			);

			eta_hat_[i * local_ny_ + j] = H_hat_[i * local_ny_ + j] - local_grid_.D(i, j);
		}
	}
}

void FourNeighborSubGrid::ComputeFullStepSpherical() {
	// Константы для сферической системы
	constexpr double R_EARTH = 6371000.0;
	constexpr double OMEGA = 7.2921e-5;

	// FIXED: Используем center_latitude_rad_ из параметров сценария
	const double PHI_0 = center_latitude_rad_;

	// FIXED: Правильная формула для Δφ
	if (delta_y_ <= 0.0) {
		throw std::runtime_error("FourNeighborSubGrid::ComputeFullStepSpherical: delta_y <= 0");
	}
	const double delta_phi = delta_y_ / R_EARTH;

	// Получаем размер глобальной сетки для правильного вычисления центра
	const int global_ny = global_grid_.GetNy();

	// Вычисления на полном временном шаге
	for (int i = 1; i < local_nx_ - 1; ++i) {
		for (int j = 1; j < local_ny_ - 1; ++j) {
			// FIXED: Вычисляем широту относительно ГЛОБАЛЬНОГО индекса
			const int global_j = global_j_start_ + (j - 1);  // -1 т.к. j=1 соответствует первой внутренней ячейке
			const double phi = PHI_0 + (global_j - global_ny / 2.0) * delta_phi;
			const double sin_phi = std::sin(phi);
			const double f = 2.0 * OMEGA * sin_phi;

			// FIXED: Для метрической сетки не нужны метрические коэффициенты
			const double dx = delta_x_(i, j);
			const double inv_dx = 1.0 / dx;
			const double inv_dy = 1.0 / delta_y_;

			// FIXED: Обычные потоки без лишних коэффициентов
			double flux_H_hat_x = (
				H_hat_[(i + 1) * local_ny_ + j] * u_hat_[(i + 1) * local_ny_ + j] -
				H_hat_[i * local_ny_ + j] * u_hat_[i * local_ny_ + j]) * inv_dx;
			double flux_H_hat_y = (
				H_hat_[i * local_ny_ + (j + 1)] * v_hat_[i * local_ny_ + (j + 1)] -
				H_hat_[i * local_ny_ + j] * v_hat_[i * local_ny_ + j]) * inv_dy;

			local_grid_.H(i, j) = 0.5 * (local_grid_.H(i, j) + H_hat_[i * local_ny_ + j]) -
								  0.5 * tau_ * (flux_H_hat_x + flux_H_hat_y);

			// FIXED: Обычные производные без лишних коэффициентов
			double du_hat_dx = (u_hat_[(i + 1) * local_ny_ + j] - u_hat_[i * local_ny_ + j]) * inv_dx;
			double du_hat_dy = (u_hat_[i * local_ny_ + (j + 1)] - u_hat_[i * local_ny_ + j]) * inv_dy;

			double dv_hat_dx = (v_hat_[(i + 1) * local_ny_ + j] - v_hat_[i * local_ny_ + j]) * inv_dx;
			double dv_hat_dy = (v_hat_[i * local_ny_ + (j + 1)] - v_hat_[i * local_ny_ + j]) * inv_dy;

			double deta_hat_dx = (eta_hat_[(i + 1) * local_ny_ + j] - eta_hat_[i * local_ny_ + j]) * inv_dx;
			double deta_hat_dy = (eta_hat_[i * local_ny_ + (j + 1)] - eta_hat_[i * local_ny_ + j]) * inv_dy;

			// Обновление с силой Кориолиса
			local_grid_.u(i, j) = 0.5 * (local_grid_.u(i, j) + u_hat_[i * local_ny_ + j]) -
								  0.5 * tau_ * (
									  u_hat_[i * local_ny_ + j] * du_hat_dx +
									  v_hat_[i * local_ny_ + j] * du_hat_dy +
									  g_ * deta_hat_dx -
									  f * v_hat_[i * local_ny_ + j]  // Сила Кориолиса
								  );

			local_grid_.v(i, j) = 0.5 * (local_grid_.v(i, j) + v_hat_[i * local_ny_ + j]) -
								  0.5 * tau_ * (
									  u_hat_[i * local_ny_ + j] * dv_hat_dx +
									  v_hat_[i * local_ny_ + j] * dv_hat_dy +
									  g_ * deta_hat_dy +
									  f * u_hat_[i * local_ny_ + j]  // Сила Кориолиса
								  );

			local_grid_.eta(i, j) = local_grid_.H(i, j) - local_grid_.D(i, j);
		}
	}
}

// ============================================================================
// Реализация для декартовой системы координат
// ============================================================================

void FourNeighborSubGrid::ComputeHalfStepCartesian() {
	// Вычисления на половинном временном шаге
	for (int i = 1; i < local_nx_ - 1; ++i) {
		for (int j = 1; j < local_ny_ - 1; ++j) {
			double flux_H_x = (
				local_grid_.H(i, j) * local_grid_.u(i, j) -
				local_grid_.H(i - 1, j) * local_grid_.u(i - 1, j)) / delta_x_(i, j);
			double flux_H_y = (
				local_grid_.H(i, j) * local_grid_.v(i, j) -
				local_grid_.H(i, j - 1) * local_grid_.v(i, j - 1)) / delta_y_;

			H_hat_[i * local_ny_ + j] = local_grid_.H(i, j) - tau_ * (flux_H_x + flux_H_y);

			// Вычисляем производные скоростей и высоты для обновления u_hat_ и v_hat_
			double du_dx = (local_grid_.u(i, j) - local_grid_.u(i - 1, j)) / delta_x_(i, j);
			double du_dy = (local_grid_.u(i, j) - local_grid_.u(i, j - 1)) / delta_y_;

			double dv_dx = (local_grid_.v(i, j) - local_grid_.v(i - 1, j)) / delta_x_(i, j);
			double dv_dy = (local_grid_.v(i, j) - local_grid_.v(i, j - 1)) / delta_y_;

			double deta_dx = (local_grid_.eta(i, j) - local_grid_.eta(i - 1, j)) / delta_x_(i, j);
			double deta_dy = (local_grid_.eta(i, j) - local_grid_.eta(i, j - 1)) / delta_y_;

			// Обновляем u_hat_[i * local_ny_+j]
			u_hat_[i * local_ny_ + j] = local_grid_.u(i, j) - tau_ * (
				local_grid_.u(i, j) * du_dx +
				local_grid_.v(i, j) * du_dy +
				g_ * deta_dx
				);

			// Обновляем v_hat_[i * local_ny_+j]
			v_hat_[i * local_ny_ + j] = local_grid_.v(i, j) - tau_ * (
				local_grid_.u(i, j) * dv_dx +
				local_grid_.v(i, j) * dv_dy +
				g_ * deta_dy
				);

			// Обновляем eta_hat_[i * local_ny_+j]
			eta_hat_[i * local_ny_ + j] = H_hat_[i * local_ny_ + j] - local_grid_.D(i, j);
		}
	}
}

void FourNeighborSubGrid::ComputeFullStepCartesian() {
	// Вычисления на полном временном шаге
	for (int i = 1; i < local_nx_ - 1; ++i) {
		for (int j = 1; j < local_ny_ - 1; ++j) {
			// Вычисляем потоки для обновления H
			double flux_H_hat_x = (H_hat_[(i + 1) * local_ny_ + j] * u_hat_[(i + 1) * local_ny_ + j] - H_hat_[i * local_ny_ + j] * u_hat_[i * local_ny_ + j]) / delta_x_(i, j);
			double flux_H_hat_y = (H_hat_[i * local_ny_ + j + 1] * v_hat_[i * local_ny_ + j + 1] - H_hat_[i * local_ny_ + j] * v_hat_[i * local_ny_ + j]) / delta_y_;

			local_grid_.H(i, j) = 0.5 * (local_grid_.H(i, j) + H_hat_[i * local_ny_ + j]) - 0.5 * tau_ * (flux_H_hat_x + flux_H_hat_y);

			// Вычисляем производные скоростей и высоты для обновления u и v
			double du_hat_dx = (u_hat_[(i + 1) * local_ny_ + j] - u_hat_[i * local_ny_ + j]) / delta_x_(i, j);
			double du_hat_dy = (u_hat_[i * local_ny_ + j + 1] - u_hat_[i * local_ny_ + j]) / delta_y_;

			double dv_hat_dx = (v_hat_[(i + 1) * local_ny_ + j] - v_hat_[i * local_ny_ + j]) / delta_x_(i, j);
			double dv_hat_dy = (v_hat_[i * local_ny_ + j + 1] - v_hat_[i * local_ny_ + j]) / delta_y_;

			double deta_hat_dx = (eta_hat_[(i + 1) * local_ny_ + j] - eta_hat_[i * local_ny_ + j]) / delta_x_(i, j);
			double deta_hat_dy = (eta_hat_[i * local_ny_ + j + 1] - eta_hat_[i * local_ny_ + j]) / delta_y_;

			// Обновляем u
			double u_new = 0.5 * (local_grid_.u(i, j) + u_hat_[i * local_ny_ + j]) - 0.5 * tau_ * (
				u_hat_[i * local_ny_ + j] * du_hat_dx +
				v_hat_[i * local_ny_ + j] * du_hat_dy +
				g_ * deta_hat_dx
				);

			// Обновляем v
			double v_new = 0.5 * (local_grid_.v(i, j) + v_hat_[i * local_ny_ + j]) - 0.5 * tau_ * (
				u_hat_[i * local_ny_ + j] * dv_hat_dx +
				v_hat_[i * local_ny_ + j] * dv_hat_dy +
				g_ * deta_hat_dy
				);

			local_grid_.u(i, j) = u_new;
			local_grid_.v(i, j) = v_new;

			// Обновляем eta
			local_grid_.eta(i, j) = local_grid_.H(i, j) - local_grid_.D(i, j);
		}
	}
}


void FourNeighborSubGrid::UpdateGlobalGrid(int row, int col, int num_rows_, int num_cols_) {
	// Определение начального и конечного индекса для локальной сетки по i
	int local_i_start = (row > 0) ? 1 : 0;
	int local_i_end = (row < num_rows_ - 1) ? local_nx_ - 1 : local_nx_;

	// Определение начального и конечного индекса для локальной сетки по j
	int local_j_start = (col > 0) ? 1 : 0;
	int local_j_end = (col < num_cols_ - 1) ? local_ny_ - 1 : local_ny_;

	// Проход по локальной сетке
	for (int i = local_i_start; i < local_i_end; ++i) {
		// Вычисление глобального индекса по i
		int global_i = global_i_start_ + i - 1;

		// Проверка корректности глобального индекса по i
		if (global_i >= 0 && global_i < global_grid_.GetNx()) {
			for (int j = local_j_start; j < local_j_end; ++j) {
				// Вычисление глобального индекса по j
				int global_j = global_j_start_ + j - 1;

				// Проверка корректности глобального индекса по j
				if (global_j >= 0 && global_j < global_grid_.GetNy()) {
					// Копируем значения из локальной сетки в глобальную
					global_grid_.H(global_i, global_j) = local_grid_.H(i, j);
					global_grid_.u(global_i, global_j) = local_grid_.u(i, j);
					global_grid_.v(global_i, global_j) = local_grid_.v(i, j);
					global_grid_.eta(global_i, global_j) = local_grid_.eta(i, j);
				}
			}
		}
	}
}

void FourNeighborSubGrid::ExchangeHatBoundaryData2(std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
	// Обмен граничными данными с соседними подрешётками

	// Верхний граничный слой
	if (s_grid->neighbor_left_) {
		// Отправляем свои данные верхней сетке и получаем его данные
		// Пример обмена H_hat_
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			// Получаем данные от соседа
			s_grid->H_hat_[j] = s_grid->neighbor_left_->H_hat_[(s_grid->neighbor_left_->local_nx_ - 2) * s_grid->local_ny_ + j];
			s_grid->u_hat_[j] = s_grid->neighbor_left_->u_hat_[(s_grid->neighbor_left_->local_nx_ - 2) * s_grid->local_ny_ + j];
			s_grid->v_hat_[j] = s_grid->neighbor_left_->v_hat_[(s_grid->neighbor_left_->local_nx_ - 2) * s_grid->local_ny_ + j];
			s_grid->eta_hat_[j] = s_grid->neighbor_left_->eta_hat_[(s_grid->neighbor_left_->local_nx_ - 2) * s_grid->local_ny_ + j];
		}
	}
	else {
		// Граничные условия на верхней границе
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->H_hat_[j] = s_grid->H_hat_[s_grid->local_ny_ + j];
			s_grid->u_hat_[j] = s_grid->u_hat_[s_grid->local_ny_ + j];
			s_grid->v_hat_[j] = s_grid->v_hat_[s_grid->local_ny_ + j];
			s_grid->eta_hat_[j] = s_grid->eta_hat_[s_grid->local_ny_ + j];
		}
	}

	// Нижний граничный слой
	if (s_grid->neighbor_right_) {
		// Отправляем свои данные нижней сетке и получаем его данные
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			//neighbor_right_->H_hat_[j] = H_hat_[local_nx_ - 2][j];
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
		// Граничные условия на нижней границе
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

	for (int i = 0; i < s_grid->local_nx_; ++i) {
		s_grid->H_hat_[i * s_grid->local_ny_] = s_grid->H_hat_[i * s_grid->local_ny_ + 1];
		s_grid->u_hat_[i * s_grid->local_ny_] = s_grid->u_hat_[i * s_grid->local_ny_ + 1];
		s_grid->v_hat_[i * s_grid->local_ny_] = 0;//v_hat_[i*local_ny_+1];
		s_grid->eta_hat_[i * s_grid->local_ny_] = s_grid->eta_hat_[i * s_grid->local_ny_ + 1];
		s_grid->H_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
			s_grid->H_hat_[(i)*s_grid->local_ny_ + s_grid->local_ny_ - 2];
		s_grid->u_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
			s_grid->u_hat_[(i)*s_grid->local_ny_ + s_grid->local_ny_ - 2];
		s_grid->v_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] = 0;
		s_grid->eta_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
			s_grid->eta_hat_[(i)*s_grid->local_ny_ + s_grid->local_ny_ - 2];
	}
}

void FourNeighborSubGrid::ExchangeBoundaryData2(std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
	// Обмен граничными данными с соседними подрешётками

	// Верхний граничный слой
	if (s_grid->neighbor_left_) {
		// Отправляем свои данные верхней сетке и получаем его данные
		// Пример обмена H_hat_
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			// Получаем данные от соседа
			s_grid->local_grid_.H(0, j) = s_grid->neighbor_left_->local_grid_.H(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->local_grid_.u(0, j) = s_grid->neighbor_left_->local_grid_.u(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->local_grid_.v(0, j) = s_grid->neighbor_left_->local_grid_.v(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->local_grid_.eta(0, j) = s_grid->neighbor_left_->local_grid_.eta(s_grid->neighbor_left_->local_nx_ - 2, j);
		}
	}
	else {
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
			s_grid->local_grid_.u(0, j) = s_grid->local_grid_.u(1, j);
			s_grid->local_grid_.v(0, j) = s_grid->local_grid_.v(1, j);
			s_grid->local_grid_.eta(0, j) = s_grid->local_grid_.eta(1, j);
		}
	}

	// Нижний граничный слой
	if (s_grid->neighbor_right_) {
		// Отправляем свои данные нижней сетке и получаем его данные
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			//neighbor_right_->H_hat_[j] = H_hat_[local_nx_ - 2][j];
			s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.H(1, j);
			s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.u(1, j);
			s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.v(1, j);
			s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.eta(1, j);
		}
	}
	else {
		// Граничные условия на нижней границе
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.H(s_grid->local_nx_ - 2, j);
			s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.u(s_grid->local_nx_ - 2, j);
			s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.v(s_grid->local_nx_ - 2, j);
			s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.eta(s_grid->local_nx_ - 2, j);
		}
	}

	for (int i = 0; i < s_grid->local_nx_; ++i) {
		s_grid->local_grid_.H(i, 0) = s_grid->local_grid_.H(i, 1);
		s_grid->local_grid_.u(i, 0) = s_grid->local_grid_.u(i, 1);
		s_grid->local_grid_.v(i, 0) = 0;
		s_grid->local_grid_.eta(i, 0) = s_grid->local_grid_.eta(i, 1);
		s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.H(i, s_grid->local_ny_ - 2);
		s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.u(i, s_grid->local_ny_ - 2);
		s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) = 0;
		s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.eta(i, s_grid->local_ny_ - 2);
	}

}

void FourNeighborSubGrid::ExchangeHatBoundaryData1(std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
	// Левая граничная область
	if (s_grid->neighbor_left_) {
		// Отправляем свои данные верхней сетке и получаем его данные
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			// Получаем данные от соседа
			s_grid->H_hat(0, j) =
				s_grid->neighbor_left_->H_hat(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->u_hat(0, j) =
				s_grid->neighbor_left_->u_hat(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->v_hat(0, j) =
				s_grid->neighbor_left_->v_hat(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->eta_hat(0, j) =
				s_grid->neighbor_left_->eta_hat(s_grid->neighbor_left_->local_nx_ - 2, j);
		}
	}
	else {
		// Граничные условия на левой границе
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->H_hat(0, j) = s_grid->H_hat(1, j);
			s_grid->u_hat(0, j) = s_grid->u_hat(1, j);
			s_grid->v_hat(0, j) = s_grid->v_hat(1, j);
			s_grid->eta_hat(0, j) = s_grid->eta_hat(1, j);
		}
	}

	// Правая граничная область
	if (s_grid->neighbor_right_) {
		// Отправляем свои данные нижней сетке и получаем его данные
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->H_hat(s_grid->local_nx_ - 1, j) =
				s_grid->neighbor_right_->H_hat(1, j);
			s_grid->u_hat(s_grid->local_nx_ - 1, j) =
				s_grid->neighbor_right_->u_hat(1, j);
			s_grid->v_hat(s_grid->local_nx_ - 1, j) =
				s_grid->neighbor_right_->v_hat(1, j);
			s_grid->eta_hat(s_grid->local_nx_ - 1, j) =
				s_grid->neighbor_right_->eta_hat(1, j);
		}
	}
	else {
		// Граничные условия на правой границе
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->H_hat(s_grid->local_nx_ - 1, j) =
				s_grid->H_hat(s_grid->local_nx_ - 2, j);
			s_grid->u_hat(s_grid->local_nx_ - 1, j) =
				s_grid->u_hat(s_grid->local_nx_ - 2, j);
			s_grid->v_hat(s_grid->local_nx_ - 1, j) =
				s_grid->v_hat(s_grid->local_nx_ - 2, j);
			s_grid->eta_hat(s_grid->local_nx_ - 1, j) =
				s_grid->eta_hat(s_grid->local_nx_ - 2, j);
		}
	}

	// Верхний граничный слой
	if (s_grid->neighbor_up_) {
		// Отправляем свои данные левой сетке и получаем её данные
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->H_hat(i, 0) =
				s_grid->neighbor_up_->H_hat(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->u_hat(i, 0) =
				s_grid->neighbor_up_->u_hat(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->v_hat(i, 0) =
				s_grid->neighbor_up_->v_hat(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->eta_hat(i, 0) =
				s_grid->neighbor_up_->eta_hat(i, s_grid->neighbor_up_->local_ny_ - 2);
		}
	}
	else {
		// Граничные условия на верхней границе
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->H_hat(i, 0) =
				s_grid->H_hat(i, 1);
			s_grid->u_hat(i, 0) =
				s_grid->u_hat(i, 1);
			s_grid->v_hat(i, 0) =
				s_grid->v_hat(i, 1);
			s_grid->eta_hat(i, 0) =
				s_grid->eta_hat(i, 1);
		}
	}

	// Нижний граничный слой
	if (s_grid->neighbor_down_) {
		// Отправляем свои данные правой сетке и получаем её данные
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->H_hat(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->H_hat(i, 1);
			s_grid->u_hat(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->u_hat(i, 1);
			s_grid->v_hat(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->v_hat(i, 1);
			s_grid->eta_hat(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->eta_hat(i, 1);
		}
	}
	else {
		// Граничные условия на нижней границе
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->H_hat(i, s_grid->local_ny_ - 1) =
				s_grid->H_hat_[(i + 1) * s_grid->local_ny_ - 2];
			s_grid->u_hat(i, s_grid->local_ny_ - 1) =
				s_grid->u_hat(i, s_grid->local_ny_ - 2);
			s_grid->v_hat(i, s_grid->local_ny_ - 1) =
				s_grid->v_hat(i, s_grid->local_ny_ - 2);
			s_grid->eta_hat(i, s_grid->local_ny_ - 1) =
				s_grid->eta_hat(i, s_grid->local_ny_ - 2);
		}
	}

	//for (int i = 0; i < s_grid->local_nx_; ++i) {
	//	s_grid->H_hat(i, 0) = s_grid->H_hat(i, 1);
	//	s_grid->u_hat(i, 0) = s_grid->u_hat(i, 1);
	//	s_grid->v_hat(i, 0) = s_grid->v_hat(i, 1);
	//	s_grid->eta_hat(i, 0) = s_grid->eta_hat(i, 1);
	//	s_grid->H_hat(i, s_grid->local_ny_ - 1) = s_grid->H_hat(i, s_grid->local_ny_ - 2);
	//	s_grid->u_hat(i, s_grid->local_ny_ - 1) = s_grid->u_hat(i, s_grid->local_ny_ - 2);
	//	s_grid->v_hat(i, s_grid->local_ny_ - 1) = s_grid->v_hat(i, s_grid->local_ny_ - 2);
	//	s_grid->eta_hat(i, s_grid->local_ny_ - 1) = s_grid->eta_hat(i, s_grid->local_ny_ - 2);
	//}
	//
	//for (int j = 0; j < s_grid->local_ny_; ++j) {
	//	s_grid->H_hat(0, j) = s_grid->H_hat(1, j);
	//	s_grid->u_hat(0, j) = s_grid->u_hat(1, j);
	//	s_grid->v_hat(0, j) = s_grid->v_hat(1, j);
	//	s_grid->eta_hat(0, j) = s_grid->eta_hat(1, j);
	//	s_grid->H_hat(s_grid->local_nx_ - 1, j) = s_grid->H_hat(s_grid->local_nx_ - 2, j);
	//	s_grid->u_hat(s_grid->local_nx_ - 1, j) = s_grid->u_hat(s_grid->local_nx_ - 2, j);
	//	s_grid->v_hat(s_grid->local_nx_ - 1, j) = s_grid->v_hat(s_grid->local_nx_ - 2, j);
	//	s_grid->eta_hat(s_grid->local_nx_ - 1, j) = s_grid->eta_hat(s_grid->local_nx_ - 2, j);
	//}
}

void FourNeighborSubGrid::ExchangeBoundaryData1(std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
	// Левая граничная область
	if (s_grid->neighbor_left_) {
		// Отправляем свои данные верхней сетке и получаем её данные
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->local_grid_.H(0, j) = s_grid->neighbor_left_->local_grid_.H(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->local_grid_.u(0, j) = s_grid->neighbor_left_->local_grid_.u(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->local_grid_.v(0, j) = s_grid->neighbor_left_->local_grid_.v(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->local_grid_.eta(0, j) = s_grid->neighbor_left_->local_grid_.eta(s_grid->neighbor_left_->local_nx_ - 2, j);
		}
	}
	else {
		// Граничные условия на левой границе
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
			s_grid->local_grid_.u(0, j) = s_grid->local_grid_.u(1, j);
			s_grid->local_grid_.v(0, j) = s_grid->local_grid_.v(1, j);
			s_grid->local_grid_.eta(0, j) = s_grid->local_grid_.eta(1, j);
		}
	}

	// Правая граничная область
	if (s_grid->neighbor_right_) {
		// Отправляем свои данные нижней сетке и получаем её данные
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.H(1, j);
			s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.u(1, j);
			s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.v(1, j);
			s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.eta(1, j);
		}
	}
	else {
		// Граничные условия на правой границе
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.H(s_grid->local_nx_ - 2, j);
			s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.u(s_grid->local_nx_ - 2, j);
			s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.v(s_grid->local_nx_ - 2, j);
			s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.eta(s_grid->local_nx_ - 2, j);
		}
	}

	// Верхний граничная область
	if (s_grid->neighbor_up_) {
		// Отправляем свои данные левой сетке и получаем её данные
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->local_grid_.H(i, 0) =
				s_grid->neighbor_up_->local_grid_.H(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->local_grid_.u(i, 0) =
				s_grid->neighbor_up_->local_grid_.u(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->local_grid_.v(i, 0) =
				s_grid->neighbor_up_->local_grid_.v(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->local_grid_.eta(i, 0) =
				s_grid->neighbor_up_->local_grid_.eta(i, s_grid->neighbor_up_->local_ny_ - 2);
		}
	}
	else {
		// Граничные условия на верхней границе
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->local_grid_.H(i, 0) =
				s_grid->local_grid_.H(i, 1);
			s_grid->local_grid_.u(i, 0) =
				s_grid->local_grid_.u(i, 1);
			s_grid->local_grid_.v(i, 0) =
				s_grid->local_grid_.v(i, 1);
			s_grid->local_grid_.eta(i, 0) =
				s_grid->local_grid_.eta(i, 1);
		}
	}

	// Нижний граничный слой
	if (s_grid->neighbor_down_) {
		// Отправляем свои данные правой сетке и получаем её данные
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->local_grid_.H(i, 1);
			s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->local_grid_.u(i, 1);
			s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->local_grid_.v(i, 1);
			s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->local_grid_.eta(i, 1);
		}
	}
	else {
		// Граничные условия на нижней границе
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) =
				s_grid->local_grid_.H(i, s_grid->local_ny_ - 2);
			s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) =
				s_grid->local_grid_.u(i, s_grid->local_ny_ - 2);
			s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) =
				s_grid->local_grid_.v(i, s_grid->local_ny_ - 2);
			s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) =
				s_grid->local_grid_.eta(i, s_grid->local_ny_ - 2);
		}
	}

	//for (int i = 0; i < s_grid->local_nx_; ++i) {
	//	s_grid->local_grid_.H(i, 0) = s_grid->local_grid_.H(i, 1);
	//	s_grid->local_grid_.u(i, 0) = s_grid->local_grid_.u(i, 1);
	//	s_grid->local_grid_.v(i, 0) = s_grid->local_grid_.v(i, 1);
	//	s_grid->local_grid_.eta(i, 0) = s_grid->local_grid_.eta(i, 1);
	//	s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.H(i, s_grid->local_ny_ - 2);
	//	s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.u(i, s_grid->local_ny_ - 2);
	//	s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.v(i, s_grid->local_ny_ - 2);
	//	s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) = s_grid->local_grid_.eta(i, s_grid->local_ny_ - 2);
	//}
	//
	//for (int j = 0; j < s_grid->local_ny_; ++j) {
	//	s_grid->local_grid_.H(0, j) = s_grid->local_grid_.H(1, j);
	//	s_grid->local_grid_.u(0, j) = s_grid->local_grid_.u(1, j);
	//	s_grid->local_grid_.v(0, j) = s_grid->local_grid_.v(1, j);
	//	s_grid->local_grid_.eta(0, j) = s_grid->local_grid_.eta(1, j);
	//	s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.H(s_grid->local_nx_ - 2, j);
	//	s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.u(s_grid->local_nx_ - 2, j);
	//	s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.v(s_grid->local_nx_ - 2, j);
	//	s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.eta(s_grid->local_nx_ - 2, j);
	//}
}

void FourNeighborSubGrid::ExchangeHatBoundaryData3(std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
	// Левая граничная область
	if (s_grid->neighbor_left_) {
		// Отправляем свои данные верхней сетке и получаем его данные
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			// Получаем данные от соседа
			s_grid->H_hat(0, j) =
				s_grid->neighbor_left_->H_hat(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->u_hat(0, j) =
				s_grid->neighbor_left_->u_hat(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->v_hat(0, j) =
				s_grid->neighbor_left_->v_hat(s_grid->neighbor_left_->local_nx_ - 2, j);
			s_grid->eta_hat(0, j) =
				s_grid->neighbor_left_->eta_hat(s_grid->neighbor_left_->local_nx_ - 2, j);
		}
	}
	else {
		// Граничные условия на левой границе
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->H_hat(0, j) = s_grid->H_hat(1, j);
			s_grid->u_hat(0, j) = s_grid->u_hat(1, j);
			s_grid->v_hat(0, j) = s_grid->v_hat(1, j);
			s_grid->eta_hat(0, j) = s_grid->eta_hat(1, j);
		}
	}

	// Правая граничная область
	if (s_grid->neighbor_right_) {
		// Отправляем свои данные нижней сетке и получаем его данные
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->H_hat(s_grid->local_nx_ - 1, j) =
				s_grid->neighbor_right_->H_hat(1, j);
			s_grid->u_hat(s_grid->local_nx_ - 1, j) =
				s_grid->neighbor_right_->u_hat(1, j);
			s_grid->v_hat(s_grid->local_nx_ - 1, j) =
				s_grid->neighbor_right_->v_hat(1, j);
			s_grid->eta_hat(s_grid->local_nx_ - 1, j) =
				s_grid->neighbor_right_->eta_hat(1, j);
		}
	}
	else {
		// Граничные условия на правой границе
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->H_hat(s_grid->local_nx_ - 1, j) =
				s_grid->H_hat(s_grid->local_nx_ - 2, j);
			s_grid->u_hat(s_grid->local_nx_ - 1, j) =
				s_grid->u_hat(s_grid->local_nx_ - 2, j);
			s_grid->v_hat(s_grid->local_nx_ - 1, j) =
				s_grid->v_hat(s_grid->local_nx_ - 2, j);
			s_grid->eta_hat(s_grid->local_nx_ - 1, j) =
				s_grid->eta_hat(s_grid->local_nx_ - 2, j);
		}
	}

	// Верхний граничный слой
	if (s_grid->neighbor_up_) {
		// Отправляем свои данные левой сетке и получаем её данные
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->H_hat(i, 0) =
				s_grid->neighbor_up_->H_hat(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->u_hat(i, 0) =
				s_grid->neighbor_up_->u_hat(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->v_hat(i, 0) =
				s_grid->neighbor_up_->v_hat(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->eta_hat(i, 0) =
				s_grid->neighbor_up_->eta_hat(i, s_grid->neighbor_up_->local_ny_ - 2);
		}
	}
	else {
		// Граничные условия на верхней границе
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->H_hat(i, 0) =
				s_grid->H_hat(i, 1);
			s_grid->u_hat(i, 0) =
				s_grid->u_hat(i, 1);
			s_grid->v_hat(i, 0) =
				s_grid->v_hat(i, 1);
			s_grid->eta_hat(i, 0) =
				s_grid->eta_hat(i, 1);
		}
	}

	// Нижний граничный слой
	if (s_grid->neighbor_down_) {
		// Отправляем свои данные правой сетке и получаем её данные
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->H_hat(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->H_hat(i, 1);
			s_grid->u_hat(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->u_hat(i, 1);
			s_grid->v_hat(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->v_hat(i, 1);
			s_grid->eta_hat(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->eta_hat(i, 1);
		}
	}
	else {
		// Граничные условия на нижней границе
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->H_hat(i, s_grid->local_ny_ - 1) =
				s_grid->H_hat_[(i + 1) * s_grid->local_ny_ - 2];
			s_grid->u_hat(i, s_grid->local_ny_ - 1) =
				s_grid->u_hat(i, s_grid->local_ny_ - 2);
			s_grid->v_hat(i, s_grid->local_ny_ - 1) =
				s_grid->v_hat(i, s_grid->local_ny_ - 2);
			s_grid->eta_hat(i, s_grid->local_ny_ - 1) =
				s_grid->eta_hat(i, s_grid->local_ny_ - 2);
		}
	}

	//for (int i = 0; i < s_grid->local_nx_; ++i) {
	//	s_grid->H_hat_[i * s_grid->local_ny_] =
	//		s_grid->H_hat_[i * s_grid->local_ny_ + 1];
	//	s_grid->u_hat_[i * s_grid->local_ny_] =
	//		s_grid->u_hat_[i * s_grid->local_ny_ + 1];
	//	s_grid->v_hat_[i * s_grid->local_ny_] =
	//		s_grid->v_hat_[i * s_grid->local_ny_ + 1];
	//	s_grid->eta_hat_[i * s_grid->local_ny_] =
	//		s_grid->eta_hat_[i * s_grid->local_ny_ + 1];
	//	s_grid->H_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
	//		s_grid->H_hat_[(i)*s_grid->local_ny_ + s_grid->local_ny_ - 2];
	//	s_grid->u_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
	//		s_grid->u_hat_[(i)*s_grid->local_ny_ + s_grid->local_ny_ - 2];
	//	s_grid->v_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] = 0;
	//	s_grid->eta_hat_[i * s_grid->local_ny_ + s_grid->local_ny_ - 1] =
	//		s_grid->eta_hat_[(i)*s_grid->local_ny_ + s_grid->local_ny_ - 2];
	//}
}

void FourNeighborSubGrid::ExchangeBoundaryData3(std::shared_ptr<FourNeighborSubGrid> s_grid, double t) {
	// Левый граничный слой
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

	// Правый граничный слой
	if (s_grid->neighbor_right_) {
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.H(1, j);
			s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.u(1, j);
			s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.v(1, j);
			s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) = s_grid->neighbor_right_->local_grid_.eta(1, j);
		}
	}
	else {
		// Граничные условия на правой границе
		for (int j = 0; j < s_grid->local_ny_; ++j) {
			s_grid->local_grid_.H(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.H(s_grid->local_nx_ - 2, j);
			s_grid->local_grid_.u(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.u(s_grid->local_nx_ - 2, j);
			s_grid->local_grid_.v(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.v(s_grid->local_nx_ - 2, j);
			s_grid->local_grid_.eta(s_grid->local_nx_ - 1, j) = s_grid->local_grid_.eta(s_grid->local_nx_ - 2, j);
		}
	}

	// Верхний граничная область
	if (s_grid->neighbor_up_) {
		// Отправляем свои данные левой сетке и получаем её данные
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->local_grid_.H(i, 0) =
				s_grid->neighbor_up_->local_grid_.H(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->local_grid_.u(i, 0) =
				s_grid->neighbor_up_->local_grid_.u(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->local_grid_.v(i, 0) =
				s_grid->neighbor_up_->local_grid_.v(i, s_grid->neighbor_up_->local_ny_ - 2);
			s_grid->local_grid_.eta(i, 0) =
				s_grid->neighbor_up_->local_grid_.eta(i, s_grid->neighbor_up_->local_ny_ - 2);
		}
	}
	else {
		// Граничные условия на верхней границе
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->local_grid_.H(i, 0) =
				s_grid->local_grid_.H(i, 1);
			s_grid->local_grid_.u(i, 0) =
				s_grid->local_grid_.u(i, 1);
			s_grid->local_grid_.v(i, 0) =
				s_grid->local_grid_.v(i, 1);
			s_grid->local_grid_.eta(i, 0) =
				s_grid->local_grid_.eta(i, 1);
		}
	}

	// Нижний граничный слой
	if (s_grid->neighbor_down_) {
		// Отправляем свои данные правой сетке и получаем её данные
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->local_grid_.H(i, 1);
			s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->local_grid_.u(i, 1);
			s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->local_grid_.v(i, 1);
			s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) =
				s_grid->neighbor_down_->local_grid_.eta(i, 1);
		}
	}
	else {
		// Граничные условия на нижней границе
		for (int i = 0; i < s_grid->local_nx_; ++i) {
			s_grid->local_grid_.H(i, s_grid->local_ny_ - 1) =
				s_grid->local_grid_.H(i, s_grid->local_ny_ - 2);
			s_grid->local_grid_.u(i, s_grid->local_ny_ - 1) =
				s_grid->local_grid_.u(i, s_grid->local_ny_ - 2);
			s_grid->local_grid_.v(i, s_grid->local_ny_ - 1) =
				s_grid->local_grid_.v(i, s_grid->local_ny_ - 2);
			s_grid->local_grid_.eta(i, s_grid->local_ny_ - 1) =
				s_grid->local_grid_.eta(i, s_grid->local_ny_ - 2);
		}
	}

}

double& FourNeighborSubGrid::H_hat(int i, int j) {
	return H_hat_[i * local_ny_ + j];
}

double& FourNeighborSubGrid::u_hat(int i, int j) {
	return u_hat_[i * local_ny_ + j];
}

double& FourNeighborSubGrid::v_hat(int i, int j) {
	return v_hat_[i * local_ny_ + j];
}

double& FourNeighborSubGrid::eta_hat(int i, int j) {
	return eta_hat_[i * local_ny_ + j];
}

const double& FourNeighborSubGrid::H_hat(int i, int j) const {
	return H_hat_[i * local_ny_ + j];
}

const double& FourNeighborSubGrid::u_hat(int i, int j) const {
	return u_hat_[i * local_ny_ + j];
}

const double& FourNeighborSubGrid::v_hat(int i, int j) const {
	return v_hat_[i * local_ny_ + j];
}

const double& FourNeighborSubGrid::eta_hat(int i, int j) const {
	return eta_hat_[i * local_ny_ + j];
}
