// SimulationRunner.h - Главный цикл симуляции для гибридного домена
// Заменяет SingleLineGridSimulation::Run() для новой архитектуры

#ifndef SIMULATION_RUNNER_H
#define SIMULATION_RUNNER_H

#include "HybridDomain.h"
#include <memory>
#include <string>
#include <functional>

/// Главный цикл симуляции для гибридного домена
///
/// Управляет временной эволюцией системы:
/// 1. HalfStep для всех патчей
/// 2. Exchange hat-полей
/// 3. FullStep для всех патчей
/// 4. Exchange основных полей
/// 5. Сохранение результатов
///
/// Структура цикла идентична SingleLineGridSimulation::Run()
class SimulationRunner {
public:
    /// Конструктор
    /// @param domain Гибридный домен для симуляции
    /// @param tau Временной шаг
    /// @param timesteps Количество временных шагов
    /// @param g Ускорение свободного падения (default 9.81)
    SimulationRunner(std::shared_ptr<HybridDomain> domain,
                     double tau,
                     int timesteps,
                     double g = 9.81);

    ~SimulationRunner() = default;

    // ========== НАСТРОЙКА ==========

    /// Установить интервал сохранения результатов
    /// @param interval Интервал в шагах (по умолчанию 100)
    void SetSaveInterval(int interval) { save_interval_ = interval; }

    /// Получить интервал сохранения
    int GetSaveInterval() const { return save_interval_; }

    /// Установить функцию сохранения данных
    /// @param save_func Функция сохранения (timestep, grid)
    void SetSaveFunction(std::function<void(int, const Grid&)> save_func) {
        save_function_ = save_func;
    }

    /// Включить/выключить вывод прогресса
    void SetVerbose(bool verbose) { verbose_ = verbose; }

    // ========== ВЫПОЛНЕНИЕ ==========

    /// Запустить симуляцию
    void Run();

    /// Сохранить результаты для текущего timestep
    /// @param timestep Номер текущего временного шага
    void SaveResults(int timestep);

    // ========== ИНФОРМАЦИЯ ==========

    /// Получить количество временных шагов
    int GetTimesteps() const { return timesteps_; }

    /// Получить временной шаг
    double GetTimeStep() const { return tau_; }

    /// Получить домен
    std::shared_ptr<HybridDomain> GetDomain() { return domain_; }
    const std::shared_ptr<HybridDomain> GetDomain() const { return domain_; }

private:
    // ========== ВНУТРЕННИЕ ДАННЫЕ ==========

    std::shared_ptr<HybridDomain> domain_;  // Домен для симуляции
    double g_;                              // Ускорение свободного падения
    double tau_;                            // Временной шаг
    int timesteps_;                         // Количество временных шагов
    int save_interval_;                     // Интервал сохранения

    bool verbose_;                          // Вывод прогресса

    // First-offender diagnostic state: fires ONCE at the first timestep where a
    // non-finite value or |eta| > offender_eta_bound_ appears, logging exactly
    // where (DG stripe vs MC stripe, x/y, distance to seam, local D) the blow-up
    // is born. Enabled by default; disable with env DG_OFFENDER=0; bound via
    // DG_OFFENDER_ETA (default 50 m).
    bool offender_fired_ = false;
    bool offender_enabled_ = true;
    double offender_eta_bound_ = -1.0;  // <0 = not yet read from env

    /// Функция сохранения (опциональна)
    std::function<void(int, const Grid&)> save_function_;

    // ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

    /// Вывести прогресс симуляции
    /// @param current_step Текущий шаг
    void PrintProgress(int current_step);

    /// First-offender localization. Scans every DG (MeshPatch) and the global
    /// structured grid for the first non-finite value or |eta| beyond a physical
    /// bound; on the first offending timestep prints a detailed [FIRST-OFFENDER]
    /// block (step, DG vs MC, x/y, distance to DG seam, H/D/eta/u/v) and then
    /// stays silent. Self-gates via offender_fired_; cheap after firing.
    void ReportFirstOffender(int t);

    /// Transfer DG HLL flux densities from every MeshPatch to the matching
    /// structured receiver side (legacy stripes and nested rectangles).
    /// Called once after AfterFullStep exchange each iteration; the buffers
    /// are then read by the next step's predictor/corrector.
    void TransferDGFluxesToStructured();
};

#endif // SIMULATION_RUNNER_H
