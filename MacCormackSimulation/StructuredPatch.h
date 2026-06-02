// StructuredPatch.h - Обертка для структурированной подсетки (SubGrid)
// Адаптирует существующий SubGrid к интерфейсу IPatch без изменения его кода

#ifndef STRUCTURED_PATCH_H
#define STRUCTURED_PATCH_H

#include "IPatch.h"
#include "SubGrid.h"
#include <memory>
#include <functional>

/// StructuredPatch - патч для структурированной (i-j) сетки
///
/// Это обертка над существующим SubGrid, которая:
/// - Делегирует все вычисления на SubGrid (без изменений!)
/// - Предоставляет унифицированный API через IPatch
/// - Управляет интерфейсами для обмена с соседними патчами
///
/// Принцип композиции: SubGrid остается неизменным, адаптация в обертке
class StructuredPatch : public IPatch {
public:
    /// Конструктор
    /// @param patch_id Уникальный идентификатор патча
    /// @param subgrid Существующий SubGrid для обертывания
    /// @param delta_x Функция шага по X (нужна для вычисления координат интерфейсов)
    /// @param delta_y Шаг по Y (нужна для вычисления координат интерфейсов)
    StructuredPatch(int patch_id,
                    std::shared_ptr<SubGrid> subgrid,
                    std::function<double(int, int)> delta_x,
                    double delta_y);

    virtual ~StructuredPatch() = default;

    // ========== IPatch Interface Implementation ==========

    void Initialize() override;

    void ComputeHalfStep() override;

    void ComputeFullStep() override;

    void PrepareExchange(double dt, double t) override;

    void ApplyExchange(double dt, double t) override;

    void ApplyBoundaryConditions(ExchangePhase phase, double t) override;

    Real* GetFieldData(FieldType field) override;

    const Real* GetFieldData(FieldType field) const override;

    int GetPatchId() const override;

    std::vector<PatchInterface> GetInterfaces() const override;

    void GetLocalSize(int& nx, int& ny) const override;

    void UpdateGlobalGrid(void* global_grid) override;

    // ========== Structured-specific methods ==========

    /// Построить интерфейсы патча на основе геометрии подсетки
    /// Вызывается автоматически в Initialize()
    void BuildInterfaces();

    /// Получить прямой доступ к внутреннему SubGrid (для отладки/диагностики)
    std::shared_ptr<SubGrid> GetSubGrid() { return subgrid_; }

    const std::shared_ptr<SubGrid> GetSubGrid() const { return subgrid_; }

private:
    // ========== Внутренние данные ==========

    int patch_id_;                           // Уникальный ID патча
    std::shared_ptr<SubGrid> subgrid_;       // Внутренний SubGrid (композиция!)

    std::vector<PatchInterface> interfaces_; // Интерфейсы для обмена

    // Геометрические параметры (для построения интерфейсов)
    std::function<double(int, int)> delta_x_;
    double delta_y_;

    // Coupling flags: when true, ExchangeEngine handles this side.
    // Legacy BC callbacks are skipped for coupled sides.
    bool has_coupled_left_ = false;
    bool has_coupled_right_ = false;

public:
    void SetCoupledLeft(bool v) { has_coupled_left_ = v; }
    void SetCoupledRight(bool v) { has_coupled_right_ = v; }
private:

    // ========== Вспомогательные методы ==========

    /// Вычислить физическую X-координату для глобального индекса i
    double ComputePhysicalX(int global_i, int j) const;

    /// Вычислить физическую Y-координату для индекса j
    double ComputePhysicalY(int j) const;

    /// Создать левый интерфейс
    void BuildLeftInterface();

    /// Создать правый интерфейс
    void BuildRightInterface();

    /// Генерация уникального interface_id из patch_id и стороны (0=left, 1=right)
    static int GenerateInterfaceId(int patch_id, int side);
};

#endif // STRUCTURED_PATCH_H
