// IPatch.h - Абстрактный интерфейс патча для гибридной архитектуры
// Определяет единый API для всех типов патчей (structured, mesh, и т.д.)

#ifndef IPATCH_H
#define IPATCH_H

#include "PatchTypes.h"
#include "Precision.h"  // Real (float or double via USE_FLOAT_PRECISION)
#include <vector>
#include <memory>

/// Абстрактный интерфейс патча (подсетки)
///
/// Патч - это независимая подобласть домена, которая:
/// - Хранит свои физические поля (H, u, v, eta, D)
/// - Вычисляет свою эволюцию по времени
/// - Обменивается данными с соседними патчами через интерфейсы
///
/// Разные типы патчей (structured, mesh) реализуют этот интерфейс
class IPatch {
public:
    virtual ~IPatch() = default;

    // ========== LIFECYCLE ==========

    /// Инициализация патча (выделение памяти, начальные условия)
    virtual void Initialize() = 0;

    // ========== TIME STEPPING (MacCormack scheme) ==========

    /// Вычислить половинный шаг MacCormack (предварительное решение)
    /// Результат: hat-поля (H_hat, u_hat, v_hat, eta_hat)
    virtual void ComputeHalfStep() = 0;

    /// Вычислить полный шаг MacCormack (коррекция со вторым порядком)
    /// Вход: hat-поля, Результат: обновленные основные поля (H, u, v, eta)
    virtual void ComputeFullStep() = 0;

    // ========== INTERFACE EXCHANGE ==========

    /// Подготовить данные для обмена с соседними патчами
    /// (копировать граничные значения в интерфейсные буферы)
    ///
    /// @param dt Временной шаг
    /// @param t  Текущее время
    virtual void PrepareExchange(double dt, double t) = 0;

    /// Применить полученные данные от соседних патчей
    /// (копировать из интерфейсных буферов в halo-зоны)
    ///
    /// @param dt Временной шаг
    /// @param t  Текущее время
    virtual void ApplyExchange(double dt, double t) = 0;

    // ========== FIELD ACCESS ==========

    /// Получить прямой доступ к массиву данных поля (для ЧТЕНИЯ, экспорт данных патча)
    ///
    /// @param field Тип поля (H, U, V, Eta, D, H_hat, etc.)
    /// @return Указатель на массив данных (может быть nullptr если поле не существует)
    virtual Real* GetFieldData(FieldType field) = 0;

    /// Константный доступ к массиву данных поля
    virtual const Real* GetFieldData(FieldType field) const = 0;

    /// Получить буфер для ЗАПИСИ входящих данных от соседнего патча.
    /// По умолчанию совпадает с GetFieldData() (для StructuredPatch ghost-ячейки — те же буферы).
    /// MeshPatch переопределяет: возвращает отдельные incoming-буферы, чтобы не перезаписывать
    /// DG-решение при CopyBufferToPatch.
    virtual Real* GetIncomingFieldData(FieldType field) { return GetFieldData(field); }

    // ========== GEOMETRY INFO ==========

    /// Получить уникальный идентификатор патча
    virtual int GetPatchId() const = 0;

    /// Получить список интерфейсов этого патча
    /// (границы для обмена с соседними патчами)
    virtual std::vector<PatchInterface> GetInterfaces() const = 0;

    /// Получить количество точек в локальной сетке патча
    /// @param nx Размер по X (выходной параметр)
    /// @param ny Размер по Y (выходной параметр)
    virtual void GetLocalSize(int& nx, int& ny) const = 0;

    // ========== BOUNDARY CONDITIONS ==========

    /// Apply physical boundary conditions after exchange phase
    /// Default: no-op. StructuredPatch overrides to call legacy boundary functions.
    /// @param phase Which exchange phase just completed
    /// @param t Current timestep index
    virtual void ApplyBoundaryConditions(ExchangePhase phase, double t) { (void)phase; (void)t; }

    // ========== UTILITY ==========

    /// Обновить глобальную сетку результатами из этого патча
    /// (для сбора данных перед сохранением)
    ///
    /// @param global_grid Глобальная сетка для записи результатов
    virtual void UpdateGlobalGrid(void* global_grid) = 0;
};

#endif // IPATCH_H
