// ExchangeEngine.h - Координатор обменов данными между патчами
// Управляет всеми обменами через CouplingMap'ы

#ifndef EXCHANGE_ENGINE_H
#define EXCHANGE_ENGINE_H

#include "IPatch.h"
#include "CouplingMap.h"
#include "PatchTypes.h"
#include <memory>
#include <unordered_map>
#include <vector>

/// Движок обмена данными между патчами
///
/// Координирует все обмены данными:
/// 1. PrepareExchange() у всех патчей
/// 2. Копирование данных из патчей в интерфейсные буферы
/// 3. Применение всех CouplingMap (интерполяция/агрегация)
/// 4. Копирование из буферов обратно в патчи
/// 5. ApplyExchange() у всех патчей
///
/// Поддерживает фазированный обмен для MacCormack (hat-поля и основные поля)
class ExchangeEngine {
public:
    ExchangeEngine() = default;
    ~ExchangeEngine() = default;

    // ========== РЕГИСТРАЦИЯ ==========

    /// Зарегистрировать патч для обменов
    /// @param patch_id Уникальный ID патча
    /// @param patch Указатель на патч
    void RegisterPatch(int patch_id, std::shared_ptr<IPatch> patch);

    /// Зарегистрировать карту связей между интерфейсами
    /// @param coupling CouplingMap для пары интерфейсов
    void RegisterCoupling(std::shared_ptr<CouplingMap> coupling);

    // ========== ОБМЕН ДАННЫМИ ==========

    /// Выполнить обмен полями между всеми патчами для заданной фазы
    /// @param dt Временной шаг
    /// @param t Текущее время
    /// @param phase Фаза обмена (AfterHalfStep или AfterFullStep)
    void ExchangeFieldsPhased(double dt, double t, ExchangePhase phase);

    /// Выполнить обмен для конкретного набора полей
    /// @param dt Временной шаг
    /// @param t Текущее время
    /// @param fields Список полей для обмена
    void ExchangeFields(double dt, double t, const std::vector<FieldType>& fields);

    // ========== ДИАГНОСТИКА ==========

    /// Получить количество зарегистрированных патчей
    size_t GetNumPatches() const { return patches_.size(); }

    /// Получить количество зарегистрированных связей
    size_t GetNumCouplings() const { return couplings_.size(); }

    /// Очистить все регистрации (для повторного использования)
    void Clear();

private:
    // ========== ВНУТРЕННИЕ ДАННЫЕ ==========

    /// Зарегистрированные патчи (patch_id → IPatch)
    std::unordered_map<int, std::shared_ptr<IPatch>> patches_;

    /// Зарегистрированные связи между интерфейсами
    std::vector<std::shared_ptr<CouplingMap>> couplings_;

    /// Буфер данных для интерфейса
    struct InterfaceBuffer {
        std::unordered_map<FieldType, std::vector<double>> fields;

        /// Получить буфер для поля (создать если не существует)
        std::vector<double>& GetFieldBuffer(FieldType field, size_t size);
    };

    /// Буферы для каждого интерфейса (interface_id → InterfaceBuffer)
    std::unordered_map<int, InterfaceBuffer> interface_buffers_;

    /// Кеш интерфейсов (interface_id → PatchInterface)
    /// Предотвращает UB от указателей на временные объекты
    std::unordered_map<int, PatchInterface> interface_cache_;

    // ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

    /// Построить кеш интерфейсов из всех зарегистрированных патчей
    void RebuildInterfaceCache();

    /// Получить список полей для обмена в зависимости от фазы
    std::vector<FieldType> GetFieldsForPhase(ExchangePhase phase) const;

    /// Заполнить буфер интерфейса из патча
    void FillBufferFromPatch(int interface_id,
                             std::shared_ptr<IPatch> patch,
                             const std::vector<FieldType>& fields);

    /// Скопировать из буфера интерфейса в патч
    void CopyBufferToPatch(int interface_id,
                           std::shared_ptr<IPatch> patch,
                           const std::vector<FieldType>& fields);

    /// Применить одну карту связей (интерполяция через CouplingMap)
    void ApplyCoupling(std::shared_ptr<CouplingMap> coupling,
                       const std::vector<FieldType>& fields);

    /// Найти интерфейс по ID (из кеша, без UB)
    const PatchInterface* FindInterface(int interface_id) const;
};

#endif // EXCHANGE_ENGINE_H
