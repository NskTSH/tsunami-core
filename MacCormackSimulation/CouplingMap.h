// CouplingMap.h - Описание связей между интерфейсами патчей
// Определяет как данные передаются от одного интерфейса к другому

#ifndef COUPLING_MAP_H
#define COUPLING_MAP_H

#include "PatchTypes.h"
#include <vector>
#include <string>

/// Точка связи между двумя интерфейсами
struct CouplingPoint {
    int source_index;      // Индекс точки в исходном интерфейсе
    int target_index;      // Индекс точки в целевом интерфейсе
    double weight;         // Вес интерполяции (1.0 для точного совпадения)

    CouplingPoint() : source_index(-1), target_index(-1), weight(0.0) {}

    CouplingPoint(int src, int tgt, double w)
        : source_index(src), target_index(tgt), weight(w) {}
};

/// Карта связей между двумя интерфейсами
///
/// Описывает как данные передаются между двумя интерфейсами:
/// - Геометрическое соответствие (какие точки связаны)
/// - Веса интерполяции (для несовпадающих сеток)
/// - Типы полей для обмена
///
/// Поддерживает двунаправленный обмен (A→B и B→A)
class CouplingMap {
public:
    /// Конструктор
    /// @param interface_a_id ID первого интерфейса
    /// @param interface_b_id ID второго интерфейса
    CouplingMap(int interface_a_id, int interface_b_id);

    /// Добавить связь от A к B
    /// @param src_idx Индекс точки в интерфейсе A
    /// @param tgt_idx Индекс точки в интерфейсе B
    /// @param weight Вес интерполяции
    void AddCouplingAtoB(int src_idx, int tgt_idx, double weight);

    /// Добавить связь от B к A
    /// @param src_idx Индекс точки в интерфейсе B
    /// @param tgt_idx Индекс точки в интерфейсе A
    /// @param weight Вес интерполяции
    void AddCouplingBtoA(int src_idx, int tgt_idx, double weight);

    /// Добавить симметричную связь (одинаковые индексы и веса в обе стороны)
    /// Используется для one-to-one mapping
    void AddSymmetricCoupling(int idx_a, int idx_b, double weight = 1.0);

    /// Получить ID интерфейса A
    int GetInterfaceAId() const { return interface_a_id_; }

    /// Получить ID интерфейса B
    int GetInterfaceBId() const { return interface_b_id_; }

    /// Получить связи A→B
    const std::vector<CouplingPoint>& GetCouplingAtoB() const { return a_to_b_; }

    /// Получить связи B→A
    const std::vector<CouplingPoint>& GetCouplingBtoA() const { return b_to_a_; }

    /// Добавить тип поля для обмена
    void AddFieldType(FieldType field) {
        fields_to_exchange_.push_back(field);
    }

    /// Получить список полей для обмена
    const std::vector<FieldType>& GetFieldsToExchange() const {
        return fields_to_exchange_;
    }

    /// Передать данные от A к B
    /// @param source_data Массив данных интерфейса A
    /// @param target_data Массив данных интерфейса B (выход)
    void TransferAtoB(const double* source_data, double* target_data) const;

    /// Передать данные от B к A
    /// @param source_data Массив данных интерфейса B
    /// @param target_data Массив данных интерфейса A (выход)
    void TransferBtoA(const double* source_data, double* target_data) const;

    /// Получить количество связей A→B
    size_t GetNumCouplingAtoB() const { return a_to_b_.size(); }

    /// Получить количество связей B→A
    size_t GetNumCouplingBtoA() const { return b_to_a_.size(); }

private:
    int interface_a_id_;               // ID интерфейса A
    int interface_b_id_;               // ID интерфейса B

    std::vector<CouplingPoint> a_to_b_;  // Связи A → B
    std::vector<CouplingPoint> b_to_a_;  // Связи B → A

    std::vector<FieldType> fields_to_exchange_;  // Поля для обмена

    /// Вспомогательная функция передачи данных
    void Transfer(const double* source_data, double* target_data,
                  const std::vector<CouplingPoint>& couplings) const;
};

#endif // COUPLING_MAP_H
