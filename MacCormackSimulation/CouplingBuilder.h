// CouplingBuilder.h - Построение карт связей между интерфейсами
// Автоматически создает CouplingMap на основе геометрии интерфейсов

#ifndef COUPLING_BUILDER_H
#define COUPLING_BUILDER_H

#include "CouplingMap.h"
#include "PatchTypes.h"
#include <memory>

/// Построитель карт связей между интерфейсами
///
/// Анализирует геометрию двух интерфейсов и создает CouplingMap
/// с соответствующими связями и весами интерполяции
class CouplingBuilder {
public:
    /// Типы mapping'а между интерфейсами
    enum class MappingType {
        OneToOne,      // Точное совпадение индексов (для совпадающих сеток)
        Nearest,       // Ближайший сосед (для несовпадающих сеток)
        Linear         // Линейная интерполяция вдоль интерфейса
    };

    /// Построить связь между двумя structured интерфейсами
    ///
    /// @param iface_a Первый интерфейс
    /// @param iface_b Второй интерфейс
    /// @param type Тип mapping'а
    /// @return Карта связей между интерфейсами
    static std::shared_ptr<CouplingMap> BuildStructuredToStructured(
        const PatchInterface& iface_a,
        const PatchInterface& iface_b,
        MappingType type = MappingType::OneToOne
    );

    /// Построить связь между structured и mesh интерфейсами
    /// (пока заглушка для будущей реализации)
    static std::shared_ptr<CouplingMap> BuildStructuredToMesh(
        const PatchInterface& structured_iface,
        const PatchInterface& mesh_iface,
        MappingType type = MappingType::Nearest
    );

    /// Построить связь между двумя mesh интерфейсами
    /// (пока заглушка для будущей реализации)
    static std::shared_ptr<CouplingMap> BuildMeshToMesh(
        const PatchInterface& iface_a,
        const PatchInterface& iface_b,
        MappingType type = MappingType::Nearest
    );

    /// Проверить совместимость двух интерфейсов
    /// (могут ли они быть связаны)
    static bool AreCompatible(const PatchInterface& iface_a,
                              const PatchInterface& iface_b);

private:
    /// Построить OneToOne mapping (индекс к индексу с весом 1.0)
    static void BuildOneToOneMapping(CouplingMap& map,
                                     const PatchInterface& iface_a,
                                     const PatchInterface& iface_b);

    /// Построить Nearest neighbor mapping
    static void BuildNearestMapping(CouplingMap& map,
                                    const PatchInterface& iface_a,
                                    const PatchInterface& iface_b);

    /// Построить Linear interpolation mapping
    static void BuildLinearMapping(CouplingMap& map,
                                   const PatchInterface& iface_a,
                                   const PatchInterface& iface_b);

    /// Вычислить расстояние между двумя точками
    static double Distance(const InterfacePoint& p1, const InterfacePoint& p2);
};

#endif // COUPLING_BUILDER_H
