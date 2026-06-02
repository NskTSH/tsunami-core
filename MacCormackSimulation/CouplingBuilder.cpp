// CouplingBuilder.cpp - Реализация построителя карт связей

#include "CouplingBuilder.h"
#include <cmath>
#include <algorithm>
#include <limits>

std::shared_ptr<CouplingMap> CouplingBuilder::BuildStructuredToStructured(
    const PatchInterface& iface_a,
    const PatchInterface& iface_b,
    MappingType type) {

    auto map = std::make_shared<CouplingMap>(iface_a.interface_id, iface_b.interface_id);

    // Выбор типа построения связей
    switch (type) {
    case MappingType::OneToOne:
        BuildOneToOneMapping(*map, iface_a, iface_b);
        break;

    case MappingType::Nearest:
        BuildNearestMapping(*map, iface_a, iface_b);
        break;

    case MappingType::Linear:
        BuildLinearMapping(*map, iface_a, iface_b);
        break;
    }

    // Добавить стандартные поля для обмена (основные + hat)
    map->AddFieldType(FieldType::H);
    map->AddFieldType(FieldType::U);
    map->AddFieldType(FieldType::V);
    map->AddFieldType(FieldType::Eta);
    map->AddFieldType(FieldType::H_hat);
    map->AddFieldType(FieldType::U_hat);
    map->AddFieldType(FieldType::V_hat);
    map->AddFieldType(FieldType::Eta_hat);

    return map;
}

std::shared_ptr<CouplingMap> CouplingBuilder::BuildStructuredToMesh(
    const PatchInterface& structured_iface,
    const PatchInterface& mesh_iface,
    MappingType type) {

    auto map = std::make_shared<CouplingMap>(
        structured_iface.interface_id,
        mesh_iface.interface_id
    );

    // Build mapping based on type
    switch (type) {
    case MappingType::OneToOne:
        BuildOneToOneMapping(*map, structured_iface, mesh_iface);
        break;
    case MappingType::Linear:
        BuildLinearMapping(*map, structured_iface, mesh_iface);
        break;
    case MappingType::Nearest:
    default:
        BuildNearestMapping(*map, structured_iface, mesh_iface);
        break;
    }

    // Register standard field types for exchange (same as S2S)
    map->AddFieldType(FieldType::H);
    map->AddFieldType(FieldType::U);
    map->AddFieldType(FieldType::V);
    map->AddFieldType(FieldType::Eta);
    map->AddFieldType(FieldType::H_hat);
    map->AddFieldType(FieldType::U_hat);
    map->AddFieldType(FieldType::V_hat);
    map->AddFieldType(FieldType::Eta_hat);

    return map;
}

std::shared_ptr<CouplingMap> CouplingBuilder::BuildMeshToMesh(
    const PatchInterface& iface_a,
    const PatchInterface& iface_b,
    MappingType type) {

    // Пока заглушка - будет реализовано позже
    return std::make_shared<CouplingMap>(iface_a.interface_id, iface_b.interface_id);
}

bool CouplingBuilder::AreCompatible(const PatchInterface& iface_a,
                                    const PatchInterface& iface_b) {
    // Базовая проверка: должны быть точки в обоих интерфейсах
    if (iface_a.points.empty() || iface_b.points.empty()) {
        return false;
    }

    // Для structured↔structured: количество точек должно совпадать (для OneToOne)
    // Но для других типов mapping это не обязательно
    return true;
}

// ========== PRIVATE: Построение разных типов mapping ==========

void CouplingBuilder::BuildOneToOneMapping(CouplingMap& map,
                                           const PatchInterface& iface_a,
                                           const PatchInterface& iface_b) {
    // OneToOne: предполагаем что индексы точно совпадают
    // Это случай для соседних structured патчей с одинаковой дискретизацией

    size_t num_points = std::min(iface_a.points.size(), iface_b.points.size());

    for (size_t i = 0; i < num_points; ++i) {
        // Симметричная связь: i↔i с весом 1.0
        map.AddSymmetricCoupling(static_cast<int>(i), static_cast<int>(i), 1.0);
    }
}

void CouplingBuilder::BuildNearestMapping(CouplingMap& map,
                                          const PatchInterface& iface_a,
                                          const PatchInterface& iface_b) {
    // TARGET-DRIVEN nearest neighbor mapping.
    // Each direction iterates over TARGET points to ensure every target gets a value.
    // This is critical when interfaces have different sizes (e.g., 4801 structured vs 300 DG).

    // A → B: source-driven phase. Each A source picks its nearest B target;
    // multiple sources to the same target are averaged via 1/count weights.
    // This alone is not enough — when |B| ≥ |A| (or even ≈ |A| with a 0.5*dy
    // staggering, as happens for StructuredPatch points at j*dy vs DG edge
    // midpoints at (j+0.5)*dy), some B targets receive NO source and their
    // incoming buffer stays at its default zero. DG's HLL then sees H=0
    // ghosts on those edges and produces a dam-break flux against the rest
    // state, draining DG H by tens of metres per step (observed in
    // HybridFlatTest: H_min crashes from 1000 → 967 in one step, → 490 by
    // step 100). The target-driven fallback below patches every unfilled
    // B target with its nearest A source so no coupled edge is left with
    // a zero ghost.
    {
        std::vector<int> a_to_b(iface_a.points.size(), -1);
        std::vector<int> count_per_b(iface_b.points.size(), 0);

        for (size_t i = 0; i < iface_a.points.size(); ++i) {
            int nearest = -1;
            double min_dist = std::numeric_limits<double>::max();
            for (size_t j = 0; j < iface_b.points.size(); ++j) {
                double dist = Distance(iface_a.points[i], iface_b.points[j]);
                if (dist < min_dist) { min_dist = dist; nearest = static_cast<int>(j); }
            }
            a_to_b[i] = nearest;
            if (nearest >= 0) count_per_b[nearest]++;
        }
        for (size_t i = 0; i < iface_a.points.size(); ++i) {
            int tgt = a_to_b[i];
            if (tgt >= 0 && count_per_b[tgt] > 0)
                map.AddCouplingAtoB(static_cast<int>(i), tgt, 1.0 / count_per_b[tgt]);
        }

        // Target-driven backfill: every B point that did not receive any
        // A source above must be paired with its nearest A source. Weight 1
        // since exactly one source supplies it.
        for (size_t j = 0; j < iface_b.points.size(); ++j) {
            if (count_per_b[j] > 0) continue;
            int nearest_a = -1;
            double min_dist = std::numeric_limits<double>::max();
            for (size_t i = 0; i < iface_a.points.size(); ++i) {
                double dist = Distance(iface_a.points[i], iface_b.points[j]);
                if (dist < min_dist) { min_dist = dist; nearest_a = static_cast<int>(i); }
            }
            if (nearest_a >= 0) {
                map.AddCouplingAtoB(nearest_a, static_cast<int>(j), 1.0);
            }
        }
    }

    // B → A: for each TARGET point in A, find nearest SOURCE in B
    // This ensures every A point receives data (critical when A is larger/structured).
    for (size_t i = 0; i < iface_a.points.size(); ++i) {
        const auto& pt_a = iface_a.points[i];
        int nearest_b = -1;
        double min_dist = std::numeric_limits<double>::max();

        for (size_t j = 0; j < iface_b.points.size(); ++j) {
            double dist = Distance(pt_a, iface_b.points[j]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_b = static_cast<int>(j);
            }
        }
        if (nearest_b >= 0) {
            map.AddCouplingBtoA(nearest_b, static_cast<int>(i), 1.0);
        }
    }
}

void CouplingBuilder::BuildLinearMapping(CouplingMap& map,
                                         const PatchInterface& iface_a,
                                         const PatchInterface& iface_b) {
    // Linear interpolation: для каждой точки в A найти два ближайших в B
    // и вычислить веса линейной интерполяции
    // Пока используем nearest для простоты (полная реализация позже)

    BuildNearestMapping(map, iface_a, iface_b);
}

double CouplingBuilder::Distance(const InterfacePoint& p1, const InterfacePoint& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}
