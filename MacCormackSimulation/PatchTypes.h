// PatchTypes.h - Базовые типы для гибридной архитектуры патчей
// Определяет структуры для описания интерфейсов между патчами и типов полей

#ifndef PATCH_TYPES_H
#define PATCH_TYPES_H

#include <vector>
#include <string>

// ========== ТИПЫ ПОЛЕЙ ==========

/// Перечисление типов физических полей для обмена между патчами
enum class FieldType {
    H,        // Полная глубина воды
    U,        // Горизонтальная скорость (X-направление)
    V,        // Вертикальная скорость (Y-направление)
    Eta,      // Высота свободной поверхности
    D,        // Глубина дна (батиметрия, обычно статическая)

    // Hat-поля для двухшаговой схемы MacCormack
    H_hat,    // Промежуточное значение H
    U_hat,    // Промежуточное значение u
    V_hat,    // Промежуточное значение v
    Eta_hat   // Промежуточное значение eta
};

// ========== ГЕОМЕТРИЯ ИНТЕРФЕЙСОВ ==========

/// Точка на интерфейсе патча (граничная точка для обмена данными)
struct InterfacePoint {
    double x;              // Физическая координата X
    double y;              // Физическая координата Y
    int local_index;       // Локальный индекс для ЧТЕНИЯ данных из патча
    int write_index;       // Локальный индекс для ЗАПИСИ данных в патч (ghost cell)
                           // Если == -1, используется local_index для записи (default)

    InterfacePoint() : x(0.0), y(0.0), local_index(-1), write_index(-1) {}

    InterfacePoint(double x_, double y_, int idx)
        : x(x_), y(y_), local_index(idx), write_index(-1) {}

    InterfacePoint(double x_, double y_, int read_idx, int write_idx)
        : x(x_), y(y_), local_index(read_idx), write_index(write_idx) {}

    /// Индекс для записи (ghost cell). Если не задан, fallback на local_index.
    int GetWriteIndex() const { return write_index >= 0 ? write_index : local_index; }
};

/// Интерфейс патча - описание граничной области для обмена с соседними патчами
struct PatchInterface {
    int interface_id;           // Уникальный идентификатор интерфейса
    int patch_id_owner;         // ID патча-владельца этого интерфейса
    std::string boundary_name;  // Название границы: "left", "right", "top", "bottom"

    /// Набор точек на интерфейсе (упорядоченные вдоль границы)
    std::vector<InterfacePoint> points;

    PatchInterface() : interface_id(-1), patch_id_owner(-1), boundary_name("") {}

    PatchInterface(int iface_id, int owner_id, const std::string& name)
        : interface_id(iface_id), patch_id_owner(owner_id), boundary_name(name) {}

    /// Добавить точку к интерфейсу
    void AddPoint(double x, double y, int local_idx) {
        points.emplace_back(x, y, local_idx);
    }

    /// Количество точек на интерфейсе
    size_t GetNumPoints() const {
        return points.size();
    }
};

// ========== ФАЗЫ ОБМЕНА ==========

/// Фаза обмена данными (для MacCormack двухшаговой схемы)
enum class ExchangePhase {
    AfterHalfStep,    // Обмен hat-полями после половинного шага
    AfterFullStep     // Обмен основными полями после полного шага
};

#endif // PATCH_TYPES_H
