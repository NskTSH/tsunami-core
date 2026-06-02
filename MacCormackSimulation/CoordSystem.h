// CoordSystem.h
// Перечисление типов систем координат для симуляции

#ifndef COORD_SYSTEM_H
#define COORD_SYSTEM_H

#include <string>
#include <stdexcept>

// Перечисление систем координат
enum class CoordSystem {
    Cartesian,   // Декартова система координат
    Spherical    // Сферическая система координат
};

// Направление разбивки домена на полосы (SubGrid)
enum class DecompositionDirection {
    Horizontal,  // Разбивка по X (горизонтальные полосы) — по умолчанию
    Vertical     // Разбивка по Y (вертикальные полосы)
};

// Вспомогательные функции для работы с CoordSystem

// Преобразование строки в CoordSystem
inline CoordSystem StringToCoordSystem(const std::string& str) {
    if (str == "Cartesian" || str == "cartesian" || str == "0") {
        return CoordSystem::Cartesian;
    }
    else if (str == "Spherical" || str == "spherical" || str == "1") {
        return CoordSystem::Spherical;
    }
    throw std::runtime_error("Unknown coordinate system: " + str);
}

// Преобразование CoordSystem в строку
inline std::string CoordSystemToString(CoordSystem cs) {
    switch (cs) {
        case CoordSystem::Cartesian: return "Cartesian";
        case CoordSystem::Spherical: return "Spherical";
        default: return "Unknown";
    }
}

#endif // COORD_SYSTEM_H
