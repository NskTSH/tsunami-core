// DiagnosticTools.h - Инструменты отладки и визуализации для гибридной архитектуры
// Экспорт геометрии интерфейсов, coupling map, метрики качества

#ifndef DIAGNOSTIC_TOOLS_H
#define DIAGNOSTIC_TOOLS_H

#include "PatchTypes.h"
#include "CouplingMap.h"
#include "HybridDomain.h"
#include <string>
#include <memory>

/// Инструменты диагностики для отладки гибридной архитектуры
///
/// Предоставляет методы для:
/// - Экспорта геометрии интерфейсов в текстовые файлы
/// - Экспорта coupling map для визуализации
/// - Вычисления метрик качества сопряжения
/// - Тестирования обмена данными
class DiagnosticTools {
public:
    // ========== ЭКСПОРТ ГЕОМЕТРИИ ==========

    /// Экспортировать геометрию интерфейса в текстовый файл
    /// Формат: x y local_index
    ///
    /// @param iface Интерфейс для экспорта
    /// @param filename Имя выходного файла
    static void ExportInterfaceGeometry(const PatchInterface& iface,
                                        const std::string& filename);

    /// Экспортировать все интерфейсы патча
    ///
    /// @param patch Патч
    /// @param base_filename Базовое имя файла (будет добавлен _iface_N.txt)
    static void ExportPatchInterfaces(std::shared_ptr<IPatch> patch,
                                      const std::string& base_filename);

    /// Экспортировать все интерфейсы домена
    ///
    /// @param domain Домен
    /// @param output_dir Директория для вывода (по умолчанию "diagnostics/")
    static void ExportDomainInterfaces(std::shared_ptr<HybridDomain> domain,
                                       const std::string& output_dir = "diagnostics/");

    // ========== ЭКСПОРТ COUPLING MAP ==========

    /// Экспортировать coupling map в текстовый файл
    /// Формат: source_idx target_idx weight
    ///
    /// @param coupling CouplingMap для экспорта
    /// @param filename Имя выходного файла
    static void ExportCouplingMap(std::shared_ptr<CouplingMap> coupling,
                                  const std::string& filename);

    /// Экспортировать все coupling map домена
    ///
    /// @param domain Домен
    /// @param output_dir Директория для вывода
    static void ExportDomainCouplings(std::shared_ptr<HybridDomain> domain,
                                      const std::string& output_dir = "diagnostics/");

    // ========== МЕТРИКИ КАЧЕСТВА ==========

    /// Вычислить метрики качества coupling map
    ///
    /// @param iface_a Первый интерфейс
    /// @param iface_b Второй интерфейс
    /// @param coupling CouplingMap между ними
    /// @return Структура с метриками
    struct CouplingMetrics {
        double avg_distance;      // Средняя дистанция между связанными точками
        double max_distance;      // Максимальная дистанция
        int num_unmapped_a;       // Количество несопоставленных точек в A
        int num_unmapped_b;       // Количество несопоставленных точек в B
        int total_couplings_a_to_b;  // Общее количество связей A→B
        int total_couplings_b_to_a;  // Общее количество связей B→A
    };

    static CouplingMetrics ComputeCouplingMetrics(
        const PatchInterface& iface_a,
        const PatchInterface& iface_b,
        std::shared_ptr<CouplingMap> coupling
    );

    /// Напечатать метрики в консоль
    static void PrintCouplingMetrics(const CouplingMetrics& metrics,
                                     const std::string& coupling_name = "");

    // ========== ТЕСТИРОВАНИЕ ОБМЕНА ==========

    /// Протестировать обмен данными без вычислений
    /// Инициализирует поля тестовыми значениями и прогоняет exchange
    ///
    /// @param domain Домен для тестирования
    /// @param fields Поля для обмена
    /// @return true если обмен прошел успешно
    static bool TestExchangeOnly(std::shared_ptr<HybridDomain> domain,
                                 const std::vector<FieldType>& fields);

    /// Проверить консистентность интерфейсов домена
    /// (все ли интерфейсы корректно сопоставлены)
    ///
    /// @param domain Домен для проверки
    /// @return true если все интерфейсы консистентны
    static bool CheckDomainConsistency(std::shared_ptr<HybridDomain> domain);

private:
    /// Вычислить дистанцию между двумя точками
    static double Distance(const InterfacePoint& p1, const InterfacePoint& p2);

    /// Создать директорию если не существует
    static void CreateDirectoryIfNeeded(const std::string& dir);
};

#endif // DIAGNOSTIC_TOOLS_H
