// DiagnosticTools.cpp - Реализация инструментов диагностики

#include "DiagnosticTools.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <filesystem>

void DiagnosticTools::ExportInterfaceGeometry(const PatchInterface& iface,
                                              const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // Заголовок
    out << "# Interface " << iface.interface_id
        << " of patch " << iface.patch_id_owner << "\n";
    out << "# boundary: " << iface.boundary_name << "\n";
    out << "# num_points: " << iface.points.size() << "\n";
    out << "# Format: x y local_index\n";

    // Данные
    out << std::fixed << std::setprecision(10);
    for (const auto& pt : iface.points) {
        out << pt.x << " " << pt.y << " " << pt.local_index << "\n";
    }

    out.close();
}

void DiagnosticTools::ExportPatchInterfaces(std::shared_ptr<IPatch> patch,
                                            const std::string& base_filename) {
    auto interfaces = patch->GetInterfaces();
    int patch_id = patch->GetPatchId();

    for (size_t i = 0; i < interfaces.size(); ++i) {
        std::string filename = base_filename + "_patch" + std::to_string(patch_id) +
                               "_iface" + std::to_string(i) + ".txt";
        ExportInterfaceGeometry(interfaces[i], filename);
    }
}

void DiagnosticTools::ExportDomainInterfaces(std::shared_ptr<HybridDomain> domain,
                                             const std::string& output_dir) {
    CreateDirectoryIfNeeded(output_dir);

    const auto& patches = domain->GetPatches();
    for (const auto& patch : patches) {
        std::string base_filename = output_dir + "/interface";
        ExportPatchInterfaces(patch, base_filename);
    }

    std::cout << "Exported interfaces for " << patches.size()
              << " patches to " << output_dir << "/" << std::endl;
}

void DiagnosticTools::ExportCouplingMap(std::shared_ptr<CouplingMap> coupling,
                                        const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // Заголовок
    out << "# CouplingMap between interfaces "
        << coupling->GetInterfaceAId() << " and "
        << coupling->GetInterfaceBId() << "\n";
    out << "# Direction: A → B\n";
    out << "# Format: source_idx target_idx weight\n";

    // A → B
    const auto& a_to_b = coupling->GetCouplingAtoB();
    for (const auto& cp : a_to_b) {
        out << cp.source_index << " " << cp.target_index << " "
            << std::fixed << std::setprecision(6) << cp.weight << "\n";
    }

    out << "\n# Direction: B → A\n";
    out << "# Format: source_idx target_idx weight\n";

    // B → A
    const auto& b_to_a = coupling->GetCouplingBtoA();
    for (const auto& cp : b_to_a) {
        out << cp.source_index << " " << cp.target_index << " "
            << std::fixed << std::setprecision(6) << cp.weight << "\n";
    }

    out.close();
}

void DiagnosticTools::ExportDomainCouplings(std::shared_ptr<HybridDomain> domain,
                                            const std::string& output_dir) {
    CreateDirectoryIfNeeded(output_dir);

    auto engine = domain->GetExchangeEngine();
    size_t num_couplings = engine->GetNumCouplings();

    // Примечание: ExchangeEngine не предоставляет прямого доступа к couplings
    // Для полной реализации нужно добавить метод GetCouplings() в ExchangeEngine

    std::cout << "Domain has " << num_couplings << " couplings" << std::endl;
    std::cout << "Note: Full coupling export requires ExchangeEngine::GetCouplings() method" << std::endl;
}

DiagnosticTools::CouplingMetrics DiagnosticTools::ComputeCouplingMetrics(
    const PatchInterface& iface_a,
    const PatchInterface& iface_b,
    std::shared_ptr<CouplingMap> coupling) {

    CouplingMetrics metrics;
    metrics.avg_distance = 0.0;
    metrics.max_distance = 0.0;
    metrics.num_unmapped_a = 0;
    metrics.num_unmapped_b = 0;

    const auto& a_to_b = coupling->GetCouplingAtoB();
    const auto& b_to_a = coupling->GetCouplingBtoA();

    metrics.total_couplings_a_to_b = static_cast<int>(a_to_b.size());
    metrics.total_couplings_b_to_a = static_cast<int>(b_to_a.size());

    // Вычислить средние и максимальные дистанции для A → B
    double sum_dist = 0.0;
    for (const auto& cp : a_to_b) {
        if (cp.source_index >= 0 && cp.source_index < static_cast<int>(iface_a.points.size()) &&
            cp.target_index >= 0 && cp.target_index < static_cast<int>(iface_b.points.size())) {

            double dist = Distance(iface_a.points[cp.source_index],
                                   iface_b.points[cp.target_index]);
            sum_dist += dist;
            metrics.max_distance = std::max(metrics.max_distance, dist);
        }
    }

    if (!a_to_b.empty()) {
        metrics.avg_distance = sum_dist / a_to_b.size();
    }

    // Подсчитать несопоставленные точки
    std::vector<bool> mapped_a(iface_a.points.size(), false);
    std::vector<bool> mapped_b(iface_b.points.size(), false);

    for (const auto& cp : a_to_b) {
        if (cp.source_index >= 0 && cp.source_index < static_cast<int>(mapped_a.size())) {
            mapped_a[cp.source_index] = true;
        }
    }

    for (const auto& cp : b_to_a) {
        if (cp.source_index >= 0 && cp.source_index < static_cast<int>(mapped_b.size())) {
            mapped_b[cp.source_index] = true;
        }
    }

    metrics.num_unmapped_a = static_cast<int>(
        std::count(mapped_a.begin(), mapped_a.end(), false));
    metrics.num_unmapped_b = static_cast<int>(
        std::count(mapped_b.begin(), mapped_b.end(), false));

    return metrics;
}

void DiagnosticTools::PrintCouplingMetrics(const CouplingMetrics& metrics,
                                           const std::string& coupling_name) {
    std::cout << "===== Coupling Metrics";
    if (!coupling_name.empty()) {
        std::cout << ": " << coupling_name;
    }
    std::cout << " =====" << std::endl;

    std::cout << "  Couplings A→B: " << metrics.total_couplings_a_to_b << std::endl;
    std::cout << "  Couplings B→A: " << metrics.total_couplings_b_to_a << std::endl;
    std::cout << "  Average distance: " << std::scientific
              << std::setprecision(6) << metrics.avg_distance << std::endl;
    std::cout << "  Max distance: " << metrics.max_distance << std::endl;
    std::cout << "  Unmapped in A: " << metrics.num_unmapped_a << std::endl;
    std::cout << "  Unmapped in B: " << metrics.num_unmapped_b << std::endl;
    std::cout << "===============================" << std::endl;
}

bool DiagnosticTools::TestExchangeOnly(std::shared_ptr<HybridDomain> domain,
                                       const std::vector<FieldType>& fields) {
    auto& patches = domain->GetPatches();
    auto engine = domain->GetExchangeEngine();

    // Инициализировать патчи тестовыми данными
    for (size_t i = 0; i < patches.size(); ++i) {
        Real* H = patches[i]->GetFieldData(FieldType::H);
        if (H) {
            int nx, ny;
            patches[i]->GetLocalSize(nx, ny);
            double test_value = 10.0 + i;  // Разные значения для разных патчей

            for (int idx = 0; idx < nx * ny; ++idx) {
                H[idx] = test_value;
            }
        }
    }

    // Прогнать обмен
    try {
        engine->ExchangeFields(0.0, 0.0, fields);
        std::cout << "Exchange test completed successfully!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exchange test failed: " << e.what() << std::endl;
        return false;
    }
}

bool DiagnosticTools::CheckDomainConsistency(std::shared_ptr<HybridDomain> domain) {
    const auto& patches = domain->GetPatches();

    std::cout << "Checking domain consistency..." << std::endl;
    std::cout << "  Patches: " << patches.size() << std::endl;

    int total_interfaces = 0;
    for (const auto& patch : patches) {
        auto interfaces = patch->GetInterfaces();
        total_interfaces += static_cast<int>(interfaces.size());
    }

    std::cout << "  Total interfaces: " << total_interfaces << std::endl;
    std::cout << "  Couplings: " << domain->GetExchangeEngine()->GetNumCouplings() << std::endl;

    // Проверка: количество coupling должно быть <= total_interfaces / 2
    // (так как каждая пара интерфейсов дает один coupling)
    bool consistent = domain->GetExchangeEngine()->GetNumCouplings() <=
                      static_cast<size_t>(total_interfaces / 2);

    if (consistent) {
        std::cout << "  Status: CONSISTENT" << std::endl;
    } else {
        std::cout << "  Status: INCONSISTENT (too many couplings)" << std::endl;
    }

    return consistent;
}

// ========== PRIVATE ==========

double DiagnosticTools::Distance(const InterfacePoint& p1, const InterfacePoint& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

void DiagnosticTools::CreateDirectoryIfNeeded(const std::string& dir) {
    try {
        std::filesystem::create_directories(dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create directory " << dir
                  << ": " << e.what() << std::endl;
    }
}
