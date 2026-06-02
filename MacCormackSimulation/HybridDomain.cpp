// HybridDomain.cpp - Реализация контейнера патчей

#include "HybridDomain.h"
#include "StructuredPatch.h"
#include "MeshPatch.h"
#include <cmath>
#include <algorithm>

HybridDomain::HybridDomain()
    : exchange_engine_(std::make_shared<ExchangeEngine>()),
      total_nx_(0),
      total_ny_(0) {
}

void HybridDomain::AddPatch(std::shared_ptr<IPatch> patch) {
    patches_.push_back(patch);

    // Автоматически зарегистрировать в ExchangeEngine
    exchange_engine_->RegisterPatch(patch->GetPatchId(), patch);
}

std::shared_ptr<IPatch> HybridDomain::GetPatch(int patch_id) {
    for (auto& patch : patches_) {
        if (patch->GetPatchId() == patch_id) {
            return patch;
        }
    }
    return nullptr;
}

void HybridDomain::BuildCouplings() {
    // Автоматическое построение связей между всеми парами патчей
    int total_couplings = 0;
    std::cerr << "BuildCouplings: " << patches_.size() << " patches" << std::endl;
    for (size_t i = 0; i < patches_.size(); ++i) {
        for (size_t j = i + 1; j < patches_.size(); ++j) {
            auto matching_interfaces = FindMatchingInterfaces(patches_[i], patches_[j]);
            if (!matching_interfaces.empty()) {
                std::cerr << "  Matched " << matching_interfaces.size()
                          << " interface(s) between patch " << patches_[i]->GetPatchId()
                          << " and patch " << patches_[j]->GetPatchId() << std::endl;
            }

            bool is_mesh_i = (dynamic_cast<MeshPatch*>(patches_[i].get()) != nullptr);
            bool is_mesh_j = (dynamic_cast<MeshPatch*>(patches_[j].get()) != nullptr);

            for (const auto& [iface_a, iface_b] : matching_interfaces) {
                std::shared_ptr<CouplingMap> coupling;

                if (is_mesh_i && is_mesh_j) {
                    coupling = CouplingBuilder::BuildMeshToMesh(
                        iface_a, iface_b, CouplingBuilder::MappingType::Nearest);
                } else if (is_mesh_i || is_mesh_j) {
                    // One structured, one mesh
                    const auto& s_iface = is_mesh_i ? iface_b : iface_a;
                    const auto& m_iface = is_mesh_i ? iface_a : iface_b;
                    coupling = CouplingBuilder::BuildStructuredToMesh(
                        s_iface, m_iface, CouplingBuilder::MappingType::Nearest);
                } else {
                    coupling = CouplingBuilder::BuildStructuredToStructured(
                        iface_a, iface_b, CouplingBuilder::MappingType::OneToOne);
                }

                AddCoupling(coupling);
                total_couplings++;
            }
        }
    }
    std::cerr << "BuildCouplings: created " << total_couplings << " coupling(s)" << std::endl;
}

void HybridDomain::AddCoupling(std::shared_ptr<CouplingMap> coupling) {
    exchange_engine_->RegisterCoupling(coupling);
}

void HybridDomain::Initialize() {
    // Инициализировать все патчи
    for (auto& patch : patches_) {
        patch->Initialize();
    }

    // Построить связи если еще не построены
    if (exchange_engine_->GetNumCouplings() == 0) {
        BuildCouplings();
    }
}

void HybridDomain::SetGlobalGrid(int nx, int ny) {
    global_grid_ = std::make_unique<Grid>(nx, ny);
    total_nx_ = nx;
    total_ny_ = ny;
}

void HybridDomain::SetGlobalGrid(Grid&& grid) {
    global_grid_ = std::make_unique<Grid>(std::move(grid));
    total_nx_ = global_grid_->GetNx();
    total_ny_ = global_grid_->GetNy();
}

void HybridDomain::CollectResultsToGlobalGrid() {
    if (!global_grid_) {
        return;  // Глобальная сетка не установлена
    }

    // Собрать результаты из всех патчей
    for (auto& patch : patches_) {
        patch->UpdateGlobalGrid(global_grid_.get());
    }
}

// ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

std::vector<std::pair<PatchInterface, PatchInterface>>
HybridDomain::FindMatchingInterfaces(std::shared_ptr<IPatch> patch_a,
                                     std::shared_ptr<IPatch> patch_b) {
    std::vector<std::pair<PatchInterface, PatchInterface>> matches;

    auto interfaces_a = patch_a->GetInterfaces();
    auto interfaces_b = patch_b->GetInterfaces();

    // Проверить все пары интерфейсов
    for (const auto& iface_a : interfaces_a) {
        for (const auto& iface_b : interfaces_b) {
            if (DoInterfacesMatch(iface_a, iface_b)) {
                matches.emplace_back(iface_a, iface_b);
            }
        }
    }

    return matches;
}

bool HybridDomain::DoInterfacesMatch(const PatchInterface& iface_a,
                                     const PatchInterface& iface_b) {
    if (iface_a.points.empty() || iface_b.points.empty()) {
        return false;
    }

    // Check opposite boundary names
    bool names_match =
        (iface_a.boundary_name == "left"   && iface_b.boundary_name == "right")  ||
        (iface_a.boundary_name == "right"  && iface_b.boundary_name == "left")   ||
        (iface_a.boundary_name == "top"    && iface_b.boundary_name == "bottom") ||
        (iface_a.boundary_name == "bottom" && iface_b.boundary_name == "top");

    if (!names_match) {
        return false;
    }


    // Geometric check: bounding box overlap along the interface direction.
    // Tolerance scales with domain size to handle floating-point accumulation.
    const double abs_tolerance = 1e-3;  // minimum absolute tolerance

    // Compute bounding boxes
    double min_ya = iface_a.points.front().y, max_ya = iface_a.points.front().y;
    double min_xa = iface_a.points.front().x, max_xa = iface_a.points.front().x;
    for (const auto& p : iface_a.points) {
        min_xa = std::min(min_xa, p.x); max_xa = std::max(max_xa, p.x);
        min_ya = std::min(min_ya, p.y); max_ya = std::max(max_ya, p.y);
    }

    double min_yb = iface_b.points.front().y, max_yb = iface_b.points.front().y;
    double min_xb = iface_b.points.front().x, max_xb = iface_b.points.front().x;
    for (const auto& p : iface_b.points) {
        min_xb = std::min(min_xb, p.x); max_xb = std::max(max_xb, p.x);
        min_yb = std::min(min_yb, p.y); max_yb = std::max(max_yb, p.y);
    }

    // For vertical interfaces (left/right): x-coords should be close, y-ranges should overlap
    bool is_vertical = (iface_a.boundary_name == "left" || iface_a.boundary_name == "right" ||
                        iface_b.boundary_name == "left" || iface_b.boundary_name == "right");

    if (is_vertical) {
        // X-coordinates of interface should be close.
        // For lat/lon grids, x varies with y (cos(lat) correction).
        // Compare x at the MEDIAN y-position, not the bounding box average.
        double median_y = 0.5 * (std::max(min_ya, min_yb) + std::min(max_ya, max_yb));

        // Find x near median_y on each interface
        auto find_x_at_y = [](const PatchInterface& iface, double target_y) -> double {
            double best_x = 0.5 * (iface.points.front().x + iface.points.back().x);
            double best_dist = 1e30;
            for (const auto& p : iface.points) {
                double dy = std::abs(p.y - target_y);
                if (dy < best_dist) {
                    best_dist = dy;
                    best_x = p.x;
                }
            }
            return best_x;
        };

        double x_a = find_x_at_y(iface_a, median_y);
        double x_b = find_x_at_y(iface_b, median_y);
        double tol = std::max(abs_tolerance, 1e-4 * std::max(std::abs(x_a), std::abs(x_b)));
        if (std::abs(x_a - x_b) > tol) return false;

        // Y-ranges should overlap significantly
        double overlap_min = std::max(min_ya, min_yb);
        double overlap_max = std::min(max_ya, max_yb);
        double range_a = max_ya - min_ya;
        double range_b = max_yb - min_yb;
        double min_range = std::min(range_a, range_b);
        if (min_range < 1e-30) return false;

        double overlap = overlap_max - overlap_min;
        return overlap > 0.5 * min_range; // At least 50% overlap
    } else {
        // Horizontal: Y close, X overlap
        double y_a = 0.5 * (min_ya + max_ya);
        double y_b = 0.5 * (min_yb + max_yb);
        double tol = std::max(abs_tolerance, 1e-6 * std::max(std::abs(y_a), std::abs(y_b)));
        if (std::abs(y_a - y_b) > tol) return false;

        double overlap_min = std::max(min_xa, min_xb);
        double overlap_max = std::min(max_xa, max_xb);
        double range_a = max_xa - min_xa;
        double range_b = max_xb - min_xb;
        double min_range = std::min(range_a, range_b);
        if (min_range < 1e-30) return false;

        double overlap = overlap_max - overlap_min;
        return overlap > 0.5 * min_range;
    }
}
