// SimulationRunner.cpp - Реализация главного цикла симуляции

#include "SimulationRunner.h"
#include "MeshPatch.h"
#include "NestedStructuredPatch.h"
#include "StructuredPatch.h"
#include "SubGrid.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>

namespace {

struct StructuredFluxTarget {
    std::shared_ptr<IPatch> patch;
    PatchInterface iface;
};

bool DoInterfacesMatchForFlux(const PatchInterface& iface_a,
                              const PatchInterface& iface_b) {
    if (iface_a.points.empty() || iface_b.points.empty()) {
        return false;
    }

    const bool names_match =
        (iface_a.boundary_name == "left"   && iface_b.boundary_name == "right")  ||
        (iface_a.boundary_name == "right"  && iface_b.boundary_name == "left")   ||
        (iface_a.boundary_name == "top"    && iface_b.boundary_name == "bottom") ||
        (iface_a.boundary_name == "bottom" && iface_b.boundary_name == "top");
    if (!names_match) {
        return false;
    }

    const double abs_tolerance = 1e-3;

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

    const bool is_vertical =
        (iface_a.boundary_name == "left" || iface_a.boundary_name == "right" ||
         iface_b.boundary_name == "left" || iface_b.boundary_name == "right");

    if (is_vertical) {
        const double median_y = 0.5 * (std::max(min_ya, min_yb) + std::min(max_ya, max_yb));
        auto find_x_at_y = [](const PatchInterface& iface, double target_y) -> double {
            double best_x = 0.5 * (iface.points.front().x + iface.points.back().x);
            double best_dist = 1e30;
            for (const auto& p : iface.points) {
                const double dy = std::abs(p.y - target_y);
                if (dy < best_dist) {
                    best_dist = dy;
                    best_x = p.x;
                }
            }
            return best_x;
        };

        const double x_a = find_x_at_y(iface_a, median_y);
        const double x_b = find_x_at_y(iface_b, median_y);
        const double tol = std::max(abs_tolerance, 1e-4 * std::max(std::abs(x_a), std::abs(x_b)));
        if (std::abs(x_a - x_b) > tol) return false;

        const double overlap_min = std::max(min_ya, min_yb);
        const double overlap_max = std::min(max_ya, max_yb);
        const double min_range = std::min(max_ya - min_ya, max_yb - min_yb);
        if (min_range < 1e-30) return false;
        return (overlap_max - overlap_min) > 0.5 * min_range;
    }

    const double y_a = 0.5 * (min_ya + max_ya);
    const double y_b = 0.5 * (min_yb + max_yb);
    const double tol = std::max(abs_tolerance, 1e-6 * std::max(std::abs(y_a), std::abs(y_b)));
    if (std::abs(y_a - y_b) > tol) return false;

    const double overlap_min = std::max(min_xa, min_xb);
    const double overlap_max = std::min(max_xa, max_xb);
    const double min_range = std::min(max_xa - min_xa, max_xb - min_xb);
    if (min_range < 1e-30) return false;
    return (overlap_max - overlap_min) > 0.5 * min_range;
}

bool FindStructuredFluxTarget(const PatchInterface& dg_iface,
                              const std::vector<std::shared_ptr<IPatch>>& patches,
                              int dg_patch_id,
                              StructuredFluxTarget& out_target) {
    for (const auto& patch : patches) {
        if (!patch || patch->GetPatchId() == dg_patch_id) {
            continue;
        }

        const bool is_structured =
            (dynamic_cast<StructuredPatch*>(patch.get()) != nullptr) ||
            (dynamic_cast<NestedStructuredPatch*>(patch.get()) != nullptr);
        if (!is_structured) {
            continue;
        }

        const auto interfaces = patch->GetInterfaces();
        for (const auto& iface : interfaces) {
            if (DoInterfacesMatchForFlux(dg_iface, iface)) {
                out_target.patch = patch;
                out_target.iface = iface;
                return true;
            }
        }
    }

    return false;
}

double InferNestedSpacing(const PatchInterface& iface,
                          const Grid& global_grid,
                          const NestedStructuredPatch& patch) {
    if (iface.points.size() >= 2) {
        if (iface.boundary_name == "left" || iface.boundary_name == "right") {
            return std::abs(iface.points[1].y - iface.points[0].y);
        }
        return std::abs(iface.points[1].x - iface.points[0].x);
    }

    if (iface.boundary_name == "left" || iface.boundary_name == "right") {
        return global_grid.GetDeltaY();
    }

    const int global_j =
        (iface.boundary_name == "bottom")
        ? patch.GetRectSubGrid()->GetGlobalJStart()
        : std::max(0, patch.GetRectSubGrid()->GetGlobalJEnd() - 1);
    return global_grid.GetDeltaXFunc()(patch.GetRectSubGrid()->GetGlobalIStart(), global_j);
}

}

SimulationRunner::SimulationRunner(std::shared_ptr<HybridDomain> domain,
                                   double tau,
                                   int timesteps,
                                   double g)
    : domain_(domain),
      g_(g),
      tau_(tau),
      timesteps_(timesteps),
      save_interval_(100),
      verbose_(true),
      save_function_(nullptr) {
}

void SimulationRunner::Run() {
    if (!domain_) {
        std::cerr << "Error: Domain is not set!" << std::endl;
        return;
    }

    auto engine = domain_->GetExchangeEngine();
    auto& patches = domain_->GetPatches();

    if (patches.empty()) {
        std::cerr << "Error: No patches in domain!" << std::endl;
        return;
    }

    // ========== CFL CHECK ==========
    if (domain_->HasGlobalGrid()) {
        const Grid& grid = domain_->GetGlobalGrid();
        int nx = grid.GetNx();
        int ny = grid.GetNy();
        double max_depth = 0.0;
        double min_dx = std::numeric_limits<double>::max();
        auto& dx_func = grid.GetDeltaXFunc();
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                double d = grid.D(i, j);
                if (d > max_depth) max_depth = d;
                double dx = dx_func(i, j);
                if (dx > 0.0 && dx < min_dx) min_dx = dx;
            }
        }
        double c = std::sqrt(g_ * max_depth);
        double dy = grid.GetDeltaY();
        double cfl = tau_ * (c / min_dx + c / dy);

        if (verbose_) {
            std::cout << "=== Simulation Parameters ===" << std::endl;
            std::cout << "  Patches: " << patches.size() << std::endl;
            std::cout << "  Couplings: " << engine->GetNumCouplings() << std::endl;
            std::cout << "  Grid: " << nx << " x " << ny << std::endl;
            std::cout << "  tau: " << tau_ << " s" << std::endl;
            std::cout << "  dx_min: " << min_dx << " m, dy: " << dy << " m" << std::endl;
            std::cout << "  Max depth: " << max_depth << " m" << std::endl;
            std::cout << "  Wave speed (sqrt(gH)): " << c << " m/s" << std::endl;
            std::cout << "  CFL: " << cfl << std::endl;
            std::cout << "  Timesteps: " << timesteps_ << std::endl;
            std::cout << "=============================" << std::endl;
        }
        if (cfl > 1.0) {
            std::cerr << "WARNING: CFL = " << cfl << " > 1.0 — simulation may be unstable!" << std::endl;
        }
    } else if (verbose_) {
        std::cout << "Starting simulation:" << std::endl;
        std::cout << "  Patches: " << patches.size() << std::endl;
        std::cout << "  Couplings: " << engine->GetNumCouplings() << std::endl;
        std::cout << "  Timesteps: " << timesteps_ << std::endl;
        std::cout << "  dt: " << tau_ << std::endl;
        std::cout << std::endl;
    }

    // ========== СИНХРОНИЗАЦИЯ DT ДЛЯ ВСЕХ ПАТЧЕЙ ==========
    // Определить минимальный стабильный dt по всем DG-патчам.
    // Если глобальный tau_ больше — автоматически уменьшить.
    for (auto& patch : patches) {
        auto mesh_patch = dynamic_cast<MeshPatch*>(patch.get());
        if (mesh_patch && mesh_patch->GetDGSolver().NumElements() > 0) {
            double dt_dg_stable = mesh_patch->GetDGSolver().ComputeTimeStep(0.2);
            if (tau_ > dt_dg_stable) {
                double old_tau = tau_;
                tau_ = dt_dg_stable;
                // Пересчитать количество шагов для сохранения физического времени
                double total_time = old_tau * timesteps_;
                timesteps_ = static_cast<int>(std::ceil(total_time / tau_));
                std::cerr << "[AUTO] Reduced tau: " << old_tau << " -> " << tau_
                          << " s (DG CFL, patch " << patch->GetPatchId()
                          << "). Timesteps: " << timesteps_ << std::endl;
            }
        }
    }

    // Установить синхронизированный tau_ во все патчи
    for (auto& patch : patches) {
        auto mesh_patch = dynamic_cast<MeshPatch*>(patch.get());
        if (mesh_patch) {
            mesh_patch->SetTimeStep(tau_);
        }
        auto struct_patch = dynamic_cast<StructuredPatch*>(patch.get());
        if (struct_patch) {
            struct_patch->GetSubGrid()->SetTau(tau_);
        }
        auto nested_patch = dynamic_cast<NestedStructuredPatch*>(patch.get());
        if (nested_patch) {
            nested_patch->GetRectSubGrid()->SetTau(tau_);
        }
    }

    // ========== INITIAL EXCHANGE ==========
    // Fill coupling buffers with initial data so DG has valid coupled states
    // before the first ComputeHalfStep. Without this, all coupled edges
    // fall back to OpenBC on the first step, causing mass loss.
    engine->ExchangeFieldsPhased(tau_, 0.0, ExchangePhase::AfterFullStep);
    for (auto& patch : patches) {
        patch->ApplyBoundaryConditions(ExchangePhase::AfterFullStep, 0.0);
    }

    // ========== ГЛАВНЫЙ ЦИКЛ ПО ВРЕМЕНИ ==========
    for (int t = 0; t < timesteps_; ++t) {
        double time = t * tau_;

        // ========== ПОЛОВИННЫЙ ШАГ (MacCormack) ==========
        for (auto& patch : patches) {
            patch->ComputeHalfStep();
        }

        // Обмен hat-полями после половинного шага
        engine->ExchangeFieldsPhased(tau_, time, ExchangePhase::AfterHalfStep);

        // Apply physical boundary conditions for hat fields (y-boundaries, external x-boundaries)
        for (auto& patch : patches) {
            patch->ApplyBoundaryConditions(ExchangePhase::AfterHalfStep, static_cast<double>(t));
        }

        // ========== ПОЛНЫЙ ШАГ (MacCormack) ==========
        for (auto& patch : patches) {
            patch->ComputeFullStep();
        }

        // Обмен основными полями после полного шага
        engine->ExchangeFieldsPhased(tau_, time, ExchangePhase::AfterFullStep);

        // Apply physical boundary conditions for main fields
        for (auto& patch : patches) {
            patch->ApplyBoundaryConditions(ExchangePhase::AfterFullStep, static_cast<double>(t));
        }

        // Hand DG's freshly-computed HLL fluxes to the adjacent MC stripes so
        // the next iteration's predictor and corrector both use the same
        // Φ_Γ on the interface (single common flux, opposite signs). Without
        // this, MC's FD flux and DG's HLL flux disagree by O(amp) every step
        // and the coupled HybridFlatTest amplifies the wave by ~3000×.
        // Set HYBRID_DISABLE_FLUX_COUPLING=1 to fall back to legacy state-only
        // coupling (useful for isolating DG-only divergence vs coupling bugs).
        {
            static int dis = -1;
            if (dis < 0) {
                char* buf = nullptr; size_t sz = 0;
                _dupenv_s(&buf, &sz, "HYBRID_DISABLE_FLUX_COUPLING");
                dis = (buf && *buf == '1') ? 1 : 0;
                if (buf) free(buf);
                if (dis) std::cerr << "[SimulationRunner] HYBRID_DISABLE_FLUX_COUPLING=1 active (no flux export)\n";
                else      std::cerr << "[SimulationRunner] HYBRID_DISABLE_FLUX_COUPLING=0 (flux export enabled)\n";
            }
            if (!dis) TransferDGFluxesToStructured();
        }

        // ========== ОБНОВЛЕНИЕ ГЛОБАЛЬНОЙ СЕТКИ ==========
        if (domain_->HasGlobalGrid()) {
            domain_->CollectResultsToGlobalGrid();
        }

        // ========== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
        if ((t + 1) % save_interval_ == 0) {
            SaveResults(t + 1);
        }

        // ========== FIRST-OFFENDER ЛОКАЛИЗАЦИЯ ==========
        // Один раз ловит первый шаг, где появляется inf/NaN или |eta| превышает
        // физический предел, и печатает где именно (DG-полоса vs MC, x/y, до шва).
        ReportFirstOffender(t);

        // ========== DG ДИАГНОСТИКА (первые 50 шагов) ==========
        // Verbose-gated: floods the log in multi-case sweeps when verbose off.
        if (verbose_ && (t < 50 || (t < 500 && t % 50 == 0))) {
            for (auto& patch : patches) {
                auto mp = dynamic_cast<MeshPatch*>(patch.get());
                if (mp && mp->GetDGSolver().NumElements() > 0) {
                    const auto& dg = mp->GetDGSolver();
                    double max_H = 0, min_H = 1e30;
                    int nan_count = 0;
                    for (int k = 0; k < dg.NumElements(); ++k) {
                        auto avg = dg.GetCellAverage(k);
                        if (std::isnan(avg.H) || std::isinf(avg.H)) { nan_count++; continue; }
                        if (avg.H > max_H) max_H = avg.H;
                        if (avg.H < min_H) min_H = avg.H;
                    }
                    std::cerr << "[DG t=" << t << "] H: [" << min_H << ", " << max_H
                              << "], NaN: " << nan_count << "/" << dg.NumElements() << std::endl;
                }
            }
        }

        // ========== ВЫВОД ПРОГРЕССА ==========
        if (verbose_) {
            PrintProgress(t + 1);
        }
    }

    if (verbose_) {
        std::cout << std::endl;
        std::cout << "Simulation completed!" << std::endl;
    }
}

void SimulationRunner::ReportFirstOffender(int t) {
    if (offender_fired_ || !offender_enabled_) return;

    // Lazy env init (once): DG_OFFENDER=0 disables; DG_OFFENDER_ETA sets bound.
    if (offender_eta_bound_ < 0.0) {
        offender_eta_bound_ = 50.0; // default physical |eta| ceiling [m]
        char* en = nullptr; size_t en_sz = 0;
        _dupenv_s(&en, &en_sz, "DG_OFFENDER");
        if (en && *en == '0') offender_enabled_ = false;
        if (en) free(en);
        char* eb = nullptr; size_t eb_sz = 0;
        _dupenv_s(&eb, &eb_sz, "DG_OFFENDER_ETA");
        if (eb && *eb) { double v = std::atof(eb); if (v > 0.0) offender_eta_bound_ = v; }
        if (eb) free(eb);
        if (!offender_enabled_) return;
    }

    const auto& patches = domain_->GetPatches();

    // Collect DG-seam x-coordinates (interface midpoints) to report distance-to-seam.
    std::vector<double> seam_x;
    for (const auto& patch : patches) {
        auto mp = dynamic_cast<MeshPatch*>(patch.get());
        if (!mp || mp->GetDGSolver().NumElements() == 0) continue;
        for (const auto& iface : mp->GetInterfaces()) {
            if (iface.points.empty()) continue;
            // All points of one vertical interface share ~same x; take the first.
            seam_x.push_back(iface.points.front().x);
        }
    }
    auto dist_to_seam = [&](double x) {
        double best = std::numeric_limits<double>::infinity();
        for (double sx : seam_x) best = std::min(best, std::abs(x - sx));
        return best;
    };

    auto bad = [&](double v) { return !std::isfinite(v); };

    // ---- Scan DG (MeshPatch) elements ----
    for (const auto& patch : patches) {
        auto mp = dynamic_cast<MeshPatch*>(patch.get());
        if (!mp || mp->GetDGSolver().NumElements() == 0) continue;
        const auto& dg = mp->GetDGSolver();
        const auto& mesh = mp->GetMesh();
        for (int k = 0; k < dg.NumElements(); ++k) {
            SWEState avg = dg.GetCellAverage(k);
            double D = dg.GetBathymetry(k);
            double eta = avg.H - D;
            bool nonfinite = bad(avg.H) || bad(avg.Hu) || bad(avg.Hv);
            bool over = std::isfinite(eta) && std::abs(eta) > offender_eta_bound_;
            if (nonfinite || over) {
                const auto& el = mesh.GetElement(k);
                double u = (std::isfinite(avg.H) && avg.H > SWE_H_DRY) ? avg.Hu / avg.H : 0.0;
                double v = (std::isfinite(avg.H) && avg.H > SWE_H_DRY) ? avg.Hv / avg.H : 0.0;
                std::cerr << "\n[FIRST-OFFENDER] step=" << t
                          << " region=DG patch=" << mp->GetPatchId()
                          << " elem=" << k
                          << " x=" << el.cx << " y=" << el.cy
                          << " dist_to_seam=" << dist_to_seam(el.cx)
                          << " H=" << avg.H << " D=" << D << " eta=" << eta
                          << " u=" << u << " v=" << v
                          << (nonfinite ? " [NONFINITE]" : " [OVER-BOUND]")
                          << std::endl;
                offender_fired_ = true;
                return;
            }
        }
    }

    // ---- Scan structured (MacCormack) global grid ----
    if (domain_->HasGlobalGrid()) {
        // Refresh global grid so we read post-step structured state.
        domain_->CollectResultsToGlobalGrid();
        const Grid& g = domain_->GetGlobalGrid();
        int gnx = g.GetNx(), gny = g.GetNy();
        double dy = g.GetDeltaY();
        for (int i = 0; i < gnx; ++i) {
            double x_acc = 0.0; // physical x at cell i (sum of dx up to i), mid-row
            for (int ii = 0; ii < i; ++ii) x_acc += g.GetDeltaXFunc()(ii, gny / 2);
            for (int j = 0; j < gny; ++j) {
                double H = g.H(i, j), eta = g.eta(i, j);
                bool nonfinite = bad(H) || bad(eta) || bad(g.u(i, j)) || bad(g.v(i, j));
                bool over = std::isfinite(eta) && std::abs(eta) > offender_eta_bound_;
                if (nonfinite || over) {
                    std::cerr << "\n[FIRST-OFFENDER] step=" << t
                              << " region=MC i=" << i << " j=" << j
                              << " x=" << x_acc << " y=" << (j * dy)
                              << " dist_to_seam=" << dist_to_seam(x_acc)
                              << " H=" << H << " D=" << g.D(i, j) << " eta=" << eta
                              << " u=" << g.u(i, j) << " v=" << g.v(i, j)
                              << (nonfinite ? " [NONFINITE]" : " [OVER-BOUND]")
                              << std::endl;
                    offender_fired_ = true;
                    return;
                }
            }
        }
    }
}

void SimulationRunner::SaveResults(int timestep) {
    // Собрать результаты в глобальную сетку
    if (domain_->HasGlobalGrid()) {
        domain_->CollectResultsToGlobalGrid();

        // Вызвать функцию сохранения если установлена
        if (save_function_) {
            save_function_(timestep, domain_->GetGlobalGrid());
        }
    }
}

void SimulationRunner::PrintProgress(int current_step) {
    // Вывод прогресса в процентах
    int percentage = static_cast<int>(100.0 * current_step / timesteps_);

    // Вывод с возвратом каретки для обновления одной строки
    std::cout << "\rSimulation progress: " << std::setw(3) << percentage << "%";
    std::cout.flush();
}

void SimulationRunner::TransferDGFluxesToStructured() {
    if (!domain_ || !domain_->HasGlobalGrid()) return;
    const Grid& gg = domain_->GetGlobalGrid();
    auto& patches = domain_->GetPatches();

    for (const auto& patch : patches) {
        if (auto sp = std::dynamic_pointer_cast<StructuredPatch>(patch)) {
            auto sg = sp->GetSubGrid();
            if (!sg) continue;
            sg->use_flux_left_ = false;
            sg->use_flux_right_ = false;
            sg->flux_left_.clear();
            sg->flux_right_.clear();
        } else if (auto np = std::dynamic_pointer_cast<NestedStructuredPatch>(patch)) {
            auto sg = np->GetRectSubGrid();
            if (sg) {
                sg->ClearFluxCoupling();
            }
        }
    }

    for (const auto& patch : patches) {
        auto mp = std::dynamic_pointer_cast<MeshPatch>(patch);
        if (!mp) continue;

        const auto dg_interfaces = mp->GetInterfaces();
        for (const auto& dg_iface : dg_interfaces) {
            StructuredFluxTarget target;
            if (!FindStructuredFluxTarget(dg_iface, patches, mp->GetPatchId(), target)) {
                continue;
            }

            if (auto sp = std::dynamic_pointer_cast<StructuredPatch>(target.patch)) {
                auto sg = sp->GetSubGrid();
                if (!sg) continue;

                std::vector<double> flux = mp->ProjectCoupledFlux(
                    dg_iface.boundary_name,
                    sg->local_ny_,
                    gg.GetDeltaY(),
                    0.0);

                if (dg_iface.boundary_name == "left" && target.iface.boundary_name == "right") {
                    for (auto& value : flux) value = -value;
                    sg->flux_right_ = std::move(flux);
                    sg->use_flux_right_ = true;
                } else if (dg_iface.boundary_name == "right" && target.iface.boundary_name == "left") {
                    sg->flux_left_ = std::move(flux);
                    sg->use_flux_left_ = true;
                }
                continue;
            }

            auto np = std::dynamic_pointer_cast<NestedStructuredPatch>(target.patch);
            if (!np) continue;

            auto sg = np->GetRectSubGrid();
            if (!sg) continue;

            const double spacing = InferNestedSpacing(target.iface, gg, *np);
            const double origin =
                target.iface.points.empty()
                ? 0.0
                : ((target.iface.boundary_name == "left" || target.iface.boundary_name == "right")
                   ? (target.iface.points.front().y - spacing)
                   : (target.iface.points.front().x - spacing));
            const int num_points =
                (target.iface.boundary_name == "left" || target.iface.boundary_name == "right")
                ? sg->local_ny_
                : sg->local_nx_;

            std::vector<double> flux = mp->ProjectCoupledFlux(
                dg_iface.boundary_name,
                num_points,
                spacing,
                origin);

            if (dg_iface.boundary_name == "left" && target.iface.boundary_name == "right") {
                for (auto& value : flux) value = -value;
                sg->flux_right_ = std::move(flux);
                sg->use_flux_right_ = true;
            } else if (dg_iface.boundary_name == "right" && target.iface.boundary_name == "left") {
                sg->flux_left_ = std::move(flux);
                sg->use_flux_left_ = true;
            } else if (dg_iface.boundary_name == "bottom" && target.iface.boundary_name == "top") {
                for (auto& value : flux) value = -value;
                sg->flux_top_ = std::move(flux);
                sg->use_flux_top_ = true;
            } else if (dg_iface.boundary_name == "top" && target.iface.boundary_name == "bottom") {
                sg->flux_bottom_ = std::move(flux);
                sg->use_flux_bottom_ = true;
            }
        }
    }
}
