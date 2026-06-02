// ActiveRegionManager.cpp — Phase 1 skeleton (Disabled / AllActive only)
//
// Phase 1 scope (this commit):
//   * Mode enum + string conversion
//   * Config::FromEnv()
//   * Manager with Disabled / AllActive activation logic (no signal/geometric)
//
// Phase 3 will fill InitializeFromGridAndSources (grid scan + source bbox).
// Phase 4-6 will fill UpdateActivation (geometric / hybrid).

#include "ActiveRegionManager.h"

#include <algorithm>
#include <cctype>    // std::tolower
#include <climits>   // INT_MAX, INT_MIN
#include <cmath>     // std::isfinite, std::abs, std::sin, std::cos
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdlib.h>  // _dupenv_s on MSVC

#include "Grid.h"
#include "ISource.h"
#include "SubGrid.h"  // Phase 6b: needed for SubGrid::local_grid_/local_nx_ in RunSignalDiagnostic
// EllipsoidalSource.h not needed yet — source-bbox path is Phase 6+ (see M1 note).

namespace active_region {

const char* ModeToString(Mode m) {
    switch (m) {
        case Mode::Disabled:   return "disabled";
        case Mode::AllActive:  return "all-active";
        case Mode::Geometric:  return "geometric";
        case Mode::Hybrid:     return "hybrid";
    }
    return "disabled";
}

Mode ModeFromString(const std::string& s) {
    std::string v;
    v.reserve(s.size());
    for (char c : s) v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (v == "disabled" || v == "off" || v == "0" || v.empty()) return Mode::Disabled;
    if (v == "all-active" || v == "all" || v == "allactive")    return Mode::AllActive;
    if (v == "geometric"  || v == "geo")                         return Mode::Geometric;
    if (v == "hybrid")                                           return Mode::Hybrid;
    std::cerr << "[ActiveRegion] Unknown ACTIVE_REGION_MODE='" << s
              << "', falling back to Disabled" << std::endl;
    return Mode::Disabled;
}

namespace {

// MSVC-safe env getter (project convention — see ScenarioBuilder/Sandbox).
// Returns empty string if var not set or empty.
std::string GetEnvString(const char* name) {
    char* buf = nullptr;
    size_t sz = 0;
    if (_dupenv_s(&buf, &sz, name) != 0 || buf == nullptr) {
        return std::string();
    }
    std::string out(buf);
    free(buf);
    return out;
}

double GetEnvDouble(const char* name, double def) {
    std::string v = GetEnvString(name);
    if (v.empty()) return def;
    try {
        return std::stod(v);
    } catch (...) {
        std::cerr << "[ActiveRegion] Invalid " << name << "='" << v
                  << "', using default " << def << std::endl;
        return def;
    }
}

int GetEnvInt(const char* name, int def) {
    std::string v = GetEnvString(name);
    if (v.empty()) return def;
    try {
        return std::stoi(v);
    } catch (...) {
        std::cerr << "[ActiveRegion] Invalid " << name << "='" << v
                  << "', using default " << def << std::endl;
        return def;
    }
}

bool GetEnvBool(const char* name, bool def) {
    std::string s = GetEnvString(name);
    if (s.empty()) return def;
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "1" || s == "true" || s == "on" || s == "yes") return true;
    if (s == "0" || s == "false" || s == "off" || s == "no") return false;
    return def;
}

}  // namespace

Config Config::FromEnv() {
    Config c;
    std::string mode_str = GetEnvString("ACTIVE_REGION_MODE");
    c.mode = mode_str.empty() ? Mode::Disabled : ModeFromString(mode_str);
    c.eta_threshold            = GetEnvDouble("ACTIVE_REGION_THRESHOLD", 1e-7);
    c.guard_band_cells         = GetEnvInt   ("ACTIVE_REGION_GUARD", 16);
    c.c_max_mps                = GetEnvDouble("ACTIVE_REGION_CMAX", -1.0);
    c.geometric_margin_cells   = GetEnvInt   ("ACTIVE_REGION_MARGIN", 32);
    c.check_period_steps       = GetEnvInt   ("ACTIVE_REGION_CHECK_PERIOD", 1);
    c.log_active_count_interval= GetEnvInt   ("ACTIVE_REGION_LOG_INTERVAL", 100);
    c.log_late_risk            = GetEnvBool  ("ACTIVE_REGION_LOG_LATE_RISK", true);

    // TVD knobs — duplicate of SubGrid env reading so manager can compute
    // algebraic stencil cone independently. Keep symbols in sync with
    // SubGrid::tvd_config() (TVD_ENABLED, TVD_APPLY_EVERY).
    c.tvd_enabled     = GetEnvBool("TVD_ENABLED", false);
    c.tvd_apply_every = GetEnvInt ("TVD_APPLY_EVERY", 1);
    if (c.tvd_apply_every < 1) c.tvd_apply_every = 1;

    // Codex Phase E P1: detect FV Godunov backend so stencil cone matches.
    // FV SSP-RK2 first-order has stencil radius 2 cells/side (L(U0)→U1 reads
    // i±1, then L(U1) reads i±1 of U1 which already has signal from i±1 of U0,
    // so combined signal from i±2 reaches cell). MUSCL adds another cell.
    {
        std::string sv = GetEnvString("MC_SOLVER");
        for (char& ch : sv) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        c.solver_is_fv = (sv == "fv_godunov" || sv == "fv" || sv == "godunov");
        c.fv_order     = GetEnvInt("FV_ORDER", 1);
        if (c.fv_order < 1) c.fv_order = 1;
        if (c.fv_order > 2) c.fv_order = 2;
    }

    // Phase E P1: boundary seeds for wavemaker-driven scenarios.
    c.seed_left   = GetEnvBool("ACTIVE_REGION_SEED_LEFT", false);
    c.seed_right  = GetEnvBool("ACTIVE_REGION_SEED_RIGHT", false);
    c.seed_bottom = GetEnvBool("ACTIVE_REGION_SEED_BOTTOM", false);
    c.seed_top    = GetEnvBool("ACTIVE_REGION_SEED_TOP", false);

    c.hybrid_wakeup_enabled = GetEnvBool("ACTIVE_REGION_HYBRID_WAKEUP", false);

    // Validate / clamp: avoid div-by-zero, NaN, negatives in later phases.
    auto warn_clamp = [](const char* name, auto bad, auto good) {
        std::cerr << "[ActiveRegion] " << name << "=" << bad
                  << " invalid, clamped to " << good << std::endl;
    };
    if (!std::isfinite(c.eta_threshold) || c.eta_threshold < 0.0) {
        warn_clamp("threshold", c.eta_threshold, 1e-7);
        c.eta_threshold = 1e-7;
    }
    if (c.guard_band_cells < 0) {
        warn_clamp("guard", c.guard_band_cells, 0);
        c.guard_band_cells = 0;
    }
    if (c.geometric_margin_cells < 0) {
        warn_clamp("margin", c.geometric_margin_cells, 0);
        c.geometric_margin_cells = 0;
    }
    if (c.check_period_steps < 1) {
        warn_clamp("check_period", c.check_period_steps, 1);
        c.check_period_steps = 1;
    }
    if (c.log_active_count_interval < 0) {
        warn_clamp("log_interval", c.log_active_count_interval, 0);
        c.log_active_count_interval = 0;
    }
    // c_max: allow -1 (auto) или strictly positive; anything else → auto.
    if (c.c_max_mps != -1.0 && !(std::isfinite(c.c_max_mps) && c.c_max_mps > 0.0)) {
        warn_clamp("c_max", c.c_max_mps, -1.0);
        c.c_max_mps = -1.0;
    }

    if (c.mode != Mode::Disabled) {
        std::cerr << "[ActiveRegion] Config:"
                  << " mode=" << ModeToString(c.mode)
                  << " threshold=" << c.eta_threshold
                  << " guard=" << c.guard_band_cells
                  << " c_max=" << (c.c_max_mps > 0 ? std::to_string(c.c_max_mps) : std::string("auto"))
                  << " margin=" << c.geometric_margin_cells
                  << " check_period=" << c.check_period_steps
                  << " tvd_enabled=" << (c.tvd_enabled ? 1 : 0)
                  << " tvd_apply_every=" << c.tvd_apply_every
                  << std::endl;
    }
    return c;
}

ActiveRegionManager::ActiveRegionManager() = default;

void ActiveRegionManager::Configure(const Config& cfg) {
    cfg_ = cfg;
}

namespace {

// Phase 6 helper: classify a cell as "non-rest" (part of an initial source).
inline bool CellIsNonRest(const Grid& grid, int i, int j, double threshold) {
    const double e = static_cast<double>(grid.eta(i, j));
    const double u = static_cast<double>(grid.u(i, j));
    const double v = static_cast<double>(grid.v(i, j));
    return std::abs(e) > threshold || std::abs(u) > threshold || std::abs(v) > threshold;
}

// Scan global grid for non-rest cells; return bbox in global cell indices
// or empty bbox (i_max<i_min) if grid is at rest.
// Uses const accessors grid.eta/u/v (Codex recommendation).
ComponentBbox ScanGridForNonRest(const Grid& grid, int nx, int ny, double threshold) {
    ComponentBbox b{ INT_MAX, INT_MIN, INT_MAX, INT_MIN };
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (CellIsNonRest(grid, i, j, threshold)) {
                if (i < b.i_min) b.i_min = i;
                if (i > b.i_max) b.i_max = i;
                if (j < b.j_min) b.j_min = j;
                if (j > b.j_max) b.j_max = j;
            }
        }
    }
    return b;
}

// Phase 6: 8-connectivity flood fill — return one ComponentBbox per spatially
// disjoint cluster of non-rest cells. Codex recommendation: 8-connectivity
// (diagonal contact) handles ellipse/raster discretization bridges.
//
// Memory: 1 byte/cell visited mask; for Pacific 9601x6781 ≈ 62 MB — acceptable
// once-only init scan. Uses explicit stack to avoid recursion depth issues.
std::vector<ComponentBbox> ScanGridForComponents(
    const Grid& grid, int nx, int ny, double threshold)
{
    std::vector<ComponentBbox> out;
    if (nx <= 0 || ny <= 0) return out;
    std::vector<uint8_t> visited(static_cast<size_t>(nx) * ny, 0);
    auto idx = [ny](int i, int j) -> size_t {
        return static_cast<size_t>(i) * ny + static_cast<size_t>(j);
    };

    // Worklist (stack) of (i, j) cells to expand.
    std::vector<std::pair<int,int>> stack;
    stack.reserve(1024);

    for (int i0 = 0; i0 < nx; ++i0) {
        for (int j0 = 0; j0 < ny; ++j0) {
            if (visited[idx(i0, j0)]) continue;
            if (!CellIsNonRest(grid, i0, j0, threshold)) {
                visited[idx(i0, j0)] = 1;
                continue;
            }

            ComponentBbox b{ i0, i0, j0, j0 };
            stack.clear();
            stack.emplace_back(i0, j0);
            visited[idx(i0, j0)] = 1;

            while (!stack.empty()) {
                const auto [ci, cj] = stack.back();
                stack.pop_back();
                if (ci < b.i_min) b.i_min = ci;
                if (ci > b.i_max) b.i_max = ci;
                if (cj < b.j_min) b.j_min = cj;
                if (cj > b.j_max) b.j_max = cj;

                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        const int ni = ci + di;
                        const int nj = cj + dj;
                        if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                        const size_t ix = idx(ni, nj);
                        if (visited[ix]) continue;
                        visited[ix] = 1;
                        if (CellIsNonRest(grid, ni, nj, threshold)) {
                            stack.emplace_back(ni, nj);
                        }
                    }
                }
            }
            out.push_back(b);
        }
    }
    return out;
}

// NOTE Phase 3 (Codex review M1): source-bbox path is intentionally disabled.
// EllipsoidalSource stores center/radii in SOURCE units; for ASC-based scenarios
// (Grid::SetParamsTest6) these are world coords (lon/lat or m-from-xll), while
// dx_min/dy here are simulator cell metres. Without proper world->cell mapping
// (xll, yll, cellsize, y-flip), source-only fallback could activate the wrong
// stripe when grid scan is empty. Until Phase 6 adds the mapping, we rely solely
// on grid scan; source list is accepted in the API for forward-compatibility.

// Union of two bboxes; empty (i_max<i_min) operands are ignored.
ComponentBbox BboxUnion(const ComponentBbox& a, const ComponentBbox& b) {
    const bool a_empty = a.i_max < a.i_min || a.j_max < a.j_min;
    const bool b_empty = b.i_max < b.i_min || b.j_max < b.j_min;
    if (a_empty) return b;
    if (b_empty) return a;
    ComponentBbox r;
    r.i_min = std::min(a.i_min, b.i_min);
    r.i_max = std::max(a.i_max, b.i_max);
    r.j_min = std::min(a.j_min, b.j_min);
    r.j_max = std::max(a.j_max, b.j_max);
    return r;
}

// True iff bbox primary-axis interval intersects [span_start, span_end).
bool BboxIntersectsSpan(const ComponentBbox& b, const SubgridSpan& sp) {
    const bool empty = b.i_max < b.i_min || b.j_max < b.j_min;
    if (empty) return false;
    int lo, hi;
    if (sp.decomp_dir == DecompositionDirection::Horizontal) {
        lo = b.i_min; hi = b.i_max;
    } else {
        lo = b.j_min; hi = b.j_max;
    }
    // span is [primary_start, primary_end); bbox interval [lo, hi] inclusive.
    return !(hi < sp.primary_start || lo >= sp.primary_end);
}

}  // namespace

void ActiveRegionManager::InitializeFromGridAndSources(
    const Grid* global_grid,
    const std::vector<std::unique_ptr<ISource>>* sources_or_null,
    const std::vector<SubgridSpan>& spans,
    int nx, int ny,
    double dx_min, double dy,
    double g, double max_depth)
{
    spans_ = spans;
    const size_t n = spans.size();
    is_active_.assign(n, 0);
    activation_step_.assign(n, -1);
    initial_components_.clear();
    total_activated_so_far_ = 0;

    cached_dx_min_ = dx_min;
    cached_dy_     = dy;
    cached_nx_     = nx;
    cached_ny_     = ny;
    if (cfg_.c_max_mps > 0.0) {
        effective_c_max_ = cfg_.c_max_mps;
    } else if (g > 0.0 && max_depth > 0.0) {
        effective_c_max_ = std::sqrt(g * max_depth);
    } else {
        effective_c_max_ = 0.0;
        std::cerr << "[ActiveRegion] WARNING: cannot derive effective c_max "
                     "(c_max_cfg=" << cfg_.c_max_mps << ", g=" << g
                  << ", max_depth=" << max_depth
                  << "); geometric expansion will be disabled." << std::endl;
    }

    if (cfg_.mode == Mode::Disabled) {
        return;
    }
    if (cfg_.mode == Mode::AllActive) {
        std::fill(is_active_.begin(), is_active_.end(), uint8_t{1});
        std::fill(activation_step_.begin(), activation_step_.end(), 0);
        total_activated_so_far_ = static_cast<int>(n);
        std::cerr << "[ActiveRegion] Mode=all-active: " << n << " subgrids active from t=0"
                  << std::endl;
        return;
    }

    // Geometric / Hybrid: build initial seed components.
    // Phase 6: 8-connectivity flood fill returns one bbox per spatially disjoint
    // cluster (multi-source support). Single source → one component; Pacific
    // raster → one component; multi-source non-overlapping → N components.
    if (global_grid != nullptr) {
        const double seed_thr = 0.0;  // strict !=0 for grid scan seed
        initial_components_ = ScanGridForComponents(*global_grid, nx, ny, seed_thr);
        if (!initial_components_.empty()) {
            std::cerr << "[ActiveRegion] Grid scan found " << initial_components_.size()
                      << " component(s):" << std::endl;
            for (size_t k = 0; k < initial_components_.size(); ++k) {
                const auto& c = initial_components_[k];
                std::cerr << "[ActiveRegion]   [" << k << "] i=[" << c.i_min
                          << "," << c.i_max << "] j=[" << c.j_min
                          << "," << c.j_max << "]" << std::endl;
            }
        }
    }

    // Phase 3 (Codex M1): source-bbox path disabled until proper world->cell
    // mapping lands. Log parsed source count for diagnostics only.
    if (sources_or_null != nullptr && !sources_or_null->empty()) {
        std::cerr << "[ActiveRegion] Parser reports " << sources_or_null->size()
                  << " source(s); using grid scan as seed (source-bbox mapping is Phase 6+)"
                  << std::endl;
    }

    // Activate subgrids whose primary-axis interval intersects ANY component.
    for (const auto& seed : initial_components_) {
        for (const auto& sp : spans_) {
            if (sp.id < 0 || sp.id >= static_cast<int>(is_active_.size())) continue;
            if (is_active_[sp.id]) continue;
            if (BboxIntersectsSpan(seed, sp)) {
                is_active_[sp.id] = 1;
                activation_step_[sp.id] = 0;
                ++total_activated_so_far_;
            }
        }
    }

    // Phase E P1 (Codex): boundary seeds for wavemaker/forcing-driven scenarios.
    // The grid scan above misses scenarios where initial state is at rest but
    // signal will be injected through a domain boundary (e.g. Monai wavemaker).
    // Without these seeds Geometric/Hybrid would leave everything inactive.
    //
    // We add BOTH a stripe activation AND a virtual component bbox so that
    // UpdateActivation has something to expand geometrically each timestep.
    auto activate_subgrid_at = [&](int span_idx) {
        if (span_idx < 0 || span_idx >= static_cast<int>(spans_.size())) return;
        const int sg_id = spans_[span_idx].id;
        if (sg_id < 0 || sg_id >= static_cast<int>(is_active_.size())) return;
        if (!is_active_[sg_id]) {
            is_active_[sg_id] = 1;
            activation_step_[sg_id] = 0;
            ++total_activated_so_far_;
        }
        // Virtual component covering this stripe — used as expansion seed.
        ComponentBbox b;
        if (spans_[span_idx].decomp_dir == DecompositionDirection::Horizontal) {
            b.i_min = spans_[span_idx].primary_start;
            b.i_max = spans_[span_idx].primary_end - 1;
            b.j_min = 0;
            b.j_max = cached_ny_ - 1;  // full cross-axis extent
        } else {
            b.i_min = 0;
            b.i_max = cached_nx_ - 1;
            b.j_min = spans_[span_idx].primary_start;
            b.j_max = spans_[span_idx].primary_end - 1;
        }
        // Clamp.
        if (b.i_min < 0) b.i_min = 0;
        if (b.j_min < 0) b.j_min = 0;
        if (b.i_max >= cached_nx_) b.i_max = cached_nx_ - 1;
        if (b.j_max >= cached_ny_) b.j_max = cached_ny_ - 1;
        initial_components_.push_back(b);
    };
    if (cfg_.seed_left || cfg_.seed_bottom) {
        activate_subgrid_at(0);
    }
    if (cfg_.seed_right || cfg_.seed_top) {
        activate_subgrid_at(static_cast<int>(spans_.size()) - 1);
    }

    if (initial_components_.empty() && total_activated_so_far_ == 0) {
        std::cerr << "[ActiveRegion] WARNING: no non-rest cells detected and "
                     "no boundary seeds; all subgrids will be inactive (no work)."
                  << std::endl;
        return;
    }

    std::cerr << "[ActiveRegion] Mode=" << ModeToString(cfg_.mode)
              << ": " << GetActiveCount() << " / " << n
              << " subgrids initially active" << std::endl;
}

void ActiveRegionManager::UpdateActivation(
    const std::vector<std::shared_ptr<SubGrid>>& /*subgrids*/,
    int t, double tau, double dx_min)
{
    // Disabled/AllActive: nothing to update — IsActive() already returns true.
    if (cfg_.mode == Mode::Disabled || cfg_.mode == Mode::AllActive) return;
    if (initial_components_.empty()) return;  // nothing seeded

    if (cfg_.check_period_steps > 1 && (t % cfg_.check_period_steps) != 0) return;

    const double dx_use = (dx_min > 0.0 && dx_min < cached_dx_min_) ? dx_min : cached_dx_min_;
    const double dy_use = cached_dy_;
    if (dx_use <= 0.0 || dy_use <= 0.0) return;

    // Codex Phase 4 spec — radius computation in cells.
    //
    // UpdateActivation is called BEFORE half-step of step `t`. To stay safe
    // until the NEXT activation update, we must cover steps_to_cover steps:
    //   steps_to_cover = t + max(1, check_period_steps)
    //
    // Two parallel constraints:
    //   physical_radius_cells = ceil(c_phys * steps_to_cover * tau / dx_use)
    //   stencil_radius_cells  = mc_per_step + tvd_per_step (per side)
    //     where mc_per_step  = 1  (MacCormack 3-pt stencil)
    //           tvd_per_step = 2  (TVD ±2 stencil), only every tvd_apply_every steps
    // The active region must enclose BOTH cones; we take max() and add margin.
    const int lookahead = std::max(1, cfg_.check_period_steps);
    const long long steps_to_cover = static_cast<long long>(t) + lookahead;

    // Solver-aware per-step stencil width (Codex Phase E P1):
    //   MC: 1 cell/side/step (3-point stencil per substep).
    //   FV SSP-RK2 first-order: 2 cells/side/step (two stages × 1 cell each).
    //   FV SSP-RK2 + MUSCL second-order: 4 cells/side/step.
    long long per_step_cells = 1;  // default MC
    if (cfg_.solver_is_fv) {
        per_step_cells = (cfg_.fv_order >= 2) ? 4 : 2;
    }
    const long long mc_radius = per_step_cells * steps_to_cover;

    // Conservative: assume TVD fires within steps_to_cover even if apply_every>1.
    long long tvd_extra = 0;
    if (cfg_.tvd_enabled) {
        const int ae = std::max(1, cfg_.tvd_apply_every);
        tvd_extra = 2LL * ((steps_to_cover + ae - 1) / ae);
    }
    const long long stencil_radius = mc_radius + tvd_extra;

    const double phys_rx_d = std::ceil((effective_c_max_ * static_cast<double>(steps_to_cover) * tau) / dx_use);
    const double phys_ry_d = std::ceil((effective_c_max_ * static_cast<double>(steps_to_cover) * tau) / dy_use);
    const long long phys_rx = (effective_c_max_ > 0.0) ? static_cast<long long>(phys_rx_d) : 0;
    const long long phys_ry = (effective_c_max_ > 0.0) ? static_cast<long long>(phys_ry_d) : 0;

    const long long rx_ll = std::max(phys_rx, stencil_radius) + cfg_.geometric_margin_cells;
    const long long ry_ll = std::max(phys_ry, stencil_radius) + cfg_.geometric_margin_cells;
    const int rx = static_cast<int>(std::min(rx_ll, static_cast<long long>(cached_nx_)));
    const int ry = static_cast<int>(std::min(ry_ll, static_cast<long long>(cached_ny_)));

    int newly_activated = 0;
    for (const auto& seed : initial_components_) {
        ComponentBbox expanded;
        expanded.i_min = std::max(0,            seed.i_min - rx);
        expanded.i_max = std::min(cached_nx_-1, seed.i_max + rx);
        expanded.j_min = std::max(0,            seed.j_min - ry);
        expanded.j_max = std::min(cached_ny_-1, seed.j_max + ry);

        for (const auto& sp : spans_) {
            if (sp.id < 0 || sp.id >= static_cast<int>(is_active_.size())) continue;
            if (is_active_[sp.id]) continue;
            if (BboxIntersectsSpan(expanded, sp)) {
                is_active_[sp.id] = 1;
                activation_step_[sp.id] = t;
                ++total_activated_so_far_;
                ++newly_activated;
            }
        }
    }
    if (newly_activated > 0 && cfg_.log_active_count_interval > 0) {
        std::cerr << "[ActiveRegion] t=" << t << " activated " << newly_activated
                  << " new subgrid(s); now " << GetActiveCount() << " / "
                  << is_active_.size() << " active (rx=" << rx << ", ry=" << ry
                  << ", stencil_r=" << stencil_radius << ", phys_r=" << phys_rx << ")"
                  << std::endl;
    }
}

void ActiveRegionManager::RunSignalDiagnostic(
    const std::vector<std::shared_ptr<SubGrid>>& subgrids,
    int t)
{
    // Only Hybrid mode runs the diagnostic — Geometric stays cone-only.
    if (cfg_.mode != Mode::Hybrid) return;
    if (cfg_.guard_band_cells <= 0) return;
    if (subgrids.empty()) return;

    const int N = static_cast<int>(spans_.size());
    if (N == 0) return;

    // For each active subgrid, look at its primary-axis boundary cells
    // (within `guard_band_cells` from the shared boundary with each inactive
    // neighbor). If max|eta| exceeds the threshold, geometric cone may be
    // insufficient: log warning (always) and optionally wake up neighbor.
    // NOTE: only scans owned interior rows (excludes ghost row 0 and ghost
    // row local_nx-1) so we observe what active stripe computed this step.
    int wakeups_this_step = 0;
    int warnings_this_step = 0;
    (void)warnings_this_step;  // accumulated below via total_late_risk_warnings_

    for (int i = 0; i < N; ++i) {
        if (!is_active_[i]) continue;
        const auto& sg = subgrids[i];
        if (!sg) continue;
        const int nx = sg->local_nx_;
        const int ny = sg->local_ny_;
        // Codex Phase 6 M1: branch on decomposition direction.
        //   Horizontal stripes: stripe axis = i_local; scan owned i bands
        //                       toward neighbors (i-1) and (i+1).
        //   Vertical stripes:   stripe axis = j_local; scan owned j bands
        //                       toward neighbors (i-1) and (i+1) in spans_,
        //                       which correspond to j primary-axis intervals.
        const bool vertical = sg->GetDecompDir() == DecompositionDirection::Vertical;
        const int stripe_axis_size = vertical ? ny : nx;
        const int cross_axis_size  = vertical ? nx : ny;
        if (stripe_axis_size < 4 || cross_axis_size < 1) continue;

        for (int side = 0; side < 2; ++side) {
            const int neighbor_id = (side == 0) ? (i - 1) : (i + 1);
            if (neighbor_id < 0 || neighbor_id >= N) continue;
            if (is_active_[neighbor_id]) continue;   // already active — irrelevant

            // Owned cells span [1 .. axis_size-2] (ghost 0 and axis_size-1).
            const int guard = cfg_.guard_band_cells;
            int lo, hi;
            if (side == 0) {
                lo = 1;
                hi = std::min(stripe_axis_size - 2, guard);
            } else {
                lo = std::max(1, stripe_axis_size - 1 - guard);
                hi = stripe_axis_size - 2;
            }
            if (lo > hi) continue;

            double max_abs_eta = 0.0;
            if (vertical) {
                // axis is j; cross is i. iterate j in guard band, i across owned cells.
                for (int jj = lo; jj <= hi; ++jj) {
                    for (int ii = 0; ii < cross_axis_size; ++ii) {
                        const double e = std::abs(static_cast<double>(sg->local_grid_.eta(ii, jj)));
                        if (e > max_abs_eta) max_abs_eta = e;
                    }
                }
            } else {
                // axis is i; cross is j.
                for (int ii = lo; ii <= hi; ++ii) {
                    for (int jj = 0; jj < cross_axis_size; ++jj) {
                        const double e = std::abs(static_cast<double>(sg->local_grid_.eta(ii, jj)));
                        if (e > max_abs_eta) max_abs_eta = e;
                    }
                }
            }
            if (max_abs_eta > cfg_.eta_threshold) {
                ++warnings_this_step;
                ++total_late_risk_warnings_;
                if (cfg_.log_late_risk) {
                    std::cerr << "[ActiveRegion] t=" << t << " late-risk: subgrid " << i
                              << " (active, " << (vertical ? "vert" : "horiz")
                              << ") has |eta|=" << max_abs_eta
                              << " in guard band toward inactive neighbor " << neighbor_id
                              << " (threshold=" << cfg_.eta_threshold << ")" << std::endl;
                }
                if (cfg_.hybrid_wakeup_enabled) {
                    is_active_[neighbor_id] = 1;
                    activation_step_[neighbor_id] = t + 1;
                    ++total_activated_so_far_;
                    ++wakeups_this_step;
                    ++total_hybrid_wakeups_;
                }
            }
        }
    }
    if (wakeups_this_step > 0) {
        std::cerr << "[ActiveRegion] t=" << t << " hybrid wakeup activated "
                  << wakeups_this_step << " subgrid(s); now "
                  << GetActiveCount() << " / " << N << " active" << std::endl;
    }
}

bool ActiveRegionManager::IsActive(int subgrid_id) const {
    if (cfg_.mode == Mode::Disabled) return true;  // never skip in Disabled
    if (subgrid_id < 0 || subgrid_id >= static_cast<int>(is_active_.size())) {
        return true;
    }
    return is_active_[subgrid_id] != 0;
}

bool ActiveRegionManager::ShouldSkipExchange(int sg_a, int sg_b) const {
    if (cfg_.mode == Mode::Disabled || cfg_.mode == Mode::AllActive) return false;
    const bool a_active = IsActive(sg_a);
    const bool b_active = IsActive(sg_b);
    // Skip only when BOTH inactive — active↔inactive must still copy ghosts.
    return !a_active && !b_active;
}

int ActiveRegionManager::GetActiveCount() const {
    int n = 0;
    for (uint8_t v : is_active_) if (v) ++n;
    return n;
}

void ActiveRegionManager::WriteSummary(std::ofstream& out) const {
    out << "active_region_mode=" << ModeToString(cfg_.mode) << std::endl;
    if (cfg_.mode == Mode::Disabled) return;
    out << "active_region_threshold=" << cfg_.eta_threshold << std::endl;
    out << "active_region_guard=" << cfg_.guard_band_cells << std::endl;
    out << "active_region_c_max="
        << (cfg_.c_max_mps > 0 ? std::to_string(cfg_.c_max_mps) : std::string("auto"))
        << std::endl;
    out << "active_region_margin=" << cfg_.geometric_margin_cells << std::endl;
    out << "active_region_check_period=" << cfg_.check_period_steps << std::endl;
    out << "active_region_tvd_enabled=" << (cfg_.tvd_enabled ? 1 : 0) << std::endl;
    out << "active_region_tvd_apply_every=" << cfg_.tvd_apply_every << std::endl;
    out << "active_region_hybrid_wakeup_enabled="
        << (cfg_.hybrid_wakeup_enabled ? 1 : 0) << std::endl;
    out << "active_region_total_subgrids=" << GetTotalSubgrids() << std::endl;
    out << "active_region_initial_components=" << initial_components_.size() << std::endl;
    out << "active_region_final_active=" << GetActiveCount() << std::endl;
    out << "active_region_total_activated_so_far=" << total_activated_so_far_ << std::endl;
    // Phase 6b diagnostic provenance.
    out << "active_region_late_risk_warnings_total=" << total_late_risk_warnings_ << std::endl;
    out << "active_region_hybrid_wakeups_total=" << total_hybrid_wakeups_ << std::endl;
}

}  // namespace active_region
