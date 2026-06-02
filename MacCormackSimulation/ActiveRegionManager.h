// ActiveRegionManager.h
//
// Optional optimization mode for MacCormack solver: skip computation/exchange
// in subgrids that remain at lake-at-rest (no wave signal yet). Design and
// validation plan: Docs/Plans/Active_Region_Tracking_Plan.md
//
// Invariants (Codex review 2026-05-27):
//   * Inactive subgrid is skipped only when its owned cells AND required halo
//     are OUTSIDE the conservative numerical dependency cone of the baseline
//     scheme. Violating this is a hard failure (cannot be recovered).
//   * Activation MUST be recomputed BEFORE half-step of the current timestep.
//   * Skip exchange only for inactive↔inactive pairs. Active↔inactive must
//     still copy ghosts so active side reads exact lake-at-rest values.
//   * `Mode::Disabled` must produce bit-for-bit identical output vs baseline.
//   * `Mode::AllActive` must produce bit-for-bit identical output vs baseline
//     (validation harness — proves the manager itself doesn't perturb state).
//
// Not used yet by Run() — Phase 1 introduces only the type and env parsing.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

#include "CoordSystem.h"

class Grid;
class SubGrid;
class ISource;

namespace active_region {

enum class Mode {
    Disabled,    // No active-region logic; baseline path.
    AllActive,   // Manager engaged but all subgrids marked active (validation harness).
    Geometric,   // Activation expanded by ceil(c_max*t*tau/dx) + margin (no signal check).
    Hybrid       // Geometric expansion + signal-based guard band check.
};

const char* ModeToString(Mode m);
Mode ModeFromString(const std::string& s);

// Subgrid metadata snapshot. Filled by SingleLineGridSimulation (friend of
// SubGrid) so manager doesn't need access to SubGrid private fields.
struct SubgridSpan {
    int id;                              // subgrid index in subgrids_
    int primary_start;                   // global_i_start_ (Horizontal) or
    int primary_end;                     // global_j_start_/end_ (Vertical)
    DecompositionDirection decomp_dir;
};

// Initial component bounding box (in global cell indices).
// Multi-source: one component per spatially-disjoint initial perturbation.
struct ComponentBbox {
    int i_min;
    int i_max;
    int j_min;
    int j_max;
};

struct Config {
    Mode mode = Mode::Disabled;
    double eta_threshold = 1e-7;          // signal activation threshold (m)
    int guard_band_cells = 16;            // safety guard near subgrid boundary
    double c_max_mps = -1.0;              // -1 = auto = sqrt(g*max_depth)
    int geometric_margin_cells = 32;      // bbox padding (cells)
    int check_period_steps = 1;           // recompute activation every N steps
    int log_active_count_interval = 100;  // log every N steps; 0 = disable
    bool log_late_risk = true;            // warn if signal close to boundary

    // Stencil cone defaults — read from TVD env vars so that we know how many
    // cells per timestep the algebraic dependency cone of the scheme covers.
    // MacCormack without TVD: 1 cell/step/side (Codex Phase 4 review).
    // MacCormack + TVD every step: 1 + 2 = 3 cells/step/side.
    bool tvd_enabled = false;
    int  tvd_apply_every = 1;

    // Phase 6b: Hybrid signal scan. Default = diagnostic-only (log late-risk
    // when signal enters guard band of an active stripe adjacent to inactive).
    // If hybrid_wakeup_enabled, also additively activate the inactive neighbor
    // for NEXT timestep (NEVER replaces geometric — Codex spec).
    bool hybrid_wakeup_enabled = false;  // ACTIVE_REGION_HYBRID_WAKEUP

    // Phase E P1 (Codex 2026-05-28): solver-aware stencil radius.
    // MacCormack: 1 cell/side per timestep (Phase 4 formula).
    // FV SSP-RK2 first-order: 2 cells/side (two stages × 1 cell each).
    // FV SSP-RK2 + MUSCL second-order: 4 cells/side.
    bool solver_is_fv = false;  // MC_SOLVER=fv_godunov
    int  fv_order     = 1;      // FV_ORDER=1|2

    // Phase E P1: boundary-driven scenarios (wavemaker/forcing on a domain
    // edge) need an explicit seed because initial η/u/v are at rest. Without
    // it Geometric/Hybrid leave everything inactive → wave never propagates.
    // ACTIVE_REGION_SEED_LEFT=1 marks the leftmost stripe active at t=0;
    // similar for RIGHT/BOTTOM/TOP. Combine as needed.
    bool seed_left   = false;
    bool seed_right  = false;
    bool seed_bottom = false;
    bool seed_top    = false;

    // Parse env vars (ACTIVE_REGION_*, TVD_ENABLED, TVD_APPLY_EVERY).
    // Unknown values -> defaults. Logs config to std::cerr if enabled.
    static Config FromEnv();
};

class ActiveRegionManager {
public:
    ActiveRegionManager();

    void Configure(const Config& cfg);
    const Config& GetConfig() const { return cfg_; }

    bool IsEnabled() const { return cfg_.mode != Mode::Disabled; }
    bool UseGlobalTvdSchedule() const { return IsEnabled(); }

    // Phase 3: scan global_grid_ for non-rest cells (eta!=0 || u!=0 || v!=0)
    // and optionally union with bbox built from parser's source list.
    // Both args may be null in Disabled mode.
    // g + max_depth are used to derive effective c_max when cfg.c_max_mps == -1.
    void InitializeFromGridAndSources(
        const Grid* global_grid,
        const std::vector<std::unique_ptr<ISource>>* sources_or_null,
        const std::vector<SubgridSpan>& spans,
        int nx, int ny,
        double dx_min, double dy,
        double g, double max_depth);

    // Phase 4-6: called BEFORE half-step of timestep t.
    // No-op in Disabled and AllActive modes.
    void UpdateActivation(
        const std::vector<std::shared_ptr<SubGrid>>& subgrids,
        int t, double tau, double dx_min);

    // Phase 6b: signal-based late-risk diagnostic + optional additive wakeup.
    // Called AFTER full-step + boundary exchange of timestep t — so it reads
    // actual post-step eta. Skipped unless mode == Hybrid.
    //
    // For each active subgrid adjacent to an inactive neighbor: scan the guard
    // band (cfg.guard_band_cells nearest to the shared boundary). If
    // max|eta| > eta_threshold there:
    //   * always log a late-risk warning (geometric cone may be insufficient);
    //   * if hybrid_wakeup_enabled, additively activate that neighbor for t+1.
    void RunSignalDiagnostic(
        const std::vector<std::shared_ptr<SubGrid>>& subgrids,
        int t);

    // Always true for Disabled/AllActive. For Geometric/Hybrid: per-subgrid flag.
    bool IsActive(int subgrid_id) const;

    // True iff BOTH sg_a and sg_b are inactive — exchange between them
    // copies lake-at-rest into lake-at-rest, can be skipped safely.
    // For Disabled/AllActive returns false (never skip).
    bool ShouldSkipExchange(int sg_a, int sg_b) const;

    int GetActiveCount() const;
    int GetTotalSubgrids() const { return static_cast<int>(is_active_.size()); }

    // Diagnostic output for summary.txt.
    void WriteSummary(std::ofstream& out) const;

private:
    Config cfg_;
    std::vector<uint8_t> is_active_;       // 1 = active, 0 = inactive
    std::vector<int> activation_step_;     // -1 if not yet activated
    std::vector<ComponentBbox> initial_components_;
    std::vector<SubgridSpan> spans_;
    int total_activated_so_far_ = 0;
    // Cached physics for geometric expansion (filled by InitializeFromGridAndSources).
    double effective_c_max_ = 0.0;
    double cached_dx_min_   = 0.0;
    double cached_dy_       = 0.0;
    int cached_nx_ = 0;
    int cached_ny_ = 0;
    // Phase 6 provenance counters (Codex Phase 6 L3): cumulative totals for
    // WriteSummary so a published run can report diagnostic activity.
    long long total_late_risk_warnings_ = 0;
    long long total_hybrid_wakeups_     = 0;
};

}  // namespace active_region
