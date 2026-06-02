// FVSolverConfig.h — env-controlled config for FV Godunov backend
//
// Phase A (Codex Audusse FV design 2026-05-28). Env-only — no API surface
// outside this header. Reading is one-time at first call (static init).
//
// Env vars:
//   MC_SOLVER          = "mc" (default, legacy MacCormack) or "fv_godunov"
//   FV_FLUX            = "hll" (default; "hllc"/"roe" reserved, not impl)
//   FV_ORDER           = 1 (default first-order); 2 for MUSCL (Phase C)
//   FV_DRY_TOL         = wet/dry threshold in metres (default 1e-6)
//   FV_CFL_MAX         = max CFL (default 0.45 for SSP-RK2)
//   FV_BOUNDARY_MODE   = "reflective" (default for FV) | "absorbing"
//   FV_AUDUSSE_PRESSURE= 1 (default; well-balanced) | 0 (ablation — disables
//                        Audusse hydrostatic pressure correction, leaves bare
//                        HLL flux divergence → no well-balancedness)

#pragma once

#include <string>

namespace fv {

enum class Solver { MacCormack, Godunov };
enum class Flux   { HLL };
enum class BoundaryMode { Reflective, Absorbing };

struct Config {
    Solver        solver        = Solver::MacCormack;
    Flux          flux          = Flux::HLL;
    int           order         = 1;
    double        dry_tol       = 1e-6;
    double        cfl_max       = 0.45;
    BoundaryMode  boundary_mode = BoundaryMode::Reflective;
    bool          audusse_pressure = true;  // Phase E pre-paper ablation toggle

    // Convenience checks called from SubGrid dispatcher and CFL block.
    bool fv_enabled() const { return solver == Solver::Godunov; }
};

// One-time env parse. Logs config to stderr if FV enabled.
const Config& config();

}  // namespace fv
