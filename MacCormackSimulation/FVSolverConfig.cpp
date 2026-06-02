// FVSolverConfig.cpp — env parser implementation
#include "FVSolverConfig.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdlib.h>  // _dupenv_s on MSVC

namespace fv {

namespace {

std::string GetEnvString(const char* name) {
    char* buf = nullptr; size_t sz = 0;
    if (_dupenv_s(&buf, &sz, name) != 0 || buf == nullptr) return {};
    std::string v(buf);
    free(buf);
    return v;
}

double GetEnvDouble(const char* name, double def) {
    std::string s = GetEnvString(name);
    if (s.empty()) return def;
    try { return std::stod(s); } catch (...) {
        std::cerr << "[FV] invalid " << name << "='" << s << "', using " << def << std::endl;
        return def;
    }
}

int GetEnvInt(const char* name, int def) {
    std::string s = GetEnvString(name);
    if (s.empty()) return def;
    try { return std::stoi(s); } catch (...) {
        std::cerr << "[FV] invalid " << name << "='" << s << "', using " << def << std::endl;
        return def;
    }
}

Solver ParseSolver(const std::string& s) {
    if (s.empty() || s == "mc" || s == "maccormack") return Solver::MacCormack;
    if (s == "fv_godunov" || s == "fv" || s == "godunov") return Solver::Godunov;
    std::cerr << "[FV] unknown MC_SOLVER='" << s << "', defaulting to mc" << std::endl;
    return Solver::MacCormack;
}

Flux ParseFlux(const std::string& s) {
    if (s.empty() || s == "hll") return Flux::HLL;
    std::cerr << "[FV] FV_FLUX='" << s << "' not implemented yet — using HLL" << std::endl;
    return Flux::HLL;
}

BoundaryMode ParseBoundary(const std::string& s) {
    if (s.empty() || s == "reflective" || s == "wall") return BoundaryMode::Reflective;
    if (s == "absorbing" || s == "radiation") return BoundaryMode::Absorbing;
    std::cerr << "[FV] unknown FV_BOUNDARY_MODE='" << s << "', defaulting to reflective"
              << std::endl;
    return BoundaryMode::Reflective;
}

Config MakeFromEnv() {
    Config c;
    c.solver        = ParseSolver(GetEnvString("MC_SOLVER"));
    c.flux          = ParseFlux  (GetEnvString("FV_FLUX"));
    c.order         = GetEnvInt  ("FV_ORDER", 1);
    c.dry_tol       = GetEnvDouble("FV_DRY_TOL", 1e-6);
    c.cfl_max       = GetEnvDouble("FV_CFL_MAX", 0.45);
    c.boundary_mode = ParseBoundary(GetEnvString("FV_BOUNDARY_MODE"));
    c.audusse_pressure = (GetEnvInt("FV_AUDUSSE_PRESSURE", 1) != 0);

    if (c.order < 1) c.order = 1;
    if (c.order > 2) c.order = 2;  // MUSCL only up to second (Phase C)
    if (c.dry_tol < 0) c.dry_tol = 0;
    if (c.cfl_max <= 0 || c.cfl_max > 1.0) c.cfl_max = 0.45;

    if (c.fv_enabled()) {
        std::cerr << "[FV] solver=Godunov flux=HLL order=" << c.order
                  << " dry_tol=" << c.dry_tol
                  << " cfl_max=" << c.cfl_max
                  << " boundary=" << (c.boundary_mode == BoundaryMode::Reflective
                                      ? "reflective" : "absorbing")
                  << " audusse_pressure=" << (c.audusse_pressure ? "on" : "OFF (ablation)")
                  << std::endl;
    }
    return c;
}

}  // namespace

const Config& config() {
    static const Config c = MakeFromEnv();
    return c;
}

}  // namespace fv
