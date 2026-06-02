// MeshPatch.cpp - DG solver MeshPatch implementation

#include "MeshPatch.h"
#include "Grid.h"
#include <algorithm>
#include <cmath>
#include <iostream>

MeshPatch::MeshPatch(int patch_id, int num_nodes)
    : patch_id_(patch_id), num_nodes_(num_nodes) {
    AllocateBuffers();
}

void MeshPatch::AllocateBuffers() {
    H_buf_.resize(num_nodes_, 0.0);
    u_buf_.resize(num_nodes_, 0.0);
    v_buf_.resize(num_nodes_, 0.0);
    eta_buf_.resize(num_nodes_, 0.0);
    H_hat_buf_.resize(num_nodes_, 0.0);
    u_hat_buf_.resize(num_nodes_, 0.0);
    v_hat_buf_.resize(num_nodes_, 0.0);
    eta_hat_buf_.resize(num_nodes_, 0.0);
    // Incoming buffers: written by CopyBufferToPatch, read by ApplyExchange.
    // Separate from the outgoing (DG-export) buffers above.
    H_in_buf_.resize(num_nodes_, 0.0);
    u_in_buf_.resize(num_nodes_, 0.0);
    v_in_buf_.resize(num_nodes_, 0.0);
    eta_in_buf_.resize(num_nodes_, 0.0);
    H_hat_in_buf_.resize(num_nodes_, 0.0);
    u_hat_in_buf_.resize(num_nodes_, 0.0);
    v_hat_in_buf_.resize(num_nodes_, 0.0);
    eta_hat_in_buf_.resize(num_nodes_, 0.0);
}

bool MeshPatch::LoadMesh(const std::string& filename) {
    if (!MeshReader::ReadMsh(filename, mesh_)) {
        std::cerr << "MeshPatch: Failed to load mesh: " << MeshReader::GetLastError() << std::endl;
        return false;
    }
    mesh_.PrintSummary();
    return true;
}

void MeshPatch::SetMesh(TriMesh&& mesh) {
    mesh_ = std::move(mesh);
}

void MeshPatch::SetBathymetry(std::function<double(double, double)> func) {
    if (dg_initialized_) {
        dg_solver_.SetBathymetryFunction(func);
    }
}

void MeshPatch::SetInitialCondition(std::function<SWEState(double, double)> func) {
    if (dg_initialized_) {
        dg_solver_.SetInitialCondition(func);
        UpdateBuffersFromDG();
    }
}

void MeshPatch::SetInitialConditionWellBalanced(
    std::function<SWEState(double, double)> q_func,
    std::function<double(double, double)> d_func) {
    if (dg_initialized_) {
        dg_solver_.SetInitialConditionWellBalanced(q_func, d_func);
        UpdateBuffersFromDG();
    }
}

void MeshPatch::SetBoundaryCondition(int tag, DGBoundaryType type) {
    dg_solver_.SetBoundaryCondition(tag, type);
}

void MeshPatch::Initialize() {
    if (mesh_.NumElements() == 0) {
        // No mesh loaded - operate in legacy stub mode
        return;
    }

    // Precompute DG matrices
    elem_matrices_.Precompute(mesh_);

    // Initialize DG solver
    dg_solver_.Initialize(&mesh_, &elem_matrices_);
    dg_initialized_ = true;

    std::cerr << "MeshPatch " << patch_id_ << ": DG solver initialized, "
              << mesh_.NumElements() << " elements, "
              << dg_solver_.TotalDOFs() << " DOFs" << std::endl;
}

void MeshPatch::ComputeHalfStep() {
    if (!dg_initialized_) {
        // Legacy stub: copy to hat buffers
        H_hat_buf_ = H_buf_;
        u_hat_buf_ = u_buf_;
        v_hat_buf_ = v_buf_;
        eta_hat_buf_ = eta_buf_;
        return;
    }

    // Use global dt (set by SimulationRunner via SetTimeStep).
    // DG advances the same physical time as MacCormack.
    // SSP-RK3 for P2 stability at the given dt.
    double dt = dt_;
    if (dt <= 0.0) {
        // Fallback: auto-compute from DG CFL (standalone mode)
        dt = dg_solver_.ComputeTimeStep(cfl_target_);
    }

    dg_solver_.StepRK3(dt);

    UpdateBuffersFromDG();

    // DG treats the whole RK3 step as its "half-step" result.  Copy the updated
    // main buffers into the hat buffers so that AfterHalfStep exchange gives
    // StructuredPatch a meaningful intermediate DG state.
    H_hat_buf_   = H_buf_;
    u_hat_buf_   = u_buf_;
    v_hat_buf_   = v_buf_;
    eta_hat_buf_ = eta_buf_;
}

void MeshPatch::ComputeFullStep() {
    if (!dg_initialized_) return;

    // DG already did the full step in ComputeHalfStep.
    // This method exists for IPatch interface compatibility.
    // In a true hybrid, we might do exchange between half and full steps.
}

void MeshPatch::PrepareExchange(double dt, double t) {
    // Update interface buffers from DG solution
    if (dg_initialized_) {
        UpdateBuffersFromDG();
    }
}

void MeshPatch::ApplyExchange(double dt, double t) {
    if (!dg_initialized_) return;

    // Diagnostic toggle (2026-05-31): DG_DISABLE_ETA_IMPORT=1 skips importing the
    // MC free-surface into the DG coupled ghosts, so the coupled edges fall back to
    // OpenBC (self-state). Used to test whether the eta-exchange at the coarse-DG /
    // fine-MC seam is seeding the interior 2dx checkerboard mode at lake-at-rest.
    static int no_import_cache = -1;
    if (no_import_cache < 0) {
        char* buf = nullptr; size_t sz = 0;
        _dupenv_s(&buf, &sz, "DG_DISABLE_ETA_IMPORT");
        no_import_cache = (buf && buf[0] == '1') ? 1 : 0;
        if (buf) free(buf);
        if (no_import_cache) std::cerr << "[DG] eta import DISABLED via DG_DISABLE_ETA_IMPORT=1 (coupled->OpenBC)\n";
    }
    if (no_import_cache) { dg_solver_.ClearCoupledStates(); return; }

    // NOTE: ApplyExchange runs in BOTH exchange phases (AfterHalfStep and
    // AfterFullStep) but is phase-blind and always reads the MAIN incoming eta
    // buffer. The DG solver consumes coupled_surface_ only inside the NEXT
    // iteration's ComputeHalfStep (StepRK3), after the AfterFullStep exchange has
    // refreshed it, so the AfterHalfStep re-injection from (stale) main buffers is
    // harmless — it is overwritten before it is ever read.
    dg_solver_.ClearCoupledStates();

    int set_count = 0;
    int multi_edge_hits = 0;
    for (auto it = buf_to_edge_.begin(); it != buf_to_edge_.end(); ++it) {
        int buf_idx = it->first;   // == owning element index (see BuildBoundaryInterface)
        int edge_idx = it->second;
        if (buf_idx < 0 || buf_idx >= static_cast<int>(eta_in_buf_.size())) continue;

        // Coupling currency is the free surface eta (continuous across the seam)
        // plus primitive velocity. DGSolver reconstructs H = max(0, eta + D_dg) per
        // edge-quadrature point using the DG element's own bathymetry, so the
        // coupling is well-balanced even when D_dg != D_structured at the interface.
        double eta = eta_in_buf_[buf_idx];
        double u   = u_in_buf_[buf_idx];
        double v   = v_in_buf_[buf_idx];

        dg_solver_.SetCoupledSurface(edge_idx, eta, u, v);
        set_count++;

        // Hazard guard: an element owning >1 coupled edge shares one eta/u/v slot
        // across those edges (O(dy) accuracy approximation, not a stability issue).
        if (buf_to_edge_.count(buf_idx) > 1) ++multi_edge_hits;
    }

    static int log_count = 0;
    if (log_count < 5) {
        // Count how many coupled edges have state vs fallback to OpenBC
        int total_coupled = 0, has_state = 0;
        for (int e = 0; e < dg_solver_.GetMesh()->NumEdges(); ++e) {
            if (dg_solver_.GetBoundaryType(e) == DGBoundaryType::Coupled) {
                total_coupled++;
                if (dg_solver_.HasCoupledState(e)) has_state++;
            }
        }
        std::cerr << "[DG COUPLING] patch=" << patch_id_
                  << " buf_to_edge=" << buf_to_edge_.size()
                  << " set=" << set_count
                  << " coupled_edges=" << total_coupled
                  << " with_state=" << has_state
                  << " fallback_open=" << (total_coupled - has_state) << std::endl;
        if (multi_edge_hits > 0) {
            std::cerr << "[DG COUPLING] WARNING: " << multi_edge_hits
                      << " coupled-edge hit(s) on multi-edge element(s) — shared eta/u/v"
                      << std::endl;
        }
        log_count++;
    }
}

void MeshPatch::UpdateBuffersFromDG() {
    // Export cell-averaged values from DG to flat buffers
    // For interface exchange, we use boundary element averages
    int n = dg_solver_.NumElements();

    // Resize if needed
    if (static_cast<int>(H_buf_.size()) < n) {
        num_nodes_ = n;
        AllocateBuffers();
    }

    for (int k = 0; k < n; ++k) {
        SWEState avg = dg_solver_.GetCellAverage(k);
        double D = dg_solver_.GetBathymetry(k);

        H_buf_[k] = avg.H;
        double H_safe = std::max(avg.H, SWE_H_DRY);
        u_buf_[k] = avg.Hu / H_safe;
        v_buf_[k] = avg.Hv / H_safe;
        eta_buf_[k] = avg.H - D;  // eta = H - D (perturbation from rest)
    }

    // Do NOT copy into hat buffers here.  Hat buffers represent an intermediate
    // half-step state for the MacCormack scheme.  For DG the "half-step" is the
    // full RK3 result (ComputeHalfStep does the whole time advance), so the hat
    // buffers should equal the main buffers — but they are filled explicitly in
    // ComputeHalfStep after UpdateBuffersFromDG() so that the timing is clear.
    // Auto-copying them here caused StructuredPatch to see n+1 DG data as hat
    // fields, breaking the MacCormack coupling on the AfterHalfStep exchange.
}

Real* MeshPatch::GetFieldData(FieldType field) {
    switch (field) {
    case FieldType::H:       return H_buf_.data();
    case FieldType::U:       return u_buf_.data();
    case FieldType::V:       return v_buf_.data();
    case FieldType::Eta:     return eta_buf_.data();
    case FieldType::H_hat:   return H_hat_buf_.data();
    case FieldType::U_hat:   return u_hat_buf_.data();
    case FieldType::V_hat:   return v_hat_buf_.data();
    case FieldType::Eta_hat: return eta_hat_buf_.data();
    default:                 return nullptr;
    }
}

const Real* MeshPatch::GetFieldData(FieldType field) const {
    return const_cast<MeshPatch*>(this)->GetFieldData(field);
}

Real* MeshPatch::GetIncomingFieldData(FieldType field) {
    // CopyBufferToPatch writes here — separate from the DG-export buffers.
    // The DG receiver consumes the free surface Eta (continuous across a
    // bathymetry mismatch) plus primitive U/V; the total depth H is rebuilt inside
    // DGSolver from eta + local D. Raw H/H_hat are NOT consumed: return nullptr so
    // CopyBufferToPatch skips them and no stale raw H can be injected.
    switch (field) {
    case FieldType::U:       return u_in_buf_.data();
    case FieldType::V:       return v_in_buf_.data();
    case FieldType::Eta:     return eta_in_buf_.data();
    case FieldType::U_hat:   return u_hat_in_buf_.data();
    case FieldType::V_hat:   return v_hat_in_buf_.data();
    case FieldType::Eta_hat: return eta_hat_in_buf_.data();
    // H, H_hat, D: not consumed by the DG receiver — returning nullptr causes
    // CopyBufferToPatch to skip these fields silently.
    default:                 return nullptr;
    }
}

std::vector<double> MeshPatch::ProjectCoupledFlux(
    const std::string& boundary_name,
    int num_points,
    double spacing,
    double origin) const {
    std::vector<double> flux(static_cast<size_t>(std::max(num_points, 0)) * 3, 0.0);
    if (!dg_initialized_ || num_points <= 0 || spacing <= 0.0) return flux;

    const PatchInterface* iface = nullptr;
    for (const auto& pi : interfaces_) {
        if (pi.boundary_name == boundary_name) { iface = &pi; break; }
    }
    if (!iface || iface->points.empty()) return flux;

    const bool sample_along_y = (boundary_name == "left" || boundary_name == "right");

    struct EdgeFlux {
        double coordinate;
        SWEState flux;
    };
    std::vector<EdgeFlux> edge_fluxes;
    for (const auto& pt : iface->points) {
        int edge_idx = -1;
        auto range = buf_to_edge_.equal_range(pt.local_index);
        for (auto it = range.first; it != range.second; ++it) {
            edge_idx = it->second;
            break;
        }
        if (edge_idx < 0) continue;

        SWEState f;
        if (dg_solver_.GetCoupledFlux(edge_idx, f)) {
            const double edge_len = dg_solver_.GetMesh()->GetEdge(edge_idx).length;
            if (edge_len > 1e-30) {
                f.H  /= edge_len;
                f.Hu /= edge_len;
                f.Hv /= edge_len;
            }
            edge_fluxes.push_back({ sample_along_y ? pt.y : pt.x, f });
        }
    }

    if (edge_fluxes.empty()) return flux;

    std::sort(edge_fluxes.begin(), edge_fluxes.end(),
              [](const EdgeFlux& a, const EdgeFlux& b) { return a.coordinate < b.coordinate; });

    for (int point_idx = 0; point_idx < num_points; ++point_idx) {
        const double target_coordinate = origin + point_idx * spacing;
        int best = 0;
        double best_dist = std::abs(edge_fluxes[0].coordinate - target_coordinate);
        for (size_t k = 1; k < edge_fluxes.size(); ++k) {
            const double d = std::abs(edge_fluxes[k].coordinate - target_coordinate);
            if (d < best_dist) {
                best_dist = d;
                best = static_cast<int>(k);
            }
        }

        flux[point_idx * 3 + 0] = edge_fluxes[best].flux.H;
        flux[point_idx * 3 + 1] = edge_fluxes[best].flux.Hu;
        flux[point_idx * 3 + 2] = edge_fluxes[best].flux.Hv;
    }

    return flux;
}

int MeshPatch::GetPatchId() const { return patch_id_; }

std::vector<PatchInterface> MeshPatch::GetInterfaces() const { return interfaces_; }

void MeshPatch::GetLocalSize(int& nx, int& ny) const {
    if (dg_initialized_) {
        nx = dg_solver_.NumElements();
        ny = 1;
    } else {
        nx = num_nodes_;
        ny = 1;
    }
}

void MeshPatch::UpdateGlobalGrid(void* global_grid) {
    if (!dg_initialized_ || !global_grid) return;

    Grid& grid = *static_cast<Grid*>(global_grid);
    int gnx = grid.GetNx();
    int gny = grid.GetNy();

    // For each DG element, find which global grid cells its centroid falls into
    // and write the cell-averaged values there.
    // This is nearest-neighbor projection: DG centroid → closest structured cell.
    for (int k = 0; k < dg_solver_.NumElements(); ++k) {
        const auto& elem = mesh_.GetElement(k);
        double cx = elem.cx;
        double cy = elem.cy;

        // Convert physical coords to grid indices using global grid's delta_x/delta_y
        // The grid stores delta_y as uniform, delta_x may vary
        double dy = grid.GetDeltaY();
        int j = static_cast<int>(cy / dy);
        if (j < 0) j = 0;
        if (j >= gny) j = gny - 1;

        // delta_x may vary (e.g., cos(lat) for Japan Sea); use function
        double dx = grid.GetDeltaXFunc()(0, j);
        if (dx <= 0.0) dx = grid.GetDeltaX(); // fallback to cached
        int i = static_cast<int>(cx / dx);
        if (i < 0) i = 0;
        if (i >= gnx) i = gnx - 1;

        SWEState avg = dg_solver_.GetCellAverage(k);
        double D = dg_solver_.GetBathymetry(k);
        double H_safe = std::max(avg.H, 1e-10);

        grid.eta(i, j) = avg.H - D;           // surface elevation
        grid.H(i, j) = avg.H;                 // total depth
        grid.u(i, j) = avg.Hu / H_safe;       // x-velocity
        grid.v(i, j) = avg.Hv / H_safe;       // y-velocity
    }
}

// ========== Legacy compatibility ==========

void MeshPatch::SetNumNodes(int num_nodes) {
    num_nodes_ = num_nodes;
    AllocateBuffers();
}

void MeshPatch::AddInterface(const PatchInterface& iface) {
    interfaces_.push_back(iface);
}

void MeshPatch::BuildBoundaryInterface(int boundary_tag, const std::string& name) {
    const auto& edge_indices = mesh_.GetBoundaryEdges(boundary_tag);
    if (edge_indices.empty()) {
        std::cerr << "MeshPatch " << patch_id_
                  << ": No boundary edges with tag " << boundary_tag << std::endl;
        return;
    }

    PatchInterface iface;
    iface.interface_id = (patch_id_ << 16) | static_cast<int>(interfaces_.size());
    iface.patch_id_owner = patch_id_;
    iface.boundary_name = name;

    // Create one interface point per boundary edge at the edge midpoint.
    // local_index = owning element index (for buffer read/write).
    for (int ei : edge_indices) {
        const auto& edge = mesh_.GetEdge(ei);
        int elem_idx = edge.elem_left; // boundary edges always have elem_left

        InterfacePoint pt;
        pt.x = edge.mx;
        pt.y = edge.my;
        pt.local_index = elem_idx;

        iface.points.push_back(pt);

        // Store mapping for ApplyExchange: elem_idx -> edge index (multimap for corners)
        buf_to_edge_.insert({elem_idx, ei});
    }

    // Sort points along the interface (by y for vertical, by x for horizontal)
    bool is_vertical = (name == "left" || name == "right");
    std::sort(iface.points.begin(), iface.points.end(),
              [is_vertical](const InterfacePoint& a, const InterfacePoint& b) {
                  return is_vertical ? (a.y < b.y) : (a.x < b.x);
              });

    interfaces_.push_back(iface);

    std::cerr << "MeshPatch " << patch_id_ << ": Built interface '" << name
              << "' with " << iface.points.size() << " points (tag=" << boundary_tag << ")"
              << std::endl;
}

void MeshPatch::InitializeWithConstants(double H_val, double u_val, double v_val, double eta_val) {
    std::fill(H_buf_.begin(), H_buf_.end(), H_val);
    std::fill(u_buf_.begin(), u_buf_.end(), u_val);
    std::fill(v_buf_.begin(), v_buf_.end(), v_val);
    std::fill(eta_buf_.begin(), eta_buf_.end(), eta_val);
    H_hat_buf_ = H_buf_;
    u_hat_buf_ = u_buf_;
    v_hat_buf_ = v_buf_;
    eta_hat_buf_ = eta_buf_;
}
