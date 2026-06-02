// MeshPatch.h - DG solver patch for unstructured triangular mesh
// Implements IPatch interface using DGSolver for time integration

#ifndef MESH_PATCH_H
#define MESH_PATCH_H

#include "IPatch.h"
#include "TriMesh.h"
#include "MeshReader.h"
#include "ElementMatrices.h"
#include "DGSolver.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>

/// MeshPatch - DG solver on unstructured triangular mesh
///
/// Wraps DGSolver and implements IPatch for integration with
/// the hybrid architecture (ExchangeEngine, SimulationRunner).
///
/// Two-step scheme for compatibility with MacCormack architecture:
/// - ComputeHalfStep: RK2 stage 1
/// - ComputeFullStep: RK2 stage 2
class MeshPatch : public IPatch {
public:
    /// Construct with patch ID and number of interface nodes
    /// @param patch_id Unique patch identifier
    /// @param num_nodes Number of interface nodes (for legacy compatibility)
    MeshPatch(int patch_id, int num_nodes);

    virtual ~MeshPatch() = default;

    // ========== IPatch Interface Implementation ==========

    void Initialize() override;
    void ComputeHalfStep() override;
    void ComputeFullStep() override;
    void PrepareExchange(double dt, double t) override;
    void ApplyExchange(double dt, double t) override;
    Real* GetFieldData(FieldType field) override;
    const Real* GetFieldData(FieldType field) const override;

    /// Overridden to return incoming buffers (H_in_buf_ etc.) so that
    /// CopyBufferToPatch does not overwrite the DG-solution export buffers.
    Real* GetIncomingFieldData(FieldType field) override;
    int GetPatchId() const override;
    std::vector<PatchInterface> GetInterfaces() const override;
    void GetLocalSize(int& nx, int& ny) const override;
    void UpdateGlobalGrid(void* global_grid) override;

    // ========== Mesh-specific methods ==========

    /// Load mesh from Gmsh .msh file
    bool LoadMesh(const std::string& filename);

    /// Set mesh directly (e.g., programmatically generated)
    void SetMesh(TriMesh&& mesh);

    /// Set bathymetry from function D(x, y)
    void SetBathymetry(std::function<double(double, double)> func);

    /// Set initial conditions from function -> (H, Hu, Hv)
    void SetInitialCondition(std::function<SWEState(double, double)> func);

    /// Well-balanced IC: build H from projected bathymetry + projected eta so the
    /// DG stripe is lake-at-rest consistent on real bathymetry. Call after
    /// SetBathymetry. d_func must be the SAME bathymetry passed to SetBathymetry.
    void SetInitialConditionWellBalanced(std::function<SWEState(double, double)> q_func,
                                         std::function<double(double, double)> d_func);

    /// Set boundary condition for a physical tag
    void SetBoundaryCondition(int tag, DGBoundaryType type);

    /// Set time step (if 0, auto-computed from CFL)
    void SetTimeStep(double dt) { dt_ = dt; }

    /// Set CFL target for auto time step
    void SetCFL(double cfl) { cfl_target_ = cfl; }

    /// Add interface for exchange with other patches
    void AddInterface(const PatchInterface& iface);

    /// Build interface from mesh boundary edges with a given tag
    /// Creates interface points at edge midpoints, sorted along the boundary.
    /// @param boundary_tag Gmsh physical tag for the boundary edges
    /// @param name Interface name ("left", "right", "top", "bottom")
    void BuildBoundaryInterface(int boundary_tag, const std::string& name);

    /// Project DG coupled fluxes onto a structured interface.
    /// Returns [num_points * 3]: (F_H, F_Hu, F_Hv) sampled along the interface.
    /// `spacing` and `origin` describe the receiver grid in physical space.
    std::vector<double> ProjectCoupledFlux(const std::string& boundary_name,
                                           int num_points,
                                           double spacing,
                                           double origin = 0.0) const;

    /// Legacy: initialize with constants
    void InitializeWithConstants(double H_val, double u_val, double v_val, double eta_val);

    /// Legacy: set number of nodes
    void SetNumNodes(int num_nodes);

    /// Access DG solver directly
    const DGSolver& GetDGSolver() const { return dg_solver_; }
    DGSolver& GetDGSolver() { return dg_solver_; }

    /// Access mesh
    const TriMesh& GetMesh() const { return mesh_; }

private:
    int patch_id_;
    int num_nodes_; // Interface node count (legacy)

    // DG infrastructure
    TriMesh mesh_;
    ElementMatrices elem_matrices_;
    DGSolver dg_solver_;
    bool dg_initialized_ = false;

    double dt_ = 0.0;           // Time step (0 = auto)
    double cfl_target_ = 0.3;   // CFL target for auto dt

    // Interface exchange buffers — OUTGOING (DG solution → neighbour)
    // Populated by UpdateBuffersFromDG(), read by FillBufferFromPatch.
    std::vector<Real> H_buf_;
    std::vector<Real> u_buf_;
    std::vector<Real> v_buf_;
    std::vector<Real> eta_buf_;
    std::vector<Real> H_hat_buf_;
    std::vector<Real> u_hat_buf_;
    std::vector<Real> v_hat_buf_;
    std::vector<Real> eta_hat_buf_;

    // Interface exchange buffers — INCOMING (neighbour → DG coupled boundary)
    // Written by CopyBufferToPatch (via GetIncomingFieldData), read by ApplyExchange
    // to set the DG coupled surface state. The DG receiver consumes the free
    // surface eta (continuous across a bathymetry mismatch) plus primitive (u,v);
    // total depth H is reconstructed inside DGSolver per edge-quadrature point as
    // max(0, eta + D_dg). Raw H is NOT consumed (H_in_* buffers retained only for
    // transport compatibility and are unused on the DG side).
    // Kept separate from H_buf_ so CopyBufferToPatch cannot corrupt the DG export.
    std::vector<Real> H_in_buf_;
    std::vector<Real> u_in_buf_;
    std::vector<Real> v_in_buf_;
    std::vector<Real> eta_in_buf_;
    std::vector<Real> H_hat_in_buf_;
    std::vector<Real> u_hat_in_buf_;
    std::vector<Real> v_hat_in_buf_;
    std::vector<Real> eta_hat_in_buf_;

    // Patch interfaces
    std::vector<PatchInterface> interfaces_;

    // Mapping: buffer index (elem_idx) -> mesh edge indices (for coupled boundary injection)
    // An element can have multiple boundary edges (e.g., corner elements)
    std::unordered_multimap<int, int> buf_to_edge_;

    /// Rebuild exchange buffers from DG solution
    void UpdateBuffersFromDG();

    /// Allocate flat buffer arrays
    void AllocateBuffers();
};

#endif // MESH_PATCH_H
