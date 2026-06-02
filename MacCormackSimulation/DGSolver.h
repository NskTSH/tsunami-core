// DGSolver.h - Discontinuous Galerkin solver for shallow water equations
// Assembles the DG system and advances the solution in time

#ifndef DG_SOLVER_H
#define DG_SOLVER_H

#include "TriMesh.h"
#include "TaylorBasis.h"
#include "ElementMatrices.h"
#include "RiemannSolver.h"
#include "BlockSparseMatrix.h"
#include <vector>
#include <functional>

/// Boundary condition type for DG edges
enum class DGBoundaryType {
    Wall,     // Solid wall (free-slip, reflect normal velocity)
    Open,     // Open boundary (radiation)
    Coupled   // Coupled boundary (state provided by external patch)
};

/// DG solver configuration: centralizes numerical parameters and tolerances
struct DGConfig {
    // --- Slope limiter ---
    double limiter_jump_threshold = 0.1;    // Face jump / scale threshold for troubled cell
    double limiter_oscillation_threshold = 0.5; // Mode energy / avg^2 threshold
    double limiter_bj_tol = 1e-14;          // BJ limiter floating-point tolerance

    // --- Positivity limiter ---
    double dry_element_factor = 10.0;       // H_avg < factor * H_DRY → zero momentum

    // --- Wet/dry handling ---
    double wet_dry_threshold = 10.0;        // H_avg below this triggers semi-dry treatment (P0 fallback + velocity limiting)
    double max_velocity = 20.0;             // Maximum velocity [m/s] — cap Hu/Hv to prevent explosion at wet/dry fronts

    // --- Numerical safety ---
    double zero_area_guard = 1e-30;         // Guard for 1/area divisions
    double zero_dist_guard = 1e-30;         // Guard for 1/dist^2 divisions
    double min_wave_speed = 1e-10;          // Skip CFL contribution if wave speed below this
    double dt_upper_bound = 1e30;           // Initial dt bound for CFL reduction

    // --- Well-balanced ---
    bool well_balanced = true;              // Use hydrostatic reconstruction (Audusse et al.)
};

/// DG solver for 2D shallow water equations on triangular mesh
///
/// Conservative variables: q = (H, Hu, Hv)
/// H = total depth, Hu = x-momentum, Hv = y-momentum
///
/// Each element stores DG_BASIS_SIZE coefficients per variable,
/// total DOF per element = DG_BLOCK_SIZE = 3 * DG_BASIS_SIZE
///
/// Time integration: forward Euler, SSP-RK2, SSP-RK3 (Shu-Osher)
class DGSolver {
public:
    DGSolver() = default;

    /// Initialize solver with mesh and precomputed matrices
    /// @param mesh Triangular mesh (must outlive the solver)
    /// @param elem_matrices Precomputed element matrices (must outlive the solver)
    void Initialize(const TriMesh* mesh, const ElementMatrices* elem_matrices);

    /// Set bathymetry (still-water depth D, convention: eta = H - D) at element centroids
    /// @param bathymetry Vector of size NumElements, D values at centroids
    void SetBathymetry(const std::vector<double>& bathymetry);

    /// Set bathymetry from a function D(x, y)
    void SetBathymetryFunction(std::function<double(double, double)> func);

    /// Set initial conditions from a function q(x, y) -> (H, Hu, Hv)
    void SetInitialCondition(std::function<SWEState(double, double)> func);

    /// Well-balanced initial condition: build H coefficients from the ALREADY
    /// projected bathymetry D_coeffs_ plus the projected free-surface eta, so that
    /// H_pt - D_pt = eta_pt holds at every quadrature point (= 0 at lake-at-rest).
    /// Projecting H and D independently (as SetInitialCondition + SetBathymetry do)
    /// leaves H_pt - D_pt != eta on steep/discontinuous real bathymetry (measured
    /// 158 m rest-imbalance on Japan Sea S7), which seeds a spurious bed-slope
    /// source that amplifies into divergence. Call AFTER SetBathymetryFunction.
    /// @param q_func  (x,y) -> (H, Hu, Hv), same field as SetInitialCondition
    /// @param d_func  (x,y) -> D, the SAME bathymetry passed to SetBathymetryFunction
    void SetInitialConditionWellBalanced(std::function<SWEState(double, double)> q_func,
                                         std::function<double(double, double)> d_func);

    /// Set boundary condition type for a given boundary tag
    void SetBoundaryCondition(int boundary_tag, DGBoundaryType type);

    /// Default boundary condition for untagged boundaries
    void SetDefaultBoundaryCondition(DGBoundaryType type);

    /// Advance solution by one time step dt
    /// Uses explicit forward Euler: q^{n+1} = q^n + dt * M^{-1} * L(q^n)
    /// @param dt Time step
    void Step(double dt);

    /// Advance solution using SSP-RK2 (Heun's method)
    /// @param dt Time step
    void StepRK2(double dt);

    /// Advance solution using SSP-RK3 (Shu-Osher)
    /// Unlocks full spatial order O(h^3) for P2 elements
    /// @param dt Time step
    void StepRK3(double dt);

    /// Apply TVB-modified Barth-Jespersen slope limiter for DG P2
    /// Detects troubled cells via vertex-based indicator, reduces to P1 + limits linear modes
    void ApplySlopeLimiter();

    /// Apply positivity-preserving limiter (Zhang & Shu)
    /// Ensures H >= 0 at vertices and edge quadrature points
    void ApplyPositivityLimiter();

    /// Well-balanced spectral-viscosity filter: damp the QUADRATIC (top) P2 modes
    /// of the deviation eta = H - D (and of momentum) by a factor sigma in (0,1) to
    /// suppress the weakly-unstable 2Δx mode from quadrature-aliased steep-bathymetry
    /// pressure/source feedback. Strict no-op at lake-at-rest (eta_l = Hu_l = Hv_l = 0).
    /// Env DG_MODAL_FILTER (default ON), DG_FILTER_SIGMA (default 0.5).
    void ApplyModalFilter();

    /// Sensor-gated P0 (cell-average) Rayleigh damping of momentum (Hu_0, Hv_0) on
    /// STEEP-bathymetry cells, to stabilize the mode-0 mass instability via the
    /// momentum leg (Codex review 2026-05-31). No-op at rest (Hu=Hv=0) and on flat
    /// bathymetry (sensor chi=0). Env DG_STEEP_DAMP (default ON). @param dt RK step.
    void ApplySteepMomentumDamping(double dt);

    /// Freeze permanently dry elements (D < wet_dry_threshold):
    /// set H=D, Hu=Hv=0 after every RK stage. Called from ApplyPositivityLimiter.
    void EnforceDryCells();

    /// Compute CFL-limited time step
    /// @param cfl_target Target CFL number (typically 0.3-0.5 for DG)
    double ComputeTimeStep(double cfl_target = 0.3) const;

    /// Enable/disable slope limiter
    void SetSlopeLimiter(bool enabled, double tvb_M = 0.0) {
        use_slope_limiter_ = enabled;
        tvb_M_ = tvb_M;
    }

    /// Access solver configuration
    DGConfig& Config() { return config_; }
    const DGConfig& Config() const { return config_; }

    /// Mesh and boundary access (for coupling diagnostics)
    const TriMesh* GetMesh() const { return mesh_; }
    DGBoundaryType GetBoundaryType(int edge_idx) const;
    bool HasCoupledState(int edge_idx) const { return coupled_surface_.count(edge_idx) > 0; }

    /// Diagnostic: max |H_pt − D_pt| over all volume quadrature points.
    /// For lake-at-rest this should be ~roundoff if init/projection is correct.
    double MaxRestImbalance() const;

    /// Diagnostic: the DISCRETE OPERATOR residual at the current state, i.e.
    /// max_k || M^{-1} L(q) ||_2 over elements (and, via the out-params, the
    /// momentum-only component and the worst element index). For a TRUE
    /// well-balanced scheme at lake-at-rest this must be ~roundoff. Unlike
    /// MaxRestImbalance (which only checks H_pt==D_pt pointwise), this measures
    /// whether the time derivative dq/dt actually vanishes at rest — exposing
    /// quadrature-aliasing C-property violations (degree-5 volume/source vs
    /// degree-4 rule on quadratic bathymetry). Runs one ComputeResidual; not const.
    /// @param mom_only   out: max element momentum-only residual norm
    /// @param worst_elem out: index of the element with the largest residual
    double RestResidual(double& mom_only, int& worst_elem,
                        double* worst_wet = nullptr, int* worst_wet_elem = nullptr);

    /// Diagnostic: max over wet elements (D_[k] >= wet_dry_threshold) and over the
    /// three H modes l of |q_[H mode l] - D_coeffs_[l]|. Fix A makes the rest
    /// operator residual IDENTICALLY zero when H_coeffs == D_coeffs (eta == 0 at
    /// every quad point), so any residual at lake-at-rest must come from a MODAL
    /// mismatch H_coeffs != D_coeffs. MaxRestImbalance only samples 6 volume
    /// points and can miss a P2 modal difference that vanishes there; this
    /// inspects the modal coefficients directly. @param worst_elem out: worst element.
    double MaxModalRestDiff(int& worst_elem) const;

    /// Diagnostic: min reconstructed bathymetry D_pt over volume+edge quadrature
    /// points of WET elements. A NEGATIVE value reveals a P2 overshoot D_pt<0 on
    /// steep seafloor that the H_safe=max(H,0)/Audusse clipping turns into a
    /// spurious rest residual (eta = 0 - D_pt > 0) even when H_coeffs==D_coeffs.
    double MinBathymetryTrace(int& worst_elem) const;

    /// Diagnostic: per-mode max |H_coeff_l - D_coeff_l| over wet elements (size DG_BASIS_SIZE).
    void ModalImbalanceBreakdown(double* out_per_mode) const;

    /// Get stored HLL flux on a coupled boundary edge (set during last ComputeResidual)
    /// Returns (F_H, F_Hu, F_Hv) integrated over edge length (not per unit length)
    bool GetCoupledFlux(int edge_idx, SWEState& flux) const {
        auto it = coupled_fluxes_.find(edge_idx);
        if (it != coupled_fluxes_.end()) { flux = it->second; return true; }
        return false;
    }
    const std::unordered_map<int, SWEState>& GetCoupledFluxes() const { return coupled_fluxes_; }

    // ========== FIELD ACCESS ==========

    /// Get total number of DOFs
    int TotalDOFs() const { return num_elements_ * DG_BLOCK_SIZE; }

    /// Get solution coefficients (H, Hu, Hv) for all elements
    /// Layout: [elem0_H_coeffs, elem0_Hu_coeffs, elem0_Hv_coeffs, elem1_H_coeffs, ...]
    const std::vector<double>& GetSolution() const { return q_; }
    std::vector<double>& GetSolution() { return q_; }

    /// Get the (positivity-limited) modal bathymetry coefficients D_coeffs_.
    /// Layout: D_coeffs_[k * DG_BASIS_SIZE + l]. Needed by the CUDA DG port to
    /// upload the exact host bathymetry (incl. the bathy-positivity fix) to device.
    const std::vector<double>& GetDCoeffs() const { return D_coeffs_; }

    /// Access the mesh / element matrices (for the CUDA port to read precomputed
    /// per-element/edge geometry + mass/quadrature data to upload to the device).
    const TriMesh* GetMeshPtr() const { return mesh_; }
    const ElementMatrices* GetElementMatricesPtr() const { return elem_matrices_; }

    /// Evaluate solution at a physical point (x, y) in element elem_idx
    SWEState EvaluateAt(int elem_idx, double x, double y) const;

    /// Get cell-averaged values for element (first coefficient of each variable)
    SWEState GetCellAverage(int elem_idx) const;

    /// Get bathymetry at element centroid
    double GetBathymetry(int elem_idx) const { return D_[elem_idx]; }

    int NumElements() const { return num_elements_; }

    /// Set external free-surface state for a coupled boundary edge.
    /// Stores the PRIMITIVE surface (eta, u, v); the total depth H is
    /// reconstructed per edge-quadrature point as max(0, eta + D_dg) using the
    /// DG element's own bathymetry, so the coupling stays well-balanced even when
    /// the neighbour's bathymetry differs (eta is continuous, depth is local).
    /// @param edge_idx Global edge index in the mesh
    /// @param eta Free-surface elevation from the external patch (eta = H - D)
    /// @param u,v Primitive velocity from the external patch
    void SetCoupledSurface(int edge_idx, double eta, double u, double v);

    /// Clear all coupled boundary states (call before each exchange cycle)
    void ClearCoupledStates();

    /// Set a uniform far-field reference state for all Open boundary edges.
    /// Used by the characteristic / Flather radiation BC. When set (default
    /// disabled), Open boundaries use Riemann-invariant decomposition with
    /// this state as the ambient; otherwise they fall back to zero-gradient
    /// (qGhost = qInner) for backward compatibility.
    void SetUniformOpenReference(const SWEState& q_far);

    /// Set the far-field reference for Open edges from a function of (x, y).
    /// Sampled at edge midpoints and stored per-edge. Useful when the rest
    /// state varies across the boundary (e.g. non-uniform bathymetry).
    void SetOpenReferenceFunction(std::function<SWEState(double, double)> func);

    /// Clear all Open boundary far-field references (fall back to legacy BC).
    void ClearOpenReference();

private:
    const TriMesh* mesh_ = nullptr;
    const ElementMatrices* elem_matrices_ = nullptr;

    int num_elements_ = 0;

    // Solution vector: q[k * DG_BLOCK_SIZE + var * DG_BASIS_SIZE + l]
    // var: 0=H, 1=Hu, 2=Hv
    // l: basis function index
    std::vector<double> q_;

    // Right-hand side / residual
    std::vector<double> rhs_;

    // Temporary storage for RK stages
    std::vector<double> q_tmp_;
    std::vector<double> q_tmp2_; // Second temp for RK3

    // Bathymetry (still-water depth D, convention: eta = H - D) at element centroids
    std::vector<double> D_;

    // Bathymetry DG coefficients: D_coeffs_[k * DG_BASIS_SIZE + l]
    // Projected onto Taylor basis for well-balanced source term and hydrostatic reconstruction
    std::vector<double> D_coeffs_;

    // Boundary condition map: boundary_tag -> DGBoundaryType
    std::unordered_map<int, DGBoundaryType> bc_map_;
    DGBoundaryType default_bc_ = DGBoundaryType::Wall;

    // Coupled boundary external state: edge_idx -> primitive surface (eta, u, v).
    // H is reconstructed per edge-quadrature point as max(0, eta + D_dg) using the
    // DG element's own bathymetry, so the coupling is well-balanced across a
    // bathymetry mismatch at the seam (eta is continuous, depth is each side's own).
    struct CoupledSurface { double eta = 0.0; double u = 0.0; double v = 0.0; };
    std::unordered_map<int, CoupledSurface> coupled_surface_;
    std::unordered_map<int, SWEState> coupled_fluxes_; // HLL flux on coupled edges (from last residual)

    // Far-field reference for Open boundary radiation BC.
    // When use_open_radiation_bc_ is true and an edge has a reference state
    // (per-edge map or uniform default), Open BC uses Flather/characteristic
    // form; otherwise falls back to qGhost = qInner.
    bool use_open_radiation_bc_ = false;
    SWEState open_ref_uniform_{0.0, 0.0, 0.0};
    bool open_ref_uniform_set_ = false;
    std::unordered_map<int, SWEState> open_ref_per_edge_;

    // Solver configuration
    DGConfig config_;

    // Slope limiter settings
    bool use_slope_limiter_ = false;
    double tvb_M_ = 0.0;  // TVB constant (0 = strict minmod)

    // Precomputed diagonal inverse of expanded mass matrix (3x3 block-diagonal of M_inv)
    std::vector<std::array<double, DG_BLOCK_SIZE * DG_BLOCK_SIZE>> mass_inv_expanded_;

    /// Compute the spatial residual L(q): volume + surface integrals
    /// Stores result in rhs_
    void ComputeResidual(const double* q_in);

    /// Volume integral contribution for one element
    void VolumeIntegral(int elem_idx, const double* q_in, double* rhs_out);

    /// Surface integral for one element (processes its 3 edges)
    /// Thread-safe: only writes to this element's rhs block
    void SurfaceIntegralElement(int elem_idx, const double* q_in, double* rhs_out);

    /// Edge-based surface integral: processes each edge ONCE (D2 optimization).
    /// Interior edges: compute flux once, accumulate into both adjacent elements.
    /// Eliminates redundant Riemann solves and basis evaluations.
    /// NOT thread-safe (writes to two elements' rhs blocks per edge).
    void SurfaceIntegralEdgeBased(const double* q_in, double* rhs_out);

    /// Find local edge index within an element (private helper)
    // GetBoundaryType, GetMesh, HasCoupledState moved to public section

    /// Build expanded mass matrix inverse (block-diagonal 3*M_inv)
    void BuildExpandedMassInverse();

    /// Project bathymetry D onto DG basis via L2 projection (from centroid values, neighbor gradient)
    void ProjectBathymetry();

    /// Project bathymetry from a function using Gauss quadrature (exact for polynomials up to order 4)
    void ProjectBathymetryFromFunction(std::function<double(double, double)> func);

    /// Source term: bathymetry gradient contribution
    void SourceTerm(int elem_idx, const double* q_in, double* rhs_out);
};

#endif // DG_SOLVER_H
