// MacCormackCUDA.cuh
// Template-based CUDA simulator for MacCormack shallow water equations.
// Supports FP32 and FP64 via runtime dispatch on top of compile-time MacCormackCUDA<Real> instantiations.
//
// Usage:
//   auto sim = CreateMacCormackCUDA(nx, ny, params, /*use_fp32=*/false);
//   sim->LoadInitialData(...); sim->Step(t); sim->CopyToHost(...);
//
// Host-facing API uses double everywhere; conversion to/from Real happens inside the simulator.

#ifndef MACCORMACK_CUDA_CUH
#define MACCORMACK_CUDA_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>

// ================== CUDA Error Checking ==================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ================== Configuration ==================
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 16
#endif

#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 16
#endif

// ================== GPU Data Structure (templated on Real) ==================
template <typename Real>
struct DeviceGridDataT {
    Real* H = nullptr;          // Total water depth (H = D + eta)
    Real* u = nullptr;          // Velocity x
    Real* v = nullptr;          // Velocity y
    Real* eta = nullptr;        // Water level deviation
    Real* D = nullptr;          // Bathymetry (bottom depth)

    Real* H_hat = nullptr;      // Intermediate values (predictor step)
    Real* u_hat = nullptr;
    Real* v_hat = nullptr;
    Real* eta_hat = nullptr;

    Real* eta_max = nullptr;    // Maximum deviation (optional)
    Real* u_max = nullptr;      // Maximum |u| (optional, for save_velocities)
    Real* v_max = nullptr;      // Maximum |v| (optional, for save_velocities)

    Real* tvd_dEta = nullptr;   // TVD delta buffers (allocated only if tvd_enabled)
    Real* tvd_dHu = nullptr;
    Real* tvd_dHv = nullptr;

    Real* delta_x = nullptr;    // Delta x array (if variable)

    int nx = 0;
    int ny = 0;
    size_t pitch = 0;
};

// ================== Coordinate System (matches CPU CoordSystem.h) ==================
enum class CudaCoordSystem : int {
    Cartesian = 0,
    Spherical = 1
};

// ================== Simulation Parameters ==================
struct SimulationParams {
    double g;               // Gravity (9.81 m/s²)
    double tau;             // Time step
    double delta_x_const;   // Constant delta x (if used)
    double delta_y;         // Delta y
    double min_calc_value;  // Minimum depth for calculation (wet/dry threshold)
    bool use_variable_dx;   // Use variable delta_x

    // Boundary condition type for BASE variables (0 = static, 2 = test3, 3 = test4/5).
    // Maps to CPU BoundaryConditionFunctions index: 0=Data1, 2=Data3, 3=Data4.
    int boundaryConditionType = 0;
    // Boundary condition type for HAT variables (predictor stage).
    // CPU uses different functions for Hat vs Base (e.g. Hat3 zeroes v_hat on top, Data3 doesn't).
    // Defaults to boundaryConditionType when not explicitly set.
    int hatBoundaryConditionType = 0;

    // Coordinate system: Cartesian (no Coriolis) or Spherical (Coriolis on beta-plane).
    CudaCoordSystem coord_system = CudaCoordSystem::Cartesian;
    // Whether to apply Coriolis force (only meaningful for Spherical coords).
    bool coriolis_enabled = false;
    // Central latitude in radians (45° default). Used by Spherical kernels to compute
    // delta_phi = delta_y / R_EARTH and f = 2*OMEGA*sin(phi).
    double center_latitude_rad = 0.785398163;

    // === Davis 1987 TVD post-correction (port of CPU SubGrid::ApplyTvdPostCorrection) ===
    // Stabilizes coastal-shoaling Gibbs oscillations. Applied to (eta, Hu, Hv) for
    // well-balancedness. Default OFF → existing tests bitwise-identical.
    bool   tvd_enabled = false;
    bool   tvd_force_minmod = false;   // true → minmod everywhere; false → van Leer (+ minmod near shallow)
    double tvd_shallow_depth = 100.0;  // auto-switch to minmod where any stencil D < this (m)
    int    tvd_boundary_skip = 0;      // skip TVD within N cells of physical edges (i and j)
    int    tvd_apply_every = 1;        // apply TVD every Nth step

    // === North absorbing (Rayleigh) layer on velocities (CPU ApplyNorthAbsorbing*) ===
    bool   absorb_enabled = false;
    int    absorb_layer_width = 96;    // cells from north boundary (j=0)
    double absorb_alpha_max = 0.05;    // = sigma_max*dt; damping at j=0

    // === Active Region (GPU): compute only the active row band [0, j_hi) ===
    // Port of CPU ActiveRegionManager (geometric expansion). Source is near the
    // north edge (j small); wave propagates south (increasing j). We anchor the
    // band at j=0 and grow j_hi southward by the physical cone each step, so the
    // 2D interior kernels run only over active rows. Cells beyond j_hi stay at
    // lake-at-rest; correctness requires j_hi to enclose the dependency cone
    // (physical c_max·t + stencil + margin), identical guarantee to the CPU AR.
    bool   ar_enabled = false;
    double ar_c_max = 0.0;             // sqrt(g*max_depth); 0 → derive disabled
    int    ar_seed_j_max = 0;          // southernmost source row (device j)
    int    ar_margin_cells = 32;       // safety pad beyond the cone
};

// ================== Subgrid Info ==================
// Information about this subgrid's position in the global domain
struct SubgridInfo {
    bool has_left_neighbor;   // True if there's a subgrid to the left
    bool has_right_neighbor;  // True if there's a subgrid to the right
    int subgrid_index;        // Index of this subgrid (0 to num_subgrids-1)
    int num_subgrids;         // Total number of subgrids

    SubgridInfo() : has_left_neighbor(false), has_right_neighbor(false),
                    subgrid_index(0), num_subgrids(1) {}
};

// ================== Abstract Interface (precision-agnostic) ==================
// The wrapper class uses this interface so the same exe can carry both fp32 and fp64
// implementations and pick at runtime via CLI flag.
class IMacCormackCUDA {
public:
    virtual ~IMacCormackCUDA() = default;

    virtual void LoadInitialData(const double* h_H, const double* h_u, const double* h_v,
                                 const double* h_eta, const double* h_D) = 0;
    virtual void LoadDeltaXArray(const double* h_delta_x, int size) = 0;
    virtual void SetSubgridInfo(const SubgridInfo& info) = 0;

    virtual void Step(int timestep) = 0;
    virtual void ComputePredictorStep() = 0;
    virtual void ApplyHatBoundaryConditions(int timestep) = 0;
    virtual void ComputeCorrectorStep() = 0;
    virtual void ApplyBaseBoundaryConditions(int timestep) = 0;

    virtual void UpdateEtaMax() = 0;
    virtual void UpdateUVMax() = 0;
    virtual double GetGlobalEtaMax() = 0;

    virtual void CopyToHost(double* h_H, double* h_u, double* h_v, double* h_eta) = 0;
    virtual void CopyEtaToHost(double* h_eta) = 0;
    virtual void CopyEtaMaxToHost(double* h_eta_max) = 0;
    virtual void CopyUMaxToHost(double* h_u_max) = 0;
    virtual void CopyVMaxToHost(double* h_v_max) = 0;

    virtual void Synchronize() = 0;
    virtual int GetNx() const = 0;
    virtual int GetNy() const = 0;
    virtual int GetActiveRows() const = 0;   // active row count (AR); ny if AR off

    virtual const char* GetPrecisionName() const = 0;

    // Direct device access (used by subgrid pipeline). Only callable from the matching
    // precision; mostly used to copy hat buffers between subgrids on the host.
    virtual void DownloadHatToHost(double* h_H_hat, double* h_u_hat,
                                   double* h_v_hat, double* h_eta_hat) = 0;
    virtual void UploadHatFromHost(const double* h_H_hat, const double* h_u_hat,
                                   const double* h_v_hat, const double* h_eta_hat) = 0;
};

// ================== Templated Simulator Class ==================
template <typename Real>
class MacCormackCUDA : public IMacCormackCUDA {
public:
    MacCormackCUDA(int nx, int ny, const SimulationParams& params);
    ~MacCormackCUDA() override;

    MacCormackCUDA(const MacCormackCUDA&) = delete;
    MacCormackCUDA& operator=(const MacCormackCUDA&) = delete;

    // ---- IMacCormackCUDA implementation ----
    void LoadInitialData(const double* h_H, const double* h_u, const double* h_v,
                         const double* h_eta, const double* h_D) override;
    void LoadDeltaXArray(const double* h_delta_x, int size) override;
    void SetSubgridInfo(const SubgridInfo& info) override { subgrid_info_ = info; }

    void Step(int timestep) override;
    void ComputePredictorStep() override;
    void ApplyHatBoundaryConditions(int timestep) override;
    void ComputeCorrectorStep() override;
    void ApplyBaseBoundaryConditions(int timestep) override;

    void UpdateEtaMax() override;
    void UpdateUVMax() override;
    double GetGlobalEtaMax() override;

    void CopyToHost(double* h_H, double* h_u, double* h_v, double* h_eta) override;
    void CopyEtaToHost(double* h_eta) override;
    void CopyEtaMaxToHost(double* h_eta_max) override;
    void CopyUMaxToHost(double* h_u_max) override;
    void CopyVMaxToHost(double* h_v_max) override;

    void Synchronize() override;
    int GetNx() const override { return nx_; }
    int GetNy() const override { return ny_; }
    int GetActiveRows() const override { return ar_enabled_ ? ar_j_hi_ : ny_; }
    const char* GetPrecisionName() const override;

    void DownloadHatToHost(double* h_H_hat, double* h_u_hat,
                           double* h_v_hat, double* h_eta_hat) override;
    void UploadHatFromHost(const double* h_H_hat, const double* h_u_hat,
                           const double* h_v_hat, const double* h_eta_hat) override;

    // Direct access to typed device data (for advanced inter-GPU operations within same precision).
    DeviceGridDataT<Real>& GetDeviceData() { return d_data_; }

private:
    int nx_;
    int ny_;
    SimulationParams params_;
    SubgridInfo subgrid_info_;
    DeviceGridDataT<Real> d_data_;
    Real* d_eta_max_reduced_ = nullptr;

    dim3 blockDim_;
    dim3 gridDim_;
    dim3 boundaryBlockDim_;
    dim3 boundaryGridDim_X_;  // For X boundaries (process columns)
    dim3 boundaryGridDim_Y_;  // For Y boundaries (process rows)

    void AllocateDeviceMemory();
    void FreeDeviceMemory();
    void SetupCudaDimensions();

    // Stabilizers (port of CPU SubGrid). Called from Step() when enabled in params_.
    void ApplyTvdPostCorrection(int timestep);
    void ApplyNorthAbsorbingMain();
    void ApplyNorthAbsorbingHat();

    // Active Region: current active row count [0, ar_j_hi_); grows southward.
    int ar_j_hi_ = 0;
    void UpdateActiveWindow(int timestep);     // recompute ar_j_hi_ from the cone
    dim3 ActiveGridDim() const;                // gridDim covering [0, ar_j_hi_) rows

    bool is_initialized_;
    bool has_variable_dx_;
    bool ar_enabled_ = false;   // mirror of params_.ar_enabled (set in ctor)

    // Host scratch buffers reused across LoadInitialData / CopyToHost when Real != double
    // (avoids allocating a temporary every step).
    std::vector<Real> h_scratch_;
};

// Factory: creates fp32 or fp64 instance behind the IMacCormackCUDA interface.
std::unique_ptr<IMacCormackCUDA> CreateMacCormackCUDA(int nx, int ny,
                                                     const SimulationParams& params,
                                                     bool use_fp32);

#endif // MACCORMACK_CUDA_CUH
