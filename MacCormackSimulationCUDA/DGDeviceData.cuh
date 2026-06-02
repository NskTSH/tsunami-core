// DGDeviceData.cuh - flat Structure-of-Arrays for the DG-on-GPU solver.
//
// All host-side DG setup (mesh, mass matrices, quadrature, IC, bathy-positivity)
// is built by the reused CPU DGSolver; this file defines the flat device layout
// and a host "snapshot" that extracts everything the per-step kernels need, in a
// CANONICAL per-edge orientation (so the element-/edge-based surface flux is
// identical from both sides — required for conservation + well-balancedness).
// FP64 throughout (well-balanced cancellation on deep bathymetry needs it).

#ifndef DG_DEVICE_DATA_CUH
#define DG_DEVICE_DATA_CUH

#include <vector>

// DG sizes (mirror TaylorBasis.h / ElementMatrices.h; checked at upload time).
#define DGC_BASIS   6      // DG_BASIS_SIZE (P2)
#define DGC_NVARS   3      // H, Hu, Hv
#define DGC_BLOCK   18     // DGC_NVARS * DGC_BASIS
#define DGC_VOLNP   6      // volume Dunavant points (TRI_ORDER4_NPTS)
#define DGC_EDGENP  3      // edge Gauss points (EDGE_ORDER3_NPTS)

// Edge boundary kinds for the DG surface kernel (match DGBoundaryType + interior).
enum DGEdgeKind : int { DG_EDGE_INTERIOR = 0, DG_EDGE_WALL = 1, DG_EDGE_OPEN = 2, DG_EDGE_COUPLED = 3 };

// ---- Host snapshot (extracted once from the CPU DGSolver; std::vector flat) ----
struct DGSnap {
    int N = 0;  // num elements
    int E = 0;  // num edges

    // Per-element (size N * stride)
    std::vector<double> q;          // [N*18] state (H,Hu,Hv x 6 modes)
    std::vector<double> Dcoeffs;    // [N*6] positivity-limited modal bathymetry
    std::vector<double> Dcell;      // [N]   centroid depth D_[k] (dry/wet threshold test)
    std::vector<double> Minv;       // [N*36] inverse mass matrix (per variable block)
    std::vector<double> vol_phi;    // [N*6*6]  phi_l at 6 vol quad pts
    std::vector<double> vol_dphidx; // [N*6*6]
    std::vector<double> vol_dphidy; // [N*6*6]
    std::vector<double> vol_w;      // [N*6] weight*detJ
    std::vector<double> area;       // [N]
    // Limiter topology (slope + positivity)
    std::vector<int>    nbr;        // [N*3] neighbor element indices (-1 boundary)
    std::vector<double> cx;         // [N] centroid x
    std::vector<double> cy;         // [N] centroid y
    std::vector<double> Hx;         // [N] basis x scale
    std::vector<double> Hy;         // [N] basis y scale
    std::vector<double> phi_vert;   // [N*3*6] basis at the 3 vertices
    std::vector<double> edge_phi_own; // [N*3*EDGENP*6] element's own basis at its 3 edges' quad pts

    // Per-edge canonical data (size E * stride)
    std::vector<int>    e_L;        // [E] left element
    std::vector<int>    e_R;        // [E] right element (-1 boundary)
    std::vector<int>    e_kind;     // [E] DGEdgeKind
    std::vector<double> e_nx;       // [E] unit normal x (L->R)
    std::vector<double> e_ny;       // [E] unit normal y
    std::vector<double> e_w;        // [E*EDGENP] quad weight * edge length
    std::vector<double> e_phiL;     // [E*EDGENP*6] left basis at edge quad pts
    std::vector<double> e_phiR;     // [E*EDGENP*6] right basis at SAME physical pts
    // Hybrid coupling: imported free-surface (eta,u,v) per COUPLED boundary edge.
    std::vector<double> e_ceta;     // [E] coupled eta (free surface from MC neighbor)
    std::vector<double> e_cu;       // [E] coupled u
    std::vector<double> e_cv;       // [E] coupled v
};

// ---- Device data (raw pointers; FP64) ----
struct DGDev {
    int N = 0, E = 0;
    // per-element
    double *q = nullptr, *q_n = nullptr, *q_t1 = nullptr, *q_t2 = nullptr; // state + RK buffers
    double *q_frozen = nullptr;     // double-buffer for the slope limiter neighbor-gather
    double *rhs = nullptr;          // residual accumulator [N*18]
    double *Dcoeffs = nullptr, *Dcell = nullptr, *Minv = nullptr;
    double *vol_phi = nullptr, *vol_dphidx = nullptr, *vol_dphidy = nullptr, *vol_w = nullptr, *area = nullptr;
    int *nbr = nullptr;
    double *cx = nullptr, *cy = nullptr, *Hx = nullptr, *Hy = nullptr, *phi_vert = nullptr, *edge_phi_own = nullptr;
    // per-edge
    int *e_L = nullptr, *e_R = nullptr, *e_kind = nullptr;
    double *e_nx = nullptr, *e_ny = nullptr, *e_w = nullptr, *e_phiL = nullptr, *e_phiR = nullptr;
    double *e_ceta = nullptr, *e_cu = nullptr, *e_cv = nullptr; // coupled seam surface
};

#endif // DG_DEVICE_DATA_CUH
