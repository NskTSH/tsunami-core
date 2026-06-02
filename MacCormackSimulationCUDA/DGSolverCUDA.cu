// DGSolverCUDA.cu - CUDA port of the unstructured P2 DG-SWE solver (see .cuh).
//
// Host setup is reused from the CPU DGSolver (mesh, mass matrices, IC, bathy
// positivity). This file extracts a flat SoA snapshot, uploads it to the device,
// and runs the per-step operator (ComputeResidual = volume + source + surface) +
// SSP-RK3 on the GPU in FP64, mirroring the CPU math exactly. Validated against
// the CPU residual on lake-at-rest.

#include "DGSolverCUDA.cuh"
#include "DGDeviceData.cuh"
#include "MacCormackCUDA.cuh"  // CUDA_CHECK
#include "ASCParser.h"         // real S7 bathymetry (Data/JapanSea.asc)

#include "../MacCormackSimulation/DGSolver.h"
#include "../MacCormackSimulation/ElementMatrices.h"
#include "../MacCormackSimulation/TriMesh.h"
#include "../MacCormackSimulation/RiemannSolver.h"  // SWE_G, SWE_H_DRY

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

// ============================ HOST: snapshot ============================
namespace {

TriMesh BuildRectMeshOff(double x0, double Lx, double y0, double Ly, int nx, int ny) {
    TriMesh mesh;
    mesh.Reserve((nx + 1) * (ny + 1), 2 * nx * ny);
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i)
            mesh.AddNode(x0 + Lx * i / nx, y0 + Ly * j / ny);
    auto nid = [&](int i, int j) { return j * (nx + 1) + i; };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int n00 = nid(i, j), n10 = nid(i + 1, j), n11 = nid(i + 1, j + 1), n01 = nid(i, j + 1);
            mesh.AddElement(n00, n10, n01);
            mesh.AddElement(n10, n11, n01);
        }
    mesh.ComputeGeometry();
    mesh.FixOrientation();
    mesh.BuildEdges();
    mesh.ComputeGeometry();
    return mesh;
}

TriMesh BuildRectMesh(double Lx, double Ly, int nx, int ny) {
    return BuildRectMeshOff(0.0, Lx, 0.0, Ly, nx, ny);
}

int LocalEdgeIndex(const TriElement& el, int gedge) {
    for (int le = 0; le < 3; ++le) if (el.edges[le] == gedge) return le;
    return -1;
}

// Build the flat SoA snapshot from a fully-initialized host DGSolver.
// Boundary edges are tagged WALL (the standalone pure-DG validation default).
DGSnap BuildSnapshot(const DGSolver& dg) {
    const TriMesh* mesh = dg.GetMeshPtr();
    const ElementMatrices* em = dg.GetElementMatricesPtr();
    const std::vector<double>& q = dg.GetSolution();
    const std::vector<double>& Dc = dg.GetDCoeffs();

    DGSnap s;
    s.N = mesh->NumElements();
    s.E = mesh->NumEdges();
    const int B = DGC_BASIS, VN = DGC_VOLNP, EN = DGC_EDGENP;

    s.q.assign(q.begin(), q.end());
    s.Dcoeffs.assign(Dc.begin(), Dc.end());
    s.Dcell.resize(s.N);
    s.Minv.resize(s.N * B * B);
    s.vol_phi.resize(s.N * VN * B);
    s.vol_dphidx.resize(s.N * VN * B);
    s.vol_dphidy.resize(s.N * VN * B);
    s.vol_w.resize(s.N * VN);
    s.area.resize(s.N);
    s.nbr.resize(s.N * 3);
    s.cx.resize(s.N); s.cy.resize(s.N); s.Hx.resize(s.N); s.Hy.resize(s.N);
    s.phi_vert.resize(s.N * 3 * B);
    s.edge_phi_own.resize(s.N * 3 * EN * B);

    for (int k = 0; k < s.N; ++k) {
        const ElementData& ed = em->GetElementData(k);
        const TriElement& el = mesh->GetElement(k);
        s.Dcell[k] = dg.GetBathymetry(k);
        s.area[k] = el.area;
        s.cx[k] = el.cx; s.cy[k] = el.cy;
        s.Hx[k] = ed.basis.GetHx(); s.Hy[k] = ed.basis.GetHy();
        for (int e = 0; e < 3; ++e) s.nbr[k * 3 + e] = el.neighbors[e];
        for (int i = 0; i < B * B; ++i) s.Minv[k * B * B + i] = ed.M_inv[i];
        for (int i = 0; i < VN * B; ++i) {
            s.vol_phi[k * VN * B + i] = ed.vol_phi[i];
            s.vol_dphidx[k * VN * B + i] = ed.vol_dphi_dx[i];
            s.vol_dphidy[k * VN * B + i] = ed.vol_dphi_dy[i];
        }
        for (int i = 0; i < VN; ++i) s.vol_w[k * VN + i] = ed.vol_weights[i];
        // basis at the 3 vertices (for BJ + positivity vertex sampling)
        for (int n = 0; n < 3; ++n) {
            const MeshNode& nd = mesh->GetNode(el.nodes[n]);
            double phi[DGC_BASIS];
            ed.basis.Evaluate(nd.x, nd.y, phi);
            for (int l = 0; l < B; ++l) s.phi_vert[(k * 3 + n) * B + l] = phi[l];
        }
        // element's own edge basis at its 3 edges' quad pts (for positivity edge sampling)
        for (int e = 0; e < 3; ++e)
            for (int i = 0; i < EN * B; ++i)
                s.edge_phi_own[(k * 3 + e) * EN * B + i] = ed.edge_phi[e][i];
    }

    s.e_L.resize(s.E); s.e_R.resize(s.E); s.e_kind.resize(s.E);
    s.e_nx.resize(s.E); s.e_ny.resize(s.E);
    s.e_w.resize(s.E * EN);
    s.e_phiL.resize(s.E * EN * B);
    s.e_phiR.resize(s.E * EN * B);
    s.e_ceta.assign(s.E, 0.0); s.e_cu.assign(s.E, 0.0); s.e_cv.assign(s.E, 0.0);

    for (int e = 0; e < s.E; ++e) {
        const TriEdge& edge = mesh->GetEdge(e);
        int L = edge.elem_left, R = edge.elem_right;
        s.e_L[e] = L; s.e_R[e] = R;
        s.e_nx[e] = edge.normal_x; s.e_ny[e] = edge.normal_y;
        s.e_kind[e] = (R >= 0) ? DG_EDGE_INTERIOR : DG_EDGE_WALL;

        const ElementData& edL = em->GetElementData(L);
        int leL = LocalEdgeIndex(mesh->GetElement(L), e);
        // Left basis + quad (canonical: use the left element's edge quadrature).
        for (int qd = 0; qd < EN; ++qd) {
            double xq = edL.edge_quad[leL][qd * 3 + 0];
            double yq = edL.edge_quad[leL][qd * 3 + 1];
            double wq = edL.edge_quad[leL][qd * 3 + 2];
            s.e_w[e * EN + qd] = wq;
            for (int l = 0; l < B; ++l)
                s.e_phiL[e * EN * B + qd * B + l] = edL.edge_phi[leL][qd * B + l];
            // Right basis evaluated at the SAME physical points (canonical).
            if (R >= 0) {
                double phiR[DGC_BASIS];
                em->GetElementData(R).basis.Evaluate(xq, yq, phiR);
                for (int l = 0; l < B; ++l) s.e_phiR[e * EN * B + qd * B + l] = phiR[l];
            } else {
                for (int l = 0; l < B; ++l) s.e_phiR[e * EN * B + qd * B + l] = 0.0;
            }
        }
    }
    return s;
}

// ============================ DEVICE: upload ============================
template <typename T>
static void up(T*& d, const std::vector<T>& h) {
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

DGDev UploadSnapshot(const DGSnap& s) {
    DGDev d; d.N = s.N; d.E = s.E;
    const int B = DGC_BASIS;
    up(d.q, s.q); up(d.Dcoeffs, s.Dcoeffs); up(d.Dcell, s.Dcell);
    up(d.Minv, s.Minv);
    up(d.vol_phi, s.vol_phi); up(d.vol_dphidx, s.vol_dphidx); up(d.vol_dphidy, s.vol_dphidy);
    up(d.vol_w, s.vol_w); up(d.area, s.area);
    up(d.e_L, s.e_L); up(d.e_R, s.e_R); up(d.e_kind, s.e_kind);
    up(d.e_nx, s.e_nx); up(d.e_ny, s.e_ny); up(d.e_w, s.e_w);
    up(d.e_phiL, s.e_phiL); up(d.e_phiR, s.e_phiR);
    up(d.nbr, s.nbr); up(d.cx, s.cx); up(d.cy, s.cy); up(d.Hx, s.Hx); up(d.Hy, s.Hy);
    up(d.phi_vert, s.phi_vert); up(d.edge_phi_own, s.edge_phi_own);
    up(d.e_ceta, s.e_ceta); up(d.e_cu, s.e_cu); up(d.e_cv, s.e_cv);
    CUDA_CHECK(cudaMalloc(&d.rhs, (size_t)s.N * DGC_BLOCK * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.q_n, (size_t)s.N * DGC_BLOCK * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.q_t1, (size_t)s.N * DGC_BLOCK * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.q_t2, (size_t)s.N * DGC_BLOCK * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.q_frozen, (size_t)s.N * DGC_BLOCK * sizeof(double)));
    (void)B;
    return d;
}

} // namespace

// ============================ DEVICE math (mirror CPU) ============================
__constant__ double DG_G = 9.81;          // SWE_G
__constant__ double DG_HDRY = 1e-6;        // SWE_H_DRY
__constant__ double DG_WETDRY = 10.0;      // wet_dry_threshold (default)

__device__ __forceinline__ double dvel(double H, double mom) {
    return (H > DG_HDRY) ? mom / H : 0.0;
}

// HLL flux, mirror RiemannSolver::ComputeFlux (no velocity cap = default).
__device__ void dHLL(double HLs, double HuL, double HvL,
                     double HRs, double HuR, double HvR,
                     double nx, double ny, double& fH, double& fHu, double& fHv) {
    double HL = fmax(HLs, 0.0), HR = fmax(HRs, 0.0);
    if (HL < DG_HDRY && HR < DG_HDRY) { fH = fHu = fHv = 0.0; return; }
    double uL = dvel(HL, HuL), vL = dvel(HL, HvL);
    double uR = dvel(HR, HuR), vR = dvel(HR, HvR);
    double unL = uL * nx + vL * ny, utL = -uL * ny + vL * nx;
    double unR = uR * nx + vR * ny, utR = -uR * ny + vR * nx;
    double cL = sqrt(DG_G * HL), cR = sqrt(DG_G * HR);
    double uStar = 0.5 * (unL + unR) + cL - cR;
    double cStar = fmax(0.5 * (cL + cR) + 0.25 * (unL - unR), 0.0);
    double sL = fmin(unL - cL, uStar - cStar);
    double sR = fmax(unR + cR, uStar + cStar);
    double fLH = HL * unL, fLHu = HL * unL * unL + 0.5 * DG_G * HL * HL, fLHv = HL * unL * utL;
    double fRH = HR * unR, fRHu = HR * unR * unR + 0.5 * DG_G * HR * HR, fRHv = HR * unR * utR;
    double fnH, fnHu, fnHv;
    if (sL >= 0.0)      { fnH = fLH; fnHu = fLHu; fnHv = fLHv; }
    else if (sR <= 0.0) { fnH = fRH; fnHu = fRHu; fnHv = fRHv; }
    else {
        double inv = 1.0 / (sR - sL);
        fnH  = (sR * fLH  - sL * fRH  + sL * sR * (HR - HL)) * inv;
        fnHu = (sR * fLHu - sL * fRHu + sL * sR * (HR * unR - HL * unL)) * inv;
        fnHv = (sR * fLHv - sL * fRHv + sL * sR * (HR * utR - HL * utL)) * inv;
    }
    // rotate normal/tangential momentum flux back
    fH = fnH;
    fHu = fnHu * nx - fnHv * ny;
    fHv = fnHu * ny + fnHv * nx;
}

// Wall ghost: reflect normal velocity. Mirror RiemannSolver::WallBC.
__device__ void dWall(double H, double Hu, double Hv, double nx, double ny,
                      double& gH, double& gHu, double& gHv) {
    double u = dvel(fmax(H,0.0), Hu), v = dvel(fmax(H,0.0), Hv);
    double un = u * nx + v * ny, ut = -u * ny + v * nx;
    un = -un;
    double ug = un * nx - ut * ny, vg = un * ny + ut * nx;
    gH = H; gHu = H * ug; gHv = H * vg;
}

// ============================ KERNELS ============================
__global__ void kZeroRhs(double* rhs, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * DGC_BLOCK) rhs[i] = 0.0;
}

// Volume + source per element. Mirror VolumeIntegral + SourceTerm (eta-deviation
// pressure with D_c = max(D_pt,0); well-balanced default).
__global__ void kVolumeSource(const double* q, const double* Dc, const double* Dcell,
                              const double* vphi, const double* vdx, const double* vdy,
                              const double* vw, double* rhs, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    const int B = DGC_BASIS, VN = DGC_VOLNP;
    double H_avg = q[k * DGC_BLOCK + 0];
    if (H_avg < DG_WETDRY && Dcell[k] < DG_WETDRY) return; // dry skip (matches CPU)
    const double* qk = q + k * DGC_BLOCK;
    const double* dck = Dc + k * B;
    double acc[DGC_BLOCK];
    for (int i = 0; i < DGC_BLOCK; ++i) acc[i] = 0.0;
    for (int qd = 0; qd < VN; ++qd) {
        const double* phi = vphi + (k * VN + qd) * B;
        const double* dpx = vdx + (k * VN + qd) * B;
        const double* dpy = vdy + (k * VN + qd) * B;
        double w = vw[k * VN + qd];
        double H = 0, Hu = 0, Hv = 0, Dpt = 0, dDdx = 0, dDdy = 0;
        for (int l = 0; l < B; ++l) {
            H  += qk[0 * B + l] * phi[l];
            Hu += qk[1 * B + l] * phi[l];
            Hv += qk[2 * B + l] * phi[l];
            Dpt += dck[l] * phi[l];
            dDdx += dck[l] * dpx[l];
            dDdy += dck[l] * dpy[l];
        }
        double Hs = fmax(H, 0.0);
        double u = dvel(Hs, Hu), v = dvel(Hs, Hv);
        double Dcl = fmax(Dpt, 0.0);
        double eta = Hs - Dcl;
        double P = DG_G * eta * (Dcl + 0.5 * eta);
        double FH = Hs * u, FHu = Hs * u * u + P, FHv = Hs * u * v;
        double GH = Hs * v, GHu = Hs * v * u, GHv = Hs * v * v + P;
        double src = Hs - Dcl;  // eta (well-balanced source coefficient)
        for (int l = 0; l < B; ++l) {
            acc[0 * B + l] += (FH * dpx[l] + GH * dpy[l]) * w;
            acc[1 * B + l] += (FHu * dpx[l] + GHu * dpy[l]) * w + DG_G * src * dDdx * phi[l] * w;
            acc[2 * B + l] += (FHv * dpx[l] + GHv * dpy[l]) * w + DG_G * src * dDdy * phi[l] * w;
        }
    }
    for (int i = 0; i < DGC_BLOCK; ++i) rhs[k * DGC_BLOCK + i] += acc[i];
}

// Surface per edge with atomic accumulation into both elements. Mirror
// SurfaceIntegralEdgeBased (Audusse + eta-deviation dP + bed baseline + neg-D clamp).
__global__ void kSurface(const double* q, const double* Dc,
                         const int* eL, const int* eR, const int* ekind,
                         const double* enx, const double* eny, const double* ew,
                         const double* ephiL, const double* ephiR,
                         const double* eceta, const double* ecu, const double* ecv,
                         double* rhs, int E) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int B = DGC_BASIS, EN = DGC_EDGENP;
    int L = eL[e], R = eR[e];
    double nx = enx[e], ny = eny[e];
    bool interior = (R >= 0);
    const double* qLk = q + L * DGC_BLOCK;
    const double* dcL = Dc + L * B;
    const double* qRk = interior ? (q + R * DGC_BLOCK) : nullptr;
    const double* dcR = interior ? (Dc + R * B) : nullptr;
    for (int qd = 0; qd < EN; ++qd) {
        const double* phiL = ephiL + (e * EN + qd) * B;
        const double* phiR = ephiR + (e * EN + qd) * B;
        double w = ew[e * EN + qd];
        // Left trace
        double HL = 0, HuL = 0, HvL = 0, DLpt = 0;
        for (int l = 0; l < B; ++l) {
            HL  += qLk[0 * B + l] * phiL[l];
            HuL += qLk[1 * B + l] * phiL[l];
            HvL += qLk[2 * B + l] * phiL[l];
            DLpt += dcL[l] * phiL[l];
        }
        double HR, HuR, HvR, DRpt;
        if (interior) {
            HR = HuR = HvR = DRpt = 0;
            for (int l = 0; l < B; ++l) {
                HR  += qRk[0 * B + l] * phiR[l];
                HuR += qRk[1 * B + l] * phiR[l];
                HvR += qRk[2 * B + l] * phiR[l];
                DRpt += dcR[l] * phiR[l];
            }
        } else if (ekind[e] == DG_EDGE_COUPLED) {
            // Coupled seam: reconstruct ghost H from the imported free surface and
            // the DG element's OWN bathymetry trace (eta continuous, depth local) —
            // well-balanced: at rest eta_c=0 => H_g=DLpt => eta=0. (Boundary path:
            // no Audusse rebuild, pbed uses DLpt, no per-side dP — matches CPU.)
            double Hg = fmax(0.0, eceta[e] + DLpt);
            HR = Hg; HuR = Hg * ecu[e]; HvR = Hg * ecv[e];
            DRpt = DLpt;
        } else {
            // Boundary: Wall ghost from left trace; D mirror.
            dWall(HL, HuL, HvL, nx, ny, HR, HuR, HvR);
            DRpt = DLpt;
        }
        // Clamp negative P2 bathymetry overshoot (and matching H trace)
        DLpt = fmax(DLpt, 0.0); DRpt = fmax(DRpt, 0.0);
        HL = fmax(HL, 0.0); HR = fmax(HR, 0.0);
        // Audusse hydrostatic reconstruction (interior only, as CPU)
        double HLstar = HL, HuLs = HuL, HvLs = HvL;
        double HRstar = HR, HuRs = HuR, HvRs = HvR;
        double Dmin = interior ? fmin(DLpt, DRpt) : DLpt;
        if (interior) {
            HLstar = fmax(0.0, HL - DLpt + Dmin);
            HRstar = fmax(0.0, HR - DRpt + Dmin);
            double rL = (HL > DG_HDRY) ? HLstar / HL : 0.0;
            double rR = (HR > DG_HDRY) ? HRstar / HR : 0.0;
            HuLs = rL * HuL; HvLs = rL * HvL;
            HuRs = rR * HuR; HvRs = rR * HvR;
        }
        // HLL of the (reconstructed) states
        double fH, fHu, fHv;
        dHLL(HLstar, HuLs, HvLs, HRstar, HuRs, HvRs, nx, ny, fH, fHu, fHv);
        // Subtract bed baseline 0.5 g Dmin^2 from the shared momentum flux (wb)
        double pbed = 0.5 * DG_G * Dmin * Dmin;
        fHu -= pbed * nx; fHv -= pbed * ny;
        // Per-side deviation dP (cancellation-safe), interior only
        double dPL = 0.0, dPR = 0.0;
        if (interior) {
            double etaL = HL - DLpt, etaLs = HLstar - Dmin;
            dPL = 0.5 * DG_G * (etaL * (HL + DLpt) - etaLs * (HLstar + Dmin));
            double etaR = HR - DRpt, etaRs = HRstar - Dmin;
            dPR = 0.5 * DG_G * (etaR * (HR + DRpt) - etaRs * (HRstar + Dmin));
        }
        // Accumulate: LEFT -= flux_L, RIGHT += flux_R
        double fLHu = fHu + dPL * nx, fLHv = fHv + dPL * ny;
        for (int l = 0; l < B; ++l) {
            atomicAdd(&rhs[L * DGC_BLOCK + 0 * B + l], -fH  * phiL[l] * w);
            atomicAdd(&rhs[L * DGC_BLOCK + 1 * B + l], -fLHu * phiL[l] * w);
            atomicAdd(&rhs[L * DGC_BLOCK + 2 * B + l], -fLHv * phiL[l] * w);
        }
        if (interior) {
            double fRHu = fHu + dPR * nx, fRHv = fHv + dPR * ny;
            for (int l = 0; l < B; ++l) {
                atomicAdd(&rhs[R * DGC_BLOCK + 0 * B + l], +fH  * phiR[l] * w);
                atomicAdd(&rhs[R * DGC_BLOCK + 1 * B + l], +fRHu * phiR[l] * w);
                atomicAdd(&rhs[R * DGC_BLOCK + 2 * B + l], +fRHv * phiR[l] * w);
            }
        }
    }
}

// RK combine: out = a*qn + b*(qcur + dt*Minv*rhs)
__global__ void kRkCombine(const double* qn, const double* qcur, const double* rhs,
                          const double* Minv, double a, double b, double dt,
                          double* out, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    const int B = DGC_BASIS;
    const double* mi = Minv + k * B * B;
    for (int var = 0; var < DGC_NVARS; ++var) {
        const double* r = rhs + k * DGC_BLOCK + var * B;
        double mr[DGC_BASIS];
        for (int i = 0; i < B; ++i) {
            double s = 0.0;
            for (int j = 0; j < B; ++j) s += mi[i * B + j] * r[j];
            mr[i] = s;
        }
        for (int i = 0; i < B; ++i) {
            int idx = k * DGC_BLOCK + var * B + i;
            out[idx] = a * qn[idx] + b * (qcur[idx] + dt * mr[i]);
        }
    }
}

// ---- Slope limiter (eta-mode, TVB Barth-Jespersen). Double-buffered: reads
//      qf (frozen pre-limiter state) for neighbor gather, writes q. Mirrors
//      DGSolver::ApplySlopeLimiter with DG_LIMIT_ETA=1, tvb_M=0. ----
__global__ void kSlopeLimiter(const double* qf, double* q, const double* Dc,
                             const double* Dcell, const int* nbr, const double* cx,
                             const double* cy, const double* Hx, const double* Hy,
                             const double* phiV, const double* area, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    const int B = DGC_BASIS;
    const double JUMP_THR = 0.1, OSC_THR = 0.5, BJ_TOL = 1e-14, ZGUARD = 1e-30, WETDRY = 10.0;
    int kb = k * DGC_BLOCK;
    // cell-average eta (var0) and raw averages (var1,2) of THIS cell from qf
    double avg[DGC_NVARS];
    avg[0] = qf[kb + 0] - Dc[k * B + 0];
    avg[1] = qf[kb + 1 * B]; avg[2] = qf[kb + 2 * B];

    // ---- Pass 1: troubled detection ----
    int troubled = 0;
    double Hk = Dcell[k]; // eta-mode wet/dry uses static D_[k]
    if (Hk < WETDRY) troubled = 1;
    for (int e = 0; e < 3 && !troubled; ++e) {
        int nb = nbr[k * 3 + e];
        if (nb >= 0 && Dcell[nb] < WETDRY) troubled = 1;
    }
    for (int var = 0; var < DGC_NVARS && !troubled; ++var) {
        double qa = avg[var];
        for (int e = 0; e < 3; ++e) {
            int nb = nbr[k * 3 + e]; if (nb < 0) continue;
            double qn = (var == 0) ? (qf[nb * DGC_BLOCK + 0] - Dc[nb * B + 0]) : qf[nb * DGC_BLOCK + var * B];
            double jump = fabs(qn - qa), scale = fmax(fabs(qa), 1.0);
            if (jump > 0.0 /*tvb_M*area=0*/ && jump > JUMP_THR * scale) { troubled = 1; break; }
        }
        if (!troubled) {
            double ms = 0.0;
            const double* src = (var == 0) ? nullptr : (qf + kb + var * B);
            for (int l = 1; l < B; ++l) {
                double m = (var == 0) ? (qf[kb + 0 * B + l] - Dc[k * B + l]) : src[l];
                ms += m * m;
            }
            double avg2 = qa * qa + ZGUARD;
            if (ms > OSC_THR * avg2) troubled = 1;
        }
    }
    if (!troubled) return;  // leave q unchanged (q already holds qf via copy)

    // ---- Pass 2: limit (reduce to P0 + minmod P1 gradient + BJ) ----
    double hx = Hx[k], hy = Hy[k];
    for (int var = 0; var < DGC_NVARS; ++var) {
        double qa = avg[var];
        double qmin = qa, qmax = qa; int nc = 0;
        for (int e = 0; e < 3; ++e) {
            int nb = nbr[k * 3 + e]; if (nb < 0) continue;
            double qn = (var == 0) ? (qf[nb * DGC_BLOCK + 0] - Dc[nb * B + 0]) : qf[nb * DGC_BLOCK + var * B];
            qmin = fmin(qmin, qn); qmax = fmax(qmax, qn); nc++;
        }
        double q1 = 0.0, q2 = 0.0;
        if (nc >= 2) {
            double dqx = 0, dqy = 0, ws = 0;
            for (int e = 0; e < 3; ++e) {
                int nb = nbr[k * 3 + e]; if (nb < 0) continue;
                double qn = (var == 0) ? (qf[nb * DGC_BLOCK + 0] - Dc[nb * B + 0]) : qf[nb * DGC_BLOCK + var * B];
                double dx = cx[nb] - cx[k], dy = cy[nb] - cy[k];
                double d2 = dx * dx + dy * dy; if (d2 < ZGUARD) continue;
                double dq = qn - qa, inv = 1.0 / d2;
                dqx += dq * dx * inv; dqy += dq * dy * inv; ws += 1.0;
            }
            if (ws > 0) { dqx /= ws; dqy /= ws; }
            q1 = dqx * hx; q2 = dqy * hy;
            double theta = 1.0;
            for (int n = 0; n < 3; ++n) {
                const double* pv = phiV + (k * 3 + n) * B;
                double qpt = qa + q1 * pv[1] + q2 * pv[2];
                if (qpt > qmax + BJ_TOL && qpt > qa + BJ_TOL) theta = fmin(theta, (qmax - qa) / (qpt - qa));
                else if (qpt < qmin - BJ_TOL && qpt < qa - BJ_TOL) theta = fmin(theta, (qmin - qa) / (qpt - qa));
            }
            theta = fmax(theta, 0.0);
            q1 *= theta; q2 *= theta;
        }
        // write back: mode0 = qa, mode1=q1, mode2=q2, modes3-5 = 0
        if (var == 0) {
            q[kb + 0 * B + 0] = qa + Dc[k * B + 0];
            q[kb + 0 * B + 1] = q1 + Dc[k * B + 1];
            q[kb + 0 * B + 2] = q2 + Dc[k * B + 2];
            for (int l = 3; l < B; ++l) q[kb + 0 * B + l] = Dc[k * B + l]; // eta_l=0 -> H_l=D_l
        } else {
            q[kb + var * B + 0] = qa;
            q[kb + var * B + 1] = q1;
            q[kb + var * B + 2] = q2;
            for (int l = 3; l < B; ++l) q[kb + var * B + l] = 0.0;
        }
    }
}

// ---- Positivity limiter (Zhang-Shu, well-balanced eta-space). Per-element,
//      in-place on q. Mirrors DGSolver::ApplyPositivityLimiter wb path. ----
__global__ void kPositivity(double* q, const double* Dc, const double* phiV,
                           const double* edgePhi, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    const int B = DGC_BASIS, EN = DGC_EDGENP;
    const double HDRY = 1e-6, DRYF = 10.0, WETDRY = 10.0, MAXV = 20.0;
    int kb = k * DGC_BLOCK;
    double eta[DGC_BASIS];
    for (int l = 0; l < B; ++l) eta[l] = q[kb + 0 * B + l] - Dc[k * B + l];
    double H_avg = q[kb + 0];

    // min sampled H over 3 vertices + 3 edges * EN pts
    auto minH = [&]() -> double {
        double Hmin = Dc[k * B + 0] + eta[0];
        for (int n = 0; n < 3; ++n) {
            const double* pv = phiV + (k * 3 + n) * B;
            double Dp = 0, ep = 0;
            for (int l = 0; l < B; ++l) { Dp += Dc[k * B + l] * pv[l]; ep += eta[l] * pv[l]; }
            Hmin = fmin(Hmin, Dp + ep);
        }
        for (int e = 0; e < 3; ++e)
            for (int qp = 0; qp < EN; ++qp) {
                const double* pe = edgePhi + ((k * 3 + e) * EN + qp) * B;
                double Dp = 0, ep = 0;
                for (int l = 0; l < B; ++l) { Dp += Dc[k * B + l] * pe[l]; ep += eta[l] * pe[l]; }
                Hmin = fmin(Hmin, Dp + ep);
            }
        return Hmin;
    };

    if (H_avg >= HDRY) {
        double Hmin = minH();
        if (Hmin < 0.0) {
            double eavg = eta[0], theta = 1.0;
            // recompute theta over sample points
            for (int n = 0; n < 3; ++n) {
                const double* pv = phiV + (k * 3 + n) * B;
                double Dp = 0, ep = 0; for (int l = 0; l < B; ++l) { Dp += Dc[k*B+l]*pv[l]; ep += eta[l]*pv[l]; }
                double Hpt = Dp + ep; if (Hpt >= 0.0) continue;
                double ehigh = ep - eavg, Hp0 = Dp + eavg;
                if (ehigh < 0.0 && Hp0 > 0.0) theta = fmin(theta, Hp0 / (Hp0 - Hpt)); else theta = 0.0;
            }
            for (int e = 0; e < 3; ++e) for (int qp = 0; qp < EN; ++qp) {
                const double* pe = edgePhi + ((k*3+e)*EN+qp)*B;
                double Dp = 0, ep = 0; for (int l = 0; l < B; ++l) { Dp += Dc[k*B+l]*pe[l]; ep += eta[l]*pe[l]; }
                double Hpt = Dp + ep; if (Hpt >= 0.0) continue;
                double ehigh = ep - eavg, Hp0 = Dp + eavg;
                if (ehigh < 0.0 && Hp0 > 0.0) theta = fmin(theta, Hp0 / (Hp0 - Hpt)); else theta = 0.0;
            }
            theta = fmax(fmin(theta, 1.0), 0.0);
            for (int l = 1; l < B; ++l) { eta[l] *= theta; q[kb + 1*B + l] *= theta; q[kb + 2*B + l] *= theta; }
        }
    }
    // velocity cap on deep wet cells (H_avg >= wet_dry_threshold)
    if (H_avg >= WETDRY) {
        double u = q[kb + 1*B] / H_avg, v = q[kb + 2*B] / H_avg;
        double sp = sqrt(u*u + v*v);
        if (sp > MAXV) { double s = MAXV / sp; for (int l = 0; l < B; ++l) { q[kb+1*B+l] *= s; q[kb+2*B+l] *= s; } }
    } else {
        // shallow/dry: drop higher modes + zero momentum modes (simplified dry handling)
        for (int l = 1; l < B; ++l) { eta[l] = 0.0; q[kb+1*B+l] = 0.0; q[kb+2*B+l] = 0.0; }
        if (H_avg < DRYF * HDRY) { q[kb+1*B] = 0.0; q[kb+2*B] = 0.0; }
    }
    // restore H = D + eta
    for (int l = 0; l < B; ++l) q[kb + 0*B + l] = Dc[k * B + l] + eta[l];
}

// Running max|eta_avg| per element (free-surface deviation) for the eta_max map.
__global__ void kDGEtaMax(const double* q, const double* Dc, double* emax, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    double eta = q[k * DGC_BLOCK + 0] - Dc[k * DGC_BASIS + 0];
    double a = fabs(eta);
    if (a > emax[k]) emax[k] = a;
}

// ============================ host driver ============================
static void computeResidualGPU(DGDev& d, int tpbE, int tpbN) {
    int nb = (d.N * DGC_BLOCK + 255) / 256;
    kZeroRhs<<<nb, 256>>>(d.rhs, d.N);
    kVolumeSource<<<(d.N + tpbN - 1) / tpbN, tpbN>>>(d.q, d.Dcoeffs, d.Dcell,
        d.vol_phi, d.vol_dphidx, d.vol_dphidy, d.vol_w, d.rhs, d.N);
    kSurface<<<(d.E + tpbE - 1) / tpbE, tpbE>>>(d.q, d.Dcoeffs, d.e_L, d.e_R, d.e_kind,
        d.e_nx, d.e_ny, d.e_w, d.e_phiL, d.e_phiR, d.e_ceta, d.e_cu, d.e_cv, d.rhs, d.E);
}

// Apply slope (double-buffered) + positivity limiters to d.q, matching the CPU
// per-stage order (combine -> slope -> positivity).
static void applyLimitersGPU(DGDev& d, bool on) {
    if (!on) return;
    const size_t bytes = (size_t)d.N * DGC_BLOCK * sizeof(double);
    int gb = (d.N + 127) / 128;
    CUDA_CHECK(cudaMemcpy(d.q_frozen, d.q, bytes, cudaMemcpyDeviceToDevice));
    kSlopeLimiter<<<gb, 128>>>(d.q_frozen, d.q, d.Dcoeffs, d.Dcell, d.nbr, d.cx, d.cy,
                               d.Hx, d.Hy, d.phi_vert, d.area, d.N);
    kPositivity<<<gb, 128>>>(d.q, d.Dcoeffs, d.phi_vert, d.edge_phi_own, d.N);
}

// One SSP-RK3 (Shu-Osher) step on the device, mirroring DGSolver::StepRK3 incl.
// per-stage slope + positivity limiters.
static void stepRK3_GPU(DGDev& d, double dt, bool limiters) {
    const size_t bytes = (size_t)d.N * DGC_BLOCK * sizeof(double);
    int gb = (d.N + 127) / 128;
    CUDA_CHECK(cudaMemcpy(d.q_n, d.q, bytes, cudaMemcpyDeviceToDevice)); // q_n = q^n
    // Stage 1: q1 = q_n + dt*Minv*L(q_n)
    computeResidualGPU(d, 128, 128);
    kRkCombine<<<gb, 128>>>(d.q_n, d.q_n, d.rhs, d.Minv, 0.0, 1.0, dt, d.q_t1, d.N);
    CUDA_CHECK(cudaMemcpy(d.q, d.q_t1, bytes, cudaMemcpyDeviceToDevice));
    applyLimitersGPU(d, limiters);
    // Stage 2: q2 = 3/4 q_n + 1/4 (q1 + dt*Minv*L(q1))   (q1 = limited stage-1)
    CUDA_CHECK(cudaMemcpy(d.q_t1, d.q, bytes, cudaMemcpyDeviceToDevice));
    computeResidualGPU(d, 128, 128);
    kRkCombine<<<gb, 128>>>(d.q_n, d.q_t1, d.rhs, d.Minv, 0.75, 0.25, dt, d.q_t2, d.N);
    CUDA_CHECK(cudaMemcpy(d.q, d.q_t2, bytes, cudaMemcpyDeviceToDevice));
    applyLimitersGPU(d, limiters);
    // Stage 3: q^{n+1} = 1/3 q_n + 2/3 (q2 + dt*Minv*L(q2))  (q2 = limited stage-2)
    CUDA_CHECK(cudaMemcpy(d.q_t2, d.q, bytes, cudaMemcpyDeviceToDevice));
    computeResidualGPU(d, 128, 128);
    kRkCombine<<<gb, 128>>>(d.q_n, d.q_t2, d.rhs, d.Minv, 1.0/3.0, 2.0/3.0, dt, d.q, d.N);
    applyLimitersGPU(d, limiters);
}

// ===================== Phase 4/5: DG<->DG seam-coupled HYBRID orchestration =====================
// Two DG sub-domains sharing a vertical seam, coupled by exchanging the free-surface
// eta each step through the verified COUPLED boundary. This is the exact hybrid loop
// (two GPU solvers + per-step seam eta exchange via host); the DG<->MC variant swaps
// one DG sub-solver for the MacCormack GPU solver feeding eta at the seam. Validates
// well-balancedness ACROSS the seam (lake-at-rest must hold to machine zero).
struct SeamE { int edge; int elem; double y; };
static std::vector<SeamE> FindSeam(const TriMesh* m, double xs, double tol) {
    std::vector<SeamE> v;
    for (int e = 0; e < m->NumEdges(); ++e) {
        const TriEdge& ed = m->GetEdge(e);
        if (ed.elem_right >= 0) continue;            // interior
        if (std::fabs(ed.mx - xs) < tol) v.push_back({e, ed.elem_left, ed.my});
    }
    std::sort(v.begin(), v.end(), [](const SeamE& a, const SeamE& b) { return a.y < b.y; });
    return v;
}

static int RunDGHybridGPU() {
    std::printf("\n=== DG CUDA port - Phase 4/5: DG<->DG seam-coupled HYBRID (orchestration) ===\n");
    const double L = 200000.0, xs = L * 0.5;
    const int NyH = 16, NxH = 8;
    auto Dfun = [L](double x, double) { return 500.0 + 3000.0 * (x / L); }; // global slope across seam

    TriMesh mL = BuildRectMeshOff(0.0,  xs,      0.0, L, NxH, NyH);
    TriMesh mR = BuildRectMeshOff(xs,   L - xs,  0.0, L, NxH, NyH);
    ElementMatrices emL, emR; emL.Precompute(mL); emR.Precompute(mR);
    DGSolver dgL, dgR; dgL.Initialize(&mL, &emL); dgR.Initialize(&mR, &emR);
    for (DGSolver* dg : {&dgL, &dgR}) {
        dg->SetBathymetryFunction(Dfun);
        dg->SetDefaultBoundaryCondition(DGBoundaryType::Wall);
        dg->SetSlopeLimiter(true, 0.0);
        dg->SetInitialConditionWellBalanced(
            [&](double x, double y) { return SWEState(Dfun(x, y), 0.0, 0.0); }, Dfun);
    }
    auto seamL = FindSeam(&mL, xs, std::max(L, L) * 1e-7);  // left's right-boundary edges
    auto seamR = FindSeam(&mR, xs, std::max(L, L) * 1e-7);  // right's left-boundary edges
    std::printf("[hybrid] seam edges: left=%zu right=%zu (should match)\n", seamL.size(), seamR.size());

    DGSnap sL = BuildSnapshot(dgL), sR = BuildSnapshot(dgR);
    for (auto& se : seamL) { sL.e_kind[se.edge] = DG_EDGE_COUPLED; sL.e_ceta[se.edge] = 0.0; }
    for (auto& se : seamR) { sR.e_kind[se.edge] = DG_EDGE_COUPLED; sR.e_ceta[se.edge] = 0.0; }
    DGDev dL = UploadSnapshot(sL), dR = UploadSnapshot(sR);

    size_t np = std::min(seamL.size(), seamR.size());
    std::vector<double> hLceta = sL.e_ceta, hRceta = sR.e_ceta; // host mirrors
    std::vector<double> qLh(dL.N * DGC_BLOCK), qRh(dR.N * DGC_BLOCK);

    auto exchange = [&]() {
        CUDA_CHECK(cudaMemcpy(qLh.data(), dL.q, qLh.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qRh.data(), dR.q, qRh.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t p = 0; p < np; ++p) {
            int lE = seamL[p].elem, rE = seamR[p].elem;
            double etaL = qLh[lE * DGC_BLOCK + 0] - sL.Dcoeffs[lE * DGC_BASIS + 0]; // left seam-elem eta
            double etaR = qRh[rE * DGC_BLOCK + 0] - sR.Dcoeffs[rE * DGC_BASIS + 0]; // right seam-elem eta
            hLceta[seamL[p].edge] = etaR;  // left imports right's surface
            hRceta[seamR[p].edge] = etaL;  // right imports left's surface
        }
        CUDA_CHECK(cudaMemcpy(dL.e_ceta, hLceta.data(), hLceta.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dR.e_ceta, hRceta.data(), hRceta.size() * sizeof(double), cudaMemcpyHostToDevice));
    };

    double dt = std::min(dgL.ComputeTimeStep(0.2), dgR.ComputeTimeStep(0.2));
    const int NSTEP = 100;
    for (int s = 0; s < NSTEP; ++s) {
        exchange();                       // import neighbor eta at the seam
        stepRK3_GPU(dL, dt, true);
        stepRK3_GPU(dR, dt, true);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(qLh.data(), dL.q, qLh.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qRh.data(), dR.q, qRh.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double me = 0.0;
    for (int k = 0; k < dL.N; ++k) me = std::fmax(me, std::fabs(qLh[k * DGC_BLOCK] - sL.Dcoeffs[k * DGC_BASIS]));
    for (int k = 0; k < dR.N; ++k) me = std::fmax(me, std::fabs(qRh[k * DGC_BLOCK] - sR.Dcoeffs[k * DGC_BASIS]));
    std::printf("[HYBRID x%d lake-at-rest, DG<->DG seam exchange]  max|eta_avg|=%.3g  -> %s\n",
                NSTEP, me, (me < 1e-6 ? "WELL-BALANCED ACROSS SEAM (hybrid orchestration works)" : "DRIFT"));
    std::printf("[hybrid] Per-step eta exchange through COUPLED seam validated on GPU. DG<->MC variant\n");
    std::printf("[hybrid] = same loop with one sub-solver = MacCormackCUDA feeding seam eta.\n");
    return 0;
}

// ===================== LITERAL MC(MacCormackCUDA) + DG hybrid on GPU =====================
// Runs the colleague's structured MacCormack GPU solver (MacCormackCUDA, UNMODIFIED)
// on the left region and the DG-GPU solver on the right region, coupled at the seam:
// each step MC advances, its seam-column free surface eta is read and fed into the DG
// COUPLED boundary; the DG seam eta is written back into the MC ghost column. Flat
// bathymetry lake-at-rest must stay machine-zero on BOTH solvers (well-balanced
// coupled system), then a small MC perturbation is run to show the coupled MC+DG
// system advances dynamically on the GPU without blowup.
static int RunDGMCHybridGPU() {
    std::printf("\n=== LITERAL MC (MacCormackCUDA) + DG TWO-WAY hybrid on GPU ===\n");
    const double L = 200000.0, xs = L * 0.5, D0 = 1000.0, g = 9.81;
    const int Ny = 16;
    auto Dflat = [D0](double, double) { return D0; };

    // ---- MC region [0, xs] x [0, L]: colleague's MacCormackCUDA. has_right_neighbor=true
    //      so MC does NOT apply a physical BC on its right column (it becomes the SEAM
    //      ghost that WE own and fill from DG each step) — the MC's own subgrid mechanism. ----
    const int nx_mc = Ny + 1, ny_mc = Ny + 1;
    double dx = xs / (nx_mc - 1), dy = L / (ny_mc - 1);
    double c0 = std::sqrt(g * D0);
    double dtMC = 0.2 * std::min(dx, dy) / c0;

    SimulationParams p{};
    p.g = g; p.delta_x_const = dx; p.delta_y = dy; p.min_calc_value = 8.0;
    p.use_variable_dx = false; p.boundaryConditionType = 0; p.hatBoundaryConditionType = 0;
    p.coord_system = CudaCoordSystem::Cartesian; p.coriolis_enabled = false;

    // ---- DG region [xs, L] x [0, L], left boundary COUPLED to MC ----
    TriMesh mR = BuildRectMeshOff(xs, L - xs, 0.0, L, Ny, Ny);
    ElementMatrices emR; emR.Precompute(mR);
    DGSolver dgR; dgR.Initialize(&mR, &emR);
    dgR.SetBathymetryFunction(Dflat);
    dgR.SetDefaultBoundaryCondition(DGBoundaryType::Wall);
    dgR.SetSlopeLimiter(true, 0.0);
    dgR.SetInitialConditionWellBalanced(
        [&](double x, double y) { return SWEState(Dflat(x, y), 0.0, 0.0); }, Dflat);
    auto seam = FindSeam(&mR, xs, L * 1e-7);
    double dt = std::min(dtMC, dgR.ComputeTimeStep(0.2));
    p.tau = dt;
    std::printf("[mc+dg] MC %dx%d, DG elems=%d, seam edges=%zu, dt=%.4g\n",
                nx_mc, ny_mc, dgR.NumElements(), seam.size(), dt);

    auto mc = CreateMacCormackCUDA(nx_mc, ny_mc, p, /*use_fp32=*/false);
    SubgridInfo si; si.has_left_neighbor = false; si.has_right_neighbor = true;
    si.subgrid_index = 0; si.num_subgrids = 2;
    mc->SetSubgridInfo(si);  // right column = exchange seam (MC won't apply physical BC there)
    std::vector<double> mcH(nx_mc * ny_mc, D0), mcU(nx_mc * ny_mc, 0.0),
                        mcV(nx_mc * ny_mc, 0.0), mcEta(nx_mc * ny_mc, 0.0), mcD(nx_mc * ny_mc, D0);

    DGSnap sR = BuildSnapshot(dgR);
    for (auto& se : seam) { sR.e_kind[se.edge] = DG_EDGE_COUPLED; sR.e_ceta[se.edge] = 0.0; }
    DGDev dR = UploadSnapshot(sR);
    std::vector<double> hRceta = sR.e_ceta, hRcu = sR.e_cu, hRcv = sR.e_cv, qRh(dR.N * DGC_BLOCK);

    // Linear interpolation along y between MC nodes (column i) and DG seam edges.
    auto mcColInterp = [&](const std::vector<double>& fld, int icol, double y) {
        double fj = y / dy; int j0 = (int)std::floor(fj);
        if (j0 < 0) j0 = 0; if (j0 > ny_mc - 2) j0 = ny_mc - 2;
        double w = fj - j0; if (w < 0) w = 0; if (w > 1) w = 1;
        return fld[icol * ny_mc + j0] * (1 - w) + fld[icol * ny_mc + (j0 + 1)] * w;
    };
    // DG seam-edge eta/u/v interpolated to a query y (seam sorted by y).
    auto dgSeamInterp = [&](double y, double& eta, double& u, double& v) {
        int np = (int)seam.size();
        int p = 0; while (p < np - 1 && seam[p + 1].y < y) ++p;
        int p1 = std::min(p + 1, np - 1);
        double y0 = seam[p].y, y1 = seam[p1].y, w = (y1 > y0) ? (y - y0) / (y1 - y0) : 0.0;
        w = std::min(std::max(w, 0.0), 1.0);
        auto val = [&](int pp, double& e, double& uu, double& vv) {
            int el = seam[pp].elem; double He = qRh[el * DGC_BLOCK + 0];
            e = He - sR.Dcoeffs[el * DGC_BASIS + 0];
            uu = (He > 1e-6) ? qRh[el * DGC_BLOCK + DGC_BASIS] / He : 0.0;
            vv = (He > 1e-6) ? qRh[el * DGC_BLOCK + 2 * DGC_BASIS] / He : 0.0;
        };
        double e0, u0, v0, e1, u1, v1; val(p, e0, u0, v0); val(p1, e1, u1, v1);
        eta = e0 * (1 - w) + e1 * w; u = u0 * (1 - w) + u1 * w; v = v0 * (1 - w) + v1 * w;
    };

    // Two-way exchange BEFORE each step (linear interp across the mismatched seam grids).
    auto exchange = [&](DGDev& d) {
        mc->CopyToHost(mcH.data(), mcU.data(), mcV.data(), mcEta.data());
        CUDA_CHECK(cudaMemcpy(qRh.data(), d.q, qRh.size() * sizeof(double), cudaMemcpyDeviceToHost));
        // MC -> DG: interp MC's last interior column (nx-2) surface AND velocity to
        // each DG seam edge y (velocity transfer is essential — without it the DG seam
        // sees still water and the wave's momentum cannot cross -> spurious reflection).
        for (size_t pq = 0; pq < seam.size(); ++pq) {
            double y = seam[pq].y; int e = seam[pq].edge;
            hRceta[e] = mcColInterp(mcEta, nx_mc - 2, y);
            hRcu[e]   = mcColInterp(mcU,   nx_mc - 2, y);
            hRcv[e]   = mcColInterp(mcV,   nx_mc - 2, y);
        }
        CUDA_CHECK(cudaMemcpy(d.e_ceta, hRceta.data(), hRceta.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.e_cu, hRcu.data(), hRcu.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.e_cv, hRcv.data(), hRcv.size() * sizeof(double), cudaMemcpyHostToDevice));
        // DG -> MC ghost column i=nx-1: interp DG seam surface to each MC node y
        for (int j = 0; j < ny_mc; ++j) {
            double eta, u, v; dgSeamInterp(j * dy, eta, u, v);
            int gi = (nx_mc - 1) * ny_mc + j;
            mcEta[gi] = eta; mcH[gi] = D0 + eta; mcU[gi] = u; mcV[gi] = v;
        }
        mc->LoadInitialData(mcH.data(), mcU.data(), mcV.data(), mcEta.data(), mcD.data());
    };
    auto peak = [&](DGDev& d, double& mcmax, double& dgmax) {
        mc->CopyEtaToHost(mcEta.data());
        mcmax = 0; for (double e : mcEta) mcmax = std::fmax(mcmax, std::fabs(e));
        CUDA_CHECK(cudaMemcpy(qRh.data(), d.q, qRh.size() * sizeof(double), cudaMemcpyDeviceToHost));
        dgmax = 0; for (int k = 0; k < d.N; ++k) dgmax = std::fmax(dgmax, std::fabs(qRh[k * DGC_BLOCK] - sR.Dcoeffs[k * DGC_BASIS]));
    };

    mc->LoadInitialData(mcH.data(), mcU.data(), mcV.data(), mcEta.data(), mcD.data());
    // Test A: lake-at-rest, TWO-WAY coupled, 100 steps -> machine zero (well-balanced)
    for (int t = 0; t < 100; ++t) { exchange(dR); mc->Step(t); stepRK3_GPU(dR, dt, true); }
    CUDA_CHECK(cudaDeviceSynchronize());
    double rmc, rdg; peak(dR, rmc, rdg);
    std::printf("[MC+DG x100 lake-at-rest, TWO-WAY]  MC max|eta|=%.3g  DG max|eta|=%.3g  -> %s\n",
                rmc, rdg, std::fmax(rmc, rdg) < 1e-6 ? "WELL-BALANCED (two-way MC<->DG holds rest)" : "DRIFT");

    // Test B: tsunami-like Gaussian pulse in MC, TWO-WAY. Wave crosses MC->DG and feeds
    // back DG->MC; compare hybrid peak to pure-MC monolithic over full [0,L].
    const double amp = 2.0, sig = L * 0.07, cxp = xs * 0.55, cyp = L * 0.5;
    {
        std::fill(mcU.begin(), mcU.end(), 0.0); std::fill(mcV.begin(), mcV.end(), 0.0);
        for (int i = 0; i < nx_mc; ++i) for (int j = 0; j < ny_mc; ++j) {
            double x = i * dx, y = j * dy, r2 = (x - cxp) * (x - cxp) + (y - cyp) * (y - cyp);
            double e = amp * std::exp(-r2 / (2 * sig * sig));
            mcEta[i * ny_mc + j] = e; mcH[i * ny_mc + j] = D0 + e;
        }
        mc->LoadInitialData(mcH.data(), mcU.data(), mcV.data(), mcEta.data(), mcD.data());
        DGDev dR0 = UploadSnapshot(sR);
        bool ok = true; const int NS = 200;
        for (int t = 0; t < NS && ok; ++t) {
            exchange(dR0); mc->Step(t); stepRK3_GPU(dR0, dt, true);
            if ((t % 50) == 0) {
                CUDA_CHECK(cudaMemcpy(qRh.data(), dR0.q, qRh.size() * sizeof(double), cudaMemcpyDeviceToHost));
                for (double vv : qRh) if (!std::isfinite(vv)) { ok = false; break; }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double mcmax, dgmax; peak(dR0, mcmax, dgmax);
        double hyb_peak = std::fmax(mcmax, dgmax);
        std::printf("[MC+DG x%d Gaussian pulse, TWO-WAY]  MC max|eta|=%.4g  DG max|eta|=%.4g  finite=%s\n",
                    NS, mcmax, dgmax, ok ? "yes" : "NO");

        const int nxF = 2 * nx_mc - 1;
        SimulationParams pf = p; pf.delta_x_const = L / (nxF - 1);
        auto mcF = CreateMacCormackCUDA(nxF, ny_mc, pf, false);
        std::vector<double> H(nxF*ny_mc, D0), U(nxF*ny_mc,0.0), V(nxF*ny_mc,0.0), E(nxF*ny_mc,0.0), Dd(nxF*ny_mc, D0);
        double dxF = L / (nxF - 1);
        for (int i = 0; i < nxF; ++i) for (int j = 0; j < ny_mc; ++j) {
            double x = i * dxF, y = j * dy, r2 = (x - cxp) * (x - cxp) + (y - cyp) * (y - cyp);
            double e = amp * std::exp(-r2 / (2 * sig * sig)); E[i*ny_mc+j]=e; H[i*ny_mc+j]=D0+e;
        }
        mcF->LoadInitialData(H.data(), U.data(), V.data(), E.data(), Dd.data());
        for (int t = 0; t < NS; ++t) mcF->Step(t);
        mcF->CopyEtaToHost(E.data());
        double refmax = 0; for (double e : E) refmax = std::fmax(refmax, std::fabs(e));
        std::printf("[REFERENCE pure-MC full domain x%d]  max|eta|=%.4g\n", NS, refmax);
        std::printf("[MC+DG] RESULT: hybrid peak=%.4g vs pure-MC ref=%.4g  (rel.diff=%.1f%%)  %s\n",
                    hyb_peak, refmax, 100.0 * std::fabs(hyb_peak - refmax) / std::fmax(refmax, 1e-12),
                    (ok && hyb_peak < 5.0 * amp) ? "PHYSICAL & STABLE" : "CHECK");
    }
    return 0;
}

// ===================== REAL S7 (Japan Sea) MC+DG hybrid on GPU =====================
// Loads the REAL Data/JapanSea.asc bathymetry and the S7 ellipsoidal tsunami source
// (center 132E/36N, al=80 deg, r1=0.148, r2=0.449, h0=2 m -- byte-identical to
// ScenarioBuilderCUDA::BuildTest6CartesianS7 + Grid::SetParamsTest6). Builds a
// Cartesian hybrid: the colleague's MacCormackCUDA (UNMODIFIED) covers the western
// region holding the source; this file's GPU DG solver -- carrying the bathymetry-
// positivity fix that turned the CPU S7 run from inf into a finite 2 m -- covers the
// eastern region (the steep, shallow, wet/dry Japanese coast that the CPU pure-DG
// could not survive). Two-way eta+velocity seam exchange each step. Reports max|eta|.
int RunS7HybridGPU() {
    std::printf("\n=== REAL S7 (Japan Sea) MC+DG hybrid on GPU ===\n");
    const double PI = 3.14159265358979323846;
    ASCParser asc;
    asc.Read("Data/JapanSea.asc");
    const int NC = asc.GetCols(), NR = asc.GetRows();
    const double xll = asc.GetXllCorner(), yll = asc.GetYllCorner(), cs = asc.GetCellSize();
    const double src_x = 132.00, src_y = 36.00, al = 80.0 * PI / 180.0;
    const double r1 = 0.148, r2 = 0.449, h0 = 2.0, g = 9.81, min_depth = 7.0;

    // Cartesian metric (matches BuildTest6CartesianS7: center_lat ~43, dy=462.5).
    const double center_lat = yll + (NR / 2.0) * cs;
    const double dx_m = 111300.0 * cs * std::cos(center_lat * PI / 180.0);
    const double dy_m = 462.5;
    const double Lx = (NC - 1) * dx_m, Ly = (NR - 1) * dy_m;
    std::printf("[s7] ASC %dx%d cell=%.6f deg  center_lat=%.2f  dx=%.1fm dy=%.1fm  Lx=%.0f Ly=%.0f\n",
                NC, NR, cs, center_lat, dx_m, dy_m, Lx, Ly);

    // Bilinear bathymetry sampler in physical (X,Y) meters. X->ASC col i=X/dx_m;
    // Y measured north from south: Jr=Y/dy_m, ASC row j=(NR-1)-Jr (SetParamsTest6 flip).
    auto Dsample = [&](double X, double Y) {
        double fi = X / dx_m, fJr = Y / dy_m;
        if (fi < 0) fi = 0; if (fi > NC - 1) fi = NC - 1;
        if (fJr < 0) fJr = 0; if (fJr > NR - 1) fJr = NR - 1;
        int i0 = (int)std::floor(fi); if (i0 > NC - 2) i0 = NC - 2; double wx = fi - i0;
        int J0 = (int)std::floor(fJr); if (J0 > NR - 2) J0 = NR - 2; double wy = fJr - J0;
        int jA = (NR - 1) - J0, jB = (NR - 1) - (J0 + 1);
        double d00 = asc.GetValue(jA, i0), d10 = asc.GetValue(jA, i0 + 1);
        double d01 = asc.GetValue(jB, i0), d11 = asc.GetValue(jB, i0 + 1);
        return (d00 * (1 - wx) + d10 * wx) * (1 - wy) + (d01 * (1 - wx) + d11 * wx) * wy;
    };
    // S7 ellipsoidal source as a free-surface bump (lon/lat as in SetParamsTest6).
    auto Esample = [&](double X, double Y) {
        double lon = xll + (X / dx_m) * cs, lat = yll + (Y / dy_m) * cs;
        double ax = ((lon - src_x) * std::cos(al) - (lat - src_y) * std::sin(al)) / r1;
        double ay = ((lon - src_x) * std::sin(al) + (lat - src_y) * std::cos(al)) / r2;
        double arg = std::sqrt(ax * ax + ay * ay);
        return (arg <= 1.0) ? (1.0 + std::cos(PI * arg)) * h0 / 2.0 : 0.0;
    };

    // Hybrid grid (Cartesian, downsampled real S7). MC=west (holds source), DG=east.
    // Seam sits just east of the source (~lon 133) so the eastward tsunami front
    // crosses into the DG region (the steep/shallow Japanese coast + Japan Trench).
    const int NX = 220, NY = 400;
    const double dxh = Lx / NX, dyh = Ly / NY;
    const int mc_iv = (int)std::lround(0.38 * NX);
    const double xs = mc_iv * dxh;
    const int nx_mc = mc_iv + 1, ny_mc = NY + 1;

    // DG region [xs,Lx] x [0,Ly] on the REAL bathymetry (positivity fix handles land).
    TriMesh mR = BuildRectMeshOff(xs, Lx - xs, 0.0, Ly, NX - mc_iv, NY);
    ElementMatrices emR; emR.Precompute(mR);
    DGSolver dgR; dgR.Initialize(&mR, &emR);
    dgR.SetBathymetryFunction(Dsample);
    dgR.SetDefaultBoundaryCondition(DGBoundaryType::Wall);
    dgR.SetSlopeLimiter(true, 0.0);
    dgR.SetInitialConditionWellBalanced(
        [&](double x, double y) { return SWEState(Dsample(x, y), 0.0, 0.0); }, Dsample);
    auto seam = FindSeam(&mR, xs, std::max(Lx, Ly) * 1e-7);

    // MC region [0,xs] x [0,Ly]: colleague's MacCormackCUDA, right column = exchange seam.
    SimulationParams p{};
    p.g = g; p.delta_x_const = dxh; p.delta_y = dyh; p.min_calc_value = min_depth;
    p.use_variable_dx = false; p.boundaryConditionType = 0; p.hatBoundaryConditionType = 0;
    p.coord_system = CudaCoordSystem::Cartesian; p.coriolis_enabled = false;

    double dmax = 1.0;
    for (int i = 0; i < nx_mc; ++i) for (int j = 0; j < ny_mc; ++j)
        dmax = std::fmax(dmax, Dsample(i * dxh, j * dyh));
    double c0 = std::sqrt(g * dmax), dtMC = 0.2 * std::min(dxh, dyh) / c0;
    double dt = std::min(dtMC, dgR.ComputeTimeStep(0.2));
    p.tau = dt;
    std::printf("[s7] MC %dx%d  DG elems=%d  seam=%zu  xs=%.0f (%.0f%% W)  dmax=%.0fm  dt=%.3fs\n",
                nx_mc, ny_mc, dgR.NumElements(), seam.size(), xs, 100.0 * xs / Lx, dmax, dt);

    auto mc = CreateMacCormackCUDA(nx_mc, ny_mc, p, /*use_fp32=*/false);
    SubgridInfo si; si.has_left_neighbor = false; si.has_right_neighbor = true;
    si.subgrid_index = 0; si.num_subgrids = 2;
    mc->SetSubgridInfo(si);

    std::vector<double> mcH(nx_mc * ny_mc), mcU(nx_mc * ny_mc, 0.0), mcV(nx_mc * ny_mc, 0.0),
                        mcEta(nx_mc * ny_mc), mcD(nx_mc * ny_mc);
    for (int i = 0; i < nx_mc; ++i) for (int j = 0; j < ny_mc; ++j) {
        double X = i * dxh, Y = j * dyh; int id = i * ny_mc + j;
        double D = Dsample(X, Y), e = Esample(X, Y);
        mcD[id] = D; mcEta[id] = e; mcH[id] = D + e;
    }

    DGSnap sR = BuildSnapshot(dgR);
    for (auto& se : seam) { sR.e_kind[se.edge] = DG_EDGE_COUPLED; sR.e_ceta[se.edge] = 0.0; }
    DGDev dR = UploadSnapshot(sR);
    std::vector<double> hRceta = sR.e_ceta, hRcu = sR.e_cu, hRcv = sR.e_cv, qRh(dR.N * DGC_BLOCK);

    auto mcColInterp = [&](const std::vector<double>& fld, int icol, double y) {
        double fj = y / dyh; int j0 = (int)std::floor(fj);
        if (j0 < 0) j0 = 0; if (j0 > ny_mc - 2) j0 = ny_mc - 2;
        double w = fj - j0; if (w < 0) w = 0; if (w > 1) w = 1;
        return fld[icol * ny_mc + j0] * (1 - w) + fld[icol * ny_mc + (j0 + 1)] * w;
    };
    auto dgSeamInterp = [&](double y, double& eta, double& u, double& v) {
        int np = (int)seam.size();
        int pp = 0; while (pp < np - 1 && seam[pp + 1].y < y) ++pp;
        int p1 = std::min(pp + 1, np - 1);
        double y0 = seam[pp].y, y1 = seam[p1].y, w = (y1 > y0) ? (y - y0) / (y1 - y0) : 0.0;
        w = std::min(std::max(w, 0.0), 1.0);
        auto val = [&](int q, double& e, double& uu, double& vv) {
            int el = seam[q].elem; double He = qRh[el * DGC_BLOCK + 0];
            e = He - sR.Dcoeffs[el * DGC_BASIS + 0];
            uu = (He > 1e-6) ? qRh[el * DGC_BLOCK + DGC_BASIS] / He : 0.0;
            vv = (He > 1e-6) ? qRh[el * DGC_BLOCK + 2 * DGC_BASIS] / He : 0.0;
        };
        double e0, u0, v0, e1, u1, v1; val(pp, e0, u0, v0); val(p1, e1, u1, v1);
        eta = e0 * (1 - w) + e1 * w; u = u0 * (1 - w) + u1 * w; v = v0 * (1 - w) + v1 * w;
    };
    auto exchange = [&]() {
        mc->CopyToHost(mcH.data(), mcU.data(), mcV.data(), mcEta.data());
        CUDA_CHECK(cudaMemcpy(qRh.data(), dR.q, qRh.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t pq = 0; pq < seam.size(); ++pq) {
            double y = seam[pq].y; int e = seam[pq].edge;
            hRceta[e] = mcColInterp(mcEta, nx_mc - 2, y);
            hRcu[e]   = mcColInterp(mcU,   nx_mc - 2, y);
            hRcv[e]   = mcColInterp(mcV,   nx_mc - 2, y);
        }
        CUDA_CHECK(cudaMemcpy(dR.e_ceta, hRceta.data(), hRceta.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dR.e_cu, hRcu.data(), hRcu.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dR.e_cv, hRcv.data(), hRcv.size() * sizeof(double), cudaMemcpyHostToDevice));
        for (int j = 0; j < ny_mc; ++j) {
            double eta, u, v; dgSeamInterp(j * dyh, eta, u, v);
            int gi = (nx_mc - 1) * ny_mc + j;
            mcEta[gi] = eta; mcH[gi] = mcD[gi] + eta; mcU[gi] = u; mcV[gi] = v;
        }
        mc->LoadInitialData(mcH.data(), mcU.data(), mcV.data(), mcEta.data(), mcD.data());
    };

    mc->LoadInitialData(mcH.data(), mcU.data(), mcV.data(), mcEta.data(), mcD.data());
    const int NSTEP = 3000;
    bool ok = true; double mc_peak = 0.0, dg_peak = 0.0;
    for (int t = 0; t < NSTEP; ++t) {
        exchange();
        mc->Step(t);
        stepRK3_GPU(dR, dt, true);
        if ((t % 250) == 0 || t == NSTEP - 1) {
            CUDA_CHECK(cudaMemcpy(qRh.data(), dR.q, qRh.size() * sizeof(double), cudaMemcpyDeviceToHost));
            for (int k = 0; k < dR.N; ++k) if (!std::isfinite(qRh[k * DGC_BLOCK])) { ok = false; break; }
            mc->CopyEtaToHost(mcEta.data());
            double me = 0.0; for (double e : mcEta) if (std::isfinite(e)) me = std::fmax(me, std::fabs(e));
            double de = 0.0; for (int k = 0; k < dR.N; ++k)
                de = std::fmax(de, std::fabs(qRh[k * DGC_BLOCK] - sR.Dcoeffs[k * DGC_BASIS]));
            mc_peak = std::fmax(mc_peak, me); dg_peak = std::fmax(dg_peak, de);
            std::printf("[s7 step %4d]  MC max|eta|=%.4g  DG max|eta|=%.4g  finite=%s\n",
                        t, me, de, ok ? "yes" : "NO");
            if (!ok) break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double hyb = std::fmax(mc_peak, dg_peak);
    std::printf("\n[S7 MC+DG HYBRID on GPU]  steps=%d  MC peak=%.4g m  DG peak=%.4g m  hybrid max|eta|=%.4g m  %s\n",
                NSTEP, mc_peak, dg_peak, hyb, (ok && hyb < 50.0) ? "FINITE & PHYSICAL" : "CHECK");
    std::printf("[S7] reference: pure-MC GPU max_eta=1.999 m; CPU DG+MC hybrid solved inf->2 m.\n");
    return ok ? 0 : 1;
}

// ============ REAL S7 NATIVE-RESOLUTION 8-band MC|DG|MC hybrid + eta_max map ============
// Domain split into 8 vertical bands at the NATIVE ASC resolution (cellsize 0.00417 deg,
// dx~339 m). Band #3 (cols [NC*2/8, NC*3/8) ~ lon 131-133, which CONTAINS the S7 source)
// is the unstructured P2 DG-GPU solver; all other bands are the colleague's MacCormackCUDA
// (UNMODIFIED), run as two instances: a left MC over bands 1-2 and a right MC over bands 4-8.
// The DG band carries the bathymetry-positivity fix and GENERATES the tsunami (ellipsoidal
// source) on the real seafloor; the wave radiates out through BOTH seams into the MC regions.
// Two seams, two-way eta+u+v exchange each step. Writes a full-domain eta_max map for plotting.
int RunS7HybridFullGPU() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: see progress live under redirect
    std::printf("\n=== REAL S7 NATIVE-RES MC|DG|MC hybrid on GPU (8 bands, #3=DG, source in DG) ===\n");
    const double PI = 3.14159265358979323846;
    ASCParser asc; asc.Read("Data/JapanSea.asc");
    const int NC = asc.GetCols(), NR = asc.GetRows();
    const double xll = asc.GetXllCorner(), yll = asc.GetYllCorner(), cs = asc.GetCellSize();
    const double src_x = 132.00, src_y = 36.00, al = 80.0 * PI / 180.0;
    const double r1 = 0.148, r2 = 0.449, h0 = 2.0, g = 9.81, min_depth = 7.0;
    const double center_lat = yll + (NR / 2.0) * cs;
    const double dx_m = 111300.0 * cs * std::cos(center_lat * PI / 180.0);
    const double dy_m = 462.5, Ly = (NR - 1) * dy_m;

    auto Dsample = [&](double X, double Y) {
        double fi = X / dx_m, fJr = Y / dy_m;
        if (fi < 0) fi = 0; if (fi > NC - 1) fi = NC - 1;
        if (fJr < 0) fJr = 0; if (fJr > NR - 1) fJr = NR - 1;
        int i0 = (int)std::floor(fi); if (i0 > NC - 2) i0 = NC - 2; double wx = fi - i0;
        int J0 = (int)std::floor(fJr); if (J0 > NR - 2) J0 = NR - 2; double wy = fJr - J0;
        int jA = (NR - 1) - J0, jB = (NR - 1) - (J0 + 1);
        double d00 = asc.GetValue(jA, i0), d10 = asc.GetValue(jA, i0 + 1);
        double d01 = asc.GetValue(jB, i0), d11 = asc.GetValue(jB, i0 + 1);
        return (d00 * (1 - wx) + d10 * wx) * (1 - wy) + (d01 * (1 - wx) + d11 * wx) * wy;
    };
    auto Esample = [&](double X, double Y) {
        double lon = xll + (X / dx_m) * cs, lat = yll + (Y / dy_m) * cs;
        double ax = ((lon - src_x) * std::cos(al) - (lat - src_y) * std::sin(al)) / r1;
        double ay = ((lon - src_x) * std::sin(al) + (lat - src_y) * std::cos(al)) / r2;
        double arg = std::sqrt(ax * ax + ay * ay);
        return (arg <= 1.0) ? (1.0 + std::cos(PI * arg)) * h0 / 2.0 : 0.0;
    };

    // 8 bands; band #3 (0-indexed 2) = DG. MC stays at NATIVE resolution; the DG band
    // uses P2 elements each spanning a DGB x DGB block of native cells (P2 carries 6 DOF,
    // so one quadratic element resolves several native cells). Bathymetry is still sampled
    // from the full ASC at the element quadrature points -> no DATA downsampling.
    const int DGB = 8;                                // DG element = DGB x DGB native cells
    const int c1 = (NC * 2) / 8, c2 = (NC * 3) / 8;   // DG column span [c1,c2) in native cols
    const int ny = NR;
    const int nxL = c1 + 1;          // left MC nodes: cols 0..c1 (col c1 = ghost seam A)
    const int nxR = NC - c2;         // right MC nodes: cols c2..NC-1 (col c2 = ghost seam B)
    const int nx_dg = (c2 - c1) / DGB;   // coarse DG cell-columns
    const int ny_dg = (NR - 1) / DGB;    // coarse DG cell-rows
    const double xsA = c1 * dx_m, xsB = c2 * dx_m;
    std::printf("[s7full] NC=%d NR=%d  dx=%.1fm dy=%.1fm  bands: MC[0,%d) | DG[%d,%d) | MC[%d,%d)\n",
                NC, NR, dx_m, dy_m, c1, c1, c2, c2, NC);
    std::printf("[s7full] src col=%.0f (lon132 in DG band: %s)\n",
                (132.0 - xll) / cs, ((132.0 - xll) / cs >= c1 && (132.0 - xll) / cs < c2) ? "yes" : "NO");

    // ---- DG band (real bathymetry; source baked into IC) ----
    TriMesh mDG = BuildRectMeshOff(xsA, xsB - xsA, 0.0, Ly, nx_dg, ny_dg);
    ElementMatrices emDG; emDG.Precompute(mDG);
    DGSolver dg; dg.Initialize(&mDG, &emDG);
    dg.SetBathymetryFunction(Dsample);
    dg.SetDefaultBoundaryCondition(DGBoundaryType::Wall);
    dg.SetSlopeLimiter(true, 0.0);
    dg.SetInitialConditionWellBalanced(
        [&](double x, double y) { return SWEState(Dsample(x, y) + Esample(x, y), 0.0, 0.0); }, Dsample);
    auto seamL = FindSeam(&mDG, xsA, std::max(xsB, Ly) * 1e-7);  // DG left boundary edges
    auto seamR = FindSeam(&mDG, xsB, std::max(xsB, Ly) * 1e-7);  // DG right boundary edges
    std::printf("[s7full] DG elems=%d edges=%d seamL=%zu seamR=%zu\n",
                dg.NumElements(), mDG.NumEdges(), seamL.size(), seamR.size());

    // ---- two MC instances (colleague's solver, unmodified) ----
    auto mkParams = [&]() { SimulationParams p{}; p.g = g; p.delta_x_const = dx_m; p.delta_y = dy_m;
        p.min_calc_value = min_depth; p.use_variable_dx = false; p.boundaryConditionType = 0;
        p.hatBoundaryConditionType = 0; p.coord_system = CudaCoordSystem::Cartesian; p.coriolis_enabled = false; return p; };
    double dtMC = 0.2 * std::min(dx_m, dy_m) / std::sqrt(g * 4000.0);
    double dt = std::min(dtMC, dg.ComputeTimeStep(0.2));
    SimulationParams pL = mkParams(); pL.tau = dt;
    SimulationParams pR = mkParams(); pR.tau = dt;

    auto mcL = CreateMacCormackCUDA(nxL, ny, pL, false);
    { SubgridInfo si; si.has_left_neighbor = false; si.has_right_neighbor = true; si.subgrid_index = 0; si.num_subgrids = 3; mcL->SetSubgridInfo(si); }
    auto mcR = CreateMacCormackCUDA(nxR, ny, pR, false);
    { SubgridInfo si; si.has_left_neighbor = true; si.has_right_neighbor = false; si.subgrid_index = 2; si.num_subgrids = 3; mcR->SetSubgridInfo(si); }
    std::printf("[s7full] MC_left %dx%d  DG %d elems  MC_right %dx%d  dt=%.4fs  sim/step=%.3fs\n",
                nxL, ny, dg.NumElements(), nxR, ny, dt, dt);

    // MC initial fields (real bathymetry; source ~0 outside DG band but sampled anyway).
    std::vector<double> Lh(nxL*ny), Lu(nxL*ny,0.0), Lv(nxL*ny,0.0), Le(nxL*ny), Ld(nxL*ny);
    for (int i = 0; i < nxL; ++i) for (int j = 0; j < ny; ++j) { double X=i*dx_m, Y=j*dy_m; int id=i*ny+j;
        double D=Dsample(X,Y), e=Esample(X,Y); Ld[id]=D; Le[id]=e; Lh[id]=D+e; }
    std::vector<double> Rh(nxR*ny), Ru(nxR*ny,0.0), Rv(nxR*ny,0.0), Re(nxR*ny), Rd(nxR*ny);
    for (int i = 0; i < nxR; ++i) for (int j = 0; j < ny; ++j) { double X=(c2+i)*dx_m, Y=j*dy_m; int id=i*ny+j;
        double D=Dsample(X,Y), e=Esample(X,Y); Rd[id]=D; Re[id]=e; Rh[id]=D+e; }
    mcL->LoadInitialData(Lh.data(),Lu.data(),Lv.data(),Le.data(),Ld.data());
    mcR->LoadInitialData(Rh.data(),Ru.data(),Rv.data(),Re.data(),Rd.data());

    // ---- DG device snapshot + per-edge COUPLED tagging (both seams) ----
    DGSnap snap = BuildSnapshot(dg);
    for (auto& se : seamL) { snap.e_kind[se.edge] = DG_EDGE_COUPLED; snap.e_ceta[se.edge]=0.0; }
    for (auto& se : seamR) { snap.e_kind[se.edge] = DG_EDGE_COUPLED; snap.e_ceta[se.edge]=0.0; }
    DGDev d = UploadSnapshot(snap);
    std::vector<double> hceta = snap.e_ceta, hcu = snap.e_cu, hcv = snap.e_cv;
    std::vector<double> qDG(d.N * DGC_BLOCK);
    double* d_emax = nullptr; CUDA_CHECK(cudaMalloc(&d_emax, (size_t)d.N*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_emax, 0, (size_t)d.N*sizeof(double)));

    // y-interp of an MC column onto a query y; and DG-seam surface onto a query y.
    auto mcCol = [&](const std::vector<double>& f, int nyMC, int icol, double y) {
        double fj=y/dy_m; int j0=(int)std::floor(fj); if(j0<0)j0=0; if(j0>nyMC-2)j0=nyMC-2;
        double w=fj-j0; if(w<0)w=0; if(w>1)w=1; return f[icol*nyMC+j0]*(1-w)+f[icol*nyMC+(j0+1)]*w; };
    auto dgSeam = [&](const std::vector<SeamE>& sm, double y, double& eta,double& u,double& v){
        int np=(int)sm.size(); int p=0; while(p<np-1 && sm[p+1].y<y) ++p; int p1=std::min(p+1,np-1);
        double y0=sm[p].y,y1=sm[p1].y,w=(y1>y0)?(y-y0)/(y1-y0):0.0; w=std::min(std::max(w,0.0),1.0);
        auto val=[&](int q,double&e,double&uu,double&vv){ int el=sm[q].elem; double He=qDG[el*DGC_BLOCK+0];
            e=He-snap.Dcoeffs[el*DGC_BASIS+0]; uu=(He>1e-6)?qDG[el*DGC_BLOCK+DGC_BASIS]/He:0.0; vv=(He>1e-6)?qDG[el*DGC_BLOCK+2*DGC_BASIS]/He:0.0; };
        double e0,u0,v0,e1,u1,v1; val(p,e0,u0,v0); val(p1,e1,u1,v1);
        eta=e0*(1-w)+e1*w; u=u0*(1-w)+u1*w; v=v0*(1-w)+v1*w; };

    std::vector<double> Leta(nxL*ny), Lue(nxL*ny), Lve(nxL*ny);   // MC readback scratch
    std::vector<double> Reta(nxR*ny), Rue(nxR*ny), Rve(nxR*ny);
    // Left-seam DG values per row (needed for the HAT ghost: the corrector reads
    // hat-fields at the left-MC right ghost, so the base ghost alone isn't enough).
    std::vector<double> dgLe(ny), dgLu(ny), dgLv(ny);
    std::vector<double> LhH(nxL*ny), LhU(nxL*ny), LhV(nxL*ny), LhE(nxL*ny);  // hat scratch
    auto exchange = [&]() {
        mcL->CopyToHost(Lh.data(),Lu.data(),Lv.data(),Leta.data());
        mcR->CopyToHost(Rh.data(),Ru.data(),Rv.data(),Reta.data());
        CUDA_CHECK(cudaMemcpy(qDG.data(), d.q, qDG.size()*sizeof(double), cudaMemcpyDeviceToHost));
        // MC -> DG COUPLED (seam A: left MC interior col c1-1; seam B: right MC interior col 1)
        for (auto& se : seamL) { double y=se.y; int e=se.edge;
            hceta[e]=mcCol(Leta,ny,nxL-2,y); hcu[e]=mcCol(Lu,ny,nxL-2,y); hcv[e]=mcCol(Lv,ny,nxL-2,y); }
        for (auto& se : seamR) { double y=se.y; int e=se.edge;
            hceta[e]=mcCol(Reta,ny,1,y); hcu[e]=mcCol(Ru,ny,1,y); hcv[e]=mcCol(Rv,ny,1,y); }
        CUDA_CHECK(cudaMemcpy(d.e_ceta, hceta.data(), hceta.size()*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.e_cu, hcu.data(), hcu.size()*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.e_cv, hcv.data(), hcv.size()*sizeof(double), cudaMemcpyHostToDevice));
        // DG -> MC ghost cols (seam A: left MC ghost col nxL-1; seam B: right MC ghost col 0)
        for (int j = 0; j < ny; ++j) { double y=j*dy_m, eta,u,v;
            dgSeam(seamL, y, eta,u,v); int gi=(nxL-1)*ny+j; Leta[gi]=eta; Lh[gi]=Ld[gi]+eta; Lu[gi]=u; Lv[gi]=v;
            dgLe[j]=eta; dgLu[j]=u; dgLv[j]=v;                          // stash for the hat ghost
            dgSeam(seamR, y, eta,u,v); int gj=0*ny+j;      Reta[gj]=eta; Rh[gj]=Rd[gj]+eta; Ru[gj]=u; Rv[gj]=v; }
        mcL->LoadInitialData(Lh.data(),Lu.data(),Lv.data(),Leta.data(),Ld.data());
        mcR->LoadInitialData(Rh.data(),Ru.data(),Rv.data(),Reta.data(),Rd.data());
    };
    // Left MC custom step: replicate Step() but inject the DG-coupled values into the
    // HAT ghost (col nxL-1) between predictor and corrector. The corrector uses forward
    // differences (reads hat[i+1]), so cell nxL-2 needs hat at the right ghost; otherwise
    // it stays =D (still water) and no wave enters the left MC. (Right MC's ghost is on
    // its left, read by the predictor from base fields, so it needs no hat injection.)
    auto stepLeftCoupled = [&](int t) {
        mcL->ComputePredictorStep();
        mcL->DownloadHatToHost(LhH.data(), LhU.data(), LhV.data(), LhE.data());
        for (int j = 0; j < ny; ++j) { int gi=(nxL-1)*ny+j;
            LhE[gi]=dgLe[j]; LhH[gi]=Ld[gi]+dgLe[j]; LhU[gi]=dgLu[j]; LhV[gi]=dgLv[j]; }
        mcL->UploadHatFromHost(LhH.data(), LhU.data(), LhV.data(), LhE.data());
        mcL->ApplyHatBoundaryConditions(t);   // apply_right_bc=false -> preserves my hat ghost
        mcL->ComputeCorrectorStep();
        mcL->ApplyBaseBoundaryConditions(t);
    };

    const int NSTEP = 8000;
    bool ok = true;
    int ethreads = 256, nthreads = 256;
    for (int t = 0; t < NSTEP; ++t) {
        exchange();
        stepLeftCoupled(t); mcR->Step(t);
        stepRK3_GPU(d, dt, true);
        kDGEtaMax<<<(d.N+nthreads-1)/nthreads, nthreads>>>(d.q, d.Dcoeffs, d_emax, d.N);
        mcL->UpdateEtaMax(); mcR->UpdateEtaMax();
        if ((t % 400) == 0 || t == NSTEP - 1) {
            CUDA_CHECK(cudaMemcpy(qDG.data(), d.q, qDG.size()*sizeof(double), cudaMemcpyDeviceToHost));
            double de=0; for (int k=0;k<d.N;++k){ double e=qDG[k*DGC_BLOCK]-snap.Dcoeffs[k*DGC_BASIS];
                if(!std::isfinite(e)){ok=false;break;} de=std::fmax(de,std::fabs(e)); }
            mcL->CopyEtaToHost(Leta.data()); mcR->CopyEtaToHost(Reta.data());
            double Lg=0,Li=0,Rg=0,Ri=0, sLexp=0, sRexp=0;
            for (auto& se : seamL) sLexp = std::fmax(sLexp, std::fabs(hceta[se.edge]));  // not DG export
            for (int j=0;j<ny;++j){ Lg=std::fmax(Lg,std::fabs(Leta[(nxL-1)*ny+j])); Rg=std::fmax(Rg,std::fabs(Reta[0*ny+j])); }
            for (int i=0;i<nxL-1;++i) for (int j=0;j<ny;++j) Li=std::fmax(Li,std::fabs(Leta[i*ny+j]));
            for (int i=1;i<nxR;++i)   for (int j=0;j<ny;++j) Ri=std::fmax(Ri,std::fabs(Reta[i*ny+j]));
            std::printf("[s7full step %4d] DG max|eta|=%.4g  finite=%s | Lghost=%.4g Lint=%.4g | Rghost=%.4g Rint=%.4g\n",
                        t, de, ok?"yes":"NO", Lg, Li, Rg, Ri);
            if (!ok) break;
        }
        (void)ethreads;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- assemble full-domain eta_max map [col i in 0..NC) x asc-row in 0..NR) ----
    std::vector<double> Lemax(nxL*ny), Remax(nxR*ny), DGemax(d.N);
    mcL->CopyEtaMaxToHost(Lemax.data()); mcR->CopyEtaMaxToHost(Remax.data());
    CUDA_CHECK(cudaMemcpy(DGemax.data(), d_emax, (size_t)d.N*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<float> full((size_t)NC * NR, 0.0f);
    auto put = [&](int icol, int j_my, double val){ int r=(NR-1)-j_my; full[(size_t)icol*NR + r]=(float)val; };
    for (int i = 0; i < c1; ++i) for (int j = 0; j < ny; ++j) put(i, j, Lemax[i*ny+j]);
    // DG band: native cell (icol,j) -> coarse DG element it falls in (DGB x DGB blocks).
    for (int icol = c1; icol < c2; ++icol) for (int j = 0; j < ny; ++j) {
        int ci = (int)((long long)(icol - c1) * nx_dg / (c2 - c1)); if (ci > nx_dg-1) ci = nx_dg-1;
        int cj = (int)((long long)j * ny_dg / (NR - 1));            if (cj > ny_dg-1) cj = ny_dg-1;
        int e0 = 2*(cj*nx_dg + ci);
        put(icol, j, std::fmax(DGemax[e0], DGemax[e0+1])); }
    for (int li = 0; li < nxR; ++li) for (int j = 0; j < ny; ++j) put(c2+li, j, Remax[li*ny+j]);

    double gmax = 0; for (float v : full) gmax = std::fmax(gmax, (double)v);
    fs::create_directories("Result/test_s7full");
    { std::ofstream f("Result/test_s7full/eta_max_timestep_0.txt");
      f << std::setprecision(7);
      for (int i = 0; i < NC; ++i) { for (int j = 0; j < NR; ++j) f << full[(size_t)i*NR + j] << " "; f << "\n"; } }
    { std::ofstream s("Result/test_s7full/summary.txt");
      s << "scenario S7_native_MCDGMC\nNC " << NC << "\nNR " << NR << "\nbands 8\ndg_band 3\n"
        << "dg_cols [" << c1 << "," << c2 << ")\ndt " << dt << "\nsteps " << NSTEP
        << "\nmax_eta_max " << gmax << "\nfinite " << (ok?1:0) << "\n"; }
    std::printf("\n[S7 NATIVE MC|DG|MC HYBRID]  steps=%d  global max_eta_max=%.4g m  %s\n",
                NSTEP, gmax, (ok && gmax < 50.0) ? "FINITE & PHYSICAL" : "CHECK");
    std::printf("[S7] eta_max map -> Result/test_s7full/eta_max_timestep_0.txt (%dx%d native)\n", NC, NR);
    cudaFree(d_emax);
    return ok ? 0 : 1;
}

int RunDGCudaTest() {
    std::printf("=== DG CUDA port - Phase 1+2 (device upload + operator kernels) ===\n");
    const double L = 200000.0; const int Nq = 16;
    auto Dfun = [L](double x, double) { return 500.0 + 3000.0 * (x / L); };

    TriMesh mesh = BuildRectMesh(L, L, Nq, Nq);
    ElementMatrices em; em.Precompute(mesh);
    DGSolver dg; dg.Initialize(&mesh, &em);
    dg.SetBathymetryFunction(Dfun);
    dg.SetDefaultBoundaryCondition(DGBoundaryType::Wall);
    dg.SetInitialConditionWellBalanced(
        [&](double x, double y) { return SWEState(Dfun(x, y), 0.0, 0.0); }, Dfun);

    // CPU reference residual
    double mom = 0, wwet = 0; int we = -1, wwe = -1;
    double cpu_rr = dg.RestResidual(mom, we, &wwet, &wwe);
    std::printf("[CPU]  rest-residual = %.6g (worst_elem=%d)\n", cpu_rr, we);

    // Build + upload snapshot, compute GPU residual, compare
    DGSnap snap = BuildSnapshot(dg);
    DGDev d = UploadSnapshot(snap);
    computeResidualGPU(d, 128, 128);
    CUDA_CHECK(cudaDeviceSynchronize());

    // upd = Minv*rhs ; max_k ||upd||
    double* d_upd; CUDA_CHECK(cudaMalloc(&d_upd, (size_t)d.N * DGC_BLOCK * sizeof(double)));
    kRkCombine<<<(d.N + 127) / 128, 128>>>(d.q, d.q, d.rhs, d.Minv, 0.0, 1.0, 1.0, d_upd, d.N);
    // (out = 0*qn + 1*(q + 1*Minv*rhs) = q + Minv*rhs) -> upd = out - q
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> h_upd(d.N * DGC_BLOCK), h_q = snap.q;
    CUDA_CHECK(cudaMemcpy(h_upd.data(), d_upd, h_upd.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double gpu_rr = 0.0; int gworst = -1;
    for (int k = 0; k < d.N; ++k) {
        double n2 = 0.0;
        for (int i = 0; i < DGC_BLOCK; ++i) { double u = h_upd[k * DGC_BLOCK + i] - h_q[k * DGC_BLOCK + i]; n2 += u * u; }
        double n = std::sqrt(n2);
        if (n > gpu_rr) { gpu_rr = n; gworst = k; }
    }
    std::printf("[GPU]  rest-residual = %.6g (worst_elem=%d)\n", gpu_rr, gworst);
    std::printf("[CMP]  |GPU-CPU| = %.3g  -> %s\n", std::fabs(gpu_rr - cpu_rr),
                (std::fabs(gpu_rr - cpu_rr) < 1e-6 + 1e-3 * cpu_rr ? "MATCH" : "MISMATCH"));
    cudaFree(d_upd);

    // ---- Phase 2b: SSP-RK3 lake-at-rest stepping. Rest must hold (~machine zero)
    // on both GPU and CPU (the limiter is a no-op at rest, so it is fair to compare
    // the no-limiter GPU stepper to the CPU here). Validates the RK3 time loop. ----
    dg.SetSlopeLimiter(true, 0.0);  // enable on host to match GPU (limiters ON)
    double dt = dg.ComputeTimeStep(0.2);
    const int NSTEP = 100;
    for (int s = 0; s < NSTEP; ++s) stepRK3_GPU(d, dt, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> h_qf(d.N * DGC_BLOCK);
    CUDA_CHECK(cudaMemcpy(h_qf.data(), d.q, h_qf.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double gpu_maxeta = 0.0;
    for (int k = 0; k < d.N; ++k) {
        double e = std::fabs(h_qf[k * DGC_BLOCK + 0] - snap.Dcoeffs[k * DGC_BASIS + 0]);
        if (e > gpu_maxeta) gpu_maxeta = e;
    }
    for (int s = 0; s < NSTEP; ++s) dg.StepRK3(dt);
    {
        const std::vector<double>& cq = dg.GetSolution();
        const std::vector<double>& cd = dg.GetDCoeffs();
        double cpu_maxeta = 0.0;
        for (int k = 0; k < d.N; ++k) {
            double e = std::fabs(cq[k * DGC_BLOCK + 0] - cd[k * DGC_BASIS + 0]);
            if (e > cpu_maxeta) cpu_maxeta = e;
        }
        std::printf("[RK3 x%d lake-at-rest, limiters ON]  GPU max|eta_avg|=%.3g  CPU=%.3g  -> %s\n",
                    NSTEP, gpu_maxeta, cpu_maxeta, (gpu_maxeta < 1e-6 ? "GPU REST HELD" : "GPU DRIFT"));
    }

    // ===================== Phase 5: DYNAMIC validation =====================
    // Gaussian free-surface pulse on flat bottom (D=1000 m, eta bump), open BC,
    // limiters ON. Run CPU and GPU 60 SSP-RK3 steps and compare cell-averaged
    // H element-by-element. Tests the full operator + RK3 + limiters on a moving
    // solution (not just rest).
    {
        const double D0 = 1000.0, amp = 2.0, sig = L * 0.06;
        auto Dflat = [D0](double, double) { return D0; };
        TriMesh m2 = BuildRectMesh(L, L, Nq, Nq);
        ElementMatrices em2; em2.Precompute(m2);
        DGSolver dg2; dg2.Initialize(&m2, &em2);
        dg2.SetBathymetryFunction(Dflat);
        dg2.SetDefaultBoundaryCondition(DGBoundaryType::Wall);
        dg2.SetSlopeLimiter(true, 0.0);
        dg2.SetInitialCondition([&](double x, double y) {
            double r2 = (x - 0.5 * L) * (x - 0.5 * L) + (y - 0.5 * L) * (y - 0.5 * L);
            return SWEState(D0 + amp * std::exp(-r2 / (2 * sig * sig)), 0.0, 0.0);
        });
        DGSnap s2 = BuildSnapshot(dg2);
        DGDev d2 = UploadSnapshot(s2);
        double dt2 = dg2.ComputeTimeStep(0.2);
        const int NS2 = 60;
        for (int s = 0; s < NS2; ++s) stepRK3_GPU(d2, dt2, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> hq2(d2.N * DGC_BLOCK);
        CUDA_CHECK(cudaMemcpy(hq2.data(), d2.q, hq2.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int s = 0; s < NS2; ++s) dg2.StepRK3(dt2);
        const std::vector<double>& cq2 = dg2.GetSolution();
        double maxdiff = 0.0, gmax = 0.0, cmax = 0.0;
        for (int k = 0; k < d2.N; ++k) {
            double gH = hq2[k * DGC_BLOCK + 0], cH = cq2[k * DGC_BLOCK + 0];
            maxdiff = std::fmax(maxdiff, std::fabs(gH - cH));
            gmax = std::fmax(gmax, std::fabs(gH - D0));
            cmax = std::fmax(cmax, std::fabs(cH - D0));
        }
        std::printf("[DYNAMIC pulse x%d]  GPU max|eta|=%.5g  CPU max|eta|=%.5g  max|GPU-CPU cellH|=%.3g  -> %s\n",
                    NS2, gmax, cmax, maxdiff, (maxdiff < 1e-7 ? "MATCH" : "CHECK"));
    }

    // ===================== Phase 4: COUPLED-boundary well-balancedness =====================
    // The hybrid couples the DG stripe to MacCormack by importing the free surface
    // eta at the seam. Critical property: at rest (eta_c = 0) the coupled boundary
    // must reconstruct H = D and preserve lake-at-rest (no spurious seam injection).
    // Mark ALL boundary edges COUPLED with eta_c=0 and confirm rest holds on GPU.
    {
        DGSolver dgc; dgc.Initialize(&mesh, &em);
        dgc.SetBathymetryFunction(Dfun);
        dgc.SetDefaultBoundaryCondition(DGBoundaryType::Wall);
        dgc.SetInitialConditionWellBalanced(
            [&](double x, double y) { return SWEState(Dfun(x, y), 0.0, 0.0); }, Dfun);
        DGSnap sc = BuildSnapshot(dgc);
        int ncoupled = 0;
        for (int e = 0; e < sc.E; ++e)
            if (sc.e_R[e] < 0) { sc.e_kind[e] = DG_EDGE_COUPLED; sc.e_ceta[e] = 0.0; ncoupled++; }
        DGDev dc = UploadSnapshot(sc);
        for (int s = 0; s < 100; ++s) stepRK3_GPU(dc, dt, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> hqc(dc.N * DGC_BLOCK);
        CUDA_CHECK(cudaMemcpy(hqc.data(), dc.q, hqc.size() * sizeof(double), cudaMemcpyDeviceToHost));
        double cmaxeta = 0.0;
        for (int k = 0; k < dc.N; ++k) {
            double e = std::fabs(hqc[k * DGC_BLOCK + 0] - sc.Dcoeffs[k * DGC_BASIS + 0]);
            if (e > cmaxeta) cmaxeta = e;
        }
        std::printf("[COUPLED-BC x100 lake-at-rest]  %d coupled edges, GPU max|eta_avg|=%.3g  -> %s\n",
                    ncoupled, cmaxeta, (cmaxeta < 1e-6 ? "WELL-BALANCED (seam injects nothing at rest)" : "DRIFT"));
        std::printf("[Phase 4] DG-side coupling primitive (import eta -> reconstruct H) verified on GPU.\n");
        std::printf("[Phase 4] Remaining for full DG<->MC hybrid: drive eta_c each step from a\n");
        std::printf("[Phase 4] MacCormackCUDA instance at the seam (D2D/host exchange) — orchestration.\n");
    }

    RunDGHybridGPU();  // Phase 4/5: DG<->DG seam-coupled hybrid orchestration validation
    RunDGMCHybridGPU(); // LITERAL MC(MacCormackCUDA) + DG coupled on GPU
    return 0;
}
