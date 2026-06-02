// MacCormackCUDA.cu
// Template-based CUDA implementation of MacCormack method for shallow water equations.
// Both float and double instantiations are linked into the same binary; the wrapper
// chooses one at runtime via the CLI --precision=fp32|fp64 flag.

#include "MacCormackCUDA.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <limits>
#include <type_traits>

// ================== Indexing Macros ==================
#define IDX(i, j, ny) ((i) * (ny) + (j))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ================== Device Functions ==================

template <typename Real>
__device__ __forceinline__ Real get_delta_x(int /*i*/, int j,
    Real delta_x_const, const Real* delta_x_arr, bool use_variable_dx) {
    return use_variable_dx && delta_x_arr != nullptr ? delta_x_arr[j] : delta_x_const;
}

// ---- Davis 1987 / Liang 2007 TVD post-correction helpers (port of CPU SubGrid) ----
__device__ __forceinline__ double tvd_phi_van_leer(double r) {
    if (r <= 0.0) return 0.0;
    return 2.0 * r / (1.0 + r);
}
__device__ __forceinline__ double tvd_phi_minmod(double r) {
    if (r <= 0.0) return 0.0;
    return r < 1.0 ? r : 1.0;
}
// Safe ratio: 0 if denominator near zero (extremum → no high-order correction).
__device__ __forceinline__ double tvd_ratio(double d_up, double d_down) {
    const double eps = 1e-12;
    if (fabs(d_down) < eps) return 0.0;
    return d_up / d_down;
}
// B(nu, r) = 0.5*nu*(1-nu)*(1-phi(r)). phi and nu clamped to [0,1] so the
// post-correction is always dissipative (never anti-diffusive → blow-up).
__device__ __forceinline__ double tvd_B(double nu, double r, bool use_minmod) {
    double phi = use_minmod ? tvd_phi_minmod(r) : tvd_phi_van_leer(r);
    if (phi < 0.0) phi = 0.0;
    if (phi > 1.0) phi = 1.0;
    double nuc = nu;
    if (nuc < 0.0) nuc = 0.0;
    if (nuc > 1.0) nuc = 1.0;
    return 0.5 * nuc * (1.0 - nuc) * (1.0 - phi);
}
// Symmetric (Davis) face coefficient: average of upwind/downwind limiter contributions.
__device__ __forceinline__ double tvd_Bface(double nu, double d_up, double d_face,
                                             double d_down_right, bool use_minmod) {
    const double r_up = tvd_ratio(d_up, d_face);
    const double r_dn = tvd_ratio(d_down_right, d_face);
    return 0.5 * (tvd_B(nu, r_up, use_minmod) + tvd_B(nu, r_dn, use_minmod));
}

// ================== CUDA Kernels ==================

// Predictor Step (Half step - forward differences)
template <typename Real>
__global__ void computeHalfStepKernel(
    Real* __restrict__ H,
    Real* __restrict__ u,
    Real* __restrict__ v,
    Real* __restrict__ eta,
    const Real* __restrict__ D,
    Real* __restrict__ H_hat,
    Real* __restrict__ u_hat,
    Real* __restrict__ v_hat,
    Real* __restrict__ eta_hat,
    const Real* __restrict__ delta_x_arr,
    int nx, int ny,
    Real g, Real tau, Real delta_x_const, Real inv_dy,
    Real min_calc_value, bool use_variable_dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        const int ij = IDX(i, j, ny);
        const int imj = IDX(i - 1, j, ny);
        const int ijm = IDX(i, j - 1, ny);
        const int ipj = IDX(i + 1, j, ny);
        const int ijp = IDX(i, j + 1, ny);

        const Real Dij = D[ij];

        if (Dij < min_calc_value) {
            H_hat[ij] = Dij;
            u_hat[ij] = Real(0);
            v_hat[ij] = Real(0);
            eta_hat[ij] = Real(0);
            return;
        }

        const Real dx = get_delta_x<Real>(i, j, delta_x_const, delta_x_arr, use_variable_dx);
        const Real inv_dx = Real(1) / dx;

        const Real eij = eta[ij];
        const Real eimj = eta[imj];
        const Real eijm = eta[ijm];

        const Real uij = u[ij];
        const Real uimj = u[imj];
        const Real uijm = u[ijm];

        const Real vij = v[ij];
        const Real vimj = v[imj];
        const Real vijm = v[ijm];

        const Real Hij = H[ij];
        const Real Himj = H[imj];
        const Real Hijm = H[ijm];

        const Real Dimj = D[imj];
        const Real Dijm = D[ijm];
        const Real Dipj = D[ipj];
        const Real Dijp = D[ijp];

        const Real Hu_ij = Hij * uij;
        const Real Hu_imj = Himj * uimj;

        const Real Hv_ij = Hij * vij;
        const Real Hv_ijm = Hijm * vijm;

        const Real flux_H_x = (Hu_ij - Hu_imj) * inv_dx;
        const Real flux_H_y = (Hv_ij - Hv_ijm) * inv_dy;

        H_hat[ij] = Hij - tau * (flux_H_x + flux_H_y);

        const Real du_dx = (uij - uimj) * inv_dx;
        const Real du_dy = (uij - uijm) * inv_dy;

        const Real dv_dx = (vij - vimj) * inv_dx;
        const Real dv_dy = (vij - vijm) * inv_dy;

        const Real deta_dx = (eij - eimj) * inv_dx;
        const Real deta_dy = (eij - eijm) * inv_dy;

        Real u_hat_new = uij - tau * (uij * du_dx + vij * du_dy + g * deta_dx);
        Real v_hat_new = vij - tau * (uij * dv_dx + vij * dv_dy + g * deta_dy);

        // Zero velocity at land interface (matches CPU SubGrid::ComputeHalfStepCartesian)
        if (u_hat_new != Real(0)) {
            if ((i + 1 < nx - 1 && Dipj <= min_calc_value) ||
                (i - 1 > 0 && Dimj <= min_calc_value)) {
                u_hat_new = Real(0);
            }
        }
        if (v_hat_new != Real(0)) {
            if ((j + 1 < ny - 1 && Dijp <= min_calc_value) ||
                (j - 1 > 0 && Dijm <= min_calc_value)) {
                v_hat_new = Real(0);
            }
        }

        u_hat[ij] = u_hat_new;
        v_hat[ij] = v_hat_new;
        eta_hat[ij] = H_hat[ij] - Dij;
    }
}

// Corrector Step (Full step - backward differences)
template <typename Real>
__global__ void computeFullStepKernel(
    Real* __restrict__ H,
    Real* __restrict__ u,
    Real* __restrict__ v,
    Real* __restrict__ eta,
    const Real* __restrict__ D,
    const Real* __restrict__ H_hat,
    const Real* __restrict__ u_hat,
    const Real* __restrict__ v_hat,
    const Real* __restrict__ eta_hat,
    const Real* __restrict__ delta_x_arr,
    int nx, int ny,
    Real g, Real tau, Real delta_x_const, Real inv_dy,
    Real min_calc_value, bool use_variable_dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        const int ij = IDX(i, j, ny);
        const int ipj = IDX(i + 1, j, ny);
        const int ijp = IDX(i, j + 1, ny);
        const int imj = IDX(i - 1, j, ny);
        const int ijm = IDX(i, j - 1, ny);

        const Real Dij = D[ij];

        if (Dij < min_calc_value) {
            H[ij] = Dij;
            u[ij] = Real(0);
            v[ij] = Real(0);
            eta[ij] = Real(0);
            return;
        }

        const Real dx = get_delta_x<Real>(i, j, delta_x_const, delta_x_arr, use_variable_dx);
        const Real inv_dx = Real(1) / dx;

        const Real Dimj = D[imj];
        const Real Dipj = D[ipj];
        const Real Dijm = D[ijm];
        const Real Dijp = D[ijp];

        const bool at_x_interface = (Dimj < min_calc_value) || (Dipj < min_calc_value);
        const bool at_y_interface = (Dijm < min_calc_value) || (Dijp < min_calc_value);

        const Real H_hat_ij = H_hat[ij];
        const Real H_hat_ipj = H_hat[ipj];
        const Real H_hat_ijp = H_hat[ijp];

        const Real u_hat_ij = u_hat[ij];
        const Real u_hat_ipj = u_hat[ipj];
        const Real u_hat_ijp = u_hat[ijp];

        const Real v_hat_ij = v_hat[ij];
        const Real v_hat_ipj = v_hat[ipj];
        const Real v_hat_ijp = v_hat[ijp];

        const Real eta_hat_ij = eta_hat[ij];
        const Real eta_hat_ipj = eta_hat[ipj];
        const Real eta_hat_ijp = eta_hat[ijp];

        const Real Hu_hat_ij = H_hat_ij * u_hat_ij;
        const Real Hu_hat_ipj = H_hat_ipj * u_hat_ipj;

        const Real Hv_hat_ij = H_hat_ij * v_hat_ij;
        const Real Hv_hat_ijp = H_hat_ijp * v_hat_ijp;

        const Real flux_H_hat_x = (Hu_hat_ipj - Hu_hat_ij) * inv_dx;
        const Real flux_H_hat_y = (Hv_hat_ijp - Hv_hat_ij) * inv_dy;

        Real H_new = Real(0.5) * (H[ij] + H_hat_ij)
                   - Real(0.5) * tau * (flux_H_hat_x + flux_H_hat_y);

        const Real du_hat_dx = (u_hat_ipj - u_hat_ij) * inv_dx;
        const Real du_hat_dy = (u_hat_ijp - u_hat_ij) * inv_dy;
        const Real dv_hat_dx = (v_hat_ipj - v_hat_ij) * inv_dx;
        const Real dv_hat_dy = (v_hat_ijp - v_hat_ij) * inv_dy;
        const Real deta_hat_dx = (eta_hat_ipj - eta_hat_ij) * inv_dx;
        const Real deta_hat_dy = (eta_hat_ijp - eta_hat_ij) * inv_dy;

        Real u_new = Real(0.5) * (u[ij] + u_hat_ij)
                   - Real(0.5) * tau * (u_hat_ij * du_hat_dx + v_hat_ij * du_hat_dy + g * deta_hat_dx);
        Real v_new = Real(0.5) * (v[ij] + v_hat_ij)
                   - Real(0.5) * tau * (u_hat_ij * dv_hat_dx + v_hat_ij * dv_hat_dy + g * deta_hat_dy);

        if (at_x_interface) u_new = Real(0);
        if (at_y_interface) v_new = Real(0);

        if (H_new < min_calc_value) {
            H[ij] = Dij;
            u[ij] = Real(0);
            v[ij] = Real(0);
            eta[ij] = Real(0);
        } else {
            H[ij] = H_new;
            u[ij] = u_new;
            v[ij] = v_new;
            eta[ij] = H_new - Dij;
        }
    }
}

// X boundary conditions (left and right boundaries)
template <typename Real>
__global__ void applyXBoundaryKernel(
    Real* __restrict__ H,
    Real* __restrict__ u,
    Real* __restrict__ v,
    Real* __restrict__ eta,
    const Real* __restrict__ /*D*/,
    int nx, int ny,
    Real /*min_calc_value*/,
    bool apply_left, bool apply_right)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ny) {
        if (apply_left) {
            const int idx_0 = IDX(0, j, ny);
            const int idx_1 = IDX(1, j, ny);
            H[idx_0] = H[idx_1];
            u[idx_0] = u[idx_1];
            v[idx_0] = v[idx_1];
            eta[idx_0] = eta[idx_1];
        }
        if (apply_right) {
            const int idx_last = IDX(nx - 1, j, ny);
            const int idx_prev = IDX(nx - 2, j, ny);
            H[idx_last] = H[idx_prev];
            u[idx_last] = u[idx_prev];
            v[idx_last] = v[idx_prev];
            eta[idx_last] = eta[idx_prev];
        }
    }
}

// Y boundary conditions (top and bottom boundaries)
template <typename Real>
__global__ void applyYBoundaryKernel(
    Real* __restrict__ H,
    Real* __restrict__ u,
    Real* __restrict__ v,
    Real* __restrict__ eta,
    const Real* __restrict__ /*D*/,
    int nx, int ny,
    Real /*min_calc_value*/)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        const int idx_0 = IDX(i, 0, ny);
        const int idx_1 = IDX(i, 1, ny);
        H[idx_0] = H[idx_1];
        u[idx_0] = u[idx_1];
        v[idx_0] = v[idx_1];
        eta[idx_0] = eta[idx_1];

        const int idx_last = IDX(i, ny - 1, ny);
        const int idx_prev = IDX(i, ny - 2, ny);
        H[idx_last] = H[idx_prev];
        u[idx_last] = u[idx_prev];
        v[idx_last] = v[idx_prev];
        eta[idx_last] = eta[idx_prev];
    }
}

// Y boundary conditions with v=0 option (Hat3 needs zero_v_top=true)
template <typename Real>
__global__ void applyYBoundaryKernelWithVZero(
    Real* __restrict__ H,
    Real* __restrict__ u,
    Real* __restrict__ v,
    Real* __restrict__ eta,
    const Real* __restrict__ /*D*/,
    int nx, int ny,
    Real /*min_calc_value*/,
    bool zero_v_bottom, bool zero_v_top)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        const int idx_0 = IDX(i, 0, ny);
        const int idx_1 = IDX(i, 1, ny);
        H[idx_0] = H[idx_1];
        u[idx_0] = u[idx_1];
        v[idx_0] = zero_v_bottom ? Real(0) : v[idx_1];
        eta[idx_0] = eta[idx_1];

        const int idx_last = IDX(i, ny - 1, ny);
        const int idx_prev = IDX(i, ny - 2, ny);
        H[idx_last] = H[idx_prev];
        u[idx_last] = u[idx_prev];
        v[idx_last] = zero_v_top ? Real(0) : v[idx_prev];
        eta[idx_last] = eta[idx_prev];
    }
}

// Update eta_max
template <typename Real>
__global__ void updateEtaMaxKernel(
    const Real* __restrict__ eta,
    const Real* __restrict__ D,
    Real* __restrict__ eta_max,
    int nx, int ny, Real min_calc_value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        const int idx = IDX(i, j, ny);
        if (D[idx] < min_calc_value) return;
        const Real current_eta = eta[idx];
        if (current_eta > eta_max[idx]) {
            eta_max[idx] = current_eta;
        }
    }
}

// Accumulate max|u| and max|v| (matches CPU SubGrid: u_max/v_max store max abs velocity).
template <typename Real>
__global__ void updateUVMaxKernel(
    const Real* __restrict__ u,
    const Real* __restrict__ v,
    const Real* __restrict__ D,
    Real* __restrict__ u_max,
    Real* __restrict__ v_max,
    int nx, int ny, Real min_calc_value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        const int idx = IDX(i, j, ny);
        if (D[idx] < min_calc_value) return;
        const Real au = (u[idx] < Real(0)) ? -u[idx] : u[idx];
        const Real av = (v[idx] < Real(0)) ? -v[idx] : v[idx];
        if (au > u_max[idx]) u_max[idx] = au;
        if (av > v_max[idx]) v_max[idx] = av;
    }
}

// ================== Davis TVD post-correction kernels (port of CPU SubGrid) ==================
// Pass 1: compute (deltaEta, deltaHu, deltaHv) per interior cell from the post-corrector
// state. Gated identically to CPU pass 1 (bounds with boundary-skip, D>=grnd, all 8
// stencil neighbors wet). Reads only pre-TVD arrays → race-free.
template <typename Real>
__global__ void tvdComputeDeltasKernel(
    const Real* __restrict__ H, const Real* __restrict__ u, const Real* __restrict__ v,
    const Real* __restrict__ eta, const Real* __restrict__ D,
    const Real* __restrict__ delta_x_arr,
    Real* __restrict__ dEta, Real* __restrict__ dHu, Real* __restrict__ dHv,
    int nx, int ny, Real g, Real tau, Real delta_x_const, Real inv_dy,
    bool use_variable_dx, Real grnd, double shallow, bool force_minmod,
    int i_lo, int i_hi, int j_lo, int j_hi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < i_lo || i >= i_hi || j < j_lo || j >= j_hi) return;

    const int ij = IDX(i, j, ny);
    const double D_ij = D[ij];
    if (D_ij < grnd) return;

    const int im2 = IDX(i-2,j,ny), im1 = IDX(i-1,j,ny), ip1 = IDX(i+1,j,ny), ip2 = IDX(i+2,j,ny);
    const int jm2 = IDX(i,j-2,ny), jm1 = IDX(i,j-1,ny), jp1 = IDX(i,j+1,ny), jp2 = IDX(i,j+2,ny);
    if (D[im2] < grnd || D[im1] < grnd || D[ip1] < grnd || D[ip2] < grnd) return;
    if (D[jm2] < grnd || D[jm1] < grnd || D[jp1] < grnd || D[jp2] < grnd) return;

    bool use_minmod = force_minmod;
    if (!use_minmod) {
        if (D_ij < shallow ||
            D[im2] < shallow || D[im1] < shallow || D[ip1] < shallow || D[ip2] < shallow ||
            D[jm2] < shallow || D[jm1] < shallow || D[jp1] < shallow || D[jp2] < shallow)
            use_minmod = true;
    }

    const double dx = (double)get_delta_x<Real>(i, j, delta_x_const, delta_x_arr, use_variable_dx);
    const double inv_dx = 1.0 / dx;
    const double c0 = sqrt((double)g * (double)H[ij]);
    const double u0 = (double)u[ij], v0 = (double)v[ij];
    const double nu_x = (fabs(u0) + c0) * (double)tau * inv_dx;
    const double nu_y = (fabs(v0) + c0) * (double)tau * (double)inv_dy;

    // --- x-direction: limiter on eta (well-balanced), Hu/Hv conservative ---
    const double e_im2=(double)eta[im2], e_im1=(double)eta[im1], e_ij=(double)eta[ij],
                 e_ip1=(double)eta[ip1], e_ip2=(double)eta[ip2];
    const double H_im2=D[im2]+e_im2, H_im1=D[im1]+e_im1, H_ij=D_ij+e_ij, H_ip1=D[ip1]+e_ip1, H_ip2=D[ip2]+e_ip2;
    const double Hu_im2=H_im2*(double)u[im2], Hu_im1=H_im1*(double)u[im1], Hu_ij=H_ij*u0, Hu_ip1=H_ip1*(double)u[ip1], Hu_ip2=H_ip2*(double)u[ip2];
    const double Hv_im2=H_im2*(double)v[im2], Hv_im1=H_im1*(double)v[im1], Hv_ij=H_ij*v0, Hv_ip1=H_ip1*(double)v[ip1], Hv_ip2=H_ip2*(double)v[ip2];

    const double dE_xm=e_im1-e_im2, dE_x=e_ij-e_im1, dE_xp=e_ip1-e_ij, dE_xpp=e_ip2-e_ip1;
    const double dHu_xm=Hu_im1-Hu_im2, dHu_x=Hu_ij-Hu_im1, dHu_xp=Hu_ip1-Hu_ij, dHu_xpp=Hu_ip2-Hu_ip1;
    const double dHv_xm=Hv_im1-Hv_im2, dHv_x=Hv_ij-Hv_im1, dHv_xp=Hv_ip1-Hv_ij, dHv_xpp=Hv_ip2-Hv_ip1;

    const double Bx_E_p =tvd_Bface(nu_x,dE_x, dE_xp, dE_xpp,use_minmod), Bx_E_m =tvd_Bface(nu_x,dE_xm, dE_x, dE_xp, use_minmod);
    const double Bx_Hu_p=tvd_Bface(nu_x,dHu_x,dHu_xp,dHu_xpp,use_minmod),Bx_Hu_m=tvd_Bface(nu_x,dHu_xm,dHu_x,dHu_xp,use_minmod);
    const double Bx_Hv_p=tvd_Bface(nu_x,dHv_x,dHv_xp,dHv_xpp,use_minmod),Bx_Hv_m=tvd_Bface(nu_x,dHv_xm,dHv_x,dHv_xp,use_minmod);
    const double DxE =Bx_E_p *dE_xp -Bx_E_m *dE_x;
    const double DxHu=Bx_Hu_p*dHu_xp-Bx_Hu_m*dHu_x;
    const double DxHv=Bx_Hv_p*dHv_xp-Bx_Hv_m*dHv_x;

    // --- y-direction ---
    const double e_jm2=(double)eta[jm2], e_jm1=(double)eta[jm1], e_jp1=(double)eta[jp1], e_jp2=(double)eta[jp2];
    const double H_jm2=D[jm2]+e_jm2, H_jm1=D[jm1]+e_jm1, H_jp1=D[jp1]+e_jp1, H_jp2=D[jp2]+e_jp2;
    const double Hu_jm2=H_jm2*(double)u[jm2], Hu_jm1=H_jm1*(double)u[jm1], Hu_jp1=H_jp1*(double)u[jp1], Hu_jp2=H_jp2*(double)u[jp2];
    const double Hv_jm2=H_jm2*(double)v[jm2], Hv_jm1=H_jm1*(double)v[jm1], Hv_jp1=H_jp1*(double)v[jp1], Hv_jp2=H_jp2*(double)v[jp2];

    const double dE_ym=e_jm1-e_jm2, dE_y=e_ij-e_jm1, dE_yp=e_jp1-e_ij, dE_ypp=e_jp2-e_jp1;
    const double dHu_ym=Hu_jm1-Hu_jm2, dHu_y=Hu_ij-Hu_jm1, dHu_yp=Hu_jp1-Hu_ij, dHu_ypp=Hu_jp2-Hu_jp1;
    const double dHv_ym=Hv_jm1-Hv_jm2, dHv_y=Hv_ij-Hv_jm1, dHv_yp=Hv_jp1-Hv_ij, dHv_ypp=Hv_jp2-Hv_jp1;

    const double By_E_p =tvd_Bface(nu_y,dE_y, dE_yp, dE_ypp,use_minmod), By_E_m =tvd_Bface(nu_y,dE_ym, dE_y, dE_yp, use_minmod);
    const double By_Hu_p=tvd_Bface(nu_y,dHu_y,dHu_yp,dHu_ypp,use_minmod),By_Hu_m=tvd_Bface(nu_y,dHu_ym,dHu_y,dHu_yp,use_minmod);
    const double By_Hv_p=tvd_Bface(nu_y,dHv_y,dHv_yp,dHv_ypp,use_minmod),By_Hv_m=tvd_Bface(nu_y,dHv_ym,dHv_y,dHv_yp,use_minmod);
    const double DyE =By_E_p *dE_yp -By_E_m *dE_y;
    const double DyHu=By_Hu_p*dHu_yp-By_Hu_m*dHu_y;
    const double DyHv=By_Hv_p*dHv_yp-By_Hv_m*dHv_y;

    dEta[ij] = (Real)(DxE  + DyE);
    dHu[ij]  = (Real)(DxHu + DyHu);
    dHv[ij]  = (Real)(DxHv + DyHv);
}

// Pass 2: apply accumulated deltas (eta-form, well-balanced) with positivity floor.
// Re-checks the SAME gate as pass 1 so it only reads deltas that pass 1 wrote.
template <typename Real>
__global__ void tvdApplyDeltasKernel(
    Real* __restrict__ H, Real* __restrict__ u, Real* __restrict__ v, Real* __restrict__ eta,
    const Real* __restrict__ D,
    const Real* __restrict__ dEta, const Real* __restrict__ dHu, const Real* __restrict__ dHv,
    int nx, int ny, Real grnd, int i_lo, int i_hi, int j_lo, int j_hi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < i_lo || i >= i_hi || j < j_lo || j >= j_hi) return;

    const int ij = IDX(i, j, ny);
    const double D_ij = D[ij];
    if (D_ij < grnd) return;
    const int im2 = IDX(i-2,j,ny), im1 = IDX(i-1,j,ny), ip1 = IDX(i+1,j,ny), ip2 = IDX(i+2,j,ny);
    const int jm2 = IDX(i,j-2,ny), jm1 = IDX(i,j-1,ny), jp1 = IDX(i,j+1,ny), jp2 = IDX(i,j+2,ny);
    if (D[im2] < grnd || D[im1] < grnd || D[ip1] < grnd || D[ip2] < grnd) return;
    if (D[jm2] < grnd || D[jm1] < grnd || D[jp1] < grnd || D[jp2] < grnd) return;

    const double eta_old = (double)eta[ij];
    const double H_old   = D_ij + eta_old;
    const double Hu_old  = H_old * (double)u[ij];
    const double Hv_old  = H_old * (double)v[ij];

    const double eta_new = eta_old + (double)dEta[ij];
    const double H_new   = D_ij + eta_new;
    const double Hu_new  = Hu_old + (double)dHu[ij];
    const double Hv_new  = Hv_old + (double)dHv[ij];

    if (H_new < (double)grnd) {
        H[ij] = (Real)D_ij; u[ij] = Real(0); v[ij] = Real(0); eta[ij] = Real(0);
        return;
    }
    H[ij]   = (Real)H_new;
    u[ij]   = (Real)(Hu_new / H_new);
    v[ij]   = (Real)(Hv_new / H_new);
    eta[ij] = (Real)eta_new;
}

// North absorbing (Rayleigh) layer: damp u,v in first W rows by 1-alpha_max*cos^2(pi/2*j/(W-1)).
// Works on either main (u,v) or hat (u_hat,v_hat) arrays.
template <typename Real>
__global__ void absorbNorthKernel(
    Real* __restrict__ u, Real* __restrict__ v,
    int nx, int ny, int W, double alpha_max)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny || j >= W) return;
    const double x = (W > 1) ? double(j) / double(W - 1) : 0.0;
    const double c = cos(0.5 * 3.14159265358979323846 * x);
    const double factor = 1.0 - alpha_max * c * c;
    const int ij = IDX(i, j, ny);
    u[ij] = (Real)((double)u[ij] * factor);
    v[ij] = (Real)((double)v[ij] * factor);
}

// Reduction kernel (used by GetGlobalEtaMax via host scan; left as scaffold)
template <typename Real>
__global__ void reduceMaxKernel(
    const Real* __restrict__ input,
    Real* __restrict__ output,
    int n)
{
    extern __shared__ unsigned char sdata_raw[];
    Real* sdata = reinterpret_cast<Real*>(sdata_raw);

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? input[i] : Real(-1e30);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Dynamic left boundary (base) for tests 3, 4, 5
template <typename Real>
__global__ void applyDynamicLeftBoundaryKernel(
    Real* __restrict__ H,
    Real* __restrict__ u,
    Real* __restrict__ v,
    Real* __restrict__ eta,
    const Real* __restrict__ D,
    int nx, int ny,
    Real min_calc_value,
    int timestep,
    Real tau,
    Real g,
    Real T,
    Real waveScale)
{
    (void)nx; (void)min_calc_value;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ny) {
        const int idx_0 = IDX(0, j, ny);
        const int idx_1 = IDX(1, j, ny);

        const Real maxWaveTime = T / tau;
        const Real t = static_cast<Real>(timestep);

        // Match CPU ExchangeBoundaryData3/4: NO D-check guard at column 0.
        // u uses (D > 0) guard only; eta and H are set unconditionally to the wave value.
        if (t <= maxWaveTime) {
            const Real waveHeight = waveScale * (Real(1) - cos(t * tau * Real(2) * Real(M_PI) / T));
            const Real depth = D[idx_0];
            const Real uVal = (depth > Real(0)) ? sqrt(g / depth) : Real(0);
            eta[idx_0] = waveHeight;
            u[idx_0] = uVal;
            v[idx_0] = Real(0);
            H[idx_0] = D[idx_0] + eta[idx_0];
        } else {
            H[idx_0] = H[idx_1];
            u[idx_0] = u[idx_1];
            v[idx_0] = v[idx_1];
            eta[idx_0] = eta[idx_1];
        }
    }
}

// ================== Sommerfeld radiation (open) boundary — index 4 ==================
//
// Matches CPU BoundaryCondition::Radiation{Left,Right,Bottom,Top}Conditions
// (BoundaryCondition.cpp:322-401). Characteristic update of the ghost cell:
//     ghost = interior1 + sigma * (interior1 - interior2),
//     sigma = SafeWaveSpeed(g, H1) * tau / dn,   SafeWaveSpeed = (H1>0 ? sqrt(g*H1) : 0).
// dn = delta_x(j) for left/right edges (latitude-dependent if variable), delta_y for
// bottom/top. The device grid is X-padded (nx == gpu_nx), so column 0 / nx-1 are the
// ghost columns, exactly like the CPU SubGrid local grid (index 0 / NX-1).
//
// To reproduce the CPU corner ordering (Left/Right first, then Bottom/Top overwrite the
// corner rows) the host launches the X kernel and then the Y kernel on the same stream.
template <typename Real>
__global__ void radiationXBoundaryKernel(
    Real* __restrict__ H, Real* __restrict__ u, Real* __restrict__ v, Real* __restrict__ eta,
    const Real* __restrict__ delta_x_arr,
    int nx, int ny,
    Real g, Real tau, Real delta_x_const, bool use_variable_dx,
    bool apply_left, bool apply_right)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;

    const Real dx = get_delta_x<Real>(0, j, delta_x_const, delta_x_arr, use_variable_dx);
    const Real inv_dx = Real(1) / dx;

    if (apply_left) {
        const int g0 = IDX(0, j, ny);
        const int i1 = IDX(1, j, ny);
        const int i2 = IDX(2, j, ny);
        const Real H1 = H[i1];
        const Real sigma = (H1 > Real(0) ? sqrt(g * H1) : Real(0)) * tau * inv_dx;
        H[g0]   = H1       + sigma * (H1       - H[i2]);
        u[g0]   = u[i1]    + sigma * (u[i1]    - u[i2]);
        v[g0]   = v[i1]    + sigma * (v[i1]    - v[i2]);
        eta[g0] = eta[i1]  + sigma * (eta[i1]  - eta[i2]);
    }
    if (apply_right) {
        const int gL = IDX(nx - 1, j, ny);
        const int i1 = IDX(nx - 2, j, ny);
        const int i2 = IDX(nx - 3, j, ny);
        const Real H1 = H[i1];
        const Real sigma = (H1 > Real(0) ? sqrt(g * H1) : Real(0)) * tau * inv_dx;
        H[gL]   = H1       + sigma * (H1       - H[i2]);
        u[gL]   = u[i1]    + sigma * (u[i1]    - u[i2]);
        v[gL]   = v[i1]    + sigma * (v[i1]    - v[i2]);
        eta[gL] = eta[i1]  + sigma * (eta[i1]  - eta[i2]);
    }
}

template <typename Real>
__global__ void radiationYBoundaryKernel(
    Real* __restrict__ H, Real* __restrict__ u, Real* __restrict__ v, Real* __restrict__ eta,
    int nx, int ny,
    Real g, Real tau, Real inv_dy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;

    // Bottom edge (j=0) from interior rows j=1, j=2.
    {
        const int g0 = IDX(i, 0, ny);
        const int j1 = IDX(i, 1, ny);
        const int j2 = IDX(i, 2, ny);
        const Real H1 = H[j1];
        const Real sigma = (H1 > Real(0) ? sqrt(g * H1) : Real(0)) * tau * inv_dy;
        H[g0]   = H1       + sigma * (H1       - H[j2]);
        u[g0]   = u[j1]    + sigma * (u[j1]    - u[j2]);
        v[g0]   = v[j1]    + sigma * (v[j1]    - v[j2]);
        eta[g0] = eta[j1]  + sigma * (eta[j1]  - eta[j2]);
    }
    // Top edge (j=ny-1) from interior rows j=ny-2, j=ny-3.
    {
        const int gL = IDX(i, ny - 1, ny);
        const int j1 = IDX(i, ny - 2, ny);
        const int j2 = IDX(i, ny - 3, ny);
        const Real H1 = H[j1];
        const Real sigma = (H1 > Real(0) ? sqrt(g * H1) : Real(0)) * tau * inv_dy;
        H[gL]   = H1       + sigma * (H1       - H[j2]);
        u[gL]   = u[j1]    + sigma * (u[j1]    - u[j2]);
        v[gL]   = v[j1]    + sigma * (v[j1]    - v[j2]);
        eta[gL] = eta[j1]  + sigma * (eta[j1]  - eta[j2]);
    }
}

// ================== Spherical (Coriolis) Kernels ==================
//
// Matches CPU SubGrid::ComputeHalfStepSpherical / ComputeFullStepSpherical (SubGrid.cpp:179, :820).
// Coriolis: f = 2*OMEGA*sin(phi), where phi = PHI_0 + (j - ny/2) * delta_phi,
//           delta_phi = delta_y / R_EARTH.
// Predictor adds 3 extra boundary passes: j=0 forward-difference in y, j=ny-1 backward,
// and ghost columns i=0/i=nx-1 init (copy from base fields if wet, zeros if dry).

constexpr double R_EARTH = 6371000.0;
constexpr double OMEGA_2 = 2.0 * 7.2921e-5;  // 2 * Earth rotation rate (rad/s)

template <typename Real>
__global__ void computeHalfStepSphericalKernel(
    Real* __restrict__ H, Real* __restrict__ u, Real* __restrict__ v, Real* __restrict__ eta,
    const Real* __restrict__ D,
    Real* __restrict__ H_hat, Real* __restrict__ u_hat,
    Real* __restrict__ v_hat, Real* __restrict__ eta_hat,
    const Real* __restrict__ delta_x_arr,
    int nx, int ny,
    Real g, Real tau, Real delta_x_const, Real inv_dy,
    Real min_calc_value, bool use_variable_dx,
    Real PHI_0, Real delta_phi, Real two_omega, bool coriolis_enabled)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        const int ij = IDX(i, j, ny);
        const int imj = IDX(i - 1, j, ny);
        const int ijm = IDX(i, j - 1, ny);
        const int ipj = IDX(i + 1, j, ny);
        const int ijp = IDX(i, j + 1, ny);

        const Real Dij = D[ij];
        if (Dij < min_calc_value) {
            H_hat[ij] = Dij; u_hat[ij] = Real(0); v_hat[ij] = Real(0); eta_hat[ij] = Real(0);
            return;
        }

        const Real phi = PHI_0 + (Real(j) - Real(ny) * Real(0.5)) * delta_phi;
        const Real f = coriolis_enabled ? (two_omega * sin(phi)) : Real(0);

        const Real dx = get_delta_x<Real>(i, j, delta_x_const, delta_x_arr, use_variable_dx);
        const Real inv_dx = Real(1) / dx;

        const Real eij = eta[ij], eimj = eta[imj], eijm = eta[ijm];
        const Real uij = u[ij],   uimj = u[imj],   uijm = u[ijm];
        const Real vij = v[ij],   vimj = v[imj],   vijm = v[ijm];
        const Real Hij = H[ij],   Himj = H[imj],   Hijm = H[ijm];
        const Real Dimj = D[imj], Dijm = D[ijm], Dipj = D[ipj], Dijp = D[ijp];

        const Real flux_H_x = (Hij * uij - Himj * uimj) * inv_dx;
        const Real flux_H_y = (Hij * vij - Hijm * vijm) * inv_dy;
        H_hat[ij] = Hij - tau * (flux_H_x + flux_H_y);

        const Real du_dx = (uij - uimj) * inv_dx;
        const Real du_dy = (uij - uijm) * inv_dy;
        const Real dv_dx = (vij - vimj) * inv_dx;
        const Real dv_dy = (vij - vijm) * inv_dy;
        const Real deta_dx = (eij - eimj) * inv_dx;
        const Real deta_dy = (eij - eijm) * inv_dy;

        // Spherical predictor includes Coriolis terms: -f*v in u-eq, +f*u in v-eq.
        Real u_hat_new = uij - tau * (uij * du_dx + vij * du_dy + g * deta_dx - f * vij);
        Real v_hat_new = vij - tau * (uij * dv_dx + vij * dv_dy + g * deta_dy + f * uij);

        if (u_hat_new != Real(0)) {
            if ((i + 1 < nx - 1 && Dipj <= min_calc_value) ||
                (i - 1 > 0 && Dimj <= min_calc_value)) {
                u_hat_new = Real(0);
            }
        }
        if (v_hat_new != Real(0)) {
            if ((j + 1 < ny - 1 && Dijp <= min_calc_value) ||
                (j - 1 > 0 && Dijm <= min_calc_value)) {
                v_hat_new = Real(0);
            }
        }

        u_hat[ij] = u_hat_new;
        v_hat[ij] = v_hat_new;
        eta_hat[ij] = H_hat[ij] - Dij;
    }
}

// Spherical predictor: j=0 boundary using FORWARD difference in y.
template <typename Real>
__global__ void computeHalfStepSphericalKernelY0Forward(
    Real* __restrict__ H, Real* __restrict__ u, Real* __restrict__ v, Real* __restrict__ eta,
    const Real* __restrict__ D,
    Real* __restrict__ H_hat, Real* __restrict__ u_hat,
    Real* __restrict__ v_hat, Real* __restrict__ eta_hat,
    const Real* __restrict__ delta_x_arr,
    int nx, int ny,
    Real g, Real tau, Real delta_x_const, Real inv_dy,
    Real min_calc_value, bool use_variable_dx,
    Real PHI_0, Real delta_phi, Real two_omega, bool coriolis_enabled)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1 || i >= nx - 1) return;
    const int j = 0;

    const int ij = IDX(i, j, ny);
    const int imj = IDX(i - 1, j, ny);
    const int ijp = IDX(i, j + 1, ny);  // forward in y

    const Real Dij = D[ij];
    if (Dij < min_calc_value) {
        H_hat[ij] = Dij; u_hat[ij] = Real(0); v_hat[ij] = Real(0); eta_hat[ij] = Real(0);
        return;
    }

    const Real phi = PHI_0 + (Real(j) - Real(ny) * Real(0.5)) * delta_phi;
    const Real f = coriolis_enabled ? (two_omega * sin(phi)) : Real(0);

    const Real dx = get_delta_x<Real>(i, j, delta_x_const, delta_x_arr, use_variable_dx);
    const Real inv_dx = Real(1) / dx;

    const Real eij = eta[ij], eimj = eta[imj], eijp = eta[ijp];
    const Real uij = u[ij],   uimj = u[imj],   uijp = u[ijp];
    const Real vij = v[ij],   vimj = v[imj],   vijp = v[ijp];
    const Real Hij = H[ij],   Himj = H[imj],   Hijp = H[ijp];

    const Real flux_H_x = (Hij * uij - Himj * uimj) * inv_dx;
    const Real flux_H_y = (Hijp * vijp - Hij * vij) * inv_dy;  // forward
    H_hat[ij] = Hij - tau * (flux_H_x + flux_H_y);

    const Real du_dx = (uij - uimj) * inv_dx;
    const Real du_dy = (uijp - uij) * inv_dy;  // forward
    const Real dv_dx = (vij - vimj) * inv_dx;
    const Real dv_dy = (vijp - vij) * inv_dy;
    const Real deta_dx = (eij - eimj) * inv_dx;
    const Real deta_dy = (eijp - eij) * inv_dy;

    u_hat[ij] = uij - tau * (uij * du_dx + vij * du_dy + g * deta_dx - f * vij);
    v_hat[ij] = vij - tau * (uij * dv_dx + vij * dv_dy + g * deta_dy + f * uij);
    eta_hat[ij] = H_hat[ij] - Dij;
}

// Spherical predictor: j=ny-1 boundary using BACKWARD difference in y.
template <typename Real>
__global__ void computeHalfStepSphericalKernelYNBackward(
    Real* __restrict__ H, Real* __restrict__ u, Real* __restrict__ v, Real* __restrict__ eta,
    const Real* __restrict__ D,
    Real* __restrict__ H_hat, Real* __restrict__ u_hat,
    Real* __restrict__ v_hat, Real* __restrict__ eta_hat,
    const Real* __restrict__ delta_x_arr,
    int nx, int ny,
    Real g, Real tau, Real delta_x_const, Real inv_dy,
    Real min_calc_value, bool use_variable_dx,
    Real PHI_0, Real delta_phi, Real two_omega, bool coriolis_enabled)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1 || i >= nx - 1) return;
    const int j = ny - 1;

    const int ij = IDX(i, j, ny);
    const int imj = IDX(i - 1, j, ny);
    const int ijm = IDX(i, j - 1, ny);

    const Real Dij = D[ij];
    if (Dij < min_calc_value) {
        H_hat[ij] = Dij; u_hat[ij] = Real(0); v_hat[ij] = Real(0); eta_hat[ij] = Real(0);
        return;
    }

    const Real phi = PHI_0 + (Real(j) - Real(ny) * Real(0.5)) * delta_phi;
    const Real f = coriolis_enabled ? (two_omega * sin(phi)) : Real(0);

    const Real dx = get_delta_x<Real>(i, j, delta_x_const, delta_x_arr, use_variable_dx);
    const Real inv_dx = Real(1) / dx;

    const Real eij = eta[ij], eimj = eta[imj], eijm = eta[ijm];
    const Real uij = u[ij],   uimj = u[imj],   uijm = u[ijm];
    const Real vij = v[ij],   vimj = v[imj],   vijm = v[ijm];
    const Real Hij = H[ij],   Himj = H[imj],   Hijm = H[ijm];

    const Real flux_H_x = (Hij * uij - Himj * uimj) * inv_dx;
    const Real flux_H_y = (Hij * vij - Hijm * vijm) * inv_dy;  // backward
    H_hat[ij] = Hij - tau * (flux_H_x + flux_H_y);

    const Real du_dx = (uij - uimj) * inv_dx;
    const Real du_dy = (uij - uijm) * inv_dy;
    const Real dv_dx = (vij - vimj) * inv_dx;
    const Real dv_dy = (vij - vijm) * inv_dy;
    const Real deta_dx = (eij - eimj) * inv_dx;
    const Real deta_dy = (eij - eijm) * inv_dy;

    u_hat[ij] = uij - tau * (uij * du_dx + vij * du_dy + g * deta_dx - f * vij);
    v_hat[ij] = vij - tau * (uij * dv_dx + vij * dv_dy + g * deta_dy + f * uij);
    eta_hat[ij] = H_hat[ij] - Dij;
}

// Spherical predictor: init ghost columns i=0 and i=nx-1 from base fields
// (matches CPU SubGrid::ComputeHalfStepSpherical:441-474).
template <typename Real>
__global__ void initSphereHatGhostColumnsKernel(
    Real* __restrict__ H, Real* __restrict__ u, Real* __restrict__ v, Real* __restrict__ eta,
    const Real* __restrict__ D,
    Real* __restrict__ H_hat, Real* __restrict__ u_hat,
    Real* __restrict__ v_hat, Real* __restrict__ eta_hat,
    int nx, int ny, Real min_calc_value)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;

    for (int side = 0; side < 2; ++side) {
        const int i = (side == 0) ? 0 : (nx - 1);
        const int ij = IDX(i, j, ny);
        const Real Dij = D[ij];
        if (Dij < min_calc_value) {
            H_hat[ij] = Dij; u_hat[ij] = Real(0); v_hat[ij] = Real(0); eta_hat[ij] = Real(0);
        } else {
            H_hat[ij] = H[ij]; u_hat[ij] = u[ij]; v_hat[ij] = v[ij]; eta_hat[ij] = eta[ij];
        }
    }
}

// Spherical corrector (matches CPU SubGrid::ComputeFullStepSpherical:820-947).
// Only the interior loop — no special j=0/ny-1 handling needed (full step uses i+1,j+1 forward),
// matching CPU which also runs only the interior loop.
template <typename Real>
__global__ void computeFullStepSphericalKernel(
    Real* __restrict__ H, Real* __restrict__ u, Real* __restrict__ v, Real* __restrict__ eta,
    const Real* __restrict__ D,
    const Real* __restrict__ H_hat, const Real* __restrict__ u_hat,
    const Real* __restrict__ v_hat, const Real* __restrict__ eta_hat,
    const Real* __restrict__ delta_x_arr,
    int nx, int ny,
    Real g, Real tau, Real delta_x_const, Real inv_dy,
    Real min_calc_value, bool use_variable_dx,
    Real PHI_0, Real delta_phi, Real two_omega, bool coriolis_enabled)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        const int ij = IDX(i, j, ny);
        const int ipj = IDX(i + 1, j, ny);
        const int ijp = IDX(i, j + 1, ny);
        const int imj = IDX(i - 1, j, ny);
        const int ijm = IDX(i, j - 1, ny);

        const Real Dij = D[ij];
        if (Dij < min_calc_value) {
            H[ij] = Dij; u[ij] = Real(0); v[ij] = Real(0); eta[ij] = Real(0);
            return;
        }

        const Real phi = PHI_0 + (Real(j) - Real(ny) * Real(0.5)) * delta_phi;
        const Real f = coriolis_enabled ? (two_omega * sin(phi)) : Real(0);

        const Real dx = get_delta_x<Real>(i, j, delta_x_const, delta_x_arr, use_variable_dx);
        const Real inv_dx = Real(1) / dx;

        const Real Dimj = D[imj], Dipj = D[ipj], Dijm = D[ijm], Dijp = D[ijp];
        const bool at_x_interface = (Dimj < min_calc_value) || (Dipj < min_calc_value);
        const bool at_y_interface = (Dijm < min_calc_value) || (Dijp < min_calc_value);

        const Real H_hat_ij = H_hat[ij], H_hat_ipj = H_hat[ipj], H_hat_ijp = H_hat[ijp];
        const Real u_hat_ij = u_hat[ij], u_hat_ipj = u_hat[ipj], u_hat_ijp = u_hat[ijp];
        const Real v_hat_ij = v_hat[ij], v_hat_ipj = v_hat[ipj], v_hat_ijp = v_hat[ijp];
        const Real eta_hat_ij = eta_hat[ij], eta_hat_ipj = eta_hat[ipj], eta_hat_ijp = eta_hat[ijp];

        const Real flux_H_hat_x = (H_hat_ipj * u_hat_ipj - H_hat_ij * u_hat_ij) * inv_dx;
        const Real flux_H_hat_y = (H_hat_ijp * v_hat_ijp - H_hat_ij * v_hat_ij) * inv_dy;
        Real H_new = Real(0.5) * (H[ij] + H_hat_ij)
                   - Real(0.5) * tau * (flux_H_hat_x + flux_H_hat_y);

        const Real du_hat_dx = (u_hat_ipj - u_hat_ij) * inv_dx;
        const Real du_hat_dy = (u_hat_ijp - u_hat_ij) * inv_dy;
        const Real dv_hat_dx = (v_hat_ipj - v_hat_ij) * inv_dx;
        const Real dv_hat_dy = (v_hat_ijp - v_hat_ij) * inv_dy;
        const Real deta_hat_dx = (eta_hat_ipj - eta_hat_ij) * inv_dx;
        const Real deta_hat_dy = (eta_hat_ijp - eta_hat_ij) * inv_dy;

        // Spherical corrector with Coriolis.
        Real u_new = Real(0.5) * (u[ij] + u_hat_ij)
                   - Real(0.5) * tau * (u_hat_ij * du_hat_dx + v_hat_ij * du_hat_dy
                                        + g * deta_hat_dx - f * v_hat_ij);
        Real v_new = Real(0.5) * (v[ij] + v_hat_ij)
                   - Real(0.5) * tau * (u_hat_ij * dv_hat_dx + v_hat_ij * dv_hat_dy
                                        + g * deta_hat_dy + f * u_hat_ij);

        if (at_x_interface) u_new = Real(0);
        if (at_y_interface) v_new = Real(0);

        if (H_new < min_calc_value) {
            H[ij] = Dij; u[ij] = Real(0); v[ij] = Real(0); eta[ij] = Real(0);
        } else {
            H[ij] = H_new; u[ij] = u_new; v[ij] = v_new;
            eta[ij] = H_new - Dij;
        }
    }
}

// ================== Class Implementation ==================

template <typename Real>
MacCormackCUDA<Real>::MacCormackCUDA(int nx, int ny, const SimulationParams& params)
    : nx_(nx), ny_(ny), params_(params),
      d_eta_max_reduced_(nullptr),
      is_initialized_(false), has_variable_dx_(false)
{
    d_data_.nx = nx;
    d_data_.ny = ny;

    AllocateDeviceMemory();
    SetupCudaDimensions();

    // Active Region: start the band enclosing the source (seed_j_max + margin),
    // grown southward each step by UpdateActiveWindow. Full domain if disabled.
    ar_enabled_ = params_.ar_enabled && params_.ar_c_max > 0.0;
    ar_j_hi_ = ny_;
    if (ar_enabled_) {
        ar_j_hi_ = params_.ar_seed_j_max + params_.ar_margin_cells + 2;
        if (ar_j_hi_ > ny_) ar_j_hi_ = ny_;
        if (ar_j_hi_ < 4) ar_j_hi_ = 4;
    }

    h_scratch_.resize(static_cast<size_t>(nx_) * ny_);

    printf("MacCormackCUDA<%s> initialized: %dx%d grid\n", GetPrecisionName(), nx_, ny_);
    printf("Block size: %dx%d, Grid size: %dx%d\n",
           blockDim_.x, blockDim_.y, gridDim_.x, gridDim_.y);
}

template <typename Real>
MacCormackCUDA<Real>::~MacCormackCUDA() {
    FreeDeviceMemory();
}

template <typename Real>
void MacCormackCUDA<Real>::AllocateDeviceMemory() {
    size_t total_size = static_cast<size_t>(nx_) * ny_;
    size_t bytes = total_size * sizeof(Real);

    printf("Allocating GPU memory: %.2f MB per array (precision=%s)\n",
           bytes / (1024.0 * 1024.0), GetPrecisionName());

    CUDA_CHECK(cudaMalloc(&d_data_.H, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_.u, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_.v, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_.eta, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_.D, bytes));

    CUDA_CHECK(cudaMalloc(&d_data_.H_hat, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_.u_hat, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_.v_hat, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_.eta_hat, bytes));

    CUDA_CHECK(cudaMalloc(&d_data_.eta_max, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_.u_max, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_.v_max, bytes));
    CUDA_CHECK(cudaMalloc(&d_eta_max_reduced_, sizeof(Real)));

    CUDA_CHECK(cudaMemset(d_data_.H_hat, 0, bytes));
    CUDA_CHECK(cudaMemset(d_data_.u_hat, 0, bytes));
    CUDA_CHECK(cudaMemset(d_data_.v_hat, 0, bytes));
    CUDA_CHECK(cudaMemset(d_data_.eta_hat, 0, bytes));
    CUDA_CHECK(cudaMemset(d_data_.eta_max, 0, bytes));
    CUDA_CHECK(cudaMemset(d_data_.u_max, 0, bytes));
    CUDA_CHECK(cudaMemset(d_data_.v_max, 0, bytes));

    // TVD delta buffers — only when the Davis post-corrector is active.
    double n_arrays = 12.0;
    if (params_.tvd_enabled) {
        CUDA_CHECK(cudaMalloc(&d_data_.tvd_dEta, bytes));
        CUDA_CHECK(cudaMalloc(&d_data_.tvd_dHu, bytes));
        CUDA_CHECK(cudaMalloc(&d_data_.tvd_dHv, bytes));
        n_arrays = 15.0;
    }

    d_data_.delta_x = nullptr;

    printf("Total GPU memory allocated: %.2f MB\n", n_arrays * bytes / (1024.0 * 1024.0));
}

template <typename Real>
void MacCormackCUDA<Real>::FreeDeviceMemory() {
    if (d_data_.H) cudaFree(d_data_.H);
    if (d_data_.u) cudaFree(d_data_.u);
    if (d_data_.v) cudaFree(d_data_.v);
    if (d_data_.eta) cudaFree(d_data_.eta);
    if (d_data_.D) cudaFree(d_data_.D);
    if (d_data_.H_hat) cudaFree(d_data_.H_hat);
    if (d_data_.u_hat) cudaFree(d_data_.u_hat);
    if (d_data_.v_hat) cudaFree(d_data_.v_hat);
    if (d_data_.eta_hat) cudaFree(d_data_.eta_hat);
    if (d_data_.eta_max) cudaFree(d_data_.eta_max);
    if (d_data_.u_max) cudaFree(d_data_.u_max);
    if (d_data_.v_max) cudaFree(d_data_.v_max);
    if (d_data_.tvd_dEta) cudaFree(d_data_.tvd_dEta);
    if (d_data_.tvd_dHu) cudaFree(d_data_.tvd_dHu);
    if (d_data_.tvd_dHv) cudaFree(d_data_.tvd_dHv);
    if (d_data_.delta_x) cudaFree(d_data_.delta_x);
    if (d_eta_max_reduced_) cudaFree(d_eta_max_reduced_);
}

template <typename Real>
void MacCormackCUDA<Real>::SetupCudaDimensions() {
    blockDim_ = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    gridDim_ = dim3(
        (nx_ + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (ny_ + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        1
    );
    boundaryBlockDim_ = dim3(256, 1, 1);
    boundaryGridDim_X_ = dim3((ny_ + 255) / 256, 1, 1);
    boundaryGridDim_Y_ = dim3((nx_ + 255) / 256, 1, 1);
}

// ---- Host scratch helpers ----

template <typename Real>
static inline void copy_double_to_real(const double* src, Real* dst, size_t n) {
    if constexpr (std::is_same_v<Real, double>) {
        std::memcpy(dst, src, n * sizeof(double));
    } else {
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<Real>(src[i]);
    }
}

template <typename Real>
static inline void copy_real_to_double(const Real* src, double* dst, size_t n) {
    if constexpr (std::is_same_v<Real, double>) {
        std::memcpy(dst, src, n * sizeof(double));
    } else {
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(src[i]);
    }
}

template <typename Real>
void MacCormackCUDA<Real>::LoadInitialData(const double* h_H, const double* h_u, const double* h_v,
                                            const double* h_eta, const double* h_D) {
    const size_t total = static_cast<size_t>(nx_) * ny_;
    const size_t bytes = total * sizeof(Real);
    h_scratch_.resize(total);

    auto upload = [&](const double* host_src, Real* dev_dst) {
        copy_double_to_real<Real>(host_src, h_scratch_.data(), total);
        CUDA_CHECK(cudaMemcpy(dev_dst, h_scratch_.data(), bytes, cudaMemcpyHostToDevice));
    };

    upload(h_H, d_data_.H);
    upload(h_u, d_data_.u);
    upload(h_v, d_data_.v);
    upload(h_eta, d_data_.eta);
    upload(h_D, d_data_.D);

    // Active Region correctness: cells south of the active band are never written
    // by the predictor, so the corrector reading their stale H_hat must see the
    // lake-at-rest value H_hat=D (not the memset 0). Initialize H_hat:=D once;
    // u/v/eta_hat stay 0 (rest). Inside the band the predictor overwrites these.
    if (ar_enabled_) {
        CUDA_CHECK(cudaMemcpy(d_data_.H_hat, d_data_.D, bytes, cudaMemcpyDeviceToDevice));
    }

    is_initialized_ = true;
}

template <typename Real>
void MacCormackCUDA<Real>::LoadDeltaXArray(const double* h_delta_x, int size) {
    if (d_data_.delta_x) cudaFree(d_data_.delta_x);
    const size_t bytes = static_cast<size_t>(size) * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&d_data_.delta_x, bytes));
    std::vector<Real> tmp(size);
    copy_double_to_real<Real>(h_delta_x, tmp.data(), static_cast<size_t>(size));
    CUDA_CHECK(cudaMemcpy(d_data_.delta_x, tmp.data(), bytes, cudaMemcpyHostToDevice));
    has_variable_dx_ = true;
    params_.use_variable_dx = true;
    printf("Variable delta_x array loaded (%d elements, precision=%s)\n", size, GetPrecisionName());
}

// ---- Pipeline phases ----

// Active Region: gridDim covering only active rows [0, ar_j_hi_).
template <typename Real>
dim3 MacCormackCUDA<Real>::ActiveGridDim() const {
    const int jh = ar_enabled_ ? ar_j_hi_ : ny_;
    return dim3((nx_ + blockDim_.x - 1) / blockDim_.x,
                (jh   + blockDim_.y - 1) / blockDim_.y, 1);
}

// Grow the active band southward by the physical dependency cone + stencil + margin.
// Mirror of CPU ActiveRegionManager::UpdateActivation radius formula (called BEFORE
// the half-step of step t, so cover steps_to_cover = t+1).
template <typename Real>
void MacCormackCUDA<Real>::UpdateActiveWindow(int timestep) {
    if (!ar_enabled_) return;
    if (ar_j_hi_ >= ny_) return;  // already full
    const double dy = params_.delta_y;
    if (dy <= 0.0) { ar_j_hi_ = ny_; return; }
    const long long steps = static_cast<long long>(timestep) + 1;
    // physical cone (rows) + MacCormack/TVD stencil per step + margin
    long long per_step = 1;                              // MacCormack 3-pt
    if (params_.tvd_enabled) per_step += 2;              // Davis TVD ±2
    const double phys = std::ceil(params_.ar_c_max * (double)steps * params_.tau / dy);
    long long r = (long long)phys + per_step * steps + params_.ar_margin_cells;
    long long jh = (long long)params_.ar_seed_j_max + r;
    if (jh > ny_) jh = ny_;
    if (jh > ar_j_hi_) ar_j_hi_ = (int)jh;   // monotonic growth
}

template <typename Real>
void MacCormackCUDA<Real>::ComputePredictorStep() {
    if (!is_initialized_) return;
    const Real inv_dy = Real(1) / static_cast<Real>(params_.delta_y);
    const Real g_r = static_cast<Real>(params_.g);
    const Real tau_r = static_cast<Real>(params_.tau);
    const Real dx_const = static_cast<Real>(params_.delta_x_const);
    const Real min_calc = static_cast<Real>(params_.min_calc_value);

    const dim3 agd = ActiveGridDim();
    if (params_.coord_system == CudaCoordSystem::Spherical) {
        const Real delta_y_v = static_cast<Real>(params_.delta_y);
        const Real delta_phi = delta_y_v / static_cast<Real>(R_EARTH);
        const Real PHI_0 = static_cast<Real>(params_.center_latitude_rad);
        const Real two_omega = static_cast<Real>(OMEGA_2);

        // Main interior loop (1 <= i < nx-1, 1 <= j < ny-1), active rows only
        computeHalfStepSphericalKernel<Real><<<agd, blockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta, d_data_.D,
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            d_data_.delta_x,
            nx_, ny_,
            g_r, tau_r, dx_const, inv_dy, min_calc, has_variable_dx_,
            PHI_0, delta_phi, two_omega, params_.coriolis_enabled
        );

        // j=0 forward-difference boundary
        computeHalfStepSphericalKernelY0Forward<Real><<<boundaryGridDim_Y_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta, d_data_.D,
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            d_data_.delta_x,
            nx_, ny_,
            g_r, tau_r, dx_const, inv_dy, min_calc, has_variable_dx_,
            PHI_0, delta_phi, two_omega, params_.coriolis_enabled
        );

        // j=ny-1 backward-difference boundary
        computeHalfStepSphericalKernelYNBackward<Real><<<boundaryGridDim_Y_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta, d_data_.D,
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            d_data_.delta_x,
            nx_, ny_,
            g_r, tau_r, dx_const, inv_dy, min_calc, has_variable_dx_,
            PHI_0, delta_phi, two_omega, params_.coriolis_enabled
        );

        // Ghost columns i=0 and i=nx-1 init from base
        initSphereHatGhostColumnsKernel<Real><<<boundaryGridDim_X_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta, d_data_.D,
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            nx_, ny_, min_calc
        );
    } else {
        computeHalfStepKernel<Real><<<agd, blockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta, d_data_.D,
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            d_data_.delta_x,
            nx_, ny_,
            g_r, tau_r, dx_const, inv_dy, min_calc, has_variable_dx_
        );
    }
}

template <typename Real>
void MacCormackCUDA<Real>::ApplyHatBoundaryConditions(int timestep) {
    const int hatBC = params_.hatBoundaryConditionType;

    bool apply_left_bc = !subgrid_info_.has_left_neighbor;
    bool apply_right_bc = !subgrid_info_.has_right_neighbor;

    // Sommerfeld radiation (open) BC on the hat (predictor) variables — index 4.
    // Matches CPU ExchangeHatBoundaryDataRadiation: Left/Right (or neighbor copy) then
    // Bottom/Top, all on the *_hat fields. X kernel first, then Y kernel (corner order).
    if (hatBC == 4) {
        const Real g_r = static_cast<Real>(params_.g);
        const Real tau_r = static_cast<Real>(params_.tau);
        const Real dx_const = static_cast<Real>(params_.delta_x_const);
        const Real inv_dy = Real(1) / static_cast<Real>(params_.delta_y);
        radiationXBoundaryKernel<Real><<<boundaryGridDim_X_, boundaryBlockDim_>>>(
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            d_data_.delta_x, nx_, ny_,
            g_r, tau_r, dx_const, has_variable_dx_, apply_left_bc, apply_right_bc);
        radiationYBoundaryKernel<Real><<<boundaryGridDim_Y_, boundaryBlockDim_>>>(
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            nx_, ny_, g_r, tau_r, inv_dy);
        return;
    }

    applyXBoundaryKernel<Real><<<boundaryGridDim_X_, boundaryBlockDim_>>>(
        d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
        d_data_.D, nx_, ny_, static_cast<Real>(params_.min_calc_value),
        apply_left_bc, apply_right_bc
    );

    // Y-BC: CPU ExchangeHatBoundaryData{N} mapping
    //   Hat1 (index=0) — plain copy (DefaultLeftHat + DefaultRightHat + Y zero-gradient)
    //   Hat2 (index=1) — v_hat=0 on BOTH Y boundaries (bottom+top)
    //   Hat3 (index=2) — v_hat=0 on TOP only
    //   Hat4/Hat5 (index=3+) — plain copy
    const bool zero_v_bottom_hat = (hatBC == 1);
    const bool zero_v_top_hat = (hatBC == 1) || (hatBC == 2);
    if (zero_v_bottom_hat || zero_v_top_hat) {
        applyYBoundaryKernelWithVZero<Real><<<boundaryGridDim_Y_, boundaryBlockDim_>>>(
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            d_data_.D, nx_, ny_, static_cast<Real>(params_.min_calc_value),
            zero_v_bottom_hat, zero_v_top_hat
        );
    } else {
        applyYBoundaryKernel<Real><<<boundaryGridDim_Y_, boundaryBlockDim_>>>(
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            d_data_.D, nx_, ny_, static_cast<Real>(params_.min_calc_value)
        );
    }
    (void)timestep;
}

template <typename Real>
void MacCormackCUDA<Real>::ComputeCorrectorStep() {
    if (!is_initialized_) return;
    const Real inv_dy = Real(1) / static_cast<Real>(params_.delta_y);
    const Real g_r = static_cast<Real>(params_.g);
    const Real tau_r = static_cast<Real>(params_.tau);
    const Real dx_const = static_cast<Real>(params_.delta_x_const);
    const Real min_calc = static_cast<Real>(params_.min_calc_value);

    const dim3 agd = ActiveGridDim();
    if (params_.coord_system == CudaCoordSystem::Spherical) {
        const Real delta_y_v = static_cast<Real>(params_.delta_y);
        const Real delta_phi = delta_y_v / static_cast<Real>(R_EARTH);
        const Real PHI_0 = static_cast<Real>(params_.center_latitude_rad);
        const Real two_omega = static_cast<Real>(OMEGA_2);

        computeFullStepSphericalKernel<Real><<<agd, blockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta, d_data_.D,
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            d_data_.delta_x,
            nx_, ny_,
            g_r, tau_r, dx_const, inv_dy, min_calc, has_variable_dx_,
            PHI_0, delta_phi, two_omega, params_.coriolis_enabled
        );
    } else {
        computeFullStepKernel<Real><<<agd, blockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta, d_data_.D,
            d_data_.H_hat, d_data_.u_hat, d_data_.v_hat, d_data_.eta_hat,
            d_data_.delta_x,
            nx_, ny_,
            g_r, tau_r, dx_const, inv_dy, min_calc, has_variable_dx_
        );
    }
}

template <typename Real>
void MacCormackCUDA<Real>::ApplyBaseBoundaryConditions(int timestep) {
    const Real T = Real(300);
    const int baseBC = params_.boundaryConditionType;
    const Real waveScale = (baseBC == 2) ? Real(1) : Real(0.5);

    bool apply_left_bc = !subgrid_info_.has_left_neighbor;
    bool apply_right_bc = !subgrid_info_.has_right_neighbor;

    // Sommerfeld radiation (open) BC on the base variables — index 4. Must be handled
    // BEFORE the dynamic-wave path below (baseBC==4 would otherwise satisfy baseBC>=2 and
    // inject a wave). Matches CPU ExchangeBoundaryDataRadiation: Left/Right then Bottom/Top.
    if (baseBC == 4) {
        const Real g_r = static_cast<Real>(params_.g);
        const Real tau_r = static_cast<Real>(params_.tau);
        const Real dx_const = static_cast<Real>(params_.delta_x_const);
        const Real inv_dy = Real(1) / static_cast<Real>(params_.delta_y);
        radiationXBoundaryKernel<Real><<<boundaryGridDim_X_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta,
            d_data_.delta_x, nx_, ny_,
            g_r, tau_r, dx_const, has_variable_dx_, apply_left_bc, apply_right_bc);
        radiationYBoundaryKernel<Real><<<boundaryGridDim_Y_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta,
            nx_, ny_, g_r, tau_r, inv_dy);
        return;
    }

    const bool useDynamicBC = (baseBC >= 2);

    if (useDynamicBC && apply_left_bc) {
        applyDynamicLeftBoundaryKernel<Real><<<boundaryGridDim_X_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta,
            d_data_.D, nx_, ny_, static_cast<Real>(params_.min_calc_value),
            timestep, static_cast<Real>(params_.tau),
            static_cast<Real>(params_.g), T, waveScale
        );
        applyXBoundaryKernel<Real><<<boundaryGridDim_X_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta,
            d_data_.D, nx_, ny_, static_cast<Real>(params_.min_calc_value),
            /*apply_left=*/false, apply_right_bc
        );
    } else {
        applyXBoundaryKernel<Real><<<boundaryGridDim_X_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta,
            d_data_.D, nx_, ny_, static_cast<Real>(params_.min_calc_value),
            apply_left_bc, apply_right_bc
        );
    }

    // Y-BC: CPU ExchangeBoundaryData{N} mapping
    //   Data1 (index=0) — plain copy
    //   Data2 (index=1) — v=0 on BOTH Y boundaries
    //   Data3 (index=2) / Data4 (index=3) — plain copy on Y
    //   Data5 (index=4, unused — test 5 maps to Data4) — v=0 on TOP only
    const bool zero_v_bottom_base = (baseBC == 1);
    const bool zero_v_top_base = (baseBC == 1) || (baseBC == 5);
    if (zero_v_bottom_base || zero_v_top_base) {
        applyYBoundaryKernelWithVZero<Real><<<boundaryGridDim_Y_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta,
            d_data_.D, nx_, ny_, static_cast<Real>(params_.min_calc_value),
            zero_v_bottom_base, zero_v_top_base
        );
    } else {
        applyYBoundaryKernel<Real><<<boundaryGridDim_Y_, boundaryBlockDim_>>>(
            d_data_.H, d_data_.u, d_data_.v, d_data_.eta,
            d_data_.D, nx_, ny_, static_cast<Real>(params_.min_calc_value)
        );
    }
}

template <typename Real>
void MacCormackCUDA<Real>::ApplyNorthAbsorbingHat() {
    if (params_.absorb_layer_width <= 0) return;
    absorbNorthKernel<Real><<<gridDim_, blockDim_>>>(
        d_data_.u_hat, d_data_.v_hat, nx_, ny_,
        params_.absorb_layer_width, params_.absorb_alpha_max);
}

template <typename Real>
void MacCormackCUDA<Real>::ApplyNorthAbsorbingMain() {
    if (params_.absorb_layer_width <= 0) return;
    absorbNorthKernel<Real><<<gridDim_, blockDim_>>>(
        d_data_.u, d_data_.v, nx_, ny_,
        params_.absorb_layer_width, params_.absorb_alpha_max);
}

// Davis TVD post-correction: two race-free kernel passes (compute deltas → apply),
// gated by boundary-skip and the apply-every schedule. Port of CPU SubGrid::ApplyTvdPostCorrection.
template <typename Real>
void MacCormackCUDA<Real>::ApplyTvdPostCorrection(int timestep) {
    const int ae = (params_.tvd_apply_every > 0) ? params_.tvd_apply_every : 1;
    if (ae > 1 && ((timestep + 1) % ae) != 0) return;

    const int bskip = params_.tvd_boundary_skip;
    const int i_lo = (2 > bskip ? 2 : bskip);
    const int i_hi = ((nx_ - 2) < (nx_ - 1 - bskip) ? (nx_ - 2) : (nx_ - 1 - bskip));
    const int j_lo = (2 > bskip ? 2 : bskip);
    const int j_hi = ((ny_ - 2) < (ny_ - 1 - bskip) ? (ny_ - 2) : (ny_ - 1 - bskip));
    if (i_lo >= i_hi || j_lo >= j_hi) return;

    const Real g_r = static_cast<Real>(params_.g);
    const Real tau_r = static_cast<Real>(params_.tau);
    const Real dx_const = static_cast<Real>(params_.delta_x_const);
    const Real inv_dy = Real(1) / static_cast<Real>(params_.delta_y);
    const Real grnd = static_cast<Real>(params_.min_calc_value);

    const dim3 agd = ActiveGridDim();
    tvdComputeDeltasKernel<Real><<<agd, blockDim_>>>(
        d_data_.H, d_data_.u, d_data_.v, d_data_.eta, d_data_.D, d_data_.delta_x,
        d_data_.tvd_dEta, d_data_.tvd_dHu, d_data_.tvd_dHv,
        nx_, ny_, g_r, tau_r, dx_const, inv_dy, has_variable_dx_,
        grnd, params_.tvd_shallow_depth, params_.tvd_force_minmod,
        i_lo, i_hi, j_lo, j_hi);

    tvdApplyDeltasKernel<Real><<<agd, blockDim_>>>(
        d_data_.H, d_data_.u, d_data_.v, d_data_.eta, d_data_.D,
        d_data_.tvd_dEta, d_data_.tvd_dHu, d_data_.tvd_dHv,
        nx_, ny_, grnd, i_lo, i_hi, j_lo, j_hi);
}

template <typename Real>
void MacCormackCUDA<Real>::Step(int timestep) {
    if (ar_enabled_) UpdateActiveWindow(timestep);  // grow active band BEFORE half-step
    ComputePredictorStep();
    if (params_.absorb_enabled) ApplyNorthAbsorbingHat();
    ApplyHatBoundaryConditions(timestep);
    ComputeCorrectorStep();
    if (params_.absorb_enabled) ApplyNorthAbsorbingMain();
    if (params_.tvd_enabled) ApplyTvdPostCorrection(timestep);
    ApplyBaseBoundaryConditions(timestep);
}

template <typename Real>
void MacCormackCUDA<Real>::UpdateEtaMax() {
    updateEtaMaxKernel<Real><<<ActiveGridDim(), blockDim_>>>(
        d_data_.eta, d_data_.D, d_data_.eta_max, nx_, ny_,
        static_cast<Real>(params_.min_calc_value)
    );
}

template <typename Real>
void MacCormackCUDA<Real>::UpdateUVMax() {
    updateUVMaxKernel<Real><<<ActiveGridDim(), blockDim_>>>(
        d_data_.u, d_data_.v, d_data_.D, d_data_.u_max, d_data_.v_max, nx_, ny_,
        static_cast<Real>(params_.min_calc_value)
    );
}

template <typename Real>
double MacCormackCUDA<Real>::GetGlobalEtaMax() {
    Synchronize();
    const size_t total = static_cast<size_t>(nx_) * ny_;
    h_scratch_.resize(total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.eta_max,
                          total * sizeof(Real), cudaMemcpyDeviceToHost));
    Real max_val = h_scratch_[0];
    for (size_t i = 1; i < total; ++i) {
        if (h_scratch_[i] > max_val) max_val = h_scratch_[i];
    }
    return static_cast<double>(max_val);
}

template <typename Real>
void MacCormackCUDA<Real>::CopyToHost(double* h_H, double* h_u, double* h_v, double* h_eta) {
    const size_t total = static_cast<size_t>(nx_) * ny_;
    const size_t bytes = total * sizeof(Real);
    h_scratch_.resize(total);

    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.H, bytes, cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_H, total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.u, bytes, cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_u, total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.v, bytes, cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_v, total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.eta, bytes, cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_eta, total);
}

template <typename Real>
void MacCormackCUDA<Real>::CopyEtaToHost(double* h_eta) {
    const size_t total = static_cast<size_t>(nx_) * ny_;
    h_scratch_.resize(total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.eta,
                          total * sizeof(Real), cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_eta, total);
}

template <typename Real>
void MacCormackCUDA<Real>::CopyEtaMaxToHost(double* h_eta_max) {
    const size_t total = static_cast<size_t>(nx_) * ny_;
    h_scratch_.resize(total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.eta_max,
                          total * sizeof(Real), cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_eta_max, total);
}

template <typename Real>
void MacCormackCUDA<Real>::CopyUMaxToHost(double* h_u_max) {
    const size_t total = static_cast<size_t>(nx_) * ny_;
    h_scratch_.resize(total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.u_max,
                          total * sizeof(Real), cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_u_max, total);
}

template <typename Real>
void MacCormackCUDA<Real>::CopyVMaxToHost(double* h_v_max) {
    const size_t total = static_cast<size_t>(nx_) * ny_;
    h_scratch_.resize(total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.v_max,
                          total * sizeof(Real), cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_v_max, total);
}

template <typename Real>
void MacCormackCUDA<Real>::DownloadHatToHost(double* h_H_hat, double* h_u_hat,
                                              double* h_v_hat, double* h_eta_hat) {
    const size_t total = static_cast<size_t>(nx_) * ny_;
    const size_t bytes = total * sizeof(Real);
    h_scratch_.resize(total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.H_hat, bytes, cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_H_hat, total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.u_hat, bytes, cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_u_hat, total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.v_hat, bytes, cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_v_hat, total);
    CUDA_CHECK(cudaMemcpy(h_scratch_.data(), d_data_.eta_hat, bytes, cudaMemcpyDeviceToHost));
    copy_real_to_double<Real>(h_scratch_.data(), h_eta_hat, total);
}

template <typename Real>
void MacCormackCUDA<Real>::UploadHatFromHost(const double* h_H_hat, const double* h_u_hat,
                                              const double* h_v_hat, const double* h_eta_hat) {
    const size_t total = static_cast<size_t>(nx_) * ny_;
    const size_t bytes = total * sizeof(Real);
    h_scratch_.resize(total);
    copy_double_to_real<Real>(h_H_hat, h_scratch_.data(), total);
    CUDA_CHECK(cudaMemcpy(d_data_.H_hat, h_scratch_.data(), bytes, cudaMemcpyHostToDevice));
    copy_double_to_real<Real>(h_u_hat, h_scratch_.data(), total);
    CUDA_CHECK(cudaMemcpy(d_data_.u_hat, h_scratch_.data(), bytes, cudaMemcpyHostToDevice));
    copy_double_to_real<Real>(h_v_hat, h_scratch_.data(), total);
    CUDA_CHECK(cudaMemcpy(d_data_.v_hat, h_scratch_.data(), bytes, cudaMemcpyHostToDevice));
    copy_double_to_real<Real>(h_eta_hat, h_scratch_.data(), total);
    CUDA_CHECK(cudaMemcpy(d_data_.eta_hat, h_scratch_.data(), bytes, cudaMemcpyHostToDevice));
}

template <typename Real>
void MacCormackCUDA<Real>::Synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <>
const char* MacCormackCUDA<float>::GetPrecisionName() const { return "fp32"; }

template <>
const char* MacCormackCUDA<double>::GetPrecisionName() const { return "fp64"; }

// ================== Explicit instantiations ==================
template class MacCormackCUDA<float>;
template class MacCormackCUDA<double>;

// ================== Factory ==================
std::unique_ptr<IMacCormackCUDA> CreateMacCormackCUDA(int nx, int ny,
                                                     const SimulationParams& params,
                                                     bool use_fp32) {
    if (use_fp32) {
        return std::unique_ptr<IMacCormackCUDA>(new MacCormackCUDA<float>(nx, ny, params));
    }
    return std::unique_ptr<IMacCormackCUDA>(new MacCormackCUDA<double>(nx, ny, params));
}

#undef IDX
