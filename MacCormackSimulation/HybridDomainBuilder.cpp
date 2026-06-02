// HybridDomainBuilder.cpp - Реализация построителя гибридных доменов

#include "HybridDomainBuilder.h"
#include "BoundaryCondition.h"
#include "DGValidation.h"
#include <algorithm>
#include <barrier>
#include <cmath>
#include <stdexcept>

namespace {
    // Map boundary index to callback (same order as SingleLineGridSimulation constructor)
    std::function<void(std::shared_ptr<SubGrid>, double)> GetBoundaryFunc(int idx) {
        switch (idx) {
        case 0: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeBoundaryData1(s, t); };
        case 1: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeBoundaryData2(s, t); };
        case 2: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeBoundaryData3(s, t); };
        case 3: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeBoundaryData4(s, t); };
        default: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeBoundaryData1(s, t); };
        }
    }
    std::function<void(std::shared_ptr<SubGrid>, double)> GetHatBoundaryFunc(int idx) {
        switch (idx) {
        case 0: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeHatBoundaryData1(s, t); };
        case 1: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeHatBoundaryData2(s, t); };
        case 2: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeHatBoundaryData3(s, t); };
        case 3: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeHatBoundaryData4(s, t); };
        default: return [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeHatBoundaryData1(s, t); };
        }
    }

    enum NestedPatchId {
        PatchNW = 0,
        PatchN = 1,
        PatchNE = 2,
        PatchW = 3,
        PatchDG = 4,
        PatchE = 5,
        PatchSW = 6,
        PatchS = 7,
        PatchSE = 8
    };

    void CopyGridData(Grid& dst, const Grid& src) {
        for (int i = 0; i < dst.GetNx(); ++i) {
            for (int j = 0; j < dst.GetNy(); ++j) {
                dst.H(i, j) = src.H(i, j);
                dst.u(i, j) = src.u(i, j);
                dst.v(i, j) = src.v(i, j);
                dst.eta(i, j) = src.eta(i, j);
                dst.D(i, j) = src.D(i, j);
            }
        }
    }
}

std::shared_ptr<HybridDomain> HybridDomainBuilder::BuildFromParameters(
    DefaultParametersParser& params,
    Grid& global_grid) {

    // Извлечь параметры из парсера
    double g = 9.81;  // Константа
    double tau = params.GetDeltaT();
    int nx = global_grid.GetNx();
    int ny = global_grid.GetNy();
    int num_subgrids = params.GetNumSubgrids();
    CoordSystem coord_system = params.GetCoordSystem();
    double min_calc_value = params.GetMinDepthToCalc();

    // Создать функцию delta_x
    double delta_x_const = params.GetDeltaXStep();
    auto delta_x = [delta_x_const](int, int) { return delta_x_const; };

    double delta_y = params.GetDeltaYStep();

    // TODO: Извлечь center_latitude_rad из params если добавлено в DefaultParametersParser
    double center_latitude_rad = 0.785398163;  // По умолчанию 45°

    // Построить домен в legacy режиме (полосы)
    return BuildStructuredStripes(
        g, tau, delta_x, delta_y, nx, ny, num_subgrids,
        global_grid, coord_system, min_calc_value, center_latitude_rad
    );
}

std::shared_ptr<HybridDomain> HybridDomainBuilder::BuildStructuredStripes(
    double g,
    double tau,
    std::function<double(int, int)> delta_x,
    double delta_y,
    int nx,
    int ny,
    int num_subgrids,
    Grid& global_grid,
    CoordSystem coord_system,
    double min_calc_value,
    double center_latitude_rad,
    int exchangeBoundaryIndex,
    int exchangeHatBoundaryIndex) {

    auto bcFunc = GetBoundaryFunc(exchangeBoundaryIndex);
    auto bcHatFunc = GetHatBoundaryFunc(exchangeHatBoundaryIndex);

    // Создать домен
    auto domain = std::make_shared<HybridDomain>();

    // Установить глобальную сетку с корректными delta_x/delta_y
    domain->SetGlobalGrid(Grid(nx, ny, delta_x, delta_y));
    domain->SetDomainSize(nx, ny);

    // Копировать начальные данные из global_grid
    Grid& target_grid = domain->GetGlobalGrid();
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            target_grid.H(i, j) = global_grid.H(i, j);
            target_grid.u(i, j) = global_grid.u(i, j);
            target_grid.v(i, j) = global_grid.v(i, j);
            target_grid.eta(i, j) = global_grid.eta(i, j);
            target_grid.D(i, j) = global_grid.D(i, j);
        }
    }

    // Разбить на полосы (логика из SingleLineGridSimulation::Initialize)
    int m = nx / num_subgrids;  // Ширина полосы

    for (int pid = 0; pid < num_subgrids; ++pid) {
        int global_i_start = pid * m;
        int global_i_end = (pid == num_subgrids - 1) ? nx : (pid + 1) * m;

        // Создать SubGrid
        auto subgrid = CreateSubGrid(
            pid, global_i_start, global_i_end, target_grid,
            g, tau, delta_x, delta_y, min_calc_value, coord_system, center_latitude_rad
        );

        // Set boundary condition callbacks (same as legacy SingleLineGridSimulation)
        subgrid->ExchangeBoundaryData = bcFunc;
        subgrid->ExchangeHatBoundaryData = bcHatFunc;

        // Инициализировать SubGrid
        subgrid->Initialize();

        // Обернуть в StructuredPatch
        auto patch = WrapSubGrid(subgrid, pid, delta_x, delta_y);

        // Mark coupled sides
        if (pid > 0) patch->SetCoupledLeft(true);
        if (pid < num_subgrids - 1) patch->SetCoupledRight(true);

        // Добавить в домен
        domain->AddPatch(patch);
    }

    // Инициализировать домен (построит связи автоматически)
    domain->Initialize();

    return domain;
}

std::shared_ptr<HybridDomain> HybridDomainBuilder::BuildStructuredMeshDomain(
    double Lx, double Ly,
    int struct_nx, int struct_ny,
    int mesh_nx, int mesh_ny,
    double depth, double g, double tau) {

    double half_Lx = Lx * 0.5;
    double dx = half_Lx / struct_nx;
    double dy = Ly / struct_ny;

    auto delta_x_fn = [dx](int, int) { return dx; };

    // --- Create global grid covering left half (for StructuredPatch) ---
    Grid global_grid(struct_nx, struct_ny, delta_x_fn, dy);
    for (int i = 0; i < struct_nx; ++i) {
        for (int j = 0; j < struct_ny; ++j) {
            global_grid.D(i, j) = depth;
            global_grid.H(i, j) = depth;
            global_grid.u(i, j) = 0.0;
            global_grid.v(i, j) = 0.0;
            global_grid.eta(i, j) = 0.0;
        }
    }

    // --- Build domain ---
    auto domain = std::make_shared<HybridDomain>();
    domain->SetGlobalGrid(Grid(struct_nx, struct_ny, delta_x_fn, dy));

    // Copy initial data to domain's global grid
    Grid& target_grid = domain->GetGlobalGrid();
    for (int i = 0; i < struct_nx; ++i)
        for (int j = 0; j < struct_ny; ++j) {
            target_grid.D(i, j) = global_grid.D(i, j);
            target_grid.H(i, j) = global_grid.H(i, j);
            target_grid.u(i, j) = global_grid.u(i, j);
            target_grid.v(i, j) = global_grid.v(i, j);
            target_grid.eta(i, j) = global_grid.eta(i, j);
        }

    // --- Structured patch (left half, patch_id = 0) ---
    auto subgrid = CreateSubGrid(
        0, 0, struct_nx, target_grid,
        g, tau, delta_x_fn, dy, 0.01, CoordSystem::Cartesian
    );
    subgrid->ExchangeBoundaryData =
        [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeBoundaryData1(s, t); };
    subgrid->ExchangeHatBoundaryData =
        [](std::shared_ptr<SubGrid> s, double t) { BoundaryCondition::ExchangeHatBoundaryData1(s, t); };
    subgrid->Initialize();
    auto struct_patch = WrapSubGrid(subgrid, 0, delta_x_fn, dy);
    struct_patch->SetCoupledRight(true); // Right side coupled to MeshPatch
    domain->AddPatch(struct_patch);

    // --- Mesh patch (right half, patch_id = 1) ---
    // Generate triangulation for [0, half_Lx] x [0, Ly], then shift x by half_Lx
    TriMesh mesh = DGValidation::GenerateRectMesh(half_Lx, Ly, mesh_nx, mesh_ny);
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        mesh.GetNode(n).x += half_Lx;
    }
    // Recompute geometry after shifting nodes
    mesh.ComputeGeometry();

    // Update boundary tags: after shifting, left boundary is at x=half_Lx
    // GenerateRectMesh tags: 1=bottom, 2=right, 3=top, 4=left
    // Tag 4 (left, now at x=half_Lx) is the coupling interface

    auto mesh_patch = std::make_shared<MeshPatch>(1, mesh.NumElements());
    mesh_patch->SetMesh(std::move(mesh));
    mesh_patch->Initialize(); // Must init DG before setting bathymetry/IC

    // Set flat-bottom bathymetry and calm initial condition
    mesh_patch->SetBathymetry([depth](double, double) { return depth; });
    mesh_patch->SetInitialCondition([depth](double, double) {
        return SWEState(depth, 0.0, 0.0);
    });

    // Build coupling interface from boundary tag 4 (left boundary = coupling interface)
    mesh_patch->BuildBoundaryInterface(4, "left");

    // Set coupling BC on tag 4, wall on others
    mesh_patch->SetBoundaryCondition(4, DGBoundaryType::Coupled);
    mesh_patch->SetBoundaryCondition(1, DGBoundaryType::Wall);
    mesh_patch->SetBoundaryCondition(2, DGBoundaryType::Open);
    mesh_patch->SetBoundaryCondition(3, DGBoundaryType::Wall);

    domain->AddPatch(mesh_patch);

    // Build couplings (don't call Initialize — it would re-initialize patches
    // and reset the initial conditions we just set)
    domain->BuildCouplings();

    return domain;
}

// ========== ПРИВАТНЫЕ МЕТОДЫ ==========

std::shared_ptr<SubGrid> HybridDomainBuilder::CreateSubGrid(
    int patch_id,
    int global_i_start,
    int global_i_end,
    Grid& global_grid,
    double g,
    double tau,
    std::function<double(int, int)> delta_x,
    double delta_y,
    double min_calc_value,
    CoordSystem coord_system,
    double center_latitude_rad) {

    // Создать фиктивные барьеры (как в SingleLineGridSimulation для последовательного режима)
    auto dummy_barrier = std::make_shared<std::barrier<>>(1);

    // Создать SubGrid (конструктор из SubGrid.h)
    auto subgrid = std::make_shared<SubGrid>(
        patch_id,
        global_i_start,
        global_i_end,
        global_grid,
        g,
        tau,
        delta_x,
        delta_y,
        min_calc_value,
        dummy_barrier,      // barrier_half_step
        dummy_barrier,      // barrier_full_step
        1,                  // spec_step (для сохранения, пока не используется)
        coord_system,
        center_latitude_rad
    );

    return subgrid;
}

std::shared_ptr<StructuredPatch> HybridDomainBuilder::WrapSubGrid(
    std::shared_ptr<SubGrid> subgrid,
    int patch_id,
    std::function<double(int, int)> delta_x,
    double delta_y) {

    // Обернуть SubGrid в StructuredPatch
    auto patch = std::make_shared<StructuredPatch>(
        patch_id,
        subgrid,
        delta_x,
        delta_y
    );

    // Построить интерфейсы для coupling
    patch->Initialize();

    return patch;
}

std::shared_ptr<HybridDomain> HybridDomainBuilder::BuildHybridStripesOneDG(
    double g, double tau,
    std::function<double(int, int)> delta_x, double delta_y,
    int nx, int ny, int num_subgrids,
    Grid& global_grid, CoordSystem coord_system,
    double min_calc_value, int dg_stripe_index,
    int dg_cells_per_structured_cell,
    double center_latitude_rad,
    int exchangeBoundaryIndex,
    int exchangeHatBoundaryIndex) {

    auto bcFunc = GetBoundaryFunc(exchangeBoundaryIndex);
    auto bcHatFunc = GetHatBoundaryFunc(exchangeHatBoundaryIndex);

    auto domain = std::make_shared<HybridDomain>();
    domain->SetGlobalGrid(Grid(nx, ny, delta_x, delta_y));
    domain->SetDomainSize(nx, ny);

    // Copy initial data
    Grid& target_grid = domain->GetGlobalGrid();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            target_grid.H(i, j) = global_grid.H(i, j);
            target_grid.u(i, j) = global_grid.u(i, j);
            target_grid.v(i, j) = global_grid.v(i, j);
            target_grid.eta(i, j) = global_grid.eta(i, j);
            target_grid.D(i, j) = global_grid.D(i, j);
        }

    int m = nx / num_subgrids;

    for (int pid = 0; pid < num_subgrids; ++pid) {
        int gi_start = pid * m;
        int gi_end = (pid == num_subgrids - 1) ? nx : (pid + 1) * m;
        int stripe_nx = gi_end - gi_start;

        if (pid == dg_stripe_index) {
            // ========== DG MESH STRIPE ==========
            // Physical dimensions of this stripe.
            // delta_x может зависеть от j (широты) — для сферических координат.
            // Используем среднюю строку (j=ny/2) для позиционирования DG-меша.
            int j_mid = ny / 2;
            double x0 = 0.0;
            for (int i = 0; i < gi_start; ++i) {
                x0 += delta_x(i, j_mid);
            }
            double stripe_Lx = 0.0;
            for (int i = gi_start; i < gi_end; ++i) {
                stripe_Lx += delta_x(i, j_mid);
            }
            double stripe_Ly = ny * delta_y;

            int mesh_nx = stripe_nx * dg_cells_per_structured_cell;
            int mesh_ny = ny * dg_cells_per_structured_cell;
            // Cap mesh resolution to keep DG tractable.
            // 150x300 balances DG accuracy vs compute cost (~8x cheaper than 300x600).
            // 2026-05-30: caps overridable via env DG_MESH_NX_CAP / DG_MESH_NY_CAP to test
            // whether the lake-at-rest imbalance (158 m on S7) is bathymetry undersampling:
            // each P2 triangle currently averages ~16 structured cells in y, so steep real
            // bathymetry cannot be represented. Higher caps -> finer D projection.
            int nx_cap = 150, ny_cap = 300;
            {
                char* b = nullptr; size_t s = 0;
                _dupenv_s(&b, &s, "DG_MESH_NX_CAP");
                if (b && *b) { int v = std::atoi(b); if (v > 0) nx_cap = v; }
                if (b) free(b);
                b = nullptr; s = 0;
                _dupenv_s(&b, &s, "DG_MESH_NY_CAP");
                if (b && *b) { int v = std::atoi(b); if (v > 0) ny_cap = v; }
                if (b) free(b);
            }
            mesh_nx = std::min(mesh_nx, nx_cap);
            mesh_ny = std::min(mesh_ny, ny_cap);
            std::cerr << "[DG] mesh resolution " << mesh_nx << "x" << mesh_ny
                      << " (caps " << nx_cap << "x" << ny_cap << ")" << std::endl;

            TriMesh mesh = DGValidation::GenerateRectMesh(stripe_Lx, stripe_Ly, mesh_nx, mesh_ny);

            // Shift mesh to correct position
            for (int n = 0; n < mesh.NumNodes(); ++n) {
                mesh.GetNode(n).x += x0;
            }
            mesh.ComputeGeometry();

            auto mesh_patch = std::make_shared<MeshPatch>(pid, mesh.NumElements());
            mesh_patch->SetMesh(std::move(mesh));
            mesh_patch->Initialize();

            // Hybrid DG patches inherit possibly-discontinuous initial data from
            // the structured global grid (e.g. Test1's 10×10 eta=1 square pulse).
            // P2 DG on a triangular mesh with no slope limiter develops
            // dispersive oscillations at such discontinuities, which then
            // amplify through the HLL flux. Enable the TVB-modified Barth-
            // Jespersen limiter (M=0 = strict minmod). Also tighten the
            // troubled-cell detector: the default jump threshold of 10% of
            // |q_avg| never fires for small perturbations over a large
            // background (e.g. eta=1m above D=1000m → 0.1%). With the
            // tighter 1e-4 threshold the limiter still ignores rest state
            // (zero jumps) but actively damps any oscillation, however small,
            // along the wave front.
            auto& dg_solver = mesh_patch->GetDGSolver();
            // Diagnostic toggle (2026-05-20): set DG_DISABLE_SLOPE_LIMITER=1 to
            // turn off the slope limiter. Codex Q3 hypothesis: limiting H,Hu,Hv
            // independently of fixed D breaks well-balancing η=H−D for moving
            // waves. If the post-t=300 runaway disappears with the limiter
            // disabled, that mechanism is confirmed.
            {
                char* dsl = nullptr; size_t dsl_sz = 0;
                _dupenv_s(&dsl, &dsl_sz, "DG_DISABLE_SLOPE_LIMITER");
                bool sl_on = !(dsl && *dsl == '1');
                if (dsl) free(dsl);
                dg_solver.SetSlopeLimiter(sl_on, 0.0);
                if (!sl_on) std::cerr << "[DG] slope limiter DISABLED via DG_DISABLE_SLOPE_LIMITER=1\n";
            }
            // 2026-05-20 evening: env DG_DISABLE_WELL_BALANCED=1 turns off the
            // Audusse hydrostatic reconstruction AND the matching pressure
            // correction (Codex Q2 #2 / #4: HR+P2 incompatibility).
            {
                char* dwb = nullptr; size_t dwb_sz = 0;
                _dupenv_s(&dwb, &dwb_sz, "DG_DISABLE_WELL_BALANCED");
                bool wb_on = !(dwb && *dwb == '1');
                if (dwb) free(dwb);
                dg_solver.Config().well_balanced = wb_on;
                if (!wb_on) std::cerr << "[DG] well_balanced DISABLED via DG_DISABLE_WELL_BALANCED=1\n";
            }
            dg_solver.Config().limiter_jump_threshold = 1e-4;
            dg_solver.Config().limiter_oscillation_threshold = 1e-3;

            // Set bathymetry from global grid (nearest neighbor interpolation)
            // Use j_mid-based dx_inv consistent with DG stripe positioning
            double dx_inv = 1.0 / delta_x(gi_start, j_mid);
            double dy_inv = 1.0 / delta_y;
            // DG: clamp dry cells to min_calc_value to avoid wet/dry instability.
            // MacCormack handles dry cells via special logic (skip computation),
            // but DG solver's positivity limiter struggles with H≈0 next to H≈3000.
            double mcv = min_calc_value;
            auto bathy_func = [&target_grid, dx_inv, dy_inv, nx, ny, mcv](double x, double y) {
                int i = std::min(std::max(static_cast<int>(x * dx_inv), 0), nx - 1);
                int j = std::min(std::max(static_cast<int>(y * dy_inv), 0), ny - 1);
                double D = target_grid.D(i, j);
                return std::max(D, mcv); // Clamp: no dry cells in DG
            };
            mesh_patch->SetBathymetry(bathy_func);

            // Initial condition from global grid, with the SAME clamps as bathy_func
            // so eta = H - D is consistent at rest.
            auto ic_func = [&target_grid, dx_inv, dy_inv, nx, ny, mcv](double x, double y) {
                int i = std::min(std::max(static_cast<int>(x * dx_inv), 0), nx - 1);
                int j = std::min(std::max(static_cast<int>(y * dy_inv), 0), ny - 1);
                double H = target_grid.H(i, j);
                double D = target_grid.D(i, j);
                // Clamp: ensure H >= D_clamped for consistency with clamped bathymetry
                if (D < mcv) { H = mcv; }
                else { H = std::max(H, mcv); }
                double u = target_grid.u(i, j);
                double v = target_grid.v(i, j);
                return SWEState(H, u * H, v * H);
            };
            // Well-balanced IC (default): build H = projected D + projected eta so
            // the DG stripe is lake-at-rest consistent on real bathymetry (kills the
            // ~158 m initial rest-imbalance that seeded the S7 blow-up). Opt out with
            // DG_WB_IC=0 to restore the legacy independent H/D projection.
            bool wb_ic = true;
            {
                char* wbuf = nullptr; size_t wsz = 0;
                _dupenv_s(&wbuf, &wsz, "DG_WB_IC");
                if (wbuf && wbuf[0] == '0') wb_ic = false;
                if (wbuf) free(wbuf);
            }
            if (wb_ic) {
                mesh_patch->SetInitialConditionWellBalanced(ic_func, bathy_func);
            } else {
                std::cerr << "[DG] well-balanced IC DISABLED via DG_WB_IC=0\n";
                mesh_patch->SetInitialCondition(ic_func);
            }

            // Pre-limit the projected IC so any structured-grid discontinuity
            // (e.g. Test1's 10×10 square pulse) doesn't seed dispersive
            // oscillations on the first DG residual.
            // 2026-05-20 evening: env DG_DISABLE_PRELIMIT=1 skips this step so
            // we can isolate whether the slope limiter is breaking
            // initial well-balance (Codex Q3 hypothesis).
            {
                char* dpl = nullptr; size_t dpl_sz = 0;
                _dupenv_s(&dpl, &dpl_sz, "DG_DISABLE_PRELIMIT");
                bool pre_on = !(dpl && *dpl == '1');
                if (dpl) free(dpl);
                if (pre_on) {
                    mesh_patch->GetDGSolver().ApplySlopeLimiter();
                    mesh_patch->GetDGSolver().ApplyPositivityLimiter();
                } else {
                    std::cerr << "[DG] pre-limit DISABLED via DG_DISABLE_PRELIMIT=1\n";
                }
            }

            // Coupling interfaces
            // Tag 4 = left boundary (coupling with stripe pid-1)
            // Tag 2 = right boundary (coupling with stripe pid+1)
            if (pid > 0) {
                mesh_patch->BuildBoundaryInterface(4, "left");
                mesh_patch->SetBoundaryCondition(4, DGBoundaryType::Coupled);
            } else {
                mesh_patch->SetBoundaryCondition(4, DGBoundaryType::Wall);
            }
            if (pid < num_subgrids - 1) {
                mesh_patch->BuildBoundaryInterface(2, "right");
                mesh_patch->SetBoundaryCondition(2, DGBoundaryType::Coupled);
            } else {
                mesh_patch->SetBoundaryCondition(2, DGBoundaryType::Open);
            }
            // Top/bottom: Open (matches zero-gradient BC on structured patches)
            mesh_patch->SetBoundaryCondition(1, DGBoundaryType::Open); // bottom
            mesh_patch->SetBoundaryCondition(3, DGBoundaryType::Open); // top

            domain->AddPatch(mesh_patch);

            std::cerr << "DG stripe " << pid << ": x=[" << x0 << ", " << x0 + stripe_Lx
                      << "], mesh " << mesh_nx << "x" << mesh_ny
                      << " (" << mesh_nx * mesh_ny * 2 << " triangles)" << std::endl;
        } else {
            // ========== STRUCTURED STRIPE ==========
            auto subgrid = CreateSubGrid(
                pid, gi_start, gi_end, target_grid,
                g, tau, delta_x, delta_y, min_calc_value, coord_system, center_latitude_rad
            );
            subgrid->ExchangeBoundaryData = bcFunc;
            subgrid->ExchangeHatBoundaryData = bcHatFunc;
            subgrid->Initialize();
            auto patch = WrapSubGrid(subgrid, pid, delta_x, delta_y);

            // Mark coupled sides: interior stripes have neighbors on both sides.
            // Only the first (pid=0) and last (pid=num_subgrids-1) have external boundaries.
            if (pid > 0) patch->SetCoupledLeft(true);
            if (pid < num_subgrids - 1) patch->SetCoupledRight(true);

            domain->AddPatch(patch);
        }
    }

    // Set SubGrid neighbor pointers for structured↔structured ghost cell exchange.
    // This enables the legacy ExchangeBoundaryData mechanism which is required
    // because ExchangeEngine coupling for S2S has a bug (data not transferred correctly).
    // DG stripe neighbors are set to nullptr — coupling via ExchangeEngine.
    std::vector<std::shared_ptr<SubGrid>> subgrids(num_subgrids);
    for (auto& patch : domain->GetPatches()) {
        auto sp = dynamic_cast<StructuredPatch*>(patch.get());
        if (sp) {
            subgrids[patch->GetPatchId()] = sp->GetSubGrid();
        }
    }
    for (int pid = 0; pid < num_subgrids; ++pid) {
        if (!subgrids[pid]) continue; // DG stripe
        SubGrid* left_nbr = (pid > 0 && subgrids[pid - 1]) ? subgrids[pid - 1].get() : nullptr;
        SubGrid* right_nbr = (pid < num_subgrids - 1 && subgrids[pid + 1]) ? subgrids[pid + 1].get() : nullptr;
        subgrids[pid]->SetNeighbors(left_nbr, right_nbr);
    }

    // Flux-based coupling disabled — unit conversion issue causes divergence.
    // State-based coupling via ExchangeEngine works (wave propagates but with amplification).

    domain->BuildCouplings();
    return domain;
}

std::shared_ptr<HybridDomain> HybridDomainBuilder::BuildNestedDGPatch(
    double g,
    double tau,
    std::function<double(int, int)> delta_x,
    double delta_y,
    int nx,
    int ny,
    Grid& global_grid,
    CoordSystem coord_system,
    double min_calc_value,
    int i_dg0,
    int i_dg1,
    int j_dg0,
    int j_dg1,
    int dg_cells_per_struct,
    int exchangeBoundaryIndex) {

    if (coord_system != CoordSystem::Cartesian) {
        throw std::runtime_error("BuildNestedDGPatch currently supports Cartesian only");
    }
    if (!(0 < i_dg0 && i_dg0 < i_dg1 && i_dg1 < nx &&
          0 < j_dg0 && j_dg0 < j_dg1 && j_dg1 < ny)) {
        throw std::runtime_error("BuildNestedDGPatch requires an interior DG window with half-open bounds");
    }

    auto domain = std::make_shared<HybridDomain>();
    domain->SetGlobalGrid(Grid(nx, ny, delta_x, delta_y));
    domain->SetDomainSize(nx, ny);

    Grid& target_grid = domain->GetGlobalGrid();
    CopyGridData(target_grid, global_grid);

    auto make_nested_patch =
        [&](int patch_id,
            int i0,
            int i1,
            int j0,
            int j1,
            bool coupled_left,
            bool coupled_right,
            bool coupled_bottom,
            bool coupled_top) {
            auto rect = std::make_shared<RectSubGrid>(
                patch_id,
                i0,
                i1,
                j0,
                j1,
                target_grid,
                g,
                tau,
                delta_x,
                delta_y,
                min_calc_value,
                coord_system);
            auto patch = std::make_shared<NestedStructuredPatch>(
                patch_id,
                rect,
                delta_x,
                delta_y,
                exchangeBoundaryIndex);
            patch->SetCoupledLeft(coupled_left);
            patch->SetCoupledRight(coupled_right);
            patch->SetCoupledBottom(coupled_bottom);
            patch->SetCoupledTop(coupled_top);
            patch->Initialize();
            domain->AddPatch(patch);
            return patch;
        };

    make_nested_patch(PatchNW, 0, i_dg0, j_dg1, ny, false, true, true, false);
    make_nested_patch(PatchN, i_dg0, i_dg1, j_dg1, ny, true, true, true, false);
    make_nested_patch(PatchNE, i_dg1, nx, j_dg1, ny, true, false, true, false);
    make_nested_patch(PatchW, 0, i_dg0, j_dg0, j_dg1, false, true, true, true);
    make_nested_patch(PatchE, i_dg1, nx, j_dg0, j_dg1, true, false, true, true);
    make_nested_patch(PatchSW, 0, i_dg0, 0, j_dg0, false, true, false, true);
    make_nested_patch(PatchS, i_dg0, i_dg1, 0, j_dg0, true, true, false, true);
    make_nested_patch(PatchSE, i_dg1, nx, 0, j_dg0, true, false, false, true);

    const int dg_nx_cells = i_dg1 - i_dg0;
    const int dg_ny_cells = j_dg1 - j_dg0;
    const int j_mid = std::min(std::max((j_dg0 + j_dg1) / 2, 0), ny - 1);

    double x_dg0 = 0.0;
    for (int i = 0; i < i_dg0; ++i) {
        x_dg0 += delta_x(i, j_mid);
    }

    double dg_Lx = 0.0;
    for (int i = i_dg0; i < i_dg1; ++i) {
        dg_Lx += delta_x(i, j_mid);
    }
    const double y_dg0 = j_dg0 * delta_y;
    const double dg_Ly = dg_ny_cells * delta_y;

    const int mesh_nx = std::max(1, dg_nx_cells * std::max(dg_cells_per_struct, 1));
    const int mesh_ny = std::max(1, dg_ny_cells * std::max(dg_cells_per_struct, 1));
    TriMesh mesh = DGValidation::GenerateRectMesh(dg_Lx, dg_Ly, mesh_nx, mesh_ny);
    for (int n = 0; n < mesh.NumNodes(); ++n) {
        mesh.GetNode(n).x += x_dg0;
        mesh.GetNode(n).y += y_dg0;
    }
    mesh.ComputeGeometry();

    auto mesh_patch = std::make_shared<MeshPatch>(PatchDG, mesh.NumElements());
    mesh_patch->SetMesh(std::move(mesh));
    mesh_patch->Initialize();

    auto& dg_solver = mesh_patch->GetDGSolver();
    {
        char* dsl = nullptr; size_t dsl_sz = 0;
        _dupenv_s(&dsl, &dsl_sz, "DG_DISABLE_SLOPE_LIMITER");
        bool sl_on = !(dsl && *dsl == '1');
        if (dsl) free(dsl);
        dg_solver.SetSlopeLimiter(sl_on, 0.0);
        if (!sl_on) std::cerr << "[DG nested] slope limiter DISABLED via DG_DISABLE_SLOPE_LIMITER=1\n";
    }
    dg_solver.Config().limiter_jump_threshold = 1e-4;
    dg_solver.Config().limiter_oscillation_threshold = 1e-3;

    const double dx_ref = delta_x(std::min(i_dg0, nx - 1), j_mid);
    const double dx_inv = (dx_ref > 0.0) ? (1.0 / dx_ref) : 0.0;
    const double dy_inv = (delta_y > 0.0) ? (1.0 / delta_y) : 0.0;
    const double mcv = min_calc_value;

    mesh_patch->SetBathymetry([&target_grid, nx, ny, dx_inv, dy_inv, mcv](double x, double y) {
        const int i = (dx_inv > 0.0)
            ? std::min(std::max(static_cast<int>(x * dx_inv), 0), nx - 1)
            : 0;
        const int j = (dy_inv > 0.0)
            ? std::min(std::max(static_cast<int>(y * dy_inv), 0), ny - 1)
            : 0;
        return std::max(target_grid.D(i, j), mcv);
    });

    mesh_patch->SetInitialCondition([&target_grid, nx, ny, dx_inv, dy_inv, mcv](double x, double y) {
        const int i = (dx_inv > 0.0)
            ? std::min(std::max(static_cast<int>(x * dx_inv), 0), nx - 1)
            : 0;
        const int j = (dy_inv > 0.0)
            ? std::min(std::max(static_cast<int>(y * dy_inv), 0), ny - 1)
            : 0;
        double H = target_grid.H(i, j);
        const double D = target_grid.D(i, j);
        H = (D < mcv) ? mcv : std::max(H, mcv);
        return SWEState(H, target_grid.u(i, j) * H, target_grid.v(i, j) * H);
    });

    dg_solver.ApplySlopeLimiter();
    dg_solver.ApplyPositivityLimiter();

    // Keep the build order aligned with the side code used by structured patches.
    mesh_patch->BuildBoundaryInterface(4, "left");
    mesh_patch->SetBoundaryCondition(4, DGBoundaryType::Coupled);
    mesh_patch->BuildBoundaryInterface(2, "right");
    mesh_patch->SetBoundaryCondition(2, DGBoundaryType::Coupled);
    mesh_patch->BuildBoundaryInterface(1, "bottom");
    mesh_patch->SetBoundaryCondition(1, DGBoundaryType::Coupled);
    mesh_patch->BuildBoundaryInterface(3, "top");
    mesh_patch->SetBoundaryCondition(3, DGBoundaryType::Coupled);

    domain->AddPatch(mesh_patch);
    domain->BuildCouplings();
    return domain;
}
