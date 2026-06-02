// StructuredPatch.cpp - Реализация обертки для структурированной подсетки

#include "StructuredPatch.h"
#include "Grid.h"

// Макрос индексации (как в SubGrid)
#define IDX(i, j, ny) ((i) * (ny) + (j))

StructuredPatch::StructuredPatch(int patch_id,
                                 std::shared_ptr<SubGrid> subgrid,
                                 std::function<double(int, int)> delta_x,
                                 double delta_y)
    : patch_id_(patch_id),
      subgrid_(subgrid),
      delta_x_(delta_x),
      delta_y_(delta_y) {
}

void StructuredPatch::Initialize() {
    // SubGrid уже инициализирован снаружи (в SingleLineGridSimulation)
    // Но мы можем построить интерфейсы здесь
    BuildInterfaces();
}

void StructuredPatch::ComputeHalfStep() {
    // Делегация на SubGrid без изменений!
    subgrid_->ComputeHalfStep();
}

void StructuredPatch::ComputeFullStep() {
    // Делегация на SubGrid без изменений!
    subgrid_->ComputeFullStep();
}

void StructuredPatch::PrepareExchange(double dt, double t) {
    // На данном этапе ничего не делаем
    // ExchangeEngine будет напрямую читать данные через GetFieldData()
    // В будущем здесь можно копировать в специальные буферы обмена
}

void StructuredPatch::ApplyExchange(double dt, double t) {
    // ExchangeEngine writes data via GetFieldData() + CopyBufferToPatch.
    // Physical boundary conditions are applied separately via ApplyBoundaryConditions().
}

void StructuredPatch::ApplyBoundaryConditions(ExchangePhase phase, double t) {
    auto sg = subgrid_;
    if (!sg) return;

    // Structured↔Structured ghost exchange uses legacy neighbor pointers.
    // Structured↔DG exchange uses ExchangeEngine (coupling).
    // Legacy callbacks with NULL neighbor overwrite ghost cells with zero-gradient,
    // destroying ExchangeEngine data. Skip legacy on coupled sides with no neighbor.
    //
    // Strategy: run legacy callback (handles Y-BC + neighbor-based X ghost exchange),
    // then if a coupled side has no neighbor pointer (DG neighbor), the legacy
    // callback already applied zero-gradient which we must NOT override.
    // But ExchangeEngine already wrote the correct data BEFORE ApplyBoundaryConditions!
    // Legacy callback destroys it. So: skip the entire callback if there's a coupled
    // side without a neighbor, and apply Y-BC + non-null-neighbor X-BC manually.

    bool left_has_neighbor = (sg->neighbor_left_ != nullptr);
    bool right_has_neighbor = (sg->neighbor_right_ != nullptr);

    // If both neighbors exist (or both sides are external), legacy callback is safe
    if ((!has_coupled_left_ || left_has_neighbor) && (!has_coupled_right_ || right_has_neighbor)) {
        if (phase == ExchangePhase::AfterHalfStep) {
            if (sg->ExchangeHatBoundaryData) sg->ExchangeHatBoundaryData(sg, t);
        } else {
            if (sg->ExchangeBoundaryData) sg->ExchangeBoundaryData(sg, t);
        }
        return;
    }

    // Has a coupled side with DG neighbor (no pointer) — apply BC manually
    int ny = sg->local_ny_;
    int nx = sg->local_nx_;

    if (phase == ExchangePhase::AfterHalfStep) {
        Real* Hh = sg->GetHHat(); Real* Uh = sg->GetUHat();
        Real* Vh = sg->GetVHat(); Real* Eh = sg->GetEtaHat();
        // Y-BC
        for (int i = 0; i < nx; ++i) {
            int ij0 = i*ny, ij1 = ij0+1, ijn2 = ij0+ny-2, ijn1 = ij0+ny-1;
            Hh[ij0]=Hh[ij1]; Hh[ijn1]=Hh[ijn2]; Uh[ij0]=Uh[ij1]; Uh[ijn1]=Uh[ijn2];
            Vh[ij0]=Vh[ij1]; Vh[ijn1]=Vh[ijn2]; Eh[ij0]=Eh[ij1]; Eh[ijn1]=Eh[ijn2];
        }
        // X-BC: only for sides with neighbor pointers (legacy exchange) or external
        if (left_has_neighbor && sg->neighbor_left_) {
            auto* nb = sg->neighbor_left_;
            for (int j=0; j<ny; ++j) {
                Hh[j]=nb->GetHHat()[(nb->local_nx_-2)*ny+j]; Uh[j]=nb->GetUHat()[(nb->local_nx_-2)*ny+j];
                Vh[j]=nb->GetVHat()[(nb->local_nx_-2)*ny+j]; Eh[j]=nb->GetEtaHat()[(nb->local_nx_-2)*ny+j];
            }
        } else if (!has_coupled_left_) {
            for (int j=0; j<ny; ++j) { Hh[j]=Hh[ny+j]; Uh[j]=Uh[ny+j]; Vh[j]=Vh[ny+j]; Eh[j]=Eh[ny+j]; }
        }
        // Right
        if (right_has_neighbor && sg->neighbor_right_) {
            auto* nb = sg->neighbor_right_;
            for (int j=0; j<ny; ++j) {
                int d=(nx-1)*ny+j;
                Hh[d]=nb->GetHHat()[ny+j]; Uh[d]=nb->GetUHat()[ny+j];
                Vh[d]=nb->GetVHat()[ny+j]; Eh[d]=nb->GetEtaHat()[ny+j];
            }
        } else if (!has_coupled_right_) {
            for (int j=0; j<ny; ++j) { int d=(nx-1)*ny+j, s=(nx-2)*ny+j; Hh[d]=Hh[s]; Uh[d]=Uh[s]; Vh[d]=Vh[s]; Eh[d]=Eh[s]; }
        }
        // Reconstruct H_hat from eta_hat on coupled-DG sides (bathymetry mismatch fix)
        Real* D = sg->local_grid_.GetD();
        if (has_coupled_left_ && !left_has_neighbor) {
            for (int j = 0; j < ny; ++j) {
                double eta_h = Eh[j];
                double D_local = D[j]; // D at i=0
                Hh[j] = std::max(0.0, eta_h + D_local);
            }
        }
        if (has_coupled_right_ && !right_has_neighbor) {
            for (int j = 0; j < ny; ++j) {
                int idx = (nx - 1) * ny + j;
                double eta_h = Eh[idx];
                double D_local = D[idx];
                Hh[idx] = std::max(0.0, eta_h + D_local);
            }
        }
    } else {
        auto& grid = sg->local_grid_;
        // Y-BC
        for (int i=0; i<nx; ++i) {
            grid.H(i,0)=grid.H(i,1); grid.H(i,ny-1)=grid.H(i,ny-2);
            grid.u(i,0)=grid.u(i,1); grid.u(i,ny-1)=grid.u(i,ny-2);
            grid.v(i,0)=grid.v(i,1); grid.v(i,ny-1)=grid.v(i,ny-2);
            grid.eta(i,0)=grid.eta(i,1); grid.eta(i,ny-1)=grid.eta(i,ny-2);
        }
        // Left neighbor exchange
        if (left_has_neighbor && sg->neighbor_left_) {
            auto& nb = sg->neighbor_left_->local_grid_;
            for (int j=0; j<ny; ++j) {
                grid.H(0,j)=nb.H(sg->neighbor_left_->local_nx_-2,j);
                grid.u(0,j)=nb.u(sg->neighbor_left_->local_nx_-2,j);
                grid.v(0,j)=nb.v(sg->neighbor_left_->local_nx_-2,j);
                grid.eta(0,j)=nb.eta(sg->neighbor_left_->local_nx_-2,j);
            }
        } else if (!has_coupled_left_) {
            for (int j=0; j<ny; ++j) { grid.H(0,j)=grid.H(1,j); grid.u(0,j)=grid.u(1,j); grid.v(0,j)=grid.v(1,j); grid.eta(0,j)=grid.eta(1,j); }
        }
        // Right neighbor exchange
        if (right_has_neighbor && sg->neighbor_right_) {
            auto& nb = sg->neighbor_right_->local_grid_;
            for (int j=0; j<ny; ++j) {
                grid.H(nx-1,j)=nb.H(1,j); grid.u(nx-1,j)=nb.u(1,j);
                grid.v(nx-1,j)=nb.v(1,j); grid.eta(nx-1,j)=nb.eta(1,j);
            }
        } else if (!has_coupled_right_) {
            for (int j=0; j<ny; ++j) { grid.H(nx-1,j)=grid.H(nx-2,j); grid.u(nx-1,j)=grid.u(nx-2,j); grid.v(nx-1,j)=grid.v(nx-2,j); grid.eta(nx-1,j)=grid.eta(nx-2,j); }
        }
        // Coupled side with DG: ExchangeEngine wrote H from DG. But DG bathymetry
        // differs from local (coarse DG mesh vs fine structured). Reconstruct H from
        // exchanged eta using local D: H = max(0, eta + D). This preserves the
        // free surface (eta) — the physically continuous quantity — including
        // depression waves (eta < 0); clamping to D would clip troughs at the seam.
        if (has_coupled_left_ && !left_has_neighbor) {
            Real* D = sg->local_grid_.GetD();
            for (int j = 0; j < ny; ++j) {
                double eta = grid.eta(0, j);
                double D_local = D[0 * ny + j];
                grid.H(0, j) = std::max(0.0, eta + D_local);
            }
        }
        if (has_coupled_right_ && !right_has_neighbor) {
            Real* D = sg->local_grid_.GetD();
            for (int j = 0; j < ny; ++j) {
                double eta = grid.eta(nx - 1, j);
                double D_local = D[(nx - 1) * ny + j];
                grid.H(nx - 1, j) = std::max(0.0, eta + D_local);
            }
        }
    }
}

Real* StructuredPatch::GetFieldData(FieldType field) {
    switch (field) {
    case FieldType::H:
        return subgrid_->local_grid_.GetH();

    case FieldType::U:
        return subgrid_->local_grid_.GetU();

    case FieldType::V:
        return subgrid_->local_grid_.GetV();

    case FieldType::Eta:
        return subgrid_->local_grid_.GetEta();

    case FieldType::D:
        return subgrid_->local_grid_.GetD();

    case FieldType::H_hat:
        return subgrid_->GetHHat();

    case FieldType::U_hat:
        return subgrid_->GetUHat();

    case FieldType::V_hat:
        return subgrid_->GetVHat();

    case FieldType::Eta_hat:
        return subgrid_->GetEtaHat();

    default:
        return nullptr;
    }
}

const Real* StructuredPatch::GetFieldData(FieldType field) const {
    // Используем const_cast для переиспользования кода (безопасно для чтения)
    return const_cast<StructuredPatch*>(this)->GetFieldData(field);
}

int StructuredPatch::GetPatchId() const {
    return patch_id_;
}

std::vector<PatchInterface> StructuredPatch::GetInterfaces() const {
    return interfaces_;
}

void StructuredPatch::GetLocalSize(int& nx, int& ny) const {
    nx = subgrid_->local_nx_;
    ny = subgrid_->local_ny_;
}

void StructuredPatch::UpdateGlobalGrid(void* global_grid) {
    // Делегация на существующий метод SubGrid
    Grid* grid_ptr = static_cast<Grid*>(global_grid);
    if (grid_ptr) {
        subgrid_->UpdateGlobalGrid(patch_id_);
    }
}

// ========== Построение интерфейсов ==========

void StructuredPatch::BuildInterfaces() {
    interfaces_.clear();

    // Построить левый интерфейс (граница i=1, если есть сосед слева или это внешняя граница)
    BuildLeftInterface();

    // Построить правый интерфейс (граница i=nx-2, если есть сосед справа или это внешняя граница)
    BuildRightInterface();
}

void StructuredPatch::BuildLeftInterface() {
    PatchInterface left_iface;
    left_iface.interface_id = GenerateInterfaceId(patch_id_, 0);
    left_iface.patch_id_owner = patch_id_;
    left_iface.boundary_name = "left";

    // Добавить все точки вдоль левой границы
    for (int j = 0; j < subgrid_->local_ny_; ++j) {
        // Чтение: i=1 (первый вычислительный слой)
        int read_idx = IDX(1, j, subgrid_->local_ny_);
        // Запись: i=0 (ghost cell слева) — сюда пишутся данные от соседа
        int write_idx = IDX(0, j, subgrid_->local_ny_);

        int global_i = subgrid_->global_i_start_;
        double x = ComputePhysicalX(global_i, j);
        double y = ComputePhysicalY(j);

        left_iface.points.emplace_back(x, y, read_idx, write_idx);
    }

    interfaces_.push_back(left_iface);
}

void StructuredPatch::BuildRightInterface() {
    PatchInterface right_iface;
    right_iface.interface_id = GenerateInterfaceId(patch_id_, 1);
    right_iface.patch_id_owner = patch_id_;
    right_iface.boundary_name = "right";

    int nx = subgrid_->local_nx_;

    // Добавить все точки вдоль правой границы
    for (int j = 0; j < subgrid_->local_ny_; ++j) {
        // Чтение: i=nx-2 (последний вычислительный слой)
        int read_idx = IDX(nx - 2, j, subgrid_->local_ny_);
        // Запись: i=nx-1 (ghost cell справа) — сюда пишутся данные от соседа
        int write_idx = IDX(nx - 1, j, subgrid_->local_ny_);

        // Координата: граница домена (для matching с соседним патчем)
        int global_i = subgrid_->global_i_end_;
        double x = ComputePhysicalX(global_i, j);
        double y = ComputePhysicalY(j);

        right_iface.points.emplace_back(x, y, read_idx, write_idx);
    }

    interfaces_.push_back(right_iface);
}

// ========== Вспомогательные методы ==========

int StructuredPatch::GenerateInterfaceId(int patch_id, int side) {
    // Unique ID: high bits = patch_id, low bits = side
    // Supports up to 65535 patches with 65536 sides each
    // Use unsigned shift to avoid UB for patch_id >= 32768
    return static_cast<int>(
        (static_cast<unsigned int>(patch_id) << 16) |
         static_cast<unsigned int>(side));
}

double StructuredPatch::ComputePhysicalX(int global_i, int j) const {
    // Вычислить физическую X-координату
    // Накопительная сумма шагов от 0 до global_i
    double x = 0.0;
    for (int i = 0; i < global_i; ++i) {
        x += delta_x_(i, j);
    }
    return x;
}

double StructuredPatch::ComputePhysicalY(int j) const {
    // Вычислить физическую Y-координату
    return j * delta_y_;
}
