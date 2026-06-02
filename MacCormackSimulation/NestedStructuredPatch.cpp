#include "NestedStructuredPatch.h"
#include <algorithm>

#define IDX(i, j, ny) ((i) * (ny) + (j))

namespace {
    constexpr int kLeftSide = 0;
    constexpr int kRightSide = 1;
    constexpr int kBottomSide = 2;
    constexpr int kTopSide = 3;
}

NestedStructuredPatch::NestedStructuredPatch(int patch_id,
                                             std::shared_ptr<RectSubGrid> subgrid,
                                             std::function<double(int, int)> delta_x,
                                             double delta_y,
                                             int exchange_boundary_index)
    : patch_id_(patch_id),
      subgrid_(subgrid),
      delta_x_(delta_x),
      delta_y_(delta_y),
      exchange_boundary_index_(exchange_boundary_index) {
}

void NestedStructuredPatch::Initialize() {
    if (subgrid_) {
        subgrid_->Initialize();
    }
    BuildInterfaces();
}

void NestedStructuredPatch::ComputeHalfStep() {
    subgrid_->ComputeHalfStep();
}

void NestedStructuredPatch::ComputeFullStep() {
    subgrid_->ComputeFullStep();
}

void NestedStructuredPatch::PrepareExchange(double dt, double t) {
    (void)dt;
    (void)t;
}

void NestedStructuredPatch::ApplyExchange(double dt, double t) {
    (void)dt;
    (void)t;
}

void NestedStructuredPatch::ApplyBoundaryConditions(ExchangePhase phase, double t) {
    (void)t;
    if (!subgrid_) {
        return;
    }

    // TODO: preserve the boundary-index hook for parity with the legacy
    // scenarios once the nested path needs more than zero-gradient outer BC.
    (void)exchange_boundary_index_;

    if (!has_coupled_left_) {
        ApplyZeroGradientSide(phase, "left");
    }
    if (!has_coupled_right_) {
        ApplyZeroGradientSide(phase, "right");
    }
    if (!has_coupled_bottom_) {
        ApplyZeroGradientSide(phase, "bottom");
    }
    if (!has_coupled_top_) {
        ApplyZeroGradientSide(phase, "top");
    }

    if (has_coupled_left_) {
        ReconstructCoupledDepth(phase, "left");
    }
    if (has_coupled_right_) {
        ReconstructCoupledDepth(phase, "right");
    }
    if (has_coupled_bottom_) {
        ReconstructCoupledDepth(phase, "bottom");
    }
    if (has_coupled_top_) {
        ReconstructCoupledDepth(phase, "top");
    }
}

Real* NestedStructuredPatch::GetFieldData(FieldType field) {
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

const Real* NestedStructuredPatch::GetFieldData(FieldType field) const {
    return const_cast<NestedStructuredPatch*>(this)->GetFieldData(field);
}

int NestedStructuredPatch::GetPatchId() const {
    return patch_id_;
}

std::vector<PatchInterface> NestedStructuredPatch::GetInterfaces() const {
    return interfaces_;
}

void NestedStructuredPatch::GetLocalSize(int& nx, int& ny) const {
    nx = subgrid_->local_nx_;
    ny = subgrid_->local_ny_;
}

void NestedStructuredPatch::UpdateGlobalGrid(void* global_grid) {
    (void)global_grid;
    subgrid_->UpdateGlobalGrid(patch_id_);
}

void NestedStructuredPatch::BuildInterfaces() {
    interfaces_.clear();
    BuildLeftInterface();
    BuildRightInterface();
    BuildBottomInterface();
    BuildTopInterface();
}

void NestedStructuredPatch::BuildLeftInterface() {
    PatchInterface iface;
    iface.interface_id = GenerateInterfaceId(patch_id_, kLeftSide);
    iface.patch_id_owner = patch_id_;
    iface.boundary_name = "left";

    const int ny = subgrid_->local_ny_;
    for (int local_j = 1; local_j < ny - 1; ++local_j) {
        const int read_idx = IDX(1, local_j, ny);
        const int write_idx = IDX(0, local_j, ny);
        const int global_j = subgrid_->GetGlobalJStart() + local_j - 1;
        iface.points.emplace_back(
            ComputePhysicalX(subgrid_->GetGlobalIStart(), global_j),
            ComputePhysicalY(global_j),
            read_idx,
            write_idx);
    }

    interfaces_.push_back(iface);
}

void NestedStructuredPatch::BuildRightInterface() {
    PatchInterface iface;
    iface.interface_id = GenerateInterfaceId(patch_id_, kRightSide);
    iface.patch_id_owner = patch_id_;
    iface.boundary_name = "right";

    const int nx = subgrid_->local_nx_;
    const int ny = subgrid_->local_ny_;
    for (int local_j = 1; local_j < ny - 1; ++local_j) {
        const int read_idx = IDX(nx - 2, local_j, ny);
        const int write_idx = IDX(nx - 1, local_j, ny);
        const int global_j = subgrid_->GetGlobalJStart() + local_j - 1;
        iface.points.emplace_back(
            ComputePhysicalX(subgrid_->GetGlobalIEnd(), global_j),
            ComputePhysicalY(global_j),
            read_idx,
            write_idx);
    }

    interfaces_.push_back(iface);
}

void NestedStructuredPatch::BuildBottomInterface() {
    PatchInterface iface;
    iface.interface_id = GenerateInterfaceId(patch_id_, kBottomSide);
    iface.patch_id_owner = patch_id_;
    iface.boundary_name = "bottom";

    const int ny = subgrid_->local_ny_;
    const int global_j = subgrid_->GetGlobalJStart();
    for (int local_i = 1; local_i < subgrid_->local_nx_ - 1; ++local_i) {
        const int global_i = subgrid_->GetGlobalIStart() + local_i - 1;
        const int read_idx = IDX(local_i, 1, ny);
        const int write_idx = IDX(local_i, 0, ny);
        iface.points.emplace_back(
            ComputePhysicalX(global_i, global_j),
            ComputePhysicalY(global_j),
            read_idx,
            write_idx);
    }

    interfaces_.push_back(iface);
}

void NestedStructuredPatch::BuildTopInterface() {
    PatchInterface iface;
    iface.interface_id = GenerateInterfaceId(patch_id_, kTopSide);
    iface.patch_id_owner = patch_id_;
    iface.boundary_name = "top";

    const int ny = subgrid_->local_ny_;
    const int global_j = subgrid_->GetGlobalJEnd();
    for (int local_i = 1; local_i < subgrid_->local_nx_ - 1; ++local_i) {
        const int global_i = subgrid_->GetGlobalIStart() + local_i - 1;
        const int read_idx = IDX(local_i, ny - 2, ny);
        const int write_idx = IDX(local_i, ny - 1, ny);
        iface.points.emplace_back(
            ComputePhysicalX(global_i, global_j - 1),
            ComputePhysicalY(global_j),
            read_idx,
            write_idx);
    }

    interfaces_.push_back(iface);
}

void NestedStructuredPatch::ApplyZeroGradientSide(ExchangePhase phase, const std::string& side) {
    const int nx = subgrid_->local_nx_;
    const int ny = subgrid_->local_ny_;

    if (phase == ExchangePhase::AfterHalfStep) {
        Real* Hh = subgrid_->GetHHat();
        Real* Uh = subgrid_->GetUHat();
        Real* Vh = subgrid_->GetVHat();
        Real* Eh = subgrid_->GetEtaHat();

        if (side == "left") {
            for (int j = 1; j < ny - 1; ++j) {
                Hh[IDX(0, j, ny)] = Hh[IDX(1, j, ny)];
                Uh[IDX(0, j, ny)] = Uh[IDX(1, j, ny)];
                Vh[IDX(0, j, ny)] = Vh[IDX(1, j, ny)];
                Eh[IDX(0, j, ny)] = Eh[IDX(1, j, ny)];
            }
        } else if (side == "right") {
            for (int j = 1; j < ny - 1; ++j) {
                Hh[IDX(nx - 1, j, ny)] = Hh[IDX(nx - 2, j, ny)];
                Uh[IDX(nx - 1, j, ny)] = Uh[IDX(nx - 2, j, ny)];
                Vh[IDX(nx - 1, j, ny)] = Vh[IDX(nx - 2, j, ny)];
                Eh[IDX(nx - 1, j, ny)] = Eh[IDX(nx - 2, j, ny)];
            }
        } else if (side == "bottom") {
            for (int i = 1; i < nx - 1; ++i) {
                Hh[IDX(i, 0, ny)] = Hh[IDX(i, 1, ny)];
                Uh[IDX(i, 0, ny)] = Uh[IDX(i, 1, ny)];
                Vh[IDX(i, 0, ny)] = Vh[IDX(i, 1, ny)];
                Eh[IDX(i, 0, ny)] = Eh[IDX(i, 1, ny)];
            }
        } else if (side == "top") {
            for (int i = 1; i < nx - 1; ++i) {
                Hh[IDX(i, ny - 1, ny)] = Hh[IDX(i, ny - 2, ny)];
                Uh[IDX(i, ny - 1, ny)] = Uh[IDX(i, ny - 2, ny)];
                Vh[IDX(i, ny - 1, ny)] = Vh[IDX(i, ny - 2, ny)];
                Eh[IDX(i, ny - 1, ny)] = Eh[IDX(i, ny - 2, ny)];
            }
        }
        return;
    }

    if (side == "left") {
        for (int j = 1; j < ny - 1; ++j) {
            subgrid_->local_grid_.H(0, j) = subgrid_->local_grid_.H(1, j);
            subgrid_->local_grid_.u(0, j) = subgrid_->local_grid_.u(1, j);
            subgrid_->local_grid_.v(0, j) = subgrid_->local_grid_.v(1, j);
            subgrid_->local_grid_.eta(0, j) = subgrid_->local_grid_.eta(1, j);
        }
    } else if (side == "right") {
        for (int j = 1; j < ny - 1; ++j) {
            subgrid_->local_grid_.H(nx - 1, j) = subgrid_->local_grid_.H(nx - 2, j);
            subgrid_->local_grid_.u(nx - 1, j) = subgrid_->local_grid_.u(nx - 2, j);
            subgrid_->local_grid_.v(nx - 1, j) = subgrid_->local_grid_.v(nx - 2, j);
            subgrid_->local_grid_.eta(nx - 1, j) = subgrid_->local_grid_.eta(nx - 2, j);
        }
    } else if (side == "bottom") {
        for (int i = 1; i < nx - 1; ++i) {
            subgrid_->local_grid_.H(i, 0) = subgrid_->local_grid_.H(i, 1);
            subgrid_->local_grid_.u(i, 0) = subgrid_->local_grid_.u(i, 1);
            subgrid_->local_grid_.v(i, 0) = subgrid_->local_grid_.v(i, 1);
            subgrid_->local_grid_.eta(i, 0) = subgrid_->local_grid_.eta(i, 1);
        }
    } else if (side == "top") {
        for (int i = 1; i < nx - 1; ++i) {
            subgrid_->local_grid_.H(i, ny - 1) = subgrid_->local_grid_.H(i, ny - 2);
            subgrid_->local_grid_.u(i, ny - 1) = subgrid_->local_grid_.u(i, ny - 2);
            subgrid_->local_grid_.v(i, ny - 1) = subgrid_->local_grid_.v(i, ny - 2);
            subgrid_->local_grid_.eta(i, ny - 1) = subgrid_->local_grid_.eta(i, ny - 2);
        }
    }
}

void NestedStructuredPatch::ReconstructCoupledDepth(ExchangePhase phase, const std::string& side) {
    const int nx = subgrid_->local_nx_;
    const int ny = subgrid_->local_ny_;

    if (phase == ExchangePhase::AfterHalfStep) {
        Real* Hh = subgrid_->GetHHat();
        Real* Eh = subgrid_->GetEtaHat();
        Real* D = subgrid_->local_grid_.GetD();

        if (side == "left") {
            for (int j = 1; j < ny - 1; ++j) {
                const int idx = IDX(0, j, ny);
                Hh[idx] = static_cast<Real>(std::max<double>(D[idx], Eh[idx] + D[idx]));
            }
        } else if (side == "right") {
            for (int j = 1; j < ny - 1; ++j) {
                const int idx = IDX(nx - 1, j, ny);
                Hh[idx] = static_cast<Real>(std::max<double>(D[idx], Eh[idx] + D[idx]));
            }
        } else if (side == "bottom") {
            for (int i = 1; i < nx - 1; ++i) {
                const int idx = IDX(i, 0, ny);
                Hh[idx] = static_cast<Real>(std::max<double>(D[idx], Eh[idx] + D[idx]));
            }
        } else if (side == "top") {
            for (int i = 1; i < nx - 1; ++i) {
                const int idx = IDX(i, ny - 1, ny);
                Hh[idx] = static_cast<Real>(std::max<double>(D[idx], Eh[idx] + D[idx]));
            }
        }
        return;
    }

    Real* D = subgrid_->local_grid_.GetD();
    if (side == "left") {
        for (int j = 1; j < ny - 1; ++j) {
            const int idx = IDX(0, j, ny);
            subgrid_->local_grid_.H(0, j) = static_cast<Real>(std::max<double>(D[idx], subgrid_->local_grid_.eta(0, j) + D[idx]));
        }
    } else if (side == "right") {
        for (int j = 1; j < ny - 1; ++j) {
            const int idx = IDX(nx - 1, j, ny);
            subgrid_->local_grid_.H(nx - 1, j) = static_cast<Real>(std::max<double>(D[idx], subgrid_->local_grid_.eta(nx - 1, j) + D[idx]));
        }
    } else if (side == "bottom") {
        for (int i = 1; i < nx - 1; ++i) {
            const int idx = IDX(i, 0, ny);
            subgrid_->local_grid_.H(i, 0) = static_cast<Real>(std::max<double>(D[idx], subgrid_->local_grid_.eta(i, 0) + D[idx]));
        }
    } else if (side == "top") {
        for (int i = 1; i < nx - 1; ++i) {
            const int idx = IDX(i, ny - 1, ny);
            subgrid_->local_grid_.H(i, ny - 1) = static_cast<Real>(std::max<double>(D[idx], subgrid_->local_grid_.eta(i, ny - 1) + D[idx]));
        }
    }
}

int NestedStructuredPatch::GenerateInterfaceId(int patch_id, int side) {
    return static_cast<int>((static_cast<unsigned int>(patch_id) << 16) |
                            static_cast<unsigned int>(side));
}

double NestedStructuredPatch::ComputePhysicalX(int global_i, int global_j) const {
    double x = 0.0;
    for (int i = 0; i < global_i; ++i) {
        x += delta_x_(i, global_j);
    }
    return x;
}

double NestedStructuredPatch::ComputePhysicalY(int global_j) const {
    return global_j * delta_y_;
}
