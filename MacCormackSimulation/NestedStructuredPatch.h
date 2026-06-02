#ifndef NESTED_STRUCTURED_PATCH_H
#define NESTED_STRUCTURED_PATCH_H

#include "IPatch.h"
#include "RectSubGrid.h"
#include <functional>
#include <memory>

class NestedStructuredPatch : public IPatch {
public:
    NestedStructuredPatch(int patch_id,
                          std::shared_ptr<RectSubGrid> subgrid,
                          std::function<double(int, int)> delta_x,
                          double delta_y,
                          int exchange_boundary_index = 0);

    void Initialize() override;
    void ComputeHalfStep() override;
    void ComputeFullStep() override;
    void PrepareExchange(double dt, double t) override;
    void ApplyExchange(double dt, double t) override;
    void ApplyBoundaryConditions(ExchangePhase phase, double t) override;
    Real* GetFieldData(FieldType field) override;
    const Real* GetFieldData(FieldType field) const override;
    int GetPatchId() const override;
    std::vector<PatchInterface> GetInterfaces() const override;
    void GetLocalSize(int& nx, int& ny) const override;
    void UpdateGlobalGrid(void* global_grid) override;

    std::shared_ptr<RectSubGrid> GetRectSubGrid() { return subgrid_; }
    const std::shared_ptr<RectSubGrid> GetRectSubGrid() const { return subgrid_; }

    void BuildInterfaces();

    void SetCoupledLeft(bool v) { has_coupled_left_ = v; }
    void SetCoupledRight(bool v) { has_coupled_right_ = v; }
    void SetCoupledTop(bool v) { has_coupled_top_ = v; }
    void SetCoupledBottom(bool v) { has_coupled_bottom_ = v; }

private:
    int patch_id_;
    std::shared_ptr<RectSubGrid> subgrid_;
    std::vector<PatchInterface> interfaces_;
    std::function<double(int, int)> delta_x_;
    double delta_y_;
    int exchange_boundary_index_ = 0;

    bool has_coupled_left_ = false;
    bool has_coupled_right_ = false;
    bool has_coupled_top_ = false;
    bool has_coupled_bottom_ = false;

    double ComputePhysicalX(int global_i, int global_j) const;
    double ComputePhysicalY(int global_j) const;

    void BuildLeftInterface();
    void BuildRightInterface();
    void BuildBottomInterface();
    void BuildTopInterface();

    void ApplyZeroGradientSide(ExchangePhase phase, const std::string& side);
    void ReconstructCoupledDepth(ExchangePhase phase, const std::string& side);

    static int GenerateInterfaceId(int patch_id, int side);
};

#endif // NESTED_STRUCTURED_PATCH_H
