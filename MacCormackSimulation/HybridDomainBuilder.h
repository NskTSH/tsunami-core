// HybridDomainBuilder.h - Построение гибридного домена из конфигураций
// Обеспечивает обратную совместимость с существующей системой

#ifndef HYBRID_DOMAIN_BUILDER_H
#define HYBRID_DOMAIN_BUILDER_H

#include "HybridDomain.h"
#include "StructuredPatch.h"
#include "NestedStructuredPatch.h"
#include "MeshPatch.h"
#include "DefaultParametersParser.h"
#include "Grid.h"
#include <memory>
#include <functional>

/// Построитель гибридных доменов
///
/// Создает HybridDomain из конфигурационных файлов
/// Поддерживает два режима:
/// 1. Legacy режим - разбиение на полосы как в SingleLineGridSimulation
/// 2. Новый режим - создание патчей по явному описанию
class HybridDomainBuilder {
public:
    /// Построить домен из DefaultParametersParser
    ///
    /// @param params Параметры симуляции
    /// @param global_grid Глобальная сетка (будет скопирована)
    /// @return Готовый HybridDomain
    static std::shared_ptr<HybridDomain> BuildFromParameters(
        DefaultParametersParser& params,
        Grid& global_grid
    );

    /// Построить домен из параметров (прямая передача)
    ///
    /// @param g Ускорение свободного падения
    /// @param tau Временной шаг
    /// @param delta_x Функция шага по X
    /// @param delta_y Шаг по Y
    /// @param nx Размер сетки по X
    /// @param ny Размер сетки по Y
    /// @param num_subgrids Количество подсеток/полос
    /// @param global_grid Глобальная сетка
    /// @param coord_system Система координат
    /// @param min_calc_value Минимальная глубина для расчета
    /// @return Готовый HybridDomain
    static std::shared_ptr<HybridDomain> BuildStructuredStripes(
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
        double center_latitude_rad = 0.785398163,
        int exchangeBoundaryIndex = 0,
        int exchangeHatBoundaryIndex = 0
    );

    /// Build a mixed domain: structured left half + DG mesh right half
    ///
    /// Domain: [0, Lx] x [0, Ly], flat bottom D = depth
    /// Left half: StructuredPatch with struct_nx x struct_ny cells
    /// Right half: MeshPatch with mesh_nx x mesh_ny triangulation
    /// Coupling interface at x = Lx/2
    ///
    /// @param Lx Domain length in x
    /// @param Ly Domain length in y
    /// @param struct_nx Structured grid columns (left half)
    /// @param struct_ny Structured grid rows
    /// @param mesh_nx Mesh cells in x (right half)
    /// @param mesh_ny Mesh cells in y
    /// @param depth Background depth D
    /// @param g Gravity
    /// @param tau Time step
    /// @return HybridDomain with coupling built
    static std::shared_ptr<HybridDomain> BuildStructuredMeshDomain(
        double Lx, double Ly,
        int struct_nx, int struct_ny,
        int mesh_nx, int mesh_ny,
        double depth, double g, double tau
    );

    /// Build domain with N-1 structured stripes and 1 DG mesh stripe.
    /// The DG stripe covers the same physical region as the structured stripe
    /// it replaces, with triangulated mesh and coupling on both sides.
    ///
    /// @param dg_stripe_index Which stripe (0-based) to replace with DG
    /// @param dg_cells_per_structured_cell Mesh refinement factor (1 = same resolution)
    /// Other params same as BuildStructuredStripes
    static std::shared_ptr<HybridDomain> BuildHybridStripesOneDG(
        double g, double tau,
        std::function<double(int, int)> delta_x, double delta_y,
        int nx, int ny, int num_subgrids,
        Grid& global_grid, CoordSystem coord_system,
        double min_calc_value, int dg_stripe_index,
        int dg_cells_per_structured_cell = 1,
        double center_latitude_rad = 0.785398163,
        int exchangeBoundaryIndex = 0,
        int exchangeHatBoundaryIndex = 0
    );

    static std::shared_ptr<HybridDomain> BuildNestedDGPatch(
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
        int dg_cells_per_struct = 1,
        int exchangeBoundaryIndex = 0
    );

private:
    /// Создать SubGrid (переиспользование логики из SingleLineGridSimulation)
    static std::shared_ptr<SubGrid> CreateSubGrid(
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
        double center_latitude_rad = 0.785398163
    );

    /// Обернуть SubGrid в StructuredPatch
    static std::shared_ptr<StructuredPatch> WrapSubGrid(
        std::shared_ptr<SubGrid> subgrid,
        int patch_id,
        std::function<double(int, int)> delta_x,
        double delta_y
    );
};

#endif // HYBRID_DOMAIN_BUILDER_H
