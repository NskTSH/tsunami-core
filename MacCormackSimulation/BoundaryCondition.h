#pragma once

#include <memory>
#include <string>
#include <vector>

class SubGrid;

// Phase 8 (NTHMP Monai): static wavemaker time-series store. Loaded once at
// scenario build (BuildMonaiValley) via Load(); consulted by WavemakerLeft*
// boundary handlers each timestep via GetEtaAt(physical_time).
namespace WavemakerData {
    // Read 2-column file: header line + (Time_s, Water_surface_m) per line.
    // Replaces any previously loaded series. Throws on file error.
    void Load(const std::string& path, double tau);
    // Linear interpolation; clamps to last sample for t > last sample time.
    // Returns 0 if no series loaded.
    double GetEtaAt(double physical_time_s);
    // True iff Load() was called successfully.
    bool IsLoaded();
}

// Отдельный объект, отвечающий за реализацию граничных условий
struct BoundaryCondition
{
    // Базовые (default) условия на левой/правой границе для базовых и hat-переменных
    static void DefaultLeftConditions(std::shared_ptr<SubGrid> s_grid);
    static void DefaultRightConditions(std::shared_ptr<SubGrid> s_grid);
    static void DefaultLeftHatConditions(std::shared_ptr<SubGrid> s_grid);
    static void DefaultRightHatConditions(std::shared_ptr<SubGrid> s_grid);

    // Условия для вертикальной разбивки (обмен ghost rows по Y)
    static void DefaultBottomConditions(std::shared_ptr<SubGrid> s_grid);
    static void DefaultTopConditions(std::shared_ptr<SubGrid> s_grid);
    static void DefaultBottomHatConditions(std::shared_ptr<SubGrid> s_grid);
    static void DefaultTopHatConditions(std::shared_ptr<SubGrid> s_grid);

    // Радиационные (Sommerfeld 1D) условия на всех четырёх сторонах.
    // Используют inv_dx_arr_[j] (учитывает сферическую зависимость dx от широты)
    // для Left/Right и delta_y_ для Top/Bottom.
    static void RadiationLeftConditions(std::shared_ptr<SubGrid> s);
    static void RadiationRightConditions(std::shared_ptr<SubGrid> s);
    static void RadiationBottomConditions(std::shared_ptr<SubGrid> s);
    static void RadiationTopConditions(std::shared_ptr<SubGrid> s);
    static void RadiationLeftHatConditions(std::shared_ptr<SubGrid> s);
    static void RadiationRightHatConditions(std::shared_ptr<SubGrid> s);
    static void RadiationBottomHatConditions(std::shared_ptr<SubGrid> s);
    static void RadiationTopHatConditions(std::shared_ptr<SubGrid> s);

    // Wavemaker boundary (NTHMP Monai Valley). η at LEFT boundary is forced
    // from a loaded time series (linear interpolation in physical time). H is
    // updated as D + η. u set from shallow-water characteristic (linear:
    // c=sqrt(g·D), see WavemakerLeftConditions) so the forced wave radiates
    // inward. Tangential v copied from interior. Wet/dry threshold aligns with
    // SubGrid min_calc_value (Codex Phase 8).
    //
    // Other 3 sides use Sommerfeld RADIATION (Codex Phase 8: reflective walls
    // caused standing-wave amplification with hydrostatic reconstruction).
    // NOTE: NTHMP Monai spec uses reflective vertical sidewalls; our radiation
    // setup is for active-region equivalence testing, not full NTHMP compliance.
    //
    // Wave series must be loaded once via WavemakerData::Load(path, tau).
    static void WavemakerLeftConditions(std::shared_ptr<SubGrid> s, double t);
    static void WavemakerLeftHatConditions(std::shared_ptr<SubGrid> s, double t);
    static void ExchangeBoundaryDataWavemaker(std::shared_ptr<SubGrid> s, double t);
    static void ExchangeHatBoundaryDataWavemaker(std::shared_ptr<SubGrid> s, double t);

    // Flather 1976 characteristic open boundary condition: u_n = u_ext + sqrt(g/H) * (eta - eta_ext)
    // with eta_ext = 0, u_ext = 0 (open ocean at rest, no external forcing).
    // η ghost = zero-gradient (Codex), tangential = zero-gradient interior, H safety = max(H, grnd).
    // Use for spherical SWE with Coriolis (Coriolis stays in interior momentum).
    static void FlatherLeftConditions(std::shared_ptr<SubGrid> s);
    static void FlatherRightConditions(std::shared_ptr<SubGrid> s);
    static void FlatherBottomConditions(std::shared_ptr<SubGrid> s);
    static void FlatherTopConditions(std::shared_ptr<SubGrid> s);
    static void FlatherLeftHatConditions(std::shared_ptr<SubGrid> s);
    static void FlatherRightHatConditions(std::shared_ptr<SubGrid> s);
    static void FlatherBottomHatConditions(std::shared_ptr<SubGrid> s);
    static void FlatherTopHatConditions(std::shared_ptr<SubGrid> s);

    // Полные сценарии обмена граничных данных (используются SingleLineGridSimulation)
    static void ExchangeBoundaryData1(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeBoundaryData2(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeBoundaryData3(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeBoundaryData4(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeBoundaryData5(std::shared_ptr<SubGrid> s_grid, double t);
    // Sommerfeld radiation на всех 4 границах домена (для Кориолис-тестов,
    // gaussian pulse, lake-at-rest verification). Внутренние ghost столбцы
    // между соседними подсетками всё равно обмениваются через Default*.
    static void ExchangeBoundaryDataRadiation(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeBoundaryDataFlather(std::shared_ptr<SubGrid> s_grid, double t);

    static void ExchangeHatBoundaryData1(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeHatBoundaryData2(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeHatBoundaryData3(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeHatBoundaryData4(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeHatBoundaryData5(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeHatBoundaryDataRadiation(std::shared_ptr<SubGrid> s_grid, double t);
    static void ExchangeHatBoundaryDataFlather(std::shared_ptr<SubGrid> s_grid, double t);
};
