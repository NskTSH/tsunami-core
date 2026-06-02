// HybridDomain.h - Контейнер для всех патчей и системы обмена
// Представляет весь вычислительный домен как коллекцию патчей

#ifndef HYBRID_DOMAIN_H
#define HYBRID_DOMAIN_H

#include "IPatch.h"
#include "ExchangeEngine.h"
#include "CouplingBuilder.h"
#include "Grid.h"
#include <memory>
#include <vector>
#include <utility>

/// Гибридный домен - коллекция патчей с единой системой обмена
///
/// Инкапсулирует:
/// - Набор патчей (structured, mesh, и т.д.)
/// - ExchangeEngine для координации обменов
/// - Глобальную сетку для сбора результатов (опционально)
///
/// Автоматически строит связи между патчами на основе их интерфейсов
class HybridDomain {
public:
    HybridDomain();
    ~HybridDomain() = default;

    // ========== УПРАВЛЕНИЕ ПАТЧАМИ ==========

    /// Добавить патч в домен
    /// @param patch Патч для добавления
    void AddPatch(std::shared_ptr<IPatch> patch);

    /// Получить список всех патчей
    std::vector<std::shared_ptr<IPatch>>& GetPatches() { return patches_; }
    const std::vector<std::shared_ptr<IPatch>>& GetPatches() const { return patches_; }

    /// Получить патч по ID
    /// @param patch_id ID патча
    /// @return Патч или nullptr если не найден
    std::shared_ptr<IPatch> GetPatch(int patch_id);

    /// Получить количество патчей
    size_t GetNumPatches() const { return patches_.size(); }

    // ========== ПОСТРОЕНИЕ СВЯЗЕЙ ==========

    /// Автоматически построить связи между всеми патчами
    /// Анализирует интерфейсы и создает CouplingMap'ы
    void BuildCouplings();

    /// Добавить явную связь между двумя интерфейсами
    /// @param coupling CouplingMap для добавления
    void AddCoupling(std::shared_ptr<CouplingMap> coupling);

    // ========== ИНИЦИАЛИЗАЦИЯ ==========

    /// Инициализировать домен и все патчи
    void Initialize();

    // ========== ОБМЕН ДАННЫМИ ==========

    /// Получить ExchangeEngine
    std::shared_ptr<ExchangeEngine> GetExchangeEngine() { return exchange_engine_; }
    const std::shared_ptr<ExchangeEngine> GetExchangeEngine() const { return exchange_engine_; }

    // ========== ГЛОБАЛЬНАЯ СЕТКА ==========

    /// Установить глобальную сетку для сбора результатов
    /// @param nx Размер по X
    /// @param ny Размер по Y
    void SetGlobalGrid(int nx, int ny);

    /// Установить глобальную сетку (перемещение)
    void SetGlobalGrid(Grid&& grid);

    /// Собрать результаты из всех патчей в глобальную сетку
    void CollectResultsToGlobalGrid();

    /// Получить глобальную сетку
    Grid& GetGlobalGrid() { return *global_grid_; }
    const Grid& GetGlobalGrid() const { return *global_grid_; }

    /// Проверить существует ли глобальная сетка
    bool HasGlobalGrid() const { return global_grid_ != nullptr; }

    // ========== ГЕОМЕТРИЯ ==========

    /// Установить размеры домена (для информации)
    void SetDomainSize(int nx, int ny) { total_nx_ = nx; total_ny_ = ny; }

    /// Получить размеры домена
    void GetDomainSize(int& nx, int& ny) const { nx = total_nx_; ny = total_ny_; }

private:
    // ========== ВНУТРЕННИЕ ДАННЫЕ ==========

    /// Список всех патчей в домене
    std::vector<std::shared_ptr<IPatch>> patches_;

    /// Движок обмена данными
    std::shared_ptr<ExchangeEngine> exchange_engine_;

    /// Глобальная сетка для сбора результатов (опционально)
    std::unique_ptr<Grid> global_grid_;

    /// Размеры домена (для информации)
    int total_nx_;
    int total_ny_;

    // ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

    /// Найти совпадающие интерфейсы между двумя патчами
    /// @param patch_a Первый патч
    /// @param patch_b Второй патч
    /// @return Пары совпадающих интерфейсов
    std::vector<std::pair<PatchInterface, PatchInterface>>
    FindMatchingInterfaces(std::shared_ptr<IPatch> patch_a,
                           std::shared_ptr<IPatch> patch_b);

    /// Проверить совпадают ли два интерфейса геометрически
    /// @param iface_a Первый интерфейс
    /// @param iface_b Второй интерфейс
    /// @return true если интерфейсы совпадают
    bool DoInterfacesMatch(const PatchInterface& iface_a,
                           const PatchInterface& iface_b);
};

#endif // HYBRID_DOMAIN_H
