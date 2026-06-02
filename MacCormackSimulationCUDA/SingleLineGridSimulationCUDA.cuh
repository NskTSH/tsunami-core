// SingleLineGridSimulationCUDA.cuh
// CUDA-версия симулятора с интерфейсом, совместимым с оригинальной CPU версией
// Позволяет легко заменить CPU симулятор на GPU версию

#ifndef SINGLE_LINE_GRID_SIMULATION_CUDA_CUH
#define SINGLE_LINE_GRID_SIMULATION_CUDA_CUH

#include <memory>
#include <string>
#include <functional>
#include <vector>

// Forward declarations
struct SimulationParams;

// Простая структура Grid для совместимости с оригинальным проектом
class GridCUDA {
public:
    GridCUDA(int nx = 0, int ny = 0);
    ~GridCUDA();
    
    // Move семантика
    GridCUDA(GridCUDA&& other) noexcept;
    GridCUDA& operator=(GridCUDA&& other) noexcept;
    
    // Copy семантика
    GridCUDA(const GridCUDA& other);
    GridCUDA& operator=(const GridCUDA& other);
    
    // Доступ к элементам
    double& H(int i, int j);
    double& u(int i, int j);
    double& v(int i, int j);
    double& eta(int i, int j);
    double& D(int i, int j);
    double& eta_max(int i, int j);
    double& u_max(int i, int j);
    double& v_max(int i, int j);

    const double& H(int i, int j) const;
    const double& u(int i, int j) const;
    const double& v(int i, int j) const;
    const double& eta(int i, int j) const;
    const double& D(int i, int j) const;
    const double& eta_max(int i, int j) const;
    const double& u_max(int i, int j) const;
    const double& v_max(int i, int j) const;

    // Прямой доступ к массивам
    double* GetH() { return H_.get(); }
    double* GetU() { return u_.get(); }
    double* GetV() { return v_.get(); }
    double* GetEta() { return eta_.get(); }
    double* GetD() { return D_.get(); }
    double* GetEtaMax() { return eta_max_.get(); }
    double* GetUMax() { return u_max_.get(); }
    double* GetVMax() { return v_max_.get(); }

    const double* GetH() const { return H_.get(); }
    const double* GetU() const { return u_.get(); }
    const double* GetV() const { return v_.get(); }
    const double* GetEta() const { return eta_.get(); }
    const double* GetD() const { return D_.get(); }
    const double* GetEtaMax() const { return eta_max_.get(); }
    const double* GetUMax() const { return u_max_.get(); }
    const double* GetVMax() const { return v_max_.get(); }
    
    int GetNx() const { return nx_; }
    int GetNy() const { return ny_; }
    
    bool IsInitialized() const { return is_initialized_; }
    void SetInitialized(bool val) { is_initialized_ = val; }
    
    int subgrid_count = 1;
    
private:
    int nx_, ny_;
    std::unique_ptr<double[]> H_;
    std::unique_ptr<double[]> u_;
    std::unique_ptr<double[]> v_;
    std::unique_ptr<double[]> eta_;
    std::unique_ptr<double[]> D_;
    std::unique_ptr<double[]> eta_max_;
    std::unique_ptr<double[]> u_max_;
    std::unique_ptr<double[]> v_max_;
    bool is_initialized_ = false;
};

// Интерфейс симулятора (совместимый с оригиналом)
class IBaseSimulatorCUDA {
public:
    virtual ~IBaseSimulatorCUDA() = default;
    virtual void Initialize() = 0;
    virtual void CollectResults(int timestep) = 0;
    virtual void SetSaveInterval(int interval) = 0;
    virtual int GetTimesteps() const = 0;
    virtual void SaveDataToFile(const std::string& variable_name, const GridCUDA& grid, int timestep) = 0;
};

// Основной класс CUDA симулятора с совместимым интерфейсом
class SingleLineGridSimulationCUDA : public IBaseSimulatorCUDA {
public:
    // Конструктор по умолчанию
    SingleLineGridSimulationCUDA();
    
    // Деструктор
    ~SingleLineGridSimulationCUDA();
    
    // Запрет копирования
    SingleLineGridSimulationCUDA(const SingleLineGridSimulationCUDA&) = delete;
    SingleLineGridSimulationCUDA& operator=(const SingleLineGridSimulationCUDA&) = delete;
    
    // ================== Установка параметров ==================
    
    // Базовая установка параметров (совместимость с оригиналом с функцией delta_x)
    void SetParameters(double g, double tau, 
                       std::function<double(int, int)> delta_x, double delta_y,
                       int nx, int ny, int timesteps);
    
    // Установка параметров с опциями
    void SetParameters(double g, double tau, double delta_x_const, double delta_y,
                       int nx, int ny, int timesteps, double min_depth = 1.0);
    
    // Полная сигнатура с num_subgrids для экономии видеопамяти
    // num_subgrids - количество частей, на которые делится сетка по X
    // При num_subgrids > 1 сетка обрабатывается по частям для экономии GPU памяти
    void SetParameters(double g, double tau, double delta_x_const, double delta_y,
                       int nx, int ny, int timesteps, 
                       int num_subgrids,
                       int exchangeBoundaryIndex, int exchangeHatBoundaryIndex);
    
    // Установка начальной сетки
    void SetGrid(GridCUDA&& grid);
    void SetGrid(const GridCUDA& grid);
    
    // ================== Интерфейс IBaseSimulatorCUDA ==================
    void Initialize() override;
    void CollectResults(int timestep) override;
    void SetSaveInterval(int interval) override;
    int GetTimesteps() const override;
    void SaveDataToFile(const std::string& variable_name, const GridCUDA& grid, int timestep) override;

    void SetResultFolder(const std::string& folder) { output_folder_ = folder; }
    const std::string& GetResultFolder() const { return output_folder_; }

    // Register a variable delta_x function (e.g. latitude-dependent dx for Test 6).
    // Auto-detection in Initialize() will sample this and load the device array if it varies.
    void SetDeltaXFunction(std::function<double(int, int)> fn) { delta_x_func_ = std::move(fn); }

    // Override total timesteps after SetParameters (used by CLI --timesteps=N).
    void OverrideTimesteps(int t) { timesteps_ = t; }

    // Override min_calc_value (depth threshold for wet/dry).
    // CPU reads this from .par. ScenarioBuilderCUDA passes it via scenario.min_depth.
    void SetMinCalcValue(double v) { min_calc_value_ = v; }
    double GetMinCalcValue() const { return min_calc_value_; }

    // Save per-interval u/v snapshots and final u_max/v_max maps (Pacific 900-903).
    void SetSaveVelocities(bool v) { save_velocities_ = v; }
    bool GetSaveVelocities() const { return save_velocities_; }

    // Production-benchmark mode: skip ALL intermediate/final saves except the final
    // eta_max map. Run() reports compute-only vs compute+save time separately.
    void SetEtaMaxOnly(bool v) { save_eta_max_only_ = v; }
    bool GetEtaMaxOnly() const { return save_eta_max_only_; }
    
    // ================== Запуск симуляции ==================
    void Run();
    
    // Один шаг симуляции
    void Step();
    
    // ================== Доступ к данным ==================
    GridCUDA& GetGlobalGrid() { return global_grid_; }
    const GridCUDA& GetGlobalGrid() const { return global_grid_; }
    
    // Синхронизация GPU -> CPU
    void SyncToHost();
    
    // ================== Тестовые сценарии (идентичны CPU версии) ==================
    void SetupTest1();           // Квадратное возмущение 10x10 в координатах (45-55, 45-55)
    void SetupTest2();           // Косинусный импульс в центре (500000, 500000)
    void SetupTest3(double tau); // Волна с переменной глубиной
    void SetupTest4(double tau); // Треугольный остров в правом нижнем углу
    void SetupTest5(double tau); // Круглый остров в центре
    
private:
    // Параметры симуляции
    double g_;
    double tau_;
    double delta_x_const_;
    double delta_y_;
    int nx_;
    int ny_;
    int timesteps_;
    double min_calc_value_;
    
    // Количество подсеток для экономии видеопамяти
    int num_subgrids_;
    
    // Тип граничных условий (0 = статические, 2 = test3, 3 = test4/5)
    // Раздельные индексы для base и hat (CPU использует разные функции — Hat3 vs Data3).
    int boundary_condition_type_;
    int hat_boundary_condition_type_;

    // Использовать FP32 (single precision) на GPU. По умолчанию false (FP64).
    bool use_fp32_ = false;
    // Координатная система (Cartesian | Spherical). Spherical активирует Coriolis kernels.
    int coord_system_id_ = 0;  // 0=Cartesian, 1=Spherical (CudaCoordSystem enum value)
    bool coriolis_enabled_ = false;
    double center_latitude_rad_ = 0.785398163;  // 45° default
public:
    void SetUseFloat32(bool v) { use_fp32_ = v; }
    bool GetUseFloat32() const { return use_fp32_; }

    // 0 = Cartesian, 1 = Spherical
    void SetCoordSystemId(int v) { coord_system_id_ = v; }
    int GetCoordSystemId() const { return coord_system_id_; }

    void SetCoriolisEnabled(bool v) { coriolis_enabled_ = v; }
    bool GetCoriolisEnabled() const { return coriolis_enabled_; }

    void SetCenterLatitudeRad(double rad) { center_latitude_rad_ = rad; }
    double GetCenterLatitudeRad() const { return center_latitude_rad_; }
private:
    
    // Текущий шаг (для динамических граничных условий)
    int current_timestep_;
    
    // Функция delta_x (для совместимости)
    std::function<double(int, int)> delta_x_func_;
    bool use_variable_dx_;
    
    // Глобальная сетка (хост)
    GridCUDA global_grid_;
    
    // Устройство (GPU) данные - pimpl
    struct CudaImpl;
    std::unique_ptr<CudaImpl> cuda_impl_;
    
    // Настройки
    int save_interval_;
    bool save_velocities_ = false;
    bool save_eta_max_only_ = false;
    std::string output_folder_;
    
    // Внутренние методы
    void AllocateGPU();
    void FreeGPU();
    void CopyToGPU();
    void CopyFromGPU();
    void CopyEtaMaxFromGPU();   // minimal D2H of only eta_max (benchmark/etamax-only mode)
    
    // Методы для работы с подсетками (устаревшие, для совместимости)
    void CopySubgridToGPU(int subgrid_index);
    void CopySubgridFromGPU(int subgrid_index);
    void StepSubgrid(int subgrid_index);
    
    // ===== НОВЫЕ МЕТОДЫ ДЛЯ КОРРЕКТНОГО GLOBAL PIPELINE =====
    // Загрузка/выгрузка данных между global буферами и GPU
    void LoadSubgridFromGlobal(int s, const double* H_src, const double* u_src,
        const double* v_src, const double* eta_src, const double* D_src);
    void SaveSubgridToGlobal(int s, double* H_dst, double* u_dst,
        double* v_dst, double* eta_dst, double* eta_max_dst);
    
    // Hat данные для MacCormack predictor/corrector
    void CopyHatFromGPU(int s);  // GPU hat -> host global hat buffer
    void LoadHatToGPU(int s);    // Host global hat buffer -> GPU hat
};

#endif // SINGLE_LINE_GRID_SIMULATION_CUDA_CUH
