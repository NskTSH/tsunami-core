# tsunami-core

Shallow-water tsunami solver: MacCormack scheme with a DG/FV hybrid, on real
bathymetry. CPU (OpenMP) **and** CUDA backends. Core simulation engine of the
NskTSH ecosystem.

## Место в экосистеме NskTSH
- Потребители: **tsunami-ui** (десктоп-просмотр/запуск) подключает этот репо как git submodule
  и линкует header-only таргет `NskTSH::tsunami-io` (парсеры/форматы).
- Постобработка результатов: **tsunami-analysis**.
- Данные/результаты — на диске вне git, см. конвенцию ниже.

## Возможности
- MacCormack-схема для уравнений мелкой воды (H, u, v, eta) на батиметрии.
- Декомпозиция области: `SingleLineGridSimulation` (горизонтальные полосы, OpenMP + `std::barrier`),
  `MultiLineGridSimulation` (2D), `PartionFileGridSimulator` (out-of-core).
- DG/FV-гибрид (Audusse well-balanced reconstruction), TVD-стабилизатор, активные регионы.
- CUDA-версия (`MacCormackSimulationCUDA/`) — те же ядра на GPU.
- Парсеры входов: `ASCParser` (ESRI ASCII), `ZDataParser` (.z); источники: точечный/эллипсоидальный/…

## Требования
- C++20, Visual Studio 2022 (x64).
- CUDA-версия: NVIDIA CUDA Toolkit.
- Для экспортного таргета `NskTSH::tsunami-io`: CMake ≥ 3.20.

## Сборка
- **Полный решатель (CPU + CUDA):** открыть `MacCormackSimulation.sln`, конфигурация `Release|x64`.
- **Только I/O-таргет для потребителей:** `cmake -B build` → таргет `NskTSH::tsunami-io` (header-only).

## Запуск
```powershell
# из корня репо; пул потоков и тихие часы — см. scripts/silent_run.ps1
MacCormackSimulation\x64\Release\MacCormackSimulation.exe <scenario|par>
```

## Данные и результаты
Пути берутся из `NSKTSH_DATA` / `NSKTSH_RESULTS`
(по умолчанию `../Data` и `../Results` — папки уровнем выше репо). **Без абсолютных путей** (`F:\…`).
Крупные бати/результаты в git **не** коммитятся.

## Статус / видимость
Private, internal. Активная разработка (DG/FV-гибрид, активные регионы).

## Лицензия
© NskTSH. Internal use only — см. [LICENSE](LICENSE).
