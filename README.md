# tsunami-core

Header-only **I/O foundation** for the NskTSH tsunami stack: bathymetry/result
parsers, parameter file parsing, source definitions, and the coordinate system.
No solver, no build — just the shared source-level contract.

## Место в экосистеме NskTSH
Общий фундамент, который потребляют **оба** репозитория:
- `tsunami-simulator` (private) — SWE-решатель (MacCormack + DG/FV, CPU/CUDA).
- [`tsunami-ui`](https://github.com/NskTSH/tsunami-ui) — Qt-просмотрщик/запуск.

Граница — **header-only исходная зависимость** (НЕ DLL).

## Содержимое (`include/`)
Парсеры/форматы/параметры/координаты/источники (все header-only, самодостаточные, без зависимости от решателя):
`ASCParser.h` (legacy ESRI ASCII), `StrictEsriAsciiGrid.h` (bounded fail-closed
ESRI ASCII V1 with filesystem and verified stream+size entry points),
`StrictCartesianXyzGrid.h` (versioned bounded stream reader for explicit metric
cell-centre XYZ, independent registration, anisotropic x/y spacing and verified
south-first row-major order; no projection, sorting or regridding),
`nsktsh/io/PreparedTriangleMesh.h` (bounded prepared Gmsh 2.2 ASCII V1 reader;
CCW/manifold/rectangle boundary checks, cancellation, zero-based geometry),
`ZDataParser.h` (.z), `DefaultParametersParser.h`, `EllipsoidalSource.h`,
`ISource.h`, `CoordSystem.h`, `IBaseBathymetryParser.h`, `IBaseParametersParser.h`, `IBaseParser.h`.

`DefaultParametersParser::Read(std::istream&)` принимает уже открытый поток или
проверенные байты, в том числе после чтения Unicode-пути средствами приложения.
Прежний `Read(const std::string&)` сохранён и использует ту же грамматику;
потоковый overload не меняет интерфейс `IBaseParametersParser`. Владелец потока
отвечает за его время жизни, допустимый объём входа и предварительную проверку
данных. Потребителей заголовка необходимо пересобрать.

## Использование (CMake)
```cmake
add_subdirectory(../tsunami-core)            # соседний репо по конвенции; либо -DTSUNAMI_CORE_DIR=<path>
target_link_libraries(<your_target> PRIVATE NskTSH::tsunami-io)
```
`NskTSH::tsunami-io` — INTERFACE-таргет (C++20), добавляет `include/` в include-path. Сборки нет.

## Требования
- C++20. CMake ≥ 3.20 (для таргета).

## Версионирование
Header-only = **source ABI**: изменение заголовков требует пересборки потребителей. Помечать релизы SemVer-тегами.

## Статус / видимость / лицензия
Public. Лицензия — **Apache-2.0**, см. [LICENSE](LICENSE) и `NOTICE`.
