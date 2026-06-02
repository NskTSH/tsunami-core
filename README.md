# tsunami-core

Header-only **I/O foundation** for the NskTSH tsunami stack: bathymetry/result
parsers, parameter file parsing, source definitions, and the coordinate system.
No solver, no build — just the shared source-level contract.

## Место в экосистеме NskTSH
Общий фундамент, который потребляют **оба** репозитория:
- [`tsunami-simulator`](https://github.com/NskTSH/tsunami-simulator) — SWE-решатель (MacCormack + DG/FV, CPU/CUDA).
- [`tsunami-ui`](https://github.com/NskTSH/tsunami-ui) — Qt-просмотрщик/запуск.

Граница — **header-only исходная зависимость** (НЕ DLL).

## Содержимое (`include/`)
Парсеры/форматы/параметры/координаты/источники (все header-only, самодостаточные, без зависимости от решателя):
`ASCParser.h` (ESRI ASCII), `ZDataParser.h` (.z), `DefaultParametersParser.h`, `EllipsoidalSource.h`,
`ISource.h`, `CoordSystem.h`, `IBaseBathymetryParser.h`, `IBaseParametersParser.h`, `IBaseParser.h`.

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
Private, internal. © NskTSH — Internal use only, см. [LICENSE](LICENSE).
