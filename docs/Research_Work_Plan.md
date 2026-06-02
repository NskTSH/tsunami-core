# План исследовательских работ PostGS

Дата: 2026-04-03

---

## Обзор

Развитие симулятора цунами и численных методов. Codex CLI — математик/физик, ревьюит и даёт рекомендации, не пишет код.

**Текущий статус (2026-05-11):** Пункты 1, 3, 4, 5, 6, 7, 8, 10, 12(Кориолис v2) полностью завершены. Пункт 13 (coupling fix) — починен (3 бага). Пункт 11.re (S7 hybrid) — прогон запущен с исправленным coupling (11/11 интерфейсов). Пункт 12-DG (DG vs MacCormack) — standalone тест создан. **Пункт 9 (CUDA) — закрыт фазой 17 в DG_MacCormack_Development_Plan.md: Tests 1-5 FP64 PASS (max\|diff\|<2.6e-11), добавлена шаблонизация FP32/FP64 с runtime dispatch.**

---

## 1. Анализ DG-солвера и план доработок

**Статус:** DG-солвер завершён (14 фаз). Нужно закрыть хвосты.

| # | Задача | Приоритет | Статус |
|---|--------|-----------|--------|
| 1.1 | Запустить CouplingTest (9.6) — structured(60×60) + mesh(30×30) | Высокий | ✅ PASS, mass err 0.0005% |
| 1.2 | Формальный flux balance (10.4) — ∑flux на интерфейсах structured↔mesh | Средний | ✅ Проанализировано: coupling state-based, не flux-based. Точный flux balance → задача 2 |
| 1.3 | Codex review всего DG-солвера — знаки, нормали, thread safety | Высокий | ✅ Codex (gpt-5.4): 6/7 OK. **BUG найден и исправлен**: positivity limiter проверял edge midpoints вместо Gauss quad points |
| 1.4 | Определить: нужен ли DG для сферических координат? | Решить | ✅ **НЕТ**: геом. ошибка 0.1-0.8% на 10-50 км, Кориолис L/L_R < 0.5. Cartesian DG достаточен. Опционально: f-plane source term |
| 1.5 | Benchmark: DG vs MacCormack на одинаковой задаче | Средний | ⏳ Отложен — приоритет ниже, перешли к пункту 4 |

**Codex review итоги (gpt-5.4):** HLL Riemann OK, surface signs OK, Audusse hydrostatic OK, source terms OK, well-balanced test validates, thread safety OK, positivity point set BUG (исправлен). Доп.: комментарии к D исправлены (bed elevation, η=H+D).

---

## 2. Гибрид DG + MacCormack на одной сетке

**Цель:** На одном домене разные подсетки считаются разными солверами. Пользователь вручную задаёт, какая подсетка считается DG. У DG-подсетки граничные 2 слоя узлов — ровные прямоугольные ячейки (для корректной передачи данных в MacCormack), остальные — произвольная неравномерная треугольная сетка.

**Что уже работает (архитектура готова на ~80%):**

| Компонент | Статус |
|-----------|--------|
| IPatch полиморфизм (StructuredPatch / MeshPatch) | ✅ |
| SimulationRunner с mixed patches | ✅ |
| ExchangeEngine type-agnostic | ✅ |
| CouplingBuilder::BuildStructuredToMesh (nearest-neighbor) | ✅ |
| DGBoundaryType::Coupled + HLL flux с внешним состоянием | ✅ |
| MeshPatch::ApplyExchange → SetCoupledState | ✅ |
| CouplingTest (structured 60×60 + mesh 30×30, mass error 0.0005%) | ✅ |
| HybridDomainBuilder::BuildStructuredMeshDomain() — proof of concept | ✅ |

**Что нужно доработать:**

| # | Задача | Сложность | Статус |
|---|--------|-----------|--------|
| 2.1 | Генератор сетки со структурированным граничным слоем: `GenerateHybridMesh()` — 2 слоя rect-triangles у границы + произвольные triangles внутри | Средняя | ⏳ |
| 2.2 | Поддержка quad→2tri в MeshReader (или Gmsh structured layer) | Низкая | ⏳ |
| 2.3 | Верификация coupling accuracy с aligned cells (mass error → 0) | Низкая | ⏳ |
| 2.4 | Удобный API: ручное указание solver type через .par или ScenarioBuilder | Низкая | ⏳ |
| 2.5 | Bathymetry projection на coupling interface (сейчас D_other = D_this) | Средняя | ⏳ |
| 2.6 | Conservation audit — формальная проверка ∑flux на интерфейсе | Средняя | ⏳ |

**Архитектурная схема:**

```
┌──────────────────────────────────┬──────────────────────────────────┐
│      MacCormack (SubGrid)        │         DG (MeshPatch)           │
│      структурная сетка           │                                  │
│      прямоугольные ячейки        │  ┌─ 2 слоя прямоугольных ──┐    │
│                                  │  │  ячеек (для обмена)      │    │
│                                  │  ├──────────────────────────┤    │
│                                  │  │  неравномерные           │    │
│                                  │  │  треугольники            │    │
│                                  │  │  (произвольная сетка)    │    │
│                                  │  └──────────────────────────┘    │
└──────────────────────────────────┴──────────────────────────────────┘
                           coupling interface
```

**Ключевые моменты:**
- Прямоугольные ячейки на границе DG выровнены с MacCormack → nearest-neighbor даёт вес = 1.0, без интерполяционной ошибки
- DG-солвер поэлементный — произвольные размеры/формы треугольников внутри не проблема
- CFL ограничение определяется самым маленьким элементом DG-патча
- Два солвера шагают с одним глобальным dt (subcycling — на будущее)

**Вопросы для Codex:**
1. При nearest-neighbor с aligned cells — гарантирована ли побитовая conservation? Или нужен flux-based coupling?
2. DG P2 даёт O(h³) внутри, но какой порядок на coupled boundary при передаче cell averages (O(h) данные)?
3. Два разных CFL на одном домене — нужен ли subcycling?

---

## 3. Альтернативная разбивка на линии (вертикальная)

**Проблема:** `SingleLineGridSimulation` всегда режет по X (горизонтальные полосы). Для некоторых задач вертикальная разбивка эффективнее.

| # | Задача | Файлы | Статус |
|---|--------|-------|--------|
| 3.1 | Добавить `enum DecompositionDirection { Horizontal, Vertical }` | `CoordSystem.h` | ✅ |
| 3.2 | `SubGrid` — параметризовать ghost cells по X или по Y | `SubGrid.h/cpp` | ✅ Vertical: ghost по X+Y |
| 3.3 | `BoundaryCondition` — добавить `DefaultTopConditions`/`DefaultBottomConditions` | `BoundaryCondition.h/cpp` | ✅ 4 функции |
| 3.4 | `SingleLineGridSimulation::Initialize()` — условная разбивка | `SingleLineGridSimulation.cpp` | ✅ |
| 3.5 | `UpdateGlobalGrid` — пропускать Y-ghosts при вертикальной разбивке | `SubGrid.cpp` | ✅ |
| 3.6 | Параметр в .par файлах или ScenarioBuilder | `ScenarioBuilder.h` | ✅ decomp_dir в scenario |
| 3.7 | Регрессионный тест: horizontal vs vertical → побитовое совпадение | — | ✅ max|diff| = 0, Cells > 1e-12: 0/13456 |

**Выводы:** Численные ядра не менялись. Вертикальная разбивка добавляет ghost cells по обеим осям (X — domain edges, Y — strip boundaries). Ключевой фикс: на domain Y-edges без соседа применяется zero-gradient BC (как в горизонтальном режиме) → побитовое совпадение. Spherical variant требует отдельной доработки (j=0/j=ny-1 edge loops).

---

## 4. Япония в сферических координатах с S7

**Готовность:** Test 18 = `BuildTest6SphericalS7()` — Japan Sea с S7 в сферических координатах. Пути исправлены (Data/ relative). Для честного сравнения добавлен Test 19 = `BuildTest6CartesianS7()`.

| # | Задача | Статус |
|---|--------|--------|
| 4.1 | Проверить парсинг `JapanSeaOneSourceS7.par` + исправить пути | ✅ BuildJapanSeaScenario helper, Data/ relative |
| 4.2 | Запустить сценарий 18 (S7 spherical) через Sandbox | ✅ Завершён за 4519 сек (75 мин). Результат: `Result/eta_max_timestep_0.txt` (146 MB), сохранена копия `Result/eta_max_test18_spherical_s7.txt` |
| 4.3 | Сравнить с декартовым вариантом — визуализировать разницу | ✅ Test 19 завершён (~100 мин). Diff max=0.40m, mean rel=1.5%. Визуализация: Code/Python/s7_spherical_vs_cartesian.png |
| 4.4 | Codex: проверить корректность delta_phi, delta_lambda для JapanSea.asc | ✅ delta_y=462.5 vs exact 463.8 (0.28%), center_lat=43° OK |

**Источник S7:** `JapanSeaOneSourceS7.par` приведён в соответствие с legacy-инициализацией: lon=132.00, lat=36.00, angle=-80, `r1=0.148°`, `r2=0.449°`, `h0=2.0 m`.

**Технические правки по S7 (2026-04-03):**
- В `Data/JapanSeaOneSourceS7.par` исправлены radii: теперь `.par` соответствует legacy CPU-инициализации, а не point-source заглушке.
- Добавлен отдельный сценарий `BuildTest6CartesianS7()` и индекс **Test 19** для прямого сравнения `Cartesian S7` vs `Spherical S7`.
- `Source.cpp` теперь принимает индекс теста через аргумент командной строки: `MacCormackSimulation.exe <testIndex>`. Поведение по умолчанию сохранено: без аргумента запускается Test 18.
- CPU-проект успешно пересобран напрямую через `MacCormackSimulation.vcxproj`.
- Полная сборка `.sln` по-прежнему блокируется CUDA-частью: отсутствует `CUDA 12.8.props`. Для CPU-итераций использовать сборку только `MacCormackSimulation.vcxproj`.

---

## 5. Кейс Кориолиса: волна снизу вверх через экватор

**Параметры:**
- Плоское дно, глубина **1000 м**
- Шаг сетки **1000 м** (~0.009° на экваторе)
- Центр сетки на **экваторе** (φ₀ = 0°)
- Размах по широте **20°** (от -10° до +10°)
- Волна — плоский фронт, юг → север
- **Сравнить:** Spherical (Кориолис) vs Cartesian (без)

| # | Задача | Статус |
|---|--------|--------|
| 5.1 | Создать `BuildTestCoriolis1()` в ScenarioBuilder | ✅ Test 21 (Cartesian), Test 22 (Spherical) |
| 5.2 | Параметры: flat bottom D=1000, dx=dy=1000, φ₀=0, nx=ny=2200 | ✅ |
| 5.3 | Источник: плоский фронт 10 ячеек у нижней границы, amp=1m | ✅ Grid::SetParamsCoriolis |
| 5.4 | Запустить Spherical и Cartesian | ✅ Test 21: 256s, Test 22: 267s |
| 5.5 | Визуализация: разница eta, отклонение фронта | ✅ Снимки каждые 1000 шагов. Diff карты + u-field анализ |
| 5.6 | Codex: аналитическая оценка отклонения Кориолисом | ✅ Уточнённая оценка: отклонение 0.16-0.31 км (НЕ 6-19 км — ранее использовалась фазовая скорость c вместо скорости частиц c·η/H) |

**Результаты (2026-04-04):**
- **eta**: антисимметричная разница ±0.19m — это эффект сферической геометрии (cos(φ) коррекция dx), НЕ Кориолис
- **u-field**: антисимметричный (−SH, +NH), ~0.01 м/с — ЭТО Кориолис-отклик
- **Фронт не смещается**: для x-однородной плоской волны Кориолис создаёт поперечную u, не сдвиг η
- **Амплитуда 0.63 вместо 1.0**: начальное условие u=v=0 — стоячая волна, разлетается в обе стороны

**CFL:** c = √(9.81×1000) ≈ 99 м/с. При dt=5с: CFL ≈ 0.99 (2D: tau*(c/dx+c/dy)).

---

## 6. Кейс Кориолиса: волна по горизонтали (запад → восток)

| # | Задача | Статус |
|---|--------|--------|
| 6.1 | `BuildTestCoriolis2()` — тот же домен, источник слева | ✅ Test 23 (Cart), Test 24 (Sph), vertical decomp |
| 6.2 | Использовать вертикальную разбивку (пункт 3) для параллелизации | ✅ DecompositionDirection::Vertical |
| 6.3 | Запустить Spherical vs Cartesian | ✅ Test 23: 330s, Test 24: 348s |
| 6.4 | Визуализация: отклонение фронта по широте | ✅ Max diff 0.001m (vs 0.19m для S→N) |
| 6.5 | Codex: сравнить аналитику с результатами обоих кейсов | ✅ Подтверждено: S→N diff — геометрия, не Кориолис |

**Результат:** W→E diff 200× меньше чем S→N → подтверждает что S→N разница от cos(φ) геометрии.

---

## 7. Эллипсоидный источник для кейсов 5 и 6

| # | Задача | Статус |
|---|--------|--------|
| 7.1 | Повторить кейс 5 с `EllipsoidalSource(r1=50, r2=200)` cells | ✅ Test 25 (Cart), Test 26 (Sph) |
| 7.2 | Повторить кейс 6 с `EllipsoidalSource(r1=200, r2=50)` cells | ✅ Test 27 (Cart), Test 28 (Sph) — код создан |
| 7.3 | Попробовать размеры как у s1-s7 (пересчитать из градусов в метры) | ⏳ |
| 7.4 | Сравнить: точечный vs эллипсоидный — как меняется картина Кориолиса | ✅ diff ~0.002m — малый Кориолис-эффект на конечном источнике |
| 7.5 | Визуализация: серия снимков через равные промежутки | ✅ 4 снимка (1000-4000 шагов), eta + diff + eta_max |

**Результат:** Эллипсоидный источник: кольцевой фронт, diff ~0.002m (0.2%). Кориолис-эффект на η слабый, основной отклик в u-поле.

---

## Порядок выполнения и зависимости

```
1 (DG анализ) ─────────── независимый
2 (гибрид DG+MacCormack) ─ независимый, параллельно с остальным
3 (вертикальная разбивка) ─ нужна для 6
4 (Япония S7 сферич.) ──── независимый
5 (Кориолис ↑) ─────────── нужен новый сценарий
6 (Кориолис →) ─────────── зависит от 3 (оптимально) + 5 (паттерн)
7 (эллипсоид) ──────────── зависит от 5 и 6
```

**Рекомендуемый порядок:** 1 → 4 → 3 → 5 → 6 → 7, с 2 параллельно.

**Обновлённый порядок (2026-04-08):**
```
Coupling DG↔MacCormack:
✅ Interface matching (5 bugs, 2026-04-06)
✅ Buffer aliasing (3 bugs, 2026-04-07)  
✅ Time sync + auto tau (2026-04-07)
✅ Wet/dry handling for real bathymetry (2026-04-07)
✅ S↔S neighbor pointers (2026-04-08)
✅ Target-driven NearestMapping (2026-04-08)
✅ Eta-based H reconstruction (2026-04-08)
✅ Initial exchange for coupled states (2026-04-08)
⚠️ Mass conservation: state-based coupling non-conservative but FUNCTIONAL
   → DG HLL flux ≠ MacCormack FD flux at interface → amplitude amplification
   → HybridFlatTest (120×120 flat, 4 stripes, DG stripe 1, source eta=1):
     Step 4000: wave propagates through ALL 4 stripes ✅
     Stripe 0: [-96, 452], Stripe 1(DG): [-25, 3766], Stripe 2: [-42, 274], Stripe 3: [-47, 148]
     Amplitudes 100-3000× too high (source=1), NaN at late steps from accumulated energy
   → Codex: need flux-based coupling (unit conversion: DG integrated flux / edge_length → density)
   → Implementation attempted (SubGrid flux correction), rolled back due to unit mismatch
   → Next step: correct flux density conversion and re-enable
```

**Обновлённый порядок (2026-04-06):**
```
✅ 1 (DG анализ) — positivity limiter fix, Codex review OK
✅ 3 (вертикальная разбивка Cartesian) — побитовое совпадение
✅ 4 (S7 Spherical + Cartesian) — оба завершены, diff max=0.48m (1.5%)
✅ 5 (Кориолис ↑) — Cart/Sph diff 0.191m — cos(φ) геометрия, не Кориолис
✅ 6 (Кориолис →) — diff 0.001m, подтверждает вывод
✅ 7 (эллипсоиды) — diff 0.001m, Кориолис на η слабый
✅ 10 (визуализация) — 8 видео Кориолис + 2 видео S7 + 4 сравнения + max_wave_height
✅ 12 (Кориолис v2 φ=45°) — тесты 41-48 завершены, S→N diff +102% (Кориолис)
✅ 13 (coupling fix) — 3 бага: Initialize не вызывался, right iface на global_i_end_-1,
     write_index=read_index, j_mid для сферических координат, relative tolerance
✅ 8 (оптимизация CPU) — dry cell skip уже был реализован, проверено
⏳ 11.re (S7 hybrid) — coupling починен (11/11), прогон запущен
⏳ 12-DG (DG vs MacCormack) — standalone DG тест создан, прогон запущен
  9 (CUDA починка) ── блокер: нет CUDA 12.8 SDK
```

---

## Результаты прогонов (2026-04-04)

### Кориолис-тесты (8 тестов, 2200×2200, flat D=1000m, экватор)

| Пара | Тип | max|diff| Cart vs Sph | Вывод |
|------|-----|------------------------|-------|
| 21 vs 22 | Plane S→N | **0.191 m** | cos(φ) геометрия |
| 23 vs 24 | Plane W→E | **0.001 m** | подтверждает |
| 25 vs 26 | Ellip N-S | **0.001 m** | Кориолис на η < 0.1% |
| 27 vs 28 | Ellip W-E | **0.001 m** | Кориолис на η < 0.1% |

**Codex (gpt-5.4) выводы:**
- S→N 9% diff — частично cos(φ), частично implementation-specific обработка краёв
- W→E vs S→N анизотропия (0.477 vs 0.632) — численный артефакт Vertical decomposition
- max_eta_max = 1.00008 — нормальный MacCormack overshoot на разрывном IC
- Эллипс max_eta_max = 0.99974 < 1.0 — `eta_max` не инициализируется из t=0

### S7 Japan Sea (3601×4801, 25000 шагов)

| Тест | Coord | max_eta_final | max_eta_max | Время |
|------|-------|---------------|-------------|-------|
| 18 | Spherical | 0.939 m | **3.370 m** | ~75 мин |
| 19 | Cartesian | 0.951 m | **3.359 m** | ~75 мин |

**Codex (gpt-5.4) выводы:**
- 1% разница — нормально: для цунами f/ω ≈ 0.01-0.02
- Кориолис виден в u/v и arrival-time, не в амплитуде eta
- ~1 m в бассейне, ~3.4 m прибрежный max — правдоподобно (ист. данные 1983: run-up до 14.9 m)

### S7 гибрид (DG stripe 3) — НЕВАЛИДНЫЕ ДАННЫЕ

| Тест | Результат | Причина |
|------|-----------|---------|
| 30 | max_eta=1.999, min=0 | `MeshPatch::UpdateGlobalGrid()` был пустым TODO |
| | NaN с шага ~10000 | DG stripe не записывала результаты в global grid |

**Фикс (2026-04-05):** Реализован `MeshPatch::UpdateGlobalGrid()` — проекция DG cell averages → structured grid. Требуется повторный прогон.

### Артефакты

| Путь (относительно `Result/`) | Содержимое |
|------|-----------|
| `test{21..28}/images/` | 320 heatmap PNG (40 кадров × 8 тестов) |
| `test{21..28}/test{N}.mp4` | 8 видео (8 fps, 5 сек) |
| `test{21..28}/summary.txt` | Параметры + max wave height |
| `test{21..28}/max_wave_height.txt` | max eta на каждом шаге |
| `compare_test{A}_vs_test{B}/` | 4 сравнения Cart vs Sph (PNG + MP4) |
| `test{18,19}/images/` | 100 heatmap PNG (50 × 2) |
| `test{18,19}/test{N}.mp4` | 2 видео S7 (10 fps, 5 сек) |
| `test{18,19}/test{N}_eta_max.png` | eta_max карты |
| `compare_test18_vs_test19/` | S7 Sph vs Cart (50 PNG + MP4) |
| `test30_hybrid_s7_spherical/` | ⚠️ НЕВАЛИДНО — требуется повторный прогон |

---

## Ближайшие действия (2026-04-05)

### Шаг 0: Повторить Кориолис-тесты 5, 6, 7 на увеличенной сетке (~3 часа)

**Проблема предыдущего прогона:** Центр домена на **экваторе** (φ₀=0°), где f=2Ωsin(0)=**0**. Кориолис-эффект отсутствует по определению. Видимая разница Cart/Sph — это cos(φ) коррекция dx, не Кориолис.

**Новые параметры (тесты 41-48):**

| Параметр | Было (тесты 21-28) | Стало (тесты 41-48) | Зачем |
|----------|---------------------|----------------------|-------|
| Центр | φ₀ = 0° (экватор) | **φ₀ = 45°** (Японское море) | f = 1.03×10⁻⁴ rad/s |
| Сетка | 2200 × 2200 | **4000 × 4000** | Больше пространства для наблюдения |
| Шаги | 4000 (20000 s) | **8000** (40000 s = 11 часов) | ~60% инерционного периода (T_i=17h) |
| dx, dy | 1000 m | 1000 m | Без изменений |
| Глубина | D = 1000 m | D = 1000 m | Без изменений |
| Амплитуда | 1 m | 1 m | Без изменений |

**Физические ожидания при φ=45°:**
- f = 1.03×10⁻⁴ rad/s, инерционный период T_i = 2π/f ≈ **17 часов**
- За 11 часов: угол поворота f*t ≈ 4.1 рад (**~235°**) — больше полного оборота в инерционной системе
- Ожидаемый эффект: заметная поперечная скорость u, асимметрия фронта
- Для плоской волны: Кориолис создаёт Kelvin-wave-like поведение, отклонение вправо (NH)

**Тесты:**

| Индекс | Описание | Аналог |
|--------|----------|--------|
| 41 | S→N plane, Cartesian | 21 |
| 42 | S→N plane, **Spherical (Кориолис)** | 22 |
| 43 | W→E plane, Cartesian | 23 |
| 44 | W→E plane, **Spherical (Кориолис)** | 24 |
| 45 | Ellipsoidal N-S, Cartesian | 25 |
| 46 | Ellipsoidal N-S, **Spherical (Кориолис)** | 26 |
| 47 | Ellipsoidal W-E, Cartesian | 27 |
| 48 | Ellipsoidal W-E, **Spherical (Кориолис)** | 28 |

**Запуск (~3 часа при 4000×4000 × 8000 шагов):**
```bash
# run from the tsunami-core repo root (data resolved via $NSKTSH_DATA, default ../Data)
for t in 41 42 43 44 45 46 47 48; do
  MacCormackSimulation/x64/Release/MacCormackSimulation.exe $t
done
```

**Визуализация:**
```bash
cd Code/Python
python visualize_simulation.py Result/test{41..48}
python visualize_simulation.py --compare Result/test41 Result/test42
python visualize_simulation.py --compare Result/test43 Result/test44
python visualize_simulation.py --compare Result/test45 Result/test46
python visualize_simulation.py --compare Result/test47 Result/test48
```

**Что смотреть:**
1. Diff-карта eta: должна быть значительно больше 0.001 m
2. u-field: антисимметрия NH/SH (но теперь центр на 45°, не на экваторе — вся сетка в NH)
3. Отклонение фронта вправо для S→N волны (NH Кориолис)
4. Эллипсоидный источник: асимметрия кольцевого фронта (east-west bias)

**Результаты Кориолис v2 (2026-04-06, тесты 41-48 завершены):**

| Пара | Тип | Cart max_eta | Sph max_eta | Разница | Вывод |
|------|-----|-------------|-------------|---------|-------|
| 41 vs 42 | S→N plane | 0.637 | **1.286** | **+102%** | Сильный Кориолис-эффект |
| 43 vs 44 | W→E plane | 0.410 | 0.397 | -3% | Малый эффект |
| 45 vs 46 | Ellip NS | 0.108 | 0.108 | ~0% | Кориолис на η < 0.1% |
| 47 vs 48 | Ellip WE | 0.027 | 0.027 | ~0% | Кориолис на η < 0.1% |

**Ключевой вывод:** При φ=45° S→N плоская волна показывает **двукратное** усиление max_eta в сферической системе. На экваторе (тесты 21-22) разница была всего 0.19 m (от cos(φ)), тут 0.65 m — настоящий Кориолис.

W→E разница 3% — ожидаемо малая: для зональной волны Кориолис создаёт меридиональную u, но η не меняет.

Эллипсоидные источники: на конечном размере кольцевого фронта Кориолис-эффект усредняется до < 0.1%.

---

### Шаг 1: Повторный прогон S7 гибрид (~2 часа)

**Предыдущие баги (все исправлены 2026-04-06):**
1. `MeshPatch::UpdateGlobalGrid()` — был пустым TODO (2026-04-05)
2. `StructuredPatch::Initialize()` — не вызывался из `WrapSubGrid()` → 0 интерфейсов
3. `BuildRightInterface` — координата на `global_i_end_-1` вместо `global_i_end_`
4. `CopyBufferToPatch` — писал в `local_index` (вычислительная ячейка) вместо ghost cell
5. `BuildHybridStripesOneDG` — x0 вычислялся для j=0, а delta_x зависит от широты
6. `DoInterfacesMatch` — абсолютный tolerance 1e-3 + сравнение bbox avg вместо median

**Фикс:** write_index в InterfacePoint, j_mid для сферических координат, relative tolerance, find_x_at_y для median matching. CouplingTest: mass error 0.0016%, coupling stable. S7: 11/11 couplings.

### Шаг 2: Визуализация + сравнение S7 pure vs hybrid
```bash
cd Code/Python
python visualize_simulation.py Result/test30_hybrid_s7_spherical --cmap hot
python visualize_simulation.py --compare Result/test18 Result/test30_hybrid_s7_spherical
```

### Шаг 3: Codex review гибридных результатов
Проверить: DG stripe даёт физически осмысленные значения, coupling interface не создаёт артефактов

### Оставшиеся исследовательские вопросы:
- **Кориолис на Japan Sea**: при lat=36-45°, f≈8.6-10.3×10⁻⁵ — заметна ли разница за 3.5 часа?
- **Численная диссипация MacCormack**: затухает ли 1m волна за 2200 км?
- **DG + Кориолис**: f-plane source term (0, f*Hv, -f*Hu) — опционально
- **Disk Offload на Japan Sea (3601×4801)**: проверить производительность выгрузки подсеток на диск на реальной акватории. Оценить замедление и пиковое потребление RAM. Сравнить с in-memory прогоном (текущие ~75 мин).
- **CFL совместимость**: DG stripe с общим dt от MacCormack

**Наблюдения Codex по S7 (2026-04-03):**
- `eta_max` сохраняется один раз после завершения расчёта, layout (nx, ny) = 3601 × 4801
- Поле не нулевое: max = 3.37 m, >0.1 m — 1.4M ячеек, >1.0 m — 4377 ячеек
- Моделируется 12500 s (3.47 h), распространение по всему Японскому морю правдоподобно
- Максимум eta_max около lon≈131.88°, lat≈34.76° — coastal shoaling/reflection

---

## 8. Оптимизация производительности

**Проблема:** S7 на сетке 3601×4801 (17.3M ячеек) × 25000 шагов считается ~2 часа (12 потоков). Кориолис-тест 2200×2200 (4.8M) × 4000 шагов — ~10 минут. Для итеративной работы это слишком долго.

**Бутылочное горлышко:** MacCormack ядро — основной цикл OpenMP + SIMD. Память: ~2.3 GB для S7.

| # | Идея | Оценка ускорения | Сложность |
|---|------|-------------------|-----------|
| 8.1 | **Уменьшить сетку для отладки** — S7 на 900×1200 (1/16 от полной) | 16× | Низкая — просто параметр |
| 8.2 | **Adaptive dt** — увеличить tau при мелкой глубине (CFL < 1 на ~60% сетки) | 2-3× | Средняя — нужен CFL-контроль |
| 8.3 | **Subcycling в зонах глубокой воды** — dt_deep > dt_shallow | 2-4× | Высокая |
| 8.4 | **CUDA версия** — уже есть скелет MacCormackSimulationCUDA | 10-50× | Средняя — ядра простые |
| 8.5 | **Dry cell skip** — пропускать ячейки с D < grnd (до 30% домена — суша) | 1.3-1.5× | Низкая |
| 8.6 | **Уменьшить timesteps** — для отладки 5000 вместо 25000 | 5× | Тривиальная |
| 8.7 | **Профилирование** — VTune/perf чтобы найти реальное бутылочное горлышко | — | Низкая |
| 8.8 | **Бинарный вывод** — заменить текстовые eta_*.txt на бинарные | 2-5× I/O | Низкая |

**Рекомендации:**
- Для отладки: 8.1 + 8.6 (быстро, без изменения кода)
- Для production: 8.4 (CUDA) или 8.2 (adaptive dt)
- Прямо сейчас: 8.5 (dry cell skip) — минимальные изменения, заметный эффект

---

## 9. Починка CUDA-версии (MacCormackSimulationCUDA) — ✅ **ЗАКРЫТ (2026-05-11)**

**Результат:** все Tests 1-5 FP64 PASS с max\|diff\| < 2.6e-11 vs CPU. Детали — в [DG_MacCormack_Development_Plan.md, Фаза 17](DG_MacCormack_Development_Plan.md#фаза-17).

**Что было сделано:**
- 17.1.5 X-padding (`nx+2` ghost columns matching CPU `SubGrid::local_nx`)
- 17.2 Динамические BC: порядок применения wave → Y-BC (как CPU), separate Hat3 (zero_v_top), exchangeHatBoundaryIndex теперь не игнорируется, applyDynamicLeftBoundaryKernel без лишней `D < min_calc_value` проверки
- 17.4-17.5 Шаблонизация `MacCormackCUDA<Real>` через `IMacCormackCUDA` + factory; CLI `--precision=fp32|fp64` (one binary, runtime dispatch)
- 17.6 Python regression `Code/Python/compare_cuda_cpu.py`
- 17.8 Legacy `SubGridCUDA.cpp/.h`, `SimulationKernels.cu/.h` исключены из build

**FP32 status:** Tests 1-3 работают (rel ~5-10%), Tests 4-5 — NaN на wet/dry interfaces (известное ограничение FP32 + MacCormack без slope limiter).

**Историческая часть (для контекста, как было до фазы 17):**

**Проблема:** CUDA-версия запускается и считает, но результат не совпадает с CPU. Ревью выявил 5 проблем.

**Текущий build blocker (2026-04-03 → решён):**
- Полная сборка `MacCormackSimulation.sln` на текущей машине падает на `MacCormackSimulationCUDA.vcxproj`: отсутствует `CUDA 12.8.props`.
- Рабочий обходной путь для CPU-исследований: собирать только `MacCormackSimulation.vcxproj`.

**Файлы:** `MacCormackSimulationCUDA/SimulationKernels.cu`, `SubGridCUDA.cpp`

| # | Баг | Описание | Строки |
|---|-----|----------|--------|
| 9.1 | **Hat init для ghost columns отсутствует** | Half-step ядро не инициализирует hat-переменные при i=0 и i=local_nx-1. CPU версия явно копирует H_hat=H, u_hat=u и т.д. на этих границах. CUDA оставляет мусор → full-step читает stale данные | `SimulationKernels.cu:17-81` vs `SubGrid.cpp:585-627` |
| 9.2 | **internalBoundaryKernel копирует j=1→j=0, а CPU копирует H→H_hat** | CPU: `H_hat[0,j] = H[0,j]` (из текущих полей). CUDA: `H_hat[0,j] = H_hat[1,j]` (из соседней hat-ячейки). Принципиально разная семантика | `SimulationKernels.cu:152-184` |
| 9.3 | **Ghost columns i=0, i=nx-1 вообще не в internalBoundaryKernel** | Ядро обрабатывает только j-границы (top/bottom), X-границы игнорируются | `SimulationKernels.cu:152-184` |
| 9.4 | **Dry cell: H_hat=0 вместо H_hat=D** | CPU при D<grnd: `H_hat = D`. CUDA: `H_hat = 0.0`. При осреднении 0.5*(H+H_hat) даёт разные значения. | `SimulationKernels.cu:39` vs `SubGrid.cpp:503` |
| 9.5 | **Порядок вызовов** | Нужно убедиться что exchange + hat init + boundary → всё в правильном порядке, как в CPU Run() | Весь pipeline |

**Приоритет:** 9.1 и 9.4 — наиболее вероятные причины расхождения. 9.1 даст самую большую разницу (мусорные hat на границах → ошибка растекается внутрь).

**План починки:**
1. Добавить kernel/логику для инициализации hat при i=0, i=nx-1 (аналог CPU строк 585-627)
2. Исправить dry cell: H_hat = D[idx] вместо 0.0
3. Исправить internalBoundaryKernel: копировать из основных полей, не из hat-соседей
4. Регрессия: CPU vs CUDA на Test1 (120×120, 500 шагов) → побитовое совпадение
5. Запустить S7 на GPU — оценить ускорение

**Исследовать после починки:**
- **Сферические координаты на GPU**: CPU Spherical ядро сложнее (Кориолис + j-edge loops). Нужно портировать
- **Multi-GPU**: для сеток >10M ячеек один GPU может не хватить по памяти (S7 = 2.3 GB на CPU)
- **Mixed precision**: float32 для промежуточных вычислений, float64 для аккумуляции — потенциально 2× на GPU
- **Требуется CUDA Toolkit 12.8** — проект настроен на него, но SDK не установлен на текущей машине

**Блокер:** Для сборки CUDA нужен NVIDIA CUDA Toolkit 12.8. На текущей машине его нет (`CUDA 12.8.props not found`).

---

## 10. Визуализация Кориолис-тестов и S7 (2026-04-04)

**Цель:** Промежуточные снимки eta → картинки → видео для всех Кориолис-направлений + S7 чистый и S7 гибрид.

**Изменения в C++:**

| # | Действие | Файл | Статус |
|---|----------|------|--------|
| 10.1 | `SetResultFolder()` — вывод в `Result/test{N}/` вместо плоского `Result/` | `SingleLineGridSimulation.h/cpp` | ✅ |
| 10.2 | `save_interval_` используется в `Run()` вместо хардкодного 1000 | `SingleLineGridSimulation.cpp` | ✅ |
| 10.3 | `summary.txt` после каждого прогона: nx, ny, tau, max_eta, coord_system | `SingleLineGridSimulation.cpp` | ✅ |
| 10.4 | `Sandbox::SimulatorTests()` — авто-настройка interval и папки по testIndex | `Sandbox.h` | ✅ |
| 10.5 | Сохранение eta + u + v на каждом интервале (не только eta + u) | `SingleLineGridSimulation.cpp` | ✅ |
| 10.6 | Финальный снимок eta всегда сохраняется | `SingleLineGridSimulation.cpp` | ✅ |

**Интервалы сохранения:**
- Кориолис (тесты 21-28): каждые 100 шагов → 40 кадров за 4000 шагов
- S7 (тесты 18-19): каждые 500 шагов → 50 кадров за 25000 шагов
- Остальные: каждые 10 шагов (legacy)

**Python pipeline:**

| # | Файл | Назначение | Статус |
|---|------|-----------|--------|
| 10.7 | `Code/Python/visualize_simulation.py` | Универсальный визуализатор: eta → heatmap PNG → MP4 | ✅ |
| 10.8 | `Code/Python/run_coriolis_visualization.py` | Пакетный запуск для всех тестов + comparisons | ✅ |

**Запуск (порядок):**

```bash
# 1. Запустить все Кориолис-тесты (по 5 мин каждый, 8 тестов ≈ 40 мин)
MacCormackSimulation.exe 21  # S→N Cartesian
MacCormackSimulation.exe 22  # S→N Spherical
MacCormackSimulation.exe 23  # W→E Cartesian
MacCormackSimulation.exe 24  # W→E Spherical
MacCormackSimulation.exe 25  # Ellip N-S Cartesian
MacCormackSimulation.exe 26  # Ellip N-S Spherical
MacCormackSimulation.exe 27  # Ellip W-E Cartesian
MacCormackSimulation.exe 28  # Ellip W-E Spherical

# 2. S7 тесты (по 75-100 мин каждый)
MacCormackSimulation.exe 18  # S7 Spherical
MacCormackSimulation.exe 19  # S7 Cartesian

# 3. Визуализация
cd Code/Python
python run_coriolis_visualization.py --result-root ../../Result
```

**Выходные данные:**
- `Result/test{N}/images/` — PNG heatmaps
- `Result/test{N}/test{N}.mp4` — видео
- `Result/test{N}/summary.txt` — параметры + max wave height
- `Result/test{N}/max_wave_height.txt` — max eta на каждом сохранённом шаге
- `Result/compare_{A}_vs_{B}/` — сравнительные карты и видео

---

## 11. S7 гибрид (одна полоса DG)

**Цель:** Прогнать S7 Japan Sea с одной из 12 полос, вычисляемой DG-солвером вместо MacCormack. Сравнить визуально с чистым MacCormack.

**Реализация:**

| # | Действие | Файл | Статус |
|---|----------|------|--------|
| 11.1 | `BuildHybridStripesOneDG()` — N-1 structured + 1 DG stripe | `HybridDomainBuilder.h/cpp` | ✅ |
| 11.2 | Батиметрия из глобальной сетки → DG (nearest-neighbor) | `HybridDomainBuilder.cpp` | ✅ |
| 11.3 | IC: SWEState(H, u*H, v*H) из глобальной сетки | `HybridDomainBuilder.cpp` | ✅ |
| 11.4 | Coupling: left (tag 4) + right (tag 2) как DGBoundaryType::Coupled | `HybridDomainBuilder.cpp` | ✅ |
| 11.5 | Mesh cap: 300×600 triangulation (~360K elements) | `HybridDomainBuilder.cpp` | ✅ |
| 11.6 | `HybridS7Test()` в Sandbox с сохранением eta + summary | `Sandbox.h` | ✅ |
| 11.7 | Маршрутизация в Source.cpp: `hybrids7` / `hybrids7cart` | `Source.cpp` | ✅ |

**Запуск:**
```bash
MacCormackSimulation.exe hybrids7     # Test 30: S7 Spherical hybrid
MacCormackSimulation.exe hybrids7cart  # Test 31: S7 Cartesian hybrid
```

**DG stripe:** полоса 3 (i=900..1200), ≈132° долготы — около источника S7.

**Ограничения и риски:**
- DG mesh capped at 300×600 → коэффициент сжатия ~8× по сравнению со структурированной полосой (~300×4801)
- P2 DG на 360K треугольников (~2.2M DOF) — вычислительно тяжело, ~10-30× медленнее MacCormack на том же участке
- MacCormack и DG работают с одним глобальным dt — для P2 DG обычно нужен CFL ≈ 0.3, тогда как MacCormack стабилен при CFL ≈ 1.0. Текущий S7 tau=0.5s → проверить стабильность
- Батиметрия nearest-neighbor — на крутых склонах возможно рассогласование. Для production нужна билинейная интерполяция

**Codex review вопросы:**
1. Совместимость CFL: DG stripe с P2 при общем dt
2. Conservation на интерфейсе structured↔DG при несовпадении разрешения
3. Физическая корректность wall BC на top/bottom для DG stripe в открытом море

---

## 12. DG-солвер на тех же кейсах, что и MacCormack (сравнение)

**Проблема:** DG валидирован только на аналитических задачах (lake at rest, dam break, convergence). Нет ни одного прогона на реальных кейсах, которые уже посчитаны MacCormack'ом. Невозможно сравнить методы.

**Цель:** Прогнать чистый DG на тех же задачах, что и MacCormack. Получить сопоставимые результаты: max_eta, картину распространения, время расчёта.

| # | Задача | Статус |
|---|--------|--------|
| 12.1 | DG на плоском дне: 200×200 km, D=1000, Gaussian wave | ✅ `DGvsMacCormackTest()` создан, 100×100 mesh P2 vs 200×200 MacCormack |
| 12.2 | DG на S7 Japan Sea | ⏳ DG не поддерживает сферические координаты — только Cartesian |
| 12.3 | Сравнение DG vs MacCormack: max_eta, время расчёта | ⏳ Прогон запущен |
| 12.4 | Оценить CFL-ограничение DG на реальной батиметрии | ⏳ |
| 12.5 | Визуализация: side-by-side | ⏳ |
| 12.6 | Отчёт | ⏳ |

**Первый прогон (2026-04-06, RK2 + CFL=0.3):**
- MacCormack 200×200: 0.91s, max|eta|=0.104 m (stable)
- DG P2 100×100 (20K elements): 434s, max|eta|=**1016 m** (UNSTABLE)
- Причина: RK2 недостаточен для P2 DG, CFL=0.3 слишком высок

**Повторный прогон (RK3 + CFL=0.15):**
- MacCormack: 0.94s, max|eta|=0.104 m (stable)
- DG P2: 2013s, max|eta|=**4.85×10^10** (UNSTABLE)
- **Вывод:** DG diverges и с RK3 — проблема не в CFL, а в солвере (limiters, BC, или баг при длительных прогонах). Требует отдельного расследования.

**Запуск:** `MacCormackSimulation.exe dgvsmc`

**Риски:**
- DG на 4000×4000 может быть слишком тяжёл — начать с уменьшенной сетки (1000×1000), затем масштабировать
- DG пока не поддерживает сферические координаты — S7 кейс (12.2) возможен только в Cartesian, сравнивать с test 19
- dt для DG в ~5× меньше → время расчёта в ~5× больше при равном числе ячеек
- **P2 DG + RK2 нестабильно** — использовать только RK3 для P2

---

## 13. Починка coupling DG + MacCormack (гибрид)

**Проблема:** Test 30 (S7 hybrid) провалился — `BuildCouplings()` нашёл 0 интерфейсов. DG stripe (MeshPatch) работала изолированно с wall BC, eta разошлась до 1663 m.

**Причина:** `DoInterfacesMatch()` требует совпадения координат точек интерфейса. Граничные точки MeshPatch (рёбра DG-элементов) не совпадают с позициями ghost-ячеек StructuredPatch.

**Факт:** На мелком CouplingTest (60×60 + 30×30) coupling работает (mass error 0.0005%). На S7 (3601×4801, DG stripe 300×600) — нет. Значит баг в масштабировании или в генерации интерфейсных точек для больших сеток.

| # | Задача | Статус |
|---|--------|--------|
| 13.1 | Отладить `BuildCouplings()` — почему 0 интерфейсов | ✅ 3 бага: Initialize, right iface coord, write_index |
| 13.2 | Проверить координаты точек на обеих сторонах | ✅ delta_x(j) разная для разных широт — j_mid fix |
| 13.3 | Диагностический вывод | ✅ Добавлен, помог найти j=0 vs j_mid проблему |
| 13.4 | Исправить matching | ✅ Relative tolerance + find_x_at_y(median) |
| 13.5 | Повторный прогон Test 30 (v1) | ✅ Coupling OK (11/11), но DG diverged: max_eta=1699m. Причина: десинхрон по dt |
| 13.5b | Исправить десинхрон DG ↔ MacCormack | ✅ MeshPatch шагает на глобальный tau, auto-reduce tau до DG CFL |
| 13.5c | Повторный прогон Test 30 (v2, 300×600) | ❌ DG diverged: max_eta=1497m. Десинхрон dt исправлен, buffer aliasing исправлен, но DG blows up на шаге 5 |
| 13.5d | Диагностика DG divergence | ✅ Причина: wet/dry front на реальной батиметрии. D=[−848,1949] в stripe 3, min_H падает до 1e-6 на шаге 4, explosion на шаге 5. Clamping D≥8 не помогает — крутой батиметрический градиент создаёт source term gH∇D, который P2 DG на грубой mesh не держит |
| 13.5e | Wet/dry handling v1: semi-dry P0 + velocity cap | ✅ Стабильно, но coastal accumulation (max_eta=6261/3105) |
| 13.5f | Buffer aliasing fix (3 бага) | ✅ Incoming/outgoing буферы разделены |
| 13.5g | Absorbing BC attempt | ❌ Нарушает well-balanced, ухудшает стабильность |
| 13.5h | EnforceDryCells + zero-gradient BC (аналог MacCormack) | ✅ D < threshold → H=D, u=v=0. На интерфейсе: exterior=interior, momentum=0 |
| 13.5i | Codex review (P1: dx_inv j_mid, P2: boundary troubled) | ✅ |
| 13.5j | **Полный прогон Test 30 (v3, 150×300)** | ✅ **14228 с (~4 ч), 0 NaN, max_eta=2897 m** |
| 13.6 | Comparison Test 18 vs Test 30 — deep water | ⏳ |
| 13.7 | Отчёт с результатами и интерпретацией | ✅ `Result/test30_hybrid_s7_spherical/report.md` |

**Исправленные баги (2026-04-06):**
1. `WrapSubGrid()` не вызывал `StructuredPatch::Initialize()` → интерфейсы не строились
2. `BuildRightInterface` использовал `global_i_end_-1` → координата не совпадала с соседом
3. `CopyBufferToPatch` писал в `local_index` → перетирал вычислительные ячейки (добавлен `write_index` в `InterfacePoint`)
4. `BuildHybridStripesOneDG` вычислял x0 при j=0, а delta_x = 111300·cellsize·cos(lat) зависит от j → j_mid fix
5. `DoInterfacesMatch` использовал абсолютный tolerance 1e-3 и bbox average → relative tolerance + find_x_at_y(median)

**Верификация:** CouplingTest: mass error 0.0016%, stable. S7: 11/11 couplings created.

**Порядок:** 13.1-13.4 (починка) → 13.5 (прогон) → 12.2 (DG отдельно на S7) → 13.6 (сравнение трёх вариантов: pure MacCormack / pure DG / hybrid)

---

## Сквозные приоритеты (2026-04-10)

Согласовано с Codex (gpt-5.4). Таблица действует для всех разделов плана (1-19); внутри раздела порядок сохраняется.

| Приоритет | Задачи | Обоснование |
|-----------|--------|-------------|
| **P0 (сейчас)** | §19 flux-based coupling + unit conversion fix; acceptance tests conservation (mass + momentum) на DG↔MC интерфейсе | Блокер гибрида; без этого §18 и §11 бессмысленны |
| **P1 (далее)** | §20 Видео-верификация Кориолиса (Cart vs Sph для 41-42, 43-44, 45-46); §18 subcycling DG; повторный S7 hybrid с flux coupling; §13.6 pure MC / pure DG / hybrid сравнение на одном кейсе | Физ. верификация + масштаб гибрида |
| **P2** | §16 Disk Offload на Japan Sea (3601×4801, замер RAM и wall-time) | Нужно для статьи, не блокирует другие задачи |
| **P3** | §17.3 предвычисление $1/\Delta x(j)$; §17.4 профилирование VTune/perf; §12.2 DG на S7 Cartesian | Бесплатные оптимизации + расширение верификации DG |
| **P4** | §17.1 «буйковая» стратегия (триггер по потоку, не только по η); §17.2 SIMD (после профилирования) | Требует подтверждения бутылочного горла |
| **Заморожено** | §9 CUDA fix (блокер: CUDA 12.8 SDK); §14.2 DG на S7 Spherical (DG не умеет сферические координаты) | Внешние блокеры |

**Замечания Codex к оценкам ускорения:**
- §17.1 «буйковая» — 30-50% реалистично только для источников у края, триггер только по `|η|` рискован → добавить триггер по потоку $|Hu|, |Hv|$ или по приходу фронта
- §17.2 SIMD — **1.2-2× на ядро** (а не 2-4×) без серьёзной SoA/layout работы; memory-bound stencil
- §17.3 $1/\Delta x(j)$ — 5-15% правдоподобно, если тригонометрия сейчас в hot loop
- §18 subcycling — $dt_{MC}/dt_{DG}$ 3-5× возможно, но wall-time выигрыш меньше из-за стоимости DG и интерфейсного обмена
- §16 Disk Offload — это **экономия RAM ценой замедления**, не ускорение; I/O-доля <50% для 17M — гипотеза, нужна проверка

---

## Дополнения из PostGS_Implementation_Description (2026-04-10)

Задачи, которых нет в плане выше, но которые упомянуты в `Docs/PostGS_Implementation_Description.tex` как открытые направления. Линза Люнеберга и TsunamiSimUI уже реализованы — в план не выносятся.

### 16. Disk Offload на реальной акватории [P2]

**Статус:** алгоритм реализован в `PartionFileGridSimulator`, регрессия на малых сетках (120×120, 1000×1000) пройдена — побитовое совпадение с in-memory. Осталось прогнать на реальной батиметрии.

**Важно (Codex):** Disk Offload — это экономия RAM ценой замедления, не ускорение. Численная эквивалентность in-memory прогону должна быть явным требованием, не побочным эффектом.

| # | Задача | Статус |
|---|--------|--------|
| 16.1 | **Disk Offload на Japan Sea (3601×4801, ~17M ячеек)** — прогон до конца, замер wall-time и пикового RAM | ⏳ |
| 16.2 | Сравнить с in-memory прогоном S7 (~75 мин) — замедление (ожидание: 2-4× на NVMe) | ⏳ |
| 16.3 | **Численная эквивалентность in-memory прогону:** одинаковые ghost/halo, порядок обновления, CFL, wet/dry handling, метрики сферической сетки, σ(max|η-η_mem|) = 0 | ⏳ |
| 16.4 | I/O-доля при 17M ячеек (гипотеза: <50% на NVMe) — замерить, не постулировать | ⏳ |
| 16.5 | Профилирование: чтение / запись / сериализация / PCIe bandwidth — найти bottleneck | ⏳ |
| 16.6 | Бинарный формат страйпов (если сейчас текст) | ⏳ |
| 16.7 | Prefetch соседних страйпов — параллельная загрузка | ⏳ |
| 16.8 | Цифры для статьи: замедление, экономия RAM, I/O доля, hardware (SSD type, bandwidth) | ⏳ |

---

### 17. Дополнительные направления оптимизации (расширение пункта 8)

Упомянуты в tex (раздел «Оптимизация производительности»), но в плане пункта 8 отсутствуют:

| # | Задача | Оценка (Codex-скорректирована) | Приоритет | Статус |
|---|--------|--------|-----------|--------|
| 17.1 | **«Буйковая» стратегия** — отложенный запуск подсеток. Триггер **по потоку** $\|Hu\|, \|Hv\| > \varepsilon$ или по приходу фронта (не только по $\|\eta\|$, иначе отражения и шум пропускают порог) | 30-50% **для источников у края** | P4 | ⏳ |
| 17.2 | **SIMD-векторизация** (AVX2/AVX-512). Memory-bound stencil → без серьёзной SoA/layout работы выигрыш ограничен | **1.2-2× на ядро** (было 2-4×) | P4 | ⏳ |
| 17.3 | **Предвычисление $1/\Delta x(j)$** для сферических координат — массив при инициализации | ~5-15% (если тригонометрия сейчас в hot loop) | P3 | ⏳ |
| 17.4 | **Профилирование VTune/perf** — найти реальное бутылочное горло до SIMD/буёв | — (необходимо для P4) | P3 | ⏳ |

**Порядок (Codex):** сначала 17.3 и 17.4 → потом 17.1 → только затем 17.2 (SIMD без профилирования — дорогая догадка).

---

### 18. Масштабируемость DG-полосы в гибриде [P1]

**Проблема:** текущая DG-полоса в S7 гибриде ограничена 150×300 элементами из-за CFL. Описано в tex («Масштабируемость DG-полосы»), но в плане нет как отдельная задача.

**Важно (Codex):** subcycling сформулирован правильно, но корректность держится на **консервативном накоплении DG-потоков** через интерфейс за все substeps — иначе за каждый MC-шаг будет дисбаланс. Выигрыш wall-time будет меньше $dt_{MC}/dt_{DG}$ из-за стоимости DG и интерфейсного обмена.

| # | Задача | Статус |
|---|--------|--------|
| 18.1 | **Subcycling DG:** DG шагает $k$ раз с $dt_{DG} = dt_{MC}/k$ за один глобальный $dt_{MC}$ | ⏳ |
| 18.2 | Определить оптимальное $k = dt_{MC}/dt_{DG}$ для P2 DG (ожидается 3-5×, но реальный wall-time gain меньше) | ⏳ |
| 18.3 | **Консервативное накопление потоков:** интерфейсный поток MC ← $\frac{1}{k}\sum_{m=1}^{k} F^{(m)}_{DG}$ | ⏳ |
| 18.4 | Проверка консервативности: $\sum$ массы и импульса на интерфейсе = 0 на каждом MC-шаге | ⏳ |
| 18.5 | Адаптивная DG-сетка: крупнее в глубокой воде, мельче у побережья | ⏳ |
| 18.6 | Прогон S7 hybrid на DG-полосе большего размера (300×600 или 600×1200) | ⏳ |
| 18.7 | Wall-time замер: реальный gain vs теоретический $k$-кратный | ⏳ |

**Зависимости:** требуется §19 flux-based coupling для корректной консервативности при subcycling.

---

### 19. Flux-based coupling для гибрида [P0]

Упомянуто в блоке 13 в обновлении 2026-04-08 и описано в tex («Открытые проблемы → Flux-based coupling»). Главный блокер гибрида.

**Суть:** DG-HLL поток и конечно-разностный поток Мак-Кормака из одного состояния дают **разные** значения → дисбаланс массы и импульса на каждом шаге. Решение — обмениваться потоками, а не состояниями.

**Расширение (Codex):** нужно согласовывать не только массовый поток, но **весь вектор переменных**: масса, импульс (Hu, Hv), плюс учёт нормалей, длин рёбер, метрик/Jacobian при переходе со сферической на плоскую геометрию, source terms по батиметрии, positivity/wet-dry, lake-at-rest well-balanced режим. Всё это должно быть согласовано одновременно, иначе дисбаланс вылезет в другой компоненте.

| # | Задача | Статус |
|---|--------|--------|
| 19.1 | Экспорт интегрального потока DG через граничное ребро (массовый + импульсный) ($\sum F_{HLL} \cdot w_q$, [м³/с] и [м⁴/с²]) | ⚠️ частично (только масса) |
| 19.2 | Проекция DG-потока на грани структурированных ячеек с учётом нормалей и длин рёбер | ⏳ |
| 19.3 | **Корректная конверсия единиц:** DG [м³/с] → MC [м²/с] (деление на длину ребра $L_e$); для импульса аналогично | ❌ блокер |
| 19.4 | Метрики/Jacobian при переходе со сферической на плоскую геометрию на границе | ⏳ |
| 19.5 | Согласование source terms по батиметрии ($-gH\nabla D$) — одинаковая реконструкция $\nabla D$ на обеих сторонах | ⏳ |
| 19.6 | Использование внешнего потока в MacCormack вместо FD-аппроксимации (и в hat-полях, и в final step) | ⏳ |
| 19.7 | Сохранение positivity и wet/dry инвариантов через интерфейс | ⏳ |
| 19.8 | **Lake-at-rest well-balanced через интерфейс:** флэт η, u=v=0 → $\partial_t$ всего = 0 | ⏳ |
| 19.9 | Codex review математики конверсии (с выводом на бумаге) | ⏳ |

**Блокер:** unit mismatch в 19.3. До его решения §18 subcycling не имеет смысла.

**Acceptance tests (обязательны перед закрытием §19):**

| Тест | Метрика | Ожидание |
|------|---------|----------|
| T1. HybridFlatTest (120×120, 4 stripes, DG stripe 1, eta₀=1m) | max\|η\| | ≈ 0.098 (как pure MC), НЕ 3766 |
| T2. Mass conservation на интерфейсе | $\|\sum_{e} F_e^{mass} \cdot dt\|$ за прогон | < $10^{-10}$ от общей массы |
| T3. Momentum conservation (Hu, Hv) на интерфейсе | аналогично T2 | < $10^{-10}$ |
| T4. Lake-at-rest через hybrid interface | max\|η−η₀\|, max\|u\|, max\|v\| | 0 (машинный ноль) |
| T5. Dam break через DG↔MC интерфейс | $H_{max}$, позиция фронта | совпадение с pure MC и pure DG |
| T6. Wet/dry positivity при пересечении интерфейса | $\min H$ | $\geq 0$ на всём прогоне |
| T7. S7 hybrid с flux coupling | max\|η − η_pure_MC\| | < 1% от max η |
| T8. Отсутствие отражений на интерфейсе (gaussian, gentle passage) | амплитуда отражённой волны | < 1% от падающей |

---

### 20. Видео-верификация силы Кориолиса [P1]

**Мотивация:** Тесты 41-48 (φ=45°, 4000×4000, 8000 шагов) завершены, числовые результаты показывают +102% усиление для S→N плоской волны (§12 Кориолис v2). Индивидуальные видео `test{N}.mp4` есть, но **нет side-by-side сравнительных MP4 Cart vs Sph**, где видно, КАК именно движется волна под действием Кориолиса. Нужно увидеть эффект в динамике, не только в `max_eta`.

**Текущее состояние:**
- `Result/compare_test41_vs_test42/` — только PNG, MP4 нет (S→N plane, +102% — **самое важное**)
- `Result/compare_test43_vs_test44/` — только PNG, MP4 нет (W→E plane)
- `Result/compare_test45_vs_test46/` — только PNG, MP4 нет (Ellip NS)
- `Result/compare_test47_vs_test48/compare_47_vs_48.mp4` — ✅ есть (Ellip WE)
- Индивидуальные `test{41..48}/test{N}.mp4` — ✅ есть

| # | Задача | Статус |
|---|--------|--------|
| 20.1 | **compare_41_vs_42.mp4** — S→N plane Cart vs Sph (приоритет №1, +102% эффект) | ⏳ |
| 20.2 | compare_43_vs_44.mp4 — W→E plane Cart vs Sph | ⏳ |
| 20.3 | compare_45_vs_46.mp4 — Ellipsoidal NS Cart vs Sph | ⏳ |
| 20.4 | **Triple-panel видео:** Cart | Sph | Diff (η_Sph − η_Cart) для пары 41 vs 42 — чтобы эффект Кориолиса виден был отдельным каналом | ⏳ |
| 20.5 | Наложение u-field (антисимметрия NH/SH) на видео 41 vs 42 | ⏳ |
| 20.6 | Замер: позиция фронта на каждом шаге, отклонение вправо для S→N (NH) | ⏳ |
| 20.7 | Короткий отчёт (report.md) с ключевыми кадрами + анимированными GIF для статьи | ⏳ |

**Инструмент:** существующий `Code/Python/visualize_simulation.py --compare` (уже использовался для пары 47-48) + расширение для triple-panel режима.

**Блокер:** сырые `eta_timestep_*.txt` для тестов 41-48 — уже удалены (gitignore + disk cleanup). Если их нет на диске, нужен **повторный прогон** тестов 41-48 или восстановление из бэкапа. Проверить перед началом:
```bash
ls Result/test4{1,2}/eta_timestep_*.txt | head -5
```
Если пусто — добавить в §20.0 перезапуск:
| 20.0 | (при необходимости) Перезапуск MacCormackSimulation.exe 41..48 (~3 часа) | ⏳ |

---

### 21. Sommerfeld radiation + Sponge layer для Coriolis-тестов (2026-04-10)

**Проблема найдена при первом перезапуске test 42 с Sommerfeld BC:**
- Sommerfeld BC на всех 4 сторонах дал **тот же** max_eta_final=1.286, что и старый zero-gradient.
- Причина: волна в test 42 доминируется **Kelvin wave у восточной стенки** — плоская S→N волна + Кориолис (NH) создаёт `u>0`, частицы упираются в east wall, формируется прибрежная Kelvin wave, бегущая на север вдоль стенки с накоплением амплитуды.
- 1D Sommerfeld `∂_t φ + c·∂_n φ = 0` бессилен против **тангенциально распространяющихся** волн (cos(90°)≈0 → σ_eff=0, эквивалентно Neumann/zero-gradient).
- Spatial analysis: max находится на самом `i=3999` (east wall), движется с юга на север во времени (t=400: j=198, t=8000: j=1992) — это **физический Kelvin mode**, но его амплитуда преувеличена идеализированной вертикальной стенкой.

**Решение (consulted with Codex):** Sponge layer + interior метрика.

| # | Задача | Статус |
|---|--------|--------|
| 21.1 | `SingleLineScenario::sponge_{west,east,south,north}_cells` + `sponge_sigma_max` | ✅ |
| 21.2 | `SingleLineGridSimulation::ApplySpongeLayer()` — quadratic ramp `sigma = sigma_max * (r/W)²`, damping `q*=exp(-σ*dt)` к η, u, v, H=D+η | ✅ |
| 21.3 | Applied после `ExchangeBoundaryData` каждого шага; только к реальным ячейкам (не ghost) | ✅ |
| 21.4 | Раздельная ширина для 4 сторон: источник у южной стенки → sponge_south=0, иначе сжигает волну | ✅ |
| 21.5 | Interior metric: `max_eta_interior` в окне `[i_min..i_max)×[j_min..j_max)`, исключающем sponge. 300 ячеек отступ для 4000×4000 | ✅ |
| 21.6 | Verification test 101 (lake-at-rest, sponge на всех 4 сторонах): `max_eta = 0` (machine zero) | ✅ |
| 21.7 | Verification test 102 (gaussian pulse, sponge на всех 4 сторонах): `max_eta_final = 0.036` (3.6% от 1.0) vs 0.08 без sponge | ✅ |
| 21.8 | Test 42 с sponge E/W/N (без S — источник у j=0-10): проверить `max_eta_interior` | ⏳ |
| 21.9 | Batch 41-48 с индивидуальной конфигурацией sponge по сценариям | ⏳ |

**Sponge конфигурация (по рекомендации Codex):**
- σ_max = 0.002 s⁻¹ (дает `exp(-σ·2020s) ≈ e⁻⁴` для волны, проходящей 200-ячеечный sponge за 2020 s при c=99 m/s)
- Ширина 200 ячеек = 200 км для 4000×4000 домена
- Quadratic ramp: σ(0)=0 у внутреннего края, σ(W)=σ_max у стенки

**Индивидуальная настройка sponge для 41-48:**

| Тест | Источник | sponge W/E/S/N | Обоснование |
|------|----------|----------------|-------------|
| 41, 42 | Plane S→N, j=[0,10) | 200/200/**0**/200 | Источник у S стенки, нельзя её трогать; Kelvin у E; N для ухода фронта |
| 43, 44 | Plane W→E, i=[0,10) | **0**/200/200/200 | Источник у W стенки; Kelvin у S (в NH для W→E) |
| 45, 46 | Эллипс 50×200km центр | 200/200/200/200 | Источник в центре, sponge на всех сторонах |
| 47, 48 | Эллипс 200×50km центр | 200/200/200/200 | Аналогично 45/46 |

**Acceptance criteria для test 42:**
- Global `max_eta_final` может остаться ~1.28 (Kelvin wave на самой стенке в ghost cell)
- **`max_eta_interior` должен упасть** до 0.65-0.9 (ожидание Codex после устранения Kelvin artifact)
- Если `max_eta_interior ≈ 0.637` (= test 41 Cartesian без Кориолиса) → Кориолис-эффект на η при плоской волне S→N практически ноль (только u-field показывает)
- Если `max_eta_interior > 1.0` — остался необъяснённый эффект, нужна дальнейшая диагностика

**Результаты всех 8 тестов (2026-04-11):**

| Тест | Тип | Coord | max_eta_interior | Coriolis Δ |
|------|-----|-------|------------------|------------|
| 41 | Plane S→N | Cartesian | **0.500** | baseline |
| 42 | Plane S→N | Spherical | **0.795** | **+59%** |
| 43 | Plane W→E | Cartesian | 0.0027 | baseline |
| 44 | Plane W→E | Spherical | 0.0024 | ≈0 (−13%) |
| 45 | Ellip N-S | Cartesian | 0.0078 | baseline |
| 46 | Ellip N-S | Spherical | 0.0116 | +50% (но на уровне шума) |
| 47 | Ellip W-E | Cartesian | 0.0023 | baseline |
| 48 | Ellip W-E | Spherical | 0.0064 | +178% (абсолют 0.004 m) |

**Интерпретация (Codex-aligned):**

1. **Plane S→N (41 vs 42): +59% — значимый, реальный Coriolis эффект.**
   - В NH Coriolis отклоняет частицы вправо, создаёт восточный поток `u>0`, энергия **концентрируется в interior** вдоль оси волны, амплитуда η растёт.
   - Прошлый результат +102% был артефактом: большая часть усиления шла от отражённой Kelvin wave у идеализированной восточной стенки. После введения sponge layer 200 cells и измерения в interior-окне (300..3700 × 300..3700) остаётся чистые +59%.

2. **Plane W→E (43 vs 44): ≈0 — эффект Coriolis на амплитуду отсутствует.**
   - Coriolis создаёт только меридиональный отклик `v<0` (юг в NH), который не влияет на η плоской волны.
   - Обе волны (Cart и Sph) просто уходят на восток через sponge, к шагу 8000 в interior почти ничего не остаётся (~0.002 m — шум).

3. **Ellipsoidal sources (45-48): эффект на η практически отсутствует.**
   - Абсолютные значения interior < 0.012 m — на уровне численного шума после разлёта кольца
   - Эллипсоидный источник генерирует кольцевой фронт, и Coriolis создаёт разные локальные отклики для разных частей кольца, которые частично компенсируются при усреднении

**Главный вывод:** В реалистичной постановке (flat 1000 m бассейн 4000×4000 км, источник amp=1 m, φ=45°, 11 часов) **Coriolis эффект на амплитуду волны значим только для плоской S→N волны, где даёт +59% усиления**. Для W→E и эллипсоидных источников эффект на η незаметен на уровне 0.01 m.

**Видео:**
- [Result/test41/test41.mp4](Result/test41/test41.mp4) — S→N Cart (baseline)
- [Result/test42/test42.mp4](Result/test42/test42.mp4) — S→N Spherical (Coriolis)
- [Result/test43/test43.mp4](Result/test43/test43.mp4), [test44.mp4](Result/test44/test44.mp4) — W→E пара
- [Result/test45/test45.mp4](Result/test45/test45.mp4), [test46.mp4](Result/test46/test46.mp4) — Ellip NS пара
- [Result/test47/test47.mp4](Result/test47/test47.mp4), [test48.mp4](Result/test48/test48.mp4) — Ellip WE пара

**Сравнительные видео (side-by-side Cart vs Sph, 200 кадров, 15 fps):**
- [Result/compare_test41_vs_test42/compare_test41_vs_test42.mp4](Result/compare_test41_vs_test42/compare_test41_vs_test42.mp4) — **главное видео: S→N +59% Coriolis**
- [Result/compare_test43_vs_test44/compare_test43_vs_test44.mp4](Result/compare_test43_vs_test44/compare_test43_vs_test44.mp4)
- [Result/compare_test45_vs_test46/compare_test45_vs_test46.mp4](Result/compare_test45_vs_test46/compare_test45_vs_test46.mp4)
- [Result/compare_test47_vs_test48/compare_test47_vs_test48.mp4](Result/compare_test47_vs_test48/compare_test47_vs_test48.mp4)

**Tooling:**
- [Code/Python/viz_coriolis_v2.py](Code/Python/viz_coriolis_v2.py) — single-pass визуализатор (PNG + MP4) для 200 кадров 4000×4000 без OOM
- [Code/Python/compare_frames.py](Code/Python/compare_frames.py) — side-by-side компоновка из уже готовых PNG (не требует сырых eta файлов)
- [Result/run_coriolis_v2_batch.sh](Result/run_coriolis_v2_batch.sh) — pipeline скрипт run → viz → clean для batch 43-48
