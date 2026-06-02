# Инструкции для Claude Code — tsunami-core

## Build & Test
- Сборка: открыть `MacCormackSimulation.sln` в VS2022, `Release|x64` (CPU+CUDA).
  Только I/O-таргет: `cmake -B build` (`NskTSH::tsunami-io`).
- Валидация физики: lake-at-rest (η≈0), `dgval` (well-balanced), массовый баланс.
- CUDA-тесты — только на машине с NVIDIA GPU; иначе compile-smoke.

## Конвенция путей (ОБЯЗАТЕЛЬНО)
- Данные: `$NSKTSH_DATA` (default `../Data` — папка уровнем выше репо).
- Результаты: `$NSKTSH_RESULTS` (default `../Results`).
- НЕ хардкодить `F:\` и абсолютные пути.
- Всегда передавать `--threads=N` через CLI (scenario.num_threads перекрывает OMP_NUM_THREADS).

## Не трогать
- Артефакты сборки (`x64/`, `.vs/`, `*.obj/pdb`), крупные данные/результаты, секреты.
- Минимальный набор изменений > крупный рефактор. Читать существующий код перед правкой.

## Workflow
- Не коммитить/пушить без явной просьбы.
- Перед завершением: build → test/валидация → diff summary → риски.
- После расчётов/бенчмарков — независимая проверка физики/алгоритмов через Codex (no-sandbox full-auto).
- Общение по-русски, код и комментарии на английском.
