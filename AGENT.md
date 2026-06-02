# Инструкции для агентов — tsunami-core

## Назначение
SWE-решатель (MacCormack + DG/FV-гибрid), CPU (OpenMP) и CUDA. Ядро симуляции NskTSH.

## Как верифицировать
- Сборка: `MacCormackSimulation.sln` (`Release|x64`); I/O-таргет: `cmake -B build`.
- Валидация: lake-at-rest η≈0; `dgval` (well-balanced); сохранение массы/энергии.
- Независимая проверка физики/алгоритмов — Codex (no-sandbox full-auto).

## Границы
- Не выходить за пределы C++/CUDA-решателя.
- Большие данные/результаты — через конвенцию путей (`$NSKTSH_DATA`/`$NSKTSH_RESULTS`), НЕ в git.
- Не трогать секреты, migrations, build-артефакты.

## Чек-лист перед PR
- [ ] Сборка зелёная (CPU; CUDA — compile-smoke или GPU)
- [ ] Валидация пройдена (lake-at-rest / dgval / баланс)
- [ ] Нет `_*`-мусора/артефактов/секретов/абсолютных путей
- [ ] README/CLAUDE обновлены при смене API/команд
