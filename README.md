# Fly-NCA — $f(\text{Природа}) = \text{NCA-Мух}$

Композиционная реализация *Drosophila melanogaster* на клеточных автоматах (NCA), напрямую выражающая генетику развития, природную топологию нейропилей, глиальную компартментализацию и 3D-биомеханику мухи по данным полного коннектома ЦНС (`BANC v888`, `MCNS v1.0`, `FAFB v783`).

---

## Проектная документация (`design/`)

1. **[Архитектурная спецификация (`design/architecture.md`)](design/architecture.md)**  
   Полное устройство `NCA-Муха`:
   - Трёхуровневое разделение: Оптический радар (`OL`, `97.7%` внутри) + Когнитивный штурман (`CB`: `LH-SPIN`, `MB`, `CX`) + Бортовой 200-Гц автопилот стабилизации полёта, подвеса головы и шага 6 лапок (`VNC`);
   - Слой `L0–L1`: `NCA-Геном`, две волны нейрогенеза (`Larva 3k -> 20E -> Adult 188k`), `Hox × 17 shTF × Notch × fru/dsx`, закон одного медиатора на гемилинию и самосборка связей по замкам `DIP ↔ Dpr` и `Frazzled/shak-B`;
   - Слой `L2`: 5 биологических портов клетки (`a-d ACh+`, `a-d GABA/GluClα` шунт, `a-a` пресинаптический порт `25.8%`, `Gap Junction`, `Glia/DCV`);
   - Слой `L3`: 6 природных топологий органов в композиции (`OL`, `AL/AMMC/SEZ`, `LH-SPIN` мультимодальный дендритный гейт, `MB` кабель сквозь 15 муфт + `DN->DAN` эфферентная копия, `CX` бублик `16` + веер `9×9` с прыжком `hΔ 180°` + сон `dFB/ER5`, `VNC` автопилот `subsumption`);
   - Слой `L4`: 3D-тело (`FlyBodyAdapter`) и сменные 3D-арены (`FastTensorArena3D` на PyTorch/CUDA и мост к `FlyGym/MuJoCo` / `Unity`);
   - План поэтапной сборки и лабораторных проверок (`M1–M5`).

2. **[Алгоритмическая и математическая спецификация (`design/algorithmic-spec.md`)](design/algorithmic-spec.md)**  
   Точные размерности тензоров, уравнения обновления 8-канальной клетки и 5 портов, псевдокод двухволнового морфогенеза из генома, математика каждого из 6 NCA-органов и дерево исходного кода `src/fly_nca/`.

3. **[Лабораторный стенд и натурные протоколы (`LABORATORY.md`)](LABORATORY.md)**  
   Самодостаточное описание 9 лабораторных установок (`LaboratoryPreset`), панели из 36 приборов (`fly_nca/instruments/`) и 9 протоколов верификации по опубликованным экспериментальным статьям (`Оригинальная статья -> Постановка ровно как в статье -> Сверка с результатом`).

---

## Установка окружения с чистого клона и быстрый запуск лабораторного стенда

Требуется **Python 3.11+** и **Git LFS** (для подтягивания `data/drosophila_pan_connectome_189766.pt`).

```bash
# 0. С чистого клона: подтянуть LFS-файлы коннектома, создать виртуальное окружение и поставить зависимости
git lfs pull
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt    # устанавливает torch, numpy, pyarrow, pandas, pytest и пакет fly-nca (-e .)

# 1. Список всех сьютов и натурных экспериментальных протоколов (работает даже без установленного torch/pyarrow)
python3 verify_principles.py --list
python3 -m fly_nca.app list

# 2. Полный прогон всех проверок (чистота модели + 9 натурных протоколов + 36 приборов + слепой холдаут Shiu 2024)
python3 verify_principles.py --ship

# 3. Запуск отдельного натурного протокола или снятие панели приборов
python3 -m fly_nca.app run pmc9213513_mb_gamma1_shibire
python3 -m fly_nca.app panel pmc13518903_vt062617_wind_tunnel --steps 4

# 4. Полный набор тестов M1–M7
pytest -q
```

---

## Данные коннектомов и сборка с нуля (`data/` и `tools/`)

В папке `data/` (через Git LFS) уже лежат скомпилированный 5-портовый коннектом (`data/drosophila_pan_connectome_189766.pt`, `189,766` клеток и `6,853,423` синаптических ребра) и проверочные таблицы метаданных (`banc_888_meta.feather`, `banc_888_meta_20260521.parquet`, `shiu_et_al_2025_fafb_mapped_opto_and_model_results.csv`), поэтому все тесты и `python verify_principles.py --ship` работают сразу после клонирования репозитория.

Если нужно скачать исходные первичные таблицы из открытых репозиториев (`BANC v888`, `MaleCNS v1.0 / v0.9`, `FAFB v783`, `Fly Cell Atlas`) и пересобрать `drosophila_pan_connectome_189766.pt` с нуля:

```bash
# Проверить статус локальных файлов:
python tools/download_connectomes.py --check

# Скачать базовый комплект для сборки 189,766-клеточного коннектома (или --tier full для полного архива 68 ГБ):
python tools/download_connectomes.py --tier build --data-dir ./data/casa-data

# Пересобрать data/drosophila_pan_connectome_189766.pt из сырых таблиц:
python tools/build_pan_connectome_189766.py
```
