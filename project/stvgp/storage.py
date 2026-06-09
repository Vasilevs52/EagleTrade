# =====================================================================
# STORAGE — сохранение/загрузка лучших индивидов эволюции
# Каждый прогон (в т.ч. из N параллельных) пишет своего лучшего
# индивида в JSON. Потом стратегию можно загрузить и запустить без эволюции.
# =====================================================================

import json
import os

from deap import gp, creator
from filelock import FileLock

# Импорт stvgp создаёт классы creator.LongIndividual / ShortIndividual /
# MetaIndividual (с прикреплённым fitness) и наборы примитивов.
import stvgp  # noqa: F401  (нужен ради побочного эффекта — creator.create)
from primitives import pset_long, pset_short, pset_meta

RESULTS_FILE = "evolution_results.json"
# v2 («честная эволюция»): в записи fitness = ВАЛИДАЦИОННЫЙ скор, поэтому
# сортировка и load_best_hofs автоматически выбирают лучшее по валидации.
RESULTS_FILE_V2 = "evolution_results_v2.json"


def _record_from_result(r):
    """Извлекает сохраняемые поля из результата одного прогона."""
    rec = {
        "seed":           r.get("seed"),
        "fitness":        r.get("fitness"),
        "best_long_str":  r.get("best_long_str"),
        "best_short_str": r.get("best_short_str"),
        "best_meta_str":  r.get("best_meta_str"),
        "best_long_fit":  r.get("best_long_fit"),
        "best_short_fit": r.get("best_short_fit"),
        "best_meta_fit":  r.get("best_meta_fit"),
    }
    # Доп. поля v2 (train/val/gap и контекст данных) — пропускаем как есть.
    for opt in ("train_fit", "val_fit", "gap", "v2", "assets", "interval"):
        if opt in r:
            rec[opt] = r[opt]
    return rec


def append_result_locked(result, filename=RESULTS_FILE):
    """
    Потокобезопасно ДОПИСЫВАЕТ один результат в JSON под файловой блокировкой.

    Зачем: при долгом параллельном прогоне (N процессов x ngen поколений)
    каждый процесс сохраняет свой результат СРАЗУ по завершении, а не все
    скопом в конце. Если сервер/SSH/процесс упадёт — уже готовые стратегии
    не теряются.

    FileLock сериализует доступ: 16 процессов не перезапишут файл друг друга.
    Дедуп по (seed, best_meta_str) — повтор не плодит дубли.
    """
    rec = _record_from_result(result)
    lock = FileLock(filename + ".lock")
    with lock:
        records = []
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, OSError):
                records = []

        # Дедуп по (seed, best_meta_str): новая запись заменяет старую.
        merged = {(r.get("seed"), r.get("best_meta_str")): r for r in records}
        merged[(rec.get("seed"), rec.get("best_meta_str"))] = rec
        records = list(merged.values())
        records.sort(
            key=lambda x: (x.get("fitness") is not None, x.get("fitness", float("-inf"))),
            reverse=True)

        # Атомарная запись: пишем во временный файл, потом заменяем — чтобы
        # при сбое в момент записи не осталось «обрезанного» JSON.
        tmp = filename + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp, filename)
    return filename


# ---------------------------------------------------------------------
# СОХРАНЕНИЕ
# ---------------------------------------------------------------------

def save_results(all_results, filename=RESULTS_FILE, append=True):
    """
    Сохраняет лучших индивидов в JSON.

    all_results — список словарей, каждый описывает результат одного прогона:
      {
        "seed": int,
        "fitness": float,            # итоговый (meta) фитнес
        "best_long_str":  str,
        "best_short_str": str,
        "best_meta_str":  str,
        "best_long_fit":  float,     # опционально
        "best_short_fit": float,     # опционально
        "best_meta_fit":  float,     # опционально
      }

    append=True — дописывает к уже накопленным в файле записям.
    """
    records = []
    if append and os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            records = []

    new_records = [_record_from_result(r) for r in all_results]

    # Дедупликация по seed: новый прогон с тем же seed ЗАМЕНЯЕТ старую запись
    # (раньше append копил дубли — один seed дважды давал 2 одинаковые записи,
    # файл рос бесконечно). Идентичность стратегии определяется seed + текстом
    # деревьев; для надёжности ключ — (seed, best_meta_str).
    def _key(rec):
        return (rec.get("seed"), rec.get("best_meta_str"))

    merged = {}
    for rec in records + new_records:   # new_records позже -> перезаписывают
        merged[_key(rec)] = rec
    records = list(merged.values())

    # Сортируем по итоговому фитнесу (лучшие сверху)
    records.sort(key=lambda x: (x.get("fitness") is not None, x.get("fitness", float("-inf"))),
                 reverse=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"  → сохранено индивидов в {filename}: {len(records)} (добавлено {len(all_results)})")
    return filename


# ---------------------------------------------------------------------
# ЗАГРУЗКА
# ---------------------------------------------------------------------

def load_records(filename=RESULTS_FILE):
    """Читает все записи из файла. Возвращает список словарей (отсортирован по fitness)."""
    if not os.path.exists(filename):
        return []
    with open(filename, "r", encoding="utf-8") as f:
        records = json.load(f)
    records.sort(key=lambda x: (x.get("fitness") is not None, x.get("fitness", float("-inf"))),
                 reverse=True)
    return records


def _rebuild_individual(tree_str, pset, individual_cls, fitness=None):
    """Восстанавливает индивида из строкового представления дерева."""
    tree = gp.PrimitiveTree.from_string(tree_str, pset)
    ind = individual_cls(tree)
    if fitness is not None:
        ind.fitness.values = (float(fitness),)
    return ind


def record_to_hofs(record):
    """
    Превращает одну запись JSON в тройку «hall of fame»:
      (hof_long, hof_short, hof_meta), где каждый — список из одного индивида.
    Совместимо с validate_on_new_data / compare_strategies / plot_signals.
    """
    long_ind = _rebuild_individual(
        record["best_long_str"], pset_long, creator.LongIndividual,
        record.get("best_long_fit", record.get("fitness")))
    short_ind = _rebuild_individual(
        record["best_short_str"], pset_short, creator.ShortIndividual,
        record.get("best_short_fit", record.get("fitness")))
    meta_ind = _rebuild_individual(
        record["best_meta_str"], pset_meta, creator.MetaIndividual,
        record.get("best_meta_fit", record.get("fitness")))
    return [long_ind], [short_ind], [meta_ind]


def load_best_hofs(filename=RESULTS_FILE):
    """
    Загружает лучшую (по fitness) сохранённую стратегию.
    Возвращает (hofs, record) либо (None, None), если файл пуст/отсутствует.
    """
    records = load_records(filename)
    if not records:
        return None, None
    best = records[0]
    return record_to_hofs(best), best


def print_saved_summary(filename=RESULTS_FILE, top=10):
    """Печатает таблицу сохранённых стратегий."""
    records = load_records(filename)
    if not records:
        print(f"Файл {filename} пуст или не найден.")
        return
    print("=" * 70)
    print(f"СОХРАНЁННЫЕ СТРАТЕГИИ ({filename}) — всего {len(records)}")
    print("=" * 70)
    print(f"{'#':>3}  {'seed':>12}  {'meta fit %':>12}")
    print("-" * 70)
    for i, r in enumerate(records[:top]):
        seed = r.get("seed")
        fit = r.get("fitness")
        fit_s = f"{fit:12.2f}" if isinstance(fit, (int, float)) else f"{'?':>12}"
        seed_s = f"{seed:12d}" if isinstance(seed, int) else f"{str(seed):>12}"
        print(f"{i:>3}  {seed_s}  {fit_s}")
    print("=" * 70)
