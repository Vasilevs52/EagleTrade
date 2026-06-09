# =====================================================================
# MAIN2 — точка входа «честной эволюции» v2 (анти-переобучение)
# ---------------------------------------------------------------------
# Отдельный entrypoint (не трогает v1 main.py):
#   1 — запустить эволюцию v2 (N параллельных процессов)
#   2 — показать сохранённые v2-стратегии (train / val / gap)
#   3 — оценить выбранную стратегию на ФИНАЛЬНОМ HOLDOUT (1 раз!)
#   4 — шумовой контроль: baseline fitness на случайных данных
#   5 — обновить кеш данных (force-перекачка с Binance)
#
# Запуск: cd project/stvgp && python main2.py
# =====================================================================

import os
import random
import multiprocessing as mp

from config import CFG2
from data_loader import load_v2_dataset
from storage import (
    append_result_locked, load_records, record_to_hofs, RESULTS_FILE_V2,
)

NUM_PROC = int(os.environ.get("EAGLE_PROCS", os.cpu_count() or 1))


def _run_v2_process(seed: int, result_queue: mp.Queue):
    """Целевая функция процесса: эволюция v2 + инкрементальное сохранение."""
    try:
        from stvgp import evolve_robust
        ds = load_v2_dataset()  # из кеша (родитель прогрел до спавна)
        hofs, info = evolve_robust(ds["train"], ds["val"], seed=seed)
        rec = {
            "seed": seed,
            # fitness = ВАЛИДАЦИОННЫЙ скор -> сортировка по валидации
            "fitness": info["val_fit"],
            "best_long_str":  str(hofs[0][0]),
            "best_short_str": str(hofs[1][0]),
            "best_meta_str":  str(hofs[2][0]),
            "best_long_fit":  hofs[0][0].fitness.values[0],
            "best_short_fit": hofs[1][0].fitness.values[0],
            "best_meta_fit":  info["train_fit"],
            "train_fit": info["train_fit"],
            "val_fit": info["val_fit"],
            "gap": info["gap"],
            "v2": True,
            "assets": list(ds["assets"]),
            "interval": ds.get("interval", CFG2.interval),
        }
        append_result_locked(rec, filename=RESULTS_FILE_V2)
        result_queue.put(rec)
        print(f"[v2 seed={seed}] ГОТОВО: val={info['val_fit']:.2f} "
              f"train={info['train_fit']:.2f} gap={info['gap']:.2f}")
    except Exception as exc:
        import traceback
        print(f"[v2 seed={seed}] ERROR: {exc}")
        traceback.print_exc()
        result_queue.put(None)


def run_evolution(n_processes=None):
    """Режим 1: параллельная эволюция v2."""
    n_processes = n_processes or NUM_PROC
    load_v2_dataset()  # прогреваем кеш ДО спавна (дети читают с диска)

    seeds = random.sample(range(1, 2**30), n_processes)
    q = mp.Queue()
    procs = [mp.Process(target=_run_v2_process, args=(s, q),
                        name=f"v2-seed-{s}") for s in seeds]
    print("=" * 70)
    print(f"V2: запуск {n_processes} параллельных эволюций, seeds={seeds}")
    print("=" * 70)
    for p in procs:
        p.start()
    # Читаем очередь ДО join (см. main.py — иначе deadlock на больших объектах)
    results = []
    for _ in procs:
        r = q.get()
        if r is not None:
            results.append(r)
    for p in procs:
        p.join()

    if not results:
        raise RuntimeError("Все v2-процессы упали (см. traceback выше).")

    results.sort(key=lambda r: r["val_fit"], reverse=True)
    print("\n" + "=" * 70)
    print("ИТОГИ V2 (отсортировано по ВАЛИДАЦИИ)")
    print("=" * 70)
    print(f"{'seed':>11} {'val':>8} {'train':>8} {'gap':>8}")
    for r in results:
        print(f"{r['seed']:>11} {r['val_fit']:>8.2f} "
              f"{r['train_fit']:>8.2f} {r['gap']:>8.2f}")
    print("=" * 70)
    print("gap = train − val: большой gap у стратегии = она переобучена.")


def show_saved(top=20):
    """Режим 2: таблица сохранённых v2-стратегий."""
    records = load_records(RESULTS_FILE_V2)
    if not records:
        print(f"{RESULTS_FILE_V2} пуст — сначала запустите эволюцию (1).")
        return
    print("=" * 70)
    print(f"V2-СТРАТЕГИИ ({RESULTS_FILE_V2}) — всего {len(records)}, "
          f"сортировка по валидации")
    print("=" * 70)
    print(f"{'#':>3} {'seed':>11} {'val':>8} {'train':>8} {'gap':>8}")
    for i, r in enumerate(records[:top]):
        print(f"{i:>3} {r.get('seed'):>11} {r.get('val_fit', 0):>8.2f} "
              f"{r.get('train_fit', 0):>8.2f} {r.get('gap', 0):>8.2f}")
    print("=" * 70)
    print("ВНИМАНИЕ: val лучшей записи оптимистичен («проклятие победителя»:")
    print("много кандидатов оценено на одних val-окнах). Истинная проверка —")
    print("HOLDOUT (меню 3), и только один раз.")


def holdout_eval():
    """
    Режим 3: финальная проверка на HOLDOUT — последних holdout_days,
    которые НЕ участвовали ни в обучении, ни в отборе.

    ВАЖНО: holdout честен только при однократном использовании. Если
    гонять по нему много стратегий и выбирать лучшую — он превращается
    в ещё одну train-выборку.
    """
    from signals import quick_profit_summary

    records = load_records(RESULTS_FILE_V2)
    if not records:
        print(f"{RESULTS_FILE_V2} пуст — сначала запустите эволюцию (1).")
        return
    show_saved()
    raw = input(f"№ стратегии для HOLDOUT [0..{len(records)-1}] (Enter=0): ").strip()
    idx = int(raw) if raw.isdigit() else 0
    idx = max(0, min(idx, len(records) - 1))
    rec = records[idx]
    hofs = record_to_hofs(rec)

    ds = load_v2_dataset()
    print(f"\nHOLDOUT: последние {CFG2.holdout_days} дней, "
          f"{len(ds['holdout'])} активов. Стратегия #{idx} "
          f"(seed={rec.get('seed')}, val={rec.get('val_fit', 0):.2f})")
    total_bot, total_bh = [], []
    for sym, bars in ds["holdout"]:
        bot, bh = quick_profit_summary(bars, hofs, label=f"HOLDOUT {sym}")
        if bot is not None:
            total_bot.append(bot)
            total_bh.append(bh)
    if total_bot:
        n = len(total_bot)
        print("\n" + "=" * 60)
        print(f"HOLDOUT ИТОГО ({n} активов): "
              f"бот {sum(total_bot)/n:+.2f}% ср., "
              f"buy&hold {sum(total_bh)/n:+.2f}% ср.")
        print("=" * 60)


def noise_baseline(pop=None, off=None, gens=None):
    """
    Режим 4: шумовой контроль. Запускает ту же эволюцию на СЛУЧАЙНЫХ
    данных (random walk с волатильностью реальных окон). Результат —
    «сколько fitness GP выжимает из чистого шума». Реальная стратегия
    обязана быть значимо лучше этой базовой линии, иначе она — шум.

    Для честного сравнения параметры (pop/gens) должны совпадать с боевыми.
    """
    import numpy as np
    import pandas as pd
    from data_loader import add_indicators, build_input_vectors, normalize_segment
    from stvgp import evolve_robust
    from config import CFG

    pop = pop or CFG2.pop_size
    off = off or CFG2.offspring
    gens = gens or CFG2.ngen

    ds = load_v2_dataset()
    rng = np.random.default_rng(20260607)

    def make_noise_window(template_seg):
        """Шумовое окно random-walk с волатильностью реального окна."""
        prices = [b["cur"] for b in template_seg]
        rets = np.diff(prices) / np.asarray(prices[:-1])
        std = float(rets.std()) if len(rets) > 1 else 0.01
        n = len(template_seg) + 2 * CFG.window + 2  # прогрев индикаторов
        p = [float(prices[0])]
        for _ in range(n - 1):
            p.append(p[-1] * (1 + rng.normal(0, std)))
        df = pd.DataFrame({
            "Open Time": pd.date_range("2020-01-01", periods=n, freq="h"),
            "Price": p,
        })
        df = add_indicators(df)
        sb = build_input_vectors(df)[:len(template_seg)]
        # та же нормализация, что и у реальных данных
        return normalize_segment(sb) if CFG2.normalize else sb

    n_val = max(2, len(ds["train"]) // 6)
    print(f"Генерирую {len(ds['train'])} train + {n_val} val шумовых окон "
          f"(random walk, волатильность как у реальных)...")
    synth_train = [make_noise_window(seg) for seg in ds["train"]]
    # ОТДЕЛЬНЫЕ val-окна (другие розыгрыши шума), а не подмножество train —
    # иначе val на шуме был бы завышен (валидация на обучающих данных).
    synth_val = [make_noise_window(ds["train"][i % len(ds["train"])])
                 for i in range(n_val)]

    print(f"Эволюция на шуме: pop={pop}, ngen={gens} (займёт как боевой прогон)...")
    _, info = evolve_robust(synth_train, synth_val,
                            seed=999, pop_size=pop, offspring=off, ngen=gens)
    print("\n" + "=" * 70)
    print(f"БАЗОВАЯ ЛИНИЯ НА ШУМЕ: train={info['train_fit']:.2f}, "
          f"val={info['val_fit']:.2f}")
    print("Если train-fitness реальных стратегий не сильно выше этой цифры —")
    print("эволюция находит шум, а не сигнал.")
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    print("=" * 70)
    print("ЧЕСТНАЯ ЭВОЛЮЦИЯ V2 (анти-переобучение)")
    print(f"Активы: {', '.join(CFG2.assets)}  |  interval={CFG2.interval}")
    print(f"История: {CFG2.history_days}д + holdout {CFG2.holdout_days}д; "
          f"окна {CFG2.window_bars} баров "
          f"(train {CFG2.n_train_windows}/актив, val {CFG2.n_val_windows}/актив)")
    print("=" * 70)
    print("  1 — Запустить эволюцию v2 (параллельно, отбор по валидации)")
    print("  2 — Показать сохранённые v2-стратегии (train/val/gap)")
    print("  3 — Финальный HOLDOUT-тест выбранной стратегии (использовать 1 раз!)")
    print("  4 — Шумовой контроль (baseline fitness на случайных данных)")
    print("  5 — Обновить кеш данных (force)")
    choice = input("Ваш выбор [1-5] (Enter = 1): ").strip()

    if choice == "2":
        show_saved()
    elif choice == "3":
        holdout_eval()
    elif choice == "4":
        noise_baseline()
    elif choice == "5":
        load_v2_dataset(force=True)
    else:
        raw = input(f"Сколько процессов [{NUM_PROC}] (Enter = {NUM_PROC}): ").strip()
        n = int(raw) if raw.isdigit() and int(raw) > 0 else NUM_PROC
        run_evolution(n)
