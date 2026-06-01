# =====================================================================
# MAIN — точка входа: загрузка данных, параллельная эволюция, валидация
# =====================================================================

import random
import multiprocessing as mp

import matplotlib.pyplot as plt

from config import CFG
from data_loader import BinanceBroker, load_period
from primitives import pset_long, pset_short, pset_meta
from signals import (
    TradingSimulator,
    compare_strategies,
    quick_profit_summary,
    plot_signals,
    plot_equity_comparison,
    plot_summary_bars,
    validate_on_new_data,
)
from stvgp import evolve
from storage import (
    save_results,
    load_best_hofs,
    load_records,
    record_to_hofs,
    print_saved_summary,
    RESULTS_FILE,
)


# =====================================================================
# ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ НА СМЕСИ РЫНОЧНЫХ РЕЖИМОВ
# ---------------------------------------------------------------------
# Чтобы бот не выучился только «ловить тренд», обучаем его сразу
# на двух разных режимах рынка:
#   • январь 2024 — восходящий тренд (~+10%)
#   • сентябрь 2023 — боковой рынок (~+4%)
# Данные грузятся на уровне модуля, т.к. при spawn каждый дочерний
# процесс повторно импортирует main и должен видеть bars/ndf.
# =====================================================================

print(f"Loading TRAIN data ({CFG.symbol} {CFG.interval}): "
      f"{len(CFG.train_periods)} периодов...")
_train_dfs, _train_bars = [], []
for _start, _end in CFG.train_periods:
    _df, _b = load_period(CFG.symbol, CFG.interval, _start, _end)
    _train_dfs.append(_df)
    _train_bars.append(_b)
    print(f"  {_start}–{_end}: {len(_b)} bars")

# Объединяем бары всех периодов: фитнес будет суммой P&L по ним.
bars = [b for chunk in _train_bars for b in chunk]
ndf = _train_dfs[0]  # для plot_signals на тренировочных (берём первый кусок)
print(f"  Total: {len(bars)} bars")

NUM_THREADS = 1   # теперь это процессы


def _run_evolution_process(seed: int, result_queue: mp.Queue):
    """Целевая функция процесса."""
    try:
        # В дочернем процессе не строим графики (нет GUI / окна не нужны).
        pops, hofs = evolve(bars, ndf, seed=seed, do_plot=False)
        fitness = hofs[2][0].fitness.values[0]
        result_queue.put({
            "seed":    seed,
            "fitness": fitness,
            "best_long_str":  str(hofs[0][0]),
            "best_short_str": str(hofs[1][0]),
            "best_meta_str":  str(hofs[2][0]),
            "best_long_fit":  hofs[0][0].fitness.values[0],
            "best_short_fit": hofs[1][0].fitness.values[0],
            "best_meta_fit":  hofs[2][0].fitness.values[0],
            # hofs сериализуются через pickle (DEAP PrimitiveTree поддерживает)
            "hofs": hofs,
        })
        print(f"[seed={seed}] Process finished. Meta fitness = {fitness:.2f}%")
    except Exception as exc:
        print(f"[seed={seed}] Process ERROR: {exc}")
        result_queue.put(None)


def run_parallel_evolution(n_processes: int = 1):
    """
    Запускает n_processes процессов эволюции с уникальными seed-ами.
    Возвращает лучший результат по Meta fitness.
    """
    seeds = random.sample(range(1, 2**30), n_processes)
    result_queue = mp.Queue()

    print("=" * 70)
    print(f"ЗАПУСК {n_processes} ПАРАЛЛЕЛЬНЫХ ПРОЦЕССОВ (multiprocessing)")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    processes = []
    for seed in seeds:
        p = mp.Process(
            target=_run_evolution_process,
            args=(seed, result_queue),
            name=f"evo-seed-{seed}",
            daemon=False
        )
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # Собираем результаты
    all_results = []
    while not result_queue.empty():
        r = result_queue.get()
        if r is not None:
            all_results.append(r)

    if not all_results:
        raise RuntimeError("Все процессы упали, результатов нет!")

    best = max(all_results, key=lambda r: r["fitness"])

    print("\n" + "=" * 70)
    print("ИТОГИ ПАРАЛЛЕЛЬНОЙ ЭВОЛЮЦИИ")
    print("=" * 70)
    for r in sorted(all_results, key=lambda x: x["fitness"], reverse=True):
        marker = " <-- WINNER" if r["seed"] == best["seed"] else ""
        print(f"  seed={r['seed']:12d}  Meta fitness={r['fitness']:8.2f}%{marker}")
    print("=" * 70)

    # Сохраняем КАЖДОГО лучшего индивида (по одному с каждого процесса) в файл.
    save_results(all_results, filename=RESULTS_FILE, append=True)

    return best


# =====================================================================
# АНАЛИЗ И ВИЗУАЛИЗАЦИЯ (общая для эволюции и загрузки из файла)
# =====================================================================

def analyze_and_visualize(hofs, title_prefix="[BEST] "):
    """
    Прогоняет переданную стратегию (hofs) на train + двух OOS-периодах,
    печатает метрики и строит все графики.
    hofs = (hof_long, hof_short, hof_meta), каждый — список, [0] = лучший.
    """
    hof_long, hof_short, hof_meta = hofs

    print(f"\nBest Long fitness:  {hof_long[0].fitness.values[0]:.2f}%")
    print(f"Best Long:  {str(hof_long[0])}")
    print(f"\nBest Short fitness: {hof_short[0].fitness.values[0]:.2f}%")
    print(f"Best Short: {str(hof_short[0])}")
    print(f"\nBest Meta fitness:  {hof_meta[0].fitness.values[0]:.2f}%")
    print(f"Best Meta:  {str(hof_meta[0])}")

    # Визуализация на тренировочных данных
    plot_signals(ndf, bars, hof_long[0], hof_short[0], hof_meta[0],
                 pset_long, pset_short, pset_meta,
                 title_prefix=title_prefix)

    # ----- OOS ПЕРИОД 1: БЫЧИЙ РЫНОК (февраль 2024) -----
    print("\n\n" + "#" * 70)
    print("# OOS ПЕРИОД 1: БЫЧИЙ РЫНОК (февраль 2024)")
    print("#" * 70)
    val_df_bull, val_bars_bull = validate_on_new_data(
        hofs, val_symbol=CFG.symbol, val_interval=CFG.interval,
        val_start=CFG.val_bull[0], val_end=CFG.val_bull[1],
    )

    bot_metrics_bull, bh_metrics_bull = {}, {}
    if val_bars_bull:
        bot_metrics_bull, bh_metrics_bull, bot_eq_bull, bh_eq_bull = compare_strategies(
            val_bars_bull, hofs, risk_percent=CFG.risk_percent,
            label=f"BULL | Risk-managed {CFG.risk_percent:.0%}")
        compare_strategies(val_bars_bull, hofs, risk_percent=1.0,
                           label="BULL | Full exposure 100%")
        plot_equity_comparison(
            bot_eq_bull, bh_eq_bull,
            title="Капитал портфеля на бычьем рынке (февраль 2024): бот vs Buy & Hold",
            filename="equity_bull_feb2024.png",
        )

    # ----- OOS ПЕРИОД 2: СЛАБЫЙ МЕДВЕЖИЙ / БОКОВОЙ (май 2023) -----
    print("\n\n" + "#" * 70)
    print("# OOS ПЕРИОД 2: СЛАБЫЙ МЕДВЕЖИЙ / БОКОВОЙ (май 2023)")
    print("#" * 70)
    val_df_flat, val_bars_flat = validate_on_new_data(
        hofs, val_symbol=CFG.symbol, val_interval=CFG.interval,
        val_start=CFG.val_bear[0], val_end=CFG.val_bear[1],
    )

    bot_metrics_bear, bh_metrics_bear = {}, {}
    if val_bars_flat:
        bot_metrics_bear, bh_metrics_bear, bot_eq_bear, bh_eq_bear = compare_strategies(
            val_bars_flat, hofs, risk_percent=CFG.risk_percent,
            label=f"WEAK BEAR | Risk-managed {CFG.risk_percent:.0%}")
        compare_strategies(val_bars_flat, hofs, risk_percent=1.0,
                           label="WEAK BEAR | Full exposure 100%")
        plot_equity_comparison(
            bot_eq_bear, bh_eq_bear,
            title="Капитал портфеля на боковом/слабо медвежьем рынке (май 2023): бот vs Buy & Hold",
            filename="equity_bear_may2023.png",
        )

    # Итоговая сводка
    summary = []
    if bot_metrics_bull and bh_metrics_bull:
        summary.append({'period': 'Февраль 2024 (бычий)',
                        'bot': bot_metrics_bull, 'bh': bh_metrics_bull})
    if bot_metrics_bear and bh_metrics_bear:
        summary.append({'period': 'Май 2023 (боковой)',
                        'bot': bot_metrics_bear, 'bh': bh_metrics_bear})
    if summary:
        plot_summary_bars(summary, filename="summary_bars.png")

    # Показываем ВСЕ накопленные окна графиков разом (блокирующий вызов).
    plt.show()


def run_new_evolution():
    """Режим 1: запустить новую эволюцию, сохранить результаты, показать лучший."""
    simulator = TradingSimulator(initial_balance=10000, commission=0.001)
    print(f"\nGP data prepared: {len(bars)} bars")

    print("\n=== STARTING PARALLEL GP EVOLUTION ===\n")
    best_result = run_parallel_evolution(n_processes=NUM_THREADS)

    hofs = best_result["hofs"]
    analyze_and_visualize(hofs, title_prefix=f"[BEST seed={best_result['seed']}] ")


def run_from_file():
    """Режим 2: загрузить сохранённую стратегию из файла и сразу показать (без эволюции)."""
    records = load_records(RESULTS_FILE)
    if not records:
        print(f"\nФайл {RESULTS_FILE} пуст или не найден — сначала запустите эволюцию (вариант 1).")
        return

    print_saved_summary(RESULTS_FILE, top=20)

    # Выбор конкретной стратегии (по умолчанию — лучшая, индекс 0)
    raw = input(f"\nВведите № стратегии для запуска [0..{len(records)-1}] (Enter = 0): ").strip()
    try:
        idx = int(raw) if raw else 0
    except ValueError:
        idx = 0
    idx = max(0, min(idx, len(records) - 1))

    record = records[idx]
    hofs = record_to_hofs(record)

    # ПЕРВЫМ ДЕЛОМ — прибыль на тренировочном участке и разница с buy & hold
    quick_profit_summary(bars, hofs, label="TRAIN")

    print(f"\nЗагружена стратегия #{idx}: seed={record.get('seed')}, "
          f"meta fitness={record.get('fitness')}")

    analyze_and_visualize(hofs, title_prefix=f"[FILE #{idx} seed={record.get('seed')}] ")


def run_live_testnet():
    """Режим 3: торговля на ТЕСТОВОМ балансе (Binance Futures Testnet)
    по сохранённой стратегии из файла."""
    from live_trading import run_live_trading

    records = load_records(RESULTS_FILE)
    if not records:
        print(f"\nФайл {RESULTS_FILE} пуст — сначала обучите стратегию (вариант 1).")
        return

    print_saved_summary(RESULTS_FILE, top=20)
    raw = input(f"\n№ стратегии для торговли [0..{len(records)-1}] (Enter = 0): ").strip()
    try:
        idx = int(raw) if raw else 0
    except ValueError:
        idx = 0
    idx = max(0, min(idx, len(records) - 1))
    record = records[idx]
    hofs = record_to_hofs(record)

    print(f"\nТорговля стратегией #{idx}: seed={record.get('seed')}, "
          f"meta fitness={record.get('fitness')}")

    # Параметры торговли (можно поменять)
    interval = input(f"Таймфрейм [{CFG.interval}] (Enter): ").strip() or CFG.interval
    poll_raw = input("Опрос, секунд [60] (Enter = 60): ").strip()
    poll = int(poll_raw) if poll_raw.isdigit() else 60

    run_live_trading(
        hofs,
        symbol=CFG.symbol,
        interval=interval,
        window=CFG.window,
        risk_percent=CFG.risk_percent,
        leverage=1,
        poll_seconds=poll,
        max_iterations=None,   # бесконечно, до Ctrl+C
    )


def run_paper_trading():
    """Режим 4: быстрая симуляция стратегии по истории на живых ценах
    Binance — БЕЗ API-ключей, по виртуальному балансу."""
    from live_trading import paper_trade_history

    records = load_records(RESULTS_FILE)
    if not records:
        print(f"\nФайл {RESULTS_FILE} пуст — сначала обучите стратегию (вариант 1).")
        return

    print_saved_summary(RESULTS_FILE, top=20)
    raw = input(f"\n№ стратегии [0..{len(records)-1}] (Enter = 0): ").strip()
    try:
        idx = int(raw) if raw else 0
    except ValueError:
        idx = 0
    idx = max(0, min(idx, len(records) - 1))
    hofs = record_to_hofs(records[idx])

    interval = input(f"Таймфрейм [{CFG.interval}] (Enter): ").strip() or CFG.interval
    lim_raw = input("Сколько свечей истории [500] (Enter = 500): ").strip()
    limit = int(lim_raw) if lim_raw.isdigit() else 500

    paper_trade_history(
        hofs, symbol=CFG.symbol, interval=interval,
        window=CFG.window, limit=limit,
        initial_balance=CFG.initial_balance, risk_percent=CFG.risk_percent,
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)

    # Все функции построения только savefig() и НЕ вызывают plt.show().
    # Фигуры накапливаются в памяти, а показываются разом одним
    # блокирующим plt.show() в самом конце.

    # Загрузка через BinanceBroker (для проверки соединения)
    broker = BinanceBroker()
    data = broker.get_history_data([CFG.symbol], CFG.interval,
                                   '2024-01-01', '2024-02-01')
    print(f"Loaded {len(data)} candles")
    print(data.head())

    # ----------------------------------------------------------------
    # МЕНЮ ВЫБОРА РЕЖИМА
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EAGLETRADE — выберите режим работы:")
    print("  1 — Запустить новую эволюцию (обучение, результат сохранится в файл)")
    print(f"  2 — Загрузить готовую стратегию из файла ({RESULTS_FILE}) и показать графики")
    print("  3 — Торговля на ТЕСТОВОМ балансе (Binance Futures Testnet, нужны ключи)")
    print("  4 — Paper-trading: симуляция на живых ценах БЕЗ ключей (виртуальный баланс)")
    print("=" * 70)
    choice = input("Ваш выбор [1/2/3/4] (Enter = 1): ").strip()

    if choice == "2":
        run_from_file()
    elif choice == "3":
        run_live_testnet()
    elif choice == "4":
        run_paper_trading()
    else:
        run_new_evolution()
