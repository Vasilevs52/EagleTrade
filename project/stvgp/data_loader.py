# =====================================================================
# DATA LOADER — загрузка истории с Binance и подготовка баров
# =====================================================================

import os
import pickle
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from binance.client import Client

from config import CFG, CFG2


class BinanceBroker():
    @staticmethod
    def get_history_data(symbols: list, interval: str, start_date: str, end_date: str):
        client = Client()
        columns = [
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
        ]
        arr_df = []
        for symbol in symbols:
            data = client.get_historical_klines(symbol=symbol, interval=interval,
                                                start_str=start_date, end_str=end_date)
            df = pd.DataFrame(data, columns=columns, dtype=float)
            df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
            df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
            df["Symbol"] = symbol
            arr_df.append(df)
        if len(arr_df) == 1:
            return arr_df[0]
        else:
            return pd.concat(arr_df, axis=0, ignore_index=True)


def get_history_data(symbol, interval, start_date, end_date):
    client = Client()
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    cols = ["Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "#Trades",
            "TakerBuyBase", "TakerBuyQuote", "Ignore"]
    df = pd.DataFrame(klines, columns=cols, dtype=float)
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
    df = df[["Open Time", "Close", "Volume"]]
    df.rename(columns={"Close": "Price"}, inplace=True)
    return df


def add_indicators(df: pd.DataFrame, window: int = None):
    window = window if window is not None else CFG.window
    df[f"SMA_{window}"] = df["Price"].rolling(window).mean()
    df[f"EMA_{window}"] = df["Price"].ewm(span=window, adjust=False).mean()
    weights = np.arange(1, window + 1)
    df[f"LWMA_{window}"] = df["Price"].rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return df


def build_input_vectors(df: pd.DataFrame, min_window: int = None):
    min_window = min_window if min_window is not None else CFG.window
    bars = []
    # ВАЖНО: окно индикаторов — это срез iloc[i-min_window : i]. SMA/LWMA
    # не определены (NaN) на первых (min_window-1) барах из-за rolling().
    # Чтобы в ОКНО не попадали NaN, начинаем не с min_window, а с
    # (2*min_window - 1): тогда самый ранний индекс окна = min_window-1,
    # где индикаторы уже посчитаны. Иначе деревья получают NaN на входе и
    # сигналы молча «съедаются» (NaN > 0 == False), искажая фитнес.
    first_i = 2 * min_window - 1
    for i in range(first_i, len(df) - 1):
        start = i - min_window
        bars.append({
            "price": df["Price"].iloc[start:i].tolist(),
            "sma": df[f"SMA_{min_window}"].iloc[start:i].tolist(),
            "ema": df[f"EMA_{min_window}"].iloc[start:i].tolist(),
            "lwma": df[f"LWMA_{min_window}"].iloc[start:i].tolist(),
            "cur": df["Price"].iloc[i],
            "next": df["Price"].iloc[i + 1]
        })
    return bars


def load_period(symbol, interval, start, end, window=None):
    """Загружает один период, добавляет индикаторы, возвращает (df, bars)."""
    window = window if window is not None else CFG.window
    df = get_history_data(symbol, interval, start, end)
    df = add_indicators(df, window=window)
    bars = build_input_vectors(df, min_window=window)
    return df, bars


# =====================================================================
# V2 — мульти-активный датасет для «честной эволюции»
# =====================================================================

_BARS_PER_DAY = {"15m": 96, "30m": 48, "1h": 24, "2h": 12, "4h": 6}


def normalize_segment(seg):
    """
    Нормализует окно: все цены делятся на первую цену окна (cur[0]).

    Зачем: деревья видят сырые цены, а активы различаются масштабом в
    ~100000 раз (BTC 60000 vs XRP 0.5). Без нормализации дерево может
    «опознать актив» (if цена>1000 -> BTC-логика) и построить
    asset-specific подстратегии — это дыра в мульти-активной защите.
    После нормализации все активы выглядят как ~1.0 ± проценты.

    PnL НЕ меняется: (a-b)/b инвариантно к общему множителю, поэтому
    eval-функции и симулятор дают идентичные результаты.
    Возвращает НОВЫЕ dict'ы (не мутирует вход — окна могут пересекаться).
    """
    if not seg:
        return seg
    base = float(seg[0]["cur"])
    if base <= 0:
        return seg
    out = []
    for b in seg:
        out.append({
            "price": [x / base for x in b["price"]],
            "sma":   [x / base for x in b["sma"]],
            "ema":   [x / base for x in b["ema"]],
            "lwma":  [x / base for x in b["lwma"]],
            "cur":   b["cur"] / base,
            "next":  b["next"] / base,
        })
    return out


def load_v2_dataset(force=False, cache_path=None):
    """
    Готовит датасет v2: по каждому активу из CFG2.assets качает историю,
    строит бары и нарезает на три зоны (по времени, без перемешивания):

      • holdout — нетронутые ПОСЛЕДНИЕ holdout_days (финальный тест, 1 раз)
      • val     — n_val_windows самых свежих окон ДО holdout (отбор стратегий)
      • train   — n_train_windows окон, равномерно по остальной истории

    Результат кешируется в pickle: повторный вызов (в т.ч. из дочерних
    spawn-процессов) читает с диска, не дёргая Binance.
    """
    cache_path = cache_path or CFG2.cache_file
    # Метка конфигурации: если конфиг изменился — кеш невалиден.
    cfg_meta = {
        "assets": tuple(CFG2.assets), "interval": CFG2.interval,
        "history_days": CFG2.history_days, "holdout_days": CFG2.holdout_days,
        "n_train_windows": CFG2.n_train_windows,
        "n_val_windows": CFG2.n_val_windows,
        "window_bars": CFG2.window_bars,
        "normalize": getattr(CFG2, "normalize", False),
    }
    if not force and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("cfg_meta") == cfg_meta:
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(
                    cached["created"])
                if age.days >= 7:
                    print(f"ВНИМАНИЕ: кешу данных {age.days} дней — данные "
                          f"устарели. Обновить: python main2.py -> 5")
            except (KeyError, ValueError):
                pass
            return cached
        print("Конфигурация V2 изменилась — перекачиваю данные...")

    per_day = _BARS_PER_DAY.get(CFG2.interval, 24)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=CFG2.history_days + CFG2.holdout_days)
    wb = CFG2.window_bars

    dataset = {"assets": [], "train": [], "val": [], "holdout": [],
               "created": now.isoformat(), "interval": CFG2.interval,
               "cfg_meta": cfg_meta}

    print(f"V2: качаю {CFG2.history_days + CFG2.holdout_days} дней "
          f"{CFG2.interval} по {len(CFG2.assets)} активам...")
    for sym in CFG2.assets:
        df = get_history_data(sym, CFG2.interval,
                              start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
        df = add_indicators(df)
        bars = build_input_vectors(df)

        hold_n = CFG2.holdout_days * per_day
        need = hold_n + wb * (CFG2.n_val_windows + 1)
        if len(bars) <= need:
            print(f"  {sym}: мало данных ({len(bars)} баров < {need}), пропускаю")
            continue

        holdout = bars[-hold_n:]
        rest = bars[:-hold_n]

        # Валидация: самые свежие окна (до holdout)
        val_segs = []
        for k in range(CFG2.n_val_windows):
            end_i = len(rest) - k * wb
            st_i = end_i - wb
            if st_i < 0:
                break
            val_segs.append(rest[st_i:end_i])

        # Обучение: равномерные окна по оставшейся (более старой) истории
        train_zone = rest[:len(rest) - len(val_segs) * wb]
        L = len(train_zone)
        if L < wb:
            print(f"  {sym}: train-зона мала ({L} баров), пропускаю")
            continue
        n = CFG2.n_train_windows
        step = max(1, (L - wb) // max(1, n - 1))
        if step < wb:
            print(f"  ВНИМАНИЕ {sym}: train-окна перекрываются "
                  f"(step={step} < window_bars={wb}) — перекрытие завышает "
                  f"оценку стабильности. Увеличьте history_days или "
                  f"уменьшите n_train_windows/window_bars.")
        train_segs = [train_zone[i:i + wb]
                      for i in range(0, L - wb + 1, step)][:n]

        # Нормализация (см. normalize_segment): убирает идентификацию
        # актива по абсолютной цене. Применяется ко ВСЕМ зонам одинаково.
        if cfg_meta["normalize"]:
            train_segs = [normalize_segment(s) for s in train_segs]
            val_segs = [normalize_segment(s) for s in val_segs]
            holdout = normalize_segment(holdout)

        dataset["assets"].append(sym)
        dataset["train"].extend(train_segs)
        dataset["val"].extend(val_segs)
        dataset["holdout"].append((sym, holdout))
        print(f"  {sym}: train {len(train_segs)}x{wb}, val {len(val_segs)}x{wb}, "
              f"holdout {len(holdout)} баров"
              + (" [нормализовано]" if cfg_meta["normalize"] else ""))

    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"  кеш сохранён: {cache_path} "
          f"(train {len(dataset['train'])} окон, val {len(dataset['val'])})")
    return dataset
