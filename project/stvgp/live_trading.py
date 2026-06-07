# =====================================================================
# LIVE TRADING — торговля на ТЕСТОВОМ балансе через Binance Futures Testnet
# ---------------------------------------------------------------------
# Стратегия использует и LONG, и SHORT, поэтому берём именно фьючерсный
# testnet (https://testnet.binancefuture.com) — там виртуальный баланс
# USDT и поддержка шортов. Spot-testnet шорты не умеет.
#
# Ключи API создаются на https://testnet.binancefuture.com (Login -> API Key).
# Передаются через переменные окружения:
#   BINANCE_TESTNET_API_KEY
#   BINANCE_TESTNET_API_SECRET
# либо аргументами в run_live_trading(api_key=..., api_secret=...).
# =====================================================================

import os
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
from binance.client import Client

from deap import gp
from primitives import pset_long, pset_short, pset_meta
from signals import decide_position
from config import CFG


# ---------------------------------------------------------------------
# КЛИЕНТ TESTNET
# ---------------------------------------------------------------------

def make_testnet_client(api_key=None, api_secret=None):
    """
    Создаёт клиента Binance Futures Testnet.
    Ключи берутся из аргументов или из переменных окружения.
    """
    api_key = api_key or os.getenv("BINANCE_TESTNET_API_KEY")
    api_secret = api_secret or os.getenv("BINANCE_TESTNET_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError(
            "Не заданы ключи Binance Testnet.\n"
            "1) Зарегистрируйтесь на https://testnet.binancefuture.com\n"
            "2) Создайте API Key/Secret\n"
            "3) Задайте переменные окружения BINANCE_TESTNET_API_KEY и "
            "BINANCE_TESTNET_API_SECRET\n"
            "   (или передайте api_key=..., api_secret=... в run_live_trading)."
        )

    client = Client(api_key, api_secret)

    # ВАЖНО: новая Binance Demo Trading использует эндпоинт
    # https://demo-fapi.binance.com (НЕ старый testnet.binancefuture.com).
    # Адрес виден на странице API Management демо-аккаунта:
    #   "Futures Demo API Base Endpoint: https://demo-fapi.binance.com"
    client.FUTURES_URL = "https://demo-fapi.binance.com/fapi"
    client.FUTURES_DATA_URL = "https://demo-fapi.binance.com/futures/data"
    return client


# ---------------------------------------------------------------------
# ДАННЫЕ И ИНДИКАТОРЫ (как в build_input_vectors, но для последнего бара)
# ---------------------------------------------------------------------

def get_recent_klines(client, symbol=None, interval=None, limit=120,
                      drop_unclosed=True):
    """
    Тянет свежие свечи с фьючерсного эндпоинта и возвращает DataFrame
    со столбцом Price (= цена закрытия).

    drop_unclosed=True — отбрасывает ПОСЛЕДНЮЮ свечу, если она ещё не
    закрылась (Binance отдаёт текущую формирующуюся свечу последней).
    Без этого сигнал считается по «цене», которая ещё меняется —
    look-ahead/дёрганье в реальной торговле.
    """
    symbol = symbol or CFG.symbol
    interval = interval or CFG.interval
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "QAV", "Trades", "TBBAV", "TBQAV", "Ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["Price"] = df["Close"].astype(float)
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

    if drop_unclosed and len(df) > 0:
        # Свеча закрыта, если её Close Time уже в прошлом.
        now = pd.Timestamp.utcnow().tz_localize(None)
        if df["Close Time"].iloc[-1] > now:
            df = df.iloc[:-1]

    return df[["Open Time", "Price"]]


def build_last_input(df: pd.DataFrame, window: int = None):
    """
    Строит входной вектор для ПОСЛЕДНЕГО завершённого бара —
    в том же формате, что build_input_vectors (price/sma/ema/lwma/cur).
    """
    window = window if window is not None else CFG.window
    # Нужно минимум 2*window баров: окно iloc[i-window:i] не должно
    # содержать NaN от rolling() (см. подробности в build_input_vectors).
    if len(df) < 2 * window:
        return None

    price = df["Price"]
    sma = price.rolling(window).mean()
    ema = price.ewm(span=window, adjust=False).mean()
    weights = np.arange(1, window + 1)
    lwma = price.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # df уже без незакрытой свечи (см. get_recent_klines), поэтому
    # последний бар — это последний ЗАКРЫТЫЙ бар.
    i = len(df) - 1
    start = i - window
    return {
        "price": price.iloc[start:i].tolist(),
        "sma":   sma.iloc[start:i].tolist(),
        "ema":   ema.iloc[start:i].tolist(),
        "lwma":  lwma.iloc[start:i].tolist(),
        "cur":   float(price.iloc[i]),
    }


# ---------------------------------------------------------------------
# СИГНАЛ СТРАТЕГИИ (та же логика, что в compare_strategies)
# ---------------------------------------------------------------------

def get_public_klines(symbol=None, interval=None, limit=500):
    """
    Тянет свечи с ПУБЛИЧНОГО эндпоинта Binance Futures — БЕЗ ключей.
    Возвращает DataFrame со столбцами Open Time, Price (close).
    """
    symbol = symbol or CFG.symbol
    interval = interval or CFG.interval
    client = Client()  # без ключей: публичные данные доступны
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "QAV", "Trades", "TBBAV", "TBQAV", "Ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["Price"] = df["Close"].astype(float)
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    return df[["Open Time", "Price"]]


def paper_trade_history(hofs, symbol=None, interval=None, window=None,
                        limit=500, initial_balance=None, risk_percent=None,
                        commission=None):
    """
    БЫСТРАЯ СИМУЛЯЦИЯ по истории на ЖИВЫХ ценах Binance (без ключей).
    Прогоняет стратегию по последним `limit` свечам, исполняя сделки по
    виртуальному балансу. Печатает журнал сделок и итоговый результат.
    """
    from signals import TradingSimulator
    from data_loader import add_indicators, build_input_vectors

    symbol = symbol or CFG.symbol
    interval = interval or CFG.interval
    window = window if window is not None else CFG.window
    initial_balance = initial_balance if initial_balance is not None else CFG.initial_balance
    risk_percent = risk_percent if risk_percent is not None else CFG.risk_percent
    commission = commission if commission is not None else CFG.commission

    long_func = gp.compile(hofs[0][0], pset_long)
    short_func = gp.compile(hofs[1][0], pset_short)
    meta_func = gp.compile(hofs[2][0], pset_meta)

    print("=" * 70)
    print("PAPER TRADING — быстрая симуляция по истории (живые цены, без ключей)")
    print(f"Symbol: {symbol}  Interval: {interval}  Свечей: {limit}  "
          f"Старт: {initial_balance:.0f} USDT  Risk: {risk_percent:.0%}")
    print("=" * 70)

    # Загружаем данные и строим бары в том же формате, что при обучении
    df = get_public_klines(symbol, interval, limit=limit)
    df = add_indicators(df, window=window)
    bars = build_input_vectors(df, min_window=window)
    if not bars:
        print("Недостаточно данных для симуляции.")
        return

    sim = TradingSimulator(initial_balance=initial_balance,
                           commission=commission, risk_percent=risk_percent)

    n_trades = 0
    for b in bars:
        action = compute_action(b, long_func, short_func, meta_func)
        # FLAT -> закрыть позицию, если открыта
        if action == "FLAT":
            order = "CLOSE" if sim.position_type != "FLAT" else "HOLD"
        else:
            order = action

        prev_type = sim.position_type
        trade = sim.execute_trade(timestamp=None, action=order, price=b["cur"])
        if trade and (trade.action != prev_type) and trade.position_type != prev_type:
            n_trades += 1
            pnl_s = f"  P&L={trade.pnl:+.2f}" if trade.pnl else ""
            print(f"  {order:5s} @ {b['cur']:.2f}  -> {sim.position_type:5s}  "
                  f"баланс={sim.balance:.2f}{pnl_s}")

    # Закрываем позицию по последней цене
    sim.close_position(timestamp=None, price=bars[-1]["cur"])

    stats = sim.get_statistics()
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТ СИМУЛЯЦИИ")
    print("=" * 70)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:>12.2f}")
        else:
            print(f"  {k:<22}: {v:>12}")
    print("=" * 70)


def compute_action(bar, long_func, short_func, meta_func):
    """
    Возвращает 'LONG' / 'SHORT' / 'FLAT' по сигналу трёх популяций.
    Использует ту же decide_position(), что и бэктест — единый источник правды.
    """
    desired, _, _, _ = decide_position(bar, long_func, short_func, meta_func)
    if desired == 1:
        return "LONG"
    elif desired == -1:
        return "SHORT"
    return "FLAT"


# ---------------------------------------------------------------------
# РАБОТА С ПОЗИЦИЕЙ НА ФЬЮЧЕРСНОМ TESTNET
# ---------------------------------------------------------------------

def get_position(client, symbol):
    """Текущая позиция: ('LONG'|'SHORT'|'FLAT', amount_float)."""
    info = client.futures_position_information(symbol=symbol)
    amt = 0.0
    for p in info:
        if p["symbol"] == symbol:
            amt = float(p["positionAmt"])
            break
    if amt > 0:
        return "LONG", amt
    elif amt < 0:
        return "SHORT", abs(amt)
    return "FLAT", 0.0


def get_usdt_balance(client):
    """Свободный баланс USDT на фьючерсном testnet."""
    for b in client.futures_account_balance():
        if b["asset"] == "USDT":
            return float(b["balance"])
    return 0.0


def _round_qty(qty, step):
    """Округляет количество ВНИЗ до шага лота, без float-артефактов.
    Считает число знаков после запятой из step и квантует через округление."""
    if step <= 0:
        return qty
    # число десятичных знаков в step (0.0001 -> 4, 1.0 -> 0)
    s = f"{step:.18f}".rstrip("0")
    decimals = len(s.split(".")[1]) if "." in s else 0
    n = math.floor(qty / step + 1e-9)        # сколько целых шагов помещается
    return round(n * step, decimals)


def get_symbol_filters(client, symbol):
    """Возвращает (step_size, min_qty) для символа из exchange info."""
    info = client.futures_exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    return float(f["stepSize"]), float(f["minQty"])
    return 0.001, 0.001


def close_position(client, symbol, side, amount, step):
    """Закрывает текущую позицию рыночным ордером."""
    if side == "FLAT" or amount <= 0:
        return
    close_side = "SELL" if side == "LONG" else "BUY"
    qty = _round_qty(amount, step)
    if qty <= 0:
        return
    client.futures_create_order(
        symbol=symbol, side=close_side, type="MARKET",
        quantity=qty, reduceOnly=True)
    print(f"  [CLOSE] {side} {qty} {symbol}")


def open_position(client, symbol, side, usdt_balance, price, risk_percent, step, min_qty):
    """Открывает позицию LONG/SHORT рыночным ордером на risk_percent от баланса."""
    order_side = "BUY" if side == "LONG" else "SELL"
    notional = usdt_balance * risk_percent
    qty = _round_qty(notional / price, step)
    if qty < min_qty:
        print(f"  [SKIP] qty {qty} < min {min_qty} (баланс мал) — пропускаем вход")
        return
    client.futures_create_order(
        symbol=symbol, side=order_side, type="MARKET", quantity=qty)
    print(f"  [OPEN] {side} {qty} {symbol} @ ~{price:.2f}")


# ---------------------------------------------------------------------
# ОСНОВНОЙ ЦИКЛ
# ---------------------------------------------------------------------

def run_live_trading(hofs, symbol=None, interval=None, window=None,
                     risk_percent=None, leverage=1, poll_seconds=60,
                     max_iterations=None, api_key=None, api_secret=None):
    """
    Запускает торговлю на ТЕСТОВОМ балансе Binance Futures (demo).

    hofs          — (hof_long, hof_short, hof_meta); берём [0] каждого
    symbol        — торговая пара (по умолчанию CFG.symbol)
    interval      — таймфрейм свечей (по умолчанию CFG.interval)
    window        — окно индикаторов (как при обучении: CFG.window)
    risk_percent  — доля баланса на позицию (по умолчанию CFG.risk_percent)
    leverage      — кредитное плечо (1 = без плеча)
    poll_seconds  — пауза между проверками сигнала
    max_iterations— None = бесконечно; иначе остановиться после N итераций
    """
    symbol = symbol or CFG.symbol
    interval = interval or CFG.interval
    window = window if window is not None else CFG.window
    risk_percent = risk_percent if risk_percent is not None else CFG.risk_percent

    client = make_testnet_client(api_key, api_secret)

    # Компилируем стратегию один раз
    long_func = gp.compile(hofs[0][0], pset_long)
    short_func = gp.compile(hofs[1][0], pset_short)
    meta_func = gp.compile(hofs[2][0], pset_meta)

    # Плечо и фильтры лота
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        print(f"  (не удалось выставить плечо: {e})")
    step, min_qty = get_symbol_filters(client, symbol)

    print("=" * 70)
    print("LIVE TRADING — Binance Futures TESTNET (виртуальный баланс)")
    print(f"Symbol: {symbol}  Interval: {interval}  Risk: {risk_percent:.0%}  "
          f"Leverage: {leverage}x")
    print(f"Стартовый баланс USDT: {get_usdt_balance(client):.2f}")
    print(f"Опрос каждые {poll_seconds} c. Ctrl+C для остановки.")
    print("=" * 70)

    last_bar_time = None
    iteration = 0

    try:
        while True:
            iteration += 1
            # Тянем с запасом: нужно >= 2*window закрытых баров (без NaN
            # в окне индикаторов) + буфер на отброшенную незакрытую свечу.
            df = get_recent_klines(client, symbol, interval, limit=2 * window + 10)
            bar_time = df["Open Time"].iloc[-1]

            # Действуем только на новом баре (избегаем дёрганья внутри свечи)
            if last_bar_time is not None and bar_time == last_bar_time:
                if max_iterations and iteration >= max_iterations:
                    break
                time.sleep(poll_seconds)
                continue
            last_bar_time = bar_time

            bar = build_last_input(df, window=window)
            if bar is None:
                print("  Недостаточно данных для индикаторов, ждём...")
                time.sleep(poll_seconds)
                continue

            desired = compute_action(bar, long_func, short_func, meta_func)
            cur_side, cur_amt = get_position(client, symbol)
            price = bar["cur"]

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{ts}] bar={bar_time}  price={price:.2f}  "
                  f"signal={desired}  position={cur_side}({cur_amt})")

            # Приводим позицию к желаемой
            if desired == "FLAT":
                if cur_side != "FLAT":
                    close_position(client, symbol, cur_side, cur_amt, step)
            elif desired != cur_side:
                if cur_side != "FLAT":
                    close_position(client, symbol, cur_side, cur_amt, step)
                bal = get_usdt_balance(client)
                open_position(client, symbol, desired, bal, price,
                              risk_percent, step, min_qty)
            else:
                print("  Позиция уже соответствует сигналу — без изменений.")

            print(f"  Баланс USDT: {get_usdt_balance(client):.2f}")

            if max_iterations and iteration >= max_iterations:
                print("\nДостигнут лимит итераций — остановка.")
                break
            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        print("\nОстановлено пользователем (Ctrl+C).")

    # Финальная сводка
    final_side, final_amt = get_position(client, symbol)
    print("\n" + "=" * 70)
    print("LIVE TRADING ЗАВЕРШЕНО")
    print(f"Итоговая позиция: {final_side}({final_amt})")
    print(f"Итоговый баланс USDT: {get_usdt_balance(client):.2f}")
    print("=" * 70)
