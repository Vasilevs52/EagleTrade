# =====================================================================
# FUNDING RESEARCH — исследование ставок funding для дельта-нейтрального
# арбитража (спот + шорт перп). Скачивает историю funding rate с Binance
# по топ-монетам и считает: средние ставки, долю положительных, годовую
# доходность, окупаемость комиссий.
#
# Запуск:  cd project/funding && python research.py
# Ключи НЕ нужны — данные публичные.
# =====================================================================

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from binance.client import Client

# --- Параметры исследования ---
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TONUSDT",
]
DAYS = 180                    # глубина истории (дней)
PERIODS_PER_DAY = 3          # funding каждые 8 часов
PERIODS_PER_YEAR = 365 * PERIODS_PER_DAY

# Комиссии Binance (taker, спот + фьючерс). Round-trip оценка для
# входа+выхода дельта-нейтральной позиции: открыть спот + открыть шорт +
# закрыть спот + закрыть шорт. ~4 сделки.
TAKER_SPOT = 0.001          # 0.1%
TAKER_FUT = 0.0005          # 0.05%
ROUND_TRIP_COST = 2 * TAKER_SPOT + 2 * TAKER_FUT  # ~0.3% на полный цикл


def fetch_funding_history(client, symbol, days=DAYS):
    """Тянет историю funding rate за `days` дней (постранично по 1000)."""
    end = int(time.time() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    rows = []
    cur = start
    while cur < end:
        batch = client.futures_funding_rate(
            symbol=symbol, startTime=cur, endTime=end, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["fundingTime"]
        if last <= cur:
            break
        cur = last + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)  # вежливость к API
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df = df.drop_duplicates(subset="fundingTime").sort_values("fundingTime")
    return df


def analyze(symbol, df):
    """Считает статистику funding по символу."""
    if df.empty:
        return None
    r = df["fundingRate"].values
    n = len(r)
    mean = r.mean()
    median = np.median(r)
    pos_share = (r > 0).mean()
    # Годовая доходность дельта-нейтрала: получаем funding когда шортим перп
    # (при положительном funding шорт ПОЛУЧАЕТ выплату). Берём сумму как есть
    # — стратегия «всегда шорт перп + спот».
    annual_naive = mean * PERIODS_PER_YEAR * 100  # % годовых (всегда в позиции)
    # Умная: входим только когда funding положительный (иначе нет позиции)
    pos_only = r[r > 0]
    annual_smart = (pos_only.sum() / n) * PERIODS_PER_YEAR * 100 if n else 0
    return {
        "symbol": symbol,
        "periods": n,
        "mean_%": mean * 100,
        "median_%": median * 100,
        "pos_share_%": pos_share * 100,
        "annual_naive_%": annual_naive,
        "annual_smart_%": annual_smart,
        "max_%": r.max() * 100,
        "min_%": r.min() * 100,
    }


def main():
    client = Client()  # без ключей
    print(f"Скачиваю историю funding за {DAYS} дней по {len(SYMBOLS)} монетам...")
    print(f"Комиссия полного цикла (round-trip): {ROUND_TRIP_COST*100:.2f}%\n")

    results = []
    for sym in SYMBOLS:
        try:
            df = fetch_funding_history(client, sym)
            stat = analyze(sym, df)
            if stat:
                results.append(stat)
                print(f"  {sym:10s}: {stat['periods']:4d} периодов, "
                      f"средн {stat['mean_%']:+.4f}%, "
                      f"полож {stat['pos_share_%']:.0f}%, "
                      f"~{stat['annual_naive_%']:+.1f}%/год")
        except Exception as e:
            print(f"  {sym}: ОШИБКА {e}")

    if not results:
        print("Нет данных.")
        return

    rep = pd.DataFrame(results).sort_values("annual_naive_%", ascending=False)

    print("\n" + "=" * 90)
    print(f"FUNDING ARBITRAGE — потенциал за {DAYS} дней (дельта-нейтрал: спот + шорт перп)")
    print("=" * 90)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 200)
    print(rep.to_string(index=False))
    print("=" * 90)

    # Итоговые выводы
    avg_annual = rep["annual_naive_%"].mean()
    best = rep.iloc[0]
    print(f"\nСредняя годовая доходность по монетам (всегда в позиции): {avg_annual:+.1f}%")
    print(f"Лучшая монета: {best['symbol']} — {best['annual_naive_%']:+.1f}%/год "
          f"({best['pos_share_%']:.0f}% времени funding положительный)")
    print(f"Комиссия одного цикла входа/выхода: {ROUND_TRIP_COST*100:.2f}% "
          f"(= {ROUND_TRIP_COST/ (best['mean_%']/100) if best['mean_%']>0 else 0:.0f} периодов funding на окупаемость)")
    print("\nВЫВОД: funding-арбитраж прибылен, если держать позицию ДОЛГО "
          "(funding капает, комиссия разовая). Частые входы/выходы съедят профит.")


if __name__ == "__main__":
    main()
