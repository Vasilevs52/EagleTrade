# =====================================================================
# DATA LOADER — загрузка истории с Binance и подготовка баров
# =====================================================================

import numpy as np
import pandas as pd
from binance.client import Client


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


def add_indicators(df: pd.DataFrame, window: int = 50):
    df[f"SMA_{window}"] = df["Price"].rolling(window).mean()
    df[f"EMA_{window}"] = df["Price"].ewm(span=window, adjust=False).mean()
    weights = np.arange(1, window + 1)
    df[f"LWMA_{window}"] = df["Price"].rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return df


def build_input_vectors(df: pd.DataFrame, min_window: int = 50):
    bars = []
    for i in range(min_window, len(df) - 1):
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


def load_period(symbol, interval, start, end, window=50):
    """Загружает один период, добавляет индикаторы, возвращает (df, bars)."""
    df = get_history_data(symbol, interval, start, end)
    df = add_indicators(df, window=window)
    bars = build_input_vectors(df, min_window=window)
    return df, bars
