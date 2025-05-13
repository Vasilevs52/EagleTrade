import pandas as pd
import numpy as np


def SMA(data: pd.Series, window: int) -> pd.Series:
    """Простое скользящее среднее"""
    return data.rolling(window=window).mean()


def EMA(data: pd.Series, window: int) -> pd.Series:
    """Экспоненциальное скользящее среднее"""
    return data.ewm(span=window, adjust=False).mean()  # Исправлено ewn на ewm


