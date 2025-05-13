import numpy as np
import pandas as pd
from typing import Tuple
from infrastructure.api.brokers.Binance import BinanceBroker

broker = BinanceBroker()
data = broker.get_history_data(['BTCUSDT'], '1h', '01-01-2020', '01-01-2021')

# Параметры стратегии
sma_window = 20
ema_window = 15
lwma_window = 10
volatility_window = 10

def ruber_bend_up_strategy(
        prices: pd.Series,
        sma_window: int = 20,
        ema_window: int = 20,
        lwma_window: int = 10,
        volatility_window: int = 10,
        take_profit_pct: float = 2.0,
        stop_loss_pct: float = -2.0,
        r_threshold: float = 0.01
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Реализация стратегии "Ruber bend UP" с динамическими условиями входа/выхода.

    Параметры:
        prices: pd.Series - ряд цен
        sma_window: int - окно SMA
        ema_window: int - окно EMA
        lwma_window: int - окно LWMA
        volatility_window: int - окно волатильности
        take_profit_pct: float - уровень TP в %
        stop_loss_pct: float - уровень SL в %
        r_threshold: float - порог доходности для входа

    Возвращает:
        tuple: (sma, ema, lwma, volatility, signals)
    """
    # 1. Скользящие средние
    sma = prices.rolling(sma_window).mean()
    ema = prices.ewm(span=ema_window, adjust=False).mean()

    # 2. LWMA с лог-весами
    weights = np.log(np.arange(1, lwma_window + 1))
    weights /= weights.sum()
    lwma = prices.rolling(lwma_window).apply(lambda x: np.sum(x * weights), raw=True)

    # 3. Волатильность (лог-доходности)
    log_returns = np.log(prices / prices.shift(1))
    volatility = log_returns.rolling(volatility_window).std() * np.sqrt(252)  # Годовая волатильность

    # 4. Динамическое основание (взвешенная волатильность)
    weighted_volatility = volatility * prices / prices.rolling(volatility_window).mean()

    # 5. Сигналы
    signals = pd.Series(0, index=prices.index)
    position = 0
    entry_price = 0

    for i in range(1, len(prices)):
        # Условия входа LONG
        if position == 0 and (
                (ema[i] > sma[i]) and
                (prices[i] > lwma[i]) and
                (log_returns[i] > r_threshold) and
                (weighted_volatility[i] < 1.5 * weighted_volatility.median())
        ):
            signals.iloc[i] = 1
            position = 1
            entry_price = prices[i]

        # Условия входа SHORT
        elif position == 0 and (
                (ema[i] < sma[i]) and
                (prices[i] < lwma[i]) and
                (log_returns[i] < -r_threshold) and
                (weighted_volatility[i] > weighted_volatility.median())
        ):
            signals.iloc[i] = -1
            position = -1
            entry_price = prices[i]

        # Выход по TP/SL
        elif position != 0:
            current_return = (prices[i] - entry_price) / entry_price * 100
            if (position == 1 and (current_return >= take_profit_pct or current_return <= stop_loss_pct)) or \
                    (position == -1 and (current_return <= -take_profit_pct or current_return >= -stop_loss_pct)):
                signals.iloc[i] = 0
                position = 0

    return sma, ema, lwma, volatility, signals


import matplotlib.pyplot as plt


def plot_strategy(prices, sma, ema, lwma, signals):
    plt.figure(figsize=(14, 8))

    # Цены и индикаторы
    plt.plot(prices, label='Цены', color='black', alpha=0.8)
    plt.plot(sma, label=f'SMA {sma_window}', linestyle='--', color='blue')
    plt.plot(ema, label=f'EMA {ema_window}', linestyle='--', color='green')
    plt.plot(lwma, label=f'LWMA {lwma_window}', linestyle='--', color='purple')

    # Сигналы
    long_entries = signals == 1
    short_entries = signals == -1
    plt.plot(prices[long_entries], '^', markersize=10, color='g', label='Long')
    plt.plot(prices[short_entries], 'v', markersize=10, color='r', label='Short')

    plt.title('Ruber bend UP Strategy')
    plt.legend()
    plt.grid()
    plt.show()


sma, ema, lwma, volatility, signals = ruber_bend_up_strategy(
    data['Close'],
    sma_window=sma_window,
    ema_window=ema_window,
    lwma_window=lwma_window,
    volatility_window=volatility_window
)

plot_strategy(data['Close'], sma, ema, lwma, signals)