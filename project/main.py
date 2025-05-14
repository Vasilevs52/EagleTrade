import numpy as np
import pandas as pd
from typing import Tuple
from infrastructure.api.brokers.Binance import BinanceBroker

broker = BinanceBroker()
data = broker.get_history_data(['BTCUSDT'], '1h', '01-01-2020', '01-01-2021')

# Параметры стратегии
sma_window = 50
ema_window = 50
lwma_window = 50
volatility_window = 50
V0 = 0
δ = 0.1

def ruber_bend_up_strategy(
        prices: pd.Series,
        sma_window: int = 50,
        ema_window: int = 50,
        lwma_window: int = 50,
        volatility_window: int = 50,
        take_profit_pct: float = 2.0,
        stop_loss_pct: float = -2.0,
        r_low: float = 0.0001,
        r_high: float = 0.0004,
        k: float = 1.0,
        δ: float = 0.01
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


    """
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
    """
    # 1. SMA & EMA
    sma = prices.rolling(sma_window).mean()
    ema = prices.ewm(span=ema_window, adjust=False).mean()

    # 2. Простая волатильность для фильтра
    volatility = prices.pct_change().rolling(volatility_window).std()

    # 3. Динамическая sigma для основания логарифма
    alpha = 2 / (ema_window + 1)
    v = pd.Series(0.0, index=prices.index)
    sigma = pd.Series(0.0, index=prices.index)
    for t in range(1, len(prices)):
        diff = prices.iloc[t] - ema.iloc[t - 1]
        v.iloc[t] = alpha * diff ** 2 + (1 - alpha) * v.iloc[t - 1]
        sigma.iloc[t] = np.sqrt(v.iloc[t])
    b = 1 + δ * np.log1p(sigma)

    # 4. Динамический LWMA и sigma_lw
    lwma = pd.Series(np.nan, index=prices.index)
    sigma_lw = pd.Series(np.nan, index=prices.index)
    base_weights = np.arange(1, lwma_window + 1)
    for t in range(lwma_window - 1, len(prices)):
        # базовые веса для текущего t
        w_raw = np.log(base_weights) / np.log(b.iloc[t])
        W_t = w_raw.sum()
        w = w_raw / W_t
        window_prices = prices.iloc[t - lwma_window + 1: t + 1].values
        # LWMA
        lwma_val = np.dot(window_prices, w[::-1])
        lwma.iloc[t] = lwma_val
        # sigma_lw по формуле
        diffs = window_prices - lwma_val
        sigma_lw.iloc[t] = np.sqrt((w * diffs ** 2).sum())

    # 5. Лог-ретурны
    log_ret = np.log(prices / prices.shift(1))

    # 6. Скользящий взвешенный R_t
    R = pd.Series(np.nan, index=prices.index)
    for t in range(lwma_window - 1, len(prices)):
        # пересчёт весов для уровня t
        w_raw = np.log(base_weights) / np.log(b.iloc[t])
        W_t = w_raw.sum()
        w = w_raw / W_t
        window_r = log_ret.iloc[t - lwma_window + 1: t + 1].values
        R.iloc[t] = np.dot(window_r, w[::-1])

    # 7. Каналы Upper/Lower
    upper = lwma + k * sigma_lw
    lower = lwma - k * sigma_lw

    # 8. Генерация сигналов
    signals = pd.Series(0, index=prices.index, dtype=int)
    position = 0
    entry_price = 0.0

    # в начале функции (до цикла), задаём размер окна для спайков:
    spike_window = 3

    for i in range(lwma_window - 1, len(prices)):
        # динамическая медиана
        sigma_median = sigma_lw.iloc[:i + 1].median()
        vol_t = sigma_lw.iloc[i]
        P = prices.iloc[i]
        Rt = R.iloc[i]
        Up = upper.iloc[i]
        Lo = lower.iloc[i]

        # --- ЗАМЕНЯЕМ cond_two_bar на N-баровую проверку ---
        low_slice = prices.iloc[i - spike_window + 1: i + 1]
        high_slice = prices.iloc[i - spike_window + 1: i + 1]

        cond_spike_long = (low_slice.min() <= Lo)  # был спайк вниз в окне
        cond_spike_short = (high_slice.max() >= Up)  # был спайк вверх в окне
        # ---------------------------------------------------

        # LONG
        if position == 0 and \
                cond_spike_long and \
                (r_low < Rt < r_high) and \
                (vol_t < sigma_median * 1.5):  # слегка ослабил вола-фильтр
            signals.iloc[i] = 1
            position = 1
            entry_price = P

        # SHORT
        elif position == 0 and \
                cond_spike_short and \
                (r_low < Rt < r_high) and \
                (vol_t > sigma_median * 0.8):
            signals.iloc[i] = -1
            position = -1
            entry_price = P

        # EXIT (оставляем без изменений)
        elif position != 0:
            tp = lwma.iloc[i] + 0.5 * sigma_lw.iloc[i]
            sl = lwma.iloc[i] - 0.5 * sigma_lw.iloc[i]
            ret = (P - entry_price) / entry_price * 100
            if position == 1 and (P >= tp or ret <= stop_loss_pct):
                signals.iloc[i] = 0;
                position = 0
            elif position == -1 and (P <= sl or ret >= take_profit_pct):
                signals.iloc[i] = 0;
                position = 0

    return sma, ema, lwma, volatility, signals

import matplotlib.pyplot as plt


def plot_strategy(prices, sma, ema, lwma, signals,
                  sma_window: int, ema_window: int, lwma_window: int):
    """
    Рисует цену и индикаторы, а также точки входа и выхода из позиций.

    Параметры:
        prices       — pd.Series с ценами закрытия
        sma, ema, lwma — pd.Series ваших скользящих средних
        signals      — pd.Series сигналов (+1 = long entry, -1 = short entry, 0 = flat/exit)
        sma_window, ema_window, lwma_window — длины окон (чтобы показать в легенде)
    """
    plt.figure(figsize=(26, 16))

    # Plot price & indicators
    plt.plot(prices, label='Цена', color='black', alpha=0.8)
    plt.plot(sma, label=f'SMA {sma_window}', linestyle='--', color='blue')
    plt.plot(ema, label=f'EMA {ema_window}', linestyle='--', color='green')
    plt.plot(lwma, label=f'LWMA {lwma_window}', linestyle='--', color='purple')

    # Определяем входы и выходы
    long_entry = (signals.shift(1) == 0) & (signals == 1)
    long_exit = (signals.shift(1) == 1) & (signals == 0)
    short_entry = (signals.shift(1) == 0) & (signals == -1)
    short_exit = (signals.shift(1) == -1) & (signals == 0)

    # Рисуем стрелки
    plt.plot(prices[long_entry], '^', markersize=10, color='lime', label='Long Entry')
    plt.plot(prices[long_exit], 'v', markersize=10, color='cyan', label='Long Exit')
    plt.plot(prices[short_entry], 'v', markersize=10, color='red', label='Short Entry')
    plt.plot(prices[short_exit], '^', markersize=10, color='orange', label='Short Exit')

    plt.title('Ruber Bend UP Strategy')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Пример использования:
sma, ema, lwma, volatility, signals = ruber_bend_up_strategy(
    data['Close'],
    sma_window=sma_window,
    ema_window=ema_window,
    lwma_window=lwma_window,
    volatility_window=volatility_window
)

plot_strategy(data['Close'], sma, ema, lwma, signals,
              sma_window, ema_window, lwma_window)