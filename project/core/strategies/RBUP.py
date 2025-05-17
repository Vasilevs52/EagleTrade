import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

class RBUP(IStrategy):

    def __init__(self, settings: dict):
        self.window = settings['window']
        self.r_low = settings.get('r_low', 0.0001)
        self.r_high = settings.get('r_high', 0.0004)
        self.k = settings.get('k', 1.0)
        self.betta = settings.get('betta', 0.01)

    def get_signal(self, data: pd.DataFrame) -> int:
        if len(data) < self.window:
            return 0  # если не хватает информации

        prices = data['Close']
        window = self.window
        alpha = 2 / (window + 1)

        # EMA
        ema = prices.ewm(span=window, adjust=False).mean()

        # sigma
        v = np.zeros(len(prices))
        sigma = np.zeros(len(prices))
        for t in range(1, len(prices)):
            diff = prices.iloc[t] - ema.iloc[t - 1]
            v[t] = alpha * diff ** 2 + (1 - alpha) * v[t - 1]
            sigma[t] = np.sqrt(v[t])
        sigma = pd.Series(sigma, index=prices.index)

        # логарифм с модификатором
        b = 1 + self.betta * np.log1p(sigma)

        # LWMA с лог-весами
        base_weights = np.arange(1, window + 1)
        weights = np.log(base_weights) / np.log(b.iloc[-1])
        weights /= weights.sum()
        window_prices = prices.iloc[-window:]
        lwma_val = np.dot(window_prices.values, weights[::-1])
        diffs = window_prices.values - lwma_val
        sigma_lw = np.sqrt((weights * diffs**2).sum())

        # лог-ретурны и взвешенный R
        log_ret = np.log(prices / prices.shift(1)).dropna()
        if len(log_ret) < window:
            return 0

        recent_ret = log_ret.iloc[-window:]
        R = np.dot(recent_ret.values, weights[::-1])

        # Каналы
        upper = lwma_val + self.k * sigma_lw
        lower = lwma_val - self.k * sigma_lw
        P = prices.iloc[-1]

        # Простой фильтр по волатильности
        sigma_median = sigma[-window:].median()

        # Проверка спайков
        spike_window = 3
        if len(prices) < window + spike_window:
            return 0
        low_spike = prices.iloc[-spike_window:].min() <= lower
        high_spike = prices.iloc[-spike_window:].max() >= upper

        if low_spike and self.r_low < R < self.r_high and sigma_lw < sigma_median * 1.5:
            return 1  # long
        elif high_spike and self.r_low < R < self.r_high and sigma_lw > sigma_median * 0.8:
            return -1  # short
        else:
            return 0  # no signal

    def get_window(self):
        return self.window
