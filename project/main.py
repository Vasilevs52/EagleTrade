import operator
import random
import statistics
from functools import partial
import math
import numpy as np
from deap import base, creator, gp, tools
from deap import algorithms
import pandas as pd
from binance.client import Client
import matplotlib.pyplot as plt

from binance.client import Client
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from datetime import datetime


class BinanceBroker():
    @staticmethod
    def get_history_data(symbols: list, interval: str, start_date: str, end_date: str):
        client = Client()
        columns = [
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Volume",
            "Taker Buy Quote Volume",
            "Ignore"
        ]

        arr_df = []
        for symbol in symbols:
            data = client.get_historical_klines(symbol=symbol,
                                                interval=interval,
                                                start_str=start_date,
                                                end_str=end_date)
            df = pd.DataFrame(data, columns=columns, dtype=float)
            df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
            df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
            df["Symbol"] = symbol
            arr_df.append(df)

        if len(arr_df) == 1:
            return arr_df[0]
        else:
            df = pd.concat(arr_df, axis=0, ignore_index=True)
            return df


@dataclass
class Trade:
    """Класс для представления одной сделки"""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'CLOSE'
    price: float
    quantity: float
    balance_before: float
    balance_after: float
    pnl: float = 0.0
    position_type: str = "FLAT"  # 'LONG', 'SHORT', 'FLAT'


class TradingSimulator:
    """Симулятор торговли с тестовым балансом"""

    def __init__(self, initial_balance: float = 10000, commission: float = 0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.position = 0.0  # Текущая позиция
        self.position_price = 0.0  # Цена входа в позицию
        self.position_type = "FLAT"  # 'LONG', 'SHORT', 'FLAT'
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.max_balance = initial_balance
        self.max_drawdown = 0.0

    def reset(self):
        """Сброс симулятора"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_price = 0.0
        self.position_type = "FLAT"
        self.trades = []
        self.equity_curve = []
        self.max_balance = self.initial_balance
        self.max_drawdown = 0.0

    def execute_trade(self, timestamp: datetime, action: str, price: float,
                      risk_percent: float = 0.02) -> Optional[Trade]:
        """
        Выполнение сделки

        Args:
            timestamp: Время сделки
            action: 'LONG', 'SHORT', 'HOLD'
            price: Цена
            risk_percent: Процент риска от баланса
        """
        balance_before = self.balance
        pnl = 0.0

        if action == 'HOLD' or self.balance <= 0:
            self.equity_curve.append(self.balance)
            return None

        if action == 'LONG':
            if self.position_type == 'SHORT':
                # Закрываем короткую позицию
                pnl = self.position * (self.position_price - price)
                self.balance += pnl
                commission_cost = abs(self.position * price * self.commission)
                self.balance -= commission_cost

                # Открываем длинную позицию
                risk_amount = self.balance * risk_percent
                quantity = risk_amount / price
                self.position = quantity
                self.position_price = price
                self.position_type = 'LONG'
                commission_cost += quantity * price * self.commission
                self.balance -= commission_cost

            elif self.position_type == 'FLAT':
                # Открываем длинную позицию
                risk_amount = self.balance * risk_percent
                quantity = risk_amount / price
                commission_cost = quantity * price * self.commission
                self.balance -= commission_cost
                self.position = quantity
                self.position_price = price
                self.position_type = 'LONG'

        elif action == 'SHORT':
            if self.position_type == 'LONG':
                # Закрываем длинную позицию
                pnl = self.position * (price - self.position_price)
                self.balance += pnl
                commission_cost = abs(self.position * price * self.commission)
                self.balance -= commission_cost

                # Открываем короткую позицию
                risk_amount = self.balance * risk_percent
                quantity = risk_amount / price
                self.position = -quantity
                self.position_price = price
                self.position_type = 'SHORT'
                commission_cost += quantity * price * self.commission
                self.balance -= commission_cost

            elif self.position_type == 'FLAT':
                # Открываем короткую позицию
                risk_amount = self.balance * risk_percent
                quantity = risk_amount / price
                commission_cost = quantity * price * self.commission
                self.balance -= commission_cost
                self.position = -quantity
                self.position_price = price
                self.position_type = 'SHORT'

        # Обновляем статистику
        self.max_balance = max(self.max_balance, self.balance)
        current_drawdown = (self.max_balance - self.balance) / self.max_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        self.equity_curve.append(self.balance)

        # Создаем запись о сделке
        trade = Trade(
            timestamp=timestamp,
            action=action,
            price=price,
            quantity=abs(self.position),
            balance_before=balance_before,
            balance_after=self.balance,
            pnl=pnl,
            position_type=self.position_type
        )

        self.trades.append(trade)
        return trade

    def close_position(self, timestamp: datetime, price: float) -> Optional[Trade]:
        """Закрытие текущей позиции"""
        if self.position_type == 'FLAT':
            return None

        balance_before = self.balance

        if self.position_type == 'LONG':
            pnl = self.position * (price - self.position_price)
        else:  # SHORT
            pnl = abs(self.position) * (self.position_price - price)

        self.balance += pnl
        commission_cost = abs(self.position * price * self.commission)
        self.balance -= commission_cost

        trade = Trade(
            timestamp=timestamp,
            action='CLOSE',
            price=price,
            quantity=abs(self.position),
            balance_before=balance_before,
            balance_after=self.balance,
            pnl=pnl,
            position_type=self.position_type
        )

        self.trades.append(trade)

        # Сброс позиции
        self.position = 0.0
        self.position_price = 0.0
        self.position_type = 'FLAT'

        self.equity_curve.append(self.balance)
        return trade

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Получение нереализованной прибыли/убытка"""
        if self.position_type == 'FLAT':
            return 0.0
        elif self.position_type == 'LONG':
            return self.position * (current_price - self.position_price)
        else:  # SHORT
            return abs(self.position) * (self.position_price - current_price)

    def get_statistics(self) -> Dict:
        """Получение статистики торговли"""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'action': t.action,
                'price': t.price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'balance': t.balance_after
            } for t in self.trades
        ])

        profitable_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100

        stats = {
            'Initial Balance': self.initial_balance,
            'Final Balance': self.balance,
            'Total Return (%)': total_return,
            'Max Drawdown (%)': self.max_drawdown * 100,
            'Total Trades': len(self.trades),
            'Profitable Trades': len(profitable_trades),
            'Losing Trades': len(losing_trades),
            'Win Rate (%)': (len(profitable_trades) / len(self.trades)) * 100 if self.trades else 0,
            'Avg Profit': profitable_trades['pnl'].mean() if not profitable_trades.empty else 0,
            'Avg Loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
            'Profit Factor': abs(
                profitable_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty and
                                                                                losing_trades[
                                                                                    'pnl'].sum() != 0 else float('inf')
        }

        return stats


def backtest_strategy(df: pd.DataFrame, long_func, short_func, meta_func,
                      simulator: TradingSimulator, verbose: bool = True):
    """
    Бэктест стратегии с симулятором

    Args:
        df: DataFrame с данными OHLCV
        long_func, short_func, meta_func: Скомпилированные функции стратегий
        simulator: Объект TradingSimulator
        verbose: Выводить ли детали
    """
    simulator.reset()
    signals = []

    # Подготовка индикаторов (как в оригинальном коде)
    df_copy = df.copy()
    df_copy['SMA'] = df_copy['Close'].rolling(window=10).mean()
    df_copy['EMA'] = df_copy['Close'].ewm(span=10).mean()
    df_copy['LWMA'] = df_copy['Close'].rolling(window=10).apply(
        lambda x: (x * np.arange(1, len(x) + 1)).sum() / np.arange(1, len(x) + 1).sum()
    )

    # Создание массива баров как в оригинальном коде
    bars = []
    for i in range(len(df_copy)):
        if i < 10:  # Пропускаем первые 10 баров из-за индикаторов
            continue

        price_vector = df_copy['Close'].iloc[max(0, i - 9):i + 1].values
        sma_vector = df_copy['SMA'].iloc[max(0, i - 9):i + 1].values
        ema_vector = df_copy['EMA'].iloc[max(0, i - 9):i + 1].values
        lwma_vector = df_copy['LWMA'].iloc[max(0, i - 9):i + 1].values
        current_price = df_copy['Close'].iloc[i]

        bar = {
            "price": price_vector,
            "sma": sma_vector,
            "ema": ema_vector,
            "lwma": lwma_vector,
            "cur": current_price
        }
        bars.append(bar)

    if verbose:
        print(f"Starting backtest with {len(bars)} bars...")
        print(f"Initial balance: ${simulator.initial_balance:,.2f}")

    # Основной цикл бэктеста
    for i, bar in enumerate(bars):
        try:
            timestamp = df_copy.iloc[i + 10]['Open Time'] if 'Open Time' in df_copy.columns else datetime.now()
            current_price = bar['cur']

            # Получение сигналов
            long_raw = long_func(bar["price"], bar["sma"], bar["ema"], bar["lwma"], bar["cur"])
            short_raw = short_func(bar["price"], bar["sma"], bar["ema"], bar["lwma"], bar["cur"])

            long_bool = 1.0 if long_raw > 0 else 0.0
            short_bool = 1.0 if short_raw > 0 else 0.0

            meta_raw = meta_func(long_bool, short_bool)

            # Определение действия
            if meta_raw > 0.5:
                action = "LONG"
            elif meta_raw < -0.5:
                action = "SHORT"
            else:
                action = "HOLD"

            # Выполнение сделки
            trade = simulator.execute_trade(timestamp, action, current_price)

            # Сохранение информации о сигнале
            unrealized_pnl = simulator.get_unrealized_pnl(current_price)
            signals.append({
                'timestamp': timestamp,
                'price': current_price,
                'long_raw': long_raw,
                'short_raw': short_raw,
                'long_bool': long_bool,
                'short_bool': short_bool,
                'meta_raw': meta_raw,
                'action': action,
                'balance': simulator.balance,
                'position_type': simulator.position_type,
                'unrealized_pnl': unrealized_pnl,
                'trade_executed': trade is not None
            })

        except Exception as e:
            if verbose:
                print(f"Error at bar {i}: {e}")
            signals.append({
                'timestamp': datetime.now(),
                'price': bar['cur'],
                'long_raw': 0,
                'short_raw': 0,
                'long_bool': 0,
                'short_bool': 0,
                'meta_raw': 0,
                'action': 'ERROR',
                'balance': simulator.balance,
                'position_type': simulator.position_type,
                'unrealized_pnl': 0,
                'trade_executed': False
            })

    # Закрываем позицию в конце
    if simulator.position_type != 'FLAT':
        final_price = bars[-1]['cur']
        final_timestamp = signals[-1]['timestamp']
        simulator.close_position(final_timestamp, final_price)

    signals_df = pd.DataFrame(signals)

    if verbose:
        stats = simulator.get_statistics()
        print(f"\n=== BACKTEST RESULTS ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

    return signals_df, simulator.get_statistics()


def plot_backtest_results(df: pd.DataFrame, signals_df: pd.DataFrame, simulator: TradingSimulator):
    """Построение графиков результатов бэктеста"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

    # График цены и сигналов
    ax1.plot(df['Close'].values, label='Price', alpha=0.7)

    # Отмечаем сделки
    for trade in simulator.trades:
        if hasattr(trade, 'timestamp'):
            idx = signals_df[signals_df['timestamp'] == trade.timestamp].index
            if len(idx) > 0:
                idx = idx[0]
                if trade.action == 'LONG':
                    ax1.scatter(idx, trade.price, color='green', marker='^', s=100, label='Long' if idx == 0 else "")
                elif trade.action == 'SHORT':
                    ax1.scatter(idx, trade.price, color='red', marker='v', s=100, label='Short' if idx == 0 else "")
                elif trade.action == 'CLOSE':
                    ax1.scatter(idx, trade.price, color='blue', marker='x', s=100, label='Close' if idx == 0 else "")

    ax1.set_title('Price and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График баланса
    ax2.plot(signals_df['balance'].values, label='Balance', color='blue')
    ax2.set_title('Account Balance')
    ax2.set_ylabel('Balance ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График сигналов
    ax3.plot(signals_df['meta_raw'].values, label='Meta Signal', color='purple')
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Long Threshold')
    ax3.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Short Threshold')
    ax3.set_title('Meta Strategy Signals')
    ax3.set_ylabel('Signal Strength')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Функции интеграции с оригинальным кодом
def evaluate_with_balance(individual, bars, strategy_type, pset, simulator=None):
    """
    Оценочная функция с учетом баланса - заменяет evalLongTrading и evalShortTrading
    """
    if simulator is None:
        simulator = TradingSimulator(initial_balance=10000, commission=0.001)

    simulator.reset()

    try:
        from deap import gp
        func = gp.compile(individual, pset)

        for bar in bars[:100]:  # Ограничиваем для скорости
            try:
                signal = func(bar["price"], bar["sma"], bar["ema"], bar["lwma"], bar["cur"])

                if strategy_type == 'long':
                    action = "LONG" if signal > 0 else "HOLD"
                else:
                    action = "SHORT" if signal > 0 else "HOLD"

                simulator.execute_trade(datetime.now(), action, bar['cur'], risk_percent=0.01)

            except:
                continue

        # Закрываем позицию
        if simulator.position_type != 'FLAT' and bars:
            simulator.close_position(datetime.now(), bars[-1]['cur'])

        # Возвращаем общую доходность как фитнес
        total_return = ((simulator.balance - simulator.initial_balance) / simulator.initial_balance) * 100

        # Добавляем штраф за большую просадку
        penalty = simulator.max_drawdown * 50  # Штраф за просадку
        fitness = total_return - penalty

        return (max(fitness, -100),)  # Ограничиваем минимум

    except Exception as e:
        return (-100,)


def evaluate_meta_with_balance(meta_individual, bars, long_func, short_func, pset_meta, simulator=None):
    """
    Оценочная функция для мета-популяции с учетом баланса
    """
    if simulator is None:
        simulator = TradingSimulator(initial_balance=10000, commission=0.001)

    simulator.reset()

    try:
        from deap import gp
        meta_func = gp.compile(meta_individual, pset_meta)

        for bar in bars[:100]:
            try:
                # Получаем сигналы от основных стратегий
                long_raw = long_func(bar["price"], bar["sma"], bar["ema"], bar["lwma"], bar["cur"])
                short_raw = short_func(bar["price"], bar["sma"], bar["ema"], bar["lwma"], bar["cur"])

                long_bool = 1.0 if long_raw > 0 else 0.0
                short_bool = 1.0 if short_raw > 0 else 0.0

                # Получаем финальный сигнал от мета-стратегии
                meta_signal = meta_func(long_bool, short_bool)

                if meta_signal > 0.5:
                    action = "LONG"
                elif meta_signal < -0.5:
                    action = "SHORT"
                else:
                    action = "HOLD"

                simulator.execute_trade(datetime.now(), action, bar['cur'], risk_percent=0.02)

            except:
                continue

        # Закрываем позицию
        if simulator.position_type != 'FLAT' and bars:
            simulator.close_position(datetime.now(), bars[-1]['cur'])

        # Возвращаем общую доходность как фитнес
        total_return = ((simulator.balance - simulator.initial_balance) / simulator.initial_balance) * 100

        # Добавляем бонус за стабильность
        stability_bonus = 0
        if len(simulator.trades) > 5:  # Если есть достаточно сделок
            win_rate = len([t for t in simulator.trades if t.pnl > 0]) / len(simulator.trades)
            if win_rate > 0.4:  # Бонус за винрейт > 40%
                stability_bonus = 10

        # Штраф за просадку
        drawdown_penalty = simulator.max_drawdown * 100

        fitness = total_return + stability_bonus - drawdown_penalty

        return (max(fitness, -100),)

    except Exception as e:
        return (-100,)


# Модифицированная функция main() для использования с балансом
def main_with_balance():
    """Основная функция с интеграцией баланса"""
    import random
    from deap import tools, algorithms
    random.seed(42)

    # Создаем популяции (предполагается что toolbox'ы уже определены)
    # pop_long = toolbox_long.population(n=50)  # Уменьшаем размер для скорости
    # pop_short = toolbox_short.population(n=50)
    # pop_meta = toolbox_meta.population(n=50)

    # Hall of Fame для каждой популяции
    # hof_long = tools.HallOfFame(3)
    # hof_short = tools.HallOfFame(3)
    # hof_meta = tools.


def plot_signals(df, bars, best_long, best_short, best_meta, pset_long, pset_short, pset_meta):
    """Интерпретатор сигналов для трех популяций"""
    long_func = gp.compile(best_long, pset_long)
    short_func = gp.compile(best_short, pset_short)
    meta_func = gp.compile(best_meta, pset_meta)

    long_entries, long_exits = [], []
    short_entries, short_exits = [], []
    pos = None
    entry_price = 0.0
    total_profit = 0.0
    commission = 0.001

    for i, b in enumerate(bars):
        try:
            # Получаем сигналы от каждой популяции
            long_sig = long_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            short_sig = short_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])

            # Нормализуем к булевым значениям
            long_bool = 1.0 if long_sig > 0 else 0.0
            short_bool = 1.0 if short_sig > 0 else 0.0

            # Мета-популяция принимает решение
            final_sig = meta_func(long_bool, short_bool)

            # Интерпретируем финальный сигнал
            if final_sig > 0.5:
                sig = 1  # Long
            elif final_sig < -0.5:
                sig = -1  # Short
            else:
                sig = 0  # Hold

        except:
            sig = 0

        # Логика торговли с правильной обработкой состояний
        if pos == "long":
            if sig == -1:  # Выход из лонга
                profit = (b["cur"] - entry_price) - commission * (entry_price + b["cur"])
                total_profit += profit
                long_exits.append((i, b["cur"]))
                pos = None
            # Если sig == 1, остаемся в лонге (ничего не делаем)
            # Если sig == 0, тоже остаемся в лонге
        elif pos == "short":
            if sig == 1:  # Выход из шорта
                profit = (entry_price - b["cur"]) - commission * (entry_price + b["cur"])
                total_profit += profit
                short_exits.append((i, b["cur"]))
                pos = None
            # Если sig == -1, остаемся в шорте (ничего не делаем)
            # Если sig == 0, тоже остаемся в шорте
        else:  # pos is None
            if sig == 1:  # Вход в лонг
                long_entries.append((i, b["cur"]))
                entry_price = b["cur"]
                pos = "long"
            elif sig == -1:  # Вход в шорт
                short_entries.append((i, b["cur"]))
                entry_price = b["cur"]
                pos = "short"
            # Если sig == 0, остаемся без позиции

    # Закрываем последнюю позицию
    if pos == "long" and bars:
        profit = (bars[-1]["cur"] - entry_price) - commission * (entry_price + bars[-1]["cur"])
        total_profit += profit
        long_exits.append((len(bars) - 1, bars[-1]["cur"]))
    elif pos == "short" and bars:
        profit = (entry_price - bars[-1]["cur"]) - commission * (entry_price + bars[-1]["cur"])
        total_profit += profit
        short_exits.append((len(bars) - 1, bars[-1]["cur"]))

    # Построение графика
    offset = len(df) - len(bars)
    prices = df["Price"].iloc[offset:].values
    plt.figure(figsize=(16, 8))
    plt.plot(prices, label="Price", color='black', linewidth=1)

    if long_entries:
        xs, ys = zip(*long_entries)
        plt.scatter(xs, ys, marker="^", s=100, color='green', label="LONG ENTRY")
    if long_exits:
        xs, ys = zip(*long_exits)
        plt.scatter(xs, ys, marker="v", s=100, color='red', label="LONG EXIT")

    if short_entries:
        xs, ys = zip(*short_entries)
        plt.scatter(xs, ys, marker="v", s=100, color='orange', label="SHORT ENTRY")
    if short_exits:
        xs, ys = zip(*short_exits)
        plt.scatter(xs, ys, marker="^", s=100, color='blue', label="SHORT EXIT")

    plt.legend()
    plt.grid(True)
    plt.title(f"Three Population Trading Signals - Total Profit: {total_profit:.2f}")
    plt.show()


    trades = len(long_entries) + len(short_entries)
    print(f"Trades: {trades}, Total Profit: {total_profit:.2f}")
    if trades > 0:
        print(f"Profit per trade: {total_profit / trades:.2f}")

# ----- 1. Загрузка исторических данных и расчёт индикаторов -----
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


# Подготовка данных
ndf = get_history_data("BTCUSDT", "1h", "01-10-2020", "06-01-2020")
ndf = add_indicators(ndf, window=50)
bars = build_input_vectors(ndf, min_window=50)

# ----- 2. Определение типов и примитивов GP -----
VECTOR = list
SCALAR = float
BOOL = float  # Для мета-популяции


# Примитивы для работы с векторами
def vec_add(a: VECTOR, b: VECTOR) -> VECTOR:
    return [x + y for x, y in zip(a, b)]

def vec_sub(a: VECTOR, b: VECTOR) -> VECTOR:
    return [x - y for x, y in zip(a, b)]

def vec_mul(a: VECTOR, b: VECTOR) -> VECTOR:
    return [x * y for x, y in zip(a, b)]

def vec_div(a: VECTOR, b: VECTOR) -> VECTOR:
    return [x / (y if abs(y) > 1e-10 else 1e-10) for x, y in zip(a, b)]

def vec_add_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [x + b for x in a]

def vec_sub_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [x - b for x in a]

def vec_mul_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [x * b for x in a]

def vec_div_s(a: VECTOR, b: SCALAR) -> VECTOR:
    divisor = b if abs(b) > 1e-10 else 1e-10
    return [x / divisor for x in a]

def vec_pow_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [pow(abs(x), b) * (1 if x >= 0 else -1) for x in a]

def vec_abs(a: VECTOR) -> VECTOR:
    return [abs(x) for x in a]

def vec_neg(a: VECTOR) -> VECTOR:
    return [-x for x in a]

def vec_sin(a: VECTOR) -> VECTOR:
    return [math.sin(x) for x in a]

def vec_cos(a: VECTOR) -> VECTOR:
    return [math.cos(x) for x in a]

def vec_exp(a: VECTOR) -> VECTOR:
    return [math.exp(min(x, 100)) for x in a]  # Ограничиваем для избежания переполнения

def vec_log(a: VECTOR) -> VECTOR:
    return [math.log(abs(x) + 1e-10) for x in a]

def vec_sqrt(a: VECTOR) -> VECTOR:
    return [math.sqrt(abs(x)) for x in a]

def vec_clip01(a: VECTOR) -> VECTOR:
    return [min(max(x, 0.0), 1.0) for x in a]

def vec_concat(a: VECTOR, b: VECTOR) -> VECTOR:
    return a + b

def vec_slice(a: VECTOR, start: SCALAR, end: SCALAR) -> VECTOR:
    s = max(0, min(int(start), len(a)))
    e = max(s, min(int(end), len(a)))
    return a[s:e] if s < len(a) else []

def vec_reverse(a: VECTOR) -> VECTOR:
    return a[::-1]

def vec_sort(a: VECTOR) -> VECTOR:
    return sorted(a)

def vec_max_elem(a: VECTOR) -> SCALAR:
    return max(a) if len(a) > 0 else 0.0

def vec_min_elem(a: VECTOR) -> SCALAR:
    return min(a) if len(a) > 0 else 0.0

def vec_sum(a: VECTOR) -> SCALAR:
    return sum(a)

def vec_prod(a: VECTOR) -> SCALAR:
    result = 1.0
    for x in a:
        result *= x
    return result

def vec_length(a: VECTOR) -> SCALAR:
    return float(len(a))

def vec_dot(a: VECTOR, b: VECTOR) -> SCALAR:
    return sum(x * y for x, y in zip(a, b))

def vec_norm(a: VECTOR) -> SCALAR:
    return math.sqrt(sum(x * x for x in a))

def vec_distance(a: VECTOR, b: VECTOR) -> SCALAR:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# Скалярные операции
def scalar_mean(a: VECTOR) -> SCALAR:
    return sum(a) / len(a) if len(a) > 0 else 0.0

def scalar_median(a: VECTOR) -> SCALAR:
    if len(a) == 0:
        return 0.0
    sorted_a = sorted(a)
    n = len(sorted_a)
    if n % 2 == 0:
        return (sorted_a[n//2 - 1] + sorted_a[n//2]) / 2
    else:
        return sorted_a[n//2]

def scalar_std(a: VECTOR) -> SCALAR:
    if len(a) <= 1:
        return 0.0
    mean = sum(a) / len(a)
    variance = sum((x - mean) ** 2 for x in a) / len(a)
    return math.sqrt(variance)

def first_elem(a: VECTOR) -> SCALAR:
    return a[0] if len(a) > 0 else 0.0

def last_elem(a: VECTOR) -> SCALAR:
    return a[-1] if len(a) > 0 else 0.0

def nth_elem(a: VECTOR, n: SCALAR) -> SCALAR:
    idx = int(n) % len(a) if len(a) > 0 else 0
    return a[idx] if len(a) > 0 else 0.0

def sum_gt(a: VECTOR, b: VECTOR) -> SCALAR:
    return 1.0 if sum(a) > sum(b) else 0.0

def mean_gt(a: VECTOR, b: SCALAR) -> SCALAR:
    mean_a = sum(a) / len(a) if len(a) > 0 else 0.0
    return 1.0 if mean_a > b else 0.0

def rnd_mean_gt(a: VECTOR, low: SCALAR, high: SCALAR) -> SCALAR:
    r = random.uniform(low, high)
    mean_a = sum(a) / len(a) if len(a) > 0 else 0.0
    return 1.0 if mean_a > r else 0.0

def scalar_add(a: SCALAR, b: SCALAR) -> SCALAR:
    return a + b

def scalar_sub(a: SCALAR, b: SCALAR) -> SCALAR:
    return a - b

def scalar_mul(a: SCALAR, b: SCALAR) -> SCALAR:
    return a * b

def scalar_div(a: SCALAR, b: SCALAR) -> SCALAR:
    return a / b if abs(b) > 1e-10 else a / 1e-10

def scalar_pow(a: SCALAR, b: SCALAR) -> SCALAR:
    return pow(abs(a), b) * (1 if a >= 0 else -1)

def scalar_mod(a: SCALAR, b: SCALAR) -> SCALAR:
    return a % b if abs(b) > 1e-10 else 0.0

def scalar_abs(a: SCALAR) -> SCALAR:
    return abs(a)

def scalar_sin(a: SCALAR) -> SCALAR:
    return math.sin(a)

def scalar_cos(a: SCALAR) -> SCALAR:
    return math.cos(a)

def scalar_tan(a: SCALAR) -> SCALAR:
    return math.tan(a)

def scalar_exp(a: SCALAR) -> SCALAR:
    return math.exp(min(a, 100))

def scalar_sqrt(a: SCALAR) -> SCALAR:
    return math.sqrt(abs(a))

def scalar_floor(a: SCALAR) -> SCALAR:
    return float(math.floor(a))

def scalar_ceil(a: SCALAR) -> SCALAR:
    return float(math.ceil(a))

def scalar_round(a: SCALAR) -> SCALAR:
    return float(round(a))

def scalar_gt(a: SCALAR, b: SCALAR) -> SCALAR:
    return 1.0 if a > b else 0.0

def scalar_lt(a: SCALAR, b: SCALAR) -> SCALAR:
    return 1.0 if a < b else 0.0

def scalar_gte(a: SCALAR, b: SCALAR) -> SCALAR:
    return 1.0 if a >= b else 0.0

def scalar_lte(a: SCALAR, b: SCALAR) -> SCALAR:
    return 1.0 if a <= b else 0.0

def scalar_eq(a: SCALAR, b: SCALAR) -> SCALAR:
    return 1.0 if abs(a - b) < 1e-6 else 0.0

def scalar_neq(a: SCALAR, b: SCALAR) -> SCALAR:
    return 1.0 if abs(a - b) >= 1e-6 else 0.0

def scalar_pos(a: SCALAR) -> SCALAR:
    return 1.0 if a > 0 else 0.0

def scalar_neg(a: SCALAR) -> SCALAR:
    return 1.0 if a < 0 else 0.0

def scalar_zero(a: SCALAR) -> SCALAR:
    return 1.0 if abs(a) < 1e-6 else 0.0

def scalar_clip01(a: SCALAR) -> SCALAR:
    return min(max(a, 0.0), 1.0)

def scalar_clip(a: SCALAR, low: SCALAR, high: SCALAR) -> SCALAR:
    return min(max(a, low), high)

def scalar_log(a: SCALAR) -> SCALAR:
    return math.log(abs(a) + 1e-10)

def scalar_sigmoid(a: SCALAR) -> SCALAR:
    return 1.0 / (1.0 + math.exp(-min(max(a, -100), 100)))

def scalar_tanh(a: SCALAR) -> SCALAR:
    return math.tanh(a)

def scalar_random() -> SCALAR:
    return random.random()

def scalar_random_range(low: SCALAR, high: SCALAR) -> SCALAR:
    return random.uniform(low, high)

def scalar_random_gauss(mean: SCALAR, std: SCALAR) -> SCALAR:
    return random.gauss(mean, std)

# Условные операции
def if_else(cond: SCALAR, a: VECTOR, b: VECTOR) -> VECTOR:
    return a if cond > 0 else b

def scalar_if_else(cond: SCALAR, a: SCALAR, b: SCALAR) -> SCALAR:
    return a if cond > 0 else b

def vec_if_else_elem(cond: VECTOR, a: VECTOR, b: VECTOR) -> VECTOR:
    return [av if cv > 0 else bv for cv, av, bv in zip(cond, a, b)]

# Булевые операции
def bool_and(a: BOOL, b: BOOL) -> BOOL:
    return 1.0 if (a > 0.5 and b > 0.5) else 0.0

def bool_or(a: BOOL, b: BOOL) -> BOOL:
    return 1.0 if (a > 0.5 or b > 0.5) else 0.0

def bool_not(a: BOOL) -> BOOL:
    return 1.0 if a <= 0.5 else 0.0

def bool_xor(a: BOOL, b: BOOL) -> BOOL:
    return 1.0 if ((a > 0.5) != (b > 0.5)) else 0.0

def bool_nand(a: BOOL, b: BOOL) -> BOOL:
    return bool_not(bool_and(a, b))

def bool_nor(a: BOOL, b: BOOL) -> BOOL:
    return bool_not(bool_or(a, b))

def bool_if_then_else(cond: BOOL, a: BOOL, b: BOOL) -> BOOL:
    return a if cond > 0.5 else b

def bool_from_scalar(a: SCALAR) -> BOOL:
    return 1.0 if a > 0 else 0.0

def scalar_from_bool(a: BOOL) -> SCALAR:
    return a

# Векторные булевы операции
def vec_bool_and(a: VECTOR, b: VECTOR) -> VECTOR:
    return [bool_and(x, y) for x, y in zip(a, b)]

def vec_bool_or(a: VECTOR, b: VECTOR) -> VECTOR:
    return [bool_or(x, y) for x, y in zip(a, b)]

def vec_bool_not(a: VECTOR) -> VECTOR:
    return [bool_not(x) for x in a]

def vec_bool_xor(a: VECTOR, b: VECTOR) -> VECTOR:
    return [bool_xor(x, y) for x, y in zip(a, b)]

# Операции сравнения векторов
def vec_gt(a: VECTOR, b: VECTOR) -> VECTOR:
    return [scalar_gt(x, y) for x, y in zip(a, b)]

def vec_lt(a: VECTOR, b: VECTOR) -> VECTOR:
    return [scalar_lt(x, y) for x, y in zip(a, b)]

def vec_gte(a: VECTOR, b: VECTOR) -> VECTOR:
    return [scalar_gte(x, y) for x, y in zip(a, b)]

def vec_lte(a: VECTOR, b: VECTOR) -> VECTOR:
    return [scalar_lte(x, y) for x, y in zip(a, b)]

def vec_eq(a: VECTOR, b: VECTOR) -> VECTOR:
    return [scalar_eq(x, y) for x, y in zip(a, b)]

def vec_neq(a: VECTOR, b: VECTOR) -> VECTOR:
    return [scalar_neq(x, y) for x, y in zip(a, b)]

def vec_gt_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [scalar_gt(x, b) for x in a]

def vec_lt_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [scalar_lt(x, b) for x in a]

def vec_eq_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [scalar_eq(x, b) for x in a]

# Создание векторов
def vec_create_zeros(size: SCALAR) -> VECTOR:
    return [0.0] * max(0, int(size))

def vec_create_ones(size: SCALAR) -> VECTOR:
    return [1.0] * max(0, int(size))

def vec_create_range(start: SCALAR, end: SCALAR) -> VECTOR:
    s, e = int(start), int(end)
    return list(range(s, e)) if s < e else []

def vec_create_random(size: SCALAR) -> VECTOR:
    return [random.random() for _ in range(max(0, int(size)))]

def vec_create_const(size: SCALAR, value: SCALAR) -> VECTOR:
    return [value] * max(0, int(size))

# Агрегатные операции
def vec_all_gt(a: VECTOR, b: SCALAR) -> SCALAR:
    return 1.0 if all(x > b for x in a) else 0.0

def vec_any_gt(a: VECTOR, b: SCALAR) -> SCALAR:
    return 1.0 if any(x > b for x in a) else 0.0

def vec_count_gt(a: VECTOR, b: SCALAR) -> SCALAR:
    return float(sum(1 for x in a if x > b))

def vec_all_pos(a: VECTOR) -> SCALAR:
    return 1.0 if all(x > 0 for x in a) else 0.0

def vec_any_neg(a: VECTOR) -> SCALAR:
    return 1.0 if any(x < 0 for x in a) else 0.0

def vec_count_zeros(a: VECTOR) -> SCALAR:
    return float(sum(1 for x in a if abs(x) < 1e-6))

# Преобразования типов
def vec_to_bool(a: VECTOR) -> VECTOR:
    return [bool_from_scalar(x) for x in a]

def bool_vec_to_scalar(a: VECTOR) -> SCALAR:
    return 1.0 if any(x > 0.5 for x in a) else 0.0

def bool_vec_all_true(a: VECTOR) -> SCALAR:
    return 1.0 if all(x > 0.5 for x in a) else 0.0


# ----- 3. Настройка DEAP GP для трех популяций -----

# Популяция 1: Long стратегии
pset_long = gp.PrimitiveSetTyped(
    "LONG",
    in_types=[VECTOR, VECTOR, VECTOR, VECTOR, SCALAR],
    ret_type=SCALAR,
    prefix="IN"
)

# Популяция 2: Short стратегии
pset_short = gp.PrimitiveSetTyped(
    "SHORT",
    in_types=[VECTOR, VECTOR, VECTOR, VECTOR, SCALAR],
    ret_type=SCALAR,
    prefix="IN"
)

# Популяция 3: Мета-стратегия (принимает решения на основе двух булевых сигналов)
pset_meta = gp.PrimitiveSetTyped(
    "META",
    in_types=[BOOL, BOOL],  # Входы от Long и Short популяций
    ret_type=SCALAR,
    prefix="SIG"
)

# Добавляем примитивы для всех популяций
for pset in [pset_long, pset_short]:
    # Эфемерная константа
    pset.addEphemeralConstant("rand", partial(random.uniform, -1.0, 1.0), SCALAR)


    # Арифметические примитивы
    def add(a: SCALAR, b: SCALAR) -> SCALAR:
        return a + b


    def sub(a: SCALAR, b: SCALAR) -> SCALAR:
        return a - b


    def mul(a: SCALAR, b: SCALAR) -> SCALAR:
        return a * b


    def safe_div(a: SCALAR, b: SCALAR) -> SCALAR:
        return a / b if abs(b) > 1e-10 else 0.0


    pset.addPrimitive(add, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(sub, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(mul, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(safe_div, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_gt, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_lt, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_eq, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_pos, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_neg, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_clip01, [SCALAR], SCALAR)

    # Скалярные агрегаторы
    pset.addPrimitive(last_elem, [VECTOR], SCALAR)
    pset.addPrimitive(scalar_mean, [VECTOR], SCALAR)
    pset.addPrimitive(scalar_log, [SCALAR], SCALAR)

    # Векторные операции
    pset.addPrimitive(vec_add, [VECTOR, VECTOR], VECTOR)
    pset.addPrimitive(vec_sub, [VECTOR, VECTOR], VECTOR)
    pset.addPrimitive(vec_add_s, [VECTOR, SCALAR], VECTOR)
    pset.addPrimitive(vec_sub_s, [VECTOR, SCALAR], VECTOR)
    pset.addPrimitive(vec_mul_s, [VECTOR, SCALAR], VECTOR)

    pset.addPrimitive(sum_gt, [VECTOR, VECTOR], SCALAR)
    pset.addPrimitive(mean_gt, [VECTOR, SCALAR], SCALAR)
    pset.addPrimitive(rnd_mean_gt, [VECTOR, SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(if_else, [SCALAR, VECTOR, VECTOR], VECTOR)

# Примитивы для мета-популяции
pset_meta.addPrimitive(bool_and, [BOOL, BOOL], BOOL)
pset_meta.addPrimitive(bool_or, [BOOL, BOOL], BOOL)
pset_meta.addPrimitive(bool_not, [BOOL], BOOL)
pset_meta.addPrimitive(bool_xor, [BOOL, BOOL], BOOL)
pset_meta.addPrimitive(bool_if_then_else, [BOOL, BOOL, BOOL], BOOL)


# Арифметические операции для мета-популяции
def meta_add(a: BOOL, b: BOOL) -> SCALAR:
    return a + b


def meta_sub(a: BOOL, b: BOOL) -> SCALAR:
    return a - b


def meta_mul(a: BOOL, b: BOOL) -> SCALAR:
    return a * b


pset_meta.addPrimitive(meta_add, [BOOL, BOOL], SCALAR)
pset_meta.addPrimitive(meta_sub, [BOOL, BOOL], SCALAR)
pset_meta.addPrimitive(meta_mul, [BOOL, BOOL], SCALAR)

# Константы для мета-популяции
pset_meta.addEphemeralConstant("meta_rand", partial(random.uniform, -2.0, 2.0), SCALAR)


# ----- 4. Фитнес-функции для каждой популяции -----

def evalLongTrading(ind, bars):
    """Фитнес для Long популяции - фокус на прибыльности лонгов"""
    func = gp.compile(ind, pset_long)

    profit = 0.0
    position = 0
    entry_price = 0.0
    commission = 0.001
    trades_count = 0

    for i, b in enumerate(bars):
        try:
            sig = func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            sig = 1 if sig > 0 else 0  # Только лонги
        except:
            sig = 0

        if position == 1 and sig == 0:  # Закрываем лонг
            trade_profit = (b["cur"] - entry_price) - commission * (entry_price + b["cur"])
            profit += trade_profit
            position = 0
            trades_count += 1
        elif position == 0 and sig == 1:  # Открываем лонг
            position = 1
            entry_price = b["cur"]

    # Закрываем финальную позицию
    if position == 1 and bars:
        trade_profit = (bars[-1]["cur"] - entry_price) - commission * (entry_price + bars[-1]["cur"])
        profit += trade_profit
        trades_count += 1

    if trades_count == 0:
        return (-1000.0,)

    return (profit,)


def evalShortTrading(ind, bars):
    """Фитнес для Short популяции - фокус на прибыльности шортов"""
    func = gp.compile(ind, pset_short)

    profit = 0.0
    position = 0
    entry_price = 0.0
    commission = 0.001
    short_cost = 0.00001
    trades_count = 0

    for i, b in enumerate(bars):
        try:
            sig = func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            sig = -1 if sig > 0 else 0  # Только шорты
        except:
            sig = 0

        if position == -1 and sig == 0:  # Закрываем шорт
            trade_profit = (entry_price - b["cur"]) - commission * (entry_price + b["cur"])
            profit += trade_profit
            position = 0
            trades_count += 1
        elif position == 0 and sig == -1:  # Открываем шорт
            position = -1
            entry_price = b["cur"]

        if position == -1:
            profit -= short_cost * b["cur"]

    # Закрываем финальную позицию
    if position == -1 and bars:
        trade_profit = (entry_price - bars[-1]["cur"]) - commission * (entry_price + bars[-1]["cur"])
        profit += trade_profit
        trades_count += 1

    if trades_count == 0:
        return (-1000.0,)

    return (profit,)


def evalMetaTrading(ind, bars, best_long_func, best_short_func):
    """Фитнес для мета-популяции - комбинирует сигналы от Long и Short"""
    meta_func = gp.compile(ind, pset_meta)

    profit = 0.0
    position = 0
    entry_price = 0.0
    commission = 0.001
    short_cost = 0.00001
    trades_count = 0

    for i, b in enumerate(bars):
        try:
            # Получаем сигналы от Long и Short популяций
            long_sig = best_long_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            short_sig = best_short_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])

            long_bool = 1.0 if long_sig > 0 else 0.0
            short_bool = 1.0 if short_sig > 0 else 0.0

            # Мета-решение
            final_sig = meta_func(long_bool, short_bool)

            if final_sig > 0.5:
                sig = 1  # Long
            elif final_sig < -0.5:
                sig = -1  # Short
            else:
                sig = 0  # Hold
        except:
            sig = 0

        # Торговая логика
        if position == 1 and sig <= 0:  # Закрываем лонг
            trade_profit = (b["cur"] - entry_price) - commission * (entry_price + b["cur"])
            profit += trade_profit
            position = 0
            trades_count += 1
        elif position == -1 and sig >= 0:  # Закрываем шорт
            trade_profit = (entry_price - b["cur"]) - commission * (entry_price + b["cur"])
            profit += trade_profit
            position = 0
            trades_count += 1

        if position == 0:
            if sig > 0:  # Открываем лонг
                position = 1
                entry_price = b["cur"]
            elif sig < 0:  # Открываем шорт
                position = -1
                entry_price = b["cur"]

        if position == -1:
            profit -= short_cost * b["cur"]

    # Закрываем финальную позицию
    if position != 0 and bars:
        final_price = bars[-1]["cur"]
        if position == 1:
            trade_profit = (final_price - entry_price) - commission * (entry_price + final_price)
        else:
            trade_profit = (entry_price - final_price) - commission * (entry_price + final_price)
        profit += trade_profit
        trades_count += 1

    if trades_count == 0:
        return (-1000.0,)

    return (profit,)


# ----- 5. Toolbox настройки для трех популяций -----

# Создаем классы для каждой популяции
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("LongIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)
creator.create("ShortIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)
creator.create("MetaIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Toolbox для Long популяции
toolbox_long = base.Toolbox()
toolbox_long.register("expr", gp.genHalfAndHalf, pset=pset_long, min_=1, max_=4)
toolbox_long.register("individual", tools.initIterate, creator.LongIndividual, toolbox_long.expr)
toolbox_long.register("population", tools.initRepeat, list, toolbox_long.individual)
toolbox_long.register("evaluate", evalLongTrading, bars=bars)
toolbox_long.register("select", tools.selTournament, tournsize=3)
toolbox_long.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox_long.register("mutate", gp.mutUniform, expr=toolbox_long.expr_mut, pset=pset_long)

# Toolbox для Short популяции
toolbox_short = base.Toolbox()
toolbox_short.register("expr", gp.genHalfAndHalf, pset=pset_short, min_=1, max_=4)
toolbox_short.register("individual", tools.initIterate, creator.ShortIndividual, toolbox_short.expr)
toolbox_short.register("population", tools.initRepeat, list, toolbox_short.individual)
toolbox_short.register("evaluate", evalShortTrading, bars=bars)
toolbox_short.register("select", tools.selTournament, tournsize=3)
toolbox_short.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox_short.register("mutate", gp.mutUniform, expr=toolbox_short.expr_mut, pset=pset_short)

# Toolbox для Meta популяции
toolbox_meta = base.Toolbox()
toolbox_meta.register("expr", gp.genHalfAndHalf, pset=pset_meta, min_=1, max_=3)
toolbox_meta.register("individual", tools.initIterate, creator.MetaIndividual, toolbox_meta.expr)
toolbox_meta.register("population", tools.initRepeat, list, toolbox_meta.individual)
toolbox_meta.register("select", tools.selTournament, tournsize=3)
toolbox_meta.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox_meta.register("mutate", gp.mutUniform, expr=toolbox_meta.expr_mut, pset=pset_meta)


# Кроссовер для всех популяций
def cx_type_safe_size_fair(ind1, ind2, max_delta=2):
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    idx1 = random.randrange(len(ind1))
    slice1 = ind1.searchSubtree(idx1)
    size1 = slice1.stop - slice1.start
    type1 = ind1[idx1].ret

    candidates = []
    for idx2 in range(len(ind2)):
        if ind2[idx2].ret == type1:
            slice2 = ind2.searchSubtree(idx2)
            size2 = slice2.stop - slice2.start
            if abs(size1 - size2) <= max_delta:
                candidates.append((idx2, slice2))

    if not candidates:
        return ind1, ind2

    idx2, slice2 = random.choice(candidates)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    return ind1, ind2


toolbox_long.register("mate", cx_type_safe_size_fair, max_delta=2)
toolbox_short.register("mate", cx_type_safe_size_fair, max_delta=2)
toolbox_meta.register("mate", cx_type_safe_size_fair, max_delta=2)

# Ограничения глубины
for tb in [toolbox_long, toolbox_short, toolbox_meta]:
    tb.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=6))
    tb.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=6))


# ----- 6. Основной эволюционный алгоритм -----
def main():
    random.seed(42)

    # Создаем популяции
    pop_long = toolbox_long.population(n=100)
    pop_short = toolbox_short.population(n=100)
    pop_meta = toolbox_meta.population(n=100)

    # Hall of Fame для каждой популяции
    hof_long = tools.HallOfFame(3)
    hof_short = tools.HallOfFame(3)
    hof_meta = tools.HallOfFame(3)

    # Статистика
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("Starting co-evolution with three populations...")

    # Коэволюция
    ngen = 20
    for gen in range(ngen):
        print(f"\nGeneration {gen + 1}/{ngen}")

        # Эволюция Long популяции
        print("Evolving Long population...")
        pop_long, _ = algorithms.eaSimple(
            pop_long, toolbox_long,
            cxpb=0.6, mutpb=0.3, ngen=1,
            stats=None, halloffame=hof_long,
            verbose=False
        )

        # Эволюция Short популяции
        print("Evolving Short population...")
        pop_short, _ = algorithms.eaSimple(
            pop_short, toolbox_short,
            cxpb=0.6, mutpb=0.3, ngen=1,
            stats=None, halloffame=hof_short,
            verbose=False
        )

        # Получаем лучших представителей для мета-популяции
        if hof_long and hof_short:
            best_long_func = gp.compile(hof_long[0], pset_long)
            best_short_func = gp.compile(hof_short[0], pset_short)

            # Регистрируем фитнес для мета-популяции с текущими лучшими
            toolbox_meta.register("evaluate", evalMetaTrading,
                                  bars=bars,
                                  best_long_func=best_long_func,
                                  best_short_func=best_short_func)

            # Эволюция Meta популяции
            print("Evolving Meta population...")
            pop_meta, _ = algorithms.eaSimple(
                pop_meta, toolbox_meta,
                cxpb=0.6, mutpb=0.3, ngen=1,
                stats=None, halloffame=hof_meta,
                verbose=False
            )

            # Выводим статистику текущего поколения
            long_fitness = [ind.fitness.values[0] for ind in pop_long if ind.fitness.valid]
            short_fitness = [ind.fitness.values[0] for ind in pop_short if ind.fitness.valid]
            meta_fitness = [ind.fitness.values[0] for ind in pop_meta if ind.fitness.valid]

            print(f"Long - Max: {max(long_fitness):.2f}, Avg: {np.mean(long_fitness):.2f}")
            print(f"Short - Max: {max(short_fitness):.2f}, Avg: {np.mean(short_fitness):.2f}")
            print(f"Meta - Max: {max(meta_fitness):.2f}, Avg: {np.mean(meta_fitness):.2f}")

    print("\n=== FINAL RESULTS ===")
    print(f"Best Long fitness: {hof_long[0].fitness.values[0]:.2f}")
    print(f"Best Long individual: {str(hof_long[0])}")
    print(f"\nBest Short fitness: {hof_short[0].fitness.values[0]:.2f}")
    print(f"Best Short individual: {str(hof_short[0])}")
    print(f"\nBest Meta fitness: {hof_meta[0].fitness.values[0]:.2f}")
    print(f"Best Meta individual: {str(hof_meta[0])}")

    # Показываем результаты лучшей комбинированной стратегии
    plot_signals(ndf, bars, hof_long[0], hof_short[0], hof_meta[0],
                 pset_long, pset_short, pset_meta)

    return (pop_long, pop_short, pop_meta), (hof_long, hof_short, hof_meta)


def interpret_signals(bars, long_individual, short_individual, meta_individual,
                      pset_long, pset_short, pset_meta):
    """
    Интерпретатор сигналов для анализа работы трех популяций
    Возвращает детальную информацию о сигналах
    """
    long_func = gp.compile(long_individual, pset_long)
    short_func = gp.compile(short_individual, pset_short)
    meta_func = gp.compile(meta_individual, pset_meta)

    signals_data = []

    for i, b in enumerate(bars[:100]):  # Анализируем первые 100 баров
        try:
            # Сигналы от каждой популяции
            long_raw = long_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            short_raw = short_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])

            # Преобразуем в булевы
            long_bool = 1.0 if long_raw > 0 else 0.0
            short_bool = 1.0 if short_raw > 0 else 0.0

            # Финальный сигнал от мета-популяции
            meta_raw = meta_func(long_bool, short_bool)

            # Интерпретация финального сигнала
            if meta_raw > 0.5:
                final_action = "LONG"
            elif meta_raw < -0.5:
                final_action = "SHORT"
            else:
                final_action = "HOLD"

            signals_data.append({
                'bar': i,
                'price': b["cur"],
                'long_raw': long_raw,
                'short_raw': short_raw,
                'long_bool': long_bool,
                'short_bool': short_bool,
                'meta_raw': meta_raw,
                'final_action': final_action
            })

        except Exception as e:
            signals_data.append({
                'bar': i,
                'price': b["cur"],
                'long_raw': 0,
                'short_raw': 0,
                'long_bool': 0,
                'short_bool': 0,
                'meta_raw': 0,
                'final_action': 'ERROR'
            })

    # Создаем DataFrame для анализа
    df_signals = pd.DataFrame(signals_data)

    print("\n=== SIGNAL ANALYSIS ===")
    print("Action distribution:")
    print(df_signals['final_action'].value_counts())

    print(f"\nLong signals activated: {df_signals['long_bool'].sum():.0f} times")
    print(f"Short signals activated: {df_signals['short_bool'].sum():.0f} times")

    print("\nFirst 10 signals:")
    print(df_signals[['bar', 'price', 'long_bool', 'short_bool', 'meta_raw', 'final_action']].head(10))

    return df_signals


def test_best_individuals():
    """
    Быстрое тестирование с лучшими найденными индивидами
    Замените строки ниже на ваших лучших индивидов из результатов эволюции
    """

    # ВСТАВЬТЕ СЮДА ЛУЧШИХ ИНДИВИДОВ ИЗ ВАШЕГО ЗАПУСКА
    # Примеры (замените на реальные результаты):
    best_long_str = "scalar_gt(last_elem(vec_sub(IN0, IN1)), scalar_mean(IN2))"
    best_short_str = "scalar_lt(scalar_mean(vec_add(IN0, IN3)), last_elem(IN1))"
    best_meta_str = "bool_and(SIG0, bool_not(SIG1))"

    print("=== TESTING BEST INDIVIDUALS ===")
    print(f"Best Long: {best_long_str}")
    print(f"Best Short: {best_short_str}")
    print(f"Best Meta: {best_meta_str}")
    print()

    # Создаем индивидов из строк
    try:
        best_long = gp.PrimitiveTree.from_string(best_long_str, pset_long)
        best_short = gp.PrimitiveTree.from_string(best_short_str, pset_short)
        best_meta = gp.PrimitiveTree.from_string(best_meta_str, pset_meta)

        # Компилируем функции
        long_func = gp.compile(best_long, pset_long)
        short_func = gp.compile(best_short, pset_short)
        meta_func = gp.compile(best_meta, pset_meta)

        # Тестируем каждую популяцию отдельно
        print("=== INDIVIDUAL POPULATION PERFORMANCE ===")

        # Тест Long популяции
        long_fitness = evalLongTrading(best_long, bars)
        print(f"Long Population Fitness: {long_fitness[0]:.2f}")

        # Тест Short популяции
        short_fitness = evalShortTrading(best_short, bars)
        print(f"Short Population Fitness: {short_fitness[0]:.2f}")

        # Тест Meta популяции
        meta_fitness = evalMetaTrading(best_meta, bars, long_func, short_func)
        print(f"Meta Population Fitness: {meta_fitness[0]:.2f}")

        print("\n=== COMBINED STRATEGY RESULTS ===")

        # Показываем график комбинированной стратегии
        plot_signals(ndf, bars, best_long, best_short, best_meta,
                     pset_long, pset_short, pset_meta)

        # Детальный анализ сигналов
        signals_df = analyze_signals_detailed(bars, best_long, best_short, best_meta,
                                              pset_long, pset_short, pset_meta)

        return best_long, best_short, best_meta, signals_df

    except Exception as e:
        print(f"Ошибка при создании индивидов: {e}")
        print("Убедитесь, что строки индивидов корректны и соответствуют primitive sets")
        return None, None, None, None


def analyze_signals_detailed(bars, long_individual, short_individual, meta_individual,
                             pset_long, pset_short, pset_meta, num_bars=50):
    """
    Детальный анализ работы сигналов
    """
    long_func = gp.compile(long_individual, pset_long)
    short_func = gp.compile(short_individual, pset_short)
    meta_func = gp.compile(meta_individual, pset_meta)

    signals_data = []

    print(f"\n=== DETAILED SIGNAL ANALYSIS (First {num_bars} bars) ===")
    print("Bar | Price  | Long_Raw | Short_Raw | Long_Bool | Short_Bool | Meta_Raw | Action")
    print("-" * 80)

    for i, b in enumerate(bars[:num_bars]):
        try:
            # Сигналы от каждой популяции
            long_raw = long_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            short_raw = short_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])

            # Преобразуем в булевы
            long_bool = 1.0 if long_raw > 0 else 0.0
            short_bool = 1.0 if short_raw > 0 else 0.0

            # Финальный сигнал от мета-популяции
            meta_raw = meta_func(long_bool, short_bool)

            # Интерпретация финального сигнала
            if meta_raw > 0.5:
                final_action = "LONG"
            elif meta_raw < -0.5:
                final_action = "SHORT"
            else:
                final_action = "HOLD"

            signals_data.append({
                'bar': i,
                'price': b["cur"],
                'long_raw': long_raw,
                'short_raw': short_raw,
                'long_bool': long_bool,
                'short_bool': short_bool,
                'meta_raw': meta_raw,
                'final_action': final_action
            })

            print(f"{i:3d} | {b['cur']:6.1f} | {long_raw:8.3f} | {short_raw:9.3f} | "
                  f"{long_bool:9.1f} | {short_bool:10.1f} | {meta_raw:8.3f} | {final_action}")

        except Exception as e:
            signals_data.append({
                'bar': i,
                'price': b["cur"],
                'long_raw': 0,
                'short_raw': 0,
                'long_bool': 0,
                'short_bool': 0,
                'meta_raw': 0,
                'final_action': 'ERROR'
            })
            print(f"{i:3d} | {b['cur']:6.1f} | ERROR: {str(e)[:50]}")

    # Создаем DataFrame для анализа
    df_signals = pd.DataFrame(signals_data)

    print(f"\n=== SIGNAL STATISTICS ===")
    print("Final Action Distribution:")
    action_counts = df_signals['final_action'].value_counts()
    for action, count in action_counts.items():
        print(f"  {action}: {count} ({count / len(df_signals) * 100:.1f}%)")

    print(f"\nLong signals activated: {df_signals['long_bool'].sum():.0f}/{len(df_signals)} "
          f"({df_signals['long_bool'].sum() / len(df_signals) * 100:.1f}%)")
    print(f"Short signals activated: {df_signals['short_bool'].sum():.0f}/{len(df_signals)} "
          f"({df_signals['short_bool'].sum() / len(df_signals) * 100:.1f}%)")

    return df_signals


def quick_test_with_strings(long_str, short_str, meta_str):
    """
    Быстрый тест с конкретными строками индивидов

    Параметры:
    long_str - строковое представление лучшего Long индивида
    short_str - строковое представление лучшего Short индивида
    meta_str - строковое представление лучшего Meta индивида
    """
    print("=== QUICK TEST WITH PROVIDED STRINGS ===")
    print(f"Long: {long_str}")
    print(f"Short: {short_str}")
    print(f"Meta: {meta_str}")

    try:
        # Создаем индивидов
        best_long = gp.PrimitiveTree.from_string(long_str, pset_long)
        best_short = gp.PrimitiveTree.from_string(short_str, pset_short)
        best_meta = gp.PrimitiveTree.from_string(meta_str, pset_meta)

        # Быстрая оценка производительности
        long_fitness = evalLongTrading(best_long, bars)
        short_fitness = evalShortTrading(best_short, bars)

        long_func = gp.compile(best_long, pset_long)
        short_func = gp.compile(best_short, pset_short)
        meta_fitness = evalMetaTrading(best_meta, bars, long_func, short_func)

        print(f"\nPerformance:")
        print(f"Long: {long_fitness[0]:.2f}")
        print(f"Short: {short_fitness[0]:.2f}")
        print(f"Meta: {meta_fitness[0]:.2f}")

        # Показываем результаты
        plot_signals(ndf, bars, best_long, best_short, best_meta,
                     pset_long, pset_short, pset_meta)

        return best_long, best_short, best_meta

    except Exception as e:
        print(f"Ошибка: {e}")
        return None, None, None


# Пример использования после получения лучших индивидов:
# После запуска main() скопируй строки лучших индивидов и используй:
#
# test_best_individuals()  # с предустановленными в функции
#
# ИЛИ
#
# quick_test_with_strings(
#     "твоя_строка_long_индивида",
#     "твоя_строка_short_индивида",
#     "твоя_строка_meta_индивида"
# )

if __name__ == '__main__':
    # Получение данных
    broker = BinanceBroker()
    data = broker.get_history_data(['BTCUSDT'], '1h', '2024-01-01', '2024-02-01')

    print(f"Loaded {len(data)} candles")
    print(data.head())

    # Создание симулятора
    simulator = TradingSimulator(initial_balance=10000, commission=0.001)


    # Пример простой стратегии для тестирования
    def simple_long_func(price, sma, ema, lwma, cur):
        return 1 if cur > sma[-1] else -1


    def simple_short_func(price, sma, ema, lwma, cur):
        return 1 if cur < sma[-1] else -1


    def simple_meta_func(long_bool, short_bool):
        if long_bool > 0.5:
            return 1
        elif short_bool > 0.5:
            return -1
        else:
            return 0


    # Запуск бэктеста (требует адаптации под ваши данные)
    print("\nExample of how to run backtest:")
    print(
        "signals_df, stats = backtest_strategy(data, simple_long_func, simple_short_func, simple_meta_func, simulator)")
    print("plot_backtest_results(data, signals_df, simulator)")

    print("\n=== STARTING GP EVOLUTION ===")
    (pops, hofs) = main()

