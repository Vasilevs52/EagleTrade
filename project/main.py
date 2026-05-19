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
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional


# =====================================================================
# BROKER & DATA
# =====================================================================

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


# =====================================================================
# SIMULATOR
# =====================================================================

@dataclass
class Trade:
    timestamp: datetime
    action: str
    price: float
    quantity: float
    balance_before: float
    balance_after: float
    pnl: float = 0.0
    position_type: str = "FLAT"


class TradingSimulator:
    def __init__(self, initial_balance: float = 10000, commission: float = 0.001,
                 risk_percent: float = 0.02):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.risk_percent = risk_percent       # доля капитала на позицию
        self.position = 0.0
        self.position_price = 0.0
        self.position_type = "FLAT"
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.max_balance = initial_balance
        self.max_drawdown = 0.0

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_price = 0.0
        self.position_type = "FLAT"
        self.trades = []
        self.equity_curve = []
        self.max_balance = self.initial_balance
        self.max_drawdown = 0.0

    def execute_trade(self, timestamp, action, price, risk_percent=None):
        if risk_percent is None:
            risk_percent = self.risk_percent
        balance_before = self.balance
        pnl = 0.0

        if action == 'HOLD' or self.balance <= 0:
            self.equity_curve.append(self.balance)
            return None

        if action == 'LONG':
            if self.position_type == 'SHORT':
                pnl = self.position * (self.position_price - price)
                self.balance += pnl
                self.balance -= abs(self.position * price * self.commission)

                risk_amount = self.balance * risk_percent
                quantity = risk_amount / price
                self.position = quantity
                self.position_price = price
                self.position_type = 'LONG'
                self.balance -= quantity * price * self.commission

            elif self.position_type == 'FLAT':
                risk_amount = self.balance * risk_percent
                quantity = risk_amount / price
                self.balance -= quantity * price * self.commission
                self.position = quantity
                self.position_price = price
                self.position_type = 'LONG'

        elif action == 'SHORT':
            if self.position_type == 'LONG':
                pnl = self.position * (price - self.position_price)
                self.balance += pnl
                self.balance -= abs(self.position * price * self.commission)

                risk_amount = self.balance * risk_percent
                quantity = risk_amount / price
                self.position = -quantity
                self.position_price = price
                self.position_type = 'SHORT'
                self.balance -= quantity * price * self.commission

            elif self.position_type == 'FLAT':
                risk_amount = self.balance * risk_percent
                quantity = risk_amount / price
                self.balance -= quantity * price * self.commission
                self.position = -quantity
                self.position_price = price
                self.position_type = 'SHORT'

        elif action == 'CLOSE':
            if self.position_type == 'LONG':
                pnl = self.position * (price - self.position_price)
            elif self.position_type == 'SHORT':
                pnl = abs(self.position) * (self.position_price - price)
            self.balance += pnl
            self.balance -= abs(self.position * price * self.commission)
            self.position = 0.0
            self.position_price = 0.0
            self.position_type = 'FLAT'

        self.max_balance = max(self.max_balance, self.balance)
        current_drawdown = (self.max_balance - self.balance) / self.max_balance if self.max_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        self.equity_curve.append(self.balance)

        trade = Trade(
            timestamp=timestamp, action=action, price=price,
            quantity=abs(self.position), balance_before=balance_before,
            balance_after=self.balance, pnl=pnl, position_type=self.position_type
        )
        self.trades.append(trade)
        return trade

    def close_position(self, timestamp, price):
        if self.position_type == 'FLAT':
            return None
        return self.execute_trade(timestamp, 'CLOSE', price)

    def get_unrealized_pnl(self, current_price):
        if self.position_type == 'FLAT':
            return 0.0
        elif self.position_type == 'LONG':
            return self.position * (current_price - self.position_price)
        else:
            return abs(self.position) * (self.position_price - current_price)

    def get_statistics(self):
        if not self.trades:
            return {}
        trades_with_pnl = [t for t in self.trades if t.pnl != 0]
        profitable = [t for t in trades_with_pnl if t.pnl > 0]
        losing = [t for t in trades_with_pnl if t.pnl < 0]
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        return {
            'Initial Balance': self.initial_balance,
            'Final Balance': self.balance,
            'Total Return (%)': total_return,
            'Max Drawdown (%)': self.max_drawdown * 100,
            'Total Trades': len(self.trades),
            'Profitable Trades': len(profitable),
            'Losing Trades': len(losing),
            'Win Rate (%)': (len(profitable) / len(trades_with_pnl)) * 100 if trades_with_pnl else 0,
            'Avg Profit': np.mean([t.pnl for t in profitable]) if profitable else 0,
            'Avg Loss': np.mean([t.pnl for t in losing]) if losing else 0,
        }


# =====================================================================
# GP PRIMITIVES
# =====================================================================

VECTOR = list
SCALAR = float
BOOL = float


def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def vec_mul(a, b):
    return [x * y for x, y in zip(a, b)]

def vec_add_s(a, b):
    return [x + b for x in a]

def vec_sub_s(a, b):
    return [x - b for x in a]

def vec_mul_s(a, b):
    return [x * b for x in a]

def vec_div_s(a, b):
    divisor = b if abs(b) > 1e-10 else 1e-10
    return [x / divisor for x in a]

def vec_abs(a):
    return [abs(x) for x in a]

def vec_neg(a):
    return [-x for x in a]

def vec_sin(a):
    return [math.sin(x) for x in a]

def vec_cos(a):
    return [math.cos(x) for x in a]

def vec_log(a):
    return [math.log(abs(x) + 1e-10) for x in a]

def vec_sqrt(a):
    return [math.sqrt(abs(x)) for x in a]

def last_elem(a):
    return a[-1] if len(a) > 0 else 0.0

def first_elem(a):
    return a[0] if len(a) > 0 else 0.0

def scalar_mean(a):
    return sum(a) / len(a) if len(a) > 0 else 0.0

def scalar_std(a):
    if len(a) <= 1:
        return 0.0
    mean = sum(a) / len(a)
    return math.sqrt(sum((x - mean) ** 2 for x in a) / len(a))

def vec_max_elem(a):
    return max(a) if len(a) > 0 else 0.0

def vec_min_elem(a):
    return min(a) if len(a) > 0 else 0.0

def vec_sum(a):
    return sum(a)

def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def vec_norm(a):
    return math.sqrt(sum(x * x for x in a))

def sum_gt(a, b):
    return 1.0 if sum(a) > sum(b) else 0.0

def mean_gt(a, b):
    return 1.0 if (sum(a) / len(a) if len(a) > 0 else 0.0) > b else 0.0

def rnd_mean_gt(a, low, high):
    # Детерминистская версия: используем среднее (low+high)/2 вместо random
    # чтобы фитнес одного и того же индивида был стабильным
    threshold = (low + high) / 2.0
    return 1.0 if (sum(a) / len(a) if len(a) > 0 else 0.0) > threshold else 0.0

def scalar_add(a, b):
    return a + b

def scalar_sub(a, b):
    return a - b

def scalar_mul(a, b):
    return a * b

def scalar_div(a, b):
    return a / b if abs(b) > 1e-10 else a / 1e-10

def scalar_abs(a):
    return abs(a)

def scalar_sin(a):
    return math.sin(a)

def scalar_cos(a):
    return math.cos(a)

def scalar_exp(a):
    return math.exp(min(a, 100))

def scalar_sqrt(a):
    return math.sqrt(abs(a))

def scalar_log(a):
    return math.log(abs(a) + 1e-10)

def scalar_sigmoid(a):
    return 1.0 / (1.0 + math.exp(-min(max(a, -100), 100)))

def scalar_tanh(a):
    return math.tanh(a)

def scalar_gt(a, b):
    return 1.0 if a > b else 0.0

def scalar_lt(a, b):
    return 1.0 if a < b else 0.0

def scalar_gte(a, b):
    return 1.0 if a >= b else 0.0

def scalar_lte(a, b):
    return 1.0 if a <= b else 0.0

def scalar_eq(a, b):
    return 1.0 if abs(a - b) < 1e-6 else 0.0

def scalar_pos(a):
    return 1.0 if a > 0 else 0.0

def scalar_neg_check(a):
    return 1.0 if a < 0 else 0.0

def scalar_clip01(a):
    return min(max(a, 0.0), 1.0)

def if_else(cond, a, b):
    return a if cond > 0 else b

def scalar_if_else(cond, a, b):
    return a if cond > 0 else b

# Булевые для мета-популяции
def bool_and(a, b):
    return 1.0 if (a > 0.5 and b > 0.5) else 0.0

def bool_or(a, b):
    return 1.0 if (a > 0.5 or b > 0.5) else 0.0

def bool_not(a):
    return 1.0 if a <= 0.5 else 0.0

def bool_xor(a, b):
    return 1.0 if ((a > 0.5) != (b > 0.5)) else 0.0

def bool_if_then_else(cond, a, b):
    return a if cond > 0.5 else b


# =====================================================================
# PSET SETUP
# =====================================================================

# Long population: вход — векторы + скаляр, выход — SCALAR (>0 = позиция открыта, <=0 = закрыта)
pset_long = gp.PrimitiveSetTyped("LONG", [VECTOR, VECTOR, VECTOR, VECTOR, SCALAR], SCALAR, "IN")

# Short population: аналогично
pset_short = gp.PrimitiveSetTyped("SHORT", [VECTOR, VECTOR, VECTOR, VECTOR, SCALAR], SCALAR, "IN")

# Meta population: 2 булевых входа (long_active, short_active), выход — SCALAR
pset_meta = gp.PrimitiveSetTyped("META", [BOOL, BOOL], SCALAR, "SIG")

# Регистрируем примитивы для long и short
for pset in [pset_long, pset_short]:
    pset.addEphemeralConstant("rand", partial(random.uniform, -1.0, 1.0), SCALAR)

    pset.addPrimitive(scalar_add, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_sub, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_mul, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_div, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_gt, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_lt, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_eq, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_pos, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_neg_check, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_clip01, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_if_else, [SCALAR, SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_log, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_sigmoid, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_tanh, [SCALAR], SCALAR)

    pset.addPrimitive(last_elem, [VECTOR], SCALAR)
    pset.addPrimitive(first_elem, [VECTOR], SCALAR)
    pset.addPrimitive(scalar_mean, [VECTOR], SCALAR)
    pset.addPrimitive(scalar_std, [VECTOR], SCALAR)
    pset.addPrimitive(vec_max_elem, [VECTOR], SCALAR)
    pset.addPrimitive(vec_min_elem, [VECTOR], SCALAR)
    pset.addPrimitive(vec_sum, [VECTOR], SCALAR)

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

def meta_add(a, b):
    return a + b

def meta_sub(a, b):
    return a - b

def meta_mul(a, b):
    return a * b

def meta_neg(a):
    return -a

pset_meta.addPrimitive(meta_add, [BOOL, BOOL], SCALAR)
pset_meta.addPrimitive(meta_sub, [BOOL, BOOL], SCALAR)
pset_meta.addPrimitive(meta_mul, [BOOL, BOOL], SCALAR)
pset_meta.addPrimitive(meta_neg, [BOOL], SCALAR)
pset_meta.addPrimitive(scalar_if_else, [BOOL, SCALAR, SCALAR], SCALAR)
pset_meta.addEphemeralConstant("meta_rand", partial(random.uniform, -2.0, 2.0), SCALAR)


# =====================================================================
# FITNESS FUNCTIONS
# Новая логика:
#   Long/Short выдают true/false на каждом баре.
#   true = позиция открыта, false = позиция закрыта.
#   Профит считается за время удержания позиции.
# =====================================================================

def evalLongTrading(ind, bars):
    """
    Фитнес Long популяции.
    Сигнал > 0 = лонг открыт (true), <= 0 = лонг закрыт (false).
    Профит считается в процентах: (exit - entry) / entry * 100 - комиссия.
    """
    func = gp.compile(ind, pset_long)

    total_pct = 0.0
    commission = 0.001  # 0.1% за сделку (вход + выход = 0.2%)
    was_open = False
    entry_price = 0.0
    trades_count = 0

    for b in bars:
        try:
            sig = func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            is_open = sig > 0
        except:
            is_open = False

        if is_open and not was_open:
            entry_price = b["cur"]
            was_open = True
        elif not is_open and was_open:
            # Процентный профит за сделку минус комиссия входа и выхода
            pct = ((b["cur"] - entry_price) / entry_price) * 100 - commission * 2 * 100
            total_pct += pct
            was_open = False
            trades_count += 1

    if was_open and bars:
        pct = ((bars[-1]["cur"] - entry_price) / entry_price) * 100 - commission * 2 * 100
        total_pct += pct
        trades_count += 1

    if trades_count == 0:
        return (-1000.0,)

    return (total_pct,)


def evalShortTrading(ind, bars):
    """
    Фитнес Short популяции.
    Сигнал > 0 = шорт открыт (true), <= 0 = шорт закрыт (false).
    Профит в процентах: (entry - exit) / entry * 100 - комиссия - стоимость удержания.
    """
    func = gp.compile(ind, pset_short)

    total_pct = 0.0
    commission = 0.001  # 0.1%
    short_cost_per_bar = 0.001  # 0.001% за бар удержания шорта (funding rate)
    was_open = False
    entry_price = 0.0
    bars_held = 0
    trades_count = 0

    for b in bars:
        try:
            sig = func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            is_open = sig > 0
        except:
            is_open = False

        if is_open and not was_open:
            entry_price = b["cur"]
            bars_held = 1
            was_open = True
        elif is_open and was_open:
            bars_held += 1
        elif not is_open and was_open:
            pct = ((entry_price - b["cur"]) / entry_price) * 100 - commission * 2 * 100 - short_cost_per_bar * bars_held
            total_pct += pct
            was_open = False
            trades_count += 1

    if was_open and bars:
        pct = ((entry_price - bars[-1]["cur"]) / entry_price) * 100 - commission * 2 * 100 - short_cost_per_bar * bars_held
        total_pct += pct
        trades_count += 1

    if trades_count == 0:
        return (-1000.0,)

    return (total_pct,)


def evalMetaTrading(ind, bars, best_long_func, best_short_func):
    """
    Фитнес Meta популяции.
    Получает 2 булевых сигнала: long_active и short_active.
    Решает итоговое действие:
      > 0.5  -> LONG
      < -0.5 -> SHORT
      иначе  -> FLAT
    Профит в процентах с комиссией и short_cost.
    """
    meta_func = gp.compile(ind, pset_meta)

    total_pct = 0.0
    commission = 0.001
    short_cost_per_bar = 0.001  # 0.001% за бар удержания шорта
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    bars_held = 0
    trades_count = 0

    for b in bars:
        try:
            long_raw = best_long_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            short_raw = best_short_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])

            long_active = 1.0 if long_raw > 0 else 0.0
            short_active = 1.0 if short_raw > 0 else 0.0

            meta_sig = meta_func(long_active, short_active)

            if meta_sig > 0.5:
                desired = 1
            elif meta_sig < -0.5:
                desired = -1
            else:
                desired = 0
        except:
            desired = 0

        if position != desired:
            # Закрываем текущую
            if position == 1 and entry_price > 0:
                pct = ((b["cur"] - entry_price) / entry_price) * 100 - commission * 2 * 100
                total_pct += pct
                trades_count += 1
            elif position == -1 and entry_price > 0:
                pct = ((entry_price - b["cur"]) / entry_price) * 100 - commission * 2 * 100 - short_cost_per_bar * bars_held
                total_pct += pct
                trades_count += 1

            # Открываем новую
            if desired != 0:
                entry_price = b["cur"]
                bars_held = 0
            position = desired

        if position == -1:
            bars_held += 1

    # Закрываем в конце
    if position != 0 and bars and entry_price > 0:
        final_price = bars[-1]["cur"]
        if position == 1:
            pct = ((final_price - entry_price) / entry_price) * 100 - commission * 2 * 100
        else:
            pct = ((entry_price - final_price) / entry_price) * 100 - commission * 2 * 100 - short_cost_per_bar * bars_held
        total_pct += pct
        trades_count += 1

    if trades_count == 0:
        return (-1000.0,)

    return (total_pct,)


# =====================================================================
# DEAP TOOLBOX
# =====================================================================

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("LongIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)
creator.create("ShortIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)
creator.create("MetaIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox_long = base.Toolbox()
toolbox_long.register("expr", gp.genHalfAndHalf, pset=pset_long, min_=1, max_=4)
toolbox_long.register("individual", tools.initIterate, creator.LongIndividual, toolbox_long.expr)
toolbox_long.register("population", tools.initRepeat, list, toolbox_long.individual)
toolbox_long.register("select", tools.selTournament, tournsize=3)
toolbox_long.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox_long.register("mutate", gp.mutUniform, expr=toolbox_long.expr_mut, pset=pset_long)

toolbox_short = base.Toolbox()
toolbox_short.register("expr", gp.genHalfAndHalf, pset=pset_short, min_=1, max_=4)
toolbox_short.register("individual", tools.initIterate, creator.ShortIndividual, toolbox_short.expr)
toolbox_short.register("population", tools.initRepeat, list, toolbox_short.individual)
toolbox_short.register("select", tools.selTournament, tournsize=3)
toolbox_short.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox_short.register("mutate", gp.mutUniform, expr=toolbox_short.expr_mut, pset=pset_short)

toolbox_meta = base.Toolbox()
toolbox_meta.register("expr", gp.genHalfAndHalf, pset=pset_meta, min_=1, max_=3)
toolbox_meta.register("individual", tools.initIterate, creator.MetaIndividual, toolbox_meta.expr)
toolbox_meta.register("population", tools.initRepeat, list, toolbox_meta.individual)
toolbox_meta.register("select", tools.selTournament, tournsize=3)
toolbox_meta.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox_meta.register("mutate", gp.mutUniform, expr=toolbox_meta.expr_mut, pset=pset_meta)


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

for tb in [toolbox_long, toolbox_short, toolbox_meta]:
    tb.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=6))
    tb.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=6))


# =====================================================================
# PLOT — новая визуализация
# =====================================================================

def plot_signals(df, bars, best_long, best_short, best_meta, pset_l, pset_s, pset_m, title_prefix=""):
    """
    Визуализация сигналов трёх популяций.
    Показывает:
      1) Цену + точки входа/выхода мета-стратегии
      2) Сигналы long/short популяций (true/false зоны)
      3) Мета-сигнал
    """
    long_func = gp.compile(best_long, pset_l)
    short_func = gp.compile(best_short, pset_s)
    meta_func = gp.compile(best_meta, pset_m)

    prices = []
    long_signals = []
    short_signals = []
    meta_signals = []

    long_entries, long_exits = [], []
    short_entries, short_exits = [], []

    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    total_pct = 0.0
    commission = 0.001
    short_cost_per_bar = 0.001
    bars_held = 0

    for i, b in enumerate(bars):
        prices.append(b["cur"])

        try:
            long_raw = long_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            short_raw = short_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])

            long_active = 1.0 if long_raw > 0 else 0.0
            short_active = 1.0 if short_raw > 0 else 0.0

            meta_raw = meta_func(long_active, short_active)

            if meta_raw > 0.5:
                desired = 1
            elif meta_raw < -0.5:
                desired = -1
            else:
                desired = 0
        except:
            long_active = 0.0
            short_active = 0.0
            meta_raw = 0.0
            desired = 0

        long_signals.append(long_active)
        short_signals.append(short_active)
        meta_signals.append(meta_raw)

        # Торговая логика
        if position != desired:
            # Закрываем текущую позицию
            if position == 1 and entry_price > 0:
                pct = ((b["cur"] - entry_price) / entry_price) * 100 - commission * 2 * 100
                total_pct += pct
                long_exits.append((i, b["cur"]))
            elif position == -1 and entry_price > 0:
                pct = ((entry_price - b["cur"]) / entry_price) * 100 - commission * 2 * 100 - short_cost_per_bar * bars_held
                total_pct += pct
                short_exits.append((i, b["cur"]))

            # Открываем новую
            if desired == 1:
                long_entries.append((i, b["cur"]))
                entry_price = b["cur"]
                bars_held = 0
            elif desired == -1:
                short_entries.append((i, b["cur"]))
                entry_price = b["cur"]
                bars_held = 0

            position = desired

        if position == -1:
            bars_held += 1

    # Закрываем в конце
    if position == 1 and bars and entry_price > 0:
        pct = ((bars[-1]["cur"] - entry_price) / entry_price) * 100 - commission * 2 * 100
        total_pct += pct
        long_exits.append((len(bars) - 1, bars[-1]["cur"]))
    elif position == -1 and bars and entry_price > 0:
        pct = ((entry_price - bars[-1]["cur"]) / entry_price) * 100 - commission * 2 * 100 - short_cost_per_bar * bars_held
        total_pct += pct
        short_exits.append((len(bars) - 1, bars[-1]["cur"]))

    # ----- РИСУЕМ -----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [3, 1, 1]})

    # === Subplot 1: Цена + сделки ===
    ax1.plot(prices, label="Price", color='black', linewidth=1, alpha=0.8)

    if long_entries:
        xs, ys = zip(*long_entries)
        ax1.scatter(xs, ys, marker="^", s=120, color='green', zorder=5, label="LONG ENTRY")
    if long_exits:
        xs, ys = zip(*long_exits)
        ax1.scatter(xs, ys, marker="v", s=120, color='darkgreen', zorder=5, label="LONG EXIT")
    if short_entries:
        xs, ys = zip(*short_entries)
        ax1.scatter(xs, ys, marker="v", s=120, color='red', zorder=5, label="SHORT ENTRY")
    if short_exits:
        xs, ys = zip(*short_exits)
        ax1.scatter(xs, ys, marker="^", s=120, color='darkred', zorder=5, label="SHORT EXIT")

    # Подсветка зон лонга/шорта на ценовом графике
    for i in range(len(prices) - 1):
        if long_signals[i] > 0.5:
            ax1.axvspan(i, i + 1, alpha=0.05, color='green')
        if short_signals[i] > 0.5:
            ax1.axvspan(i, i + 1, alpha=0.05, color='red')

    ax1.set_title(f"{title_prefix}Three Population Strategy — Total Return: {total_pct:.2f}%", fontsize=14)
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # === Subplot 2: Сигналы Long/Short популяций (true/false) ===
    ax2.fill_between(range(len(long_signals)), long_signals, alpha=0.4, color='green',
                     step='post', label='Long Active (true/false)')
    ax2.fill_between(range(len(short_signals)), [-s for s in short_signals], alpha=0.4, color='red',
                     step='post', label='Short Active (true/false)')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_ylabel("Population Signals")
    ax2.set_ylim(-1.3, 1.3)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # === Subplot 3: Мета-сигнал ===
    ax3.plot(meta_signals, label='Meta Signal', color='purple', linewidth=1)
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Long threshold')
    ax3.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Short threshold')
    ax3.fill_between(range(len(meta_signals)), meta_signals, 0.5,
                     where=[m > 0.5 for m in meta_signals], alpha=0.2, color='green')
    ax3.fill_between(range(len(meta_signals)), meta_signals, -0.5,
                     where=[m < -0.5 for m in meta_signals], alpha=0.2, color='red')
    ax3.set_ylabel("Meta Signal")
    ax3.set_xlabel("Bar")
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_result.png", dpi=150, bbox_inches='tight')
    plt.show()

    trades = len(long_entries) + len(short_entries)
    print(f"\nTrades: {trades}, Total Return: {total_pct:.2f}%")
    if trades > 0:
        print(f"Avg return per trade: {total_pct / trades:.2f}%")

    # Распечатаем статистику сигналов
    total_bars = len(bars)
    long_active_count = sum(1 for s in long_signals if s > 0.5)
    short_active_count = sum(1 for s in short_signals if s > 0.5)
    print(f"\nLong population active: {long_active_count}/{total_bars} bars ({long_active_count/total_bars*100:.1f}%)")
    print(f"Short population active: {short_active_count}/{total_bars} bars ({short_active_count/total_bars*100:.1f}%)")


# =====================================================================
# PERFORMANCE METRICS  (bot vs buy & hold)
# =====================================================================

def compute_metrics(equity_curve: list, trades_pnl: list,
                    bars_per_year: float = 24 * 365,
                    risk_free_rate: float = 0.0) -> dict:
    """
    Считает риск-метрики по equity-кривой и списку pnl сделок.
    bars_per_year=24*365 для часового таймфрейма крипты (24/7).
    """
    if len(equity_curve) < 2:
        return {}

    eq = np.asarray(equity_curve, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]

    if len(rets) == 0:
        return {}

    mu = rets.mean()
    sigma = rets.std(ddof=1) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    sigma_down = downside.std(ddof=1) if len(downside) > 1 else 0.0

    ann_factor = math.sqrt(bars_per_year)
    rf_per_bar = risk_free_rate / bars_per_year
    sharpe = (mu - rf_per_bar) / sigma * ann_factor if sigma > 0 else 0.0
    sortino = (mu - rf_per_bar) / sigma_down * ann_factor if sigma_down > 0 else 0.0

    years = len(rets) / bars_per_year
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 and eq[0] > 0 else 0.0

    running_max = np.maximum.accumulate(eq)
    drawdowns = (running_max - eq) / running_max
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0

    calmar = cagr / max_dd if max_dd > 0 else float('inf')

    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p < 0]
    win_rate = len(wins) / len(trades_pnl) if trades_pnl else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')
    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    return {
        'Total Return (%)': (eq[-1] / eq[0] - 1) * 100,
        'CAGR (%)': cagr * 100,
        'Max Drawdown (%)': max_dd * 100,
        'Sharpe (ann.)': sharpe,
        'Sortino (ann.)': sortino,
        'Calmar': calmar,
        'Volatility (ann. %)': sigma * ann_factor * 100,
        'Win Rate (%)': win_rate * 100,
        'Profit Factor': profit_factor,
        'Expectancy (per trade)': expectancy,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Total Trades': len(trades_pnl),
    }


def buy_and_hold_benchmark(bars: list, initial_balance: float = 10000,
                           commission: float = 0.001) -> dict:
    """
    Эталонная стратегия: купить на первом баре, держать до конца.
    Возвращает метрики, сопоставимые с compute_metrics().
    """
    if not bars:
        return {}

    entry_price = bars[0]["cur"]
    qty = initial_balance * (1 - commission) / entry_price  # учли комиссию входа

    equity = []
    for b in bars:
        equity.append(qty * b["cur"])

    # учли комиссию выхода в конце
    equity[-1] *= (1 - commission)

    final_pnl = equity[-1] - initial_balance
    return compute_metrics(equity, [final_pnl])


def compare_strategies(bars: list, hofs, initial_balance: float = 10000,
                       risk_percent: float = 0.02, label: str = ""):
    """
    Прогоняет GP-бота и buy & hold на одних и тех же барах,
    печатает сравнительную таблицу метрик.

    risk_percent: 0.02 = 2% сайзинг (риск-менеджмент), 1.0 = полная экспозиция.
    label: подпись для заголовка таблицы.
    """
    if not bars:
        print("compare_strategies: empty bars, skipping.")
        return {}, {}

    hof_long, hof_short, hof_meta = hofs
    best_long_func = gp.compile(hof_long[0], pset_long)
    best_short_func = gp.compile(hof_short[0], pset_short)
    meta_func = gp.compile(hof_meta[0], pset_meta)

    # --- Бот ---
    sim = TradingSimulator(initial_balance=initial_balance, commission=0.001,
                           risk_percent=risk_percent)
    trades_pnl = []

    for b in bars:
        try:
            long_raw = best_long_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            short_raw = best_short_func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            la = 1.0 if long_raw > 0 else 0.0
            sa = 1.0 if short_raw > 0 else 0.0
            meta = meta_func(la, sa)
            if meta > 0.5:
                action = 'LONG'
            elif meta < -0.5:
                action = 'SHORT'
            else:
                action = 'CLOSE' if sim.position_type != 'FLAT' else 'HOLD'
        except Exception:
            action = 'HOLD'

        trade = sim.execute_trade(timestamp=None, action=action, price=b["cur"])
        if trade and trade.pnl != 0.0:
            trades_pnl.append(trade.pnl)

    sim.close_position(timestamp=None, price=bars[-1]["cur"])
    bot_metrics = compute_metrics(sim.equity_curve, trades_pnl)

    # --- Buy & Hold ---
    bh_metrics = buy_and_hold_benchmark(bars, initial_balance=initial_balance)

    # --- Печать ---
    title = f" [{label}]" if label else ""
    print("\n" + "=" * 70)
    print(f"STRATEGY COMPARISON{title}  (risk_percent = {risk_percent:.0%})")
    print("=" * 70)
    print(f"{'Metric':<25}{'Bot (GP)':>20}{'Buy & Hold':>20}")
    print("-" * 70)
    keys = sorted(set(bot_metrics) | set(bh_metrics))
    for k in keys:
        bv = bot_metrics.get(k, float('nan'))
        hv = bh_metrics.get(k, float('nan'))
        bv_s = f"{bv:>20.4f}" if isinstance(bv, (int, float)) else f"{bv:>20}"
        hv_s = f"{hv:>20.4f}" if isinstance(hv, (int, float)) else f"{hv:>20}"
        print(f"{k:<25}{bv_s}{hv_s}")
    print("=" * 70)

    # equity curve B&H для построения графиков
    bh_entry = bars[0]["cur"]
    bh_qty = initial_balance * (1 - 0.001) / bh_entry
    bh_equity = [bh_qty * b["cur"] for b in bars]
    bh_equity[-1] *= (1 - 0.001)

    return bot_metrics, bh_metrics, list(sim.equity_curve), bh_equity


# =====================================================================
# PLOT — equity curve бот vs Buy & Hold (для рис. 3.2 и 3.4 курсовой)
# =====================================================================

def plot_equity_comparison(bot_equity, bh_equity, title, filename,
                           initial_balance=10000):
    """
    Рисует две кривые капитала на одном графике: бот vs B&H.
    Дополнительно показывает зоны просадки.
    Используется в курсовой как рис. 3.2 (бычий рынок) и рис. 3.4 (боковой).
    """
    n = min(len(bot_equity), len(bh_equity))
    if n < 2:
        print(f"plot_equity_comparison: недостаточно данных для '{title}'")
        return

    bot_eq = np.asarray(bot_equity[:n], dtype=float)
    bh_eq = np.asarray(bh_equity[:n], dtype=float)
    x = np.arange(n)

    bot_run_max = np.maximum.accumulate(bot_eq)
    bot_dd = (bot_run_max - bot_eq) / bot_run_max * 100
    bh_run_max = np.maximum.accumulate(bh_eq)
    bh_dd = (bh_run_max - bh_eq) / bh_run_max * 100

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True,
    )

    ax_eq.plot(x, bot_eq, label='EagleTrade (бот)', color='#1f77b4', linewidth=1.8)
    ax_eq.plot(x, bh_eq, label='Buy & Hold', color='#d62728', linewidth=1.8, linestyle='--')
    ax_eq.axhline(y=initial_balance, color='gray', linestyle=':', linewidth=0.8,
                  label=f'Начальный капитал ({initial_balance:.0f})')
    ax_eq.set_title(title, fontsize=13)
    ax_eq.set_ylabel('Капитал, USDT')
    ax_eq.legend(loc='best')
    ax_eq.grid(True, alpha=0.3)

    ax_dd.fill_between(x, 0, -bot_dd, color='#1f77b4', alpha=0.4, label='Просадка бота')
    ax_dd.fill_between(x, 0, -bh_dd, color='#d62728', alpha=0.3, label='Просадка B&H')
    ax_dd.set_ylabel('Просадка, %')
    ax_dd.set_xlabel('Час (бар)')
    ax_dd.legend(loc='lower left', fontsize=9)
    ax_dd.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  → сохранён график: {filename}")
    plt.show()


# =====================================================================
# PLOT — сводная гистограмма по двум периодам (для рис. 3.5 курсовой)
# =====================================================================

def plot_summary_bars(results, filename="summary_bars.png"):
    """
    Сводная гистограмма доходности и максимальной просадки
    по двум отложенным периодам для бота и Buy & Hold.

    results — список словарей вида:
      {'period': 'Февраль 2024 (бычий)',
       'bot':  {'Total Return (%)': ..., 'Max Drawdown (%)': ...},
       'bh':   {'Total Return (%)': ..., 'Max Drawdown (%)': ...}}
    """
    if not results:
        print("plot_summary_bars: пустой список результатов")
        return

    periods = [r['period'] for r in results]
    bot_ret = [r['bot'].get('Total Return (%)', 0.0) for r in results]
    bh_ret = [r['bh'].get('Total Return (%)', 0.0) for r in results]
    bot_dd = [r['bot'].get('Max Drawdown (%)', 0.0) for r in results]
    bh_dd = [r['bh'].get('Max Drawdown (%)', 0.0) for r in results]

    x = np.arange(len(periods))
    width = 0.35

    fig, (ax_ret, ax_dd) = plt.subplots(1, 2, figsize=(13, 5))

    bars1 = ax_ret.bar(x - width / 2, bot_ret, width,
                       label='EagleTrade (бот)', color='#1f77b4')
    bars2 = ax_ret.bar(x + width / 2, bh_ret, width,
                       label='Buy & Hold', color='#d62728')
    ax_ret.axhline(y=0, color='black', linewidth=0.6)
    ax_ret.set_title('Доходность за период')
    ax_ret.set_ylabel('Доходность, %')
    ax_ret.set_xticks(x)
    ax_ret.set_xticklabels(periods)
    ax_ret.legend()
    ax_ret.grid(True, alpha=0.3, axis='y')
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        offset = 0.4 if h >= 0 else -1.2
        ax_ret.text(b.get_x() + b.get_width() / 2, h + offset,
                    f'{h:.2f}%', ha='center', fontsize=9)

    bars3 = ax_dd.bar(x - width / 2, bot_dd, width,
                      label='EagleTrade (бот)', color='#1f77b4')
    bars4 = ax_dd.bar(x + width / 2, bh_dd, width,
                      label='Buy & Hold', color='#d62728')
    ax_dd.set_title('Максимальная просадка')
    ax_dd.set_ylabel('Просадка, %')
    ax_dd.set_xticks(x)
    ax_dd.set_xticklabels(periods)
    ax_dd.legend()
    ax_dd.grid(True, alpha=0.3, axis='y')
    for b in list(bars3) + list(bars4):
        h = b.get_height()
        ax_dd.text(b.get_x() + b.get_width() / 2, h + 0.1,
                   f'{h:.2f}%', ha='center', fontsize=9)

    plt.suptitle('Сводное сравнение EagleTrade и Buy & Hold по двум периодам',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  → сохранён график: {filename}")
    plt.show()


# =====================================================================
# MAIN — коэволюция трёх популяций
# =====================================================================

def evolve_one_gen(pop, toolbox, hof, mu=100, lambda_=150, cxpb=0.6, mutpb=0.3):
    """
    Один шаг (μ+λ) эволюции:
    - Генерим lambda_ потомков через кроссовер и мутацию
    - Объединяем родителей + потомков
    - Отбираем mu лучших
    Лучшие ВСЕГДА выживают — нет деградации.
    """
    offspring = algorithms.varOr(pop, toolbox, lambda_=lambda_, cxpb=cxpb, mutpb=mutpb)

    invalid = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    combined = pop + offspring
    pop[:] = tools.selBest(combined, mu)

    hof.update(pop)
    return pop


def main(seed=None):
    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % (2**31)
    random.seed(seed)
    print(f"Random seed: {seed}  (передай в main(seed={seed}) чтобы воспроизвести)")

    POP_SIZE = 150
    OFFSPRING = 200
    NGEN = 30

    pop_long = toolbox_long.population(n=POP_SIZE)
    pop_short = toolbox_short.population(n=POP_SIZE)
    pop_meta = toolbox_meta.population(n=POP_SIZE)

    hof_long = tools.HallOfFame(5)
    hof_short = tools.HallOfFame(5)
    hof_meta = tools.HallOfFame(5)

    toolbox_long.register("evaluate", evalLongTrading, bars=bars)
    toolbox_short.register("evaluate", evalShortTrading, bars=bars)

    for pop, tb in [(pop_long, toolbox_long), (pop_short, toolbox_short)]:
        fitnesses = list(map(tb.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

    hof_long.update(pop_long)
    hof_short.update(pop_short)

    print("=" * 60)
    print("Starting co-evolution with three populations (μ+λ)")
    print(f"Bars: {len(bars)}, Pop: {POP_SIZE}, Offspring: {OFFSPRING}, Gen: {NGEN}")
    print("Signal logic: true = position open, false = position closed")
    print("=" * 60)

    for gen in range(NGEN):
        print(f"\n--- Generation {gen + 1}/{NGEN} ---")

        pop_long = evolve_one_gen(pop_long, toolbox_long, hof_long,
                                  mu=POP_SIZE, lambda_=OFFSPRING)

        pop_short = evolve_one_gen(pop_short, toolbox_short, hof_short,
                                   mu=POP_SIZE, lambda_=OFFSPRING)

        best_long_func = gp.compile(hof_long[0], pset_long)
        best_short_func = gp.compile(hof_short[0], pset_short)

        toolbox_meta.register("evaluate", evalMetaTrading,
                              bars=bars,
                              best_long_func=best_long_func,
                              best_short_func=best_short_func)

        if gen == 0:
            fitnesses = list(map(toolbox_meta.evaluate, pop_meta))
            for ind, fit in zip(pop_meta, fitnesses):
                ind.fitness.values = fit
            hof_meta.update(pop_meta)

        pop_meta = evolve_one_gen(pop_meta, toolbox_meta, hof_meta,
                                  mu=POP_SIZE, lambda_=OFFSPRING)

        long_fit = [ind.fitness.values[0] for ind in pop_long]
        short_fit = [ind.fitness.values[0] for ind in pop_short]
        meta_fit = [ind.fitness.values[0] for ind in pop_meta]

        print(f"  Long  — max: {max(long_fit):8.2f}%, avg: {np.mean(long_fit):8.2f}%")
        print(f"  Short — max: {max(short_fit):8.2f}%, avg: {np.mean(short_fit):8.2f}%")
        print(f"  Meta  — max: {max(meta_fit):8.2f}%, avg: {np.mean(meta_fit):8.2f}%")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Long fitness:  {hof_long[0].fitness.values[0]:.2f}%")
    print(f"Best Long:  {str(hof_long[0])}")
    print(f"\nBest Short fitness: {hof_short[0].fitness.values[0]:.2f}%")
    print(f"Best Short: {str(hof_short[0])}")
    print(f"\nBest Meta fitness:  {hof_meta[0].fitness.values[0]:.2f}%")
    print(f"Best Meta:  {str(hof_meta[0])}")

    plot_signals(ndf, bars, hof_long[0], hof_short[0], hof_meta[0],
                 pset_long, pset_short, pset_meta,
                 title_prefix="[TRAIN] ")

    return (pop_long, pop_short, pop_meta), (hof_long, hof_short, hof_meta)


def validate_on_new_data(hofs, val_symbol="BTCUSDT", val_interval="1h",
                         val_start="02-01-2024", val_end="03-01-2024"):
    """
    Прогоняет лучших индивидов на новых данных (out-of-sample).
    Загружает новый кусок, строит индикаторы, выводит график и статистику.
    Возвращает кортеж (val_df, val_bars). При ошибке возвращает (None, []).
    """
    hof_long, hof_short, hof_meta = hofs

    print("\n" + "=" * 60)
    print(f"OUT-OF-SAMPLE VALIDATION")
    print(f"Symbol: {val_symbol}, Interval: {val_interval}")
    print(f"Period: {val_start} — {val_end}")
    print("=" * 60)

    val_df = get_history_data(val_symbol, val_interval, val_start, val_end)
    val_df = add_indicators(val_df, window=50)
    val_bars = build_input_vectors(val_df, min_window=50)

    print(f"Validation bars: {len(val_bars)}")

    if len(val_bars) == 0:
        print("ERROR: Нет данных для валидации!")
        return None, []

    best_long = hof_long[0]
    best_short = hof_short[0]
    best_meta = hof_meta[0]

    long_fit = evalLongTrading(best_long, val_bars)
    short_fit = evalShortTrading(best_short, val_bars)

    long_func = gp.compile(best_long, pset_long)
    short_func = gp.compile(best_short, pset_short)
    meta_fit = evalMetaTrading(best_meta, val_bars, long_func, short_func)

    print(f"\nValidation fitness:")
    print(f"  Long:  {long_fit[0]:.2f}%")
    print(f"  Short: {short_fit[0]:.2f}%")
    print(f"  Meta:  {meta_fit[0]:.2f}%")

    plot_signals(val_df, val_bars, best_long, best_short, best_meta,
                 pset_long, pset_short, pset_meta,
                 title_prefix="[VALIDATION] ")

    return val_df, val_bars


# =====================================================================
# DATA LOADING & ENTRY POINT
# =====================================================================

def load_period(symbol, interval, start, end, window=50):
    """Загружает один период, добавляет индикаторы, возвращает (df, bars)."""
    df = get_history_data(symbol, interval, start, end)
    df = add_indicators(df, window=window)
    bars = build_input_vectors(df, min_window=window)
    return df, bars

# ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ НА СМЕСИ РЫНОЧНЫХ РЕЖИМОВ
# ---------------------------------------------------------------------
# Чтобы бот не выучился только «ловить тренд», обучаем его сразу
# на двух разных режимах рынка:
#   • январь 2024 — восходящий тренд (~+10%)
#   • сентябрь 2023 — боковой рынок (~+4%)
print("Loading TRAIN data: январь 2024 (тренд) + сентябрь 2023 (флэт)...")
ndf_trend, bars_trend = load_period("BTCUSDT", "1h", "01-01-2024", "02-01-2024")
ndf_flat, bars_flat = load_period("BTCUSDT", "1h", "09-01-2023", "10-01-2023")

# Объединяем бары: фитнес будет суммой P&L по обоим периодам.
# При этом в evalLongTrading/evalShortTrading при переходе от одного периода
# к другому будет «склейка» — это не идеально, но допустимо: всего одна сделка
# на стыке из ~1300, влияние на фитнес пренебрежимо.
bars = bars_trend + bars_flat
ndf = ndf_trend  # для plot_signals на тренировочных (берём первый кусок)
print(f"  Trend bars: {len(bars_trend)}, Flat bars: {len(bars_flat)}, Total: {len(bars)}")
# =====================================================================
# MULTI-THREADED EVOLUTION  — 15 потоков, каждый со своим seed
# =====================================================================

import threading
from typing import Dict

NUM_THREADS = 15
_results_lock = threading.Lock()
_all_results = []   # [{seed, fitness, pops, hofs}, ...]


def _run_evolution_thread(seed: int):
    """Целевая функция потока: запускает main(seed) и сохраняет результат."""
    try:
        pops, hofs = main(seed=seed)
        fitness = hofs[2][0].fitness.values[0]   # fitness лучшего Meta-индивида
        with _results_lock:
            _all_results.append({
                "seed":    seed,
                "fitness": fitness,
                "pops":    pops,
                "hofs":    hofs,
            })
        print(f"[seed={seed}] Thread finished. Meta fitness = {fitness:.2f}%")
    except Exception as exc:
        print(f"[seed={seed}] Thread ERROR: {exc}")


def run_parallel_evolution(n_threads: int = NUM_THREADS):
    """
    Запускает n_threads потоков эволюции с уникальными случайными seed-ами.
    Возвращает словарь с лучшим результатом по Meta fitness.
    """
    global _all_results
    _all_results = []

    # Генерируем уникальные seed-ы заранее
    seeds = random.sample(range(1, 2**30), n_threads)

    print("=" * 70)
    print(f"ЗАПУСК {n_threads} ПАРАЛЛЕЛЬНЫХ ЭВОЛЮЦИЙ")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    threads = []
    for seed in seeds:
        t = threading.Thread(
            target=_run_evolution_thread,
            args=(seed,),
            name=f"evo-seed-{seed}",
            daemon=False
        )
        threads.append(t)

    # Стартуем все потоки одновременно
    for t in threads:
        t.start()

    # Ждём завершения всех
    for t in threads:
        t.join()

    if not _all_results:
        raise RuntimeError("Все потоки упали, результатов нет!")

    # Выбираем лучший по Meta fitness
    best = max(_all_results, key=lambda r: r["fitness"])

    print("\n" + "=" * 70)
    print("ИТОГИ ПАРАЛЛЕЛЬНОЙ ЭВОЛЮЦИИ")
    print("=" * 70)
    for r in sorted(_all_results, key=lambda x: x["fitness"], reverse=True):
        marker = " <-- WINNER" if r["seed"] == best["seed"] else ""
        print(f"  seed={r['seed']:12d}  Meta fitness={r['fitness']:8.2f}%{marker}")
    print("=" * 70)
    print(f"Лучший seed: {best['seed']}, Meta fitness: {best['fitness']:.2f}%")
    return best


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    # Загрузка через BinanceBroker (для проверки)
    broker = BinanceBroker()
    data = broker.get_history_data(['BTCUSDT'], '1h', '2024-01-01', '2024-02-01')
    print(f"Loaded {len(data)} candles")
    print(data.head())

    simulator = TradingSimulator(initial_balance=10000, commission=0.001)
    print(f"\nGP data prepared: {len(bars)} bars")

    # ----------------------------------------------------------------
    # Запуск 15 параллельных эволюций
    # ----------------------------------------------------------------
    print("\n=== STARTING PARALLEL GP EVOLUTION (15 threads) ===\n")
    best_result = run_parallel_evolution(n_threads=15)

    # Берём лучшие HOF-ы из победившего потока
    hofs = best_result["hofs"]
    hof_long, hof_short, hof_meta = hofs

    print(f"\nBest Long fitness:  {hof_long[0].fitness.values[0]:.2f}%")
    print(f"Best Long:  {str(hof_long[0])}")
    print(f"\nBest Short fitness: {hof_short[0].fitness.values[0]:.2f}%")
    print(f"Best Short: {str(hof_short[0])}")
    print(f"\nBest Meta fitness:  {hof_meta[0].fitness.values[0]:.2f}%")
    print(f"Best Meta:  {str(hof_meta[0])}")

    # Визуализация лучшего результата
    plot_signals(ndf, bars, hof_long[0], hof_short[0], hof_meta[0],
                 pset_long, pset_short, pset_meta,
                 title_prefix=f"[BEST seed={best_result['seed']}] ")

    # ======================================================================
    # OUT-OF-SAMPLE ВАЛИДАЦИЯ
    # ======================================================================

    print("\n\n" + "#" * 70)
    print("# OOS ПЕРИОД 1: БЫЧИЙ РЫНОК (февраль 2024)")
    print("#" * 70)
    val_df_bull, val_bars_bull = validate_on_new_data(
        hofs, val_symbol="BTCUSDT", val_interval="1h",
        val_start="02-01-2024", val_end="03-01-2024",
    )

    bot_metrics_bull, bh_metrics_bull = {}, {}
    if val_bars_bull:
        bot_metrics_bull, bh_metrics_bull, bot_eq_bull, bh_eq_bull = compare_strategies(
            val_bars_bull, hofs, risk_percent=0.02, label="BULL | Risk-managed 2%")
        compare_strategies(val_bars_bull, hofs, risk_percent=1.0,
                           label="BULL | Full exposure 100%")
        plot_equity_comparison(
            bot_eq_bull, bh_eq_bull,
            title="Капитал портфеля на бычьем рынке (февраль 2024): бот vs Buy & Hold",
            filename="equity_bull_feb2024.png",
        )

    print("\n\n" + "#" * 70)
    print("# OOS ПЕРИОД 2: СЛАБЫЙ МЕДВЕЖИЙ / БОКОВОЙ (май 2023)")
    print("#" * 70)
    val_df_flat, val_bars_flat = validate_on_new_data(
        hofs, val_symbol="BTCUSDT", val_interval="1h",
        val_start="05-01-2023", val_end="06-01-2023",
    )

    bot_metrics_bear, bh_metrics_bear = {}, {}
    if val_bars_flat:
        bot_metrics_bear, bh_metrics_bear, bot_eq_bear, bh_eq_bear = compare_strategies(
            val_bars_flat, hofs, risk_percent=0.02, label="WEAK BEAR | Risk-managed 2%")
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
        plot_summary_bars(summary, filename="summary_bars.png ")