# =====================================================================
# SIGNALS — симулятор, фитнес-функции, метрики и визуализация
# Преобразование сигналов трёх популяций (long / short / meta) в сделки.
# =====================================================================

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from deap import gp

from primitives import pset_long, pset_short, pset_meta
from data_loader import get_history_data, add_indicators, build_input_vectors
from config import CFG


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
    def __init__(self, initial_balance: float = CFG.initial_balance,
                 commission: float = CFG.commission,
                 risk_percent: float = CFG.risk_percent):
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
# ТОРГОВЫЕ КОНСТАНТЫ И ОБЩАЯ РЕШАЮЩАЯ ЛОГИКА
# Единый источник правды для комиссий, порогов и решения мета-сигнала.
# Раньше эти значения и правило meta>0.5/<-0.5 были скопированы в 4 местах
# (evalMetaTrading, plot_signals, compare_strategies, live_trading) и могли
# разъехаться. Теперь — одно место.
# =====================================================================

COMMISSION = CFG.commission          # 0.1% за сделку (вход + выход = 0.2%)
SHORT_COST_PER_BAR = CFG.short_cost_per_bar  # за бар удержания шорта (funding rate)
LONG_THRESHOLD = CFG.long_threshold  # meta_sig > LONG_THRESHOLD  -> LONG
SHORT_THRESHOLD = CFG.short_threshold  # meta_sig < SHORT_THRESHOLD -> SHORT


def decide_position(bar, long_func, short_func, meta_func):
    """
    Единая решающая логика трёх популяций.
    Возвращает желаемую позицию: 1 = LONG, -1 = SHORT, 0 = FLAT.
    При любой ошибке вычисления дерева -> 0 (FLAT).
    """
    try:
        long_raw = long_func(bar["price"], bar["sma"], bar["ema"], bar["lwma"], bar["cur"])
        short_raw = short_func(bar["price"], bar["sma"], bar["ema"], bar["lwma"], bar["cur"])
        long_active = 1.0 if long_raw > 0 else 0.0
        short_active = 1.0 if short_raw > 0 else 0.0
        meta_sig = meta_func(long_active, short_active)
    except Exception:
        return 0, 0.0, 0.0, 0.0

    if meta_sig > LONG_THRESHOLD:
        desired = 1
    elif meta_sig < SHORT_THRESHOLD:
        desired = -1
    else:
        desired = 0
    return desired, long_active, short_active, meta_sig


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
    commission = COMMISSION
    was_open = False
    entry_price = 0.0
    trades_count = 0

    for b in bars:
        try:
            sig = func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            is_open = sig > 0
        except Exception:
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
    commission = COMMISSION
    short_cost_per_bar = SHORT_COST_PER_BAR
    was_open = False
    entry_price = 0.0
    bars_held = 0
    trades_count = 0

    for b in bars:
        try:
            sig = func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
            is_open = sig > 0
        except Exception:
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
    commission = COMMISSION
    short_cost_per_bar = SHORT_COST_PER_BAR
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    bars_held = 0
    trades_count = 0

    for b in bars:
        desired, _, _, _ = decide_position(b, best_long_func, best_short_func, meta_func)

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
# PLOT — визуализация сигналов трёх популяций
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
    commission = COMMISSION
    short_cost_per_bar = SHORT_COST_PER_BAR
    bars_held = 0

    for i, b in enumerate(bars):
        prices.append(b["cur"])

        desired, long_active, short_active, meta_raw = decide_position(
            b, long_func, short_func, meta_func)

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
    # Окно НЕ показываем здесь — все фигуры накапливаются и выводятся
    # одним plt.show() в конце main.py (иначе при множестве графиков
    # показывается только первый).

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


def buy_and_hold_benchmark(bars: list, initial_balance: float = CFG.initial_balance,
                           commission: float = CFG.commission) -> dict:
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


def compare_strategies(bars: list, hofs, initial_balance: float = CFG.initial_balance,
                       risk_percent: float = CFG.risk_percent, label: str = ""):
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
        desired, _, _, _ = decide_position(b, best_long_func, best_short_func, meta_func)
        if desired == 1:
            action = 'LONG'
        elif desired == -1:
            action = 'SHORT'
        else:
            action = 'CLOSE' if sim.position_type != 'FLAT' else 'HOLD'

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


def quick_profit_summary(bars, hofs, initial_balance=None, risk_percent=None,
                         label="TRAIN"):
    """
    Быстрая сводка БЕЗ большой таблицы и графиков:
    прибыль стратегии, прибыль buy & hold и разница между ними.
    Возвращает (bot_return_pct, bh_return_pct). Печатает компактный блок.
    """
    initial_balance = initial_balance if initial_balance is not None else CFG.initial_balance
    risk_percent = risk_percent if risk_percent is not None else CFG.risk_percent

    if not bars:
        print("quick_profit_summary: нет данных.")
        return None, None

    hof_long, hof_short, hof_meta = hofs
    long_func = gp.compile(hof_long[0], pset_long)
    short_func = gp.compile(hof_short[0], pset_short)
    meta_func = gp.compile(hof_meta[0], pset_meta)

    # --- Бот ---
    sim = TradingSimulator(initial_balance=initial_balance,
                           commission=CFG.commission, risk_percent=risk_percent)
    for b in bars:
        desired, _, _, _ = decide_position(b, long_func, short_func, meta_func)
        if desired == 1:
            action = 'LONG'
        elif desired == -1:
            action = 'SHORT'
        else:
            action = 'CLOSE' if sim.position_type != 'FLAT' else 'HOLD'
        sim.execute_trade(timestamp=None, action=action, price=b["cur"])
    sim.close_position(timestamp=None, price=bars[-1]["cur"])

    bot_return = (sim.balance / initial_balance - 1) * 100

    # --- Buy & Hold ---
    bh_metrics = buy_and_hold_benchmark(bars, initial_balance=initial_balance)
    bh_return = bh_metrics.get('Total Return (%)', 0.0)

    diff = bot_return - bh_return
    sign = "+" if diff >= 0 else ""

    print("\n" + "=" * 60)
    print(f"ПРИБЫЛЬ НА УЧАСТКЕ [{label}]  ({len(bars)} баров, "
          f"риск {risk_percent:.0%})")
    print("=" * 60)
    print(f"  Стратегия (бот):   {bot_return:+8.2f}%")
    print(f"  Buy & Hold:        {bh_return:+8.2f}%")
    print(f"  Разница:           {sign}{diff:7.2f} п.п.  "
          f"({'лучше' if diff >= 0 else 'хуже'} buy & hold)")
    print("=" * 60)

    return bot_return, bh_return


# =====================================================================
# PLOT — equity curve бот vs Buy & Hold
# =====================================================================

def plot_equity_comparison(bot_equity, bh_equity, title, filename,
                           initial_balance=CFG.initial_balance):
    """
    Рисует две кривые капитала на одном графике: бот vs B&H.
    Дополнительно показывает зоны просадки.
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
    # show() — один раз в конце main.py


# =====================================================================
# PLOT — сводная гистограмма по двум периодам
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
    # show() — один раз в конце main.py


# =====================================================================
# OUT-OF-SAMPLE VALIDATION
# =====================================================================

def validate_on_new_data(hofs, val_symbol=None, val_interval=None,
                         val_start="02-01-2024", val_end="03-01-2024"):
    """
    Прогоняет лучших индивидов на новых данных (out-of-sample).
    Загружает новый кусок, строит индикаторы, выводит график и статистику.
    Возвращает кортеж (val_df, val_bars). При ошибке возвращает (None, []).
    """
    val_symbol = val_symbol or CFG.symbol
    val_interval = val_interval or CFG.interval
    hof_long, hof_short, hof_meta = hofs

    print("\n" + "=" * 60)
    print(f"OUT-OF-SAMPLE VALIDATION")
    print(f"Symbol: {val_symbol}, Interval: {val_interval}")
    print(f"Period: {val_start} — {val_end}")
    print("=" * 60)

    val_df = get_history_data(val_symbol, val_interval, val_start, val_end)
    val_df = add_indicators(val_df, window=CFG.window)
    val_bars = build_input_vectors(val_df, min_window=CFG.window)

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
