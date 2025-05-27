import operator
import random
from functools import partial
import math
import numpy as np
from deap import base, creator, gp, tools
from deap import algorithms  # для эволюционных стратегий
import pandas as pd
from binance.client import Client  # для получения исторических данных
import matplotlib.pyplot as plt

# ----- 0. Визуализация торговых сигналов -----
def plot_signals(df, bars, best_individual, pset):
    # Компиляция GP-дерева в функцию
    func = gp.compile(best_individual, pset)
    buy_signals, sell_signals = [], []  # индексы сигналов на покупку/продажу
    in_position = False  # флаг, открыта ли позиция
    entries, exits = [], []  # цены входа и выхода

    for i, b in enumerate(bars):
        try:
            signal = func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
        except Exception:
            # если функция выдала ошибку — считаем сигнал ложным
            signal = False

        # открываем позицию на первом истинном сигнале
        if signal and not in_position:
            buy_signals.append(i)
            entries.append(b["cur"])
            in_position = True
        # закрываем позицию на следующем ложном сигнале
        elif not signal and in_position:
            sell_signals.append(i)
            exits.append(b["cur"])
            in_position = False

    # подгонка временных рядов: bars короче df
    offset = len(df) - len(bars)
    price_series = df["Price"].iloc[offset:]

    # рисуем график цены и метки сделок
    plt.figure(figsize=(16, 8))
    plt.plot(price_series.values, label='Price')
    plt.scatter(buy_signals, price_series.values[buy_signals], label='BUY', marker='^', s=100)
    plt.scatter(sell_signals, price_series.values[sell_signals], label='SELL', marker='v', s=100)
    plt.title("Trading Signals")
    plt.xlabel("Bars")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    # вычисляем итоговый PnL
    pnl = sum(exit - entry for entry, exit in zip(entries, exits))
    print(f"Сделок: {len(entries)} | Профит: {pnl:.2f}")


# ----- 1. Загрузка исторических данных и расчёт индикаторов -----
def get_history_data(symbol, interval, start_date, end_date):
    client = Client()  # API Binance
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    # создаём DataFrame и оставляем только полезные столбцы
    cols = ["Open Time","Open","High","Low","Close","Volume",
            "Close Time","Quote Asset Volume","#Trades",
            "TakerBuyBase","TakerBuyQuote","Ignore"]
    df = pd.DataFrame(klines, columns=cols, dtype=float)
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
    df = df[["Open Time", "Close", "Volume"]]
    df.rename(columns={"Close": "Price"}, inplace=True)
    return df


def add_indicators(df: pd.DataFrame, window: int = 50):
    # Простая скользящая средняя
    df[f"SMA_{window}"] = df["Price"].rolling(window).mean()
    # Экспоненциальная скользящая средняя
    df[f"EMA_{window}"] = df["Price"].ewm(span=window, adjust=False).mean()
    # Линейно-взвешенная SMA (LWMA)
    weights = np.arange(1, window + 1)
    df[f"LWMA_{window}"] = df["Price"].rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return df


def build_input_vectors(df: pd.DataFrame, min_window: int = 50):
    bars = []
    # строим векторы признаков для каждого бара, начиная с min_window
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


# пример подготовки данных
ndf = get_history_data("BTCUSDT", "1h", "01-10-2020", "06-01-2020")
ndf = add_indicators(ndf, window=50)
bars = build_input_vectors(ndf, min_window=50)

# ----- 2. Определение типов и примитивов GP -----
VECTOR = list
SCALAR = float
BOOL = bool
# --- 2. Примитивы ---
def vec_add(a: VECTOR, b: VECTOR) -> VECTOR:
    return [x + y for x, y in zip(a, b)]

def vec_sub(a: VECTOR, b: VECTOR) -> VECTOR:
    return [x - y for x, y in zip(a, b)]

def vec_add_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [x + b for x in a]

def vec_sub_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [x - b for x in a]

def vec_mul_s(a: VECTOR, b: SCALAR) -> VECTOR:
    return [x * b for x in a]

def scalar_mean(a: VECTOR) -> SCALAR:
    return sum(a) / len(a)

def last_elem(a: VECTOR) -> SCALAR:
    return a[-1]

def sum_gt(a: VECTOR, b: VECTOR) -> BOOL:
    return sum(a) > sum(b)

def mean_gt(a: VECTOR, b: SCALAR) -> BOOL:
    return (sum(a) / len(a)) > b

def rnd_mean_gt(a: VECTOR, low: SCALAR, high: SCALAR) -> BOOL:
    r = random.uniform(low, high)
    return (sum(a) / len(a)) > r

def if_else(cond: BOOL, a: VECTOR, b: VECTOR) -> VECTOR:
    return a if cond else b

def vec_diff(a: VECTOR) -> VECTOR:
    return [j - i for i, j in zip(a, a[1:])]

def scalar_diff(a: SCALAR) -> SCALAR:
    return 0.0

def vec_log(a: VECTOR) -> VECTOR:
    return [math.log(x) if x > 0 else 0.0 for x in a]

def scalar_log(a: SCALAR) -> SCALAR:
    return math.log(a) if a > 0 else 0.0

# --- 3. Настраиваем DEAP GP ---
pset = gp.PrimitiveSetTyped("MAIN",
    in_types=[VECTOR, VECTOR, VECTOR, VECTOR, SCALAR],
    ret_type=BOOL,
    prefix="IN"
)

# Уникальные эфемерные константы
pset.addEphemeralConstant("ephemeral_zero", partial(random.uniform, 0.0, 0.0), SCALAR)
pset.addEphemeralConstant("ephemeral_one",  partial(random.uniform, 1.0, 1.0), SCALAR)

# Булевы терминалы
pset.addTerminal(True,  BOOL)
pset.addTerminal(False, BOOL)

# Примитивы
pset.addPrimitive(vec_add,     [VECTOR, VECTOR],       VECTOR)
pset.addPrimitive(vec_sub,     [VECTOR, VECTOR],       VECTOR)
pset.addPrimitive(vec_add_s,   [VECTOR, SCALAR],       VECTOR)
pset.addPrimitive(vec_sub_s,   [VECTOR, SCALAR],       VECTOR)
pset.addPrimitive(vec_mul_s,   [VECTOR, SCALAR],       VECTOR)
pset.addPrimitive(scalar_mean, [VECTOR],               SCALAR)
pset.addPrimitive(last_elem,   [VECTOR],               SCALAR)
pset.addPrimitive(sum_gt,      [VECTOR, VECTOR],       BOOL)
pset.addPrimitive(mean_gt,     [VECTOR, SCALAR],       BOOL)
pset.addPrimitive(rnd_mean_gt, [VECTOR, SCALAR, SCALAR], BOOL)
pset.addPrimitive(if_else,     [BOOL, VECTOR, VECTOR], VECTOR)
pset.addPrimitive(vec_diff,    [VECTOR],               VECTOR)
pset.addPrimitive(vec_log,     [VECTOR],               VECTOR)
pset.addPrimitive(scalar_log,  [SCALAR],               SCALAR)

# --- 4. Toolbox и GP-настройки ---
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # максимизируем фитнес
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Фитнес-функция
def evalTrading(individual, bars):
    func = gp.compile(individual, pset)
    profit = 0.0
    for b in bars:
        try:
            sig = func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
        except Exception:
            sig = False
        if sig:
            profit += b["next"] - b["cur"]
    return profit,


  # добавочный профит за каждую сделку

def evalTrading(individual, bars):
    func = gp.compile(individual, pset)
    profit = 0.0
    trades = 0
    for b in bars:
        try:
            sig = func(b["price"], b["sma"], b["ema"], b["lwma"], b["cur"])
        except Exception:
            sig = False
        if sig:
            profit += b["next"] - b["cur"]
            trades += 1
    # вычисляем итоговый фитнес: PnL + бонусы
    fitness = profit
    # штраф за отсутствие сделок
    if trades == 0:
        fitness -= 1.0
    return (fitness,)

toolbox.register("evaluate", evalTrading, bars=bars)

# операторы отбора, мутации и ограничение глубины
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=5))



# ----- 5. Size-Fair Type-Safe Subtree Crossover -----
def cx_type_safe_size_fair(ind1, ind2, max_delta=1):
    # выбираем случайный узел из первого родителя
    idx1 = random.randrange(len(ind1))
    slice1 = ind1.searchSubtree(idx1)
    size1 = slice1.stop - slice1.start
    type1 = ind1[idx1].ret
    # ищем подходящие узлы во втором родителе с тем же типом и схожим размером
    candidates = []
    for idx2 in range(len(ind2)):
        if ind2[idx2].ret == type1:
            slice2 = ind2.searchSubtree(idx2)
            size2 = slice2.stop - slice2.start
            if abs(size1 - size2) <= max_delta:
                candidates.append((idx2, slice2))
    if not candidates:
        return ind1, ind2  # нет подходящих — возвращаем без изменений
    # случайно выбираем один кандидат и меняем поддеревья
    _, slice2 = random.choice(candidates)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    return ind1, ind2

# регистрируем новый crossover вместо gp.cxOnePoint
toolbox.register("mate", cx_type_safe_size_fair, max_delta=1)
toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=5))

# ----- 6. "Брачный рынок" по размеру деревьев -----
def pair_by_size(population):
    # сортируем особей по длине их дерева
    sorted_pop = sorted(population, key=len)
    # объединяем соседей в пары
    for i in range(0, len(sorted_pop) - 1, 2):
        yield sorted_pop[i], sorted_pop[i + 1]

# ----- 7. Эволюционный цикл с элитизмом (μ+λ) -----
def main():
    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)  # храним лучшую особь
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # μ+λ: родители + потомки соревнуются за место в след. поколении
    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=len(pop), lambda_=len(pop),
        cxpb=0.5, mutpb=0.2,
        ngen=50,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    plot_signals(ndf, bars, hof[0], pset)
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print("Лучший индивид:", hof[0], "Фитнес:", hof[0].fitness.values[0])
