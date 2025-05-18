import operator
import random
from functools import partial
import math
import numpy as np
from deap import base, creator, gp, tools
from deap import algorithms    # ← Добавил импорт алгоритмов
import pandas as pd
import numpy as np
from binance.client import Client

def get_history_data(symbol: str, interval: str, start_date: str, end_date: str):
    client = Client()  # подставь свои ключи, если нужно
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    cols = ["Open Time","Open","High","Low","Close","Volume",
            "Close Time","Quote Asset Volume","#Trades",
            "TakerBuyBase","TakerBuyQuote","Ignore"]
    df = pd.DataFrame(klines, columns=cols, dtype=float)
    df["Open Time"]  = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
    # Оставим только нужное:
    df = df[["Open Time","Close","Volume"]]
    df.rename(columns={"Close":"Price"}, inplace=True)
    return df

def add_indicators(df: pd.DataFrame, window: int=50):
    # SMA
    df[f"SMA_{window}"] = df["Price"].rolling(window).mean()
    # EWMA (EMA)
    df[f"EMA_{window}"] = df["Price"].ewm(span=window, adjust=False).mean()
    # LWMA — взвешенное с линейным весом
    weights = np.arange(1, window+1)
    df[f"LWMA_{window}"] = df["Price"].rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    return df

# Пример:
df = get_history_data("BTCUSDT", "1h", "01-01-2020", "02-01-2020")
df = add_indicators(df, window=50)



def build_input_vectors(df: pd.DataFrame, min_window: int=50):
    bars = []
    for i in range(min_window, len(df)-1):
        # Векторы из окон [0…i-1], [1…i], … [i-min_window…i-1] или просто последние min_window точек:
        start = i - min_window
        vec_price = df["Price"].iloc[start:i].tolist()
        vec_sma   = df[f"SMA_{min_window}"].iloc[start:i].tolist()
        vec_ema   = df[f"EMA_{min_window}"].iloc[start:i].tolist()
        vec_lwma  = df[f"LWMA_{min_window}"].iloc[start:i].tolist()
        cur_price = df["Price"].iloc[i]
        bars.append({
            "price": vec_price,
            "sma":   vec_sma,
            "ema":   vec_ema,
            "lwma":  vec_lwma,
            "cur":   cur_price,
            "next":  df["Price"].iloc[i+1]
        })
    return bars

bars = build_input_vectors(df, min_window=50)



# --- 1. Определяем типы ---
VECTOR = list
SCALAR = float
BOOL   = bool

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
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox.register("expr",       gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat,   list, toolbox.individual)

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


# вместо random-bars
toolbox.register("evaluate", evalTrading, bars=bars)

toolbox.register("select",   tools.selTournament, tournsize=3)
toolbox.register("mate",     gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate",   gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Ограниечение глубины
toolbox.decorate("mate",   gp.staticLimit(key=lambda ind: ind.height, max_value=5))
toolbox.decorate("mutate",gp.staticLimit(key=lambda ind: ind.height, max_value=5))

# --- 5. Эво-цикл ---
def main():
    pop = toolbox.population(n=2000)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # ==== Исправлено: вызываем алгоритм из deap.algorithms, а не из tools ====
    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=0.5, mutpb=0.2,
                                   ngen=20,
                                   stats=stats, halloffame=hof,
                                   verbose=True)
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print("Лучший индивид:", hof[0], "Фитнес:", hof[0].fitness.values[0])
