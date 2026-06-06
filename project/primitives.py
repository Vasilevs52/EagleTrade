# =====================================================================
# GP PRIMITIVES — типы, функции-примитивы и наборы примитивов (pset)
# =====================================================================

import math
import random

from deap import gp

VECTOR = list
SCALAR = float
BOOL = float


def rand_period():
    """Эфемерная константа-период для индикаторов: float в [5, 50].
    Возвращает именно float (а не int), иначе gp.from_string не восстановит
    дерево из строки (тип терминала должен совпадать со SCALAR=float)."""
    return float(random.randint(5, 50))


def rand_unit():
    """Эфемерная константа: float в [-1, 1]."""
    return random.uniform(-1.0, 1.0)


def rand_meta():
    """Эфемерная константа мета-популяции: float в [-2, 2]."""
    return random.uniform(-2.0, 2.0)


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
# ДОПОЛНИТЕЛЬНЫЕ СКАЛЯРНЫЕ ПРИМИТИВЫ
# =====================================================================

def scalar_neg(a):
    return -a

def scalar_min(a, b):
    return min(a, b)

def scalar_max(a, b):
    return max(a, b)

def scalar_sign(a):
    return 1.0 if a > 0 else (-1.0 if a < 0 else 0.0)

def scalar_relu(a):
    return a if a > 0 else 0.0

def scalar_gte(a, b):
    return 1.0 if a >= b else 0.0

def scalar_lte(a, b):
    return 1.0 if a <= b else 0.0

def scalar_avg(a, b):
    return (a + b) / 2.0

def scalar_pow2(a):
    return min(a * a, 1e12)

def protected_div(a, b):
    return a / b if abs(b) > 1e-10 else 1.0


# =====================================================================
# ДОПОЛНИТЕЛЬНЫЕ ВЕКТОРНЫЕ ПРИМИТИВЫ (технические индикаторы)
# Все возвращают SCALAR — пригодны для построения торговых сигналов.
# =====================================================================

def vec_momentum(a):
    """Моментум: разница последнего и первого элемента окна."""
    return (a[-1] - a[0]) if len(a) > 0 else 0.0

def vec_roc(a):
    """Rate of Change (%): относительное изменение за окно."""
    if len(a) == 0 or abs(a[0]) < 1e-10:
        return 0.0
    return (a[-1] - a[0]) / a[0] * 100.0

def vec_range(a):
    """Размах окна: max - min."""
    return (max(a) - min(a)) if len(a) > 0 else 0.0

def vec_median(a):
    """Медиана значений окна."""
    if len(a) == 0:
        return 0.0
    s = sorted(a)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0

def vec_slope(a):
    """Наклон линейного тренда (МНК) по окну."""
    n = len(a)
    if n < 2:
        return 0.0
    xs = range(n)
    mx = (n - 1) / 2.0
    my = sum(a) / n
    num = sum((x - mx) * (a[x] - my) for x in xs)
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-10 else 0.0

def vec_zscore(a):
    """Z-оценка последнего элемента относительно окна."""
    n = len(a)
    if n < 2:
        return 0.0
    mean = sum(a) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in a) / n)
    return (a[-1] - mean) / std if std > 1e-10 else 0.0

def vec_ema_last(a):
    """Последнее значение EMA по окну (span = длина окна)."""
    n = len(a)
    if n == 0:
        return 0.0
    alpha = 2.0 / (n + 1)
    ema = a[0]
    for x in a[1:]:
        ema = alpha * x + (1 - alpha) * ema
    return ema

def vec_rsi(a):
    """RSI по окну (0..100); 50 при недостатке данных."""
    n = len(a)
    if n < 2:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(1, n):
        d = a[i] - a[i - 1]
        if d > 0:
            gains += d
        else:
            losses -= d
    if losses < 1e-10:
        return 100.0
    rs = gains / losses
    return 100.0 - 100.0 / (1.0 + rs)

def vec_mean_diff(a, b):
    """Разница средних двух окон (например SMA - EMA)."""
    ma = sum(a) / len(a) if len(a) > 0 else 0.0
    mb = sum(b) / len(b) if len(b) > 0 else 0.0
    return ma - mb

def cross_above(a, b):
    """1.0, если ряд a пересёк ряд b снизу вверх на последнем баре."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    return 1.0 if (a[-2] <= b[-2] and a[-1] > b[-1]) else 0.0

def cross_below(a, b):
    """1.0, если ряд a пересёк ряд b сверху вниз на последнем баре."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    return 1.0 if (a[-2] >= b[-2] and a[-1] < b[-1]) else 0.0


# =====================================================================
# ФИНАНСОВЫЕ ИНДИКАТОРЫ С НАСТРАИВАЕМЫМ ПЕРИОДОМ
# GP сам подбирает период (SCALAR), индикатор считается по последним
# N значениям входного окна. Есть две формы:
#   *_n      (VECTOR, period) -> SCALAR  — последнее значение индикатора
#   *_series (VECTOR, period) -> VECTOR  — полный ряд (для пересечений)
# =====================================================================

def _clamp_period(vec, period):
    """Приводит период к целому в диапазоне [2, len(vec)].
    Безопасно к NaN/inf: дерево может подать в период не-конечное значение
    (scalar_div на ~0, scalar_exp и т.п.). Без этой защиты int(NaN)/int(inf)
    кидают ValueError/OverflowError в горячем цикле эволюции."""
    n = len(vec)
    if n < 2:
        return n
    if not math.isfinite(period):   # NaN, +inf, -inf
        return n                     # деградируем к полному окну
    p = int(abs(period))
    if p < 2:
        p = 2
    if p > n:
        p = n
    return p

def sma_n(vec, period):
    """Simple Moving Average по последним N значениям."""
    if len(vec) == 0:
        return 0.0
    p = _clamp_period(vec, period)
    window = vec[-p:]
    return sum(window) / len(window)

def ema_n(vec, period):
    """Exponential Moving Average (span=N), последнее значение."""
    if len(vec) == 0:
        return 0.0
    p = _clamp_period(vec, period)
    window = vec[-p:]
    alpha = 2.0 / (p + 1)
    ema = window[0]
    for x in window[1:]:
        ema = alpha * x + (1 - alpha) * ema
    return ema

def wma_n(vec, period):
    """Linear Weighted Moving Average по последним N (LWMA)."""
    if len(vec) == 0:
        return 0.0
    p = _clamp_period(vec, period)
    window = vec[-p:]
    weights = range(1, len(window) + 1)
    return sum(w * x for w, x in zip(weights, window)) / sum(weights)

def stddev_n(vec, period):
    """Стандартное отклонение по последним N (волатильность)."""
    if len(vec) == 0:
        return 0.0
    p = _clamp_period(vec, period)
    window = vec[-p:]
    mean = sum(window) / len(window)
    return math.sqrt(sum((x - mean) ** 2 for x in window) / len(window))

def macd_n(vec):
    """MACD-линия: EMA(быстрый) - EMA(медленный) по всему окну.
    Периоды задаются как доли длины окна (≈12/26 EMA)."""
    n = len(vec)
    if n < 2:
        return 0.0
    fast = max(2, n // 4)
    slow = max(fast + 1, n // 2)
    return ema_n(vec, fast) - ema_n(vec, slow)

def bollinger_pctb(vec, period):
    """%B Боллинджера: положение цены в полосах (0=нижняя, 1=верхняя)."""
    if len(vec) == 0:
        return 0.5
    p = _clamp_period(vec, period)
    window = vec[-p:]
    mean = sum(window) / len(window)
    std = math.sqrt(sum((x - mean) ** 2 for x in window) / len(window))
    upper = mean + 2 * std
    lower = mean - 2 * std
    if upper - lower < 1e-10:
        return 0.5
    return (vec[-1] - lower) / (upper - lower)

def sma_series(vec, period):
    """Ряд SMA той же длины (для пересечений cross_above/cross_below)."""
    n = len(vec)
    if n == 0:
        return []
    p = _clamp_period(vec, period)
    out = []
    for i in range(n):
        start = max(0, i - p + 1)
        w = vec[start:i + 1]
        out.append(sum(w) / len(w))
    return out

def ema_series(vec, period):
    """Ряд EMA той же длины (span=N)."""
    n = len(vec)
    if n == 0:
        return []
    p = _clamp_period(vec, period)
    alpha = 2.0 / (p + 1)
    out = []
    e = vec[0]
    for x in vec:
        e = alpha * x + (1 - alpha) * e
        out.append(e)
    return out

def wma_series(vec, period):
    """Ряд LWMA той же длины."""
    n = len(vec)
    if n == 0:
        return []
    p = _clamp_period(vec, period)
    out = []
    for i in range(n):
        start = max(0, i - p + 1)
        w = vec[start:i + 1]
        weights = range(1, len(w) + 1)
        out.append(sum(wt * x for wt, x in zip(weights, w)) / sum(weights))
    return out


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
    pset.addEphemeralConstant("rand", rand_unit, SCALAR)
    # Константа-период для индикаторов (sma_n/ema_n/...): осмысленные значения 5..50
    pset.addEphemeralConstant("period", rand_period, SCALAR)

    # --- Скалярная арифметика ---
    pset.addPrimitive(scalar_add, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_sub, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_mul, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_div, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(protected_div, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_min, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_max, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_avg, [SCALAR, SCALAR], SCALAR)

    # --- Скалярные унарные ---
    pset.addPrimitive(scalar_abs, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_neg, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_sign, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_relu, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_pow2, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_sqrt, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_exp, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_log, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_sin, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_cos, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_sigmoid, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_tanh, [SCALAR], SCALAR)

    # --- Скалярные сравнения / логика ---
    pset.addPrimitive(scalar_gt, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_lt, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_gte, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_lte, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_eq, [SCALAR, SCALAR], SCALAR)
    pset.addPrimitive(scalar_pos, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_neg_check, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_clip01, [SCALAR], SCALAR)
    pset.addPrimitive(scalar_if_else, [SCALAR, SCALAR, SCALAR], SCALAR)

    # --- Векторные агрегаты (VECTOR -> SCALAR) ---
    pset.addPrimitive(last_elem, [VECTOR], SCALAR)
    pset.addPrimitive(first_elem, [VECTOR], SCALAR)
    pset.addPrimitive(scalar_mean, [VECTOR], SCALAR)
    pset.addPrimitive(scalar_std, [VECTOR], SCALAR)
    pset.addPrimitive(vec_max_elem, [VECTOR], SCALAR)
    pset.addPrimitive(vec_min_elem, [VECTOR], SCALAR)
    pset.addPrimitive(vec_sum, [VECTOR], SCALAR)
    pset.addPrimitive(vec_median, [VECTOR], SCALAR)
    pset.addPrimitive(vec_range, [VECTOR], SCALAR)
    pset.addPrimitive(vec_norm, [VECTOR], SCALAR)

    # --- Технические индикаторы (VECTOR -> SCALAR) ---
    pset.addPrimitive(vec_momentum, [VECTOR], SCALAR)
    pset.addPrimitive(vec_roc, [VECTOR], SCALAR)
    pset.addPrimitive(vec_slope, [VECTOR], SCALAR)
    pset.addPrimitive(vec_zscore, [VECTOR], SCALAR)
    pset.addPrimitive(vec_ema_last, [VECTOR], SCALAR)
    pset.addPrimitive(vec_rsi, [VECTOR], SCALAR)

    # --- Финансовые индикаторы с настраиваемым периодом (VECTOR, period -> SCALAR) ---
    pset.addPrimitive(sma_n, [VECTOR, SCALAR], SCALAR)
    pset.addPrimitive(ema_n, [VECTOR, SCALAR], SCALAR)
    pset.addPrimitive(wma_n, [VECTOR, SCALAR], SCALAR)
    pset.addPrimitive(stddev_n, [VECTOR, SCALAR], SCALAR)
    pset.addPrimitive(bollinger_pctb, [VECTOR, SCALAR], SCALAR)
    pset.addPrimitive(macd_n, [VECTOR], SCALAR)

    # --- Индикаторы как ряды (VECTOR, period -> VECTOR) — для пересечений ---
    pset.addPrimitive(sma_series, [VECTOR, SCALAR], VECTOR)
    pset.addPrimitive(ema_series, [VECTOR, SCALAR], VECTOR)
    pset.addPrimitive(wma_series, [VECTOR, SCALAR], VECTOR)

    # --- Парные векторные (VECTOR, VECTOR -> SCALAR) ---
    pset.addPrimitive(vec_dot, [VECTOR, VECTOR], SCALAR)
    pset.addPrimitive(vec_mean_diff, [VECTOR, VECTOR], SCALAR)
    pset.addPrimitive(cross_above, [VECTOR, VECTOR], SCALAR)
    pset.addPrimitive(cross_below, [VECTOR, VECTOR], SCALAR)
    pset.addPrimitive(sum_gt, [VECTOR, VECTOR], SCALAR)

    # --- Векторные преобразования (VECTOR -> VECTOR) ---
    pset.addPrimitive(vec_add, [VECTOR, VECTOR], VECTOR)
    pset.addPrimitive(vec_sub, [VECTOR, VECTOR], VECTOR)
    pset.addPrimitive(vec_mul, [VECTOR, VECTOR], VECTOR)
    pset.addPrimitive(vec_add_s, [VECTOR, SCALAR], VECTOR)
    pset.addPrimitive(vec_sub_s, [VECTOR, SCALAR], VECTOR)
    pset.addPrimitive(vec_mul_s, [VECTOR, SCALAR], VECTOR)
    pset.addPrimitive(vec_div_s, [VECTOR, SCALAR], VECTOR)
    pset.addPrimitive(vec_abs, [VECTOR], VECTOR)
    pset.addPrimitive(vec_neg, [VECTOR], VECTOR)
    pset.addPrimitive(vec_log, [VECTOR], VECTOR)
    pset.addPrimitive(vec_sqrt, [VECTOR], VECTOR)

    # --- Прочее ---
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

def meta_nand(a, b):
    return 0.0 if (a > 0.5 and b > 0.5) else 1.0

def meta_nor(a, b):
    return 0.0 if (a > 0.5 or b > 0.5) else 1.0

def meta_implies(a, b):
    # a -> b
    return 1.0 if (a <= 0.5 or b > 0.5) else 0.0

# Доп. булевая логика
pset_meta.addPrimitive(meta_nand, [BOOL, BOOL], BOOL)
pset_meta.addPrimitive(meta_nor, [BOOL, BOOL], BOOL)
pset_meta.addPrimitive(meta_implies, [BOOL, BOOL], BOOL)

# BOOL/SCALAR арифметика и решение
pset_meta.addPrimitive(meta_add, [BOOL, BOOL], SCALAR)
pset_meta.addPrimitive(meta_sub, [BOOL, BOOL], SCALAR)
pset_meta.addPrimitive(meta_mul, [BOOL, BOOL], SCALAR)
pset_meta.addPrimitive(meta_neg, [BOOL], SCALAR)
pset_meta.addPrimitive(scalar_add, [SCALAR, SCALAR], SCALAR)
pset_meta.addPrimitive(scalar_sub, [SCALAR, SCALAR], SCALAR)
pset_meta.addPrimitive(scalar_mul, [SCALAR, SCALAR], SCALAR)
pset_meta.addPrimitive(scalar_neg, [SCALAR], SCALAR)
pset_meta.addPrimitive(scalar_tanh, [SCALAR], SCALAR)
pset_meta.addPrimitive(scalar_sign, [SCALAR], SCALAR)
pset_meta.addPrimitive(scalar_if_else, [BOOL, SCALAR, SCALAR], SCALAR)
pset_meta.addEphemeralConstant("meta_rand", rand_meta, SCALAR)
