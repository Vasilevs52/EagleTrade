# =====================================================================
# ROBUST — fitness-функции v2 с защитой от переобучения
# ---------------------------------------------------------------------
# Почему v1 переобучался: fitness = голая прибыль на ОДНОМ куске ОДНОЙ
# монеты, без штрафов. В пространстве из триллионов деревьев гарантированно
# находится дерево, идеальное на любых 1266 барах — даже на шуме.
#
# Принцип v2: стратегия обязана зарабатывать
#   • на НЕСКОЛЬКИХ активах (шум, случайно идеальный на BTC, почти никогда
#     не идеален ещё и на ETH/SOL/BNB/XRP),
#   • в НЕСКОЛЬКИХ временных окнах (walk-forward: шум не бывает стабилен),
#   • ПРОСТЫМ деревом (parsimony: маленькое дерево физически не может
#     запомнить много шума).
#
# fitness = mean(окна) − alpha·std(окна) − lambda·max(0, размер − лимит)
# =====================================================================

import math

from config import CFG2
from signals import (
    evalLongTrading, evalShortTrading, evalMetaTrading, precompute_active,
)

# evalXTrading возвращают -1000, когда стратегия не сделала ни одной сделки
NO_TRADE = -1000.0
# Мягкий штраф за «молчание» в отдельном окне (вместо катастрофы -1000):
# не торговать в одном спокойном режиме — не преступление, но и не плюс.
NO_TRADE_SEGMENT_PENALTY = -20.0


def aggregate(seg_returns, ind_size,
              alpha=None, lam=None, size_free=None):
    """
    Сворачивает прибыли по окнам в один robust-fitness.

    • больше половины окон без сделок -> -1000 (стратегия «мёртвая»);
    • отдельное окно без сделок -> мягкий штраф NO_TRADE_SEGMENT_PENALTY;
    • стабильность: вычитаем alpha·std по окнам — стратегия, заработавшая
      всё на одном удачном куске, проигрывает ровной;
    • parsimony: вычитаем lam за каждый узел дерева сверх size_free.
    """
    alpha = CFG2.alpha_stability if alpha is None else alpha
    lam = CFG2.lambda_size if lam is None else lam
    size_free = CFG2.size_free if size_free is None else size_free

    n = len(seg_returns)
    if n == 0:
        return -1000.0

    no_trade = sum(1 for r in seg_returns if r <= NO_TRADE + 1e-9)
    if no_trade * 2 > n:
        return -1000.0

    rets = [NO_TRADE_SEGMENT_PENALTY if r <= NO_TRADE + 1e-9 else r
            for r in seg_returns]
    mean = sum(rets) / n
    std = math.sqrt(sum((r - mean) * (r - mean) for r in rets) / n)
    size_pen = lam * max(0, ind_size - size_free)
    return mean - alpha * std - size_pen


def evalLongRobust(ind, segments):
    """Robust-fitness long-популяции: окна × активы + штрафы."""
    rs = [evalLongTrading(ind, seg)[0] for seg in segments]
    return (aggregate(rs, len(ind)),)


def evalShortRobust(ind, segments):
    """Robust-fitness short-популяции."""
    rs = [evalShortTrading(ind, seg)[0] for seg in segments]
    return (aggregate(rs, len(ind)),)


def evalMetaRobust(ind, segments, best_long_func, best_short_func, actives):
    """
    Robust-fitness meta-популяции.
    actives — список (long_active, short_active) на каждый сегмент,
    предпосчитанный один раз за поколение (см. evolve_robust).
    """
    rs = []
    for seg, (la, sa) in zip(segments, actives):
        rs.append(evalMetaTrading(ind, seg, best_long_func, best_short_func,
                                  long_active=la, short_active=sa)[0])
    return (aggregate(rs, len(ind)),)


def precompute_actives(segments, long_func, short_func):
    """precompute_active по каждому сегменту (один раз за поколение)."""
    return [precompute_active(seg, long_func, short_func) for seg in segments]
