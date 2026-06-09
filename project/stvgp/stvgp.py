# =====================================================================
# STVGP — движок генетического программирования (DEAP)
# Коэволюция трёх популяций: long / short / meta.
# =====================================================================

import random
from datetime import datetime

import numpy as np
from deap import base, creator, gp, tools, algorithms

from primitives import pset_long, pset_short, pset_meta
from signals import (
    evalLongTrading, evalShortTrading, evalMetaTrading,
    precompute_active, plot_signals,
)


# =====================================================================
# DEAP TOOLBOX
# =====================================================================

# creator.create регистрирует классы ГЛОБАЛЬНО в модуле deap.creator.
# Повторный вызов (повторный импорт в дочернем процессе, переиспользование
# в одном процессе) кидает RuntimeWarning/ошибку. Создаём идемпотентно —
# только если класс ещё не зарегистрирован.
def _safe_create(name, *args, **kwargs):
    if not hasattr(creator, name):
        creator.create(name, *args, **kwargs)

_safe_create("FitnessMax", base.Fitness, weights=(1.0,))
_safe_create("LongIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)
_safe_create("ShortIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)
_safe_create("MetaIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax)

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
# EVOLUTION
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


def evolve(bars, ndf, seed=None, pop_size=250, offspring=320, ngen=300,
           do_plot=True):
    """
    Коэволюция трёх популяций на наборе баров `bars`.
    `ndf` используется только для финальной визуализации plot_signals.
    Возвращает ((pop_long, pop_short, pop_meta), (hof_long, hof_short, hof_meta)).
    """
    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % (2**31)
    random.seed(seed)
    print(f"Random seed: {seed}  (передай в evolve(seed={seed}) чтобы воспроизвести)")

    pop_long = toolbox_long.population(n=pop_size)
    pop_short = toolbox_short.population(n=pop_size)
    pop_meta = toolbox_meta.population(n=pop_size)

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
    print(f"Bars: {len(bars)}, Pop: {pop_size}, Offspring: {offspring}, Gen: {ngen}")
    print("Signal logic: true = position open, false = position closed")
    print("=" * 60)

    prev_target = None  # (str(best_long), str(best_short)) предыдущего поколения
    for gen in range(ngen):
        print(f"\n--- Generation {gen + 1}/{ngen} ---")

        pop_long = evolve_one_gen(pop_long, toolbox_long, hof_long,
                                  mu=pop_size, lambda_=offspring)

        pop_short = evolve_one_gen(pop_short, toolbox_short, hof_short,
                                   mu=pop_size, lambda_=offspring)

        best_long_func = gp.compile(hof_long[0], pset_long)
        best_short_func = gp.compile(hof_short[0], pset_short)

        # Предпосчитываем сигналы long/short по всем барам ОДИН раз за
        # поколение — meta-особи (сотни) больше не гоняют эти деревья каждая.
        la, sa = precompute_active(bars, best_long_func, best_short_func)

        toolbox_meta.register("evaluate", evalMetaTrading,
                              bars=bars,
                              best_long_func=best_long_func,
                              best_short_func=best_short_func,
                              long_active=la, short_active=sa)

        # Meta оценивается ПРОТИВ текущих лучших long/short. Когда они
        # меняются, фитнес ранее оценённых meta-особей устаревает: selBest
        # начинает сравнивать несопоставимые числа (мишень «уплыла»).
        # Поэтому при смене мишени инвалидируем ВЕСЬ pop_meta (пересчёт
        # против новой мишени) и пересоздаём hof_meta (старые записи мерились
        # против устаревшей мишени).
        cur_target = (str(hof_long[0]), str(hof_short[0]))
        if gen == 0 or cur_target != prev_target:
            for ind in pop_meta:
                if ind.fitness.valid:
                    del ind.fitness.values
            hof_meta = tools.HallOfFame(5)
            fitnesses = list(map(toolbox_meta.evaluate, pop_meta))
            for ind, fit in zip(pop_meta, fitnesses):
                ind.fitness.values = fit
            hof_meta.update(pop_meta)
        prev_target = cur_target

        pop_meta = evolve_one_gen(pop_meta, toolbox_meta, hof_meta,
                                  mu=pop_size, lambda_=offspring)

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

    if do_plot:
        plot_signals(ndf, bars, hof_long[0], hof_short[0], hof_meta[0],
                     pset_long, pset_short, pset_meta,
                     title_prefix="[TRAIN] ")

    return (pop_long, pop_short, pop_meta), (hof_long, hof_short, hof_meta)


# =====================================================================
# EVOLUTION V2 — «честная» эволюция с защитой от переобучения
# =====================================================================

def evolve_robust(train_segments, val_segments, seed=None,
                  pop_size=None, offspring=None, ngen=None, verbose=True):
    """
    Эволюция v2 на мульти-активных walk-forward окнах (см. robust.py).

    Анти-переобучение:
      • fitness = mean(окна) − alpha·std − штраф за размер дерева;
      • финальный отбор meta — НЕ по train, а по val_segments (свежие окна,
        которых эволюция не видела);
      • возвращает и train, и val fitness — их разрыв (gap) показывает
        степень переобучения.

    Возвращает (hofs, info):
      hofs = ([best_long], [best_short], [best_meta_by_val]) — совместимо
             с storage/record_to_hofs/quick_profit_summary;
      info = {train_fit, val_fit, gap, ranked}
    """
    from robust import (evalLongRobust, evalShortRobust, evalMetaRobust,
                        precompute_actives)
    from config import CFG2

    pop_size = pop_size if pop_size is not None else CFG2.pop_size
    offspring = offspring if offspring is not None else CFG2.offspring
    ngen = ngen if ngen is not None else CFG2.ngen

    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % (2**31)
    random.seed(seed)
    if verbose:
        print(f"[v2 seed={seed}] окон train={len(train_segments)}, "
              f"val={len(val_segments)}, pop={pop_size}, ngen={ngen}")

    pop_long = toolbox_long.population(n=pop_size)
    pop_short = toolbox_short.population(n=pop_size)
    pop_meta = toolbox_meta.population(n=pop_size)

    hof_long = tools.HallOfFame(5)
    hof_short = tools.HallOfFame(5)
    # Meta-HOF шире: это пул кандидатов для отбора по валидации.
    hof_meta = tools.HallOfFame(25)

    toolbox_long.register("evaluate", evalLongRobust, segments=train_segments)
    toolbox_short.register("evaluate", evalShortRobust, segments=train_segments)

    for pop, tb in [(pop_long, toolbox_long), (pop_short, toolbox_short)]:
        fits = list(map(tb.evaluate, pop))
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit
    hof_long.update(pop_long)
    hof_short.update(pop_short)

    prev_target = None
    best_long_func = best_short_func = None
    for gen in range(ngen):
        pop_long = evolve_one_gen(pop_long, toolbox_long, hof_long,
                                  mu=pop_size, lambda_=offspring)
        pop_short = evolve_one_gen(pop_short, toolbox_short, hof_short,
                                   mu=pop_size, lambda_=offspring)

        best_long_func = gp.compile(hof_long[0], pset_long)
        best_short_func = gp.compile(hof_short[0], pset_short)
        actives = precompute_actives(train_segments,
                                     best_long_func, best_short_func)

        toolbox_meta.register("evaluate", evalMetaRobust,
                              segments=train_segments,
                              best_long_func=best_long_func,
                              best_short_func=best_short_func,
                              actives=actives)

        # Та же логика «плавающей мишени», что в v1: при смене лучших
        # long/short фитнес старых meta-особей несопоставим — пересчитываем.
        cur_target = (str(hof_long[0]), str(hof_short[0]))
        if gen == 0 or cur_target != prev_target:
            for ind in pop_meta:
                if ind.fitness.valid:
                    del ind.fitness.values
            hof_meta = tools.HallOfFame(25)
            fits = list(map(toolbox_meta.evaluate, pop_meta))
            for ind, fit in zip(pop_meta, fits):
                ind.fitness.values = fit
            hof_meta.update(pop_meta)
        prev_target = cur_target

        pop_meta = evolve_one_gen(pop_meta, toolbox_meta, hof_meta,
                                  mu=pop_size, lambda_=offspring)

        if verbose and (gen % 5 == 0 or gen == ngen - 1):
            mf = [ind.fitness.values[0] for ind in pop_meta]
            print(f"  [v2 seed={seed}] gen {gen + 1}/{ngen}  "
                  f"meta max={max(mf):.2f} avg={np.mean(mf):.2f}")

    # ----- Финальный отбор по ВАЛИДАЦИИ (окна, которых train не видел) -----
    from robust import evalMetaRobust as _emr
    val_actives = precompute_actives(val_segments,
                                     best_long_func, best_short_func)
    ranked = []
    for m in hof_meta:
        train_fit = m.fitness.values[0]
        val_fit = _emr(m, val_segments, best_long_func, best_short_func,
                       val_actives)[0]
        ranked.append((val_fit, train_fit, m))
    ranked.sort(key=lambda x: x[0], reverse=True)

    best_val, best_train, best_meta = ranked[0]
    info = {
        "train_fit": best_train,
        "val_fit": best_val,
        "gap": best_train - best_val,
        # топ-10 для анализа (val, train, дерево)
        "ranked": [(v, t, str(m)) for v, t, m in ranked[:10]],
    }
    if verbose:
        print(f"[v2 seed={seed}] ОТБОР ПО ВАЛИДАЦИИ: "
              f"val={best_val:.2f}, train={best_train:.2f}, "
              f"gap={info['gap']:.2f} (меньше gap = меньше переобучение)")

    return ((hof_long, hof_short, [best_meta]), info)
