# =====================================================================
# STVGP — движок генетического программирования (DEAP)
# Коэволюция трёх популяций: long / short / meta.
# =====================================================================

import random
from datetime import datetime

import numpy as np
from deap import base, creator, gp, tools, algorithms

from primitives import pset_long, pset_short, pset_meta
from signals import evalLongTrading, evalShortTrading, evalMetaTrading, plot_signals


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


def evolve(bars, ndf, seed=None, pop_size=25, offspring=32, ngen=10,
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

    for gen in range(ngen):
        print(f"\n--- Generation {gen + 1}/{ngen} ---")

        pop_long = evolve_one_gen(pop_long, toolbox_long, hof_long,
                                  mu=pop_size, lambda_=offspring)

        pop_short = evolve_one_gen(pop_short, toolbox_short, hof_short,
                                   mu=pop_size, lambda_=offspring)

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
