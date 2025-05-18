import random
import numpy as np
import pandas as pd
import abc
from typing import List
from geneticengine.grammar.decorators import abstract
from geneticengine.representations.tree.treebased import  TreeNode
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.problems import SingleObjectiveProblem



# ==================================================================
# 1. Абстрактный стартовый класс с использованием abc.ABC
# ==================================================================
@abstract
class MathExpr(abc.ABC,TreeNode):
    @abc.abstractmethod
    def evaluate(self, data: dict, i: int) -> float:
        pass

# ==================================================================
# 2. Конкретные узлы AST (арифметические операции)
# ==================================================================
class Add(MathExpr):
    def __init__(self, left: MathExpr, right: MathExpr):
        self.left = left
        self.right = right

    def evaluate(self, data: dict, i: int) -> float:
        return self.left.evaluate(data, i) + self.right.evaluate(data, i)

class Sub(MathExpr):
    def __init__(self, left: MathExpr, right: MathExpr):
        self.left = left
        self.right = right

    def evaluate(self, data: dict, i: int) -> float:
        return self.left.evaluate(data, i) - self.right.evaluate(data, i)

class Mul(MathExpr):
    def __init__(self, left: MathExpr, right: MathExpr):
        self.left = left
        self.right = right

    def evaluate(self, data: dict, i: int) -> float:
        return self.left.evaluate(data, i) * self.right.evaluate(data, i)

class Div(MathExpr):
    def __init__(self, left: MathExpr, right: MathExpr):
        self.left = left
        self.right = right

    def evaluate(self, data: dict, i: int) -> float:
        denom = self.right.evaluate(data, i)
        return self.left.evaluate(data, i) / (abs(denom) + 1e-6)

class Log(MathExpr):
    def __init__(self, child: MathExpr):
        self.child = child

    def evaluate(self, data: dict, i: int) -> float:
        val = self.child.evaluate(data, i)
        return np.log(abs(val) + 1e-6)

# ==================================================================
# 3. Индикаторы
# ==================================================================
class Price(MathExpr):
    def evaluate(self, data: dict, i: int) -> float:
        return data['price'].iloc[i]

class SMA50(MathExpr):
    def evaluate(self, data: dict, i: int) -> float:
        return data['sma50'].iloc[i]

class EMA50(MathExpr):
    def evaluate(self, data: dict, i: int) -> float:
        return data['ema50'].iloc[i]

# ==================================================================
# 4. Грамматика
# ==================================================================
grammar = extract_grammar([Add, Price], MathExpr)
representation = TreeBasedRepresentation(grammar)


# ==================================================================
# 3. Фитнес-функция и проблема
# ==================================================================
def fitness_function(individual: MathExpr, prices: pd.Series) -> float:
    data = {'price': prices}
    try:
        signals = [individual.evaluate(data, i) for i in range(len(prices))]
        pnl = sum(prices.diff().iloc[i] * signals[i - 1] for i in range(1, len(prices)))
        return pnl
    except:
        return -9999.0


problem = SingleObjectiveProblem(
    fitness_function=lambda ind: fitness_function(ind, prices),
    maximize=True
)


# ==================================================================
# 4. Кастомная реализация GP
# ==================================================================
class CustomGP:
    def __init__(
            self,
            representation,
            problem,
            population_size=100,
            generations=50,
            mutation_rate=0.1
    ):
        self.representation = representation
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def evolve(self) -> MathExpr:
        # Инициализация популяции
        population = [self.representation.create_node() for _ in range(self.population_size)]

        for gen in range(self.generations):
            # Оценка приспособленности
            fitness = [self.problem.evaluate(ind) for ind in population]

            # Селекция (турнир)
            selected = []
            for _ in range(self.population_size):
                candidates = random.sample(list(zip(population, fitness)), 3)
                winner = max(candidates, key=lambda x: x[1])[0]
                selected.append(winner)

            # Кроссовер и мутация
            new_population = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[i + 1] if (i + 1) < len(selected) else selected[i]
                c1, c2 = self.representation.crossover(p1, p2)
                c1 = self.representation.mutate(c1, self.mutation_rate)
                c2 = self.representation.mutate(c2, self.mutation_rate)
                new_population.extend([c1, c2])

            population = new_population

            # Лучшая особь
            best = max(zip(population, fitness), key=lambda x: x[1])[0]
            print(f"Gen {gen} | Best PnL: {self.problem.evaluate(best):.2f}")

        return max(population, key=lambda x: self.problem.evaluate(x))