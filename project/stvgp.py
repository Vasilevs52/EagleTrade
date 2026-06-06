import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────
# 1. Тестовые функции
# ─────────────────────────────────────────────────────────────
def find_root_numerically(fp, a, b):
    """Численное нахождение корня для определения точного минимума"""
    try:
        return brentq(fp, a, b)
    except:
        return None

TEST_FUNCS = {
    "1": {
        "name": "Полином: f(x) = x⁴ - 4x³ + 6x²",
        "f": lambda x: x ** 4 - 4 * x ** 3 + 6 * x ** 2,
        "fp": lambda x: 4 * x ** 3 - 12 * x ** 2 + 12 * x,  # = 4x(x² - 3x + 3)
        "interval": (-0.5, 2.5),
        "x_true_num": None  # Будет вычислено
    },
    "2": {
        "name": "Экспоненциальная: f(x) = eˣ - 2x",
        "f": lambda x: np.exp(x) - 2 * x,
        "fp": lambda x: np.exp(x) - 2,
        "interval": (0.0, 2.0),
        "x_true_num": None
    },
    "3": {
        "name": "Тригонометрическая: f(x) = sin(x) + x²/4",
        "f": lambda x: np.sin(x) + x ** 2 / 4,
        "fp": lambda x: np.cos(x) + x / 2,
        "interval": (-2.0, 1.0),
        "x_true_num": None
    },
    "4": {
        "name": "Плохая для секущих: f(x) = x³",
        "f": lambda x: x**3,
        "fp": lambda x: 3 * x**2,
        "interval": (-1.0, 1.0),
        "x_true_num": 0.0
    }
}

# Вычисляем точные корни заранее
for key in TEST_FUNCS:
    func = TEST_FUNCS[key]
    root = find_root_numerically(func['fp'], *func['interval'])
    func['x_true'] = root if root is not None else 0.0
    print(f"Функция {key}: точный корень f'(x)=0 найден: x* = {func['x_true']:.10f}")

# ─────────────────────────────────────────────────────────────
# 2. Методы оптимизации
# ─────────────────────────────────────────────────────────────
def bisection_method(f, a, b, x_true,
                     tol=1e-12,
                     max_iter=100):

    history = []
    stop_reason = "Достигнут max_iter"

    if a >= b:
        print("[WARNING] Некорректный интервал")
        return {
            "history": [],
            "iterations": 0,
            "final_error": None,
            "final_x": None,
            "stop_reason": "Некорректный интервал"
        }

    # Начальные точки
    x1 = a + (b - a) / 4
    x2 = (a + b) / 2
    x3 = b - (b - a) / 4

    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)

    for i in range(max_iter):

        # Проверка NaN
        if np.isnan(f1) or np.isnan(f2) or np.isnan(f3):
            stop_reason = f"NaN на итерации {i+1}"
            break

        # Сужение интервала
        if f1 <= f2:

            # Оставляем [a, x2]
            b = x2

            # старая x1 становится новой серединой
            x2 = x1
            f2 = f1

            # новые внутренние точки
            x1 = a + (x2 - a) / 2
            x3 = x2 + (b - x2) / 2

            f1 = f(x1)
            f3 = f(x3)

        else:

            if f2 <= f3:

                # Оставляем [x1, x3]
                a = x1
                b = x3

                # x2 уже середина нового интервала
                # f2 уже известно

                x1 = a + (x2 - a) / 2
                x3 = x2 + (b - x2) / 2

                f1 = f(x1)
                f3 = f(x3)

            else:

                # Оставляем [x2, b]
                a = x2

                # старая x3 становится новой серединой
                x2 = x3
                f2 = f3

                # новые внутренние точки
                x1 = a + (x2 - a) / 2
                x3 = x2 + (b - x2) / 2

                f1 = f(x1)
                f3 = f(x3)

        current_x = x2

        err = abs(current_x - x_true)
        history.append(err)

        # Критерий остановки
        if (b - a) / 2 < tol:
            stop_reason = "Интервал стал меньше tol"
            break

        if abs(b - a) < 1e-15:
            stop_reason = "Интервал слишком мал"
            break
    final_x = (a + b) / 2

    return {
        "history": history,
        "iterations": len(history),
        "final_error": history[-1] if history else None,
        "final_x": final_x,
        "stop_reason": stop_reason
    }

def secant_method(fp, a, b, x_true,
                  tol=1e-12,
                  max_iter=100):

    history = []
    stop_reason = "Достигнут max_iter"

    c = None
    f_a, f_b = fp(a), fp(b)

    for i in range(max_iter):

        denom = f_b - f_a

        if abs(denom) < 1e-14:
            stop_reason = f"Деление на почти ноль (итерация {i+1})"
            break

        c = a - f_a * (b - a) / denom

        err = abs(c - x_true)
        history.append(err)

        # Успешная остановка
        if err < tol:
            stop_reason = "Достигнута точность"
            break

        # Расходимость
        if abs(c) > 1e8 or np.isnan(c):
            stop_reason = f"Расходимость на итерации {i+1}"
            break

        # Переход
        a, f_a = b, f_b
        b, f_b = c, fp(c)

    final_x = c

    return {
        "history": history,
        "iterations": len(history),
        "final_error": history[-1] if history else None,
        "final_x": final_x,
        "stop_reason": stop_reason
    }

# ─────────────────────────────────────────────────────────────
# 3. Анализ сходимости
# ─────────────────────────────────────────────────────────────
def analyze_convergence(history):
    """Расчёт коэффициентов сходимости"""
    if len(history) < 2:
        return [], []

    ratios = []
    for i in range(1, len(history)):
        if history[i - 1] > 1e-15:
            ratios.append(history[i] / history[i - 1])
        else:
            ratios.append(0.0)

    # Порядок сходимости
    orders = []
    for i in range(2, min(len(history) - 1, 15)):  # Только первые 15 итераций
        e_prev, e_curr, e_next = history[i - 1], history[i], history[i + 1]

        if e_prev > 1e-10 and e_curr > 1e-10 and e_next > 1e-15:
            if e_curr < e_prev * 1.5:  # Ошибка не должна сильно расти
                r1 = e_curr / e_prev
                r2 = e_next / e_curr

                if 1e-8 < r1 < 2.0 and 1e-8 < r2 < 2.0:
                    try:
                        if abs(np.log(r1 + 1e-30)) < 1e-12:
                            continue

                        order = np.log(r2 + 1e-30) / np.log(r1 + 1e-30)

                        if 0.3 < order < 2.5:
                            orders.append(order)

                    except:
                        pass

    return ratios, orders

def print_results_table(name, history, ratios, orders):
    """Вывод результатов"""
    print(f"\n[FUNC] {name}:")
    print(f"   Итераций: {len(history)}")
    print(f"   Финальная ошибка: {history[-1]:.2e}")

    if orders:
        avg_order = np.mean(orders[-min(3, len(orders)):])
        print(f"   Порядок сходимости: {avg_order:.3f}")

    print(f"   {'k':<3} | {'Ошибка':<14} | {'e_k/e_(k-1)':<12}")
    print(f"   {'-' * 3} | {'-' * 14} | {'-' * 12}")
    for i in range(min(8, len(history))):
        err_str = f"{history[i]:.2e}"
        ratio_str = f"{ratios[i]:.4f}" if i < len(ratios) else "—"
        print(f"   {i + 1:<3} | {err_str:<14} | {ratio_str:<12}")

# ─────────────────────────────────────────────────────────────
# 4. Интерактивный режим
# ─────────────────────────────────────────────────────────────
def main():
    while True:
        print("\n" + "=" * 65)
        print("  СРАВНЕНИЕ МЕТОДОВ ОДНОМЕРНОЙ ОПТИМИЗАЦИИ")
        print("=" * 65)
        for key, val in TEST_FUNCS.items():
            print(f"{key}. {val['name']}")
        print("5. Выход")

        choice = input("\nВыберите функцию [1-5]: ").strip()
        if choice == '5':
            break
        if choice not in TEST_FUNCS:
            print("[WARNING]  Неверный выбор!")
            continue

        func = TEST_FUNCS[choice]
        print(f"\n[OK] Выбрано: {func['name']}")
        print(f"[TARGET] Точный минимум: x* = {func['x_true']:.10f}")
        print(f"[INFO] Интервал: {func['interval']}")

        try:
            print("\n[PROCESS] Запуск методов...")
            res_bis = bisection_method(
                func['f'],
                *func['interval'],
                func['x_true']
            )

            res_sec = secant_method(
                func['fp'],
                func['interval'][0],
                func['interval'][1],
                func['x_true']
            )

            hist_bis = res_bis["history"]
            hist_sec = res_sec["history"]
        except Exception as e:
            print(f"[ERROR] Ошибка: {e}")
            continue

        rat_bis, ord_bis = analyze_convergence(hist_bis)
        rat_sec, ord_sec = analyze_convergence(hist_sec)

        print("\n[RESULT] Бисекция")
        print(f"  Итераций: {res_bis['iterations']}")
        print(f"  Найденный минимум x = {res_bis['final_x']:.12f}")
        print(f"  Финальная ошибка: {res_bis['final_error']:.2e}")
        print(f"  Причина остановки: {res_bis['stop_reason']}")

        print("\n[RESULT] Метод секущих")
        print(f"  Итераций: {res_sec['iterations']}")
        print(f"  Найденный минимум x = {res_sec['final_x']:.12f}")
        print(f"  Финальная ошибка: {res_sec['final_error']:.2e}")
        print(f"  Причина остановки: {res_sec['stop_reason']}")

        print(f"\n{'=' * 65}")
        print(f"{'РЕЗУЛЬТАТЫ':^65}")
        print(f"{'=' * 65}")
        print(f"{'Метод':<30} | {'Итераций':<10} | {'Ошибка':<12}")
        print(f"{'-' * 65}")
        print(f"{'Бисекция':<30} | {len(hist_bis):<10} | {hist_bis[-1]:.2e}")
        print(f"{'Метод секущих':<30} | {len(hist_sec):<10} | {hist_sec[-1]:.2e}")

        print_results_table("Бисекция", hist_bis, rat_bis, ord_bis)
        print_results_table("Метод секущих", hist_sec, rat_sec, ord_sec)

        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # График сходимости
        ax1.semilogy(range(len(hist_bis)), hist_bis, 'bo-', label='Бисекция',
                     markersize=5, linewidth=1.5, alpha=0.8)
        ax1.semilogy(range(len(hist_sec)), hist_sec, 'rs-', label='Секущие',
                     markersize=5, linewidth=1.5, alpha=0.8)
        ax1.axhline(1e-12, color='gray', linestyle=':', alpha=0.5, label='Точность 1e-12')
        ax1.set_xlabel('Номер итерации k', fontsize=11)
        ax1.set_ylabel('Ошибка |x_k - x*| (лог)', fontsize=11)
        ax1.set_title('Сходимость методов', fontsize=12, fontweight='bold')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.legend()

        # График скорости
        if rat_bis:
            ax2.plot(range(1, len(rat_bis) + 1), rat_bis, 'bo-', label='Бисекция',
                     markersize=5, linewidth=1.5, alpha=0.8)
        if rat_sec:
            ax2.plot(range(1, len(rat_sec) + 1), rat_sec, 'rs-', label='Секущие',
                     markersize=5, linewidth=1.5, alpha=0.8)
        ax2.axhline(0.5, color='red', linestyle=':', linewidth=1.5, label='Линейная (0.5)')
        ax2.set_xlabel('Номер итерации k', fontsize=11)
        ax2.set_ylabel('Коэффициент e_k / e_(k-1)', fontsize=11)
        ax2.set_title('Скорость сходимости по шагам', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(-0.1, max(2.0, max(rat_bis + rat_sec) * 1.2) if (rat_bis and rat_sec) else 2.0)

        plt.tight_layout()
        plt.show()

        if input("\n[PROCESS] Продолжить? (y/n): ").strip().lower() != 'y':
            break

if __name__ == "__main__":
    main()