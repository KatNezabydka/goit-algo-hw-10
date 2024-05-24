"""
Завдання 2. Обчислення визначеного інтеграла.
Обчисліть значення інтеграла функції за допомогою методу Монте-Карло, інакше кажучи,
 знайдіть площу під цим графіком (сіра зона).
Перевірте правильність розрахунків, щоб підтвердити точність методу Монте-Карло,
шляхом порівняння отриманого результату та аналітичних розрахунків або результату виконання функції quad.
Зробіть висновки.
"""

import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as spi

def f(x):
    return x ** 2


# межі інтегрування
a = 0  # нижня межа
b = 2  # верхня межа

# Обчислення інтеграла
result, error = spi.quad(f, a, b)

print("Integral: ", result)


# Метод Монте-Карло
def monte_carlo_integral(f, a, b, N):
    x_random = np.random.uniform(a, b, N)

    # Обчислення значення функції в цих точках
    f_values = f(x_random)

    # Обчислення середнього значення функції
    f_mean = np.mean(f_values)

    # Оцінка інтегралу
    integral = (b - a) * f_mean

    return integral


N = 100000

integral_value = monte_carlo_integral(f, a, b, N)
print(f"Estimated value of the integral (Monte Carlo): {integral_value}")

x = np.linspace(-0.5, 2.5, 400)
y = f(x)
fig, ax = plt.subplots()
ax.plot(x, y, 'r', linewidth=2)
ix = np.linspace(a, b, 100)
iy = f(ix)
ax.fill_between(ix, iy, color='gray', alpha=0.3)
ax.set_xlim([x[0], x[-1]])
ax.set_ylim([0, max(y) + 0.1])
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.axvline(x=a, color='gray', linestyle='--')
ax.axvline(x=b, color='gray', linestyle='--')
ax.set_title(f'Графік інтегрування f(x) = x^2 від {a} до {b}')
plt.grid()
plt.show()
