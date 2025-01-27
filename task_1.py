"""
Завдання 1. Оптимізація виробництва

Компанія виробляє два види напоїв: "Лимонад" і "Фруктовий сік".
 Для виробництва цих напоїв використовуються різні інгредієнти та обмежена кількість обладнання.
 Задача полягає у максимізації виробництва, враховуючи обмежені ресурси.

Умови завдання:

"Лимонад" виготовляється з "Води", "Цукру" та "Лимонного соку".
"Фруктовий сік" виготовляється з "Фруктового пюре" та "Води".
Обмеження ресурсів: 100 од. "Води", 50 од. "Цукру", 30 од. "Лимонного соку" та 40 од. "Фруктового пюре".
Виробництво одиниці "Лимонаду" вимагає 2 од. "Води", 1 од. "Цукру" та 1 од. "Лимонного соку".
Виробництво одиниці "Фруктового соку" вимагає 2 од. "Фруктового пюре" та 1 од. "Води".
Використовуючи PuLP, створіть модель, яка визначає, скільки "Лимонаду" та "Фруктового соку" потрібно виробити для максимізації загальної кількості продуктів, дотримуючись обмежень на ресурси. Напишіть програму, код якої максимізує загальну кількість вироблених продуктів "Лимонад" та "Фруктовий сік", враховуючи обмеження на кількість ресурсів.
"""
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

model = LpProblem(name="drink-production", sense=LpMaximize)

x = LpVariable(name="lemonade", lowBound=0, cat="Integer")
y = LpVariable(name="fruit_juice", lowBound=0, cat="Integer")

# цільову функція
model += lpSum([x, y]), "Maximize number of drinks"

# обмеження на ресурси
model += (2 * x + y <= 100, "Water constraint")
model += (x <= 50, "Sugar constraint")
model += (x <= 30, "Lemon juice constraint")
model += (2 * y <= 40, "Fruit puree constraint")

model.solve()

lemonade_produced = x.value()
fruit_juice_produced = y.value()

print(f"Optimal number of Lemonade units to produce: {lemonade_produced}")
print(f"Optimal number of Fruit Juice units to produce: {fruit_juice_produced}")
