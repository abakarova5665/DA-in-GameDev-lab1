# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Абакарова Кистаман Умарасхабовна
- X21IT_AI-01BL
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами языка Python на примере реализации линейной регрессии.

## Задание 1
### Вывод строки "Hello World!" в Python
```py
print ("Hello, World!")
```
![Lab1_Task1](https://user-images.githubusercontent.com/48391156/193573415-e5cc2379-7613-4019-a9df-c50ccee8f41c.png)


### Вывод строки "Hello World!" в Unity
```py
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Lab1_Task1 : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Hello, World!!!");
    }

}
```
![Lab1_Task1,1](https://user-images.githubusercontent.com/48391156/193578053-d70f4bb1-8022-41ba-a8d6-01f873d15d21.png)

## Задание 2
### Пошагово выполнить каждый пункт с описанием и примером реализации задачи по теме лабораторной работы.

Ход работы:
- Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline
# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)
#Show the effect of a scatter plot
plt.scatter(x,y)
```
![image](https://user-images.githubusercontent.com/48391156/193639103-518d2827-f45b-48c3-bda1-c9e75fdadac1.png)



- Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py
#The basic linear regression model is wx+ b, and since this is a two-dimensional space, the model is ax+ b
def model(a, b, x):
  return a*x + b
#Tahe most commonly used loss function of linear regression model is the lossfunction of mean variance difference
def loss_function(a, b, x, y):
  num = len(x)
  prediction=model(a,b,x)
  return (0.5/num) * (np.square(prediction-y)).sum()
#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize(a,b,x,y):
  num = len(x)
  prediction = model(a,b,x)
#Update the values of A and B by finding the partial derivatives of the loss function on a and b
  da = (1.0/num) * ((prediction -y)*x).sum()
  db = (1.0/num) * ((prediction -y).sum())
  a = a - Lr*da
  b = b - Lr*db
  return a, b
#iterated function, return a and b
def iterate(a,b,x,y,times):
  for i in range(times):
    a,b = optimize(a,b,x,y)
  return a,b
```

Начать итерацию

Шаг 1. Инициализация и модель итеративной оптимизации
```py
#Initialize parameters and display
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001
#For the first iteration, the parameter values, losses, and visualization after the iteration are displayed
a,b = iterate(a,b,x,y,1)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```

[0.39804736]
[0.20904672]
[0.40229218] [0.20910828] 3040.760488335982
[<matplotlib.lines.Line2D at 0x1ac156e0490>]
![image](https://user-images.githubusercontent.com/48391156/194259282-50113ba5-d172-479a-90c9-e550d33de809.png)

Шаг 2 На второй итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации
```py
a,b = iterate(a,b,x,y,2)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[0.41074187] [0.2092308] 3005.1103630458274
[<matplotlib.lines.Line2D at 0x1ac15747460>]
![image](https://user-images.githubusercontent.com/48391156/194259392-8b80dcd9-f7a8-473e-ace7-4addd3473d78.png)


Шаг 3 Третья итерация показывает значения параметров, значения потерь и
визуализацию после итерации
```py
a,b = iterate(a,b,x,y,3)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[0.42331714] [0.20941307] 2952.4691845375987
[<matplotlib.lines.Line2D at 0x1ac1781e770>]
![image](https://user-images.githubusercontent.com/48391156/194259468-73421264-26d0-4d9f-be8f-892dee628567.png)


Шаг 4 На четвертой итерации отображаются значения параметров, значения потерь и эффекты визуализации
```py
a,b = iterate(a,b,x,y,4)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[0.43990065] [0.2096533] 2883.8086883273045
[<matplotlib.lines.Line2D at 0x1ac178b54b0>]
![image](https://user-images.githubusercontent.com/48391156/194259548-84784319-ee56-4389-95ce-f71dd5534851.png)


Шаг 5 Пятая итерация показывает значение параметра, значение потерь и эффект визуализации после итерации
```py
a,b = iterate(a,b,x,y,5)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[0.46033877] [0.20994916] 2800.377416862917
[<matplotlib.lines.Line2D at 0x1ac17a80e50>]
![image](https://user-images.githubusercontent.com/48391156/194259620-bde75e88-4048-436d-97d3-f3acb855886a.png)


Шаг 6 10000-я итерация, показывающая значения параметров, потери и визуализацию после итерации
```py
a,b = iterate(a,b,x,y,10000)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[1.7501248] [0.19898818] 189.35563420094786
[<matplotlib.lines.Line2D at 0x1ac17aebaf0>]
![image](https://user-images.githubusercontent.com/48391156/194259668-0da72e24-5cff-4dc5-8e1b-47df20851951.png)



## Выводы
При выполнения данной лабораторной работы я ознакомилась со средой разработки VS Code, облачной средой Google Colab, Jupiter, а также со средой разработки Unity. Узнала базовые функции в Python и C#. Научилась выводить строки в Unity. А так же попыталась разобраться в непростом для меня кодом на примере линейной регрессии.

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
