import sys

import numpy


def read_matrix(file_name):
    matrix = []
    with open(file_name, 'r') as file:
        for line in file:
            matrix.append(list(map(float, line.split())))

    return numpy.array(matrix, float)


EPS = 0.01


def itaration_method(matrix):
    for i in range(len(matrix)):
        matrix[i] *= 1 / matrix[i][i]
    print(matrix)
    b = numpy.array([matrix[i][:-1] for i in range(len(matrix))], float)
    b = numpy.identity(len(matrix)) - b
    c = numpy.array([matrix[i][-1:] for i in range(len(matrix))])
    if norm(b) >= 1:
        print('Метод простых итераций расходится')
        return 0
    x = [c]
    while True:
        x.append(numpy.dot(b, x[-1]) + c)
        print(f'{len(x) - 1} шаг')
        print(x[-1])

        if norm(x[-1] - x[-2]) < EPS:
            break
    print(f'Кол-во шагов: {len(x) - 1}')
    return x[-1]


def norm(matrix):
    matrix_norm = numpy.sum(numpy.abs(matrix[0]))
    for i in range(1, len(matrix)):
        sum = numpy.sum(numpy.abs(matrix[i]))
        if matrix_norm < sum:
            matrix_norm = sum
    return matrix_norm


def Seidel_method(matrix):
    for i in range(len(matrix)):
        matrix[i] *= 1 / matrix[i][i]
    b = numpy.array([matrix[i][:-1] for i in range(len(matrix))], float)
    b = numpy.identity(len(matrix)) - b
    c = numpy.array([matrix[i][-1:] for i in range(len(matrix))])
    if norm(b) >= 1:
        return 0
    x = [c.copy()]
    step = 0
    while True:
        step += 1
        x.append(c.copy())
        for i in range(len(x[-1])):
            for j in range(i):
                x[-1][i] += b[i][j] * x[-1][j]
            for j in range(i + 1, len(matrix)):
                x[-1][i] += b[i][j] * x[-2][j]
        print(f'{step} шаг\n{x[-1]}')
        if norm(x[-1] - x[-2]) < EPS:
            break

    return x[-1]


def main():
    matrix = read_matrix('input2.txt')
    a = numpy.array([matrix[i][:-1] for i in range(len(matrix))], float)
    b = numpy.array([matrix[i][-1:] for i in range(len(matrix))])
    print(f'Матрица коэффициентов системы\n{a}')
    print(f'Столбец свободных членов\n{b}')
    x = itaration_method(matrix.copy())
    if type(x) == type(0) and x == 0:
        print('Выполнение метода итераций невозможно')
    else:
        print(f'Метод итераций выполнен успешно')
        print(f'Столбец неизвестных\n{x}')
        print('Проверка: A * X = B')
        print(a.dot(x))
    x = Seidel_method(matrix.copy())
    if type(x) == type(0) and x == 0:
        print('Выполнение метода Зейделя невозможно')
    else:
        print(f'Метод Зейделя выполнен успешно')
        print(f'Столбец неизвестных\n{x}')
        print('Проверка: A * X = B')
        print(a.dot(x))


if __name__ == '__main__':
    main()