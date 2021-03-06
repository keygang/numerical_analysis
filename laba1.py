import numpy
import sys


def read_matrix(file_name):
    matrix = []
    with open(file_name, 'r') as file:
        for line in file:
            matrix.append(list(map(float, line.split())))

    return numpy.array(matrix, float)


def gauss(matrix):
    set_lines = set()
    for i in range(len(matrix)):
        first = -1
        for j in range(len(matrix)):
            if matrix[j][i] != 0 and (j not in set_lines):
                first = j
                break
        if first == -1:
            continue
        set_lines.add(first)
        for j in range(len(matrix)):
            if matrix[j][i] != 0 and (j not in set_lines):
                matrix[j] -= matrix[first].dot(matrix[j][i] / matrix[first][i])
                print(matrix)
                for k in range(10):
                    print('_', end='')
                print()
    x = []
    for i in range(len(matrix) - 1, -1, -1):
        sum = 0
        for j in range(i + 1, len(matrix)):
            sum += x[j - i - 1] * matrix[i][j]
        if matrix[i][i] == 0 and matrix[i][-1] - sum != 0:
            print('Ошибка вычесления (деление на ноль)')
            sys.exit(0)
        x.insert(0, (matrix[i][-1] - sum) / matrix[i][i])
    return numpy.array([x], float).transpose()


def norm(matrix):
    matrix_norm = numpy.sum(numpy.abs(matrix[0]))
    for i in range(1, len(matrix)):
        sum = numpy.sum(numpy.abs(matrix[i]))
        if matrix_norm < sum:
            matrix_norm = sum
    return matrix_norm


def main():
    matrix = read_matrix('input1.txt')
    B_ABSOLUTE_ERROR = 0.001
    a = numpy.array([matrix[i][:-1] for i in range(len(matrix))], float)
    b = numpy.array([matrix[i][-1:] for i in range(len(matrix))])
    print(f'Матрица коэффициентов системы\n{a}')
    print(f'Столбец свободных членов\n{b}')
    if numpy.linalg.det(a) == 0:
        print('Определитель матрицы коэффициентов равен нулю')
        sys.exit(0)
    x = gauss(matrix.copy())
    print(f'Столбец неизвестных\n{x}')
    print(f'Проверка\n A * X =\n{a.dot(x)}')
    a_inv = numpy.linalg.inv(a.copy())
    print(f'Обратная матрица коэффициентов системы\n{a_inv}')
    b_norm = norm(b)
    x_norm = norm(x)
    a_norm = norm(a)
    a_inv_norm = norm(a_inv)
    x_absolute_error = a_inv_norm * B_ABSOLUTE_ERROR / x_norm
    x_relative_error = a_norm * a_inv_norm * B_ABSOLUTE_ERROR / b_norm
    print(f'Абсолютная погрешность решения системы: {x_absolute_error * x_norm}')
    print(f'Оценка относительной погрешности решения системы: {x_absolute_error} <= {x_relative_error}')


if __name__ == "__main__":
    main()
