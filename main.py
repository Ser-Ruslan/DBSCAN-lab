"""Простейшая реализация DBSCAN.

python -m pip install numpy
python main.py
"""

import numpy as np


def find_neighbors(X, point_index, eps):
    """Индексы точек на расстоянии <= eps, включая саму точку."""
    neighbors = []
    for other_index in range(len(X)):
        # Евклидово расстояние: корень из суммы квадратов разностей.
        difference = X[point_index] - X[other_index]
        distance = np.sqrt(np.sum(difference ** 2))
        if distance <= eps:
            neighbors.append(other_index)
    return neighbors


def dbscan(X, eps=0.5, min_samples=3):
    """Вернуть метки: 0, 1, ... — кластеры; -1 — шум.

    X — числовой массив формы (число точек, число признаков).
    eps — положительный радиус окрестности.
    min_samples — минимум точек в окрестности ядровой точки.
    Сама точка тоже входит в это количество.

    Поиск соседей выполняется полным перебором. Время O(n² * d),
    дополнительная память O(n) без учёта входного массива.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X должен быть двумерным массивом с признаками.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X должен содержать только конечные числа.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps должен быть положительным конечным числом.")
    if (isinstance(min_samples, (bool, np.bool_))
            or not isinstance(min_samples, (int, np.integer))
            or min_samples < 1):
        raise ValueError("min_samples должен быть целым числом >= 1.")

    labels = np.full(len(X), -1, dtype=int)
    visited = np.zeros(len(X), dtype=bool)
    cluster_id = 0

    for point_index in range(len(X)):
        if visited[point_index]:
            continue

        visited[point_index] = True
        neighbors = find_neighbors(X, point_index, eps)

        if len(neighbors) < min_samples:
            # Пока оставляем метку -1. Позже точка может стать граничной.
            continue

        # Нашли ядровую точку: создаём и расширяем новый кластер.
        labels[point_index] = cluster_id
        queue = neighbors.copy()
        queued = set(neighbors)  # Не добавляем один индекс повторно.
        position = 0

        while position < len(queue):
            current = queue[position]
            position += 1

            if not visited[current]:
                visited[current] = True
                current_neighbors = find_neighbors(X, current, eps)

                # Кластер расширяется только через ядровые точки.
                if len(current_neighbors) >= min_samples:
                    for neighbor in current_neighbors:
                        if neighbor not in queued:
                            queue.append(neighbor)
                            queued.add(neighbor)

            if labels[current] == -1:
                # Добавляем и новые точки, и ранее отмеченный шум.
                # Точки другого кластера не переназначаем.
                labels[current] = cluster_id

        cluster_id += 1

    return labels


def main():
    # Две плотные группы и две удалённые точки шума.
    # Координаты в одной шкале, поэтому масштабирование здесь не нужно.
    X = np.array([
        [1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.2, 1.1],
        [5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.2, 5.1],
        [9.0, 1.0], [0.0, 8.0],
    ])

    labels = dbscan(X, eps=0.3, min_samples=3)
    print("Метки кластеров:", labels)
    print("Количество кластеров:", len(set(labels) - {-1}))
    print("Количество точек шума:", np.sum(labels == -1))

    for point, label in zip(X, labels):
        group = "шум" if label == -1 else "кластер {}".format(label)
        print("{} -> {}".format(point, group))


if __name__ == "__main__":
    main()

