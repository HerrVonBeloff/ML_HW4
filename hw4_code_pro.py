import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector).squeeze()
    feature_vector_sorted = feature_vector[sorted_indices]
    target_vector_sorted = target_vector[sorted_indices]
    
    thresholds = (feature_vector_sorted[1:] + feature_vector_sorted[:-1]) / 2
    
    cumsum_left = np.cumsum(target_vector_sorted[:-1])
    cumsum_right = np.sum(target_vector_sorted) - cumsum_left
    
    n_left = np.arange(1, len(target_vector_sorted))
    n_right = len(target_vector_sorted) - n_left

    p1_left = cumsum_left / n_left
    p0_left = 1 - p1_left
    p1_right = cumsum_right / n_right
    p0_right = 1 - p1_right
    
    # Вычисляем критерий Джини для левой и правой подвыборок
    H_left = 1 - p1_left**2 - p0_left**2
    H_right = 1 - p1_right**2 - p0_right**2

    # Вычисляем общий критерий Джини
    ginis = -(n_left / len(target_vector_sorted)) * H_left - (n_right / len(target_vector_sorted)) * H_right
    
    valid_indices = (n_left > 0) & (n_right > 0)
    thresholds = thresholds[valid_indices]
    ginis = ginis[valid_indices]
    
    # Находим оптимальный порог и соответствующее значение Джини
    if len(ginis) > 0:
        best_index = np.argmax(ginis)
        threshold_best = thresholds[best_index]
        gini_best = ginis[best_index]
    else:
        threshold_best = None
        gini_best = None
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        
        # Критерии остановы
        sub_X = np.array(sub_X)
        sub_y = np.array(sub_y)
        if len(sub_y) == 0:  # Если подвыборка пуста
            node["type"] = "terminal"
            # node["class"] = Counter(sub_y).most_common(1)[0][0] if len(sub_y) > 0 else 0  
            node["class"] = 0 # По умолчанию 0
            return

        if np.all(sub_y == sub_y[0]):  # Если все объекты одного класса
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth: # Критерий останова: достигнута максимальная глубина
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:  # Минимальное количество объектов для разбиения
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                feature_vector = sub_X[:, feature]
                categories_map = {cat: idx for idx, cat in enumerate(sub_X[:, feature])}
            else:
                raise ValueError("Unknown feature type")

            # Находим лучшее разбиение
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini is None:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]
                else:
                    raise ValueError("Unknown feature type")

        # Критерий останова: нельзя улучшить Джини
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self._min_samples_leaf is not None and (np.sum(split) < self._min_samples_leaf or np.sum(~split) < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
    
        # Создаем внутренний узел
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError("Unknown feature type")

        # Рекурсивно строим левое и правое поддеревья
        node["left_child"], node["right_child"] = {}, {}
        if np.sum(split) > 0:  # Проверка на пустую подвыборку
            self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        else:
            node["left_child"]["type"] = "terminal"
            node["left_child"]["class"] = Counter(sub_y).most_common(1)[0][0]

        if np.sum(~split) > 0:  # Проверка на пустую подвыборку
            self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)
        else:
            node["right_child"]["type"] = "terminal"
            node["right_child"]["class"] = Counter(sub_y).most_common(1)[0][0]

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":

            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        X = np.array(X)
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)