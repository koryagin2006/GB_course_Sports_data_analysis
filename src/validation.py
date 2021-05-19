from IPython.core.display import display
from typing import List, Tuple

import numpy as np
import pandas as pd


def create_bootstrap_samples(data: np.array, n_samples: int = 1000) -> np.array:
    """
    Создание бутстреп-выборок.

    :param data: Исходная выборка, которая будет использоваться для создания бутстреп выборок.
    :param n_samples: Количество создаваемых бутстреп выборок. Опциональный параметр, по умолчанию, равен 1000.
    :return: Матрица индексов, для создания бутстреп выборок.
    """
    bootstrap_idx = np.random.randint(low=0, high=len(data), size=(n_samples, len(data)))
    return bootstrap_idx


def create_bootstrap_metrics(y_true: np.array, y_pred: np.array,
                             metric: callable, n_samples: int = 1000) -> List[float]:
    """
    Вычисление бутстреп оценок.

    :param y_true: Вектор целевой переменной.
    :param y_pred: Вектор прогнозов.
    :param metric: Функция для вычисления метрики. Функция должна принимать 2 аргумента: y_true, y_pred.
    :param n_samples: Количество создаваемых бутстреп выборок. Опциональный параметр, по умолчанию, равен 1000.
    :return: Список со значениями метрики качества на каждой бустреп выборке.
    """
    scores = []

    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    bootstrap_idx = create_bootstrap_samples(y_true, n_samples=n_samples)

    for idx in bootstrap_idx:
        y_true_bootstrap, y_pred_bootstrap = y_true[idx], y_pred[idx]
        score = metric(y_true_bootstrap, y_pred_bootstrap)
        scores.append(score)

    return scores


def calculate_confidence_interval(scores: list, conf_interval: float = 0.95) -> Tuple[float, float]:
    """
    Вычисление доверительного интервала.

    :param scores: Список с оценками изучаемой величины.
    :param conf_interval: Уровень доверия для построения интервала. Опциональный параметр, по умолчанию, равен 0.95.
    :return: Кортеж с границами доверительного интервала.
    """
    left_bound = np.percentile(scores, ((1 - conf_interval) / 2) * 100)
    right_bound = np.percentile(scores, (conf_interval + ((1 - conf_interval) / 2)) * 100)

    return left_bound, right_bound


def make_cross_validation(X: pd.DataFrame, y: pd.Series, estimator: object, metric: callable, cv_strategy):
    """
    Кросс-валидация.

    :param X: Матрица признаков.
    :param y: Вектор целевой переменной.
    :param estimator: Объект модели для обучения.
    :param metric: Метрика для оценки качества решения.
                   Ожидается, что на вход будет передана функция, которая принимает 2 аргумента: y_true, y_pred.
    :param cv_strategy: Объект для описания стратегии кросс-валидации.
                        Ожидается, что на вход будет передан объект типа KFold или StratifiedKFold.
    :return:
        oof_score: Значение метрики качества на OOF-прогнозах.
        fold_train_scores: Значение метрики качества на каждом обучающем датасете кросс-валидации.
        fold_valid_scores: Значение метрики качества на каждом валидационном датасете кросс-валидации.
        oof_predictions: Прогнозы на OOF.
    """
    estimators, fold_train_scores, fold_valid_scores = [], [], []
    oof_predictions = np.zeros(X.shape[0])
    results = []

    for fold_number, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]

        estimator.fit(x_train, y_train)
        y_train_pred = estimator.predict(x_train)
        y_valid_pred = estimator.predict(x_valid)

        fold_train_scores.append(metric(y_train, y_train_pred))
        fold_valid_scores.append(metric(y_valid, y_valid_pred))
        oof_predictions[valid_idx] = y_valid_pred

        print(f"Fold: {fold_number + 1}", end='| ')
        results.append([fold_number + 1, len(train_idx), len(valid_idx),
                        round(fold_train_scores[fold_number], 4),
                        round(fold_valid_scores[fold_number], 4)])
        estimators.append(estimator)

    oof_score = metric(y, oof_predictions)

    display(pd.DataFrame(results, columns=['Fold', 'train-observations', 'valid-observations',
                                           'train-score', 'valid-score']))

    print(f"CV-results train: {round(np.mean(fold_train_scores), 4)} +/- {round(np.std(fold_train_scores), 3)}")
    print(f"CV-results valid: {round(np.mean(fold_valid_scores), 4)} +/- {round(np.std(fold_valid_scores), 3)}")
    print(f"OOF-score = {round(oof_score, 4)}")

    return estimators, oof_score, fold_train_scores, fold_valid_scores, oof_predictions, results


def make_modify_cross_validation(X: pd.DataFrame, y: pd.Series, estimator: object, metric: callable,
                                 cv_strategy, error_to_be_outlier: None):
    """
    Кросс-валидация с учетом выбросов.

    :param X: Матрица признаков.
    :param y: Вектор целевой переменной.
    :param estimator: Объект модели для обучения.
    :param metric: Метрика для оценки качества решения.
                   Ожидается, что на вход будет передана функция, которая принимает 2 аргумента: y_true, y_pred.
    :param cv_strategy: Объект для описания стратегии кросс-валидации.
                        Ожидается, что на вход будет передан объект типа KFold или StratifiedKFold.
    :param error_to_be_outlier: Максимальная относительная величина ошибки для того, чтобы объект считать выбросом и не
                                учитывать в итоговой ошибке алгоритма.
                                Опциональный параметр, по умолчанию не используется.
    :return:
        oof_score: Значение метрики качества на OOF-прогнозах.
        fold_train_scores: Значение метрики качества на каждом обучающем датасете кросс-валидации.
        fold_valid_scores: Значение метрики качества на каждом валидационном датасете кросс-валидации.
        oof_predictions: Прогнозы на OOF.
    """
    estimators, fold_train_scores, fold_valid_scores = [], [], []
    oof_predictions = np.zeros(X.shape[0])

    for fold_number, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]

        estimator.fit(x_train, y_train)
        y_train_pred, y_valid_pred = estimator.predict(x_train), estimator.predict(x_valid)

        fold_train_scores.append(metric(y_train, y_train_pred))
        if not error_to_be_outlier:
            fold_valid_scores.append(metric(y_valid, y_valid_pred))
        else:
            mask = ((y_valid - y_valid_pred) / y_valid) < error_to_be_outlier
            fold_valid_scores.append(metric(y_valid.loc[mask], y_valid_pred[mask]))
        oof_predictions[valid_idx] = y_valid_pred

        msg = (f"Fold: {fold_number + 1}, train-observations = {len(train_idx)}, "
               f"valid-observations = {len(valid_idx)}\n"
               f"train-score = {round(fold_train_scores[fold_number], 4)}, "
               f"valid-score = {round(fold_valid_scores[fold_number], 4)}")
        print(msg)
        print("=" * 69)
        estimators.append(estimator)

    if not error_to_be_outlier:
        oof_score = metric(y, oof_predictions)
    else:
        mask = ((y - oof_predictions) / y) < error_to_be_outlier
        oof_score = metric(y.loc[mask], oof_predictions[mask])

    print(f"CV-results train: {round(np.mean(fold_train_scores), 4)} +/- {round(np.std(fold_train_scores), 3)}")
    print(f"CV-results valid: {round(np.mean(fold_valid_scores), 4)} +/- {round(np.std(fold_valid_scores), 3)}")
    print(f"OOF-score = {round(oof_score, 4)}")

    return estimators, oof_score, fold_train_scores, fold_valid_scores, oof_predictions
