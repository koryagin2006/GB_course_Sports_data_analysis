from typing import List, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


def linear_model_fit_transform(X: np.array, y: np.array) -> np.array:
    """
    Обучение и применение линейной модели к набору данных.

    :param X: Матрица признаков.
    :param y: Вектор целевой переменной.
    :return: Вектор прогнозов.
    """
    model = LinearRegression()
    model.fit(X.reshape(X.shape[0], -1), y)
    y_pred = model.coef_ * X + model.intercept_

    return y_pred


def plot_kde_target(feature_name: str, data: pd.DataFrame):
    """
    Визуализация функции распределения признаков в зависимости от значения целевой переменной на обучающей выборке.
    Вывод коэффициента корреляции между значением признака и значением целевой переменной, вывод медианы
    значений признака в разрезе целевой переменной.

    :param feature_name: Название анализируемого признака.
    :param data: Матрица признаков для обучения.
    :return: График
    """
    corr = data["TARGET"].corr(data[feature_name])

    mask = data["TARGET"] == 1
    avg_target = data.loc[mask, feature_name].median()
    avg_non_target = data.loc[~mask, feature_name].median()

    fig = plt.figure(figsize=(12, 6))
    plt.title(f"{feature_name} Distribution", size=14)
    sns.kdeplot(data.loc[mask, feature_name], linewidth=3, color="blue", label="TARGET = 1")
    sns.kdeplot(data.loc[~mask, feature_name], linewidth=3, color="green", label="TARGET = 0")
    plt.legend(loc="best", fontsize=14)
    plt.xlabel(feature_name, size=14)
    plt.ylabel("Density", size=14)

    print(f"The correlation between {feature_name} and target = {round(corr, 4)}")
    print(f"Median-value for default-loan = {round(avg_target, 4)}")
    print(f"Median-value for non default-loan = {round(avg_target, 4)}")


def calculate_feature_separating_ability(features: pd.DataFrame,
                                         target: pd.Series,
                                         fill_value: float = -9999) -> pd.DataFrame:
    """
    Оценка разделяющей способности признаков с помощью метрики GINI.

    :param features: Матрица признаков.
    :param target: Вектор целевой переменной.
    :param fill_value: Значение для заполнения пропусков в значении признаков.
                       Опциональный параметр, по умолчанию, равен -9999;
    :return: Матрица важности признаков.
    """
    scores = {}
    for feature in features:
        score = roc_auc_score(target, features[feature].fillna(fill_value))
        scores[feature] = 2 * score - 1
    scores = pd.Series(scores)
    scores = scores.sort_values(ascending=False)
    return scores


def create_numerical_aggs(data: pd.DataFrame,
                          groupby_id: str,
                          aggs: dict,
                          prefix: Optional[str] = None,
                          suffix: Optional[str] = None,
                          ) -> pd.DataFrame:
    """
    Построение агрегаций для числовых признаков.

    :param data: Выборка для построения агрегаций.
    :param groupby_id: Название ключа, по которому нужно произвести группировку.
    :param aggs: Словарь с названием признака и списка функций.
                Ключ словаря - название признака, который используется для вычисления агрегаций,
                значение словаря - список с названием функций для вычисления агрегаций.
    :param prefix: Префикс для названия признаков. Опциональный параметр, по умолчанию, не используется.
    :param suffix: Суффикс для названия признаков. Опциональный параметр, по умолчанию, не используется.
    :return: Выборка с рассчитанными агрегациями.
    """
    if not prefix:
        prefix = ""
    if not suffix:
        suffix = ""
    stats = data \
        .groupby(groupby_id) \
        .agg(aggs)
    stats.columns = [f"{prefix}{feature}_{stat}{suffix}".upper() for feature, stat in stats]
    stats = stats.reset_index()

    return stats


def create_categorical_aggs(data: pd.DataFrame,
                            groupby_id: str,
                            features: List[str],
                            prefix: Optional[str] = None,
                            suffix: Optional[str] = None,
                            ) -> pd.DataFrame:
    """
    Построение агрегаций для категориальных признаков. Для категориальных признако считаются счетчики для
    каждого значения категории и среднее значение счетчика для каждого значения категории.

    :param data: Выборка для построения агрегаций.
    :param groupby_id: Название ключа, по которому нужно произвести группировку.
    :param features: Список с названием признаков, для которых произвести группировку.
    :param prefix: Префикс для названия признаков. Опциональный параметр, по умолчанию, не используется.
    :param suffix: Суффикс для названия признаков. Опциональный параметр, по умолчанию, не используется.
    :return: Выборка с рассчитанными агрегациями.
    """
    if not prefix:
        prefix = ""
    if not suffix:
        suffix = ""

    categorical = pd.get_dummies(data[features])
    columns_to_agg = categorical.columns

    categorical[groupby_id] = data[groupby_id]
    stats = categorical \
        .groupby(groupby_id) \
        .agg({col: ["mean", "sum"] for col in columns_to_agg})
    stats.columns = [f"{prefix}{feature}_{stat}{suffix}".upper() for feature, stat in stats]
    stats.columns = [col.replace("MEAN", "RATIO") for col in stats.columns]
    stats.columns = [col.replace("SUM", "TOTAL") for col in stats.columns]
    stats = stats.reset_index()

    return stats
