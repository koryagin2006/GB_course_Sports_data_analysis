import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import missingno as msno
import os
import warnings
from copy import deepcopy
from tqdm import tqdm
from typing import List

import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

warnings.simplefilter("ignore")
shap.initjs()


def check_missings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция для вычисления среднего и общего числа пропусков.

    :param df: Набор данных для вычисления статистики.
    :return: Датафрейм со статистикой распределения пропусков.
    """
    na = df.isnull().sum()
    result = pd.DataFrame({"Total": na,
                           "Percent": 100 * na / df.shape[0],
                           "Types": df.dtypes
                           })
    result = result[result["Total"] != 0]
    print(f"Total NA-values = {na.sum()}")
    return result.T


def _predict(estimator, x_valid):
    if hasattr(estimator, "predict_proba"):
        y_pred = estimator.predict_proba(x_valid)[:, 1]
    else:
        y_pred = estimator.predict(x_valid)
    return y_pred


def calculate_permutation_importance(estimator,
                                     metric: callable,
                                     x_valid: pd.DataFrame,
                                     y_valid: pd.DataFrame,
                                     maximize: bool = True
                                     ) -> pd.Series:
    """
    Вычисление важности признаков на основе перестановочного критерия (permutation importance).

    :param estimator: Модель машинного обучения, выполненная в sklearn-API.
                    Модель должны быть обученной (применен метод `fit`).
    :param metric: Функция для оценки качества прогнозов, функция принимает 2 аргумента:
                    вектор истинных ответов и вектор прогнозов.
    :param x_valid: Матрица признаков для оценки качества модели.
    :param y_valid: Вектор целевой переменной для оценки качества модели.
    :param maximize: Флаг максимизации метрики качества. Опциональный параметр, по умолчанию, равен `True`.
                    Если `True`, значит чем выше значение метрики качества, тем лучше. Если `False` - иначе.
    :return:
    """
    y_pred = _predict(estimator, x_valid)
    base_score = metric(y_valid, y_pred)
    scores, delta = {}, {}

    for feature in x_valid.columns:
        x_valid_ = x_valid.copy(deep=True)
        x_valid_[feature] = np.random.permutation(x_valid_[feature])
        y_pred = _predict(estimator, x_valid_)
        feature_score = metric(y_valid, y_pred)

        if maximize:
            delta[feature] = base_score - feature_score
        else:
            delta[feature] = feature_score - base_score

        scores[feature] = feature_score

    scores, delta = pd.Series(scores), pd.Series(delta)
    scores = scores.sort_values(ascending=False)
    delta = delta.sort_values(ascending=False)

    return scores, delta


class TargetEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, alpha: float = 0, folds: int = 5):
        self.folds = folds
        self.alpha = alpha
        self.features = None
        self.cv = None

    def fit(self, X, y=None):
        self.features = {}
        self.cv = KFold(n_splits=self.folds, shuffle=True, random_state=27)
        global_mean = np.mean(y)

        for fold_number, (train_idx, valid_idx) in enumerate(self.cv.split(X, y), start=1):
            x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
            y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]

            data = pd.DataFrame({"feature": x_train, "target": y_train})
            data = data.groupby(["feature"])["target"].agg([np.mean, np.size])
            data = data.reset_index()
            score = data["mean"] * data["size"] + global_mean * self.alpha
            score = score / (data["size"] + self.alpha)

            self.features[f"fold_{fold_number}"] = {key: value for key, value in zip(data["feature"], score)}

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "features")
        # TBD

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        x_transformed = X.copy(deep=True)

        for fold_number, (train_idx, valid_idx) in enumerate(self.cv.split(X, y), start=1):
            x_transformed.loc[valid_idx] = x_transformed.loc[valid_idx].map(self.features[f"fold_{fold_number}"])
        return x_transformed
