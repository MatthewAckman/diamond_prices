"""
Various functions used in diamond price project
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold


def tt_tr_split(df, test_ratio = 0.25):
    """
    Train test split function for use with Pandas df, assumed exog and endos vars remain
    in same data structure.
    :param dataframe: Pandas df, contains all vars
    :return: train df, test df
    """
    df = df.sample(frac=1)  # this shuffles dataframe

    test_size = int(len(df) * test_ratio)

    return df.iloc[test_size:], df.iloc[:test_size]


def cross_validate(data, model, exog, endog, kfold=None):
    """
    :param data: Pandas df, contains all data
    :param exog: Exog vars
    :param endog: Endog var
    :param kfold: k-Fold object from sklearn
    :param model: model
    :return: Mean absolute error (mae) and std of mae
    """

    if not kfold:
        kfold = KFold(n_splits=5, shuffle=True, random_state=1996)

    errors = []

    for tr, tt in kfold.split(data):
        xtr, ytr = data.iloc[tr][exog], data.iloc[tr][endog]
        xtt, ytt = data.iloc[tt][exog], data.iloc[tt][endog]
        model.fit(xtr, ytr)

        errors += [(model.predict(xtt) - ytt)**2]

    mse = round(float(np.mean(errors)), 2)

    stdev_err = round(float(np.std(errors)), 2)

    print(f'Cross-validation results: {mse = }, {stdev_err = }')

