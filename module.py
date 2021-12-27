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


