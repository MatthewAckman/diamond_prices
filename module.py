"""
Various functions used in diamond price project
"""


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
        xtr, ytr = data.iloc[tr][exog], df.iloc[tr][endog]
        xtt, ytt = data.iloc[tt][exog], df.iloc[tt][endog]
        model.fit(xtr, ytr)

        errors += [abs(model.predict(xtt) - ytt)]

    mae = round(float(np.mean(errors)), 2)

    stdev_err = round(float(np.std(errors)), 2)

    print(f'Cross-validation results: {mae = }, {stdev_err = }')


def get_errors(data, model, exog, endog, kfold=None):
    if not kfold:
        kfold = KFold(n_splits=5, shuffle=True, random_state=1996)

    cv_errors = pd.DataFrame()
    tt_errors = pd.DataFrame()

    data, test_data = tt_tr_split(data)

    # yield cv errors
    for tr, tt in kfold.split(data):
        xtr, ytr = data.iloc[tr][exog], df.iloc[tr][endog]
        xtt, ytt = data.iloc[tt][exog], df.iloc[tt][endog]
        model.fit(xtr, ytr)

        cv_errors = cv_errors.append(model.predict(xtt) - ytt)

    # yield test errors
    tt_errors = tt_errors.append(model.predict(test_data[exog]) - test_data[endog])

    return cv_errors, tt_errors