import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, log_loss


def header():
    return 'WEEK 5: Compositions and Neural Nets'


def run():

    #random_forest()
    gradient_boosting()

    return

def random_forest():
    df = pd.read_csv(utils.PATH.MATERIALS_FILE('abalone.csv'))
    df['Sex'] = df.replace({ 'Sex': { 'F':-1, 'M':1, 'I':0 } })

    print(df.head())

    y = df['Rings'].values
    X = df.drop(['Rings'], axis=1).values

    for i in range(1, 51):
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        rf = RandomForestRegressor(n_estimators=i, random_state=1)
        scores = []
        for train_ids, test_ids in kf.split(X):
            X_train, X_test = X[train_ids], X[test_ids]
            y_train, y_test = y[train_ids], y[test_ids]
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            scores.append(r2_score(y_test, y_pred))
        score = np.array(scores).mean()
        print(i, score, '<---' if score>0.52 else '')

    return


def gradient_boosting():
    df = pd.read_csv(utils.PATH.MATERIALS_FILE('gbm-data.csv'), dtype=float)
    print(df.head())

    data = df.values
    X = data[:, 1:]
    y = data[:, 0].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

    rfc = RandomForestClassifier(n_estimators=36, random_state=241)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict_proba(X_test)
    score = log_loss(y_test, y_pred)
    print('RandomForestClassifier score:', score)


    for lr in [1, 0.5, 0.3, 0.2, 0.1]:
        gbc = GradientBoostingClassifier(n_estimators=250, learning_rate=lr, verbose=False, random_state=241)
        gbc.fit(X_train, y_train)

        test_score  = []
        train_score = []
        for i, y_pred in enumerate(gbc.staged_decision_function(X_test)):
            y_pred = 1/(1 + np.exp(-y_pred))
            test_score.append(log_loss(y_test, y_pred))
        for i, y_pred in enumerate(gbc.staged_decision_function(X_train)):
            y_pred = 1/(1 + np.exp(-y_pred))
            train_score.append(log_loss(y_train, y_pred))

        print(lr, '\t', 'train:\t', np.array(train_score).argmin(), np.array(train_score).min(), '\t',
                        'test: \t', np.array(test_score).argmin(), np.array(test_score).min())

        plt.plot(test_score)
        plt.plot(train_score)
        plt.legend(['test score', 'train score'])
        plt.show()


    return