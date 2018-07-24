import numpy as np
import pandas as pd
import utils
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston

def header():
    return 'WEEK 2: Metric and Linear methods'


def run():

    #k_neighbor_selection()
    #metric_tuning()
    features_normalization()

    return


def k_neighbor_selection():

    df = pd.read_csv(utils.PATH.MATERIALS_FILE('wine.data'))
    feature_names = [ 'Alcohol',
                      'Malic_acid',
 	                  'Ash',
	                  'Alcalinity_of_ash',
 	                  'Magnesium',
	                  'Total_phenols',
 	                  'Flavanoids',
 	                  'Nonflavanoid_phenols',
 	                  'Proanthocyanins',
	                  'Color_intensity',
 	                  'Hue',
 	                  'OD280/OD315',
 	                  'Proline']
    target_name = [ 'target' ]

    df.columns = target_name + feature_names
    print(df.head())

    X = df[feature_names]
    y = df[target_name].values.reshape([len(df)])

    k_max = -1
    score_max = 0
    for k in range(1, 51):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        kn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(kn, X, y, cv=kf, scoring='accuracy').mean()
        if score>score_max:
            score_max = score
            k_max = k
        print(k, score)
    print('------')
    print(k_max, score_max)

    # scaling

    X = scale(X)
    k_max = -1
    score_max = 0
    for k in range(1, 51):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        kn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(kn, X, y, cv=kf, scoring='accuracy').mean()
        if score>score_max:
            score_max = score
            k_max = k
        print(k, score)
    print('------')
    print(k_max, score_max)


    return


def metric_tuning():
    data = load_boston()
    X = data['data']
    y = data['target']

    X = scale(X)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    p_max = 0
    score_max = -np.Infinity
    for p in np.linspace(1, 10, num=200):
        kr = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
        score = cross_val_score(kr, X, y, scoring='neg_mean_squared_error', cv=kf).mean()
        print(p, score)
        if score>score_max:
            score_max = score
            p_max = p
    print(p_max, score_max)

    return


def features_normalization():
    df_train = pd.read_csv(utils.PATH.MATERIALS_FILE('perceptron-train.csv'), header=None).values
    df_test  = pd.read_csv(utils.PATH.MATERIALS_FILE('perceptron-test.csv'), header=None).values

    X_train, X_test = df_train[:, 1:], df_test[:, 1:]
    y_train, y_test = df_train[:, 0], df_test[:, 0]

    p = Perceptron(random_state=241)
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    p.fit(X_train_scaled, y_train)
    y_pred = p.predict(X_test_scaled)

    score_scaled = accuracy_score(y_test, y_pred)

    print(score, score_scaled, score_scaled-score)


    return