import datetime as dt
import numpy as np
import pandas as pd
import utils
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def run():

    X_train, X_test, y_train = prepare_data()
    first_approach(X_train, X_test, y_train)
    second_approach(X_train, X_test, y_train)

    return

    
def prepare_data():
    train_df = pd.read_csv('F:/Work/My/Python/courses/courserra/ml-intro/materials/final/features.csv'), index_col='match_id')
    test_df  = pd.read_csv('F:/Work/My/Python/courses/courserra/ml-intro/materials/final/features_test.csv'), index_col='match_id')
    results_df = train_df[['duration',
                           'radiant_win',
                           'tower_status_radiant',
                           'tower_status_dire',
                           'barracks_status_radiant',
                           'barracks_status_dire']]
    train_df = train_df.drop(results_df.columns.values, axis=1)
    full_df = pd.concat([train_df, test_df], axis=0)

    print(train_df.shape)
    print(test_df.shape)
    print(full_df.shape)

    nan_cols = train_df.columns[train_df.isna().any()].tolist()
    print(nan_cols)

    full_df.fillna(0, inplace=True)

    # heroes
    hero_features = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                     'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
    heroes_bag = set()
    for feat_name in hero_features:
        heroes_bag |= set(full_df[feat_name].unique())
    heroes_feat_ids = {}
    for i, id in enumerate(heroes_bag):
        heroes_feat_ids[id] = i

    heroes_data = np.zeros([full_df.shape[0], len(heroes_bag)])
    print(len(heroes_bag))

    for i, match_id in enumerate(full_df.index):
        for p in range(1, 6):
            r_idx = heroes_feat_ids[full_df.ix[match_id, 'r%d_hero'%p]]
            d_idx = heroes_feat_ids[full_df.ix[match_id, 'd%d_hero'%p]]
            heroes_data[i, r_idx] = 1
            heroes_data[i, d_idx] = -1

    heroes_df = pd.DataFrame(data = heroes_data,
                             index=full_df.index,
                             columns=heroes_feat_ids.keys())
    full_df = pd.concat([full_df, heroes_df], axis=1)

    categ_features = ['lobby_type'] + hero_features
    full_df = full_df.drop(categ_features, axis=1)

    target = results_df['radiant_win'].values
    # целевая переменная - radiant_win

    # data sets
    X_train = full_df.iloc[:len(train_df), :]
    X_test  = full_df.iloc[len(train_df):, :]

    return X_train, X_test, target


def first_approach(X_train, X_test, y_train):
    for n_estimators in [10, 20, 30]:
        kf = KFold(n_splits=5, shuffle=True, random_state=84)
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.1, random_state=84)
        scores = []
        start = dt.datetime.now()
        for tr_ids, vl_ids in kf.split(X_train):
            X_tr, X_vl = X_train[tr_ids], X_train[vl_ids]
            y_tr, y_vl = y_train[tr_ids], y_train[vl_ids]
            clf.fit(X_tr, y_tr)
            y_pr = clf.predict_proba(X_vl)[:, 1]
            score = roc_auc_score(y_vl, y_pr)
            scores.append(score)
        score = np.mean(scores)
        end = dt.datetime.now()
        print(n_estimators, '\t', score, '\telapsed:', end-start)
    return


def second_approach(X_train, X_test, y_train):

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)

    for C in [0.1]: #, 1, 10, 100, 1000]:
        kf = KFold(n_splits=5, shuffle=True, random_state=84)
        clf = LogisticRegression(penalty='l2', C=C, random_state=84, n_jobs=-1)
        scores = []
        start = dt.datetime.now()
        for tr_ids, vl_ids in kf.split(X_train):
            X_tr, X_vl = X_train[tr_ids], X_train[vl_ids]
            y_tr, y_vl = y_train[tr_ids], y_train[vl_ids]
            clf.fit(X_tr, y_tr)
            y_pr = clf.predict_proba(X_vl)[:, 1]
            score = roc_auc_score(y_vl, y_pr)
            scores.append(score)
        score = np.mean(scores)
        end = dt.datetime.now()
        print(C, '\t', score, '\telapsed:', end-start)

    y_pred = clf.predict_proba(X_test)[:, 1]
    min = y_pred.min()
    max = y_pred.max()

    return
    
run()