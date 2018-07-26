import numpy as np
import pandas as pd
import utils
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.sparse import hstack


def header():
    return 'WEEK 4: Linear Regression and PCA'


def run():

    #linear_regression()
    pca()

    return

def linear_regression():
    # training

    df_train = pd.read_csv(utils.PATH.MATERIALS_FILE('salary-train.csv'))
    df_test = pd.read_csv(utils.PATH.MATERIALS_FILE('salary-test-mini.csv'))
    df_full = pd.concat([df_train, df_test])
    df_full.fillna('nan', inplace=True)

    str_features = ['FullDescription', 'LocationNormalized', 'ContractTime']
    for feature_name in str_features:
        df_full[feature_name] = df_full[feature_name].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)

    print(df_full.head())

    df_train = df_full.iloc[:-2]
    df_test  = df_full.iloc[-2:]

    vect = TfidfVectorizer(min_df=5)
    full_descr_tfidf = vect.fit_transform(df_train['FullDescription'])

    enc = DictVectorizer()
    categs = enc.fit_transform(df_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

    X_train = hstack([full_descr_tfidf, categs])
    y_train = df_train['SalaryNormalized']

    lr = Ridge(alpha=1, random_state=241)
    lr.fit(X_train, y_train)

    # testing

    full_descr_tfidf_test = vect.transform(df_test['FullDescription'])
    categs_test = enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
    X_test = hstack([full_descr_tfidf_test, categs_test])

    y_pred = lr.predict(X_test)
    print(y_pred)

    return


def pca():
    df = pd.read_csv(utils.PATH.MATERIALS_FILE('close_prices.csv'))
    df.drop(['date'], axis=1, inplace=True)
    print(df.head())

    pca = PCA(n_components=10)
    pca.fit(df)

    print(pca.explained_variance_ratio_)
    print(np.cumsum(pca.explained_variance_ratio_))

    df_pca = pca.transform(df)
    main_comp = df_pca[:, 0]

    df_dowj = pd.read_csv(utils.PATH.MATERIALS_FILE('djia_index.csv'))
    print(df_dowj.head())

    dowj = df_dowj['^DJI'].values

    corr = np.corrcoef(main_comp, dowj)
    print(corr)

    print(pca.components_[0, :].argmax())

    return