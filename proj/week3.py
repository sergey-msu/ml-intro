import numpy as np
import pandas as pd
import utils
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


def header():
    return 'WEEK 3: SVM, Logictic Regression, Metrics'


def run():

    #support_objects()
    #text_analysis()
    #logistic_regression()
    metrics()

    return

def support_objects():

    df = pd.read_csv(utils.PATH.MATERIALS_FILE('svm-data.csv'), header=None).values
    X  = df[:, 1:]
    y  = df[:, 0]

    svm = SVC(C=100000, kernel='linear', random_state=241)
    svm.fit(X, y)

    print(svm.support_)

    return


def text_analysis():

    newsgroup = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
    X = newsgroup.data
    y = newsgroup.target

    vect = TfidfVectorizer()
    X = vect.fit_transform(X)

    param_grid = { 'C': np.power(10.0, np.arange(-5, 6)) }
    svm = SVC(kernel='linear', random_state=241)
    kf = KFold(n_splits=5, shuffle=True, random_state=241)
    grid = GridSearchCV(svm, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)

    print(grid.best_params_)

    svm = SVC(C=1.0, kernel='linear', random_state=241)
    svm.fit(X, y)

    feature_mapping = np.array(vect.get_feature_names())
    coeffs = np.abs(np.squeeze(np.asarray(svm.coef_.todense())))
    max_ids = np.array(coeffs).argsort()[-10:][::-1].astype(int)

    max_words = feature_mapping[max_ids]

    print(max_words)

    return


def logistic_regression():
    df = pd.read_csv(utils.PATH.MATERIALS_FILE('data-logistic.csv'), header=None).values
    X = df[:, 1:]
    y = df[:, 0]

    #X = np.array([[1, 2],
    #              [0, 3]])
    #y = np.array([1, -1])

    slr = SimpleLogisticRegression(10, 0.1, 10000)
    slr.fit(X, y)

    y_pred = slr.predict_proba(X)
    score = roc_auc_score(y, y_pred)
    print(score)

    lr = LogisticRegression(C=10, max_iter=10000)
    lr.fit(X, y)
    y_pred = lr.predict_proba(X)[:, 1]
    score = roc_auc_score(y, y_pred)
    print(score)


    return


def metrics():

    df = pd.read_csv(utils.PATH.MATERIALS_FILE('classification.csv'))
    print(df.head())

    y_true = df['true'].values
    y_pred = df['pred'].values

    TP = len(df[(df['true']==1) & (df['pred']==1)])
    FP = len(df[(df['true']==0) & (df['pred']==1)])
    FN = len(df[(df['true']==1) & (df['pred']==0)])
    TN = len(df[(df['true']==0) & (df['pred']==0)])

    print('\t Act. pos\tAct. neg')
    print('Pred. pos\t{0}\t{1}'.format(TP, FP))
    print('Pred. neg\t{0}\t{1}'.format(FN, TN))

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)

    print(accuracy, precision, recall, f1)

    df = pd.read_csv(utils.PATH.MATERIALS_FILE('scores.csv'))
    print(df.head())
    y_true = df['true'].values
    score_logreg = df['score_logreg'].values
    score_svm    = df['score_svm'].values
    score_knn    = df['score_knn'].values
    score_tree   = df['score_tree'].values

    logreg_roc_auc = roc_auc_score(y_true, score_logreg)
    svm_roc_auc    = roc_auc_score(y_true, score_svm)
    knn_roc_auc    = roc_auc_score(y_true, score_knn)
    tree_roc_auc   = roc_auc_score(y_true, score_tree)

    print(logreg_roc_auc, svm_roc_auc, knn_roc_auc, tree_roc_auc)

    logreg_rc = precision_recall_curve(y_true, score_logreg)
    logreg_rc_max = logreg_rc[0][logreg_rc[1]>=0.7].max()

    svm_rc = precision_recall_curve(y_true, score_svm)
    svm_rc_max = svm_rc[0][svm_rc[1]>=0.7].max()

    knn_rc = precision_recall_curve(y_true, score_knn)
    knn_rc_max = knn_rc[0][knn_rc[1]>=0.7].max()

    tree_rc = precision_recall_curve(y_true, score_tree)
    tree_rc_max = tree_rc[0][tree_rc[1]>=0.7].max()

    print(logreg_rc_max, svm_rc_max, knn_rc_max, tree_rc_max)

    return


class SimpleLogisticRegression(BaseEstimator):
    def __init__(self, C, eta, max_iters, delta_stab=1e-15):
        self._C = C
        self._eta = eta
        self._max_iters = max_iters
        self._delta_stab = delta_stab
        self._W = None

    def fit(self, X, y):
        n_feat = X.shape[-1]
        w = np.zeros(n_feat)

        for i in range(self._max_iters):
            # calc grads
            grads = np.zeros(n_feat)
            M = y*np.dot(X, w)
            p = -y*(1 - 1/(1 + np.exp(-M)))
            for j in range(n_feat):
                pj = p*X[:, j]
                grads[j] = np.mean(pj) + self._C*w[j]

            # do step
            dw = -self._eta*grads
            if (np.linalg.norm(dw) < self._delta_stab):
                break

            w = w + dw

        self._W = w

        return

    def predict_proba(self, X):
        if self._W is None:
            raise Exception('Fit algorithm first')

        if X.shape[-1] != len(self._W):
            raise Exception('Inconsistent number of features: {0} expected'.format(len(self._W)))

        return np.dot(X, self._W)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.sign(probs)
