import numpy as np
import pandas as pd
import utils
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

def header():
    return 'WEEK 1: Intro and Logical methods'


def run():

    #numpy_check()
    #pandas_check()
    decision_tree()

    return


def numpy_check():

    X = np.random.normal(1, 10, size=(1000,50))
    print(X)

    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    Y = (X - mean)/std
    print(Y)

    Z = np.array([[1,0,3,4],
                  [2,3,1,7],
                  [1,0,9,1]])
    S = Z.sum(axis=1)
    U = np.nonzero(S > 10)
    print(U)

    E1 = np.eye(3)
    E2 = np.eye(3)
    E  = np.vstack([E1, E2])
    print(E)

    return


def pandas_check():

    df = pd.read_csv(utils.PATH.MATERIALS_FILE('titanic.csv'))
    print(df.head())
    print(df.info())

    print('+++++++++++++++++++++++++++++++++++++++++++++')

    sex_vc = df['Sex'].value_counts()
    print(sex_vc)

    surv_vc = 100*df['Survived'].value_counts()/len(df)
    print(surv_vc)

    pclass_vc = 100*df['Pclass'].value_counts()/len(df)
    print(pclass_vc)

    mean_age = df['Age'].mean()
    std_age  = df['Age'].std()
    med_age  = df['Age'].median()
    print(mean_age, std_age, med_age)

    corr = df['SibSp'].corr(df['Parch'])
    print(corr)

    women_names = df[df['Sex'] == 'female']['Name']

    def extract_name(str):
        result = str
        markers = ['(', 'Miss. ', 'Mrs. ', 'Mme. ', 'Mlle. ']
        for marker in markers:
            idx = str.index(marker) if marker in str else None
            if idx:
                result = str[idx+len(marker):].split()[0]
                break
        result = result.replace('"', '')
        result = result.replace(')', '')

        return result

    names = list()
    for full_name in women_names:
        name = extract_name(full_name)
        names.append(name)

    print(names)

    diff_names = Counter(names)
    print(diff_names)


    return


def decision_tree():
    df = pd.read_csv(utils.PATH.MATERIALS_FILE('titanic.csv'))
    print('loaded: ', len(df))


    feature_names = ['Pclass', 'Fare', 'Age', 'Sex']
    target_name = ['Survived']
    df = df[feature_names + target_name]
    df.replace({'Sex': { 'male': 0, 'female': 1 }}, inplace=True)
    df.dropna(inplace=True)

    print(df.head())

    X = df[feature_names]
    y = df[target_name]
    print(y.head())

    tree = DecisionTreeClassifier(random_state=241)
    tree.fit(X, y)

    importances = tree.feature_importances_
    print(importances)

    return