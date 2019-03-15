#removes forced warnings printed
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd
import math
import matplotlib.pyplot as plt


def get_training_data():
    file_to_use = "ENB2012_data.csv"

    full_file = pd.read_csv(file_to_use)
    column_count = len(full_file.columns)

    training_data = pd.read_csv(file_to_use, usecols=[0, 1, 2, 3, 4, 6, 7], skiprows=1, header=None, index_col=None)

    print("Features:")
    print(training_data)

    results = pd.read_csv(file_to_use, usecols=[column_count-2, column_count-1], skiprows=1, header=None,
                          index_col=None)
    print("Results:")
    print(results)

    return training_data, results


def visualise_data(X, y):
    plt.figure(figsize=(20, 10))
    index = 1
    for result in y.columns:
        for column in X.columns:
            plt.subplot(len(y.columns), len(X.columns), index)
            plt.scatter(X[column], y[result], color='black')
            index = index + 1

    plt.show()


def get_lr_classifier(X, y):
    clf = LinearRegression()
    clf.fit(X, y)
    return clf


def use_poly_pipeline(X, y):
    steps = [
        ('scalar', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', LinearRegression())
    ]

    pipeline = Pipeline(steps)

    return pipeline.fit(X, y)


def use_ridge_pipeline(X, y):
    steps = [
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', Ridge(alpha=0.001, fit_intercept=True))
    ]

    pipeline = Pipeline(steps)

    return pipeline.fit(X, y)


def use_lasso_pipeline(X, y):
    steps = [
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', Lasso(alpha=0.1, fit_intercept=True))
    ]

    pipeline = Pipeline(steps)

    return pipeline.fit(X, y)


def use_en_pipeline(X, y):
    steps = [
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', ElasticNet(alpha=0.001, l1_ratio=0.99, fit_intercept=True))
    ]

    pipeline = Pipeline(steps)

    return pipeline.fit(X, y)


def score_classifier(clf, X, y):
    y_pred = clf.predict(X)
    r2 = r2_score(y, y_pred)
    print("R2 Score: {}".format(r2))
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    print("RMSE: {}".format(rmse))


if __name__ == "__main__":
    X, y = get_training_data()
    visualise_data(X, y)

    print("=============================")
    print("Training and scoring entire data set")
    full_clf = get_lr_classifier(X, y)
    score_classifier(full_clf, X, y)
    print("=============================")

    print("Training on random 70%, validating on other 30%")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=301, test_size=0.3)
    partial_clf = get_lr_classifier(X_train, y_train)
    print("--Training score--")
    score_classifier(partial_clf, X_train, y_train)
    print("--Testing score--")
    score_classifier(partial_clf, X_test, y_test)
    print("=============================")

    print("Now using scaling and polynomial features")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=301, test_size=0.3)
    poly_clf = use_poly_pipeline(X_train, y_train)
    print("--Training score--")
    score_classifier(poly_clf, X_train, y_train)
    print("--Testing score--")
    score_classifier(poly_clf, X_test, y_test)
    print("=============================")

    print("Using Ridge")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=301, test_size=0.3)
    ridge_clf = use_ridge_pipeline(X_train, y_train)
    print("--Training score--")
    score_classifier(ridge_clf, X_train, y_train)
    print("--Testing score--")
    score_classifier(ridge_clf, X_test, y_test)
    print("=============================")

    print("Using Lasso")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=301, test_size=0.3)
    lasso_clf = use_lasso_pipeline(X_train, y_train)
    print("--Training score--")
    score_classifier(lasso_clf, X_train, y_train)
    print("--Testing score--")
    score_classifier(lasso_clf, X_test, y_test)
    print("=============================")

    print("Using Elastic Net")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=301, test_size=0.3)
    en_clf = use_en_pipeline(X_train, y_train)
    print("--Training score--")
    score_classifier(en_clf, X_train, y_train)
    print("--Testing score--")
    score_classifier(en_clf, X_test, y_test)
    print("=============================")


