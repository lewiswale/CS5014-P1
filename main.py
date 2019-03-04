from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def get_training_data():
    file_to_use = "ENB2012_data.csv"

    full_file = pd.read_csv(file_to_use)
    column_count = len(full_file.columns)

    x_cols = [i for i in range(0, column_count-2)]
    training_data = pd.read_csv(file_to_use, usecols=[0, 1, 2, 3, 4, 6], skiprows=1, header=None, index_col=None)

    print("Features:")
    print(training_data)

    results = pd.read_csv(file_to_use, usecols=[column_count-2, column_count-1], skiprows=1, header=None,
                          index_col=None)
    print("Results:")
    print(results)

    return training_data, results


def get_lr_classifier(X, y):
    clf = LinearRegression()
    clf.fit(X, y)
    print(clf)
    return clf


def get_rf_classifier(X, y):
    clf = RandomForestRegressor()
    clf.fit(X, y)
    print(clf)
    return clf


def make_prediction(clf, data):
    prediction = clf.predict(data)
    print(prediction)


def score_classifier(clf, X, y):
    print(clf.score(X, y))


if __name__ == "__main__":
    X, y = get_training_data()
    clf1 = get_lr_classifier(X, y)
    score_classifier(clf1, X, y)

    clf2 = get_rf_classifier(X, y)
    score_classifier(clf2, X, y)

