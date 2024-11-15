from linear_regressions import Regression, mse
import numpy as np
from utilities import read_data
import requests

spots = requests.get(
    'http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv', stream=True)
data = read_data(spots)
data = np.array([list(row) for row in data])


def a():
    n_runs = 20
    train_mse = 0
    test_mse = 0
    for _ in range(n_runs):
        data_size = len(data)
        indices = np.arange(data_size)
        np.random.shuffle(indices)  # Random indexing

        # Data disaggregated by 2/3 and 1/3
        train_size = int(data_size * 2 / 3)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        X = np.ones(len(data))
        Y = data[:, -1]   # last column
        X_train, Y_train = X[train_indices].reshape(
            -1, 1), Y[train_indices].reshape(-1, 1)
        X_test, Y_test = X[test_indices].reshape(
            -1, 1), Y[test_indices].reshape(-1, 1)

        model = Regression()
        model.train(X_train, Y_train)
        predict_train = model.predict(X_train)
        predict_test = model.predict(X_test)
        train_mse += mse(Y_train, predict_train)
        test_mse += mse(Y_test, predict_test)
        print((model.W, sum(Y_train) / len(Y_train)))

    train_mse = train_mse / n_runs
    test_mse = test_mse / n_runs
    return train_mse, test_mse


def c():
    n_atrr = 12
    train_mse_for_atrrs = []
    test_mse_for_attrs = []
    for i in range(n_atrr):
        n_runs = 20
        train_mse = 0
        test_mse = 0
        for _ in range(n_runs):
            data_size = len(data)
            indices = np.arange(data_size)
            np.random.shuffle(indices)  # Random indexing

            # Data disaggregated by 2/3 and 1/3
            train_size = int(data_size * 2 / 3)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            train_bias = np.ones(train_size).reshape(-1, 1)
            test_bias = np.ones(data_size - train_size).reshape(-1, 1)
            X = data[:, i]
            Y = data[:, -1]   # last column
            X_train, Y_train = np.concatenate((X[train_indices].reshape(
                -1, 1), train_bias), axis=1), Y[train_indices].reshape(-1, 1)
            X_test, Y_test = np.concatenate((X[test_indices].reshape(
                -1, 1), test_bias), axis=1), Y[test_indices].reshape(-1, 1)

            model = Regression()
            model.train(X_train, Y_train)
            predict_train = model.predict(X_train)
            predict_test = model.predict(X_test)
            train_mse += mse(Y_train, predict_train)
            test_mse += mse(Y_test, predict_test)

        train_mse = train_mse / n_runs
        test_mse = test_mse / n_runs
        train_mse_for_atrrs.append(train_mse)
        test_mse_for_attrs.append(test_mse)
    return np.array(train_mse_for_atrrs), np.array(test_mse_for_attrs)


def d():
    n_runs = 20
    train_mse = 0
    test_mse = 0
    for _ in range(n_runs):
        data_size = len(data)
        indices = np.arange(data_size)
        np.random.shuffle(indices)  # Random indexing

        # Data disaggregated by 2/3 and 1/3
        train_size = int(data_size * 2 / 3)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_bias = np.ones(train_size).reshape(-1, 1)
        test_bias = np.ones(data_size - train_size).reshape(-1, 1)
        X = data[:, :-1]
        Y = data[:, -1]   # last column
        X_train, Y_train = np.concatenate(
            (X[train_indices], train_bias), axis=1), Y[train_indices].reshape(-1, 1)
        X_test, Y_test = np.concatenate(
            (X[test_indices], test_bias), axis=1), Y[test_indices].reshape(-1, 1)

        model = Regression()
        model.train(X_train, Y_train)
        predict_train = model.predict(X_train)
        predict_test = model.predict(X_test)
        train_mse += mse(Y_train, predict_train)
        test_mse += mse(Y_test, predict_test)

    train_mse = train_mse / n_runs
    test_mse = test_mse / n_runs
    return train_mse, test_mse


if __name__ == "__main__":
    print(a())
    print(c())
    print(d())
