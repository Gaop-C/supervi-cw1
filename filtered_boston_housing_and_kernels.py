from linear_regressions import Regression, mse
import numpy as np
from utilities import read_data
import requests

spots = requests.get(
    'http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv', stream=True)
data = read_data(spots)
data = np.array([list(row) for row in data])


def a_1_2():
    n_runs = 20
    list_train_mse = []
    list_test_mse = []
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
        list_train_mse.append(mse(Y_train, predict_train))
        list_test_mse.append(mse(Y_test, predict_test))
        # print((model.W, sum(Y_train) / len(Y_train)))

    train_mse, train_std = np.mean(list_train_mse), np.std(list_train_mse)
    test_mse, test_std = np.mean(list_test_mse), np.std(list_train_mse)
    return train_mse, train_std, test_mse, test_std


def c_1_2():
    n_atrr = 12
    train_mse_for_atrrs = []
    test_mse_for_attrs = []
    train_std_for_atrrs = []
    test_std_for_attrs = []
    for i in range(n_atrr):
        n_runs = 20
        list_train_mse = []
        list_test_mse = []
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
            list_train_mse.append(mse(Y_train, predict_train))
            list_test_mse.append(mse(Y_test, predict_test))

        train_mse, train_std = np.mean(list_train_mse), np.std(list_train_mse)
        test_mse, test_std = np.mean(list_test_mse), np.std(list_train_mse)
        train_mse_for_atrrs.append(train_mse)
        train_std_for_atrrs.append(train_std)
        test_mse_for_attrs.append(test_mse)
        test_std_for_attrs.append(test_std)
    return np.array(train_mse_for_atrrs), np.array(train_std_for_atrrs), np.array(test_mse_for_attrs), np.array(test_std_for_attrs)


def d_1_2():
    n_runs = 20
    list_train_mse = []
    list_test_mse = []
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
        list_train_mse.append(mse(Y_train, predict_train))
        list_test_mse.append(mse(Y_test, predict_test))

    train_mse, train_std = np.mean(list_train_mse), np.std(list_train_mse)
    test_mse, test_std = np.mean(list_test_mse), np.std(list_train_mse)
    return train_mse, train_std, test_mse, test_std


if __name__ == "__main__":
    print(a_1_2())
    print(c_1_2())
    print(d_1_2())
