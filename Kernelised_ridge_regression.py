import numpy as np
import matplotlib.pyplot as plt
from utilities import set_axes
from numpy import sin, pi
from linear_regressions import mse
import requests
from utilities import read_data
from filtered_boston_housing_and_kernels import a_1_2, c_1_2, d_1_2
import pandas as pd

spots = requests.get(
    'http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv', stream=True)
data = read_data(spots)
data = np.array([list(row) for row in data])


def gaussian_kernel(x1, x2, sig):
    """
    Gaussian kernel function implementation:
    exp(-||x1 - x2||^2 / (2 * sigma^2))

    Parameters.
    - x1, x2: input two sample vectors of shape (n_features,)
    - sigma: parameter controlling the width of the Gaussian kernel

    Returns.
    - Kernel function value (scalar)
    """
    return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * sig**2))


class KernelisedRegression():
    def __init__(self):
        self.alpha = np.array([])

    def kernel_matrix(self, Xi, Xj, kernel_function=gaussian_kernel, **kwargs):
        """
        Batch compute RBF kernel function values.
        Input.
        - Xi: Sample matrix 1, shape (n_samples1, n_features)
        - Xj: Sample matrix 2, shape (n_samples2, n_features)
        - sigma: Gaussian kernel width parameter

        Output.
        - Kernel matrix K, shape (n_samples1, n_samples2)
        """
        n_Xi_samples = Xi.shape[0]
        n_Xj_samples = Xj.shape[0]
        # Initialise the kernel matrix
        K = np.zeros((n_Xi_samples, n_Xj_samples))
        for i in range(n_Xi_samples):
            for j in range(n_Xj_samples):
                K[i, j] = kernel_function(Xi[i], Xj[j], **kwargs)
        return K

    def train(self, X_train, Y_train, gamma, kernel_function=gaussian_kernel, **kwargs):
        """
        Solve for α* = (K + γℓ * I_ℓ)^(-1) * y

        Parameters.
        - K: kernel matrix, shape (ℓ, ℓ)
        - y: label vector in the shape of (ℓ,)
        - gamma: regularisation parameter (scalar).

        Returns: alpha: the solved dyadic matrix, in the shape of (ℓ, ℓ)
        - alpha: vector of solved pairwise coefficients, shape (ℓ,)
        """

        K = self.kernel_matrix(X_train, X_train,
                               kernel_function, **kwargs)

        # Get the size of the kernel matrix ℓ
        ℓ = K.shape[0]

        # Construct the unit matrix I_ℓ
        I_ℓ = np.eye(ℓ)

        # Calculate the regularised kernel matrix
        K_regularized = K + gamma * ℓ * I_ℓ

        # Solve a system of linear equations: K_regularized * alpha = y
        alpha = np.linalg.solve(K_regularized, Y_train)

        self.alpha = alpha

    def predict(self, X_test, X_train, kernel_function=gaussian_kernel, **kwargs):
        return np.dot(self.kernel_matrix(X_test, X_train, kernel_function, **kwargs), self.alpha)


def five_fold_cross_validation(X_train, Y_train, folds=5):
    num_samples = X_train.shape[0]
    fold_size = num_samples // folds
    list_gamma = [2**i for i in range(-40, -25)]
    list_sigma = [2**i for i in np.arange(7, 13.5, 0.5)]
    indices = np.arange(num_samples)

    best_gamma, best_sigma, best_mse = None, None, float('inf')

    for gamma in list_gamma:
        for sigma in list_sigma:
            validate_mse = 0
            # Split data for cross validation
            for fold in range(folds):
                # Validation sets for each fold
                index_start = fold * fold_size
                index_end = (fold + 1) * \
                    fold_size if fold != folds - 1 else num_samples

                # Divide the training set and validation set
                validate_indices = indices[index_start:index_end]
                train_indices = np.concatenate(
                    [indices[:index_start], indices[index_end:]])

                X_train_cv, Y_train_cv = X_train[train_indices], Y_train[train_indices]
                X_val_cv, Y_val_cv = X_train[validate_indices], Y_train[validate_indices]
                model = KernelisedRegression()
                model.train(X_train_cv, Y_train_cv, gamma, sig=sigma)
                predict_validate = model.predict(
                    X_val_cv, X_train_cv, sig=sigma)
                validate_mse += mse(Y_val_cv, predict_validate)
            average_mse = validate_mse / folds
            if average_mse < best_mse:
                best_gamma, best_sigma,  best_mse = gamma, sigma, average_mse
    return best_gamma, best_sigma, best_mse


def record_five_fold_cross_validation(X_train, Y_train, folds=5):
    num_samples = X_train.shape[0]
    fold_size = num_samples // folds
    list_gamma = [2**i for i in range(-40, -25)]
    list_sigma = [2**i for i in np.arange(7, 13.5, 0.5)]
    indices = np.arange(num_samples)

    array_mse_gamma_to_sigma = np.zeros((len(list_gamma), len(list_sigma)))

    for g_index, gamma in enumerate(list_gamma):
        for s_index, sigma in enumerate(list_sigma):
            validate_mse = 0
            # Split data for cross validation
            for fold in range(folds):
                # Validation sets for each fold
                index_start = fold * fold_size
                index_end = (fold + 1) * \
                    fold_size if fold != folds - 1 else num_samples

                # Divide the training set and validation set
                validate_indices = indices[index_start:index_end]
                train_indices = np.concatenate(
                    [indices[:index_start], indices[index_end:]])

                X_train_cv, Y_train_cv = X_train[train_indices], Y_train[train_indices]
                X_val_cv, Y_val_cv = X_train[validate_indices], Y_train[validate_indices]
                model = KernelisedRegression()
                model.train(X_train_cv, Y_train_cv, gamma, sig=sigma)
                predict_validate = model.predict(
                    X_val_cv, X_train_cv, sig=sigma)
                validate_mse += mse(Y_val_cv, predict_validate)
            average_mse = validate_mse / folds
            array_mse_gamma_to_sigma[g_index, s_index] = average_mse
    return list_gamma, list_sigma, array_mse_gamma_to_sigma


def a_1_3():
    n_runs = 1
    for _ in range(n_runs):
        data_size = len(data)
        indices = np.arange(data_size)
        np.random.shuffle(indices)  # Random indexing

        # Data disaggregated by 2/3 and 1/3
        train_size = int(data_size * 2 / 3)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        X = data[:, :-1]
        Y = data[:, -1]   # last column
        X_train, Y_train = X[train_indices], Y[train_indices].reshape(-1, 1)
        list_gamma, list_sigma, array_mse_gamma_to_sigma = record_five_fold_cross_validation(
            X_train, X_train)

    # Creating a mesh with meshgrid
    sigma_grid, gamma_grid = np.meshgrid(list_sigma, list_gamma)

    # Drawing heat maps
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(sigma_grid, gamma_grid,
                      array_mse_gamma_to_sigma, cmap='viridis')
    plt.colorbar(cp)  # Adding a colour bar

    # Setting labels and titles
    plt.xlabel('γ (sigma values)')
    plt.ylabel('σ (gamma values)')
    plt.title('Cross-validation error as a function of γ and σ')
    plt.show()


def c_1_3(n_runs):
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

        X = data[:, :-1]
        Y = data[:, -1]   # last column
        X_train, Y_train = X[train_indices], Y[train_indices].reshape(-1, 1)
        X_test, Y_test = X[test_indices], Y[test_indices].reshape(-1, 1)
        best_gamma, best_sigma, best_mse = five_fold_cross_validation(
            X_train, X_train)
        print(best_gamma, best_sigma, best_mse)
        model = KernelisedRegression()
        model.train(X_train, Y_train, best_gamma, sig=best_sigma)
        predict_train = model.predict(X_train, X_train, sig=best_sigma)
        predict_test = model.predict(X_test, X_train, sig=best_sigma)
        list_train_mse.append(mse(Y_train, predict_train))
        list_test_mse.append(mse(Y_test, predict_test))

    train_mse, train_std = np.mean(list_train_mse), np.std(list_train_mse)
    test_mse, test_std = np.mean(list_test_mse), np.std(list_train_mse)
    return train_mse, train_std, test_mse, test_std


def d_1_3(n_runs):
    NR_train_mse, NR_train_std, NR_test_mse, NR_test_std = a_1_2()
    train_mse_for_atrrs, train_std_for_atrrs, test_mse_for_attrs, test_std_for_attrs = c_1_2()
    FA_train_mse, FA_train_std, FA_test_mse, FA_test_std = d_1_2()
    KN_train_mse, KN_train_std, KN_test_mse, KN_test_std = c_1_3(n_runs)
    list_train_mse = np.concatenate(
        ([NR_train_mse], train_mse_for_atrrs, [FA_train_mse], [KN_train_mse]))
    list_train_std = np.concatenate(
        ([NR_train_std], train_std_for_atrrs, [FA_train_std], [KN_train_mse]))
    list_test_mse = np.concatenate(
        ([NR_test_mse], test_mse_for_attrs, [FA_test_mse], [KN_train_mse]))
    list_test_std = np.concatenate(
        ([NR_test_std], test_mse_for_attrs, [FA_test_std], [KN_train_mse]))
    list_method_name = np.concatenate((['Naive Regression'], [f'Linear Regression (attribute{
                                      i})' for i in range(1, 13)], ['Linear Regression (all attributes)'], ['Kernel Ridge Regression']))
    # Put them into a dictionary whose keys are the column names
    data = {
        'Method': list_method_name,
        'train_mse': list_train_mse,
        'train_std': list_train_std,
        'test_mse': list_test_mse,
        'test_std': list_test_std
    }
    df = pd.DataFrame(data)
    # create new columns
    df['MSE train'] = df.apply(lambda row: f"{row['train_mse']} ± {
                               row['train_std']}", axis=1)
    df['MSE test'] = df.apply(lambda row: f"{row['test_mse']} ± {
                              row['test_std']}", axis=1)
    # Delete column
    df = df.drop(columns="train_mse")
    df = df.drop(columns="train_std")
    df = df.drop(columns="test_mse")
    df = df.drop(columns="test_std")

    print(df)


if __name__ == "__main__":
    # a_1_3()
    # c_1_3(1)
    d_1_3(20)
