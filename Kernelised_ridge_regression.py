import numpy as np
import matplotlib.pyplot as plt
from utilities import set_axes
from numpy import sin, pi
from linear_regressions import mse
import requests
from utilities import read_data
from filtered_boston_housing_and_kernels import a_1_2, c_1_2, d_1_2

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
        train_mse += mse(Y_train, predict_train)
        test_mse += mse(Y_test, predict_test)

    train_mse = train_mse / n_runs
    test_mse = test_mse / n_runs
    return train_mse, test_mse


def d_1_3(n_runs):
    c_1_3(n_runs)
    a_1_2()
    c_1_2()
    d_1_2()
 
if __name__ == "__main__":
    a_1_3()
    c_1_3(1)