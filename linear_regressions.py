import numpy as np
import matplotlib.pyplot as plt
from utilities import set_axes
from numpy import sin, pi


class Regression():
    def __init__(self):
        self.W = np.array([])

    def polynomial_base(self, X_train, degree):
        Phi = np.hstack([X_train**d for d in range(degree)])
        return Phi

    def fourier_base(self, X_train, degree):
        Phi = np.hstack([sin(d*pi*X_train) for d in range(1, degree+1)])
        return Phi

    def train(self, X_train, Y_train, degree=None, basis=None):
        if basis == "polynomial":
            Phi = self.polynomial_base(X_train, degree)
        elif basis == "fourier":
            Phi = self.fourier_base(X_train, degree)
        else:
            Phi = X_train
        W = np.linalg.solve(Phi.T @ Phi, Phi.T @ Y_train)
        self.W = np.array(W)

    def predict(self, X_test, basis=None):
        if basis == "polynomial":
            return np.dot(self.polynomial_base(X_test, len(self.W)), self.W)
        elif basis == "fourier":
            return np.dot(self.fourier_base(X_test, len(self.W)), self.W)
        else:
            return np.dot(X_test, self.W)


def predict_by_diff_models(X_train, Y_train, list_degree, X_test, basis):
    predictions = []
    for degree in list_degree:
        model = Regression()
        model.train(X_train, Y_train, degree, basis)
        Y_predict = model.predict(X_test, basis)
        predictions.append(Y_predict)
    return np.array(predictions)


def plot_predict_by_diff_models(X_train, Y_target, list_degree, X_test, basis, line_patterns,
                                x_major_scale, y_major_scale, x_subscale, y_subscale,
                                xlim_left=None, xlim_right=None, ylim_bottom=None, ylim_top=None):

    predictions = predict_by_diff_models(
        X_train, Y_target, list_degree, X_test, basis)
    fig, ax = plt.subplots()
    set_axes(ax, x_major_scale, y_major_scale, x_subscale, y_subscale,
             xlim_left, xlim_right, ylim_bottom, ylim_top)

    for index, prediction in enumerate(predictions):
        ax.plot(X_test, prediction,
                line_patterns[index], label=f'Polynomial: {list_degree[index]}')
    ax.plot(X_train, Y_target, "b.")
    ax.set_title('Plot of Polynomial Fitting')
    ax.legend()
    plt.show()


def mse(expected, predict):
    SSE = np.sum((expected - predict) ** 2)
    MSE = SSE / len(expected)
    return MSE


def mse_by_diff_models(X_train, Y_train, list_degree, X_test, Y_test, basis):
    polys_mse_results = []
    predictions = predict_by_diff_models(
        X_train, Y_train, list_degree, X_test, basis)
    for prediction in predictions:
        polys_mse_results.append(mse(Y_test, prediction))
    return np.array(polys_mse_results)


def g(X, noise):
    return sin(2*pi*X)**2 + noise


def g_no_noise(X):
    return sin(2*pi*X)**2


def plot_noise_points_with_sin2pix(X_noise, Y_noise, X_test):
    Y = g_no_noise(X_test)
    fig, ax = plt.subplots()
    set_axes(ax, 0.2, 0.2, 0.05, 0.05, 0, 1, 0, 1)
    ax.plot(X_noise, Y_noise, "b.")
    ax.plot(X_test, Y, "b-")
    plt.show()


def c(basis):
    list_training_mse = np.zeros(18)
    list_test_mse = np.zeros(18)
    X_noise = np.random.uniform(0, 1, 30).reshape(-1, 1)
    Y_noise = g(X_noise, noise)
    X_test = np.linspace(0, 1, 1000).reshape(-1, 1)
    noise2 = np.random.normal(mean, std, 1000).reshape(-1, 1)
    Y_test = g(X_test, noise2)
    list_training_mse = mse_by_diff_models(X_noise,
                                           Y_noise, list(range(1, 19)), X_noise, Y_noise, basis)
    list_test_mse = mse_by_diff_models(X_noise,
                                       Y_noise, list(range(1, 19)), X_test, Y_test, basis)
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(list(range(1, 19)), list_training_mse,
            'r-', label='Training MSE')
    ax.plot(list(range(1, 19)), list_test_mse,
            'b-', label='Testing MSE')
    ax.set_title(f'Plot of error vs {basis} basis of dimension k (1 cycles)')
    ax.legend()
    plt.show()


def d(basis):
    list_training_mse = np.zeros(18)
    list_test_mse = np.zeros(18)
    for i in range(1, 101):
        X_noise = np.random.uniform(0, 1, 30).reshape(-1, 1)
        Y_noise = g(X_noise, noise)
        X_test = np.linspace(0, 1, 1000).reshape(-1, 1)
        noise2 = np.random.normal(mean, std, 1000).reshape(-1, 1)
        Y_test = g(X_test, noise2)
        list_training_mse += mse_by_diff_models(X_noise,
                                                Y_noise, list(range(1, 19)), X_noise, Y_noise, basis)
        list_test_mse += mse_by_diff_models(X_noise,
                                            Y_noise, list(range(1, 19)), X_test, Y_test, basis)
    list_training_mse = list_training_mse/100
    list_test_mse = list_test_mse/100
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(list(range(1, 19)), list_training_mse,
            'r-', label='Training MSE')
    ax.plot(list(range(1, 19)), list_test_mse,
            'b-', label='Testing MSE')
    ax.set_title(f'Plot of error vs {basis} basis of dimension k (100 cycles)')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    X_train = np.array([1, 2, 3, 4]).reshape(-1, 1)
    Y_train = np.array([3, 2, 0, 5]).reshape(-1, 1)
    list_degree = [1, 2, 3, 4]
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    line_patterns = ['b-', 'g-', 'r-', 'y-']
    print(mse_by_diff_models(
        X_train, Y_train, list_degree, X_train, Y_train, "polynomial"))
    plot_predict_by_diff_models(X_train, Y_train, list_degree, X_test, "polynomial",
                                line_patterns, 1, 2, 0.2, 0.5, 0, 4, -2, 6)

    mean = 0
    std = 0.07
    n_samples = 30
    # Generate noise
    noise = np.random.normal(mean, std, n_samples).reshape(-1, 1)
    X_noise = np.random.uniform(0, 1, 30).reshape(-1, 1)
    Y_noise = g(X_noise, noise)
    list_degree = [2, 5, 10, 14, 18]
    X_test = np.linspace(0, 1, 1000).reshape(-1, 1)
    line_patterns = ['b-', 'g-', 'r-', 'c-', 'k-']
    plot_noise_points_with_sin2pix(X_noise, Y_noise, X_test)
    plot_predict_by_diff_models(X_noise, Y_noise, list_degree, X_test, "polynomial",
                                line_patterns, 0.2, 0.2, 0.05, 0.05, 0, 1, -0.4, 1.5)

    c("polynomial")
    d("polynomial")
    c("fourier")
    d("fourier")
