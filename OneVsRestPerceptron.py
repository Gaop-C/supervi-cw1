import numpy as np
from utilities import read_data
import requests
from itertools import product
import pandas as pd

spots = requests.get(
    'http://www0.cs.ucl.ac.uk/staff/M.Herbster/SL/misc/zipcombo.dat', stream=True)
data = read_data(spots)
data = np.array([list(row) for row in data])


# One-versus-Rest kernel perceptron
class OneVsRestKernelPerceptron:
    def __init__(self, kernel, num_classes):
        self.kernel = kernel  # Kernel function
        self.num_classes = num_classes
        self.alpha = None # weight matrix
        
    def train(self, X_train, y_train, max_epochs=10, kernel_arg=7):
        """training One-Vs-Rest model (update weight matrix 'alpha' for different binary classifier according to the fomular)

        Args:
            X_train (_type_): matrix of training data X
            y_train (_type_): vector of label Y of trianing data
            max_epochs (int, optional): number of training repetitions. Defaults to 10.
            kernel_arg (int, optional): kernel parameter (for polynomial or gaussian). Defaults to 7.
        """
        self.m, self.n = X_train.shape
        self.alpha = np.zeros((self.num_classes, self.m))  # Initialise all α to 0
        self.binary_y_train = []
        for i in range(self.num_classes):
            self.binary_y_train.append(np.where(y_train == i, 1, -1))  # 1 for category i, -1 for other categories
        self.binary_y_train = np.array(self.binary_y_train)
        
        # Pre-calculated kernel matrices
        kernel_matrix = self.kernel(X_train, X_train, kernel_arg)

        for epoch in range(max_epochs):
            for t in range(self.m):
                prediction = np.dot(self.alpha, kernel_matrix[:, t]) # single samples
                label = np.argmax(prediction)
                for i in range(self.num_classes):
                    if np.sign(prediction[i]) != self.binary_y_train[i, t]:
                        self.alpha[i, t] += self.binary_y_train[i, t]

     
    def predict(self, X_train, X_test, kernel_arg=7):
        """predict the classes for matrix of any input X_test

        Args:
            X_train (_type_): support matrix (matrix of training X)
            X_test (_type_):  matrix of input X needs to be predicted
            kernel_arg (int, optional): kernel parameter (for polynomial or gaussian). Defaults to 7.

        Returns:
            _type_: vector of predicted label
        """
        kernel_matrix = self.kernel(X_train, X_test, kernel_arg)  # Calculate the kernel matrix of test samples and training samples
        predictions = np.dot(self.alpha, kernel_matrix)    # Batch forecasting
        # Calculate actual labels
        predict_y = np.argmax(predictions, axis=0)
        
        return predict_y    

# Kernel function


def polynomial_kernel(X_train, X_in, d=7):
    # polynomial kernel K(x_i, x_t) = (x_i · x_t)^d
    return (np.dot(X_train, X_in.T)) ** d


def gaussian_kernel(X_train, X_in, c=1):
    # Gaussian kernel K(x_i, x_t) = exp(-c ||x_i - x_t||^2)
    X_train_norm = np.sum(X_train**2, axis=1).reshape(-1, 1)  # X1**2
    X_in_norm = np.sum(X_in**2, axis=1).reshape(1, -1)  # X2**2
    dist = X_train_norm + X_in_norm - 2 * np.dot(X_train, X_in.T) # ||X1, X2|| = X1**2 - 2*X1*X2 + X2**2

    return np.exp(-c * dist)

def evaluate(X_train, y_train, X_test, y_test, kernel=polynomial_kernel, kernel_arg=7):
    """Complete one training session (number of max_epoch) and predict

    Args:
        X_train (_type_): matrix of training data X
        y_train (_type_): vector of label Y of trianing data
        X_test (_type_): matrix of test data X
        y_test (_type_): vector of label Y of test data
        kernel (_type_, optional): type of kernel function. Defaults to polynomial_kernel.
        kernel_arg (int, optional): kernel parameter (for polynomial or gaussian). Defaults to 7.

    Returns:
        _type_: train_error_rate, test_error_rate, X_test_predictions
    """
    OneVsRestClassifier = OneVsRestKernelPerceptron(kernel, 10)
    OneVsRestClassifier.train(X_train, y_train, kernel_arg = kernel_arg)
    X_train_predictions = OneVsRestClassifier.predict(X_train, X_train, kernel_arg)
    X_test_predictions = OneVsRestClassifier.predict(X_train, X_test, kernel_arg)
    train_error_rate = np.sum(X_train_predictions != y_train) / len(y_train)
    test_error_rate = np.sum(X_test_predictions != y_test) / len(y_test)
        
    return train_error_rate, test_error_rate, X_test_predictions

def Q3(kernel=polynomial_kernel, kernelargs=list(range(1, 8))):
    """Solving question 3: evaluate the error in 20 runs for different values of kernel parameters (here d for polynomial)

    Args:
        kernel (_type_, optional): type of kernel function. Defaults to polynomial_kernel.
        kernelargs (_type_, optional): list of kernel parameter (for polynomial or gaussian). Defaults to list(range(1, 8)).
    """
    n_runs = 20
    train_error_rates = {d: [] for d in kernelargs}
    test_error_rates = {d: [] for d in kernelargs}
    for run in range(n_runs):
        data_size = len(data)
        indices = np.arange(data_size)
        np.random.shuffle(indices)  # Random indexing

        # Data divided by 2/3 and 1/3
        train_size = int(data_size * 4 / 5)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        X_train, y_train = data[train_indices, 1:], data[train_indices, 0]
        X_test, y_test = data[test_indices, 1:], data[test_indices, 0]
        
        for d in kernelargs:
            train_error_rate, test_error_rate, _ = evaluate(X_train, y_train, X_test, y_test, kernel=kernel, kernel_arg=d)
            train_error_rates[d].append(train_error_rate)
            test_error_rates[d].append(test_error_rate)
            print(f'[runs = {run+1}] [d = {d}] train_error_rate: {train_error_rate}')
            print(f'[runs = {run+1}] [d = {d}] test_error_rate: {test_error_rate}')
    
    avg_train_error_rates, std_train_errors_rates = [float(np.mean(train_error_rates[d])) for d in kernelargs], [float(np.std(train_error_rates[d])) for d in kernelargs]     
    avg_test_error_rates, std_test_errors_rates = [float(np.mean(test_error_rates[d])) for d in kernelargs], [float(np.std(test_error_rates[d])) for d in kernelargs]     
    
    error_data = {
        'kernelargs(d/c)': kernelargs,
        'avg_train_error_rates': avg_train_error_rates,
        'std_train_errors_rates': std_train_errors_rates,
        'avg_test_error_rates': avg_test_error_rates,
        'std_test_errors_rates': std_test_errors_rates
    }
    df = pd.DataFrame(error_data)
    # create new columns
    df['Error rate train'] = df.apply(lambda row: f"{row['avg_train_error_rates']} ± {
                               row['std_train_errors_rates']}", axis=1)
    df['Error rate test'] = df.apply(lambda row: f"{row['avg_test_error_rates']} ± {
                              row['std_test_errors_rates']}", axis=1)
    # Delete column
    df = df.drop(columns="avg_train_error_rates")
    df = df.drop(columns="std_train_errors_rates")
    df = df.drop(columns="avg_test_error_rates")
    df = df.drop(columns="std_test_errors_rates")

    print(df)



def cross_validation(X, y, kernel=polynomial_kernel, kernelargs=list(range(1, 8)), folds=5):
    """5 folds cross validation

    Args:
        X (_type_): matrix of input X
        y (_type_): vector of input y
        kernel (_type_, optional): type of kernel function. Defaults to polynomial_kernel.
        kernelargs (_type_, optional): list of kernel parameter (for polynomial or gaussian). Defaults to list(range(1, 8)).
        folds (int, optional): number of folds. Defaults to 5.

    Returns:
        _type_: the optimal kernel function parameters found
    """
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split the data
    fold_sizes = n_samples // folds
    cv_error_rates = {d: [] for d in kernelargs}

    for fold in range(folds):
        print(f'started {fold+1} fold')
        # Define training and validation sets
        start, end = fold * fold_sizes, (fold + 1) * fold_sizes if fold != folds - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # For each parameter d train the model and calculate the error
        for d in kernelargs:
            _, test_error_rate, _ = evaluate(X_train, y_train, X_val, y_val, kernel=kernel, kernel_arg=d)
            cv_error_rates[d].append(test_error_rate)
        print(f'finished {fold+1} fold')

    # Calculate the average validation error for each d
    avg_error_rates = {d: float(np.mean(cv_error_rates[d])) for d in kernelargs}
    best_d = min(avg_error_rates, key=avg_error_rates.get)
    return best_d, avg_error_rates


def Q4Q5(kernel=polynomial_kernel, kernelargs=list(range(1, 8))):
    """Solving question 4 and 5: find the optimal kernel function parameter (here d for polynomial) 
       by 5-fold-cross-validation and use it to train and evaluate the model and construct the confusion matrices

    Args:
        kernel (_type_, optional): type of kernel function. Defaults to polynomial_kernel.
        kernelargs (_type_, optional): list of kernel parameter (for polynomial or gaussian). Defaults to list(range(1, 8)).
    """
    n_runs = 20
    best_d_results = []
    train_error_rates = []
    test_error_rates = []
    confusion_matrices = np.zeros((10, 10, 20))
    for run in range(n_runs):
        print(f'started {run+1} run')
        data_size = len(data)
        indices = np.arange(data_size)
        np.random.shuffle(indices)  # Random indexing

        # Data divided by 2/3 and 1/3
        train_size = int(data_size * 4 / 5)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        X_train, y_train = data[train_indices, 1:], data[train_indices, 0]
        X_test, y_test = data[test_indices, 1:], data[test_indices, 0]
        
        best_d, _ = cross_validation(X_train, y_train, kernel, kernelargs)
        best_d_results.append(best_d)
        print(f'[run = {run+1}] found the best d')
        train_error_rate, test_error_rate, test_predict_y = evaluate(X_train, y_train, X_test, y_test, kernel=kernel, kernel_arg=best_d)
        train_error_rates.append(train_error_rate)
        test_error_rates.append(test_error_rate)
        print(f'[run = {run+1}] finished training and testing')
        # Confusion matrix
        for i, j in product(range(10), range(10)):
            if i != j:
                conf_mask = (y_test == i) & (np.array(test_predict_y) == j)
                confusion_matrices[i, j, run] = conf_mask.sum() / (y_test == i).sum()
        print(f'finished run {run+1}')
    avg_best_d, std_best_d = np.mean(best_d_results), np.std(best_d_results)
    avg_train_error, std_train_error = np.mean(train_error_rates), np.std(train_error_rates)
    avg_test_error, std_test_error = np.mean(test_error_rates), np.std(test_error_rates)
    
    # Saved as CSV, each layer of the matrix is saved as a DataFrame
    for k in range(confusion_matrices.shape[2]):  # Traverse the results of each run
        pd.DataFrame(confusion_matrices[:, :, k]).to_csv(f"confusion_matrix_run_{k}.csv", index=False)

    print(f'avg_chosen_d: {avg_best_d}, \nstd_chosen_d: {std_best_d}, \navg_train_error: {avg_train_error}, \nstd_train_error: {std_train_error}, \navg_test_error: {avg_test_error}, \nstd_test_error: {std_test_error}')


def Q7_b():
    """Solving question 7b: evaluate the error in 20 runs for different values of kernel parameters (here c for gaussian)
    """
    Q3(kernel=gaussian_kernel, kernelargs=[10**-2, 10**-1.5, 10**-1, 10**-0.5, 10**0])
    

def Q7_c():
    """Solving question 7_c: find the optimal kernel function parameter (here c for gaussian) 
       by 5-fold-cross-validation and use it to train and evaluate the model.
    """
    Q4Q5(kernel=gaussian_kernel, kernelargs=[10**-2, 10**-1.5, 10**-1, 10**-0.5, 10**0])


if __name__ == "__main__":
    # Q3()
    Q4Q5()
    # Q7_b()
    # Q7_c()
