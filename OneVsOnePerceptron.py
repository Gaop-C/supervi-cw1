import numpy as np
import itertools
from utilities import read_data
import requests
import pandas as pd

spots = requests.get(
    'http://www0.cs.ucl.ac.uk/staff/M.Herbster/SL/misc/zipcombo.dat', stream=True)
data = read_data(spots)
data = np.array([list(row) for row in data])


class OneVsOneKernelPerceptron:
    def __init__(self, kernel):
        self.kernel = kernel
        self.classifiers = {}  # Store all binary classifiers
        self.category = np.array([])

    def train(self, X_train, y_train, max_epochs=10, kernel_arg=7):
        """training One-Vs-One model (update weight vector 'alpha' for different binary classifier according to the fomular)

        Args:
            X_train (_type_): matrix of training data X
            y_train (_type_): vector of label Y of trianing data
            max_epochs (int, optional): number of training repetitions. Defaults to 10.
            kernel_arg (int, optional): kernel parameter (for polynomial or gaussian). Defaults to 7.
        """
        self.category = np.unique(y_train)  # Get all categories
        self.classifiers = {}  # Empty the classifier store

        for category_i, category_j in itertools.combinations(self.category, 2):
            # Filtering samples belonging to the two current categories
            mask = np.logical_or(y_train == category_i, y_train == category_j)
            X_sub = X_train[mask]
            y_sub = y_train[mask]

            # Recode categories as +1 and -1
            y_binary = np.where(y_sub == category_i, 1, -1)

            # Initialise the binary classifier
            n_samples = X_sub.shape[0]
            alpha = np.zeros(n_samples)
            
            # Pre-calculated kernel matrices
            kernel_matrix = self.kernel(X_sub, X_sub, kernel_arg)
            for epoch in range(max_epochs):
                # training
                for t in range(n_samples):
                    prediction = np.sign(np.dot(alpha, kernel_matrix[:, t]))
                    if prediction != y_binary[t]:
                        alpha[t] += y_binary[t]

            # Storing Classifier Parameters
            self.classifiers[(category_i, category_j)] = {
                "alpha": alpha,
                "support_vectors": X_sub,
                "kernel_matrix": kernel_matrix,
                "kernel_arg": kernel_arg,
            }
            
    def predict(self, X_test):
        """predict the classes for matrix of any input X_test

        Args:
            X_test (_type_): matrix of input X needs to be predicted

        Returns:
            _type_: vector of predicted label
        """
        n_samples = X_test.shape[0]
        votes = np.zeros((n_samples, len(self.category)))  # Voting records for each sample

        for (category_i, category_j), classifier in self.classifiers.items():
            alpha = classifier["alpha"]
            support_vectors = classifier["support_vectors"]
            kernel_arg = classifier["kernel_arg"]

            # Calculate the kernel matrix of test samples and support vectors
            kernel_matrix = self.kernel(support_vectors, X_test, kernel_arg)

            # Prediction of binary classifiers
            predictions = np.sign(np.dot(alpha, kernel_matrix))

            # Converted to category voting
            votes[:, int(category_i)] += (predictions == 1)
            votes[:, int(category_j)] += (predictions == -1)

        # Final prediction categories
        return np.argmax(votes, axis=1)

def evaluate(X_train, y_train, X_test, y_test, kernel_arg=7):
    """Complete one training session (number of max_epoch) and predict

    Args:
        X_train (_type_): matrix of training data X
        y_train (_type_): vector of label Y of trianing data
        X_test (_type_): matrix of test data X
        y_test (_type_): vector of label Y of test data
        kernel_arg (int, optional): kernel parameter (for polynomial or gaussian). Defaults to 7.

    Returns:
        _type_: _description_
    """
    OneVsOneClassifier = OneVsOneKernelPerceptron(polynomial_kernel)
    OneVsOneClassifier.train(X_train, y_train, kernel_arg = kernel_arg)
    X_train_predictions = OneVsOneClassifier.predict(X_train)
    X_test_predictions = OneVsOneClassifier.predict(X_test)
    train_error_rate = np.sum(X_train_predictions != y_train) / len(y_train)
    test_error_rate = np.sum(X_test_predictions != y_test) / len(y_test)
        
    return train_error_rate, test_error_rate
    
    
def polynomial_kernel(X_train, X_in, d=7):
    # polynomial kernel K(x_i, x_t) = (x_i · x_t)^d
    return (np.dot(X_train, X_in.T)) ** d


def Q8_b(kernelargs=list(range(1, 8))):
    """Solving question 8_b: evaluate the error in 20 runs for different values of kernel parameters (here d for polynomial)

    Args:
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
            train_error_rate, test_error_rate = evaluate(X_train, y_train, X_test, y_test, kernel_arg=d)
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


def cross_validation(X, y, kernelargs=list(range(1, 8)), folds=5):
    """5 fold cross validation
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
            _, test_error_rate = evaluate(X_train, y_train, X_val, y_val, kernel_arg=d)
            cv_error_rates[d].append(test_error_rate)
        print(f'finished {fold+1} fold')

    # Calculate the average validation error for each d
    avg_error_rates = {d: float(np.mean(cv_error_rates[d])) for d in kernelargs}
    best_d = min(avg_error_rates, key=avg_error_rates.get)
    return best_d, avg_error_rates
    
    
def Q8_c(kernelargs=list(range(1, 8))):
    """Solving question 8_c: find the optimal kernel function parameter (here d for polynomial) 
       by 5-fold-cross-validation and use it to train and evaluate the model.

    Args:
        kernelargs (_type_, optional): list of kernel parameter (for polynomial or gaussian). Defaults to list(range(1, 8)).
    """
    n_runs = 20
    best_d_results = []
    train_error_rates = []
    test_error_rates = []
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
        
        best_d, _ = cross_validation(X_train, y_train, kernelargs)
        best_d_results.append(best_d)
        print(f'[run = {run+1}] found the best d')
        train_error_rate, test_error_rate = evaluate(X_train, y_train, X_test, y_test, kernel_arg=best_d)
        train_error_rates.append(train_error_rate)
        test_error_rates.append(test_error_rate)
        print(f'[run = {run+1}] finished training and testing')
        print(f'finished run {run+1}')
    avg_best_d, std_best_d = np.mean(best_d_results), np.std(best_d_results)
    avg_train_error, std_train_error = np.mean(train_error_rates), np.std(train_error_rates)
    avg_test_error, std_test_error = np.mean(test_error_rates), np.std(test_error_rates)
     
    print(f'avg_chosen_d: {avg_best_d}, \nstd_chosen_d: {std_best_d}, \navg_train_error: {avg_train_error}, \nstd_train_error: {std_train_error}, \navg_test_error: {avg_test_error}, \nstd_test_error: {std_test_error}')
    
    
if __name__ == "__main__":   
    # Q8_b()
    Q8_c()