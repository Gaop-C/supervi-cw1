import numpy as np
import matplotlib.pyplot as plt
from utilities import set_axes
from numpy import sin, pi
from linear_regressions import mse
from utilities import set_axes


def euclidean_distance(x1, x2,):
    """Calculate the Euclidean distance"""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def resolve_undefined(label):
    """
    Handle "undefined" cases: if there is a problem with bincount (e.g. empty labels), randomly generate a label
    """
    # Count the number of occurrences of each label
    counts = np.bincount(label)
    # Find the tag with the most votes (there may be more than one)
    max_votes = counts.max()
    candidates = np.where(counts == max_votes)[0]
    # If there are multiple tags with the same number of votes, randomly select one of them
    return np.random.choice(candidates)


def knn_classification(h, label, X_test, k):
    # Vectorised distance calculation
    distances = np.linalg.norm(h - X_test[:, np.newaxis], axis=2)
    # Sort the distances to get the index from smallest to largest.
    # Takes the first k indexes, corresponding to the positions of the k smallest values in the array.
    k_indexes = np.argsort(distances, axis=1)[:, :k]
    k_nearest_labels = label[k_indexes]

    predict = np.apply_along_axis(
        resolve_undefined, axis=1, arr=k_nearest_labels)
    return predict


def visialise_hsv():
    h = np.random.uniform(0, 1, (100, 2))
    label = np.random.randint(0, 2, size=100)

    # Generate 2D grid points
    x1_min, x1_max = 0, 1
    x2_min, x2_max = 0, 1
    grid_size = 500
    x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min, x1_max, grid_size),
                                   np.linspace(x2_min, x2_max, grid_size))

    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]  # Flatten to a 2D array

    # Classification of grid points
    grid_predict = knn_classification(h, label, X_grid, k=3)
    grid_predict = grid_predict.reshape(grid_size, grid_size)

    # Mapping decision-making boundaries
    plt.contourf(x1_grid, x2_grid, grid_predict, alpha=0.5, cmap='coolwarm')
    plt.scatter(h[:, 0], h[:, 1], c=label,
                cmap='coolwarm', edgecolor='k', s=50)
    plt.title(f"k-NN Decision Boundary (k=3)")
    plt.show()


def generating_data(h, h_label, n, k):
    # Randomly generate training data
    training_data = np.random.uniform(0, 1, (n, 2))

    # Calculate the distance from each generated point to the reference point
    distances = np.linalg.norm(
        training_data[:, np.newaxis, :] - h[np.newaxis, :, :], axis=2)

    # Find the index of the nearest k neighbours of each point
    k_indexes = np.argsort(distances, axis=1)[:, :k]

    # Find nearest neighbour tags
    k_nearest_labels = h_label[k_indexes]

    # Majority vote on labelling
    voted_labels = np.apply_along_axis(
        resolve_undefined, axis=1, arr=k_nearest_labels)

    # Adjust 20 per cent of labels to random values based on probability
    random_mask = np.random.rand(n) < 0.2
    random_labels = np.random.choice([0, 1], size=n)
    final_labels = np.where(random_mask, random_labels, voted_labels)
    return training_data, final_labels


def mse(expected, predict):
    SSE = np.sum((expected - predict) ** 2)
    MSE = SSE / len(expected)
    return MSE


def protocol_a():
    n_runs = 100
    k = 49
    mse_k_n = np.zeros((k, n_runs))
    mse_k = np.zeros(k)

    for i in range(n_runs):
        # Generate all data in advance
        h = np.random.uniform(0, 1, (100, 2))
        h_label = np.random.randint(0, 2, size=100)
        data_cache = [
            (generating_data(h, h_label, 4000, j),
             generating_data(h, h_label, 1000, j))
            for j in range(1, k + 1)
        ]
        for j in range(1, k+1):
            h_train, label_train = data_cache[j - 1][0]
            h_test, label_test = data_cache[j - 1][1]
            k_predict = knn_classification(h_train, label_train, h_test, j)
            mse_k_n[j-1, i] = mse(label_test, k_predict)
            print(
                f'[n={i+1} runs] [k={j} nearest neighbour] MSE for this run: {mse_k_n[j-1, i]}')

    mse_k = mse_k_n.mean(axis=1)  # One-time calculation of mean values

    x_axe = np.linspace(1, 49, 49)
    fig, ax = plt.subplots()
    set_axes(ax, 10, 0.05, 1, 0.01, -0.01, 51.0, -0.01, 0.3)
    ax.set_xlabel('k')
    ax.set_ylabel('error')
    ax.set_title('k neighbour vs generalisation error')
    ax.plot(x_axe, mse_k, "r-", label="MSE")
    ax.legend()
    plt.show()


def protocol_b():
    n_runs = 100
    k = 49
    list_m = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    optimal_k = []
    for m in list_m:
        mse_k_n = np.zeros((k, n_runs))
        mse_k = np.zeros(k)
        for i in range(n_runs):
            # Generate all data in advance
            h = np.random.uniform(0, 1, (100, 2))
            h_label = np.random.randint(0, 2, size=100)
            data_cache = [
                (generating_data(h, h_label, 4000, j),
                 generating_data(h, h_label, 1000, j))
                for j in range(1, k + 1)
            ]
            for j in range(1, k+1):
                h_train, label_train = data_cache[j - 1][0]
                h_test, label_test = data_cache[j - 1][1]
                k_predict = knn_classification(h_train, label_train, h_test, j)
                mse_k_n[j-1, i] = mse(label_test, k_predict)
                print(
                    f'[m={m} training points] [n={i+1} runs] [k={j} nearest neighbour] MSE for this run: {mse_k_n[j-1, i]}')
                
        mse_k = mse_k_n.mean(axis=1)  # One-time calculation of mean values
        optimal_k.append(np.argmin(mse_k) + 1)

    fig, ax = plt.subplots()
    set_axes(ax, 500, 5, 100, 1, -100, 4100, -5, 55)
    ax.set_xlabel('m training data ')
    ax.set_ylabel('k neighbour')
    ax.set_title('m versus optimal k')
    ax.plot(list_m, optimal_k, "r-")
    plt.show()


if __name__ == "__main__":
    # visialise_hsv()
    # protocol_a()
    protocol_b()
