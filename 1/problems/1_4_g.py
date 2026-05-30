import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style = 'darkgrid')


def generate_target_line():

    points = np.random.uniform(-1, 1, (10, 10))
    w_features = np.linalg.solve(points, -np.ones(10))

    return np.hstack(([1.0], w_features))



def generate_data(N, w_f):
    
    X = np.random.uniform(-1, 1, (N, 10))
    X_with_bias = np.hstack((np.ones((N, 1)), X))
    y = np.sign(np.dot(X_with_bias, w_f))

    while len(np.unique(y)) < 2:

        X = np.random.uniform(-1, 1, (N, 10))
        X_with_bias = np.hstack((np.ones((N, 1)), X))
        y = np.sign(np.dot(X_with_bias, w_f))

    return X_with_bias, y



def run_pla(X, y):

    w = np.zeros(X.shape[1])
    iterations = 0

    while True:
    
        predictions = np.sign(np.dot(X, w))
        predictions[predictions == 0] = -1
        misclassified = np.where(predictions != y)[0]

        if len(misclassified) == 0:
            break

        idx = np.random.choice(misclassified)
        w += y[idx] * X[idx]
        iterations += 1

    return w, iterations



def main() -> None:

    N = 1000
    
    w_target = generate_target_line()
    X_data, y_data = generate_data(N, w_target)

    N_exp = 100
    iteration_counts = []

    for _ in range(N_exp):
        _, iterations = run_pla(X_data, y_data)
        iteration_counts.append(iterations)

    plt.figure(figsize = (9, 6))
    sns.histplot(iteration_counts, kde = True, color = "purple", stat = "count")
    
    plt.title('Distribution of PLA Convergence Times (100 Runs on the Same Dataset)\n$N = 1000, d = 10$', fontsize = 13)
    plt.xlabel('Number of Updates to Converge', fontsize=12)
    plt.ylabel('Frequency', fontsize = 12)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__': main()








