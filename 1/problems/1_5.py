import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style = 'darkgrid')


def generate_target_line():

    x1, y1 = np.random.uniform(-1, 1, 2)
    x2, y2 = np.random.uniform(-1, 1, 2)

    w0 = y1 * x2 - y2 * x1
    w1 = y2 - y1
    w2 = x1 - x2

    return np.array([w0, w1, w2])


def generate_data(N, w_f):
    
    X = np.random.uniform(-1, 1, (N, 2))
    X_with_bias = np.hstack((np.ones((N, 1)), X))
    y = np.sign(np.dot(X_with_bias, w_f))

    while len(np.unique(y)) < 2:

        X = np.random.uniform(-1, 1, (N, 2))
        X_with_bias = np.hstack((np.ones((N, 1)), X))
        y = np.sign(np.dot(X_with_bias, w_f))

    return X_with_bias, y



def run_adaline(X, y, n, N_updates):

    w = np.zeros(X.shape[1])

    for _ in range(N_updates):
    
        idx = np.random.randint(0, X.shape[0])
        s = np.dot(X[idx], w)
        w += n * (y[idx] - s) * X[idx]

    return w


def plot_experiment(X, y, w_f, w_g) -> None:

    plt.figure(figsize = (9, 7))

    sns.scatterplot(
        x = X[:, 1], y = X[:, 2], hue = y, style = y,
        palette = {1.0: 'dodgerblue', -1.0: 'crimson'},
        markers = {1.0: 'o', -1.0: 'X'}, s = 100, edgecolor = 'w', linewidth = 1.5
    )

    x_vals = np.array([-1.2, 1.2])

    y_f = (-w_f[1] * x_vals - w_f[0]) / w_f[2]
    y_g = (-w_g[1] * x_vals - w_g[0]) / w_g[2]

    plt.plot(x_vals, y_f, label = 'Target Function', color = 'black', linestyle = '--', linewidth = 2)
    plt.plot(x_vals, y_g, label = f'Hypothesis', color = 'forestgreen', linestyle = '-', linewidth = 2)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color = 'grey', linewidth = 1, linestyle = ':')
    plt.axvline(0, color = 'grey', linewidth = 1, linestyle = ':')
    
    plt.xlabel('$x_1$', fontsize = 12)
    plt.ylabel('$x_2$', fontsize = 12)
    plt.legend(loc = 'upper right', frameon = True)
    plt.tight_layout()
    plt.show()




def main() -> None:

    N_training = 100
    N_test = 10000
    N_updates = 1000
    eta = 0.0001

    w_target = generate_target_line()
    X_training, y_training = generate_data(N_training, w_target)
    X_test, y_test = generate_data(N_test, w_target)
    
    w_hypothesis = run_adaline(X_training, y_training, eta, N_updates)

    plot_experiment(X_test, y_test, w_target, w_hypothesis)

if __name__ == '__main__': main()






