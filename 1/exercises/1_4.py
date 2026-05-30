import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style = 'darkgrid')


def generate_target_line():

    p1 = np.random.uniform(-1, 1, 2)
    p2 = np.random.uniform(-1, 1, 2)
    
    w1 = p2[1] - p1[1]
    w2 = p1[0] - p2[0]
    w0 = p2[0] * p1[1] - p1[0] * p2[1]
    
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


def plot_experiment(X, y, w_f, w_g, iterations):

    plt.figure(figsize = (9, 7))
    
    sns.scatterplot(
        x = X[:, 1], y = X[:, 2], hue = y, style = y,
        palette = {1.0: 'dodgerblue', -1.0: 'crimson'},
        markers = {1.0: 'o', -1.0: 'X'}, s = 100, edgecolor = 'w', linewidth = 1.5
    )

    x_vals = np.array([-1.2, 1.2])
    
    y_f = (-w_f[1] * x_vals - w_f[0]) / w_f[2]
    y_g = (-w_g[1] * x_vals - w_g[0]) / w_g[2]
    
    plt.plot(x_vals, y_f, label = 'Target Function $f$', color = 'black', linestyle = '--', linewidth = 2)
    plt.plot(x_vals, y_g, label = f'Hypothesis $g$ ({iterations} steps)', color = 'forestgreen', linestyle = '-', linewidth = 2)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color = 'grey', linewidth = 1, linestyle = ':')
    plt.axvline(0, color = 'grey', linewidth = 1, linestyle = ':')
    
    plt.title(f'Perceptron Convergence for $N = 20$ ({iterations} iterations)', fontsize = 14)
    plt.xlabel('$x_1$', fontsize = 12)
    plt.ylabel('$x_2$', fontsize = 12)
    plt.legend(loc = 'upper right', frameon = True)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    N = 20
    
    w_target = generate_target_line()
    X_data, y_data = generate_data(N, w_target)
    
    w_hypothesis, total_iterations = run_pla(X_data, y_data)
    
    print(f'Algorithm converged successfully in {total_iterations} iterations.')
    print(f'Target weights (f):     {w_target}')
    print(f'Hypothesis weights (g): {w_hypothesis}')
    
    plot_experiment(X_data, y_data, w_target, w_hypothesis, total_iterations)







