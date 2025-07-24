import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer


def generateData(r, thk, sep, xoff, n):

    rin = r
    rout = r + thk

    def generateSemiCircle(top):

        radii = np.sqrt(np.random.uniform(rin ** 2, rout ** 2, n))
        angles = np.random.uniform(0, np.pi, n) if top else np.random.uniform(np.pi, np.pi * 2, n)

        x = radii * np.cos(angles)
        y = radii * np.sin(angles)

        if top:
            x -= xoff
            y += sep

        else:
            x += xoff
            y -= sep

        return np.column_stack((x, y))
    
    plus = generateSemiCircle(top = True)
    minus = generateSemiCircle(top = False)

    x = np.vstack((plus, minus))
    y = np.hstack((np.ones(n), -np.ones(n)))

    return x, y


def pocket(x, y, maxIt = 100000):

    start = timer()

    x = np.hstack((np.ones((x.shape[0], 1)), x))
    w = np.zeros(x.shape[1])
    w_star = w.copy()

    minEin = float('inf')

    for _ in range(maxIt):

        misclassified = False

        for xi, yi in zip(x, y):

            if np.sign(xi @ w) != yi:

                w += yi * xi
                misclassified = True

        currEin = np.mean(np.sign(x @ w) != y)

        if currEin < minEin:

            minEin = currEin
            w_star = w.copy()

        if not misclassified:
            break

    end = timer()

    return w_star, minEin, end - start


def linearRegression(x, y):

    start = timer()

    x = np.hstack((np.ones((x.shape[0], 1)), x))
    w = np.linalg.pinv(x.T @ x) @ x.T @ y

    end = timer()

    return w, end - start


def plotResults(x, y, wP, wLR):

    plt.figure(figsize = (10, 8))
    sns.set_style('darkgrid')

    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker = '+', color = 'blue', label = '+1')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker = '_', color = 'red', label = '-1')

    xVals = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 500)

    plt.plot(xVals, -1 * (wP[0] + (wP[1] * xVals)) / wP[2], color = 'black', label = 'Pocket Algorithm')
    plt.plot(xVals, -1 * (wLR[0] + (wLR[1] * xVals)) / wLR[2], color = 'green', label = 'Linear Regression')

    plt.title('Pocket Algorithm versus Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    
    x, y = generateData(r = 10, thk = 5, sep = -0.5, xoff = 5, n = 1000)
    wP, EinP, tP = pocket(x, y)
    wLR, tLR = linearRegression(x, y)

    xBias = np.hstack((np.ones((x.shape[0], 1)), x))
    EinLR = np.mean(np.sign(xBias @ wLR) != y)

    print(f'{'Metric':<20} | {'Pocket Algorithm':<20} | {'Linear Regression':<20}')
    print(f'{'-' * 20} | {'-' * 20} | {'-' * 20}')
    print(f'{'In-Sample Error':<20} | {EinP:<20.6f} | {EinLR:<20.6f}')
    print(f'{'Computation Time':<20} | {tP:<20.6f} | {tLR:<20.6f}')

    plotResults(x, y, wP, wLR)

if __name__ == '__main__': main()
