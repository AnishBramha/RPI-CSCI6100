import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement as combo


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


def polynomialFeatureTransform(x, deg):

    nSamples, nFeatures = x.shape

    combos = [combo(range(nFeatures), d) for d in range(deg + 1)]

    nOutputFeatures = sum([len(list(c)) for c in combos])

    X = np.ones((nSamples, nOutputFeatures))

    col = 0
    for d in range(deg + 1):

        for comb in combo(range(nFeatures), d):

            if d == 0:
                pass

            else:
                X[:, col] = np.prod(x[:, comb], axis = 1)

            col += 1

    return X


def linearRegression(x, y, deg):

    x = polynomialFeatureTransform(x, deg)
    w = np.linalg.pinv(x.T @ x) @ x.T @ y

    return w


def plotResults(x, y, w, deg):

    plt.figure(figsize = (10, 8))
    sns.set_style('darkgrid')

    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker = '+', color = 'blue', label = '+1')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker = '_', color = 'red', label = '-1')

    xx, yy = np.meshgrid(np.linspace(np.min(x[:, 0]) - 1, np.max(x[:, 0]) + 1, 10000),
                         np.linspace(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1, 10000))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    GRID = polynomialFeatureTransform(grid, deg)

    z = np.sign(GRID @ w)
    z = z.reshape(xx.shape)

    plt.contour(xx, yy, z, levels = [0], colors = 'black')

    plt.title(f'Linear Regression with degree {deg} feature transform')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main() -> None:
    
    x, y = generateData(r = 10, thk = 5, sep = -0.5, xoff = 5, n = 2500)
    w = linearRegression(x, y, 3)

    plotResults(x, y, w, 3)

if __name__ == '__main__': main()
