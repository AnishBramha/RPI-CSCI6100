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


def pocket(x, y, deg, maxIt = 1000):

    x = polynomialFeatureTransform(x, deg)
    w = np.zeros(x.shape[1])
    w_star = w.copy()

    minEin = float('inf')
    Eins = []
    t = []
    tTotal = 0

    for _ in range(maxIt):

        it = 0

        misclassified = False

        for xi, yi in zip(x, y):

            if np.sign(xi @ w) != yi:

                w += yi * xi
                misclassified = True
                it += 1

        currEin = np.mean(np.sign(x @ w) != y)

        if currEin < minEin:

            minEin = currEin
            w_star = w.copy()

        tTotal += it
        t.append(tTotal)
        Eins.append(minEin)

        if not misclassified:
            break

    return w_star, np.array(Eins), np.array(t)

def plotResults(x, y, wP, Ein, t, deg):

    plt.figure(figsize = (15, 6))
    sns.set_style('darkgrid')

    plt.subplot(1, 2, 1)
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker = '+', color = 'blue', label = '+1')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker = '_', color = 'red', label = '-1')

    xx, yy = np.meshgrid(np.linspace(np.min(x[:, 0]) - 1, np.max(x[:, 0]) + 1, 1000),
                         np.linspace(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1, 1000))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    GRID = polynomialFeatureTransform(grid, deg)

    z = np.sign(GRID @ wP)
    z = z.reshape(xx.shape)

    plt.contour(xx, yy, z, levels = [0], colors = 'black')

    plt.title(f'Pocket Algorithm with degree {deg} feature transform')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t, Ein, color = 'blue', label = 'Ein')
    plt.legend()
    plt.xlabel('Iterations (t)')
    plt.ylabel('In-sample Error')
    plt.title('E$_{in}$ vs. t')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main() -> None:
    
    x, y = generateData(r = 10, thk = 5, sep = -5, xoff = 5, n = 1000)
    wP, Ein, t = pocket(x, y, 3)

    plotResults(x, y, wP, Ein, t, 3)

if __name__ == '__main__': main()
