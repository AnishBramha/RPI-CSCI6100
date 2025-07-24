import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

    x = np.hstack((np.ones((x.shape[0], 1)), x))
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

def plotResults(x, y, wP, Ein, t):

    plt.figure(figsize = (15, 6))
    sns.set_style('darkgrid')

    plt.subplot(1, 2, 1)
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker = '+', color = 'blue', label = '+1')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker = '_', color = 'red', label = '-1')

    xVals = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 500)
    plt.plot(xVals, -(wP[0] + (xVals * wP[1])) / wP[2], color = 'black', label = 'Pocket')

    plt.title('Pocket Algorithm')
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
    wP, Ein, t = pocket(x, y)

    plotResults(x, y, wP, Ein, t)

if __name__ == '__main__': main()
