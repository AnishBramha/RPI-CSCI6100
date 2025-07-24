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


def perceptron(x, y, maxIt = 10000):

    x = np.hstack((np.ones((x.shape[0], 1)), x))
    w = np.zeros(x.shape[1])

    for _ in range(maxIt):

        misclassified = False

        for xi, yi in zip(x, y):

            if np.sign(w @ xi) != yi:

                w += yi * xi
                misclassified = True

        if not misclassified:
                break

    return w

def linearRegression(x, y):

    x = np.hstack((np.ones((x.shape[0], 1)), x))
    w = np.linalg.pinv(x.T @ x) @ x.T @ y

    return w

def plotResults(x, y, wPLA, wLR):

    plt.figure(figsize = (10, 8))
    sns.set_style('darkgrid')

    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker = '+', color = 'blue', label = '+1')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker = '_', color = 'red', label = '-1')

    xVals = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 500)

    plt.plot(xVals, -1 * (wPLA[0] + (wPLA[1] * xVals)) / wPLA[2], color = 'black', label = 'Perceptron')
    plt.plot(xVals, -1 * (wLR[0] + (wLR[1] * xVals)) / wLR[2], color = 'green', label = 'Linear Regression')

    plt.title('Perceptron versus Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    
    x, y = generateData(r = 10, thk = 5, sep = 1, xoff = 5, n = 2500)
    wPLA = perceptron(x, y)
    wLR = linearRegression(x, y)

    plotResults(x, y, wPLA, wLR)

if __name__ == '__main__': main()
