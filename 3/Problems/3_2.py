import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

maxIt = 10000

def generateData(r, thk, sep, n):

    rin = r
    rout = r + thk

    def generateSemiCircle(top):

        radii = np.sqrt(np.random.uniform(rin ** 2, rout ** 2, n))
        angles = np.random.uniform(0, np.pi, n) if top else np.random.uniform(np.pi, np.pi * 2, n)

        x = radii * np.cos(angles)
        y = radii * np.sin(angles)

        y += (sep / 2) if top else -(sep / 2)

        return np.column_stack((x, y))
    
    plus = generateSemiCircle(top = True)
    minus = generateSemiCircle(top = False)

    x = np.vstack((plus, minus))
    y = np.hstack((np.ones(n), -np.ones(n)))

    return x, y


def perceptron(x, y):

    x = np.hstack((np.ones((x.shape[0], 1)), x))
    w = np.zeros(x.shape[1])

    it = 0

    for _ in range(maxIt):

        misclassified = False

        for xi, yi in zip(x, y):

            if np.sign(w @ xi) != yi:

                w += yi * xi
                misclassified = True
                it += 1

        if not misclassified:
            break
            
    return it


def runExperiment(r, thk, n):

    seps = np.round(np.arange(0.2, 5.1, 0.2), 1)

    it = []

    for sep in seps:

        x, y = generateData(r, thk, sep, n)
        it.append(perceptron(x, y))

    return seps, np.array(it)


def plotResults(sep, it):

    plt.figure(figsize = (10, 8))
    sns.set_style('darkgrid')

    plt.plot(sep, it, marker = 'o', color = 'black', linestyle = '-', label = 'Perceptron Iterations')

    xVals = np.linspace(0.2, 5)
    # plt.plot(xVals, np.array([maxIt] * xVals.shape[0]), color = 'red', linestyle = '-', label = 'Maximum Iterations')
    
    plt.xlabel('Separation between discs')
    plt.ylabel('Perceptron Iterations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    
    sep, it = runExperiment(r = 10, thk = 5, n = 1000)

    plotResults(sep, it)

if __name__ == '__main__': main()
