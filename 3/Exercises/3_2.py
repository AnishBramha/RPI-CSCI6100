import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generateData(N, noiseRatio):

    a, b, c = np.random.randn(3)
    while b == 0 or np.abs(a / b) < 0.1 or np.abs(a / b) > 10:
        a, b = np.random.randn(2)

    x = np.random.randn(N, 2)

    y = np.sign((a * x[:, 0]) + (b * x[:, 1]) + c)

    flip_idx = np.random.choice(N, size = int(N * noiseRatio), replace = False)
    y[flip_idx] *= -1

    return x, y


def error(x, y, w):

    return np.mean(np.sign(x @ w) != y)

def pocket(x, y, T):

    x = np.hstack([np.ones((x.shape[0], 1)), x])

    N, d = x.shape

    w = np.zeros(d)
    w_star = w.copy()

    minError = error(x, y, w)

    errors = [minError]

    for t in range(1, T):

        misclassified = np.where(np.sign(x @ w) != y)[0]

        if len(misclassified) == 0:
            break

        i = np.random.choice(misclassified)

        w += y[i] * x[i]

        currentError = error(x, y, w)

        if currentError < minError:

            minError = currentError
            w_star = w.copy()

        errors.append(minError)

    while len(errors) < T:

        errors.append(errors[-1])

    return w_star, np.array(errors)


def experiment(num, NTrain, NTest, T):

    eIns, avgEIns, eOuts = [], [], []

    for _ in range(num):

        xTrain, yTrain = generateData(NTrain, 0.1)

        w, eIn = pocket(xTrain, yTrain, T)

        xTest, yTest = generateData(NTest, 0)

        eOut = error(np.hstack([np.ones((NTest, 1)), xTest]), yTest, w)

        avgEIns.append(eIn)
        eIns.append(eIn[-1])
        eOuts.append(eOut)

    return np.array(eIns), np.mean(avgEIns, axis = 0), np.array(eOuts)


def plotResults(eIn, avgEIn, eOut):

    sns.set_style('darkgrid')

    plt.figure(figsize = (15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(avgEIn, color = 'orange')
    plt.title('Average $E_{in}$ Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.grid(True)
    
    # Plot 2: Final in-sample error distribution
    plt.subplot(1, 3, 2)
    sns.histplot(eIn, kde = True, color = 'green')
    plt.title('Final $E_{in}$ Distribution')
    plt.xlabel('Error')
    plt.grid(True)
    
    # Plot 3: Out-of-sample error distribution
    plt.subplot(1, 3, 3)
    sns.histplot(eOut, kde = True, color = 'red')
    plt.title('$E_{out}$ Distribution')
    plt.xlabel('Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def main() -> None:

    eIn, avgEIn, eOut = experiment(20, 100, 1000, 1000)

    plotResults(eIn, avgEIn, eOut)


if __name__ == '__main__': main()
