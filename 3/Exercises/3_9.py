import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def classificationError(s, y):

    return y != np.sign(s)

def squaredError(s, y):

    return (y - s) ** 2

def logarithmicError(s, y):

    return np.log(1.0 + np.exp(-1 * y * s))


def plotGraphs():

    ss = np.arange(-10, 10, 0.1)
    eClass = classificationError(ss, 1)
    eSq = squaredError(ss, 1)
    eLog = (1.0 * logarithmicError(ss, 1)) / np.log(2)

    sns.set_style('darkgrid')
    plt.figure(figsize = (8, 6))

    plt.plot(ss, eClass, label = 'Classification Error', color = 'black')
    plt.plot(ss, eSq, label = 'Square Error', color = 'blue')
    plt.plot(ss, eLog, label = 'Logarithmic Error', color = 'red')

    plt.xlabel('(Signal) s = w$^T$x')
    plt.ylabel('Error')
    plt.ylim(-0.1, 2)
    plt.title('Error types with bounds')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:

    plotGraphs()

if __name__ == '__main__': main()
