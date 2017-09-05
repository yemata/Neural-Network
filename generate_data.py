import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Generate a data set and plot it
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_blobs(200, centers=2)
    return X, y

def visualize(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title("Sklearn Dataset")
    plt.show()

def main():
    X, y = generate_data()
    visualize(X, y)

if __name__ == '__main__':
    main()