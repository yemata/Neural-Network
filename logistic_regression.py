import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegressionCV

# Generate a data set and plot it
def generate_data():
    np.random.seed(0)
    X, y = make_moons(200, noise=0.20)
    return X, y

def visualize(X, y, clf):
    plot_decision_boundary(clf.predict, X, y)
    plt.title("Logistic Regression")
    plt.show()

# Helper function to plot a decision boundary
def plot_decision_boundary(pred_func, X, y):

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# Train the logistic regression classifier
def classify(X, y):
    clf = LogisticRegressionCV()
    clf.fit(X, y)
    return clf

def main():
    X, y = generate_data()
    clf = classify(X, y)
    visualize(X, y , clf)

if __name__ == '__main__':
    main()