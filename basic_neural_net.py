import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Gradient descent parameters
epsilon = 0.01 # learning rate
reg_lambda = 0.01 # regularization strength

def generate_data(nn_output_dim, nn_input_dim=2, n_samples=100):
    np.random.seed(0)
    type = input("Please select one of the following data sets:\n"
                       + "\t1. moons\n"
                       + "\t2. blobs\n"
                       + "\t3. random\n")

    X, y = {}, {}

    if type == 1:
        if nn_output_dim == 2:
            X, y = datasets.make_moons(n_samples, noise=0.1)
    elif type == 2:
        X, y = datasets.make_blobs(n_samples, centers=nn_output_dim)
    elif type == 3:
        X, y = generate_random_data(nn_output_dim, nn_input_dim, n_samples)

    return X, y

def generate_random_data(n_classes, n_features, n_samples):
    max_classes = n_features * 2
    if (n_classes > max_classes):
        return {}, {}

    n_clusters_per_class = max_classes // n_classes

    n_informative = n_features
    n_redundant = 0
    n_repeated = 0

    return datasets.make_classification(n_samples, n_features, n_informative, n_redundant, n_repeated,
                                        n_classes, n_clusters_per_class)
# Helper function to evaluate the total loss on the data set
def calculate_loss(model, X, y):
    N = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    # Soft max
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Loss function
    correct_logprobs = -np.log(probs[range(N), y]) # TODO: change to understandable form
    data_loss = np.sum(correct_logprobs)
    return 1. / N * data_loss

# Helper function to predict an output
def predict(model, x, print_raw=False):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    # Soft max
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    if print_raw:
        print "Raw prediction:"
        print probs.round(decimals=2)

    return np.argmax(probs, axis=1)

def initialize_model(nn_hdim, nn_output_dim, nn_input_dim=2):
    print "Initializing network parameters..."
    # Initialize parameters to random values
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

# Learn parameters for the neural network and return model
# - nn_hdim: number of nodes in hidden layer
# - num_passes: number of passes through the training data for gradient descent
# - print_loss: if true, print loss every 1000 iterations
def build_model(model, X, y, num_passes=20000, print_loss=False):
    print "Training neural network..."

    N = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    print "Performing gradient descent..."
    # Gradient descent
    for i in xrange(0, num_passes):
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2

        # Soft max
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Back propagation
        delta3 = probs
        delta3[range(N), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # add regularization terms
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # assign new parameter values
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # optionally print loss
        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" %(i, calculate_loss(model, X, y))

    print "Training complete!"

    return model

# Helper function to plot a decision boundary
def plot_decision_boundary(pred_func, X, y):

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.022

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def display_decision_boundary(model, X, y):
    plt.close()
    plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.title("Single-Layer Network Size-%d" % len(model))
    plt.show(False)

def classify_single_point(model):
    x_data = input("Please enter x-coordinate: ")
    y_data = input("Please enter y-coordinate: ")

    data = np.array([x_data, y_data])
    prediction = predict(model, data, True)
    translation = "male" if prediction == 1 else "female"

    print "Data has been identified as %s" % translation

def prompt_init_data():
    nn_output_dim = input("Please select the number of classes to classify: ")
    nn_hdim = input("Please select the dimensionality of the hidden layer: ")

    return nn_output_dim, nn_hdim

def validate_dataset(X):
    if len(X) < 2:
        print "Invalid data set! Exiting program."
        return False

    return True

def main():
    nn_output_dim, nn_hdim = prompt_init_data()

    X, y = generate_data(nn_output_dim)
    model = initialize_model(nn_hdim, nn_output_dim)

    while True:
        if not validate_dataset(X):
            break

        option = input("Please select an option:\n"
                       + "\t0. Exit program\n"
                       + "\t1. Train neural network\n"
                       + "\t2. Display prediction boundaries\n"
                       + "\t3. Classify single data point\n")

        if option == 0:
            break
        elif option == 1:
            model = build_model(model, X, y)
        elif option == 2:
            display_decision_boundary(model, X, y)
        elif option == 3:
            classify_single_point(model)


if __name__ == '__main__':
    main()