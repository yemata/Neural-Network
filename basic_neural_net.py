import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)
X, y = datasets.make_blobs(300, centers=3)

N = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 3 # output layer dimensionality

# Gradient descent parameters
epsilon = 0.01 # learning rate
reg_lambda = 0.01 # regularization strength

# Helper function to evaluate the total loss on the data set
def calculate_loss(model):
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

# Learn parameters for the neural network and return model
# - nn_hdim: number of nodes in hidden layer
# - num_passes: number of passes through the training data for gradient descent
# - print_loss: if true, print loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):

    print "Initializing network parameters..."
    # Initialize parameters to random values
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    print "Initial network parameters:"
    print "First layer weights\n", W1,\
        "\nFirst layer biases\n", b1,\
        "\nSecond layer weights\n", W2,\
        "\nSecond layer biases\n", b2

    model = {}

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
            print "Loss after iteration %i: %f" %(i, calculate_loss(model))

    print "Final network parameters:"
    print "First layer weights\n", W1, \
        "\nFirst layer biases\n", b1, \
        "\nSecond layer weights\n", W2, \
        "\nSecond layer biases\n", b2

    return model

# Helper function to plot a decision boundary
def plot_decision_boundary(pred_func):

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

def main():
    nn_hdim = 4

    print "Training single-layer %i dimensional network...\n" %nn_hdim
    model = build_model(nn_hdim)

    option = input ("Training complete!\n\n"
                    "Please select next option:\n"
                    + "1. Classify single data point\n"
                    + "2. Display prediction boundaries\n")

    if option == 1:
        x_data = input("Please enter x-coordinate: ")
        y_data = input("Please enter y-coordinate: ")

        data = np.array([x_data, y_data])
        prediction = predict(model, data, True)
        translation = "male" if prediction==1 else "female"

        print "Data has been identified as %s" %translation

    if option == 2:
        plot_decision_boundary(lambda x: predict(model, x))
        plt.title("Single-Layer Network Size-%d" % nn_hdim)
        plt.show()

if __name__ == '__main__':
    main()