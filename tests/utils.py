import cupy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    sigx = sigmoid(x)
    return sigx * (1 - sigx)


def diff_squares(y, y_true):
    return np.sum(np.square(y - y_true)).item()


def ddiff_squares(y, y_true):
    return 2 * (y - y_true)


def classic_net_predict(weights, biases, input):
    for b, w in zip(biases, weights):
        input = sigmoid(np.dot(input, w) + b)
    return input


def classic_single_node_backprop(weights, biases, input, err_grad):
    zs = np.dot(input, weights) + biases

    delta = err_grad * dsigmoid(zs)

    nabla_b = delta
    nabla_w = np.dot(input.T, delta)
    return (nabla_b, nabla_w), np.dot(delta, weights.T)


def classic_net_backprop(weights, biases, input, exp_out):
    activations = [input]  # list to store all the activations, layer by layer
    zs = []  # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        zs.append(np.dot(activations[-1], w) + b)
        activations.append(sigmoid(zs[-1]))

    # error calc
    delta = ddiff_squares(activations[-1], exp_out)

    # gradient calc
    delta = delta * dsigmoid(zs[-1])
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(activations[-2].T, delta)
    gradients = [delta]
    for layer_idx in range(2, len(weights) + 1):
        nabla_b[-layer_idx] = np.dot(
            nabla_b[-layer_idx + 1], weights[-layer_idx + 1].T
        ) * dsigmoid(zs[-layer_idx])
        nabla_w[-layer_idx] = np.dot(activations[-layer_idx - 1].T, nabla_b[-layer_idx])
        gradients.insert(0, nabla_b[-layer_idx])
    return (nabla_b, nabla_w), gradients
