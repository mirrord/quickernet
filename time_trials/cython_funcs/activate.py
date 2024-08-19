
import cupy as cp


def softmax(inputs):
    exp = cp.exp(inputs - cp.max(inputs, axis=1, keepdims=True))
    return exp / cp.sum(exp, axis=1, keepdims=True)

def sigmoid(inputs):
    1 / (1 + cp.exp(-inputs))

def relu(inputs):
    return cp.maximum(0, inputs)

def leaky_relu(inputs):
    return cp.maximum(0.01*inputs, inputs)

def tanh(inputs):
    return cp.tanh(inputs)

def swish(inputs):
    return inputs*(1 / (1 + cp.exp(-inputs)))

def scale(inputs):
    return inputs/inputs.shape[1]

def norm(inputs):
    return inputs/cp.max(inputs,axis=1,keepdims=True)

def l2norm(inputs):
    return inputs/cp.linalg.norm(inputs,axis=1,keepdims=True)
