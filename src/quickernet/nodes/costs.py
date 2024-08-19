import cupy as np

from .optimization import OptimizableFunction


class CostFunction(OptimizableFunction):
    def __call__(self, inputs, expected_output):
        return self.forward(inputs, expected_output) / inputs.shape[0]

    def forward(self, inputs, expected_output):
        raise NotImplementedError

    def backward(self, error_gradient, inputs, expected_output):
        raise NotImplementedError

#NOTE: Cost functions usually need to divide by the batch size 
# when calculating cost

#NOTE: there seems to be some disagreement over which cost function
# is quadratic vs diffsquares
class QuadraticCost(CostFunction):
    def forward(self, inputs, expected_output):
        return 0.5 * np.sum(np.square(inputs - expected_output)).item()

    def backward(self, inputs, expected_output):
        return inputs - expected_output

class DiffSquaresCost(CostFunction):
    def forward(self, inputs, expected_output):
        return np.sum(np.square(inputs-expected_output)).item()
    def backward(self, inputs, expected_output):
        return 2*(inputs-expected_output)

class CrossEntropyCost(CostFunction):
    def forward(self, inputs, expected_output):
        return -1*np.sum( expected_output*np.log(inputs) + (1-expected_output)*np.log(1-inputs) ).item()
    def backward(self, inputs, expected_output):
        return (inputs-expected_output)/(inputs*(1-inputs))

class HellingerCost(CostFunction):
    def forward(self, inputs, expected_output):
        return 0.70710678*np.sum( np.square(np.sqrt(inputs)-np.sqrt(expected_output)) ).item()
    def backward(self, inputs, expected_output):
        rooty = np.sqrt(inputs)
        return (rooty - np.sqrt(expected_output))/(1.41421356*rooty)
