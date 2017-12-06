import math
import time
import numpy
import matplotlib.pyplot as plt

def sprandn(m, n, density):
    ret = numpy.random.randn(m, n)
    flag = numpy.random.uniform(0., 1., size=(m, n))
    ret[flag > density] = 0.
    return ret

def generate_data(n=1024, m=512, mu=1.e-3):
    A = numpy.random.randn(m, n)
    u = sprandn(n, 1, 0.1)
    b = A.dot(u)

    x0 = numpy.random.rand(n, 1)
    
    return n, m, A, u, b, mu, x0

def errfun(x1, x2):
    return numpy.linalg.norm(x1 - x2) / (1. + numpy.linalg.norm(x1))

class Tester:
    def __init__(self, n, m, A, u, b, mu, x0):
        self.n = n
        self.m = m
        self.A = A
        self.u = u
        self.b = b
        self.mu = mu
        self.x0 = x0
    
    def set_xx(self, func):
        self.xx, _ = func(self.x0, self.A, self.b, self.mu)
    
    
    def test(self, func, options={}):
        start_time = time.time()
        value, out = func(self.x0, self.A, self.b, self.mu, options)
        end_time = time.time()
        elapsed = end_time - start_time

        error = self.A.dot(value) - self.b
        check_loss = 1. / 2. * numpy.sum(error**2) + self.mu * numpy.sum(numpy.abs(value))
        approximation_loss = 1. / 2. * numpy.sum(error**2)
        regularization = numpy.sum(numpy.abs(value))

        out["name"] = func.__name__
        out["value"] = value
        out["time"] = elapsed
        out["check_loss"] = check_loss
        out["approximation_loss"] = approximation_loss
        out["regularization"] = regularization
        out["error_m"] = errfun(self.xx, value)
        out["error_g"] = errfun(self.u, value)

        print(
    """The function {name} is executed:
        cpu: {time:.5f} (setup: {setup_time:.5f}, solve: {solve_time:.5f}),
        vars: {num_var}, iters: {iters},
        loss: {loss:.5e} = check: {check_loss:.5e} (approx: {approximation_loss:5e} + mu*reg: {regularization:5e})
        error to known xx: {error_m:.5e}
        error to ground-truth: {error_g:.5e}""".format(**out)
        )

        return out

def draw_sparse_figure(value):
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(value, bins=30)

    fig.tight_layout()
    plt.show()
    
    return None

def draw_loss_curve(array, label=None, log=False):
    if label is not None:
        for arr, lab in zip(array, label):
            if log:
                plt.plot(numpy.log(arr), label=lab)
            else:
                plt.plot(arr, label=lab)
        plt.legend()
    else:
        for arr in array:
            plt.plot(arr)

    plt.show()
    
    return None

def mu_func_expo_dimish(start_mu, mu, prepare):
    """
    Return a function to modify mu, with exponentially dimishing strategy.
    
    Arguments:
        x0 (float): the starting mu
        mu (float): the ending mu
        prepare (int): an integer indicating the time of dimishment
    
    Returns:
        a function mu_func, which valued start_mu at 0, mu at and after prepare, and exponentially
        dimishing between 0 and prepare
    """
    def mu_func(i):
        if i < prepare:
            return math.pow(10., (math.log10(start_mu) - (math.log10(start_mu) - math.log10(mu)) * i / prepare))
        else:
            return mu
    return mu_func

def lr_func_frac_dimish(coefficient, offset):
    """
    Return a function to modify learning rate, with fractional dimishing strategy.
    
    Arguments:
        coefficient (float): the coefficient
        offset (int): the offset of the linear fractional
    
    Returns:
        a function lr_func, which valued coefficient / (x + offset) at x
    """
    def lr_func(i):
        return coefficient / (i + offset)
    
    return lr_func