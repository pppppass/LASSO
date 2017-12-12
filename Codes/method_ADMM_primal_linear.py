import numpy

from utils import errfun

def loss_func(A, x, b, mu):
    error = A.dot(x) - b
    loss = 1. / 2. * numpy.sum(error**2) + mu * numpy.sum(numpy.abs(x))
    return loss

def grad_func(A, x, b, y, lam, lr):
    grad_lin = lr * A.transpose().dot(A.dot(x) - b - y) + A.transpose().dot(lam)
    return grad_lin

def linear_update_x(x, g, mu, tau):
    shrink_positive = numpy.maximum(x - tau * g - tau * mu, 0.)
    shrink_negative = numpy.minimum(x - tau * g + tau * mu, 0.)
    return shrink_positive + shrink_negative

def init(A, x0, b):
    x = x0
    y = A.dot(x) - b
    lam = y
    gamma = 1.618
    return x, y, lam, gamma

def iteration(A, x, b, y, lam, mu, gamma, lr, tau):
    g = grad_func(A, x, b, y, lam, lr)
    x = linear_update_x(x, g, mu, tau)
    
    y = (lam - lr * (b - A.dot(x))) / (1. + lr)
    
    lam = lam + gamma * lr * (A.dot(x) - b - y)
    
    return x, y, lam

def l1_ADMM_primal_linear(
    x0, A, b, mu,
    iter_list=[],
    lr_list=None,
    mu_list=None,
    tau_list=None,
    sep=0, figure=False, xx=None, **opts
):
    m, n = A.shape
    
    mu0 = mu
    
    x, y, lam, gamma = init(A, x0, b)
    
    t = 0
    iter_len = len(iter_list)
    
    formal_loss_list, real_loss_list, error_xx_list = [], [], []
    
    for j in range(iter_len):
        if lr_list is not None:
            lr = lr_list[j]
        if mu_list is not None:
            mu = mu_list[j]
        if tau_list is not None:
            tau = tau_list[j]
        for i in range(iter_list[j]):
            x, y, lam = iteration(A, x, b, y, lam, mu, gamma, lr, tau)
            
            if figure:
                formal_loss_list.append(loss_func(A, x, b, mu))
                real_loss_list.append(loss_func(A, x, b, mu0))
                if xx is not None:
                    error_xx_list.append(errfun(x, xx))
            
            if sep != 0 and t % sep == 0:
                loss = loss_func(A, x, b, mu0)
                print("i: {0}, j: {1}, t: {2}, loss: {3:.5e}".format(i, j, t, loss))
                
            t += 1
    
    solution = x
    loss = loss_func(A, x, b, mu0)
    
    out = {
        "solution": solution,
        "loss": loss,
        "vars": 2*n + m,
        "iters": t,
        "formal_loss": numpy.array(formal_loss_list),
        "real_loss": numpy.array(real_loss_list),
        "error": numpy.array(error_xx_list),
    }
    
    return solution, out
