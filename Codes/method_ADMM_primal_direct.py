import numpy

from utils import errfun

def loss_func(A, x, b, mu):
    error = A.dot(x) - b
    loss = 1. / 2. * numpy.sum(error**2) + mu * numpy.sum(numpy.abs(x))
    return loss

def update_x(x, y, lam, mu, lr):
    shrink_positive = numpy.maximum((-mu - lam) / lr + y, 0.)
    shrink_negative = numpy.minimum((mu - lam) / lr + y, 0.)
    return shrink_positive + shrink_negative

def init(A, x0, b):
    x = x0
    y = x0
    lam = A.transpose().dot(A.dot(y) - b)
    gamma = 1.618
    return x, y, lam, gamma

def update_inv(n, A, lr):
    inv = numpy.linalg.inv(A.transpose().dot(A) + lr * numpy.eye(n))
    return inv

def iteration(A, x, b, y, inv, lam, mu, gamma, lr):
    x = update_x(x, y, lam, mu, lr)
    
    y = inv.dot(A.transpose().dot(b) + lam + lr * x)
    
    lam = lam + gamma * lr * (x - y)
    
    return x, y, lam

def l1_ADMM_primal_direct(
    x0, A, b, mu,
    iter_list=[],
    lr_list=None,
    mu_list=None,
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
            inv = update_inv(n, A, lr)
        if mu_list is not None:
            mu = mu_list[j]
        for i in range(iter_list[j]):
            x, y, lam = iteration(A, x, b, y, inv, lam, mu, gamma, lr)
            
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
        "vars": 3*n,
        "iters": t,
        "formal_loss": numpy.array(formal_loss_list),
        "real_loss": numpy.array(real_loss_list),
        "error": numpy.array(error_xx_list),
    }
    
    return solution, out
