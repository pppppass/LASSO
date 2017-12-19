import numpy

from utils import errfun

def loss_func(A, u, v, b, mu):
    x = (u - v) / 2.
    error = A.dot(x) - b
    loss = 1. / 2. * numpy.sum(error**2) + mu * numpy.sum(numpy.abs(x))
    return loss

def grad_func(n, A, u, v, b, mu):
    x = (u - v) / 2.
    error = A.dot(x) - b
    grad_u = 1. / 2. * A.transpose().dot(error) + 1. / 2. * mu * numpy.ones((n, 1))
    grad_v = -1. / 2. * A.transpose().dot(error) + 1. / 2. * mu * numpy.ones((n, 1))
    return grad_u, grad_v

def init(x0):
    u = 2. * numpy.maximum(x0, 0.)
    v = -2. * numpy.minimum(x0, 0.)
    return u, v

def proj(u, v):
    p_u = numpy.maximum(u, 0.)
    p_v = numpy.maximum(v, 0.)
    return p_u, p_v

def iteration(n, A, u, v, b, mu, lr):
    grad_u, grad_v = grad_func(n, A, u, v, b, mu)
    
    u = u - lr * grad_u
    v = v - lr * grad_v
    
    u, v = proj(u, v)
    
    return u, v, grad_u, grad_v

def l1_proj_grad(
    x0, A, b, mu,
    iter_list=[],
    lr_list=None,
    mu_list=None,
    res_list=None,
    sep=0, figure=False, xx=None, **opts
):
    m, n = A.shape
    
    mu0 = mu
    
    u, v = init(x0)
    
    t = 0
    iter_len = len(iter_list)
    
    formal_loss_list, real_loss_list, error_xx_list, grad_norm2_list = [], [], [], []
    
    for j in range(iter_len):
        lr = lr_list[j]
        mu = mu_list[j]
        for i in range(iter_list[j]):
            u, v, grad_u, grad_v = iteration(n, A, u, v, b, mu, lr)
            
            if figure:
                formal_loss_list.append(loss_func(A, u, v, b, mu))
                real_loss_list.append(loss_func(A, u, v, b, mu0))
                if xx is not None:
                    x = (u - v) / 2.
                    error_xx_list.append(errfun(x, xx))
            
            if sep != 0 and t % sep == 0:
                loss = loss_func(A, u, v, b, mu0)
                print("i: {0}, j: {1}, t: {2}, loss: {3:.5e}".format(i, j, t, loss))
            
            if res_list is not None:
                grad_norm2 = numpy.sum(grad_u**2) + numpy.sum(grad_v**2)
                if figure:
                    grad_norm2_list.append(grad_norm2)
                if grad_norm2 < res_list[j]:
                    break
            
            t += 1
    
    solution = (u - v) / 2.
    loss = loss_func(A, u, v, b, mu0)
    
    out = {
        "solution": solution,
        "loss": loss,
        "vars": 2*n,
        "iters": t,
        "conts": iter_len,
        "formal_loss": numpy.array(formal_loss_list),
        "real_loss": numpy.array(real_loss_list),
        "error": numpy.array(error_xx_list),
        "grad_norm2": numpy.array(grad_norm2_list),
    }
    
    return solution, out
