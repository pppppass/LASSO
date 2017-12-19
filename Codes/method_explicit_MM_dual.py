import numpy

from utils import errfun

def loss_func(A, xi, b, mu):
    x = -xi
    error = A.dot(x) - b
    loss = 1. / 2. * numpy.sum(error**2) + mu * numpy.sum(numpy.abs(x))
    return loss

def reconstruct(xi):
    x = -xi
    return x

def init(A, x0, b, mu):
    x = x0
    lam = A.dot(x) - b
    nu = A.transpose().dot(lam)
    xi = -x
    gamma = 1.618
    return lam, nu, xi, gamma

def update_inv(m, A, lr):
    inv = numpy.linalg.inv(numpy.eye(m) + lr * A.dot(A.transpose()))
    return inv

def iteration_ALM(A, b, inv, lam, nu, xi, mu, gamma, lr):
    lam_t = -b
    nu_t = A.transpose().dot(lam) + xi / lr
    
    nu = numpy.minimum(numpy.maximum(nu_t, -mu), +mu)
    lam = inv.dot(lr * A.dot(nu) - b - A.dot(xi))
    
    xi = xi + lr * (A.transpose().dot(lam) - nu)
    
    return lam, nu, xi

def iteration_ADMM(A, b, inv, lam, nu, xi, mu, gamma, lr):
    lam = inv.dot(lr * A.dot(nu) - b - A.dot(xi))
    
    nu = A.transpose().dot(lam) + xi / lr
    nu = numpy.minimum(numpy.maximum(nu, -mu), +mu)
    
    xi = xi + gamma * lr * (A.transpose().dot(lam) - nu)
    
    return lam, nu, xi

def l1_explicit_MM_dual(
    x0, A, b, mu,
    iter_func=None,
    iter_list=[],
    lr_list=None,
    mu_list=None,
    sep=0, figure=False, xx=None, **opts
):
    m, n = A.shape
    
    mu0 = mu
    
    lam, nu, xi, gamma = init(A, x0, b, mu)
    
    t = 0
    iter_len = len(iter_list)
    
    formal_loss_list, real_loss_list, error_xx_list = [], [], []
    
    for j in range(iter_len):
        lr = lr_list[j]
        inv = update_inv(m, A, lr)
        mu = mu_list[j]
        for i in range(iter_list[j]):
            lam, nu, xi = iter_func(A, b, inv, lam, nu, xi, mu, gamma, lr)
            
            if figure:
                formal_loss_list.append(loss_func(A, xi, b, mu))
                real_loss_list.append(loss_func(A, xi, b, mu0))
                if xx is not None:
                    x = reconstruct(xi)
                    error_xx_list.append(errfun(x, xx))
            
            if sep != 0 and t % sep == 0:
                loss = loss_func(A, xi, b, mu0)
                print("i: {0}, j: {1}, t: {2}, loss: {3:.5e}".format(i, j, t, loss))
                
            t += 1
    
    solution = reconstruct(xi)
    loss = loss_func(A, xi, b, mu0)
    
    out = {
        "solution": solution,
        "loss": loss,
        "vars": 2*n + m,
        "iters": t,
        "conts": iter_len,
        "formal_loss": numpy.array(formal_loss_list),
        "real_loss": numpy.array(real_loss_list),
        "error": numpy.array(error_xx_list),
    }
    
    return solution, out

def l1_ALM_dual(x0, A, b, mu, **opts):
    return l1_explicit_MM_dual(
        x0, A, b, mu,
        iter_func=iteration_ALM,
        **opts
    )

def l1_ADMM_dual(x0, A, b, mu, **opts):
    return l1_explicit_MM_dual(
        x0, A, b, mu,
        iter_func=iteration_ADMM,
        **opts
    )
