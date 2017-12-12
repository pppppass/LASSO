import numpy

from utils import errfun

def sqrt_smooth(x, eps):
    return numpy.sqrt(x**2 + eps**2) - eps

def log_exp_smooth(x, eps):
    return eps * (numpy.logaddexp(x / eps - numpy.log(2.), -x / eps - numpy.log(2.)))

def no_smooth(x, eps):
    return numpy.abs(x)

def sqrt_smooth_grad(x, eps):
    return x / numpy.sqrt(x**2 + eps**2)

def log_exp_smooth_grad(x, eps):
    return numpy.tanh(x / eps)

def loss_func(A, x, b, mu, smooth_func, eps):
    error = A.dot(x) - b
    loss = 1. / 2. * numpy.sum(error**2) + mu * numpy.sum(smooth_func(x, eps))
    return loss

def grad_func(A, x, b, mu, smooth_grad, eps):
    error = A.dot(x) - b
    grad_x = A.transpose().dot(error) + mu * smooth_grad(x, eps)
    return grad_x

def init(x0):
    x = x0
    return x

def iteration(A, x, b, mu, lr, smooth_grad, eps):
    grad_x = grad_func(A, x, b, mu, smooth_grad, eps)
    
    x = x - lr * grad_x
    
    return x, grad_x

def l1_smooth_grad(
    x0, A, b, mu,
    smooth_func=None, smooth_grad=None,
    iter_list=[],
    lr_list=None,
    mu_list=None,
    eps_list=None,
    res_list=None,
    sep=0, figure=False, xx=None, **opts
):
    m, n = A.shape
    
    mu0 = mu
    
    x = init(x0)
    
    t = 0
    iter_len = len(iter_list)
    
    formal_loss_list, real_loss_list, orig_loss_list, error_xx_list, grad_norm2_list = [], [], [], [], []
    
    for j in range(iter_len):
        if lr_list is not None:
            lr = lr_list[j]
        if mu_list is not None:
            mu = mu_list[j]
        if eps_list is not None:
            eps = eps_list[j]
        for i in range(iter_list[j]):
            x, grad_x = iteration(A, x, b, mu, lr, smooth_grad, eps)
            
            if figure:
                formal_loss_list.append(loss_func(A, x, b, mu, smooth_func, eps))
                real_loss_list.append(loss_func(A, x, b, mu0, smooth_func, eps))
                orig_loss_list.append(loss_func(A, x, b, mu0, no_smooth, eps))
                if xx is not None:
                    error_xx_list.append(errfun(x, xx))
            
            if sep != 0 and t % sep == 0:
                loss = loss_func(A, x, b, mu0, smooth_func, eps)
                print("i: {0}, j: {1}, t: {2}, loss: {3:.5e}".format(i, j, t, loss))
            
            if res_list is not None:
                grad_norm2 = numpy.sum(grad_x**2)
                if figure:
                    grad_norm2_list.append(grad_norm2)
                if grad_norm2 < res_list[j]:
                    break
                
            t += 1
    
    solution = x
    loss = loss_func(A, x, b, mu0, smooth_func, eps)
    
    out = {
        "solution": solution,
        "loss": loss,
        "vars": n,
        "iters": t,
        "formal_loss": numpy.array(formal_loss_list),
        "real_loss": numpy.array(real_loss_list),
        "orig_loss": numpy.array(orig_loss_list),
        "error": numpy.array(error_xx_list),
        "grad_norm2": numpy.array(grad_norm2_list),
    }
    
    return solution, out

def l1_smooth_grad_sqrt(x0, A, b, mu, **opts):
    return l1_smooth_grad(
        x0, A, b, mu,
        smooth_func=sqrt_smooth, smooth_grad=sqrt_smooth_grad,
        **opts
    )

def l1_smooth_grad_log_exp(x0, A, b, mu, **opts):
    return l1_smooth_grad(
        x0, A, b, mu,
        smooth_func=log_exp_smooth, smooth_grad=log_exp_smooth_grad,
        **opts
    )

def iteration_fast(A, x, x_1, b, mu, lr, smooth_grad, eps, i):
    y = x + (i - 1.) / (i + 2.) * (x - x_1)
    x_1 = x
    
    grad_y = grad_func(A, y, b, mu, smooth_grad, eps)
    
    x = y - lr * grad_y 
    return x, x_1, grad_y

def l1_fast_smooth_grad(
    x0, A, b, mu,
    smooth_func=None, smooth_grad=None,
    iter_list=[],
    lr_list=None,
    mu_list=None,
    eps_list=None,
    res_list=None,
    sep=0, figure=False, xx=None, **opts
):
    m, n = A.shape
    
    mu0 = mu
    
    x = init(x0)
    x_1 = x
    
    t = 0
    iter_len = len(iter_list)
    
    formal_loss_list, real_loss_list, orig_loss_list, error_xx_list, grad_norm2_list = [], [], [], [], []
    
    for j in range(iter_len):
        if lr_list is not None:
            lr = lr_list[j]
        if mu_list is not None:
            mu = mu_list[j]
        if eps_list is not None:
            eps = eps_list[j]
        for i in range(iter_list[j]):
            x, x_1, grad_y = iteration_fast(A, x, x_1, b, mu, lr, smooth_grad, eps, i)
            
            if figure:
                formal_loss_list.append(loss_func(A, x, b, mu, smooth_func, eps))
                real_loss_list.append(loss_func(A, x, b, mu0, smooth_func, eps))
                orig_loss_list.append(loss_func(A, x, b, mu0, no_smooth, eps))
                if xx is not None:
                    error_xx_list.append(errfun(x, xx))
            
            if sep != 0 and t % sep == 0:
                loss = loss_func(A, x, b, mu0, smooth_func, eps)
                print("i: {0}, j: {1}, t: {2}, loss: {3:.5e}".format(i, j, t, loss))
            
            if res_list is not None:
                grad_norm2 = numpy.sum(grad_y**2)
                if figure:
                    grad_norm2_list.append(grad_norm2)
                if grad_norm2 < res_list[j]:
                    break
                
            t += 1
    
    solution = x
    loss = loss_func(A, x, b, mu0, smooth_func, eps)
    
    out = {
        "solution": solution,
        "loss": loss,
        "vars": n,
        "iters": t,
        "formal_loss": numpy.array(formal_loss_list),
        "real_loss": numpy.array(real_loss_list),
        "orig_loss": numpy.array(orig_loss_list),
        "error": numpy.array(error_xx_list),
        "grad_norm2": numpy.array(grad_norm2_list),
    }
    
    return solution, out

def l1_fast_smooth_grad_sqrt(x0, A, b, mu, **opts):
    return l1_fast_smooth_grad(
        x0, A, b, mu,
        smooth_func=sqrt_smooth, smooth_grad=sqrt_smooth_grad,
        **opts
    )

def l1_fast_smooth_grad_log_exp(x0, A, b, mu, **opts):
    return l1_fast_smooth_grad(
        x0, A, b, mu,
        smooth_func=log_exp_smooth, smooth_grad=log_exp_smooth_grad,
        **opts
    )

