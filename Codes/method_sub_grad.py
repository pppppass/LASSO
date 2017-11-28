"""
Solve lasso regularized least square by sub-gradient method.
"""

import math
import numpy

def l1_sub_grad(x0, A, b, mu, options):
    m, n = A.shape
    
    def loss_func(x, mu):
        error = A.dot(x) - b
        loss = 1. / 2. * numpy.sum(error**2) + mu * numpy.sum(numpy.abs(x))
        return loss
    
    def grad(x, mu):
        error = A.dot(x) - b
        grad_x = A.transpose().dot(error) + mu * numpy.sign(x)
        return grad_x

    iters = options["iters"]
    sep = options["sep"]
    mutable_learn_rate = options["mutable_learn_rate"]
    mutable_regularization = options["mutable_regularization"]
    
    x = x0
    if mutable_learn_rate:
        lr_func = options["lr_func"]
    else:
        lr = options["init_lr"]
    if mutable_regularization:
        mu_func = options["mu_func"]
    else:
        now_mu = mu
    
    i_count = 0
    
    for i in range(iters):
        i_count += 1
    
        real_loss = loss_func(x, mu)
        
        if mutable_learn_rate:
            lr = lr_func(i)
        if mutable_regularization:
            now_mu = mu_func(i)
        
        grad_x = grad(x, now_mu)
        grad_norm2 = numpy.sum(grad_x**2)

        if grad_norm2 < 1.e-6 or lr < 1.e-10:
            break

        loss = loss_func(x, now_mu)
                    
        if sep is not 0 and i % sep == 0:
            print(
"""At the {0}-th iteration,\
 loss = {1:.5e}, lr = {2:.5e},\
 mu = {3:.5e}, grad_norm2 = {4:.5e}""".format(i, real_loss, lr, now_mu, grad_norm2))

        x = x - lr*grad_x
    
    solution = x
    loss = loss_func(x, mu)
    
    out = {
        "solution": solution,
        "loss": loss,
        "num_var": n,
        "iters": i_count,
        "setup_time": -1.,
        "solve_time": -1.
    }

    return [solution, out]
