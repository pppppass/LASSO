"""
Solve lasso regularized least square by projected gradient method.
"""

import numpy
def l1_proj_grad(x0, A, b, mu, options):
    m, n = A.shape
    
    def loss_func(u, v, mu):
        x = (u - v) / 2.
        error = A.dot(x) - b
        loss = 1. / 2. * numpy.sum(error**2) + mu * numpy.sum((u + v) / 2.)
        return loss

    def grad(u, v, mu):
        x = (u - v) / 2.
        error = A.dot(x) - b
        grad_u = 1. / 2. * A.transpose().dot(error) + 1. / 2. * mu * numpy.ones((n, 1))
        grad_v = -1. / 2. * A.transpose().dot(error) + 1. / 2. * mu * numpy.ones((n, 1))
        return (grad_u, grad_v)
    
    iters = options["iters"]
    sep = options["sep"]
    mutable_learn_rate = options["mutable_learn_rate"]
    mutable_regularization = options["mutable_regularization"]
    backtracking = options["backtracking"]
    
    u = 2. * numpy.maximum(x0, 0.)
    v = 2. * numpy.minimum(x0, 0.)
    if mutable_learn_rate:
        lr_func = options["lr_func"]
    else:
        lr = options["init_lr"]
    if mutable_regularization:
        mu_func = options["mu_func"]
    else:
        now_mu = mu
    if backtracking:
        beta = options["beta"]
    
    i_count = 0
    
    for i in range(iters):
        i_count += 1
    
        real_loss = loss_func(u, v, mu)
        
        if mutable_learn_rate:
            lr = lr_func(i)
        if mutable_regularization:
            now_mu = mu_func(i)
        
        grad_u, grad_v = grad(u, v, now_mu)
        grad_norm2 = numpy.sum(grad_u**2) + numpy.sum(grad_v**2)

        if grad_norm2 < 1.e-6 or lr < 1.e-10:
            break

        loss = loss_func(u, v, now_mu)
        
        if backtracking:
            while True:
                new_u = numpy.maximum(u - lr*grad_u, 0.)
                new_v = numpy.maximum(v - lr*grad_v, 0.)
                g_u = (u - new_u) / lr
                g_v = (v - new_v) / lr
                new_loss = loss_func(new_u, new_v, now_mu)
                g_norm2 = numpy.sum(g_u**2) + numpy.sum(g_v**2)
                grad_g = numpy.sum(grad_u*g_u) + numpy.sum(grad_v*g_v)
                if new_loss < loss - lr * grad_g + lr / 2. * g_norm2:
                    break
                else:
                    lr *= beta
                    
        if sep is not 0 and i % sep == 0:
            print(
"""At the {0}-th iteration,\
 loss = {1:.5e}, lr = {2:.5e},\
 mu = {3:.5e}, grad_norm2 = {4:.5e}""".format(i, real_loss, lr, now_mu, grad_norm2))

        u = u - lr * grad_u
        v = v - lr * grad_v

        u = numpy.maximum(u, 0.)
        v = numpy.maximum(v, 0.)
    
    solution = (u - v) / 2.
    loss = loss_func(u, v, mu)
    
    out = {
        "solution": solution,
        "loss": loss,
        "num_var": 2*n,
        "iters": i_count,
        "setup_time": -1.,
        "solve_time": -1.
    }

    return [solution, out]
