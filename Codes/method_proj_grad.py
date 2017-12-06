"""
Solve lasso regularized least square by projection gradient method.
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
    list_avail, intel_loss_avail, intel_grad_norm2_avail = False, False, False
    if "iter_list" in options.keys():
        list_avail = True
        iter_list = options["iter_list"]
        list_ctr, list_item_ctr, list_len = 0, 0, len(iter_list)
    if "intel_loss" in options.keys():
        intel_loss_avail = True
        intel_loss = options["intel_loss"]
    if "intel_grad_norm2" in options.keys():
        intel_grad_norm2_avail = True
        intel_grad_norm2 = options["intel_grad_norm2"]
    
    lr_func_avail = False
    if "lr_func" in options.keys():
        lr_func_avail = True
        lr_func = options["lr_func"]
    if not lr_func_avail:
        lr = options["lr_init"]
    
    mu_func_avail, mu_list_avail = False, False
    if "mu_func" in options.keys():
        mu_func_avail = True
        mu_func = options["mu_func"]
    if "mu_list" in options.keys():
        mu_list_avail = True
        mu_list = options["mu_list"]
    if not mu_func_avail and not mu_list_avail:
        now_mu = mu
    
    backtracking = False
    if "bt_beta" in options.keys():
        backtracking = True
        beta = options["bt_beta"]
    
    if "sep" in options.keys():
        sep = options["sep"]
    else:
        sep = 0
    
    real_loss_list = []
    formal_loss_list = []
    
    u = 2. * numpy.maximum(x0, 0.)
    v = 2. * numpy.minimum(x0, 0.)
    
    i_count = 0
    loss, real_loss, last_loss = -1., -1., -1.
    grad_norm2, last_grad_norm2 = -1., -1.
    loss_bad_ctr, grad_norm2_bad_ctr = 0, 0
    
    def check_next_stage():
        nonlocal list_item_ctr, list_ctr, list_len
        list_item_ctr = 0
        list_ctr += 1
        if sep != 0:
            print("The {}th stage".format(list_ctr))
        return list_ctr >= list_len
    
    for i in range(iters):
        i_count += 1
        if list_avail:
            list_item_ctr += 1
            if list_item_ctr >= iter_list[list_ctr]:
                if check_next_stage():
                    break
        
        if lr_func_avail:
            lr = lr_func(i)
        
        if mu_func_avail:
            now_mu = mu_func(i)
        elif mu_list_avail:
            now_mu = mu_list[list_ctr]
            
        last_loss = loss
        real_loss = loss_func(u, v, mu)
        loss = loss_func(u, v, now_mu)
        
        real_loss_list.append(real_loss)
        formal_loss_list.append(loss)
        
        last_grad_norm2 = grad_norm2
        grad_u, grad_v = grad(u, v, now_mu)
        grad_norm2 = numpy.sum(grad_u**2) + numpy.sum(grad_v**2)

        if grad_norm2 < 1.e-6 or lr < 1.e-10:
            if list_avail:
                if check_next_stage():
                    break
            else:
                break
        
        if intel_loss_avail:
            if loss > last_loss * intel_loss[0] and last_loss > 0.:
                loss_bad_ctr += 1
            if loss_bad_ctr > intel_loss[1] * list_item_ctr and list_item_ctr > intel_loss[2]:
                loss_bad_ctr = 0
                if check_next_stage():
                    break
        
        if intel_grad_norm2_avail:
            if grad_norm2 > last_grad_norm2 * intel_grad_norm2[0] and last_grad_norm2 > 0.:
                grad_norm2_bad_ctr += 1
            if grad_norm2_bad_ctr > intel_grad_norm2[1] * list_item_ctr and list_item_ctr > intel_grad_norm2[2]:
                grad_norm2_bad_ctr = 0
                if check_next_stage():
                    break
        
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
                    
        if sep != 0 and i % sep == 0:
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
        "solve_time": -1.,
        "real_loss": numpy.array(real_loss_list),
        "formal_loss": numpy.array(formal_loss_list),
    }

    return [solution, out]
