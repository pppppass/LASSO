"""
Solve lasso regularized least square by accelerated proximal gradient method.
"""

import numpy

def l1_fast_prox_grad(x0, A, b, mu, options):
    m, n = A.shape
    
    def loss_func(x, mu):
        error = A.dot(x) - b
        loss = 1. / 2. * numpy.sum(error**2) + mu * numpy.sum(numpy.abs(x))
        return loss
    
    def grad(x):
        error = A.dot(x) - b
        grad_x = A.transpose().dot(error)
        return grad_x
    
    def prox(y, mu, lr):
        shrink_positive = numpy.maximum(0., y - mu*lr)
        shrink_negative = numpy.minimum(0., y + mu*lr)
        return shrink_positive + shrink_negative
    
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
    
    x = x0
    x_old = x0
    
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
        real_loss = loss_func(x, mu)
        loss = loss_func(x, now_mu)
        
        real_loss_list.append(real_loss)
        formal_loss_list.append(loss)
        
        last_grad_norm2 = grad_norm2
        grad_x = grad(x)
        grad_norm2 = numpy.sum(grad_x**2)

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

        if sep != 0 and i % sep == 0:
            print(
"""At the {0}-th iteration,\
 loss = {1:.5e}, lr = {2:.5e},\
 mu = {3:.5e}, grad_norm2 = {4:.5e}""".format(i, real_loss, lr, now_mu, grad_norm2))
        
        y = x + (list_item_ctr - 2) / (list_item_ctr + 1) * (x - x_old)
        x_old = x
        
        grad_y = grad(y)
        x = prox(y - lr*grad_y, now_mu, lr)
    
    solution = x
    loss = loss_func(x, mu)
    
    out = {
        "solution": solution,
        "loss": loss,
        "num_var": n,
        "iters": i_count,
        "setup_time": -1.,
        "solve_time": -1.,
        "real_loss": numpy.array(real_loss_list),
        "formal_loss": numpy.array(formal_loss_list),
    }

    return [solution, out]
