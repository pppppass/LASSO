"""
Solve lasso regularized least square by calling mosek directly.
"""

import time
import numpy
import mosek

def mosek_model(m, n, A, x0, b, mu, task):
    inf = 0.

    task.appendcons(2*n + m)
    task.appendvars(2*n + m)

    task.putqobj(range(2*n, 2*n + m), range(2*n, 2*n + m), [1.]*m)

    task.putclist(range(n, 2*n), [mu]*n)

    task.putconboundlist(range(0, n), [mosek.boundkey.up]*n, [inf]*n, [0.]*n)
    task.putconboundlist(range(n, 2*n), [mosek.boundkey.lo]*n, [0.]*n, [inf]*n)
    task.putconboundlist(range(2*n, 2*n + m), [mosek.boundkey.fx]*m, b[:, 0], b[:, 0])

    task.putvarboundlist(range(2*n + m), [mosek.boundkey.fr]*(2*n + m), [inf]*(2*n + m), [inf]*(2*n + m))
    for i in range(n):
        task.putarow(i, [i, n + i], [1., -1.])
        task.putarow(i + n, [i, n + i], [1., 1.])
    for i in range(m):
        task.putarow(i + 2*n, range(n), A[i, :])
        task.putaij(i + 2*n, 2*n + i, -1.)

    task.putobjsense(mosek.objsense.minimize)

def l1_mosek_qp(x0, A, b, mu, log=None, **opts):
    m, n = A.shape

    start = time.time()
    
    with mosek.Env() as env:
        env.set_Stream(mosek.streamtype.log, log)
        
        with env.Task() as task:
            task.set_Stream(mosek.streamtype.log, log)

            mosek_model(m, n, A, x0, b, mu, task)
            
            end = time.time()
            setup_time = end - start
            
            task.optimize()
            
            xx = [0.] * (2*n + m)
            task.getxx(mosek.soltype.itr, xx)

            solution = numpy.array(xx[:n]).reshape(n, 1)
            
            loss = task.getprimalobj(mosek.soltype.itr)
    
            out = {
                "solution": solution,
                "loss": loss,
                "vars": task.getintinf(mosek.iinfitem.opt_numvar),
                "iters": task.getintinf(mosek.iinfitem.intpnt_iter),
                "setup_time": setup_time,
                "solve_time": task.getdouinf(mosek.dinfitem.optimizer_time)
            }

            return solution, out
