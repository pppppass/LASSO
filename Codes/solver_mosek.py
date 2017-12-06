"""
Solve lasso regularized least square by calling mosek directly.
"""

import time
import numpy
import mosek
from mosek.fusion import Model, Domain, Var, Expr, ObjectiveSense

def l1_mosek_qp(x0, A, b, mu, options={}):
    m, n = A.shape
    
    inf = 0.

    start_time = time.time()
    
    with mosek.Env() as env:
        env.set_Stream(mosek.streamtype.log, print)
        
        with env.Task() as task:
            task.set_Stream(mosek.streamtype.log, print)

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
            
            end_time = time.time()
            setup_time = end_time - start_time
            
            task.optimize()
            
            xx = [0.] * (2*n + m)
            task.getxx(mosek.soltype.itr, xx)

            solution = numpy.array(xx[:n]).reshape(n, 1)
            
            loss = task.getprimalobj(mosek.soltype.itr)
    
            out = {
                "solution": solution,
                "loss": loss,
                "num_var": task.getintinf(mosek.iinfitem.opt_numvar),
                "iters": task.getintinf(mosek.iinfitem.intpnt_iter),
                "setup_time": setup_time,
                "solve_time": task.getdouinf(mosek.dinfitem.optimizer_time)
            }

            return [solution, out]
            
def l1_mosek_fusion_socp(x0, A, b, mu, options={}):
    m, n = A.shape
        
    start_time  = time.time()
    
    with Model("cqo1") as M:
        
        x = M.variable("x", n, Domain.unbounded())
        t = M.variable("t", n, Domain.unbounded())
        y = M.variable("y", m, Domain.unbounded())
        u = M.variable("u", Domain.unbounded())
        I1 = M.variable("1", Domain.equalsTo(1.))
        
        error = Expr.sub(Expr.mul(A, x), b)
        
        M.constraint(Expr.sub(error, y), Domain.equalsTo(0.))
        
        M.constraint(Expr.sub(x, t), Domain.lessThan(0.))
        M.constraint(Expr.add(x, t), Domain.greaterThan(0.))
        
        z = Var.vstack(u, I1, y)
        M.constraint("qrc", z, Domain.inRotatedQCone())
        
        obj = Expr.add(u, Expr.mul(Expr.sum(t), mu))
        M.objective("obj", ObjectiveSense.Minimize, obj)
        
        end_time = time.time()
        setup_time = end_time - start_time
        
        M.solve()
        
        solution = x.level().reshape(n, 1)
        loss = M.primalObjValue()
    
        out = {
            "solution": solution,
            "loss": loss,
            "num_var": M.getSolverIntInfo("optNumvar"),
            "iters": M.getSolverIntInfo("intpntIter"),
            "setup_time": setup_time,
            "solve_time": M.getSolverDoubleInfo("optimizerTime"),
        }

        return [solution, out]