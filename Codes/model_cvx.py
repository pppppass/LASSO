"""
Solve lasso regularized least square by calling cvxpy.
"""

import numpy
import cvxpy

def l1_cvx_mosek(x0, A, b, mu, options={}):
    m, n = A.shape
    
    x = cvxpy.Variable(n)
    
    objective = cvxpy.Minimize(
          1. / 2. * cvxpy.sum_squares(A*x - b)
        + mu * cvxpy.norm(x, 1)
    )
    problem = cvxpy.Problem(objective)
    
    loss = problem.solve(solver=cvxpy.MOSEK)
    
    solution = numpy.asarray(x.value).reshape(n, 1)
    
    sizes = problem.size_metrics
    stats = problem.solver_stats
    
    out = {
        "solution": solution,
        "loss": loss,
        "num_var": -1,
        "iters": stats.num_iters,
        "setup_time": stats.setup_time,
        "solve_time": stats.solve_time,
    }
    
    return [solution, out]

def l1_cvx_gurobi(x0, A, b, mu, options={}):
    m, n = A.shape
    
    x = cvxpy.Variable(n)
    
    objective = cvxpy.Minimize(
          1. / 2. * cvxpy.sum_squares(A*x - b)
        + mu * cvxpy.norm(x, 1)
    )
    problem = cvxpy.Problem(objective)
    
    loss = problem.solve(solver=cvxpy.GUROBI)
    
    solution = numpy.asarray(x.value).reshape(n, 1)
    
    sizes = problem.size_metrics
    stats = problem.solver_stats
    
    out = {
        "solution": solution,
        "loss": loss,
        "num_var": -1,
        "iters": -1,
        "setup_time": -1.,
        "solve_time": stats.solve_time,
    }
    
    return [solution, out]
