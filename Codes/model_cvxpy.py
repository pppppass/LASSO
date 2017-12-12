import numpy
import cvxpy

def cvxpy_call(m, n, A, x0, b, mu, solver):
    x = cvxpy.Variable(n)
    
    objective = cvxpy.Minimize(
          1. / 2. * cvxpy.sum_squares(A*x - b)
        + mu * cvxpy.norm(x, 1)
    )
    problem = cvxpy.Problem(objective)
    
    loss = problem.solve(solver=solver)
    
    solution = numpy.asarray(x.value).reshape(n, 1)
    
    return solution, loss, problem

def l1_cvxpy_mosek(x0, A, b, mu, **opts):
    m, n = A.shape
    
    solution, loss, problem = cvxpy_call(m, n, A, x0, b, mu, solver=cvxpy.MOSEK)
    
    sizes = problem.size_metrics
    stats = problem.solver_stats
    
    out = {
        "solution": solution,
        "loss": loss,
        "iters": stats.num_iters,
        "setup_time": stats.setup_time,
        "solve_time": stats.solve_time,
    }
    
    return solution, out

def l1_cvxpy_gurobi(x0, A, b, mu, **opts):
    m, n = A.shape
    
    solution, loss, problem = cvxpy_call(m, n, A, x0, b, mu, solver=cvxpy.GUROBI)
    
    stats = problem.solver_stats
    
    out = {
        "solution": solution,
        "loss": loss,
        "solve_time": stats.solve_time,
    }
    
    return solution, out
