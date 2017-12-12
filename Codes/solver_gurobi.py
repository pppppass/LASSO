"""
Solve lasso regularized least square by calling gurobi directly.
"""

import time
import numpy
from gurobipy import *

def gurobi_model(m, n, A, x0, b, mu):
    M = Model("qp")
    
    x = M.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    y = M.addVars(m, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    t = M.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    
    obj = 1. / 2. * y.prod(y) + mu * t.sum()
    M.setObjective(obj)

    M.addConstrs(x[i] - t[i] <= 0. for i in range(n))
    M.addConstrs(x[i] + t[i] >= 0. for i in range(n))

    M.addConstrs(quicksum(A[i, j] * x[j] for j in range(n)) - y[i] == b[i, 0] for i in range(m))
    
    return M, x

def l1_gurobi_nonexpand(x0, A, b, mu, output="off", **opts):
    m, n = A.shape
    
    start = time.time()
    
    M, x = gurobi_model(m, n, A, x0, b, mu)
    
    flag_dict = {"on": 1, "off": 0}
    M.setParam(GRB.Param.OutputFlag, flag_dict[output])
    
    end = time.time()
    setup_time = end - start
    
    M.optimize()
    
    xtd = M.getAttr("x", x)
    solution = numpy.array([xtd[i] for i in range(n)]).reshape(n, 1)
    loss = M.getAttr("ObjVal")
    
    out = {
        "solution": solution,
        "loss": loss,
        "num_var": M.getAttr("NumVars"),
        "iters": M.getAttr("BarIterCount"),
        "setup_time": setup_time,
        "solve_time": M.getAttr("Runtime"),
    }

    return solution, out
