"""
Solve lasso regularized least square by calling gurobi directly.
"""

import time
import numpy
from gurobipy import *

def l1_gurobi_expand(x0, A, b, mu, options={}):
    m, n = A.shape
    
    start_time = time.time()
    
    M = Model("qp")
    
    x_list = [M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY) for i in range(n)]
    t_list = [M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY) for i in range(n)]
    
    AT_A = A.transpose().dot(A)
    bT_A = b.transpose().dot(A).reshape(n)
    bT_b = b.transpose().dot(b).reshape(1)
    
    for i in range(n):
        M.addConstr(x_list[i] - t_list[i], GRB.LESS_EQUAL, 0.)
        M.addConstr(x_list[i] + t_list[i], GRB.GREATER_EQUAL, 0.)
    
    obj = QuadExpr()
    
    for i in range(n):
        for j in range(n):
            obj += 1. / 2. * float(AT_A[i][j]) * x_list[i] * x_list[j]
            
    for i in range(n):
        obj -= float(bT_A[i]) * x_list[i]
    obj += 0.5*float(bT_b)
    
    for i in range(n):
        obj += mu * t_list[i]

    M.setObjective(obj)

    end_time = time.time()
    setup_time = end_time - start_time
    
    M.optimize()
    
    solution = numpy.array(M.getAttr("x", x_list)).reshape(n, 1)
    loss = M.getAttr("ObjVal")
    
    out = {
        "solution": solution,
        "loss": loss,
        "num_var": M.getAttr("NumVars"),
        "iters": M.getAttr("BarIterCount"),
        "setup_time": setup_time,
        "solve_time": M.getAttr("Runtime")
    }

    return [solution, out]

def l1_gurobi_nonexpand(x0, A, b, mu, options={}):
    m, n = A.shape
    
    start_time = time.time()
    
    M = Model("qp")
    
    x_list = [M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY) for i in range(n)]
    t_list = [M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY) for i in range(n)]
    y_list = [M.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY) for i in range(m)]
    
    for i in range(n):
        M.addConstr(x_list[i] - t_list[i], GRB.LESS_EQUAL, 0.)
        M.addConstr(x_list[i] + t_list[i], GRB.GREATER_EQUAL, 0.)
    
    obj = QuadExpr()
    for i in range(m):
        obj += 1. / 2. * y_list[i] * y_list[i]
    for i in range(n):
        obj += mu * t_list[i]

    M.setObjective(obj)

    for i in range(n):
        M.addConstr(x_list[i] - t_list[i], GRB.LESS_EQUAL, 0.)
        M.addConstr(x_list[i] + t_list[i], GRB.GREATER_EQUAL, 0.)

    for i in range(m):
        con = LinExpr()
        for j in range(n):
            con += A[i][j] * x_list[j]
        M.addConstr(con, GRB.EQUAL, b[i])
    
    end_time = time.time()
    setup_time = end_time - start_time
    
    M.optimize()

    solution = numpy.array(M.getAttr("x", x_list)).reshape(n, 1)
    loss = M.getAttr("ObjVal")
    
    out = {
        "solution": solution,
        "loss": loss,
        "num_var": M.getAttr("NumVars"),
        "iters": M.getAttr("BarIterCount"),
        "setup_time": setup_time,
        "solve_time": M.getAttr("Runtime")
    }

    return [solution, out]
