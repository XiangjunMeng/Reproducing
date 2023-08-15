import tensorflow as tf, numpy as np, pandas as pd
import matplotlib.pyplot as plt

from pyscipopt import Model, quicksum
import pyscipopt

# Data center parameters
mu_min = 0
mu_max = 20
p_peak = 0.2
p_idle = 0.1
e_usage = 1.2
n_servers = 400_000
kappa = (4/3)/3600
max_queue = 500
m = n_servers / 2

a = (p_peak - p_idle) / kappa
b = m * (p_idle + (e_usage - 1) * p_peak)

#price of electricity
omega_1 = 0.3
omega_2 = 0.5
omega_3 = 0.4

a = (p_peak - p_idle) / kappa
b = m * (p_idle + (e_usage - 1) * p_peak)

T = 24

# Initial length of the queue
init_lambda_1 = 150
init_lambda_2 = 250
init_lambda_3 = 350

# Renewable power prediction
W_1 = np.ones(T) * 10_000
W_2 = np.ones(T) * 16_000
W_3 = np.ones(T) * 18_000

# Service requests
L = np.ones(T) * 100

for t in range(T):

    if t == 0:
        lambda_1 = init_lambda_1
        lambda_2 = init_lambda_2
        lambda_3 = init_lambda_3
    else:
        lambda_1 += delta_1 - mu_1
        lambda_2 += delta_2 - mu_2
        lambda_3 += delta_3 - mu_3

    model = Model("data_center_dr")

    # Variables
    MU_1 = model.addVar(vtype='C', name='mu_1', lb=mu_min, ub=mu_max)
    MU_2 = model.addVar(vtype='C', name='mu_2', lb=mu_min, ub=mu_max)
    MU_3 = model.addVar(vtype='C', name='mu_3', lb=mu_min, ub=mu_max)

    DELTA_1 = model.addVar(vtype='C', name='delta_1', lb=0, ub=10000)
    DELTA_2 = model.addVar(vtype='C', name='delta_2', lb=0, ub=10000)
    DELTA_3 = model.addVar(vtype='C', name='delta_3', lb=0, ub=10000)

    P_1 = a * MU_1 + b
    P_2 = a * MU_2 + b
    P_3 = a * MU_3 + b

    G_1 = P_1 - W_1[t]
    G_2 = P_2 - W_2[t]
    G_3 = P_3 - W_3[t]

    # Objective
    obj_1 = G_1 * omega_1 + G_2 * omega_2 + G_3 * omega_3
    obj_2 = MU_1 * lambda_1 + MU_2 * lambda_2 + MU_3 * lambda_3

    model.addCons(DELTA_1 + DELTA_2 + DELTA_3 == L[t])
    model.addCons(lambda_1 + DELTA_1 - MU_1 <= max_queue, name='data_center_1_queue_constraint')
    model.addCons(lambda_2 + DELTA_2 - MU_2 <= max_queue, name='data_center_2_queue_constraint')
    model.addCons(lambda_3 + DELTA_3 - MU_3 <= max_queue, name='data_center_3_queue_constraint')


    # # ESS
    # P_ESS, B_ESS, E_ESS, CH, DCH = {}, {}, {}, {}, {}
    # for t in range(T):
    #     for i, ess in enumerate(env.ess_agents):
    #         eta_chr, eta_dch = ess.ch_eff, ess.dsc_eff
    #         B_ESS[t,i] = model.addVar(vtype='B', name='B_ESS_{}_{}'.format(t,i))
    #         P_ESS[t,i] = model.addVar(vtype='C', name='P_ESS_{}_{}'.format(t,i), lb=0, ub=ess.max_p_mw)
    #         E_ESS[t,i] = model.addVar(vtype='C', name='E_ESS_{}_{}'.format(t,i), lb=ess.min_e_mwh, ub=ess.max_e_mwh)
    #         CH[t,i] = model.addVar(vtype='C', name='CH_{}_{}'.format(t,i))
    #         DCH[t,i] = model.addVar(vtype='C', name='DCH_{}_{}'.format(t,i))
    #         model.addCons(CH[t,i] == dt * P_ESS[t,i] * (1 - B_ESS[t,i]))
    #         model.addCons(DCH[t,i] == dt * P_ESS[t,i] * B_ESS[t,i])
    #         if t == 0:
    #             model.addCons(E_ESS[t,i] == ess.max_e_mwh*ess.state.soc + CH[t,i] * eta_chr - DCH[t,i] / eta_dch, name='E_ESS_DYN_{}_{}'.format(t,i))
    #         else:
    #             model.addCons(E_ESS[t,i] == E_ESS[t-1,i] + CH[t,i] * eta_chr - DCH[t,i] / eta_dch, name='E_ESS_DYN_{}_{}'.format(t,i))

    # Set objective function
    model.setObjective(obj_1 - obj_2, 'minimize')

    # Execute optimization
    # model.hideOutput()
    model.setRealParam('limits/time', 60) # Maximal sovling time: 10 minutes 
    # model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    # model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    # model.disablePropagation()
    model.optimize()
    f = model.getObjVal()

    sol = model.getBestSol()

    mu_1 = sol[MU_1]
    mu_2 = sol[MU_2]
    mu_3 = sol[MU_3]

    delta_1 = sol[DELTA_1]
    delta_2 = sol[DELTA_2]
    delta_3 = sol[DELTA_3]

    p_1 = a * mu_1 + b
    p_2 = a * mu_2 + b
    p_3 = a * mu_3 + b

    if model.getStatus() == "optimal":
        print("Optimal value:", model.getObjVal())
        for v in model.getVars():
            print(v.name, " = ", model.getVal(v))
        # pass
    else:
        print("Problem could not be solved to optimality")

