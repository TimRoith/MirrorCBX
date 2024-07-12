import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
dir_mirrorcbo = os.path.dirname("/Users/dohyeon/Desktop/Academics/Resesarch/Mean_Field_Theory/Mirror_CBX/Code/MirrorCBX/mirrorcbo")
sys.path.append(dir_mirrorcbo)

import Regularized_CBO.CBO_reg as CBO_reg
import mirrorcbo as mcbo
import mirrorcbo.particledynamic as pdyn
from mirrorcbo import objectives, constraints

objective = objectives.Rastrigin()
d = 4

## original:
# constraint = Ackley.sphere_constraint
# grad_constraint = Ackley.grad_sphere_constraint

proj_ball = constraints.ProjectionBall(radius=1)
constraint = proj_ball.constraint
grad_constraint = proj_ball.grad_constraint

beta = 10
drift = 1
sigma = 0.7
delta = 0.01

epsilon = 0.01 # the smaller the epsilon is, faster and more accurate the convergence
J = 100
nsimuls = 10

# nu, epsilon, nsimuls = 1, 1, 1
# nus = [10., 1., .1]
nus = [1]
threshold = 1e-5; rate = 0

for nu in nus:
    config = CBO_reg.Config(beta, drift, sigma, delta, nu, epsilon)
    limits = np.zeros((d, J))
    np.random.seed(0)
    for s in range(1, nsimuls + 1):
        ensembles = 10 * np.random.randn(d, J)
        distance, spread, niter = 10, 1, 0
        while np.max(spread) > 1e-10:
            # print(f"iteration {spread}")
            mean = np.mean(ensembles, axis=1)
            spread = np.sqrt(np.sum(np.abs(np.cov(ensembles)), axis=1))
            distance = np.mean(ensembles - limits, axis = 1)
            ensembles = CBO_reg.step(objective, config, ensembles, eq_constraint = constraint, ineq_constraint = None, 
                        grad_eq_constraint= constraint, grad_ineq_constraint= None, verbose = False)
            # ensembles = CBO_reg.step(objective, config, ensembles, ineq_constraint=constraint)
            print(f"{niter}: Constraint: {constraint(mean)}, Distance: {distance}, Spread: {spread}")
            niter += 1
            # np.savetxt(os.path.join(datadir, f"niter={niter}.txt"), ensembles)
        print(s, mean, constraint(mean))
        limits[:, s-1] = mean
        # compute success rate
        val = objective(ensembles)
        meanval = sum(val)/val.size
        test1 = meanval < threshold
        if test1 == 1:
            rate = rate + 1
        
    rate = rate/nsimuls
    print("Success rate is ", rate)


    # np.savetxt(os.path.join(datadir, f"limits-nu={nu}-epsilon={epsilon}-J={J}.txt"), limits)
    # np.savetxt(os.path.join(datadir, "limits.txt"), limits)
