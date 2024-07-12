import numpy as np
import random
# import lib_inverse_problem as Ip

class Config:
    def __init__(self, beta, drift, sigma, delta, v, epsilon):
        self.beta = beta
        self.drift =drift
        self.sigma = sigma
        self.delta = delta
        self.v =v
        self.epsilon = epsilon

# for our purpose, we eliminate "problem" from input, and comment out "InverseProblem" ~ part
def step(problem, config, ensembles, eq_constraint=None, ineq_constraint=None,
         grad_eq_constraint=None, grad_ineq_constraint=None, verbose=False):
  
# def step(config, ensembles, eq_constraint=None, ineq_constraint=None,
#          grad_eq_constraint=None, grad_ineq_constraint=None, verbose=False):

    # if "InverseProblem" in str(type(problem)):
    #     objective = lambda u: Ip.reg_least_squares(problem, u)
    # else:
    #     objective = problem
    
    objective = problem

    v = config.v
    epsilon = config.epsilon

    def extra_objective(x):
        result = 0
        if eq_constraint is not None:
            result += (1/v) * eq_constraint(x)**2
        elif ineq_constraint is not None:
            val = ineq_constraint(x)
            result += 0 if val > 0 else (1/v) * val**2
        return result

    drift = config.drift
    sigma = config.sigma
    delta = config.delta

    d, J = ensembles.shape
    fensembles = np.zeros(J)
    for i in range(len(fensembles)):
        fensembles[i] = objective(ensembles[:, i])
        fensembles[i] += extra_objective(ensembles[:, i])
        if verbose:
            print(".", end="")
    if verbose:
        print("")

    fensembles -= np.min(fensembles)
    weights = np.exp(-config.beta * fensembles)
    weights /= np.sum(weights)

    mean = np.sum(ensembles * weights, axis=1).reshape(-1, 1)
    diff = ensembles - mean
    new_ensembles = ensembles - drift * delta * diff + sigma * np.sqrt(2 * delta)*(diff * np.random.randn(d, J))
    if grad_eq_constraint is not None:
        for i in range(len(fensembles)):
            # SPECIAL CASE !!!
            new_ensembles[:, i] = new_ensembles[:, i] / (1 + 4*(delta / epsilon) * eq_constraint(ensembles[:, i]))
            # new_ensembles[:, i] -= (2/epsilon) * delta * eq_constraint(ensembles[:, i]) * grad_eq_constraint(ensembles[:, i])

    if grad_ineq_constraint is not None:
        for i in range(len(fensembles)):
            # SPECIAL CASE !!!
            # if ineq_constraint(ensembles[:, i]) < 0:
            new_ensembles[:, i] = new_ensembles[:, i] / (1 + 4*(delta / epsilon) * ineq_constraint(ensembles[:, i]))
            # new_ensembles[:, i] -= (2/epsilon) * delta * eq_constraint(ensembles[:, i]) * grad_ineq_constraint(ensembles[:, i])

    return new_ensembles