import numpy as np
from scipy.stats import binom


def construct_fail_set(T=500, ed=0.25, delta=0.000001, ub=0.4, seed=None):

    # Basic error handling + seed repeatability
    if seed is None:
        np.random.seed()
    else:
        if isinstance(seed, int):
            np.random.seed(seed)
        else:
            raise ValueError('Seed is input but is not of type int!')

    # Randomly generate |A| failures out of T samples for Pr(a) = ed
    a=0
    sample = np.random.rand(T)
    for t in range(T):
        if sample[t] <= ed:
            a = a + 1

    # Check if sample is low-probability event (violates Hoeffding)
    Hoeff_draw = 1
    valid_cutoff = (ed*T - np.sqrt((T*np.log(1./delta))/2))

    if (a <= valid_cutoff):
        Hoeff_draw = 0

    g_fail = ub*np.random.rand(a)

    return g_fail, Hoeff_draw


def safety_failure(g_fail, e_star, ub=0.4):
    a = len(g_fail)

    Qstar = (1-e_star)*ub
    Gmax = np.max(g_fail)

    theory_fail = 0

    if Qstar > Gmax:
        theory_fail = 1

    return theory_fail


def calculate_e_star(T=500,ed=0.25,delta=0.00000001):
    """
    Calculate Hoeffding lower bound of e_star via Hoeffding so that e_alg > 0.

    Equivalent to e_star > 1/(1 + ed*T - sqrt(Tlog(1/delta)/2))

    Parameters
    ----------
    T : int, optional
        Sample Size. Default 500.
    ed : float, optional
        Expected fraction of failures within dataset. The default is 0.25.
    delta : float, optional
        Probability with which Hoeffding bound fails. The default is 0.000001.

    Returns
    -------
    e_star_low : float
        Lower bound on valid e_star. Want to choose e_star > e_star_low in the
        testing phase

    """

    A_low = ed*T - np.sqrt(0.5*T*np.log(1/delta))
    e_star_low = (1./(1. + A_low))

    return e_star_low


def est_failure_prob(T=500, e_star = 0.015, ed=0.25):
    cumProb = 0
    tmp = 1.
    rv = binom(T, ed)
    for i in range(T+1):
        cumProb = cumProb + tmp*rv.pmf(i)
        tmp = tmp*(1.-e_star)

    return cumProb


if __name__ == "__main__":
    phat = est_failure_prob()
    print('Estimated failure probability:', phat)
