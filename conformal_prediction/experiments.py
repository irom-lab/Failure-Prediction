import numpy as np
from matplotlib import pyplot as plt
from utils import construct_fail_set, calculate_e_star, safety_failure


def exp_1(N, e_star0=0.015, T=500, ed=0.25, delta=0.000001, ub=0.4):
    '''
    Experiment 1: Test the empirical failure rate of algorithms under a 
    conformal prediction framework, following the algorithm presented by 
    the Pavone group paper in Summer/Fall 2021. Default parameters show a 
    failure rate of ~15.2% of the conformal prediction guarantee; PAC-Bayes 
    guarantees would allow for this 15.2% to be reduced to an arbitrarily 
    small fraction. 

    Parameters
    ----------
    N : int
        Number of training sample sets to draw.
    e_star0 : float, optional
        Nominal desired failure rate to be guaranteed (over the T steps). 
        The default is 0.015.
    T : int, optional
        Size of each sample set, this is the sequential conformal prediction
        variable; i.e. the guarantee is over the number of t in {1, 2, ..., T}
        in which the sample fails the safety constraint. The default is 500.
    ed : float, optional
        True expected fraction of failures within the datasets being drawn. 
        The default is 0.25.
    delta : float, optional
        Confidence parameter for Hoeffding bound (the probability of a bad 
        sample set). The default is 0.000001.
    ub : float, optional
        Upper bound for conformal prediction algorithm. The default is 0.4.

    Returns
    -------
    theory_est_fail : float
        Fraction of samples n in {1, 2, ..., N} that lead to violations of the 
        conformal prediction guarantee. The idea is that this fraction cannot
        be reduced arbitrarily (as it can be for PAC-Bayes guarantees).
    theory_Hoeff_validity : float
        Theoretical fraction of sample sets that should be valid (1-delta).
    emp_Hoeff_validity : float
        Actual fraction of sample sets that should be valid. Should be close
        to 1-delta for large N. 

    '''
    e_star_low = calculate_e_star(T, ed, delta)
    e_star = e_star0
    while e_star < e_star_low:
        e_star = e_star + 0.01
    
    print('Running experiment with N =', int(N), 'samples')
    print('Running Experiment with e_star =', e_star)
    print()

    Hoeff_validity = 0
    fail_validity_theory = 0
    
    for i in range(N):
        g_fail, Hoeff_draw = construct_fail_set(T, ed, delta, ub)
        
        Hoeff_validity=Hoeff_validity + Hoeff_draw
        
        if Hoeff_draw > 0: # Valid sample for Hoeffding calcs on e_star
            TF = safety_failure(g_fail, e_star, ub)
            if TF:
                fail_validity_theory = fail_validity_theory + 1

    theory_est_fail = float(fail_validity_theory)/float(Hoeff_validity)
    emp_Hoeff_validity = float(Hoeff_validity)/float(N)
    theory_Hoeff_validity = 1-delta
    
    return theory_est_fail, theory_Hoeff_validity, emp_Hoeff_validity, 


def exp_2(e_star=0.015, T=500, ed=0.25):
    '''
    Experiment 2: Similar to experiment 1, but with a slightly different 
    methodology. 

    Parameters
    ----------
    e_star : float, optional
        Nominal desired failure rate to be guaranteed (over the T steps). 
        The default is 0.015.
    T : int, optional
        Size of each sample set, this is the sequential conformal prediction
        variable; i.e. the guarantee is over the number of t in {1, 2, ..., T}
        in which the sample fails the safety constraint. The default is 500.
    ed : float, optional
        True expected fraction of failures within the datasets being drawn. 
        The default is 0.25.

    Returns
    -------
    bool
        Verifies a valid draw. 
    safety_violation : int
        Acts as a boolean - 0 if no safety violation, 1 if safety violation.
    p_af : int/float
        Probability that algorithm fails the conformal prediction guarantee.

    '''
    training_data = np.random.rand(T)
    failure_set = training_data[np.where(training_data < ed)]
    count_failure = len(failure_set)
    safety_violation = 0
    p_af = 0

    # Case Handling - Note this implicitly accounts for the delta term
    if (e_star < (1./(1.+count_failure))):
        # Then not enough failures (unlikely) --> Bad draw (epsilon^* invalid)
        return False, safety_violation, p_af

    p_af = (1.-e_star)**(count_failure)
    if (np.max(failure_set) < ((1-e_star)*ed)):
        safety_violation = 1
        
    return True, safety_violation, p_af


def exp_3(T=500, Nruns=50000, delta=0.01):
    '''
    Experiment 3: Run conformal prediction algorithm acting as a reduction for 
    estimating the median of a distribution as a function of iid data draws. 
    This function plots the convergence results of the median estimate error
    as a function of t, where t is the number of samples in the draw (N again 
    refers to the number of draws or 'experiments' of T samples).

    Parameters
    ----------
    T : int, optional
        Number of samples to draw in the sample set. The default is 500.
    Nruns : int, optional
        Number of sample sets in the experiment. The default is 50000.
    delta : float, optional
        Confidence parameter for Hoeffding bound. The default is 0.01.

    Returns
    -------
    None.

    '''
    results = np.zeros((Nruns, T))
    vec_whp = np.zeros(T)
    vec_mean = np.zeros(T)
    vec_median = np.zeros(T)
    
    Bernstein_uniform_whp = (4./np.sqrt(12))*np.sqrt(np.log(2/delta)/np.arange(1,T+1)) + 4.*np.log(2/delta)/np.arange(1,T+1)
    
    for k in range(Nruns):
        tmp = np.random.rand(T)
        results[k, :] = np.abs((np.cumsum(tmp)/np.arange(1,T+1)) - 0.5)
    
    idx = int(np.floor(Nruns*(1-delta)) + 1)
    for i in range(T):
        vec_mean[i] = np.mean(results[:,i])
        vec_median[i] = np.median(results[:,i])
        tmp_sort = np.sort(results[:,i])
        
        if tmp_sort[-1] > tmp_sort[0]:
            vec_whp[i] = tmp_sort[idx]
        else:
            vec_whp[i] = tmp_sort[Nruns-idx]
    
    
    start_idx = 4
    t_vals = np.arange(start_idx+1, T+1)
    mean_median_oneT = vec_median[start_idx]*(t_vals[0])/(t_vals)
    mean_median_sqrt = vec_median[start_idx]*(np.sqrt(t_vals[0]))/(np.sqrt(t_vals))
    
    plt.figure(1)
    plt.plot(t_vals, vec_median[start_idx:], 'k--')
    plt.plot(t_vals, mean_median_oneT, 'r')
    plt.plot(t_vals, mean_median_sqrt, 'b')
    plt.title('Expected convergence over time')
    plt.xlabel('Number of samples T')
    plt.ylabel('Absolute mean error')
    plt.legend(['True Data', '1/T Convergence', 'Sqrt Convergence'])
    plt.show()
    
    plt.figure(3)
    plt.loglog(t_vals, vec_median[start_idx:], 'k--')
    plt.loglog(t_vals, mean_median_oneT, 'r')
    plt.loglog(t_vals, mean_median_sqrt, 'b')
    plt.title('Expected Convergence With Sample Size')
    plt.xlabel('Number of Samples t')
    plt.ylabel('Median Estimate Error')
    plt.legend(['True Data', '1/T Convergence', 'Sqrt Convergence'])
    plt.savefig('images/ConvergenceOfSampleMedian.png')
    plt.show()
    
    plt.figure(2)
    plt.plot(t_vals, vec_whp[start_idx:], 'b--')
    # plt.plot(t_vals, Bernstein_uniform_whp[start_idx:], 'r')
    plt.plot(t_vals, vec_whp[start_idx]*np.sqrt(t_vals[0])/np.sqrt(t_vals), 'r')
    plt.title('High-probability convergence over time')
    plt.xlabel('Number of samples T')
    plt.ylabel('Absolute mean error')
    plt.legend(['True Data', 'Sqrt Convergence'])
    plt.show()
    
    return
