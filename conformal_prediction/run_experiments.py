import numpy as np
from matplotlib import pyplot as plt
from experiments import exp_1, exp_2, exp_3
    
    
if __name__ == "__main__":
    '''
    Run through all of the experiments on default settings 
    '''
    
    ### Experiment 1:
    Nsamples = 50000
    TF, tHv, eHv = exp_1(N=Nsamples)
    
    print('Kept', int(eHv*Nsamples), 'of', Nsamples, 'sample sets')
    print('Theoretical percentage of failures over samples: ', TF)
    print('Theoretical Hoeffding validity: ', tHv)
    print('Empirical Hoeffding validity: ', eHv)
    
  
    ### Experiment 2 Code
    Nruns = 50001
    
    res_Bernoulli = np.zeros(Nruns)
    res_p = np.zeros(Nruns)
    
    for i in range(Nruns):
        # Ensures we ignore sample sets that aren't valid per delta param
        # Creates experiment similar to experiment 1 but with smaller delta 
        # parameter. 
        valid = False
        while (valid is not True):
            valid, dSF, dp = exp_2()
        
        res_Bernoulli[i] = dSF
        res_p[i] = dp
        
    print('Bernoulli estimate of violation rate: ', np.sum(res_Bernoulli)/float(Nruns))
    print('Probability-mass estimate of violation rate: ', np.sum(res_p)/float(Nruns))
    
    plot_axis = np.cumsum(np.ones(50001))[1:]
    plot_Ber = np.cumsum(res_Bernoulli)[1:]
    plot_p = np.cumsum(res_p)[1:]
    minPlot = 100
    
    plt.figure()
    plt.plot(plot_axis[minPlot:], plot_Ber[minPlot:]/np.arange(minPlot+1,Nruns), 'r--')
    plt.plot(plot_axis[minPlot:], plot_p[minPlot:]/np.arange(minPlot+1,Nruns), 'b--')
    plt.plot([minPlot, np.max(plot_axis)], [0.153, 0.153], 'k')
    plt.xlabel('Number of Sample Sets')
    plt.ylabel('Proportion of Bad Sample Sets')
    plt.title('Proportion of Bad Sample Sets vs. Number of Sets')
    plt.legend(['Real Failure Rate', 'Exp. Failure Rate', 'Est. Steady State 0.153'])
    plt.savefig('images/Experiment2Plot.png')
    plt.show()
    
    
    ### Experiment 3 Code
    exp_3()    
