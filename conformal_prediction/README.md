# Experimental Test of Conformal Prediction Guarantees
Author: @dasnyder5

### Requirements
```
numpy
scipy
matplotlib
```

### Description of Experiments
The code contains three experiments exploring the behavior of conformal prediction methods
for failure detection, in order to understand the meaning, assumptions, and implications of
each guarantee. In summary, we emphasize that conformal prediction guarantees hold in 
expectation over the draw of the training sample AND testing point, whereas PAC-Bayes 
guarantees hold in expectation over the training point but with high probability over the 
draw of the training sample.  

For all experiments, we use the conformal prediction algorithm (Algorithm 1) of [Luo et. al., 2022](https://stanfordasl.github.io/wp-content/papercite-data/pdf/Luo.Zhao.ICRA22.pdf).  

Experiment 1 demonstrates that the conformal prediction algorithm, for some reasonable 
choice of parameters, fails to synthesize an effective detector with nontrivial probability 
(0.152 or ~15.2%).  

Experiment 2 demonstrates the same phenomenon through a different methodology 
(again, with a ~15.2% expected failure rate).  

Experiment 3 investigates in more detail the implied consequence of the 
algorithm if the guarantee is understood to be true with high probability over the 
draw of the training sample set - specifically, the 1/T convergence of the sample median.
This is known to not be true (and more generally, for arbitrary distribution percentiles).  

### How to Run the Code
The experiment code itself is contained with `experiments.py`, which relies on utility functions stored in `utils.py`. To run the code, simply run  

`run_experiments.py`

This will output a few numerical results as well as some figures in the `./images/` directory illustrating appropriate convergence rates. Note: the plot for Experiment 2 will change with each run due to variations in the random seed (and of course, one can modify the code to control for this). We have saved example ones in the `./images/` directory for reference. 
