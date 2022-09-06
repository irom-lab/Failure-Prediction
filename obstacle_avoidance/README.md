# Code for running experiments in section 5A in the paper: Obstacle avoidance with a done
Author: @aalec

### Requirements
```
pytorch
scipy
cvxpy
Mosek (a personal academic license can be obtained here: https://www.mosek.com/products/academic-licenses/)
matplotlib
```


## Training Policy
We provide pretrained policy networks in the `weights` folder: `policy1.pt` and `policy2.py` which we use for the standard and occluded obstacle settings (respectively). If you skip this step, the pretrained weights will be used automatically. If you want to train the policy, uncomment lines `L291-L295` in `gen_data.py` and run:
```
python gen_data.py
```
This generates the synthetic depth images for training the policies. To train the policies, run
```
python train_policy.py
```
with `L20: policy_num = 1` and `policy_num = 2`. The models will be saved into the `weights` folder. 


## Trailing Failure Predictor
We also provide pretrained failure detectors in the `weights` folder for both settings as well as a variet of tunings so show the results in Figure 5 with varied tradeoff between FPR and FNR. If you skip this step, the pretrained weights will be used automatically. If you want to train failure detectors, ensure `L291-L295` in `gen_data.py` are commented and run:
```
python gen_data.py
```
This generates the depth images for training the failure predictors in the standard and occluded obstacle settings. You can train the failure predictors for the results in the paper with the following lines (setting `train_post = True`, `train_prior = True`, and `eval_bound = False`):
```
python train_fd.py --policy 1 --fn_factors "[0.3, 0.4, 0.5, 0.601, 1.001, 2]"
python train_fd.py --policy 2 --fn_factors "[0.125, 0.126, 0.3, 0.4, 0.8, 1, 1.67]"
```
This will generate a series of priors and posterior weights which will be saved into the `weights` folder. Note the tuning parameters `fn_factors` trades off the importance of false positives and false negatives. The training is subject to noise and we cannot control exactly the tradeoff with `fn_factors`. As such, we trained with a wide variety of scales and these were the values used in the paper to represent a range in values for Figure 5 in the paper.

## Evaluating bounds:
The results from the pretrained network are at the bottom of the `plots.py` file. If you skip this step, the saved numerical results will be used automatically. If you want to generate these results, set `train_post = False`, `train_prior = False`, and `eval_bound = True` and run
```
python train_fd.py --policy 1 --fn_factors "[0.3, 0.4, 0.5, 0.601, 1.001, 2]"
python train_fd.py --policy 2 --fn_factors "[0.125, 0.126, 0.3, 0.4, 0.8, 1, 1.67]" 
```
Note that these may take a while because of the number of evaluations required in the sample convergence bound. You may want to add the flag `--n_eval` and specify a number smaller than `1000` to speed computations up but loosen the guarantees. These lines will compute the corresponding PAC-Bayes bounds using Theorem 2 and class conditional bounds using theorem 3 in the paper. The output format is explained on line `L140-L145` in `plots.py`. After the bounds are computed, you can paste the results into `plots.py` for processing and plotting in the same format as `data_p1` and `data_p2`.

## Plotting Results
Using the results computed with the networks, run 
```
python plots.py
``` 
to generate the figures in this section. The results in the tables are pulled directly from the output of the bound evaluation. See `L140-L145` of `plots.py` for specific information about the output of the bound evaluation. The predictor with the best PAC-Bayes bound (Theorem 2 in the paper) was used for the results in the tables and hardware testing. In the results we computed, this was `fn_factor = 0.4` for the standard setting and `fn_factor = 0.126` for the occluded obstacle setting. 