import torch
import numpy as np
from util_models import load_weights, save_weights
from models.policy import *
from dataloaders import DataLoader_FailureDetector as dataloader
import warnings
import ast
from copy import deepcopy
from loss_utils import Loss, kl_inv_l, compute_ccbounds, count_fnfptntp
import argparse
from gen_data import visualize
warnings.filterwarnings('ignore')
torch.set_num_threads(8)


def test(model, dl, dataset, loss_function):
    x, y = dl.get_batch(dataset, batch=500)

    model.init_xi()
    model_output = model(x)
    loss = loss_function(model_output, y)

    fn, fp, tn, tp = count_fnfptntp(model_output, y)

    if verbose >= 1:
        print("test_loss:", float(loss.to('cpu')))
        print("CASE 1: FD saves from failure:         p =", np.round(tp, 5))
        # print("CASE 1/2: FD is too late on a failure: p =", np.round(fd_cost_12, 5))
        print("CASE 2: FD misses/late on a failure    p =", np.round(fn, 5))
        print("CASE 3: FD is too conservative:        p =", np.round(fp, 5))
        print("CASE 4: FD correct about no failure:   p =", np.round(tn, 5))
        print('tp+tn:', np.round(tn + tp, 5), ';  safe:', np.round(tp + fp + tn, 5))
        print('precision', np.round(tp / (tp + fp + 1e-10), 5), 'recall', np.round(tp / (tp + fn + 1e-10), 5))
    if verbose > 0:
        print('fn:', np.round(fn, 5), '; fp:', np.round(fp, 5), '; tp:', np.round(tp, 5), '; tn:', np.round(tn, 5))
        print('fn+fp', np.round(fn + fp, 5), 'rg:', np.round(loss_function.PAC_Bayes_Reg(model).detach().to('cpu').numpy(), 5))


def train(num_steps, model, dl, dataset, optimizer, loss_function, fd_path):
    losses = []
    for n in range(1, num_steps+1):
        x, y = dl.get_batch(dataset)
        model.init_xi()
        model_output = model(x)

        loss = loss_function(model_output, y, model)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        losses.append(float(loss.to('cpu')))
        if verbose > 1:
            print('Iteration: {}, avg loss: {:.7f}'.format(n, np.mean(losses)), end='\r')
        
        if verbose > 1 and n % 100 == 0:
            print()
            test(model, dl, 'post', loss_function)
            save_weights(model, save_file_name=fd_path)
    if verbose > 1:
        test(model, dl, 'post', loss_function)
        save_weights(model, save_file_name=fd_path)


def eval(model, dl, dataset, loss_function, args):
    xb, yb = dl.data[dataset]
    num_evaluations = args.n_eval
    deltap = args.deltap

    # regularizers for PAC-Bayes and sample convergence
    rg = loss_function.PAC_Bayes_Reg(model).detach().to('cpu').numpy()
    sample_convergence_reg = np.log(2/deltap)/num_evaluations
    N = loss_function.N

    batch_size = 1000
    batchs = int(xb.shape[0] / batch_size) + 1

    fnfptntp = np.zeros((4,))
    for b in range(batchs-1):
        x = xb[b*batch_size:(b+1)*batch_size]
        y = yb[b*batch_size:(b+1)*batch_size]
        x = x.to(device)
        y = y.to(device)
        for eval in range(num_evaluations):
            model.init_xi()
            fn, fp, tn, tp = count_fnfptntp(model(x), y)
            fnfptntp += np.array([fn, fp, tn, tp])*x.shape[0]
    fnfptntps = fnfptntp
    fnfptntps /= num_evaluations*xb.shape[0]

    avg_emp_loss = fnfptntps[0] + fnfptntps[1]
    lossbound = kl_inv_l(avg_emp_loss, sample_convergence_reg) if avg_emp_loss < 1 else 1
    bound = kl_inv_l(lossbound, rg**2 * 2) if lossbound < 1 else 1

    # print("Emp loss:", fnfptntps[0] + fnfptntps[1])
    # print("PAC-Bayes reg", rg)
    # print("Lossbound", lossbound)
    # print("Final bound", bound)
    pbound_stats = [bound, lossbound, list(np.round(fnfptntps, 8)), float(rg), sample_convergence_reg]
    print(pbound_stats)

    [ccs_bound_terms, ccf_bound_terms] = compute_ccbounds(fnfptntps, N, rg, args)
    # print('ccs_bound', ccs_bound_terms)
    # print('ccf_bound', ccf_bound_terms)

    slossbound = kl_inv_l(ccs_bound_terms[0], sample_convergence_reg) if ccs_bound_terms[0] < 1 else 1
    flossbound = kl_inv_l(ccf_bound_terms[0], sample_convergence_reg) if ccf_bound_terms[0] < 1 else 1
    ccs_bound = slossbound + ccs_bound_terms[1]
    ccf_bound = flossbound + ccf_bound_terms[1]
    cc_stats = [ccs_bound, ccf_bound, ccs_bound_terms, ccf_bound_terms]
    print(cc_stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--small', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--stack', type=int, default=1)

    # arguments for training prior
    parser.add_argument('--n_iter_prior', type=int, default=5000)
    parser.add_argument('--lr_prior', type=float, default=0.001)

    # arguments for training posterior
    parser.add_argument('--n_iter_post', type=int, default=2000)
    parser.add_argument('--lr_post', type=float, default=0.0001)

    # arguments for evaluating bound
    parser.add_argument('--n_eval', type=int, default=1000)
    parser.add_argument('--delta', type=float, default=0.009)
    parser.add_argument('--deltap', type=float, default=0.001)

    # arguments for running multiple trials
    parser.add_argument('--fn_factors', type=str, default='[1, ]')
    parser.add_argument('--seeds', type=int, default=1)
    args = parser.parse_args()

    train_prior = False
    train_post = False
    eval_bound = True

    policy_num = args.policy
    verbose = args.verbose
    device = torch.device('cuda:' + args.gpu) if int(args.gpu) != -1 else torch.device('cpu')

    if args.stack:
        nsmodel = NSFailureDetectorStack
        smodel = SFailureDetectorStack
    else:
        nsmodel = NSFailureDetectorSimple
        smodel = SFailureDetectorSimple

    # a = smodel()
    # print('\nTotal parameters in model: {}'.format(
    #     sum(p.numel() for p in a.parameters()
    #         if p.requires_grad)))
    #
    # exit()

    data_dict = {'prior': '_prior' + ('small' if args.small else '') + str(policy_num),
                 'post': '_post' + ('small' if args.small else '') + str(policy_num),
                 # 'test': '_test' + ('small' if args.small else '') + str(policy_num),
                 }
    dl = dataloader(data_dict, device, args.batch_size)

    num_imgs = dl.data['post'][0].shape[1]
    N = dl.data['post'][0].shape[0]
    loss = Loss(num_imgs, N, args.delta, device, loss_fn_preset=1)

    for fn_factor in ast.literal_eval(args.fn_factors):
        if verbose == 0:
            print("\t Evaluating bound", "policy", policy_num, "fn_factor:", fn_factor)

        prior_path = 'fd' + ('stack' if args.stack else '') + str(policy_num) + '_prior_fnscale' + (str(fn_factor))
        post_path = 'fd' + ('stack' if args.stack else '') + str(policy_num) + '_post_fnscale' + (str(fn_factor))

        loss.fn_factor = fn_factor

        if train_prior:
            if verbose > 0:
                print("\t Training prior", "policy", policy_num, "fn_factor:", fn_factor, )
            prior = None
            model = smodel().to(device)
            model.init_logvar(-6, -6)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_prior)
            train(args.n_iter_prior, model, dl, 'prior', optimizer, loss, prior_path)
            save_weights(model, prior_path)

        if train_post:
            if verbose > 0:
                print("\t Training posterior", "policy", policy_num, "fn_factor:", fn_factor, )
            model = smodel().to(device)
            load_weights(model, device, prior_path)
            model.init_logvar(-6, -6)

            prior = deepcopy(model)
            loss.prior = prior
            save_weights(prior, prior_path)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_post)
            train(args.n_iter_post, model, dl, 'post', optimizer, loss, post_path)
            save_weights(model, post_path)

        if eval_bound:
            if verbose > 0:
                print("\t Evaluating bound", "policy", policy_num, "fn_factor:", fn_factor)
            model = smodel().to(device)
            load_weights(model, device, post_path)
            prior = smodel().to(device)
            load_weights(prior, device, prior_path)
            loss.prior = prior
            eval(model, dl, 'post', loss, args)
