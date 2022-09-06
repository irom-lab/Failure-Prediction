import torch
import numpy as np


class Loss:
    def __init__(self, num_imgs, N, delta, device, loss_fn_preset=None):
        self.criterion = torch.nn.BCELoss(reduce=False).to(device)
        self.num_imgs = num_imgs
        self.loss = self.loss_function
        self.device = device
        self.prior = None
        self.N = N
        self.delta = delta

        self.discount_if_collision = 0  # if we discount the loss based on how far out we predict collision
        self.discount_if_safe = 0  # if we discount the loss based on how far out we predict safe
        self.fn_factor = 1  # how much more we care about false negatives  (1 is even, > 1 is more, < 1 is less)
        self.num_pred_ahead = self.num_imgs
        self.start_image = 0

        if loss_fn_preset == 1:
            self.num_pred_ahead = 10
            # self.discount_if_collision = 0
            # self.discount_if_safe = 0

    def __call__(self, model_output, y, model=None):
        loss = self.loss(model_output, y)
        rg = self.PAC_Bayes_Reg(model) if model is not None else 0
        return loss + rg

    # need a way to exit detection, otherwise getting confused with things after the crash?
    def loss_function(self, model_output, y):
        loss = torch.tensor(0, dtype=torch.float).to(self.device)
        crash_memory = torch.zeros([y.shape[0], ], dtype=torch.float).to(self.device)
        for i in range(self.start_image, self.num_imgs):
            comp_img = min(self.num_imgs - 1, i + self.num_pred_ahead)

            first_term = y[:, comp_img, 0] * (1/(i+1))**self.discount_if_collision
            second_term = y[:, comp_img, 1] * self.fn_factor * (1/(i+1))**self.discount_if_safe
            weight = (first_term + second_term) / (self.fn_factor + 1)

            losses = weight * self.criterion(model_output[:, i], y[:, comp_img])[:, 0]

            loss += torch.mean(losses)

        return loss / self.num_imgs

    def PAC_Bayes_Reg(self, model):
        if self.prior is None:
            return torch.tensor(0)

        KL = model.calc_kl_div(self.prior, self.device)
        num = KL + np.log(2*np.sqrt(self.N)/self.delta)
        rg = torch.sqrt(num/(2*self.N))
        return rg


def kl_inv_l(q, c):
    import cvxpy as cvx
    solver = cvx.MOSEK
    # KL(q||p) <= c
    # try to solve: KLinv(q||c) = p

    # solve: sup  p
    #       s.t.  KL(q||p) <= c

    p_bernoulli = cvx.Variable(2)
    q_bernoulli = np.array([q, 1 - q])
    constraints = [c >= cvx.sum(cvx.kl_div(q_bernoulli, p_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1,
                   p_bernoulli[1] == 1.0 - p_bernoulli[0]]
    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)
    opt = prob.solve(verbose=False, solver=solver)
    return p_bernoulli.value[0]


def count_fnfptntp(model_output, label):
    model_output = model_output.max(dim=2).indices  # 0 means guess success, 1 means guess fail
    label = label.max(dim=2).indices

    test_batch = model_output.shape[0]
    fd_cost_1 = 0
    fd_cost_12 = 0
    fd_cost_2 = 0
    fd_cost_3 = 0
    fd_cost_4 = 0

    for i in range(test_batch):
        if label[i, -1]:  # collision
            if max(model_output[i]):  # detected label at some point
                first_detection = model_output[i].nonzero()[0][0]  # when was the first detection
                first_label = label[i].nonzero()[0][0]  # when was the collision
                if first_detection < first_label:  # if we detect **before** the collision
                    fd_cost_1 += 1 / test_batch  # early enough
                else:
                    fd_cost_12 += 1 / test_batch  # too late
            else:  # no collision detected
                fd_cost_2 += 1 / test_batch
        else:  # no collision
            if max(model_output[i]):  # detected label at some point
                fd_cost_3 += 1 / test_batch
            else:  # no collision detected
                fd_cost_4 += 1 / test_batch

    fn = fd_cost_12 + fd_cost_2  # crash is relevant, missed crash
    fp = fd_cost_3  # says crash but none
    tn = fd_cost_4  # no crash and is correct
    tp = fd_cost_1  # crash and is correct

    return fn, fp, tn, tp


def compute_ccbounds(fnfptntp, N, rg, args):
    # Class conditional probs calculations
    fnfptntp = np.round(fnfptntp, 5)
    pfh = fnfptntp[0] + fnfptntp[3]
    psh = fnfptntp[1] + fnfptntp[2]
    pfp = fnfptntp[1]
    pfn = fnfptntp[0]

    KdN = 100 * np.log(2 / args.delta) / (9 * N)
    coeff = [(1 + KdN), -(2*psh + KdN), psh**2]  # a ph^2 + b ph + c = 0
    psl = min(np.roots(coeff))
    coeff = [(1 + KdN), -(2*pfh + KdN), pfh**2]
    pfl = min(np.roots(coeff))

    Kfl = pfl / (pfh - pfl)
    Ksl = psl / (psh - psl)
    Kminl = min(Kfl, Ksl)

    ccbounds = []
    for lam in [[1, 0], [0, 1]]:
        # print('lambda', lam)

        C_lam = lam[0]*psl + lam[1]*pfl

        C_tilde = 1/C_lam * (lam[0] / psl * pfp + lam[1] / pfl * pfn)
        bound = C_lam * C_tilde + C_lam * rg
        scaled_loss = C_lam * C_tilde
        scaled_reg = C_lam * rg
        ccbounds.append((scaled_loss, scaled_reg))

        # C_hat = lam[0]*pfp/psh + lam[1]*pfn/pfh
        # bound2 = C_hat * (1 + 1/Kminl) + C_lam * rg
        #
        # print(lam, bound)

    return ccbounds
