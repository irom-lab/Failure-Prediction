import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from loss_utils import kl_inv_l, compute_ccbounds


def visualize_primitives(location):
    prims1 = scipy.io.loadmat('./data/policy1_prim.mat')['policy1_prim']
    prims1 = np.array(prims1)/1000
    prims2 = scipy.io.loadmat('./data/policy2_prim.mat')['policy2_prim']
    prims2 = np.array(prims2)/1000

    prim1_choice = [2, 0, 1, 0, 1, 2, 0]
    prim2_choice = [4, 0, 0, 0, 2, 2, 4]

    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1, 2)
    # plt.suptitle("  Motion Primitive Trajectories", fontsize=14)
    fig.set_size_inches(5, 3.7)
    fig.text(0.52, 0.01, 'X Position (m)', ha='center', fontsize=12)
    ax[1].set_yticks([])
    plt.sca(ax[0])
    plt.title("Standard Setting")
    plt.ylabel(' Y Position (m)', fontsize=12)
    plt.ylim([0, 11])
    plt.xlim([-2.4, 2.4])

    for i in range(7):
        traj = prims1[i, prim1_choice[i]]
        plt.plot(traj[:, 0], traj[:, 1], 'k', linewidth=1.5)

    plt.sca(ax[1])
    plt.title("Occluded Obstacle Setting")
    plt.ylim([0, 11])
    plt.xlim([-2.6, 2.5])
    # for i in [5]:
    for i in range(7):
        traj = prims2[i, prim2_choice[i]]
        plt.plot(traj[:, 0], traj[:, 1], 'k', linewidth=1.5)

    plt.savefig(location, dpi=200)


def process_data(data, returns='fnrvsfpr'):
    seeds = round(len(data)/2)
    fnrvsfpr = []
    bounds = []
    for i in range(seeds):

        fnfptntp = data[2*i][2]
        cost = (fnfptntp[0] + fnfptntp[1])
        fpr = (fnfptntp[1] / (fnfptntp[1] + fnfptntp[2]))
        fnr = (fnfptntp[0] / (fnfptntp[0] + fnfptntp[3]))
        pb_bound = (data[2*i][0])
        fpr_bound = (data[2*i+1][0])
        fnr_bound = (data[2*i+1][1])

        failure_rate = np.round(fnfptntp[0] + fnfptntp[3], 4)
        pb_bound = np.round(np.min(pb_bound), 4)
        fpr_bound = np.round(np.min(fpr_bound), 4)
        fnr_bound = np.round(np.min(fnr_bound), 4)
        cost = np.round(cost, 4)  # + ' pm ' + str(np.round(np.std(cost), 4))
        fpr = np.round(fpr, 4)  # + ' pm ' + str(np.round(np.std(fpr), 4))
        fnr = np.round(fnr, 4)  # + ' pm ' + str(np.round(np.std(fnr), 4))
        # print( fpr, ',', fnr)
        fnrvsfpr.append([fpr, fnr, fpr_bound, fnr_bound, pb_bound])


    if returns == 'fnrvsfpr':
        return fnrvsfpr


def trivial_detector():
    N = 10000
    delta = 0.009
    num = np.log(2 * np.sqrt(N) / delta)
    rg = np.sqrt(num / (2 * N))
    class Args:
        def __init__(self):
            self.delta = 0.009
    args = Args()

    # n = 10,000
    sample_convergence_reg = 0.0007600902459542082
    for policynp in [[0.2499, 0.7501], [0.5142, 0.4858]]:
        for i in range(2):
            if i == 1:
                fnfptntps = [policynp[0], 0, policynp[1], 0]
            else:
                fnfptntps = [0, policynp[1], 0, policynp[0]]

            [ccs_bound_terms, ccf_bound_terms] = compute_ccbounds(fnfptntps, N, rg, args)
            slossbound = kl_inv_l(ccs_bound_terms[0], sample_convergence_reg) if ccs_bound_terms[0] < 1 else 1
            flossbound = kl_inv_l(ccf_bound_terms[0], sample_convergence_reg) if ccf_bound_terms[0] < 1 else 1
            ccs_bound = slossbound + ccs_bound_terms[1]
            ccf_bound = flossbound + ccf_bound_terms[1]
            cc_stats = [ccs_bound, ccf_bound, ccs_bound_terms, ccf_bound_terms]
            print(cc_stats)


def plot_fnrvsfpr(location, data_p1, data_p2):
    # FORMAT: [FPR, FNR, FPRB, FNRB, FP+FNB]

    # data_p1 = [[0, 1, 0.016757644548097254, 1, None], [0.0474, 0.2562, 0.0749, 0.3165, 0.1273], [0.0552, 0.2349, 0.0837, 0.2916, 0.1278], [0.0725, 0.2042, 0.103, 0.2557, 0.1334], [0.1303, 0.1283, 0.1668, 0.1662, 0.1599], [0.1938, 0.1014, 0.2357, 0.134, 0.2037], [0.4091, 0.054, 0.4651, 0.0768, 0.3597], [1, 0, 1, 0.005634668729086153, None]]
    # data_p2 = [[0, 1, 0.010765612583032677, 1, None], [0.0464, 0.4114, 0.0702, 0.4758, 0.2722], [0.1207, 0.1566, 0.1552, 0.1955, 0.1705], [0.1835, 0.0693, 0.2255, 0.0965, 0.1544], [0.2163, 0.0494, 0.262, 0.0736, 0.1607], [0.335, 0.0179, 0.3931, 0.0361, 0.2054], [0.381, 0.0123, 0.4434, 0.0292, 0.2261], [0.5063, 0.0072, 0.5807, 0.0238, 0.2897], [1, 0, 1, 0.011397144996587096, None]]


    data_p1 = np.array(data_p1)
    data_p2 = np.array(data_p2)
    data_nolearn = np.array([[i*0.01, 1 - i*0.01] for i in range(0, 101)])

    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    ax.plot(data_p1[:, 0], data_p1[:, 1], 'b--', marker='o', linewidth=1.2)
    ax.plot(data_p2[:, 0], data_p2[:, 1], 'r--', marker='^', linewidth=1.2)
    ax.plot(data_p1[:, 2], data_p1[:, 3], 'b', marker='o', linewidth=1.2, alpha=1)
    ax.plot(data_p2[:, 2], data_p2[:, 3], 'r', marker='^', linewidth=1.2, alpha=1)
    ax.plot(data_nolearn[:, 0], data_nolearn[:, 1], 'k', linewidth=4)
    plt.legend(['Standard Setting', 'Occluded Obstacle Setting', "Guarantee in Standard Setting", "Guarantee in Occluded Obstacle Setting", 'Unlearned Performance'])

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.xlabel('FNR', fontsize=12)
    plt.ylabel('FPR', fontsize=12)
    plt.tight_layout()
    ax.set_aspect('equal')

    plt.savefig(location, dpi=200)


if __name__ == '__main__':
    
    filename = './plots/trajectories.png'
    visualize_primitives(filename)
    
    
    filename = './plots/navigationfnrfpr.png'
    # Output format from train_fd.py when evaluating bounds: 
    # [0.12826510151123985, 0.11233120589819647,    [0.064815, 0.035404, 0.711596, 0.188185], 0.0228001419454813,    0.0007600902459542082],
    # [Total Bound,         Bound on training loss, [fn rate,  fp rate,  tn rate,  tp rate],  PAC-Bayes regularizer, Sample convergence regularizer]
    # 
    # [0.07489146824172287,                  0.3164735588918691,              (0.049725484675662304, 0.016231617049779674),      (0.29348056032735725, 0.005035785672678275)],
    # [Class conditional success bound,      Class conditional failure bound, (success class loss,   success class regularizer), (failure class loss,  failure class regularizer)]

    data_p1 = [
    #	 Evaluating bound policy 1 fn_factor: 0.3
    [0.12826510151123985, 0.11233120589819647, [0.064815, 0.035404, 0.711596, 0.188185], 0.0228001419454813, 0.0007600902459542082],
    [0.07489146824172287, 0.3164735588918691, (0.049725484675662304, 0.016231617049779674), (0.29348056032735725, 0.005035785672678275)],
    #	 Evaluating bound policy 1 fn_factor: 0.4 ***
    [0.1277600993828157, 0.1128187614938471, [0.059418, 0.041265, 0.705735, 0.193582], 0.022772038355469704, 0.0007600902459542082],
    [0.08374299271386634, 0.2915908858191383, (0.0579700802870062, 0.01621184615024396), (0.26904297059719573, 0.0050293620981008945)],
    #	 Evaluating bound policy 1 fn_factor: 0.5
    [0.13342618966047173, 0.11820527729061284, [0.051665, 0.054149, 0.692851, 0.201335], 0.022760851308703423, 0.0007600902459542082],
    [0.1030328583898614, 0.2557394155797469, (0.07606313545726309, 0.016203645707700075), (0.2339423102763738, 0.005027107690487219)],
    #	 Evaluating bound policy 1 fn_factor: 0.601
    [0.1598546752926732, 0.1432795637668907, [0.032466, 0.097337, 0.649663, 0.220534], 0.022946685552597046, 0.0007600902459542082],
    [0.1668277421403448, 0.16624898088438203, (0.1367310361109878, 0.016335942703430598), (0.14701826414155075, 0.005067934138954709)],
    # 	 Evaluating bound policy 1 fn_factor: 1.001
    [0.20365604110994406, 0.18542268334042986, [0.025642, 0.14479, 0.60221, 0.227358], 0.022900793701410294, 0.0007600902459542082],
    [0.23567591962791792, 0.1340279776083477, (0.2033828510222922, 0.01630327190006674), (0.11609326432366371, 0.005057798606361295)],
    #	 Evaluating bound policy 1 fn_factor: 2
    [0.3596570107244912, 0.33762128165726124, [0.013669, 0.305599, 0.441401, 0.239331], 0.023066755384206772, 0.0007600902459542082],
    [0.4650531079934779, 0.07683024603306036, (0.4292685908723841, 0.016421421448720025), (0.06189527781998764, 0.005094452391330578)],
    ]
    data_p1 = process_data(data_p1)
    data_p1 = [[0, 1, 0.016757644548097254, 1, None], *data_p1, [1, 0, 1, 0.005634668729086153, None]]  # adding the trivial detectors


    data_p2 = [
    #	 Evaluating bound policy 2 fn_factor: 0.126
    [0.2722313950170442, 0.25087370546201787, [0.211553, 0.022547, 0.463253, 0.302647], 0.02419690415263176, 0.0007600902459542082],
    [0.07022844046789842, 0.475827029961441, (0.05041663552528909, 0.010822621996823885), (0.44489682557544574, 0.011505712738831829)],
    #	 Evaluating bound policy 2 fn_factor: 0.125
    [0.17048385345031197, 0.15303949649896126, [0.080541, 0.05864, 0.42716, 0.433659], 0.023523623123764992, 0.0007600902459542082],
    [0.1551579759938296, 0.1955195845007438, (0.13110561007551894, 0.010521481568813173), (0.16937835184044625, 0.011185565249641415)],
    #	 Evaluating bound policy 2 fn_factor: 0.3
    [0.15437705690803702, 0.1380647148692559, [0.035634, 0.089167, 0.396633, 0.478566], 0.022915301844477654, 0.0007600902459542082],
    [0.22549036792782642, 0.09652369837203333, (0.19936369799512318, 0.010249395883106345), (0.07493109853582194, 0.010896306349070965)],
    #	 Evaluating bound policy 2 fn_factor: 0.4
    [0.16068218750539123, 0.14402025510578229, [0.025424, 0.10509, 0.38071, 0.488776], 0.023018306121230125, 0.0007600902459542082],
    [0.2620463916874285, 0.07362973500225883, (0.23495717194468432, 0.010295466915347343), (0.05345912222230126, 0.010945285243714235)],
    #	 Evaluating bound policy 2 fn_factor: 0.8
    [0.2053757451424022, 0.1869618546467537, [0.009181, 0.162741, 0.323059, 0.505019], 0.02305486612021923, 0.0007600902459542082],
    [0.39305031290718356, 0.036129126290483954, (0.36384936875323937, 0.010311819215904782), (0.01930585137689715, 0.010962669651382557)],
    #	 Evaluating bound policy 2 fn_factor: 1
    [0.22611018311938533, 0.20702327514891042, [0.00631, 0.185067, 0.300733, 0.50789], 0.02305150404572487, 0.0007600902459542082],
    [0.4433673029286161, 0.029197153607332223, (0.41377413466364754, 0.010310315450747547), (0.013270144029217975, 0.010961070973175845)],
    #	 Evaluating bound policy 2 fn_factor: 1.67
    [0.28973955066547696, 0.2667742238109737, [0.003694, 0.245957, 0.239843, 0.510506], 0.02551979199051857, 0.0007600902459542082],
    [0.5806640301279287, 0.023830741751629914, (0.5499102294368118, 0.011414314013427893), (0.007760195161301796, 0.012134750542693381)],
    ]
    data_p2 = process_data(data_p2)
    data_p2 = [[0, 1, 0.010765612583032677, 1, None], *data_p2, [1, 0, 1, 0.011397144996587096, None]]  # adding the trivial detectors

    plot_fnrvsfpr(filename, data_p1, data_p2)
