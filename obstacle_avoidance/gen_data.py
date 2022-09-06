import scipy.io
import numpy as np
import time
from models.policy import Policy
from util_models import load_weights
import torch
import warnings
from scipy.special import softmax
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# Note: all distances are in mm
img_size = 50
ths = np.linspace(-60, 60, img_size)
phis = np.linspace(-60, 60, img_size)
maxrng = 7000  # range of depth sensor in mm
x_lim = [-3500, 3500]
y_lim = [4000, maxrng]
obs_radius = 235
origin = [0, 0]


def visualize(deptharray=None, obs=None, traj=None):
    if deptharray is not None:
        plt.figure()
        plt.imshow(deptharray)
    if obs is not None:
        fig, ax = plt.subplots()
        for i in range(len(obs[0])):
            o = obs[:, i]
            circle = plt.Circle(o[:2], o[2], color='k')
            ax.add_patch(circle)
        plt.xlim([-3500, 3500])
        plt.ylim([0, 10000])
        ax.set_aspect(1)
        if traj is not None:
            plt.plot(traj[:, 0], traj[:, 1])

    plt.show()


def sample_obstacles_random(num_obs, x_lim, y_lim, obs_radius, step=1, obs=None, traj=None):
    if step == 1:
        xa = x_lim[0] + obs_radius
        xb = x_lim[1] - obs_radius
        xs = (xb - xa) * np.random.random(num_obs) + xa
        ya = y_lim[0] + obs_radius
        yb = y_lim[1] - obs_radius
        ys = (yb - ya) * np.random.random(num_obs) + ya

        obs = np.zeros((3, num_obs))
        obs[0, :] += xs
        obs[1, :] += ys
        obs[2, :] += obs_radius

        return obs
    else:
        return obs


# adversarial env generation
def sample_obstacles_occlusion(num_obs, x_lim, y_lim, obs_radius, step=1, obs=None, traj=None):
    num_occlusion = int(num_obs / 2)

    if step == 1:
        obs = sample_obstacles_random(num_obs - num_occlusion, x_lim, y_lim, obs_radius)
        return obs
    else:

        obs = np.transpose(obs)

        for i in range(num_occlusion):
            o = obs[i]

            scale = np.random.random() * 0.4 + 0.1  # if we can't specifically interfere with the traj

            m = o[1]/o[0]
            for j in range(len(traj[:, 0])):  # try to interfere
                if traj[j, 1] > o[1] and abs(m * traj[j, 0] - traj[j, 1]) < obs_radius:
                    scale = traj[j, 1] / o[1] - 1
                    break

            new_obs = np.array(o + [scale*o[0], scale*o[1], 0])
            obs = np.concatenate((obs, [new_obs]), axis=0)

        obs = np.transpose(obs)

        return obs


def get_min_dist(obs, traj):
    # minimum distance of the obstacles using full state knowledge
    obs_centers = obs[:2]
    obs_radii = obs[2]
    num_obs = len(obs_radii)

    r = 167.5 # approximate radius of swing
    dist = np.zeros((num_obs, len(traj)))

    for i in range(num_obs):
        center = obs_centers[:, i]
        radius = obs_radii[i]

        delx = traj[:, 0] - center[0]
        dely = traj[:, 1] - center[1]

        dist[i,:] = np.sqrt(delx**2 + dely**2) - r - radius

    min_dist = np.min(dist, axis=0)
    min_dist = np.minimum.accumulate(min_dist)

    return min_dist


def get_depth_matrix(imu, ths, phis, obs, maxrng):
    ths = ths % 360
    x0 = imu[0]
    y0 = imu[1]
    heading = imu[5] + 180
    deptharray = maxrng * np.ones((len(phis), len(ths)))

    nobs = len(obs[0])

    horizontal_angle = (ths + heading) % 360 # viewing angle for this depth sensor
    vertical_angle = (phis % 360) * np.pi / 180
    # Slope for projection of the ray on the horizontal plane
    # print(np.tan((horizontal_angle) * np.pi / 180))
    # exit()
    m = np.tan((90 + horizontal_angle) * np.pi / 180)
    # m = np.tan(horizontal_angle * np.pi / 180)

    for ob in range(nobs):
        a = obs[0, ob]
        b = obs[1, ob]
        r = obs[2, ob]

        d = y0 - m * x0

        # begin check for intersection between like and obs
        delta = np.maximum(r**2 * (1 + m**2) - (b - m*a - d)**2, np.zeros((len(m),)))

        x1 = (a + b*m - d*m + np.sqrt(delta)) / (1 + m**2)
        y1 = (d + a*m + b*m**2 + m * np.sqrt(delta)) / (1 + m**2)
        x2 = (a + b*m - d*m - np.sqrt(delta)) / (1 + m**2)
        y2 = (d + a*m + b*m**2 - m * np.sqrt(delta)) / (1 + m**2)
        centx = x1 - x0
        centy = y1 - y0
        # something going wrong with this function, most of the distances seem to be changing blerg

        con1 = np.logical_and(np.logical_or(270 <= horizontal_angle, horizontal_angle < 90), centy >= 0)
        con2 = np.logical_and(np.logical_and(90 <= horizontal_angle, horizontal_angle < 270), centy <= 0)
        con3 = np.logical_and(np.logical_and(0 <= horizontal_angle, horizontal_angle < 180), centx <= 0)
        con4 = np.logical_and(180 <= horizontal_angle, centx >= 0)

        ints = np.logical_and(np.sum(np.array([con1, con2, con3, con4]), axis=0) == 2, delta > 0)
        lst = np.where(ints == 1)[0]
        for j in np.where(ints == 1)[0]:
            dist1 = np.sqrt((x0 - x1[j])**2 + (y0 - y1[j])**2)
            dist2 = np.sqrt((x0 - x2[j])**2 + (y0 - y2[j])**2)
            # Project dist to incorporate vertical translation as well
            dist1 = dist1 / np.cos(vertical_angle)
            dist2 = dist2 / np.cos(vertical_angle)

            deptharray[:, j] = np.minimum(np.minimum(deptharray[:, j].T, dist1), dist2).T

    deptharray /= maxrng
    deptharray = np.flip(deptharray, axis=1)  # ToDo do I need this?, compare matlab and python outputs for fixed obstacles
    return deptharray


def generate_data(policy_num, num_envs, num_obs, add_app):
    # prims = scipy.io.loadmat('./data/trajectory_primitives.mat')['prims'][0][:]
    file_name = 'policy'+str(policy_num)+'_prim'
    prims = scipy.io.loadmat('./data/'+file_name+'.mat')[file_name]
    prims = np.array(prims)
    num_prims = prims.shape[0]
    num_trials = prims.shape[1]
    t = prims.shape[2]

    depth_maps = np.zeros((num_envs, img_size, img_size))
    min_dists = np.zeros((num_envs, num_prims))
    trajectories = np.zeros((num_envs, num_prims, t, 6))

    for i in range(num_envs):
        if i % 100 == 0 and i > 0:
            print(i)
        while True:
            obs = sample_obstacles_random(num_obs, x_lim, y_lim, obs_radius)

            for j in range(num_prims):
                k = np.random.randint(num_trials)
                traj = prims[j][k]
                trajectories[i, j, :, :] = traj[:t, :]
                min_dist = get_min_dist(obs, traj)
                min_dists[i, j] = min(min_dist)

            mds = min_dists[i, :]
            if max(mds) > 100:
                cur_pos = trajectories[i, 0, 0]  # + np.array([0, 0, 1000, 0, 0, -180])
                deptharray = get_depth_matrix(cur_pos, ths, phis, obs, maxrng)
                depth_maps[i, :, :] = deptharray
                break

    dist_softmax = softmax(min_dists/1e3, axis=1)
    prim_collision = (min_dists <= 0).astype(float)

    add_app += str(policy_num)
    np.save('./data/depth_maps' + add_app + '.npy', depth_maps)
    np.save('./data/dist_softmax' + add_app + '.npy', dist_softmax)
    np.save('./data/prim_collision' + add_app + '.npy', prim_collision)


def generate_data_multi(policy, policy_num, num_envs, num_obs, num_imgs, sample_obstacles, add_app):
    # prims = scipy.io.loadmat('./data/trajectory_primitives.mat')['prims'][0][:]
    file_name = 'policy'+str(policy_num)+'_prim'
    prims = scipy.io.loadmat('./data/'+file_name+'.mat')[file_name]
    prims = np.array(prims)
    # num_prims = prims.shape[0]
    num_trials = prims.shape[1]
    t = prims.shape[2]

    depth_maps = np.zeros((num_envs, num_imgs, img_size, img_size))
    min_dists = np.zeros((num_envs, num_imgs))
    trajectories = np.zeros((num_envs, t, 6))

    temp_obs = sample_obstacles(num_obs, x_lim, y_lim, obs_radius)
    obss = np.zeros((num_envs, 3, temp_obs.shape[1]))

    # Decide which motion primitive to use
    for i in range(num_envs):
        if i % 100 == 0 and i > 0:
            print(i)

        obs = sample_obstacles(num_obs, x_lim, y_lim, obs_radius, step=1, obs=None, traj=None)
        obss[i] = obs

        cur_pos = np.array([0, 0, 0, 0, 0, -180])
        deptharray = get_depth_matrix(cur_pos, ths, phis, obs, maxrng)
        depth_maps[i, 0, :, :] = deptharray

    depth_map_tensor = torch.Tensor(depth_maps[:, 0, :, :].astype('float32')).unsqueeze(1).to('cuda')
    prim_choice = policy(depth_map_tensor).max(dim=1).indices.to('cpu').numpy()

    # With given motion primitive, generate images for that trajectory
    for i in range(num_envs):
        if i % 100 == 0 and i > 0:
            print(i)

        splits = np.floor(np.linspace(0, t, num_imgs + 1)).astype(int)

        j = prim_choice[i]
        obs = obss[i]

        k = np.random.randint(num_trials)
        traj = prims[j][k]
        obs = sample_obstacles(num_obs, x_lim, y_lim, obs_radius, step=2, obs=obs, traj=traj)

        k = np.random.randint(num_trials)
        traj = prims[j][k]
        trajectories[i, :, :] = traj[:t, :]

        # visualize(None, obs, traj)

        min_dist = get_min_dist(obs, traj)
        min_dists[i, 0] = min_dist[splits[0]]

        for img in range(1, num_imgs):
            min_dists[i, img] = min_dist[splits[img]]
            if min_dists[i, img] > 0:
                cur_pos = trajectories[i, splits[img]]
                cur_pos = cur_pos  # + np.array([0, 0, 1000, 0, 0, -180])
                deptharray = get_depth_matrix(cur_pos, ths, phis, obs, maxrng)
            else:
                # don't change the depth image if we've crashed already, otherwise after the crash it doesn't make sense
                deptharray = np.zeros((img_size, img_size))

            depth_maps[i, img, :, :] = deptharray

    # exit()
    prim_collision = (min_dists <= 0).astype(float)
    add_app += str(policy_num)
    np.save('./data/depth_maps_multi' + add_app + '.npy', depth_maps)
    np.save('./data/prim_collision_multi' + add_app + '.npy', prim_collision)


if __name__ == "__main__":

    '''
    # Data generation for policy
    generate_data(policy_num=1, num_envs=20000, num_obs=10, add_app="_policy")
    generate_data(policy_num=1, num_envs=2000, num_obs=10, add_app="_test")
    generate_data(policy_num=2, num_envs=20000, num_obs=10, add_app="_policy")
    generate_data(policy_num=2, num_envs=2000, num_obs=10, add_app="_test")
    exit()
    '''

    device = torch.device('cuda')
    num_imgs = 30  # 20 Hz for 1.5s worth of motion primitives

    for policy_num in [2, 1]:
        policy = Policy().to(device)
        load_weights(policy, device, 'policy'+str(policy_num))
        num_obs = 12 if policy_num == 1 else 16

        sample_obstacles = sample_obstacles_random if policy_num == 1 else sample_obstacles_occlusion

        ''' 
        # Data generation for testing purposes
        generate_data_multi(policy, policy_num, 2000, num_obs, num_imgs, sample_obstacles, add_app="_priorsmall")
        generate_data_multi(policy, policy_num, 2000, num_obs, num_imgs, sample_obstacles, add_app="_postsmall")
        generate_data_multi(policy, policy_num, 2000, num_obs, num_imgs, sample_obstacles, add_app="_testsmall")
        '''

        generate_data_multi(policy, policy_num, 10000, num_obs, num_imgs, sample_obstacles, add_app="_prior")
        generate_data_multi(policy, policy_num, 10000, num_obs, num_imgs, sample_obstacles, add_app="_post")
        generate_data_multi(policy, policy_num, 20000, num_obs, num_imgs, sample_obstacles, add_app="_test")






























