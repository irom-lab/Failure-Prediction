import torch
import numpy as np
DATAPATH = "./data/"


def load_data(app='', option=None):
    app = app + ".npy"
    depth_maps = torch.tensor(np.load(DATAPATH + "depth_maps" + app), dtype=torch.float32).unsqueeze(1)

    if option is None:
        option = 'prim_cost'

    if option not in ['dist_softmax', 'prim_collision', 'prim_cost']:
        raise NotImplementedError

    costs = torch.tensor(np.load(DATAPATH + option + app), dtype=torch.float32)

    return depth_maps, costs


def load_data_multi(app=''):
    app = app + ".npy"
    depth_maps = torch.tensor(np.load(DATAPATH + "depth_maps_multi" + app), dtype=torch.float32).unsqueeze(2)
    costs = torch.tensor(np.load(DATAPATH + "prim_collision_multi" + app), dtype=torch.float32)

    return depth_maps, costs


class DataLoader():
    def __init__(self, device, batch=16):
        self.device = device
        self.batch = batch
        self.data = {}

    def get_batch(self, dataset, shuffle=True, batch=None):
        x, y = self.data[dataset]

        if batch is None:
            batch = self.batch

        x_batch = torch.empty((batch, *list(x.shape[1:])))
        y_batch = torch.empty((batch, *list(y.shape[1:])))
        if shuffle:
            inds = np.random.randint(0, x.shape[0], (batch,))
        else:
            inds = np.arange(0, x.shape[0])

        for i, ind in enumerate(inds):
            x_batch[i] = x[ind]
            y_batch[i] = y[ind]

        return x_batch.detach().to(self.device), y_batch.detach().to(self.device)


class DataLoader_Policy(DataLoader):
    def __init__(self, data_dict, device, batch=16):
        super().__init__(device, batch=batch)
        for key in data_dict:
            file_name, option = data_dict[key]
            if option is None:
                option = 'prim_cost'
            self.data[key] = load_data(app=file_name, option=option)


class DataLoader_FailureDetector(DataLoader):
    def __init__(self, data_dict, device, batch=16):
        super().__init__(device, batch=batch)

        for key in data_dict:
            depth_maps, costs = load_data_multi(app=data_dict[key])
            y = []

            for cost in costs:
                path_result = []
                for collision in cost:
                    add = [0, 1] if collision else [1, 0]
                    path_result.append(add)
                y.append(path_result)

            x = depth_maps
            y = torch.tensor(y, dtype=torch.float)
            self.data[key] = (x, y)
