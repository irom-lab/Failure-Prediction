import torch
import numpy as np
from util_models import save_weights, load_weights
from models.policy import Policy
from dataloaders import DataLoader_Policy as dataloader
import warnings
warnings.filterwarnings('ignore')

def test_model(model, x, y, batch):
    model_output = model(x)
    prims = model_output.max(dim=1).indices
    emp_cost = 0
    for (j, prim) in enumerate(prims.tolist()):
        emp_cost += y[j, prim]
    emp_cost /= batch

    return round(float(emp_cost), 5)


policy_num = 2
policy_path = 'policy' + str(policy_num)
device = torch.device('cuda')
batch = 1000
test_batch = 1000


data_dict = {'train': ('_policy'+str(policy_num), 'dist_softmax'),
             'emp_test': ('_policy'+str(policy_num), 'prim_collision'),
             'test': ('_test'+str(policy_num), 'prim_collision')}
dl = dataloader(data_dict, device, batch)
criterion = torch.nn.BCELoss().to(device)


model = Policy().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

N = 1000

losses = []
missmatch = []

for n in range(1, int(N)+1):
    x, y = dl.get_batch('train')
    model_output = model(x)
    loss = criterion(model_output, y)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    losses.append(float(loss.to('cpu')))

    print('Iteration: {}, avg loss: {:.7f}'.format(n, np.mean(losses)), end='\r')

    if n % 100 == 0:
        print()
        x, y = dl.get_batch('emp_test', batch=test_batch)
        print(test_model(model, x, y, test_batch))
        x, y = dl.get_batch('test', batch=test_batch)
        print(test_model(model, x, y, test_batch))

        save_weights(model, save_file_name=policy_path)
print()
print("Done!")
