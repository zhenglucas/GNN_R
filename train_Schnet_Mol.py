import os.path as osp
import time
import torch
import numpy as np

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import SchNet
# from model.Nequip_simple import Nequip_simple
# from model.Vector_Capsule import Vector_Capsule
from Schnet import SchNet
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import sys
sys.path.append("./data")
from AtomicData import ASEDataset,MolDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



seed_everything(234)

datas=MolDataset(root='/share/home/zhengss/20221103SchNet/out',dir_name='/share/home/zhengss/20221103SchNet/in')
# datas=ASEDataset(root='./data/CuO',file_name='./data/CuO.xyz',r_cut=5)
model=SchNet(cutoff=10.0,pbc=False,calc_force=False).to(device)
ema=EMA(model,0.99)
ema.register()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
'''
It seems like train_test_split will make bug, when batch_size=64, it may sometime get a wrong 63. We should check.
'''
train_data,valid_data=train_test_split(datas,train_size=0.9,random_state=456,shuffle=True)
# train_ratio=0.8
# train_data=datas[:int(len(datas)*train_ratio)]
# valid_data=datas[int(len(datas)*train_ratio):]
train_loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0)
valid_loader=DataLoader(valid_data,batch_size=64,shuffle=False,num_workers=0)
model.train()



for epoch in range(80):
    start_time=time.time()
    train_loss_E = 0
    valid_loss_E = 0
    iter_train = 0
    iter_valid = 0

    for iter in train_loader:
        state='training'
        model.train()
        iter=iter.to(device)
        # shift_param=torch.tensor([0.5216150283813477]).to(device)
        optimizer.zero_grad()
        # out=model(iter.z.to(torch.long),iter.pos,iter.batch)
        out=model(iter.z.to(torch.long),iter.pos,iter.edge_index,None,None,iter.batch)
        # loss_E = F.mse_loss(out,iter.energy_tot.reshape(-1,1)/shift_param)
        loss_E = F.mse_loss(out,iter.total_energy)
        loss=loss_E
        loss.backward()
        optimizer.step()
        ema.update()
        train_loss_E+=loss_E.item()
        iter_train+=1
    train_E_RMSE=(train_loss_E/iter_train)**0.5


    for iter in valid_loader:
        state='validation'
        model.eval()
        iter = iter.to(device)
        ema.apply_shadow()
        out=model(iter.z.to(torch.long),iter.pos,iter.edge_index,None,None,iter.batch)
        loss_E = F.mse_loss(out.detach(),iter.total_energy)
        ema.restore()
        valid_loss_E+=loss_E.item()
        iter_valid+=1
    valid_E_RMSE=(valid_loss_E/iter_valid)**0.5
    final_time=time.time()
    print(f'EPOCH: {epoch} RMSE_E_T: {train_E_RMSE:.5f} RMSE_E_V: {valid_E_RMSE:.5f}')
   