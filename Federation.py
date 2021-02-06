import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import  DataLoader
import numpy
from torch.utils.data import Subset
from torchvision import datasets, transforms
from PIL import Image
import os
import collections
from random import shuffle
from config import Conv
import torch.nn.functional as F
import threading
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def localTrain(Model,trainloader,args):
    optimizer = optim.Adam(Model.parameters(), lr=args.lr)
    batch_index = 0
    for data, anno in trainloader:
            optimizer.zero_grad()
            data, anno = data.to(device), anno.to(device)
            out = Model(data)
            target = anno[:, 1]
            loss = F.nll_loss(out, target.long())
            loss.backward()
            optimizer.step()
            batch_index += 1
            if batch_index > args.batchN_peround:
                break

def federation(fedModel,args,trainLoader,weights):
    model = []
    thread = []
    for trainloader in trainLoader:
        Model = Conv(out_dim=args.out_dim, imshape_1=args.imshape_1, imshape_2=args.imshape_2,
                            in_chan=args.in_chan)
        Model.load_state_dict(fedModel.state_dict())
        Model.to(device)
        model.append(Model)
        worker = threading.Thread(target = localTrain,args=(Model,trainloader,args))
        worker.start()
        thread.append(worker)

    for t in thread:
        t.join()

    worker_state_dict = [x.state_dict() for x in model]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0.0
        for i in range(len(model)):
            key_sum += weights[i] * worker_state_dict[i][key]
        fed_state_dict[key] = key_sum
    #### update fed weights to fl model
    fedModel.load_state_dict(fed_state_dict)

    return fedModel