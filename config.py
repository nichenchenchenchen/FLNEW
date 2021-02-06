import argparse
import torch.nn as nn
import torch
from vit_pytorch import  ViT
import torch.nn.functional as F
import numpy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
# name, test split and train split, scale up step length, target_type, batch_size
class args():
    def __init__(self):
        self.groups = 3
        self.Groupscale = [0.9, 0.9, 0.5]
        self.stepLength = 0.001
        self.maxRounds = 500
        self.batchN_peround = 50
        self.contriSplit = 0.05
        self.targetLoc = 1
        self.groupLoc = 0
        self.in_chan = 3
        self.lr = 0.005
        self.batch_size = 50
        self.imshape_1 = int(128)
        self.imshape_2 = int(128)

    def ScaleChange(self, mode):
        if mode == 'Random':
            if numpy.random.rand() > 0.5:
                self.Groupscale[self.groups - 1] -= self.stepLength
            else:
                self.Groupscale[self.groups - 1] += self.stepLength
        if mode == 'Up':
            self.Groupscale[self.groups - 1] += self.stepLength
        if mode == 'Down':
            self.Groupscale[self.groups - 1] -= self.stepLength

# target is attractive, sensitive attr is black hair and  Blurry
class celebA_args(args):
    def __init__(self):
        super(celebA_args,self).__init__()
        self.name = "celebA"
        self.groups = 10
        self.groupSplits = [[(0, 30000)],
                            [(30000,50000)],
                            [(50000,70000)],
                            [(70000,90000)],
                            [(90000,110000)],
                            [(110000,130000)],
                            [(130000,150000)],
                            [(150000,170000)],
                            [(170000,190000)],
                            [(190000,202599)]]
        self.Groupscale = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.5]
        self.testSplit = 0.1
        self.stepLength = 0.0001
        self.maxRounds = 5000
        self.out_dim = 2

#target is smile or not, sentive attr is pose
class GENKI_args(args):
    def __init__(self):
        super(GENKI_args,self).__init__()
        self.name = "GENKI"
        self.groups = 3
        self.groupSplits = [[(0,1000),(2000,2500)],
                            [(1000,1800),(2500,3200)],
                            [(1800,2000),(3200,3999)]]
        self.Groupscale = [0.9, 0.9, 0.5]
        self.testSplit = 0.2
        self.stepLength = 0.001
        self.out_dim = 2
#target is gender, sentive attr is age and race
class UTKface_args(args):
    def __init__(self):
        super(UTKface_args, self).__init__()
        self.name = "UTKface"
        self.groups = 4
        self.groupSplits = [[(0,500),(1000,1500),(2000,2500),(4000,4500),(6000,6500),(9000,9500)],
                            [(500,1000),(2500,3000),(4500,5000),(7000,7500),(8000,9000)],
                            [(3000,3500),(5000,6000),(6500,7000),(7500,8000)],
                            [(1500,2000),(3500,4000),(9500,10137)]]
        self.Groupscale = [0.9, 0.9, 0.9, 0.5]
        self.testSplit = 0.2
        self.stepLength = 0.001
        self.out_dim = 3
#target is shape, sentive attr is Xpos and Ypos
class DSprites_args(args):
    def __init__(self):
        super(DSprites_args,self).__init__()
        self.name = "DSprites"
        self.groups = 10
        self.groupSplits = [[(0, 30000),(700000,737280)],
                            [(30000, 50000),(650000,700000)],
                            [(50000, 70000),(600000,650000)],
                            [(70000, 90000),(550000,600000)],
                            [(90000, 110000),(500000,550000)],
                            [(110000, 130000),(450000,500000)],
                            [(130000, 150000),(400000,450000)],
                            [(150000, 170000),(350000,400000)],
                            [(170000, 190000),(250000,350000)],
                            [(190000, 250000)]]
        self.Groupscale = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.5]
        self.testSplit = 0.1
        self.stepLength = 0.0001
        self.maxRounds = 5000
        self.in_chan = 1
        self.out_dim = 4
        self.imshape_1 = int(64)
        self.imshape_2 = int(64)

class Conv(nn.Module):
    def __init__(self, out_dim ,imshape_1, imshape_2,in_chan = 3):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True))
        self.mlp = nn.Sequential(
            nn.Linear(64 * imshape_1 * imshape_2 // 256, 128),
            nn.ReLU(True),
            nn.Linear(128, out_dim))
        self.imshape_1 = imshape_1
        self.imshape_2 = imshape_2
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,64 * self.imshape_1 * self.imshape_2 // 256)
        x = self.mlp(x)
        out = F.log_softmax(x,dim = 1)
        return out
def normal(weights):
    sum = 0.0
    for i in range(len(weights)):
        sum += weights[i]
    for i in range(len(weights)):
        weights[i] = weights[i] / sum
    return weights
def testBench(model,args,testLoader,weights):
    accuracyCollection = []
    lossCollection = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    with torch.no_grad():
        totLoss = 0
        totCorrect = 0

        for i in range(args.groups):
            testLoss = 0
            correct = 0
            for data, anno in testLoader[i]:
                data, anno = data.to(device), anno.to(device)
                output = model(data)
                target = anno[:, 1]
                testLoss += F.nll_loss(output, target.long(), reduction='sum').item()  # sum up batch loss
                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()


            testLoss /= len(testLoader[i].dataset)
            accuracy = 100. * correct / len(testLoader[i].dataset)
            accuracyCollection.append(accuracy)
            lossCollection.append(testLoss)

            print('Testset{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                i,testLoss, correct, len(testLoader[i].dataset),accuracy))

            totLoss += weights[i] * testLoss
            totCorrect += weights[i] * accuracy

        print('Wholeset: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(
            totLoss, totCorrect))

    return totLoss,totCorrect,accuracyCollection,lossCollection