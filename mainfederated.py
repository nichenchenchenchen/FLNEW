from config import celebA_args,GENKI_args,UTKface_args,DSprites_args,testBench,normal
from Federation import federation
from makeDataset import fedDataset,init
import argparse
import matplotlib.pyplot as plt
from vision import dynamicpainting
import os
import numpy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required = True, help = 'celebA | GENKI | UTKface | DSprites')
parser.add_argument('--mode',required = True, help = 'Random | Up | Down')
parser.add_argument('--samrounds',required = True, help = 'Up to 500')
opt = parser.parse_args()
dataset = opt.dataset
mode = opt.mode
rounds = int(opt.samrounds)

print("START:")
args = celebA_args()
if dataset == "GENKI":
    args = GENKI_args()
if dataset == "UTKface":
    args = UTKface_args()
if dataset == "DSprites":
    args = DSprites_args()
testLoader, contriData, modelPretrained = init(args)

plt.ion()
fedModel = modelPretrained
totLoss,totCorrect,accuracyCollection,lossCollection = testBench(fedModel, args, testLoader,weights = [1.0 / args.groups for i in range(args.groups)])
totlossPainting = dynamicpainting(rounds, 1, "totlossPainting", totLoss)
totcorrectPainting = dynamicpainting(rounds, 2, "totcorrectPainting", totCorrect)
for round in range(rounds):
    args.ScaleChange(mode)
    trainLoader = fedDataset(args)
    weights = [len(x.dataset) for x in trainLoader]
    weights = normal(weights)

    totLoss,totCorrect,accuracyCollection,lossCollection = testBench(fedModel, args, testLoader,weights)
    totlossPainting.addData(totLoss)
    totcorrectPainting.addData(totCorrect)
    fedModel,models = federation(fedModel,args,trainLoader,weights)
    #testBench(models[args.groups - 1], args, testLoader, weights=[1.0 / args.groups for i in range(args.groups)])

plt.ioff()
plt.show()