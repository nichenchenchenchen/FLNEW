from config import celebA_args,GENKI_args,UTKface_args,DSprites_args,testBench,normal
from Federation import federation
from makeDataset import fedDataset,init
import argparse
from vision import dynamicpainting
import os
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

fedModel = modelPretrained
totlossPainting = dynamicpainting(rounds,1)
for round in range(rounds):
    args.ScaleChange(mode)
    trainLoader = fedDataset(args)
    weights = [len(x.dataset) for x in trainLoader]
    weights = normal(weights)
    totLoss,totCorrect,accuracyCollection,lossCollection = testBench(fedModel, args, testLoader,weights)
    totlossPainting.addData(totCorrect)
    federation(fedModel,args,trainLoader,weights)


