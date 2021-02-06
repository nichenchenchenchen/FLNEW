# FLNEW
Federated Learning for potential fairness
Start this project by 'python mainfederated.py' under project repository 
arguments:
'''
parser.add_argument('--dataset', required = True, help = 'celebA | GENKI | UTKface | DSprites')
parser.add_argument('--mode',required = True, help = 'Random | Up | Down')
parser.add_argument('--samrounds',required = True, help = 'Up to 500')
'''
Now this project supports four datasets
You may get no more than four plots standing for Demographic Parity, Accuracy and Loss of our federated model in certain weight-balanced testset  
