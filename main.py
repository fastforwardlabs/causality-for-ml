import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import random

from dataset import get_dataset, get_handler
from models import get_net
from train import Train
from torchvision import transforms
from datetime import datetime

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, fbeta_score, roc_curve, auc, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

seed = 123
dataset_name = 'WILDCAM'
dataset_path = './data/wildcam_denoised'

args_pool = {
    'WILDCAM': {
        'n_restarts': 1,
        'steps': 121,
        'n_classes': 2,
        'fc_only': True,
        'model_path': "./models/",
        'transform': {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
            'test': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        },
        'loader_tr_args': {
            'batch_size': 100, 
            'num_workers': 1
        },
        'loader_te_args': {
            'batch_size': 100, 
            'num_workers': 1
        },
        'loader_sample_args': {
            'batch_size': 100, 
            'num_workers': 1
        },
        'optimizer_args': {
            'lr': 0.001,
            'l2_regularizer_weight': 0.001,
            'penalty_anneal_iters': 0, # make this 0 for ERM
            'penalty_weight': 0.0 # make this 0 for ERM
        }
    }
}

args = args_pool[dataset_name]
model_name = "wildcam_denoised_" + str(args['steps']) + "_" + str(args['optimizer_args']['lr']) + "_" + str(args['optimizer_args']['penalty_anneal_iters']) + "_" + str(args['optimizer_args']['penalty_weight']) + "_"

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load dataset
envs, x_test, y_test = get_dataset(dataset_name, dataset_path)
print("\nworking with", len(envs), "training environments: ")
for env in envs:
    print("env['images']: ", len(env['images']))
    print("env['labels']: ", env['labels'].shape[0])
    unique, counts = np.unique(env['labels'], return_counts=True)
    if counts[0] >= counts[1]:
        baseline_acc = counts[0]/(counts[0]+counts[1])
        print_statement = str(round(baseline_acc, 2)) + " (always coyote)."
    else:
        baseline_acc = counts[1]/(counts[0]+counts[1])
        print_statement = str(round(baseline_acc, 2)) + " (always raccoon)."        
    print("class distribution: ", counts[0], "coyotes and", counts[1], "raccoons. baseline accuracy", print_statement)

print("x_test: ", len(x_test))
print("y_test: ", y_test.shape[0])
unique, counts = np.unique(y_test, return_counts=True)
if counts[0] >= counts[1]:
    baseline_acc = counts[0]/(counts[0]+counts[1])
    print_statement = str(round(baseline_acc, 2)) + " (always coyote)."
else:
    baseline_acc = counts[0]/(counts[0]+counts[1])
    print_statement = str(round(baseline_acc, 2)) + " (always coyote)."    
print("test class distribution: ", counts[0], "coyotes and", counts[1], "raccoons. baseline accuracy", print_statement)

# get model and data handler
net = get_net(dataset_name)
handler = get_handler(dataset_name)

# GPU enabled?
print("Using GPU - {}".format(torch.cuda.is_available()))

final_train_accs = []
final_test_accs = []
train_process = Train(envs, x_test, y_test, net, handler, args)
    
print()
if args['optimizer_args']['penalty_weight'] > 1.0:
    model_name = model_name + "IRM"
    print("========================================IRM========================================")
else:
    model_name = model_name + "ERM"
    print("========================================ERM========================================")
print(args)

for restart in range(args['n_restarts']):  
    print("\nRestart", restart)
    train_acc, test_acc, preds, probs = train_process.train()
    preds = torch.reshape(preds, (-1,)).long().detach().cpu().numpy()
    test_prec = precision_score(y_test, preds)
    test_rec = recall_score(y_test, preds)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)

    final_train_accs.append(train_acc)
    final_test_accs.append(test_acc)

print('Final train acc (mean/std across restarts so far):')
print(round(np.mean(final_train_accs), 3), round(np.std(final_train_accs), 3))
print('Final test acc (mean/std across restarts so far):')
print(round(np.mean(final_test_accs), 3), round(np.std(final_test_accs), 3))
print('Final test precision:')
print(round(test_prec, 4))
print('Final test recall:')
print(round(test_rec, 4))
    
print('confusion matrix:')
print(confusion_matrix(y_test.numpy(), preds))
tn, fp, fn, tp = confusion_matrix(y_test.numpy(), preds).ravel() 
print(f'tn = {tn}, fp = {fp}, fn = {fn}, tp = {tp}')

torch.save(train_process.clf.state_dict(), args['model_path'] + model_name + ".pth")
