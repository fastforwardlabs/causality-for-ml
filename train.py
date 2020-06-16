import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import autograd
from torch.utils.data import DataLoader
from collections import Counter

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, fbeta_score, roc_curve, auc, roc_auc_score
    
def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))    
        
class Train:
    def __init__(self, envs, X_te, Y_te, net, handler, args):
        self.envs = envs
        self.X_te = X_te
        self.Y_te = Y_te
        self.net = net
        self.handler = handler
        self.args = args
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def get_distribution(self):
        return self.class_distribution
    
    # Define loss function helpers
    def mean_nll(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean(), preds

    def penalty(self, logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = self.mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)
    
    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y,
                                            transform=self.args['transform']['test']),
                               shuffle=True, **self.args['loader_te_args'])
        self.clf.eval()
        total_loss = nll = acc = 0.0
        preds = torch.zeros(len(Y), 1, dtype=torch.float)
        preds_Y = torch.zeros(len(Y), 1, dtype=torch.float)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                y.resize_((y.shape[0], 1))
                train_nll = self.mean_nll(out, y.float())
                train_acc, temp_preds = self.mean_accuracy(out, y.float())
                
                nll += train_nll
                acc += train_acc

                probs = torch.sigmoid(out)
                if str(self.device) == 'cuda':
                    preds[idxs] = probs.cpu()
                    preds_Y[idxs] = temp_preds.cpu()
                else:
                    preds[idxs] = probs           
                    preds_Y[idxs] = temp_preds
        '''
        Traditional way of calculating accuracy, getting probs and then using a threshold. 
        Doesn't quite work the way it should or may be the results are bad with 0.5 threshold
        but saving the code in case we would want to retrieve probs instead of logits.
        '''
        predicted_vals = (preds >= 0.5).long()        
        predicted_acc = ((preds - Y).abs() < 1e-2).float().mean()

        return predicted_acc, nll/len(loader_te), acc/len(loader_te), preds_Y, preds     

    def train(self):        
        n_classes = self.args['n_classes']
        self.clf = self.net(n_classes=n_classes).to(self.device)
        if self.args['fc_only']: # feature extraction
            optimizer = optim.Adam(self.clf.fc.parameters(), self.args['optimizer_args']['lr'])
        else:
            optimizer = optim.Adam(self.clf.parameters(), self.args['optimizer_args']['lr'])
        
        pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test nll', 'test acc', 'test prec', 'test rec')

        for step in range(self.args['steps']):  
            for env_idx, env in enumerate(self.envs):
                x = env['images']
                y = env['labels']
                loader_tr = DataLoader(self.handler(x, y, transform=self.args['transform']['train']), 
                                       shuffle=True, **self.args['loader_tr_args'])
                self.clf.train()
                nll = acc = penalty = 0.0
                
                for batch_idx, (x, y, idxs) in enumerate(loader_tr):
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    logits = self.clf(x)
            
                    y.resize_((y.shape[0], 1))
                    train_nll = self.mean_nll(logits, y.float())
                    train_acc, _ = self.mean_accuracy(logits, y.float())
                    train_penalty = self.penalty(logits, y.float())
                    
                    nll += train_nll
                    acc += train_acc
                    penalty += train_penalty
                env['nll'] = nll / len(loader_tr)
                env['acc'] = acc / len(loader_tr)
                env['penalty'] = penalty / len(loader_tr)
                
            train_nll = torch.stack([self.envs[0]['nll'], self.envs[1]['nll']]).mean()
            train_acc = torch.stack([self.envs[0]['acc'], self.envs[1]['acc']]).mean()
            train_penalty = torch.stack([self.envs[0]['penalty'], self.envs[1]['penalty']]).mean()
            weight_norm = torch.tensor(0.).cuda()
            
            if self.args['fc_only']:
                for w in self.clf.fc.parameters():
                    weight_norm += w.norm().pow(2)
            else:
                for w in self.clf.parameters():
                    weight_norm += w.norm().pow(2)     
            
            loss = train_nll.clone()
            loss += self.args['optimizer_args']['l2_regularizer_weight'] * weight_norm
            penalty_weight = (self.args['optimizer_args']['penalty_weight'] 
                              if step >= self.args['optimizer_args']['penalty_anneal_iters'] else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight

            loss.backward()
            optimizer.step()
            
            _, test_loss, test_acc, preds, probs = self.predict(self.X_te, self.Y_te)
            
            #acc_test = accuracy_score(self.Y_te.detach().cpu().numpy(), preds.detach().cpu().numpy())
            test_prec = precision_score(self.Y_te.detach().cpu().numpy(), preds.detach().cpu().numpy())
            test_rec = recall_score(self.Y_te.detach().cpu().numpy(), preds.detach().cpu().numpy())
           
            if step % 10 == 0:
                pretty_print(np.int32(step), train_nll.detach().cpu().numpy(), 
                             train_acc.detach().cpu().numpy(), train_penalty.detach().cpu().numpy(), 
                             test_loss.detach().cpu().numpy(), test_acc.detach().cpu().numpy(),
                             test_prec, test_rec)
                
        return train_acc.detach().cpu().numpy(), test_acc.detach().cpu().numpy(), preds, probs
