import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
from datetime import datetime
from gcn import GCN
from attacker import RND
from utils import *
from dataset import Dataset
from tqdm import tqdm
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('--inputFile', type=str, default='targetNodesList.txt', help='Target node file.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = int(datetime.now().timestamp())
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

current_dir = os.getcwd()
data = Dataset(root=current_dir, name='citeseer')
adj, features, labels = data.adj, data.features, data.labels

#print('adj',adj.shape) #(2110, 2110)
#print('features',features.shape) #(2110, 3703)
#print('labels',labels.shape) #(2110, )

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

#print('idx_train',idx_train.shape) #(210,)
#print('idx_val',idx_val.shape) #(211,)
#print('idx_test',idx_test.shape) #(1688,)

idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
#utils.preprocess_graph
adj_preprocess = preprocess_graph(adj)
#print('adj_preprocess test',adj_preprocess[:1])
surrogate = surrogate.to(device)
surrogate.fit(features, adj_preprocess, labels, idx_train, idx_val, patience=30)

w1,w2 = surrogate.get_weight()
#print('weight',w1.shape,w2.shape) #(3706, 16) (16, 6)
#dot如果是處理tensor，只能做一維，要讓他實現二維，要在這先把tensor轉成np.array
w1 = w1.cpu().detach().numpy() 
w2 = w2.cpu().detach().numpy()
# Setup target node list
target_nodes = []
with open(args.inputFile, 'r') as f:
	target_nodes = [int(n) for n in f.read().split('\n')] 

def single_test(adj, features, target_node):
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])

    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

def multi_test_poison():
    # test on 40 nodes on poisoining attack
    cnt = 0
    degrees = adj.sum(0).A1
    num = len(target_nodes)
    print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(target_nodes):
        # Setup Attack Model
        print('target node %d' % (target_node))
        n_perturbations = int(degrees[target_node]) + 2
        model = RND()
        model = model.to(device)
        model.attack(features, adj, labels, idx_train, w1, w2, target_node, n_added=1, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj
        modified_features = model.modified_features
        acc = single_test(modified_adj, modified_features, target_node)
        if acc == 0:
            cnt += 1
            print(colored('attack success', 'green'))
        else:
            print(colored('attack failed', 'red'))

    #print('misclassification rate : %s' % (cnt/num))
    return cnt, num
if __name__ == '__main__':
    """
    for i in range(3):
        multi_test_poison()
    """
    cnt, num = multi_test_poison()
    
    print('misclassification rate : %s' % (cnt/num))
