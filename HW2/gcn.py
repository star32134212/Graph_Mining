
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, n_classes, l1 = 128, l2 = 64, l3 = 16):
        super(GCN, self).__init__()
        # input layer
        self.layer1 = GraphConv(in_feats, l1, activation=F.relu)
        # hidden layers
        self.layer2 = GraphConv(l1, l2, activation=F.relu)
        # hidden layers
        self.layer3 = GraphConv(l2, l3, activation=F.relu)
        # output layer
        self.layer4 = GraphConv(l3, n_classes)
        self.dropout = nn.Dropout(p = 0.5)
        print(', l1:',l1,', l2:',l2,', l3:',l3)
    def forward(self, g, x):
        h = self.layer1(g, x)
        h = self.dropout(h)
        h = self.layer2(g, h)
        h = self.dropout(h)
        h = self.layer3(g, h)
        h = self.dropout(h)
        h = self.layer4(g, h)
        return h
