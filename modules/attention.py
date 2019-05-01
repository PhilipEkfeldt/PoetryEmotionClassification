import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, v_dim):
        super().__init__()
        self.w_z = nn.Linear(hidden_dim*2, v_dim)
        self.w_w = nn.Linear(hidden_dim*2, v_dim)
        self.v = nn.Linear(v_dim, 1)
        self.tanh = torch.tanh
        self.softmax = torch.softmax
        self.hidden_dim = hidden_dim
    def forward(self, hidden, z):
        sum_w = self.w_w(hidden) + self.w_z(z)
        u = self.v(self.tanh(sum_w))
        attention = self.softmax(u, 0)
        h_att = torch.mul(attention.repeat(1,1,2*self.hidden_dim), hidden)
        return h_att, attention.squeeze()
        
  
class DualAttention(nn.Module):
    def __init__(self, hidden_dim, v_dim):
        super().__init__()
        self.w_z = nn.Linear(hidden_dim*2, v_dim)
        self.w_w = nn.Linear(hidden_dim*2, v_dim)
        self.v = nn.Linear(v_dim, 1)
        self.w_a = nn.Parameter(torch.tensor(0))
        self.tanh = torch.tanh
        self.softmax = torch.softmax
    def forward(hidden, z, alpha_s):
        sum_w = self.w_w(hidden) + self.w_z(z) + w_a*alpha_s
        u = self.v(self.tanh(sum_v))
        attention = self.softmax(u, 0)
        h_att = torch.mul(attention.unsqueeze(2).repeat(1,1,2*hidden_dim), h)
        return attention