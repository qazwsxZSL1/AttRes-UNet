import torch
from torch import nn
from einops import rearrange
from math import sqrt
from math import log


class attbase(nn.Module):
    def __init__(self, input_dim, heads=None, dim_head=None, k_size=None):
        super(attbase, self).__init__()
        self.input_dim = input_dim
        if (heads == None) and (dim_head == None):
            raise ValueError("Can not be the same None")
        elif dim_head != None:
            heads = int(input_dim / dim_head)
        else:
            heads = heads
        self.heads = heads
        
        if k_size == None:
            t = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
        self.k_size = k_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False)
        self._norm_fact = 1 / sqrt(input_dim / heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x)
        y = y.squeeze(-1).transpose(-1, -2)
        Q = self.Wq(y)
        k = self.Wk(y)
        v = self.Wv(x)
        K = k
        V = v.unsqueeze(1)
        Q = Q.view(b, self.heads, 1, int(c/self.heads))
        K = rearrange(K, 'b t (g d) -> b g t d', b=b, g=self.heads, d=int(c/self.heads))
        V = rearrange(V, 'b t (g d) h w -> b g t d h w', g=self.heads, d=int(c/self.heads))
        atten = torch.einsum('... i d, ... j d -> ... i j', Q, K) * self._norm_fact
        atten = self.softmax(atten)
        V = rearrange(V, 'b g t d h w -> b g t (d h w)')
        attfeature = torch.einsum('bgit, bgtj -> bgij', atten, V)
        attfeature = attfeature.unsqueeze(2).reshape(b, c, h, w)
        return attfeature
    
