
# coding: utf-8

# In[3]:



import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np




def hidden_init(layer):
    w_in = layer.weight.data.size()[0]    
    lim = 1./np.sqrt(w_in)
    return (-lim,lim)
    



class Network(nn.Module):
    def __init__(self, input_dim ,h1, h2, h3, output_dim , actor = False):
        super(Network,self).__init__()
        
        self.input = input_dim
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.output = output_dim
        self.actor = actor
        
        self.fc1 = nn.Linear(self.input,h1)
        self.fc2 = nn.Linear(self.h1,self.h2)
        self.fc3 = nn.Linear(self.h2,self.h3)
        self.fc4 = nn.Linear(self.h3, self.output)
        self.bn = nn.BatchNorm1d(self.h1)
        #self.reset_parameters()
        
        
        
    def reset_parameters(self):
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        
        self.fc4.weight.data.uniform_(-1e-3,1e3)
        
        
    def forward(self,x):
        
        if self.actor:
            if x.dim() == 1:
                x = torch.unsqueeze(x,0)
            
            x = f.relu(self.fc1(x))
            x = self.bn(x)
            x = f.relu(self.fc2(x))
            x = f.relu(self.fc3(x))
            return torch.tanh(self.fc4(x))
        else:
            
            x = f.relu(self.fc1(x))
            x = self.bn(x)
            x = f.relu(self.fc2(x))
            x = f.relu(self.fc3(x))
            return (self.fc4(x))
        

