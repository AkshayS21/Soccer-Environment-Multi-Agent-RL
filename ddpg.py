
# coding: utf-8

# In[2]:


import model
import importlib
importlib.reload(model)
from OUNoise import OUNoise

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


import numpy as np


state_size = 24
action_size = 2
#h1 = 256
#h2 = 128
lr_act = 1e-3
lr_crt = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ddpg():
    
    def __init__(self, state_size, action_size, h1, h2 , h3, n_agents = 2  ):
        
        a1 = h1 #int(h1/2)
        a2 = h2  #int(h2/2)
        a3 = h3 #int(h3*4)
        self.action_size = action_size
        self.actor_local = model.Network(input_dim = state_size ,h1 = a1, h2=a2, h3=a3, output_dim = action_size , actor = True).to(device)
        self.actor_target = model.Network(input_dim = state_size ,h1 = a1, h2=a2, h3=a3, output_dim = action_size , actor = True).to(device)
        self.actor_optimizer = optim.Adam( self.actor_local.parameters(), lr = lr_act)
        
        critic_input = n_agents * (state_size + action_size)
        
        self.critic_local = model.Network(input_dim = critic_input ,h1 = h1, h2=h2,h3 = h3, output_dim = 1 ).to(device)
        self.critic_target = model.Network(input_dim = critic_input ,h1 = h1, h2=h2,h3 = h3, output_dim = 1 ).to(device)
        self.critic_optimizer = optim.Adam( self.critic_local.parameters(), lr = lr_crt)
        
        self.noise = OUNoise( action_size , scale=1.0)
        
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        self.tau = 1e-3
        
    def local_act(self , state , noise ,rand):
        
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()     #detach().numpy()  #data.numpy()
        self.actor_local.train() 
        if noise is not None:
            
            action +=  noise * self.noise.noise()
        if rand is not None:
            
            action = (1 - rand) * action + rand * (np.random.rand(self.action_size) - 0.5) * 2.0
            
        #action +=  noise * self.noise.noise()
        return np.clip(action, -1,1)
    
    def local_act2(self , state  ):
        
        #state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()     #detach().numpy()  #data.numpy()
        self.actor_local.train()            
        #action +=  noise * self.noise.noise()
        return action #np.clip(action, -1,1)
    
    
    
    def target_act(self, state  ):
        
        #state = torch.from_numpy(state).float().to(device)
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy() #detach().numpy() #data.numpy()
        self.actor_target.train()            
        #action +=   noise * self.noise.noise()
        return action #np.clip(action, -1,1) 
    
    def hard_update(self, target, source):
        
        for target_params, source_params in zip( target.parameters(), source.parameters()):
            target_params.data.copy_(source_params.data)
            
            
    def soft_update(self, target, local ):
        
        for target_params, local_params in zip(target.parameters() , local.parameters()):
            target_params.data.copy_( (1-self.tau)* target_params.data + (self.tau* local_params.data))
            
            
    def reset_action(self):
        
        self.noise.reset()
        
        
            
            
        
        
        
    
    
    
    

