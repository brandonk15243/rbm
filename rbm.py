import numpy as np
import torch
import torch.nn as nn

# Create model
# Based on Hinton's 'A Practical Guide to Training Restricted Boltzmann Machines'
class RBM():
    def __init__(self, num_vis, num_hid, k=1, learning_rate=1e-3, batch_size=1):
        # Params
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.k = k
        self.learning_rate = learning_rate
        self.error = 0
        self.batch_size = batch_size
        
        # Weights and Biases
        # STD of weights should be ~0.01
        self.W = torch.randn(num_vis, num_hid) * np.sqrt(np.sqrt(.1))
        # (Hinton Section 8)
        self.vis_bias = torch.ones(num_vis) * 0.15
        # Initial hidden biases of 0 is fine
        self.hid_bias = torch.zeros(num_hid)
    
    def sample_hidden(self, visible):
        # Get hidden prob. given visible prob.
        # Nodes are activated (=1) if node's prob. is greater than random number generated on [0,1)
        hid_prob = self.sigmoid(nn.functional.linear(visible, self.W.t(), self.hid_bias))
        hid_act = (hid_prob > self.rand_nums(self.num_hid)).float()
        return hid_prob, hid_act
    
    def sample_visible(self, hidden):
        # Get visible prob. and activation given hidden prob.
        # Nodes are activated (=1) if node's prob. is greater than random number generated on [0,1)
        vis_prob = self.sigmoid(nn.functional.linear(hidden, self.W, self.vis_bias))
        vis_act = (vis_prob > self.rand_nums(self.num_vis)).float()
        return vis_prob, vis_act
    
    def train(self, input_data):
        # First forward pass
        # Collect positive statistic <p_ip_j>_{data}
        pos_hid_prob, pos_hid_act = self.sample_hidden(input_data)
        pos_statistic_data = torch.matmul(input_data.t(),pos_hid_prob)
        
        # Contrastive Divergence k-times
        # "Reconstruction"
        hid_act = pos_hid_act
        for i in range(self.k):
            # Use hidden activations when getting visible prob.
            vis_prob = self.sample_visible(hid_act)[0]
            hid_prob, hid_act = self.sample_hidden(vis_prob)
        
        # Last pass
        # Collect negative statistic <p_ip_j>_{reconstructed}
        neg_statistic_recon = torch.matmul(vis_prob.t(), hid_prob)
        
        # Update weights
        # (Hinton) When using mini-batches, divide by size of mini-batch
        self.W += (self.learning_rate / self.batch_size) * (pos_statistic_data - neg_statistic_recon)
        
        # Update biases
        self.vis_bias += (self.learning_rate / self.batch_size) * torch.sum(input_data - vis_prob, dim=0)
        self.hid_bias += (self.learning_rate / self.batch_size) * torch.sum(pos_hid_prob - hid_prob, dim=0)
        
        # Compute and report squared error
        self.error = torch.sum((input_data - vis_prob)**2)
        return self.error

    def free_energy(self, v):
        visible_bias_term = torch.matmul(v.t(),self.vis_bias)
        hidden_input_term = torch.linear(v,self.W.t(),self.hid_bias)
        hidden_term = torch.sum(torch.log(1+torch.exp(hidden_input_term)))
        return -visible_bias_term - hidden_term
    
    def sigmoid(self, x):
        # Activation func. is sigmoid
        return 1/(1+torch.exp(-x))
    
    def rand_nums(self, num):
        # Return num random numbers generated on [0,1)
        return torch.rand(num)
