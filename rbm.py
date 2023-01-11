import torch

class RBM():

    def __init__(self, num_v_nodes, num_h_nodes):
        # Parameters
        self.num_v_nodes = num_v_nodes
        self.num_h_nodes = num_h_nodes

        # Weights and Biases
        # Weight matrix has dim. (num_h_nodes x num_v_nodes)
        self.weights = torch.randn(num_h_nodes, num_v_nodes)
        self.v_biases = torch.ones(num_v_nodes)
        self.h_biases = torch.ones(num_h_nodes)

        # Momentum
        # same dimensions as above
        self.weights_mom = torch.zeros_like(self.weights)
        self.v_biases_mom = torch.zeros_like(self.v_biases)
        self.h_biases_mom = torch.zeros_like(self.h_biases)
