#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import numpy as np

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, word_embed_size: int, dropout_rate: float = 0.5, is_test: bool = False):
        """
        @param word_embed_size (int): input size/ output word embedding size
        @param dropout_rate (float): dropout rate
        @param is_test (bool): testing mode
        """
        super(Highway, self).__init__()
        self.input_size = word_embed_size
        self.proj = nn.Linear(self.input_size, self.input_size)
        self.ReLU = nn.ReLU()
        self.gate = nn.Linear(self.input_size, self.input_size)
        self.sigmoid = nn.Sigmoid()
        self.test = is_test
        if is_test:
            self.proj.weight = nn.Parameter(torch.ones_like(self.proj.weight))
            self.proj.bias = nn.Parameter(torch.zeros_like(self.proj.bias))
            self.gate.weight = nn.Parameter(torch.ones_like(self.gate.weight))
            self.gate.bias = nn.Parameter(torch.zeros_like(self.gate.bias))
            dropout_rate = 0.0
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_convout):
        """
        @param x_convout: tensor of (batch_size, word_embed_size)

        @returns x_wordemb: tensor of (batch_size, word_embed_size)
        """
        x_proj = self.ReLU(self.proj(x_convout))
        x_gate = self.sigmoid(self.gate(x_convout))
        x_highway = x_gate*x_proj + (1 - x_gate)*x_convout
        if self.test:
            return self.dropout(x_highway), x_proj, x_gate
        else:
            return self.dropout(x_highway)

def test():
    H = Highway(5, is_test=True)
    x = torch.tensor([[1.0]*5, [-2.0]*5]) # batch of 2 examples/words reps. Each word has 5 elements
    y, x_proj, x_gate = H(x)
    x_proj_t = np.array([[ 5.,  5.,  5.,  5.,  5.],
               [-0., -0., -0., -0., -0.]])
    x_gate_t = np.array([[9.93307149e-01, 9.93307149e-01, 9.93307149e-01, 9.93307149e-01, 9.93307149e-01], 
        [4.53978687e-05, 4.53978687e-05, 4.53978687e-05, 4.53978687e-05, 4.53978687e-05]])
    y_t = np.array([[ 4.9732286,  4.9732286,  4.9732286,  4.9732286,  4.9732286],
               [-1.9999092, -1.9999092, -1.9999092, -1.9999092, -1.9999092]])
    assert x_proj_t.shape == x_proj.detach().numpy().shape, "Wrong dimensions for x_proj!"
    assert np.allclose(x_proj_t, x_proj.detach().numpy()), "Wrong values for proj!"
    assert x_gate_t.shape == x_gate.detach().numpy().shape, "Wrong dimensions for x_gate!"
    assert np.allclose(x_gate_t, x_gate.detach().numpy()), "Wrong values for x_gate!"
    assert y_t.shape == y.detach().numpy().shape, "Wrong dimensions for output!"
    assert np.allclose(y_t, y.detach().numpy()), "Wrong values for output!"
    print("Passed all test!")

def main():
    print("Running tests...")
    test()

if __name__ == "__main__":
    main()

        
### END YOUR CODE 

