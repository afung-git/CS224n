#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn


### YOUR CODE HERE for part 1h
class Highway(nn.module):
    def __init__(word_embed_size: int, dropout_rate: float = 0.5, is_test: bool = False):
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
        if is_test:
            self.proj.weight = nn.Parameter(torch.ones_like(self.proj.weight)))
            self.proj.bias = nn.Parameter(torch.zeros_like(self.proj.bias))
            self.gate.weight = nn.Parameter(torch.ones_like(self.gate.weight)))
            self.gate.bias = nn.Parameter(torch.zeros_like(self.gate.bias))
            dropout_rate = 0.0
        self.dropout = nn.Dropout(dropout_rate)

    def forward(x_convout):
        """
        @param x_convout: tensor of (batch_size, word_embed_size)

        @returns x_wordemb: tensor of (batch_size, word_embed_size)
        """
        x_proj = self.ReLU(self.proj(x_convout))
        x_gate = self.sigmoid(self.gate(x_convout))
        x_highway = x_convout*x_proj + (1 - x_gate)*x_convout
        return self.dropout(x_highway)

def test():
    H = Highway(5, is_test=True)
    x = torch.tensor([[1]*5, [-2]*5]) # batch of 2 examples/words reps. Each word has 5 elements
    y = H(x)
    y_t = np.array([[ 5.99330715e+00,  5.99330715e+00,  5.99330715e+00,
        5.99330715e+00,  5.99330715e+00], [-9.07957374e-05, -9.07957374e-05,
        -9.07957374e-05, -9.07957374e-05, -9.07957374e-05]])
    assert y_t.shape == y.numpy().shape, "Wrong dimensions for output!"
    assert np.allclose(y_t, y.numpy()), "Wrong values for output!"


        
### END YOUR CODE 

