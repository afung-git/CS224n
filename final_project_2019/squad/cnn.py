#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import numpy as np


### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, char_size: int, word_size: int, word_length: int, kernel_size: int, is_test: bool = False):
        """
        @param char_size (int): input char embedding size
        @param word_size (int): output word embedding size
        @param word_length (int): length of each word (in # of chars)
        @param kernel_size (int): kernel width for 1D convolution
        @param is_test (bool): testing mode
        """
        super(CNN, self).__init__()
        self.char_size = char_size
        self.word_size = word_size
        self.kernel_size = kernel_size
        self.word_length = word_length
        self.test = is_test

        self.conv = nn.Conv1d(self.char_size, self.word_size, self.kernel_size)
        if is_test:
            self.conv.weight = nn.Parameter(torch.ones_like(self.conv.weight))
            self.conv.bias = nn.Parameter(torch.zeros_like(self.conv.bias))
        self.ReLU = nn.ReLU()
        self.maxpool = nn.MaxPool1d(self.word_length - self.kernel_size + 1)

    def forward(self, x_reshape):
        """
        @param x_reshape: tensor of (batch_size, char_size, word_length)
        
        @return x_convout: tensor of (batch_size, word_size)
        """
        x_conv = self.conv(x_reshape)
        if self.test:
            return self.maxpool(self.ReLU(x_conv)).squeeze(), x_conv
        else:
            return self.maxpool(self.ReLU(x_conv)).squeeze()


def test():
    C = CNN(3, 5, 6, 4, is_test=True)
    x = torch.tensor([[[2.0,-1.0]*3]*3, [[-2.0, 1.0]*3]*3]) #2 words. Each word has 6 chars, each char 3 elements 
    y, x_conv = C(x)
    x_conv_t = np.array([[[6]*3]*5, [[-6]*3]*5])
    y_t = np.array([[6]*5, [0]*5])
    assert x_conv_t.shape == x_conv.detach().numpy().shape, "Wrong dimensions for x_conv!"
    assert np.allclose(x_conv_t, x_conv.detach().numpy()), "Wrong values for x_conv!"
    assert y_t.shape == y.detach().numpy().shape, "Wrong dimensions for output!"
    assert np.allclose(y_t, y.detach().numpy()), "Wrong values for output!"
    print("Passed all test!")


def main():
    print("Running tests...")
    test()


if __name__ == "__main__":
    main()
### END YOUR CODE

