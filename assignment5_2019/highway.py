#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn


### YOUR CODE HERE for part 1h
class Highway(nn.module):
    def __init__(word_embed_size: int, dropout_rate: float = 0.5):
        """
        @param word_embed_size (int): input size/ output word embedding size
        @param dropout_rate (float): dropout rate

        """
        super(Highway, self).__init__()
        self.input_size = word_embed_size
        self.proj = nn.Linear(self.input_size, self.input_size)
        self.ReLU = nn.ReLU()
        self.gate = nn.Linear(self.input_size, self.input_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(x_convout):
        """
        @param x_convout: tensor of (max_sentence_length, batch_size, word_embed_size)

        @returns x_wordemb: tensor of (max_sentence_length, batch_size, word_embed_size)
        """
        x_proj = self.ReLU(self.proj(x_convout))
        x_gate = self.sigmoid(self.gate(x_convout))
        x_highway = x*x_proj + (1 - x_gate)*x_convout
        return self.dropout(x_highway)

### END YOUR CODE 

