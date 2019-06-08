#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code
        self.embed_size = embed_size
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.id2char), 50, padding_idx=pad_token_idx)
        self.CNN = CNN(50, embed_size, 21, 5)
        self.HW = Highway(embed_size, 0.3)
        ### YOUR CODE HERE for part 1j


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        x_emb = self.embeddings(input)  # (sent_len, batch_size, max_word_len, char_size)
        x_reshaped = x_emb.transpose(2, 3)  # (sent_len, batch_size, char_size, max_word_len)
        sent_len, batch_size, char_size, max_word_len = x_reshaped.shape
        x_convout = self.CNN(x_reshaped.view(-1, char_size, max_word_len))  # (batch_size*sent_len, word_embed_size)
        x_word = self.HW(x_convout)  # (sent_len*batch_size, word_embed_size)
        return x_word.view(sent_len, batch_size, -1)  # (sent_len, batch_size, word_embed_size)
        
        ### END YOUR CODE

