import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchcrf import CRF
from numpy import array
from random import shuffle
import pandas as pd

def create_emb_layer(weights_matrix, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def create_char_emb_layer(targetSize, embedding_dim=30, non_trainable=True):
    num_embeddings = targetSize
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class BilstmCrf(nn.Module):
    def __init__(self, input_size, hidden_size, weight_matrix, targetSize, dropt_out=0.5, tag_size = 34):
        super(BilstmCrf, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_matrix = weight_matrix
        self.targetSize = targetSize
        self.drop_out = nn.Dropout(dropt_out)
        self.embeddingC, num_embeddingsC, embedding_dimC = create_char_emb_layer(targetSize)
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(self.weight_matrix, True)
        self.encoder = nn.LSTM(input_size = input_size, hidden_size = hidden_size, bidirectional = True)
        self.estimates = nn.Linear(hidden_size*2, tag_size)
        self.model = CRF(34, batch_first=True)
        self.model.reset_parameters()
    
    def maskCreation(self, batch, seq_lengths):
        mask = torch.zeros(batch.size(0), batch.size(1), dtype=torch.uint8)
        for i in range(batch.size(0)):
            for j in seq_lengths:
                for k in range(j):
                    mask[i, k] = 1
        return mask.to('cuda:0')
        
    def forward(self, batch, tags, seq_lengths):
        Batch = self.embedding(batch).to('cuda:0')
        emit_score = self.encode(Batch, seq_lengths)
        mask = self.maskCreation(Batch, seq_lengths)
        loss = self.model(emit_score, tags, mask=mask, reduction= 'mean')
        return -loss
    
    def encode(self, batch, seq_lengths):
        #The expected Batch size is in 1st dimension
        padded_sequences = pack_padded_sequence(batch, seq_lengths, batch_first=True)
        Hidden, _ = self.encoder(padded_sequences)
        #The returned Batch size is in 1st dimension
        Hidden, _ = pad_packed_sequence(Hidden, batch_first=True)
        emit_scores = self.estimates(Hidden)
        emit_scores = self.drop_out(emit_scores)
        return emit_scores.to('cuda:0')

    def predict(self, batch, seq_lengths):
        Batch = self.embedding(batch)
        emit_score = self.encode(Batch, seq_lengths)
        mask = self.maskCreation(Batch, seq_lengths)
        predTags = self.model.decode(emit_score, mask=mask)
        return predTags