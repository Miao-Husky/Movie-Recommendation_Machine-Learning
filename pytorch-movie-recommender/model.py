import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users+1, embedding_dim) 
        self.item_emb = nn.Embedding(num_items+1, embedding_dim)
        
    def forward(self, user, item):
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        dot = (user_vec * item_vec).sum(dim=1)
        return dot
