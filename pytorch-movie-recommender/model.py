import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """
    Basic matrix factorization model for collaborative filtering.

    Each user and item is represented by a learned embedding vector.
    The predicted rating is computed as:
        ŷ = global_bias + user_bias + item_bias + dot(user_emb, item_emb)

    Args:
        num_users (int): total number of unique users
        num_items (int): total number of unique items (movies)
        embedding_dim (int): dimensionality of latent feature vectors
    """

    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()

        # User embedding matrix: each user_id maps to a D-dimensional vector
        self.user_emb = nn.Embedding(num_users + 1, embedding_dim)

        # Item embedding matrix: each movie_id maps to a D-dimensional vector
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim)

        # User bias term: captures how much a user tends to rate higher/lower than average
        self.user_bias = nn.Embedding(num_users + 1, 1)

        # Item bias term: captures how much an item tends to receive higher/lower ratings
        self.item_bias = nn.Embedding(num_items + 1, 1)

        # Global bias: represents the average rating across all users and items
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, item):
        """
        Compute the predicted rating for a given batch of (user, item) pairs.

        Args:
            user (Tensor): user IDs (batch_size,)
            item (Tensor): item IDs (batch_size,)

        Returns:
            Tensor: predicted ratings (batch_size,)
        """
        # Look up user and item embedding vectors
        u = self.user_emb(user)   # shape: [batch_size, embedding_dim]
        i = self.item_emb(item)   # shape: [batch_size, embedding_dim]

        # Compute dot product between user and item embeddings
        dot = (u * i).sum(1)  # element-wise product + sum over embedding dimension

        # Look up user and item biases
        bu = self.user_bias(user).squeeze()  # shape: [batch_size]
        bi = self.item_bias(item).squeeze()  # shape: [batch_size]

        # Final prediction = global bias + user bias + item bias + dot product
        return self.global_bias + bu + bi + dot
