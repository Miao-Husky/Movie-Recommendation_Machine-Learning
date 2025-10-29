import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, MovieLensDataset
from tqdm import tqdm
from model import MatrixFactorization
from scipy.signal import savgol_filter


# Load MovieLens data
# ratings: contains user_id, movie_id, rating
# movies: contains metadata such as movie titles
ratings, movies = load_data()

# Find the number of unique users and items
num_users = ratings["user_id"].max()
num_items = ratings["movie_id"].max()

# Split ratings into training and testing sets (80/20)
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Build DataLoaders for mini-batch training
# Each batch will return (user, item, rating) tensors
train_loader = DataLoader(MovieLensDataset(train), batch_size=512, shuffle=True)
test_loader = DataLoader(MovieLensDataset(test), batch_size=512, shuffle=False)


def train_model(use_l2=True):
    """
    Train a matrix factorization model with or without L2 regularization.

    Args:
        use_l2 (bool): whether to apply L2 regularization (via weight decay)

    Returns:
        model: trained matrix factorization model
        train_rmse_iter: list of training RMSE values
        val_rmse_iter: list of validation RMSE values
    """
    model = MatrixFactorization(num_users, num_items, embedding_dim=50)
    criterion = nn.MSELoss()  # Mean squared error between predicted and true ratings
    optimizer = optim.Adam(model.parameters(), lr=0.005,
                           weight_decay=0.01 if use_l2 else 0.0)

    num_epochs = 5
    train_rmse_iter, val_rmse_iter = [], []
    iteration = 0

    print(f"Training {'with' if use_l2 else 'without'} L2 regularization...\n")

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (user, item, rating) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear old gradients
            preds = model(user, item)  # Forward pass
            loss = criterion(preds, rating)  # Compute MSE loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            iteration += 1

            # Compute RMSE for every 10 iterations
            if iteration % 10 == 0:
                train_rmse = np.sqrt(loss.item())
                train_rmse_iter.append(train_rmse)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for u, i, r in test_loader:
                        val_loss += criterion(model(u, i), r).item()
                val_rmse = np.sqrt(val_loss / len(test_loader))
                val_rmse_iter.append(val_rmse)
                model.train()

                print(f"Iter {iteration}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")

    return model, train_rmse_iter, val_rmse_iter


# Train two models for comparison
model_l2, train_rmse_l2, val_rmse_l2 = train_model(use_l2=True)
torch.save(model_l2.state_dict(), "model_l2.pth")

model_no_l2, train_rmse_no_l2, val_rmse_no_l2 = train_model(use_l2=False)
torch.save(model_no_l2.state_dict(), "model_no_l2.pth")


def smooth_curve(values, window=7, poly=2):
    """
    Smooth the RMSE curve using Savitzky-Golay filter for clearer visualization.
    """
    if len(values) < window:
        return values
    return savgol_filter(values, window_length=window, polyorder=poly)


# Plot training and validation RMSE (with vs without L2)
plt.style.use("dark_background")
plt.rcParams.update({
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#444444"
})

fig, axes = plt.subplots(2, 1, figsize=(9, 10), facecolor="black", sharex=True)
fig.suptitle("Ablation Study: With vs Without L2 Regularization",
             fontsize=16, fontweight="bold", color="white", y=0.96)

axes[0].plot(np.arange(len(train_rmse_l2)) * 10, smooth_curve(train_rmse_l2, 9),
             label="Train RMSE", color="#00BFFF", linewidth=3.0)
axes[0].plot(np.arange(len(val_rmse_l2)) * 10, smooth_curve(val_rmse_l2, 9),
             label="Validation RMSE", color="#FFB347", linewidth=3.0, linestyle="--")
axes[0].set_title("With L2 Regularization (λ = 0.01)", fontsize=13, color="white", pad=8)
axes[0].set_ylabel("RMSE", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

axes[1].plot(np.arange(len(train_rmse_no_l2)) * 10, smooth_curve(train_rmse_no_l2, 9),
             label="Train RMSE", color="#00BFFF", linewidth=3.0)
axes[1].plot(np.arange(len(val_rmse_no_l2)) * 10, smooth_curve(val_rmse_no_l2, 9),
             label="Validation RMSE", color="#FFB347", linewidth=3.0, linestyle="--")
axes[1].set_title("Without L2 Regularization", fontsize=13, color="white", pad=8)
axes[1].set_xlabel("Iteration", fontsize=12, fontweight="bold")
axes[1].set_ylabel("RMSE", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


def precision_recall_at_k(model, ratings, movies, k=5, threshold=4.0):
    """
    Compute average Precision@K and Recall@K across users.

    Precision@K: among top-K recommended movies, what fraction were truly liked.
    Recall@K: among all movies a user liked, how many appeared in top-K.

    Args:
        model: trained matrix factorization model
        ratings: test dataset (DataFrame)
        movies: movie metadata (DataFrame)
        k: number of top items to recommend
        threshold: rating threshold to consider an item as "liked"
    """
    model.eval()
    precisions, recalls = [], []
    user_groups = ratings.groupby("user_id")

    for user_id, user_data in tqdm(user_groups, desc=f"Evaluating P@{k} / R@{k}"):
        # Ground truth: items this user rated above threshold
        true_items = user_data[user_data["rating"] >= threshold]["movie_id"].values
        if len(true_items) == 0:
            continue

        # Predict scores for all movies for this user
        movie_ids = torch.arange(1, movies.shape[0] + 1)
        user_tensor = torch.tensor([user_id] * len(movie_ids))

        with torch.no_grad():
            preds = model(user_tensor, movie_ids)
            preds = preds.detach().cpu().numpy()

        # Select top-K predicted scores
        top_k_idx = np.argsort(preds)[::-1][:k]
        top_k_items = movie_ids[top_k_idx].numpy()

        # Calculate precision and recall for this user
        hits = np.intersect1d(top_k_items, true_items)
        precisions.append(len(hits) / k)
        recalls.append(len(hits) / len(true_items))

    mean_p = np.mean(precisions)
    mean_r = np.mean(recalls)
    print(f"Precision@{k}: {mean_p:.4f} | Recall@{k}: {mean_r:.4f}")
    return mean_p, mean_r


print("\nFinal Evaluation:")
print("With L2 Regularization:")
precision_recall_at_k(model_l2, test, movies, k=1)

print("\nWithout L2 Regularization:")
precision_recall_at_k(model_no_l2, test, movies, k=1)
