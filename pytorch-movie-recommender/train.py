import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import load_data, MovieLensDataset
from model import MatrixFactorization

# load data
ratings, movies = load_data()   
num_users = ratings["user_id"].max()
num_items = ratings["movie_id"].max()

# split train/test
train, test = train_test_split(ratings, test_size=0.2, random_state=42)
train_loader = DataLoader(MovieLensDataset(train), batch_size=512, shuffle=True)

# define model
model = MatrixFactorization(num_users, num_items, embedding_dim=50)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# train
for epoch in range(5):
    model.train()
    total_loss = 0
    for user, item, rating in train_loader:
        optimizer.zero_grad()
        preds = model(user, item)
        loss = criterion(preds, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss {total_loss/len(train_loader):.4f}")

# save model
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved as model.pth")
