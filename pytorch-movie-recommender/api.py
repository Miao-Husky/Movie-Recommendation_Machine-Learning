import torch
from fastapi import FastAPI
from model import MatrixFactorization
from utils import load_data

# initialize API
app = FastAPI(title="🎬 PyTorch Movie Recommender", version="1.0")

# load data
ratings, movies = load_data()
num_users = ratings["user_id"].max()
num_items = ratings["movie_id"].max()

# load model
model = MatrixFactorization(num_users, num_items, embedding_dim=50)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# recommendation function
def recommend_movies(user_id, top_k=5):
    movie_ids = torch.arange(1, num_items+1)
    user_ids = torch.tensor([user_id]*len(movie_ids))
    with torch.no_grad():
        scores = model(user_ids, movie_ids).numpy()
    top_idx = scores.argsort()[::-1][:top_k]
    return movies.iloc[top_idx][["movie_id", "title"]].to_dict(orient="records")

@app.get("/")
def root():
    return {"message": "Welcome to PyTorch Movie Recommender API! Try /recommend?user=1"}

@app.get("/recommend")
def recommend(user: int, k: int = 5):
    recs = recommend_movies(user, k)
    return {"user_id": user, "recommendations": recs}
