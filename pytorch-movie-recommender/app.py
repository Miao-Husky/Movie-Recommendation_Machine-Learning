import streamlit as st
import torch
from model import MatrixFactorization
from utils import load_data

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
    user_ids = torch.tensor([user_id] * len(movie_ids))
    with torch.no_grad():
        scores = model(user_ids, movie_ids).numpy()
    top_idx = scores.argsort()[::-1][:top_k]
    return movies.iloc[top_idx][["movie_id", "title"]]

# Streamlit UI
st.title("🎬 PyTorch Movie Recommender")
st.write("input ID，get recommendations!")

user_id = st.number_input("ID:", min_value=1, max_value=int(num_users), value=1)
top_k = st.slider("Recommend:", 1, 20, 5)

if st.button("Get Recommendations"):
    recs = recommend_movies(user_id, top_k)
    st.table(recs)
