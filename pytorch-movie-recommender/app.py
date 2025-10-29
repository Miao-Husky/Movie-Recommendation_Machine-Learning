import streamlit as st
import torch
import numpy as np
from model import MatrixFactorization
from utils import load_data

# Configure Streamlit app appearance
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎥", layout="wide")

# Inject custom CSS to improve UI appearance
st.markdown("""
<style>
    body { background-color: #f5f7fa; }
    .movie-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        padding: 12px;
        margin-bottom: 10px;
        text-align: left;
        transition: transform 0.15s;
    }
    .movie-card:hover { transform: scale(1.02); }
    .movie-title { font-weight: 600; color: #1E3A8A; font-size: 16px; }
    .movie-id { color: #6B7280; font-size: 13px; }
    .movie-score { color: #FACC15; font-weight: 600; font-size: 15px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """
    Load the trained matrix factorization model and MovieLens metadata.

    Returns:
        model (MatrixFactorization): trained MF model loaded from checkpoint
        movies (DataFrame): movie metadata (IDs and titles)
        num_users (int): total number of users
        num_items (int): total number of movies
    """
    ratings, movies = load_data()
    num_users = ratings["user_id"].max()
    num_items = ratings["movie_id"].max()

    # Initialize model structure and load pretrained weights
    model = MatrixFactorization(num_users, num_items, embedding_dim=50)
    model.load_state_dict(torch.load("model_l2.pth", map_location=torch.device("cpu")))
    model.eval()
    return model, movies, num_users, num_items


# Load model and movie metadata
model, movies, num_users, num_items = load_model()


def recommend_movies(user_id, top_k=5):
    """
    Generate top-K movie recommendations for a given user.

    Args:
        user_id (int): target user ID
        top_k (int): number of movies to recommend

    Returns:
        DataFrame: contains movie_id, title, and predicted rating score
    """
    # Create tensors for all movie IDs and repeated user ID
    movie_ids = torch.arange(1, num_items + 1)
    user_ids = torch.tensor([user_id] * len(movie_ids))

    # Predict scores for all movies
    with torch.no_grad():
        scores = model(user_ids, movie_ids)
        # Optional: map raw scores to 1–5 range if desired
        # scores = 1 + 4 * torch.sigmoid(scores)
        scores = scores.numpy()

    # Sort movies by predicted score (descending)
    top_idx = np.argsort(scores)[::-1][:top_k]

    # Select top-k movies and attach their predicted scores
    selected = movies.iloc[top_idx][["movie_id", "title"]].copy()
    selected["pred_score"] = scores[top_idx]
    return selected.reset_index(drop=True)


# App title and description
st.title("🎬 Movie Recommendation System")
st.caption("Discover what you'll love next — powered by Matrix Factorization")

# Sidebar layout: user controls on the left, results on the right
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("### Select User")
    user_id = st.number_input("User ID", min_value=1, max_value=int(num_users), value=1, step=1)

    st.markdown("### Recommendation Settings")
    top_k = st.slider("Top-K Movies", 1, 20, 5)

    # Button to trigger recommendation generation
    show_button = st.button("🎥 Show Recommendations")

# When user clicks the button, generate and display top-K results
if show_button:
    recs = recommend_movies(user_id, top_k)
    st.success(f"Top {top_k} Movies Recommended for User {user_id}:")
    st.write("")

    # Render each recommended movie as a styled "card"
    for i, row in recs.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{i+1}. {row['title']}</div>
            <div class="movie-id">Movie ID: {int(row['movie_id'])}</div>
            <div class="movie-score">⭐ {row['pred_score']:.1f}/5</div>
        </div>
        """, unsafe_allow_html=True)

# Before any button is clicked, show usage instruction
else:
    st.info("👆 Select a user and click 'Show Recommendations' to get started!")


# Footer section
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed by Allen Miao @2025</p>",
    unsafe_allow_html=True,
)
