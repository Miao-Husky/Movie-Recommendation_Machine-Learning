🎬 PyTorch Movie Recommendation System

A personalized movie recommender system built with PyTorch, trained on the MovieLens 100k dataset.
The system predicts user ratings using collaborative filtering with embeddings and provides real-time recommendations via a FastAPI REST API and an interactive Streamlit dashboard.

🚀 Features

Collaborative Filtering with user–item embeddings (Matrix Factorization).

PyTorch model training with MSE loss and Adam optimizer.

Prediction output constrained to [1–5] rating scale for realistic results.

REST API with FastAPI for serving recommendations.

Streamlit dashboard for interactive Top-K movie recommendations.

Modular project structure (data preprocessing, training, inference, deployment).

🛠️ Tech Stack

Backend/Model: Python, PyTorch, scikit-learn

API: FastAPI, Uvicorn

Frontend: Streamlit

Dataset: MovieLens 100k

📂 Project Structure
pytorch-movie-recommender/
│── data/                  # MovieLens dataset (place here after download)
│── model.py               # MatrixFactorization model
│── utils.py               # Dataset class + data loading
│── train.py               # Training script
│── api.py                 # FastAPI service
│── app.py                 # Streamlit front-end
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

⚡ Getting Started
1. Clone the repo
git clone https://github.com/your-username/Movie-Recommendation_Machine-Learning.git
cd Movie-Recommendation_Machine-Learning

2. Set up environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

3. Download dataset

Download MovieLens 100k dataset

Unzip and place ml-100k/ under data/

pytorch-movie-recommender/data/ml-100k/

4. Train the model
python train.py


This will generate a model.pth file.

🌐 Run the API
uvicorn api:app --reload


API docs: http://127.0.0.1:8000/docs

Example request:

GET /recommend?user=1&k=5

🎨 Run the Streamlit App
streamlit run app.py


Then open http://localhost:8501

You can input a user ID and get Top-K movie recommendations with predicted ratings.

📊 Example Output
{
  "user_id": 1,
  "recommendations": [
    {"movie_id": 50, "title": "Star Wars (1977)", "score": 4.8},
    {"movie_id": 181, "title": "Return of the Jedi (1983)", "score": 4.6},
    {"movie_id": 100, "title": "Fargo (1996)", "score": 4.5}
  ]
}

📌 Future Improvements

Add hybrid recommendation (content-based + collaborative filtering).

Use a larger dataset (MovieLens 1M/10M).

Deploy API + Streamlit via Docker or cloud (Heroku, AWS, GCP).
