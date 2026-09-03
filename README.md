# ASMR: Adaptive Semantic Movie Recommendation System

<div align="center">
  <img src="screenshots/home.png" alt="Home Page" width="800"/>
</div>

## 📌 Overview
This repository contains a state-of-the-art **Content-Based Movie Recommendation System** leveraging hybrid intelligence. It was engineered sequentially through extensive A/B testing frameworks, progressively optimizing its baseline heuristics using Differential Evolution constraints. 

Unlike traditional collaborative filtering, this system uses an **Adaptive Semantic** engine that understands the context of movies beyond simple genre matching.

## ✨ Features
- **Semantic Plot Analysis:** Powered by `SentenceTransformers`, vectorizing high-dimensional plot similarities natively.
- **Hybrid Weight Optimization:** Balances exact categorical matches (Genre) against contextual similarities (Semantic), Popularity thresholds, and Franchise heuristics.
- **Dynamic Streamlit Interface:** A modern, dark-neon futuristic UI tailored for responsive cinematic discovery.
- **Self-Healing TMDB Metadata:** Automatically resolves broken or missing local poster images using resilient TMDB API fallbacks.

## 💻 Screenshots

### Recommendation Dashboard
<img src="screenshots/recommendation.png" alt="Dashboard" width="800"/>

### Movie Detail View
<img src="screenshots/detail.png" alt="Detail View" width="800"/>

## 🛠️ Technologies Used
- **Python:** Primary programming language.
- **Pandas / NumPy / SciPy:** Core ML data computation.
- **Scikit-learn:** Meta-label binarization.
- **SentenceTransformers (Huggingface):** Semantic embedding extraction (`all-MiniLM-L6-v2`).
- **Streamlit:** Interactive web application rendering.
- **TMDB API:** Dynamic content metadata retrieval.

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/rajarshi99-ux/movie-recommendations-system.git
cd movie-recommendations-system
```

### 2. Create a virtual environment (Recommended)
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
You will need a TMDB API Key for dynamic poster loading. Create a `.env` file in the root directory:
```env
TMDB_API_KEY=your_tmdb_api_key_here
```

### 5. Dataset Requirement
Ensure that the `dataset/` and `movielens/` data files are present in the directory. You'll specifically need:
- `dataset/movies_cleaned.csv`
- Precomputed embeddings in `model/` (e.g., `movie_embeddings_v7.npy`)

## 🎮 Running the Application

To launch the web interface, execute the following command in your terminal:
```bash
streamlit run app.py
```
This will start a local server, and you can view the application in your browser at [http://localhost:8501](http://localhost:8501).

## 📊 Optimization Experiments (Phases 1-9)
This repository includes the comprehensive research methodology used to optimize the weights of the recommendation engine. 
* Scripts like `run_phase6_optimization.py` and `run_phase9_experiment.py` contain the strictly isolated Differential Evolution (DE) optimization constraint frameworks. 
* Logs and markdown evaluations document the gradient convergence ensuring popularity biases are actively mitigated without destroying semantic depth.
