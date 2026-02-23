# 🎬 Why This Movie?

> **Built a content-based movie recommendation engine that prioritizes user trust by providing natural language explanations for every suggestion.**

An explainable recommendation system built using **TF-IDF + cosine similarity**, enhanced with **personalized user taste profiling** and clean natural-language reasoning.

---

## 🚀 Project Overview

Traditional recommender systems suggest items without explanation.  
This project answers a critical question:

> **“Why was this movie recommended?”**

Instead of opaque suggestions, this system:

- Computes content similarity using TF-IDF  
- Builds user taste profiles from ratings  
- Extracts overlapping semantic features  
- Converts them into human-readable explanations  

The result is a **transparent, trust-aware recommendation engine**.

---

## 🧠 Core Features

### 1️⃣ Content-Based Recommendations
- TF-IDF vectorization of:
  - Genres
  - User-generated tags
- Cosine similarity between movie vectors
- Top-N similar movie retrieval

---

### 2️⃣ Personalized Recommendations
- Builds a **user taste vector**
- Uses weighted average of liked movie TF-IDF vectors
- Recommends unseen movies closest to the user’s profile
- Excludes already-rated movies

---

### 3️⃣ Natural Language Explanations

For every recommendation, the system:

- Identifies overlapping semantic features  
- Filters noisy tokens  
- Removes redundant bigrams  
- Generates clean explanations such as:

> Because you selected *Toy Story (1995)*, this recommendation matches on themes like **pixar**, **animation**, and **children**.

Or in personalized mode:

> Recommended because your profile resembles someone who liked *Toy Story (1995)*. It matches on themes like **family-friendly animation**.

---

## 📂 Dataset

This project uses the **MovieLens 100K dataset**, including:

- `movies.csv`
- `ratings.csv`
- `tags.csv`

Each movie is represented by:
- Structured genres
- User-generated tags
- Historical user ratings

---

## 🏗 Project Architecture

Why-This-Movie-Recommender/
│
├── data/
│ └── raw/
│ ├── movies.csv
│ ├── ratings.csv
│ └── tags.csv
│
├── models/
│ └── artifacts/
│ ├── tfidf.joblib
│ ├── tfidf_matrix.joblib
│ ├── similarity.joblib
│ └── indices.joblib
│
├── src/
│ ├── data.py
│ ├── features.py
│ ├── recommender.py
│ ├── explain.py
│ ├── utils.py
│ └── train.py
│
├── app.py
└── README.md


## ⚙️ How It Works

### 🔹 Step 1: Data Processing
- Merge movie genres and aggregated tags  
- Create a `combined_text` field  
- Clean formatting  

### 🔹 Step 2: Feature Engineering
- Apply `TfidfVectorizer`  
- Extract meaningful textual features  
- Filter noisy or overly generic tokens  

### 🔹 Step 3: Similarity Modeling
- Compute cosine similarity matrix  
- Enable movie-to-movie recommendations  

### 🔹 Step 4: User Profiling (Personalized Mode)
- Identify highly rated movies (rating ≥ threshold)  
- Compute weighted average TF-IDF vector  
- Recommend closest unseen movies  

### 🔹 Step 5: Explanation Engine
- Compute overlapping TF-IDF importance  
- Remove duplicates and awkward bigrams  
- Generate natural-language justification  

---

## 🖥️ Running the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt

```
### 2️⃣ Train the Model

```bash
python -m src.train

This generates model artifacts inside:
models/artifacts/
```
### 3️⃣ Launch the App

```bash
streamlit run app.py
```
```BASH
📊 Example Output
🎬 Movie-Based Mode

Selected: Toy Story (1995)
```
### Recommendations:

Bug's Life (1998)

Toy Story 2 (1999)

Antz (1998)

### Explanation:

It matches on themes like pixar, animation, and children.

👤 Personalized Mode

User profile built from highly rated movies.

Example explanation:

Recommended because your profile resembles someone who liked Toy Story (1995). It matches on themes like family-friendly animation.

### 📈 Why This Project Matters

This project demonstrates:

✔ Feature engineering with TF-IDF

✔ Cosine similarity modeling

✔ Sparse matrix operations

✔ Natural language explanation logic

✔ User profiling from behavioral data

✔ Clean modular ML architecture

✔ Interactive deployment with Streamlit

It bridges the gap between:

Black-box recommendations → Transparent, interpretable suggestions.

### 🛠 Tech Stack

Python

Scikit-learn

Pandas

NumPy

Streamlit

MovieLens Dataset

### 🔮 Future Improvements

Hybrid collaborative filtering

Precision@K evaluation

Cloud deployment (Streamlit Cloud)

Transformer-based embeddings (e.g., SBERT)

Tag importance weighting

### 👤 Author

Developed by MD SHAHRIAR KABIR OMI

If you found this interesting, feel free to connect or contribute.

```