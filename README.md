
# 💼 Job Recommendation System

This is a machine learning-based job recommendation system that helps users find relevant job listings based on their skills or interest keywords. The project uses **TF-IDF** vectorization and **cosine similarity** for matching, and it's built using **Flask** for the web interface.

---

## 📌 Features

- Match user input to job descriptions using unsupervised ML (no deep learning).
- Uses real-world job listing data.
- Web interface built with Flask.
- Preprocessing includes TF-IDF and cosine similarity.
- Interactive frontend with results display.
- Lightweight and beginner-friendly.

---

## 🛠️ Tech Stack

- Python
- Flask
- scikit-learn
- pandas
- nltk
- HTML/CSS

---

## 📊 Dataset

- File: `Business_analyst_job_listings_linkedin.csv`
- Columns used: `title`, `description`, and others for visualization.
- You can find similar datasets on Kaggle by searching:
  - “Business Analyst Job Listings”
  - “LinkedIn Job Data”

---

## 🧪 How It Works

1. Load job listing data.
2. Preprocess text using NLTK.
3. Vectorize job descriptions using TF-IDF.
4. Match user input against job descriptions using cosine similarity.
5. Display top recommendations via web UI.

---

## 📸 Screenshots

| Homepage                             | Recommendations Page               |
|-------------------------------------|------------------------------------|
| ![Homepage](\job_recommender\Screenshot 2025-06-20 015648.png) | ![Results](Screenshot%202025-06-20%20015714.png) |

---

## 🧩 Project Structure

```
job_recommendation_system/
│
├── app.py                # Flask backend
├── model.pkl             # Saved TF-IDF vectorizer
├── jobs.csv              # Cleaned and used dataset
├── templates/
│   └── index.html        # Web UI
├── static/
│   └── style.css         # (Optional) Custom styles
├── README.md             # Project info
```

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install flask pandas scikit-learn nltk pymupdf
```

### 2. Run the Web App

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

---

## 📈 Visualizations

- Distribution of job sectors
- Frequency of top TF-IDF tokens

These are saved as `.png` files and shown optionally on the dashboard.


Feel free to reach out via GitHub for questions or improvements.

---

