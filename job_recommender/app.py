from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import pickle
import re
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk


nltk.download('stopwords', quiet=True)
stop = set(nltk.corpus.stopwords.words('english'))

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and job data
with open('model.pkl', 'rb') as f:
    tfidf = pickle.load(f)

df = pd.read_csv('jobs.csv')

def clean(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return ' '.join(w for w in text.split() if w not in stop)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        resume = request.files['resume']
        if resume and resume.filename.endswith('.pdf'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
            resume.save(path)

            # Extract text
            doc = fitz.open(path)
            resume_text = ''
            for page in doc:
                resume_text += page.get_text()

            # Preprocess and vectorize
            resume_clean = clean(resume_text)
            resume_vec = tfidf.transform([resume_clean])
            job_vecs = tfidf.transform(df['clean'])

            scores = cosine_similarity(resume_vec, job_vecs).flatten()
            top_n = scores.argsort()[-5:][::-1]

            results = []
            for idx in top_n:
                results.append({
                    'title': df.iloc[idx]['title'],
                    'score': round(scores[idx] * 100, 2),
                    'description': df.iloc[idx]['description'][:300] + '...'
                })

    return render_template('index.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
