## Twitter Airline Sentiment Analysis (TF‑IDF + Tree‑Based Models)

A reproducible NLP project that classifies airline-related tweets into positive or negative sentiment. It preprocesses raw text, transforms it using TF‑IDF, and trains multiple tree-based classifiers (Decision Tree, Random Forest, XGBoost), evaluating them with standard classification metrics and plots.

### Key Features
- **End‑to‑end NLP pipeline**: cleaning, tokenization, stopword removal, TF‑IDF feature engineering.
- **Multiple models**: Decision Tree, Random Forest, and XGBoost variants.
- **Clear evaluation**: accuracy, F1, precision, recall, confusion matrix, and a precision‑recall curve.
- **Easy to run**: single Jupyter notebook with self‑contained steps.
- **Extensible**: swap vectorizers, tweak hyperparameters, or add more models.

### Dataset
- **Input file**: `Tweets.csv`
- **Target column**: `airline_sentiment` (converted to binary: 1 = positive, 0 = negative; neutral is excluded)
- **Text column**: `text`
- **Dropped columns**: `airline_sentiment_gold`, `negativereason_gold`, `tweet_coord`

### Tech Stack
- Python, Jupyter
- pandas, numpy, nltk, scikit‑learn
- seaborn, matplotlib, missingno
- xgboost (optional GPU with `tree_method='gpu_hist'`)

### Project Structure
- `Twitter (RF,DT).ipynb`: main notebook with data prep, modeling, and evaluation
- `Tweets.csv`: dataset file used by the notebook

### Installation

1) Create and activate a virtual environment (Windows PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Download required NLTK corpora (done in the notebook, or run once in Python):
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4) Optional GPU for XGBoost:
- Requires a compatible NVIDIA GPU and CUDA toolkit.
- If not available, remove or change `tree_method='gpu_hist'` in the XGBoost parameters in the notebook.

### How It Works
- Cleans tweets (keep letters, lowercase, tokenize), removes punctuation and English stopwords.
- Transforms text into TF‑IDF features (`TfidfVectorizer`).
- Splits into train/test.
- Trains Decision Tree, Random Forest, and XGBoost classifiers.
- Evaluates and prints metrics; plots a precision‑recall curve.

### Usage
1) Place `Tweets.csv` in the project root (same folder as the notebook).
2) Launch Jupyter and open the notebook:
```bash
jupyter notebook
```
3) Open `Twitter (RF,DT).ipynb` and Run All cells.

#### Inference Example (after training within the notebook)
```python
sample = "I loved the service and the flight was on time!"
cleaned = text_process(clean_the_tweet(sample))
X_sample = vectorizer.transform([cleaned])
pred = model.predict(X_sample)[0]
label = "positive" if pred == 1 else "negative"
print(label)
```

### Results (example on the provided split)
- Random Forest (sklearn): Accuracy 88.15%, F1 64.89%, Precision 76.89%, Recall 56.13%
- Decision Tree (sklearn): Accuracy 83.19%, F1 58.79%, Precision 56.35%, Recall 61.46%
- XGBoost (gpu_hist params): Accuracy 84.86%, F1 45.98%, Precision 75.61%, Recall 33.04%
- XGBoost (decision-tree style params): Accuracy 86.76%, F1 55.58%, Precision 80.47%, Recall 42.45%

Notes:
- Metrics may vary depending on random seeds, train/test split, and hyperparameters.
- Class imbalance impacts F1/recall; tune thresholds and class weights as needed.

### Extending the Project
- Try `CountVectorizer` or character n‑grams.
- Hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV`.
- Save models with `joblib` and build a simple CLI or Flask/FastAPI service.

### Contribution Guidelines
- **Issues & Discussions**: Use issues for bugs/enhancements; propose major changes via a discussion first.
- **Fork & Branch**: Fork the repo and create a feature branch (`feat/your-feature-name`).
- **Commits**: Write clear, scoped commit messages.
- **PRs**: Open a pull request describing the problem, your approach, and testing.
- **Code Style**: Prefer readable, well‑structured Python. Keep notebook outputs minimal; re‑run to update result cells where relevant.
- **Reproducibility**: Document seeds, parameters, and environment differences if metrics change.
