# Hate Speech Detection

A machine learning model that classifies tweets into **Hate Speech**, **Offensive Language**, or **Neither**, using classical NLP techniques and a Decision Tree classifier.

🔗 **Live demo:** [hate-speech-detection-liart.vercel.app](https://hate-speech-detection-liart.vercel.app)

---

## Overview

This project builds a text classifier that flags harmful content in social media posts. It takes a raw tweet as input, cleans and vectorizes the text, and predicts one of three labels:

| Label | Meaning |
|---|---|
| `0` | Hate Speech |
| `1` | Offensive Language |
| `2` | Neither |

## Dataset

- **Source**: `labeled dataset.csv` — 24,783 labeled tweets
- **Columns used**: `tweet` (raw text), `class` (label 0/1/2)
- Labels were remapped to readable strings (`Hate Speech`, `Offensive Language`, `No hate or offensive language`) for clarity during EDA.

## Approach

### 1. Text Cleaning
Each tweet is preprocessed before modeling:
- Lowercased
- URLs, HTML tags, and text in brackets removed
- Punctuation and digits stripped
- Stopwords removed (NLTK's English stopword list)
- Words reduced to their root form via **Porter Stemming**

### 2. Feature Extraction
- Cleaned text is converted into numeric features using **`CountVectorizer`** (bag-of-words), producing a sparse matrix of shape `(24783, 25683)`.

### 3. Model Training
- **Algorithm**: `DecisionTreeClassifier` (scikit-learn)
- **Split**: 67% train / 33% test (`random_state=42`)

### 4. Evaluation
- **Accuracy**: ~87.5% on the held-out test set
- **Confusion matrix** visualized with a Seaborn heatmap to inspect class-wise performance

### 5. Inference on New Text
The trained pipeline (clean → vectorize → predict) is tested on a custom example:

```python
sample = "Lets unite and kill all the people who are protesting against the government"
# → cleaned to: "let unit kill peopl protest govern"
# → model prediction: "Hate Speech"
```

## Tech Stack

- **Python**, **pandas**, **NumPy**
- **NLTK** — stopword removal, Porter stemming
- **scikit-learn** — `CountVectorizer`, `DecisionTreeClassifier`, train/test split, metrics
- **Matplotlib / Seaborn** — confusion matrix visualization

## Project Structure

```
Hate-Speech-Detection/
├── Hate speech detection.ipynb   # Full pipeline: EDA → cleaning → training → evaluation
├── labeled dataset.csv           # Training data (24,783 labeled tweets)
└── README.md
```

## Running It Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Saksham-coder293/Hate-Speech-Detection.git
   cd Hate-Speech-Detection
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy nltk scikit-learn matplotlib seaborn
   ```

3. Download NLTK stopwords (run once):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. Open and run the notebook:
   ```bash
   jupyter notebook "Hate speech detection.ipynb"
   ```

## Results

| Metric | Score |
|---|---|
| Accuracy | ~87.5% |

The confusion matrix shows the model is strongest at identifying "Neither" (the majority class) and weakest at distinguishing Hate Speech from Offensive Language — expected given the class imbalance and their linguistic overlap.

## Limitations & Future Work

- **Class imbalance**: "Offensive Language" and "Neither" dominate the dataset, which likely biases the Decision Tree toward these classes.
- **Bag-of-words loses context**: `CountVectorizer` ignores word order and semantics — a Transformer-based model (e.g., BERT/DistilBERT) or even TF-IDF + Logistic Regression/SVM would likely perform better.
- **Single model, no cross-validation**: results are from one train/test split; k-fold CV would give a more reliable estimate.
- **No hyperparameter tuning**: Decision Trees are prone to overfitting — pruning or switching to an ensemble (Random Forest, Gradient Boosting) could improve generalization.
- **Deployment**: the model isn't yet wrapped in an API for the live demo to consume — an obvious next step.

## Author

**Saksham Yadav**
