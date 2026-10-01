# 🏦 Bank Customer Review Sentiment Analysis using NLP

## 📌 Project Overview

Customer reviews contain valuable information about people's experiences with banks. However, manually reading and analyzing a large number of reviews can be time-consuming.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze bank customer reviews and classify them into **Positive** or **Negative** sentiment.

The project covers the complete NLP workflow, from data cleaning and text preprocessing to model training, evaluation, and deployment using **Streamlit**.

---

## 🎯 Why This Project?

Banks receive a large amount of customer feedback through reviews and other platforms. These reviews can provide useful information about customer satisfaction, problems, and experiences.

The purpose of this project is to explore how NLP and Machine Learning can be used to automatically analyze customer reviews and identify their sentiment.

This project also helps demonstrate a practical **end-to-end NLP and Machine Learning workflow** using a real-world customer review dataset.

---

## 🎯 Main Objective

The main objective of this project is to develop an **NLP-based Machine Learning system that automatically classifies bank customer reviews as Positive or Negative**.

### Project Objectives

* Clean and prepare customer review data.
* Perform Exploratory Data Analysis (EDA).
* Create sentiment labels from star ratings.
* Preprocess text data.
* Convert text into numerical features using **TF-IDF** and **CountVectorizer**.
* Train different Machine Learning models.
* Compare model performance.
* Evaluate models using appropriate classification metrics.
* Analyze important words contributing to predictions.
* Build a system that can predict the sentiment of new customer reviews.
* Deploy the model using **Streamlit**.

---

## 📊 Dataset

The dataset contains customer reviews of different banks.

### Main Features

| Column     | Description              |
| ---------- | ------------------------ |
| `author`   | Author of the review     |
| `date`     | Date of the review       |
| `location` | Location of the reviewer |
| `bank`     | Name of the bank         |
| `star`     | Customer rating          |
| `text`     | Customer review          |
| `like`     | Number of likes          |

The most important column for NLP is:

```text
text
```

because it contains the actual customer review.

---

## 🏷️ Sentiment Creation

The original dataset contains star ratings rather than a separate sentiment column.

Therefore, sentiment is created from the `star` rating.

```python
df["sentiment"] = np.where(df["star"] >= 4, 1, 0)
```

The classification is:

| Star Rating | Sentiment |
| ----------- | --------- |
| 1, 2, 3     | Negative  |
| 4, 5        | Positive  |

Where:

```text
0 = Negative
1 = Positive
```

### ⚠️ Important Note

The sentiment labels are derived from the star ratings. Therefore, the model is learning to predict **rating-derived sentiment**, rather than independently human-labeled sentiment.

---

## 🔄 Project Workflow

```text
Bank Customer Reviews
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Text Preprocessing
        ↓
Create Sentiment Target
        ↓
Train/Test Split
        ↓
TF-IDF / CountVectorizer
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Important Word Analysis
        ↓
Best Model Selection
        ↓
New Review Prediction
        ↓
Streamlit Deployment
```

---
# 🔍 Methodology
## 📊 Visualizations
Here are some visualizations from the project:

![alt text](https://github.com/ranjansinghds/Bank-Customer-Review-Sentiment-Analysis-Project/blob/main/Bank%20Customer%20Department%20Project%20Png/Countplot%20of%20Sentiment%20Distribution.png)
![alt text](https://github.com/ranjansinghds/Bank-Customer-Review-Sentiment-Analysis-Project/blob/main/Bank%20Customer%20Department%20Project%20Png/Distribution%20of%20star%20Rating.png)
![alt text](https://github.com/ranjansinghds/Bank-Customer-Review-Sentiment-Analysis-Project/blob/main/Bank%20Customer%20Department%20Project%20Png/Distribution%20of%20Review%20Length.png)
![alt text](https://github.com/ranjansinghds/Bank-Customer-Review-Sentiment-Analysis-Project/blob/main/Bank%20Customer%20Department%20Project%20Png/Top%2015%20Banks%20by%20Number%20of%20Reviews.png)
## 🧹 Data Cleaning

The following steps are performed during data cleaning:

* Remove duplicate records.
* Remove missing reviews.
* Remove empty reviews.
* Convert text into string format.
* Remove unnecessary spaces.

Example:

```python
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
```

---

## 📝 Text Preprocessing

Customer reviews are cleaned before converting them into numerical features.

The preprocessing includes:

* Converting text to lowercase.
* Removing URLs.
* Removing punctuation.
* Removing numbers.
* Removing English stopwords.
* Removing unnecessary spaces.

Example:

```text
Original:
"The banking app is AMAZING! Visit https://example.com"

Cleaned:
"banking app amazing"
```

---

## 🔢 Feature Extraction

Machine Learning models cannot directly understand text.

Therefore, the text is converted into numerical features using:

### TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** measures how important a word is within a collection of documents.

Example:

```python
TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
```

### CountVectorizer

CountVectorizer converts text into numerical features based on word frequency.

Both methods are compared to understand which representation works better for the dataset.

---

## 🤖 Machine Learning Models

The project experiments with multiple classification models:

### 1. Multinomial Naive Bayes

A simple and commonly used algorithm for text classification.

### 2. Logistic Regression

A linear classification algorithm that works well with TF-IDF features.

### 3. Linear Support Vector Machine

Linear SVM is particularly useful for high-dimensional text classification problems.

The models are compared using their evaluation metrics, and the best-performing model is used for prediction and deployment.

---

## 📏 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Why use multiple metrics?

Accuracy alone may not provide a complete picture, especially when the classes are imbalanced.

Therefore, precision, recall, and F1-score are also considered when comparing the models.

---

## 🔍 Important Word Analysis

For a linear model such as **Logistic Regression or Linear SVM**, model coefficients can be analyzed to identify words that contribute toward positive or negative predictions.

For example:

```text
Positive-related words:
excellent
amazing
helpful
easy

Negative-related words:
worst
terrible
bad
poor
```

This provides additional insight into how the model is making predictions.

---

## 💬 New Review Prediction

After training the model, a new customer review can be entered and classified.

Example:

```text
"The banking application is very easy to use."
```

Possible output:

```text
Prediction: Positive Review
```

Another example:

```text
"The app keeps crashing and customer service is terrible."
```

Possible output:

```text
Prediction: Negative Review
```

---

## 🌐 Streamlit Deployment

The trained NLP model is deployed using **Streamlit**.

The application allows a user to enter a customer review and receive a sentiment prediction.

### Example Prediction:

![sentiment prediction](https://github.com/ranjansinghds/Bank-Customer-Review-Sentiment-Analysis-Project/blob/main/Bank%20Customer%20Department%20Project%20Png/Bank%20Customer%20Review%20Sentiment%20Analysis.png)

### Run the application

First, install the required libraries:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* TF-IDF
* CountVectorizer
* Machine Learning
* Streamlit
* Jupyter Notebook
* Git & GitHub

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/ranjansinghds/BANK_CUSTOMER_REVIEW_DEPARTMENT_PROJECT.git
cd Bank-Customer-Review-Sentiment-Analysis-Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🌍 Real-World Applications

This type of sentiment analysis system can be used in:

* 🏦 Banks
* 📱 Mobile banking applications
* ☎️ Customer service departments
* 🛍️ E-commerce platforms
* 🏨 Hotels
* 🍽️ Restaurants
* 📊 Customer feedback analysis
* 💬 Social media monitoring

Businesses can use customer sentiment analysis to understand customer satisfaction, identify common complaints, and improve their products and services.

---

## ⚠️ Limitations

This project has some limitations:

1. Sentiment labels are created from star ratings rather than independent human annotations.
2. The model primarily performs binary sentiment classification.
3. Sarcasm and complex language can be difficult for traditional NLP models to understand.
4. The model may not perform equally well on reviews from completely different domains.
5. The current system focuses mainly on overall sentiment rather than identifying specific aspects of a bank's service.

---

## 🚀 Future Improvements

Possible improvements include:

* Add a **Neutral** sentiment class.
* Perform **Aspect-Based Sentiment Analysis**.
* Use Word2Vec or other word embeddings.
* Experiment with deep learning models such as LSTM.
* Experiment with Transformer models such as BERT.
* Improve sentiment labeling using manually annotated data.
* Analyze sentiment by bank.
* Analyze sentiment over time.
* Identify common topics in negative reviews.
* Improve the Streamlit interface.
* Deploy the application online.

---

## 📌 Conclusion

This project demonstrates how **Natural Language Processing and Machine Learning** can be used to analyze bank customer reviews and classify their sentiment.

Through this project, I practiced the complete NLP workflow, including **data cleaning, exploratory data analysis, text preprocessing, feature extraction, model training, model evaluation, prediction, and deployment**.

The project provides practical experience in transforming unstructured customer feedback into useful information that can support customer experience analysis and business decision-making.



