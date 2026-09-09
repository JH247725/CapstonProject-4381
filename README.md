# Analysis of Line-of-Duty Incident Risk Factors and Narrative Text Mining

## Project Overview
This repository contains a two-semester capstone research project examining historical line-of-duty fatality records. The goal is to identify operational and situational risk factors using classical machine learning and extract key context from narrative summaries using Natural Language Processing (NLP).

By combining categorical incident metrics (e.g., officer age, rank, weapon type, cause of death, length of tour) with unstructured narrative data, this project aims to uncover hidden patterns, evaluate predictive features, and segment high-risk operational profiles.

---

## Research Questions

* **Semester 1:** What operational and situational factors (such as weapon, tour length, time/day, rank, and cause) serve as the strongest predictors of line-of-duty fatality circumstances?
* **Semester 2:** Can Natural Language Processing (NLP) and feature extraction on narrative summary text uncover latent operational risk clusters that traditional categorical metrics miss?

---

## Dataset Description
The dataset consists of historical incident logs containing structured tabular attributes alongside unstructured narrative summaries:

| Feature | Description | Type |
| :--- | :--- | :--- |
| `Rank` | Title/position held by the officer | Categorical |
| `Name` | Officer full name | Text |
| `Age` | Age of the officer at the time of incident | Numeric |
| `End_Of_Watch` | Date of the incident | Datetime |
| `Day_Of_Week` | Day of the week when incident occurred | Categorical |
| `Cause` | Cause of death (e.g., Gunfire, Stabbed, Duty-related) | Categorical |
| `Department` | Department or agency name | Categorical |
| `State` | Location jurisdiction | Categorical |
| `Tour` | Length of service/experience | Categorical/Numeric |
| `Badge` | Officer badge identifier | Categorical/Text |
| `Weapon` | Weapon involved in the incident | Categorical |
| `Offender` | Legal/prosecutorial outcome or status of offender | Categorical |
| `Summary` | Detailed narrative describing the incident context | Text (Unstructured) |

---

## Project Structure & Roadmap

### Semester 1: Exploratory Data Analysis & Predictive Modeling
* **Data Preprocessing & Cleaning:** Handle missing values across columns (`Age`, `Tour`, `Badge`), standardize date formats, and parse experience strings into numeric variables.
* **Exploratory Data Analysis (EDA):** Analyze distributions of fatalities by cause, weapon type, region, time of day, and officer demographic profiles.
* **Feature Engineering:** One-hot encode categorical features, scale numerical attributes, and derive temporal features.
* **Baseline Modeling:** Train and evaluate classification algorithms (e.g., Logistic Regression, Decision Trees, Random Forests, XGBoost) to classify incident causes and fatality circumstances.
* **Feature Importance Evaluation:** Calculate Gini importance and SHAP values to isolate primary predictive variables.

### Semester 2: NLP, Clustering & Risk Profiling
* **Text Processing Pipeline:** Clean, tokenize, lemmatize, and remove stop words from the narrative `Summary` field.
* **Text Vectorization:** Apply TF-IDF vectorization and domain-specific word embeddings (e.g., Word2Vec, BERT) to convert narratives into dense vector representations.
* **Unsupervised Clustering:** Implement K-Means, Hierarchical Clustering, or LDA Topic Modeling on text embeddings to discover latent incident categories.
* **Model Integration:** Fuse tabular feature spaces with NLP-derived topics to build hybrid risk prediction models.
* **Insights & Policy Recommendations:** Synthesize statistical findings into actionable operational recommendations.

---

## Tech Stack
* **Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-Learn, XGBoost, SHAP
* **NLP & Text Mining:** NLTK, spaCy, Gensim, Transformers
* **Environment:** Jupyter Notebooks / Google Colab

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/line-of-duty-risk-analysis.git](https://github.com/your-username/line-of-duty-risk-analysis.git)
   cd line-of-duty-risk-analysis
