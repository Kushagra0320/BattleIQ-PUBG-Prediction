
# 🧠 BattleIQ: Win Rate Prediction Model for PUBG Players

*A Machine Learning Project for Competitive Ranking Analysis*

---

## 📌 Overview

**BattleIQ** is a machine learning-based predictive analytics project built to estimate a PUBG player's *win placement percentile (`winPlacePerc`)* using in-game stats like kills, distance traveled, healing items used, and more. This project focuses on creating high-quality features, applying advanced anomaly filtering, and training a fine-tuned CatBoost regression model for high accuracy.

> ✅ This project is  built from scratch — including data cleaning, feature engineering, model training, and evaluation.
---

## 📂 Dataset

- Source: [PUBG Games Dataset on Kaggle](https://www.kaggle.com/datasets/ashishjangra27/pubg-games-dataset)  
- Credit: [Ashish Jangra](https://www.kaggle.com/ashishjangra27)  
- Size: ~4.4 million player records across various game types (solo, duo, squad)

---

## 🚀 Key Features

- **Domain-Driven Feature Engineering**  
  Created new composite features to represent movement (`traveldistance`), combat (`killsNorm`, `assist`), and survival tactics (`healsnboosts`).

- **Anomaly Detection & Cleaning**  
  Removed unrealistic records like:
  - `kills > 20`
  - `longestKill >= 500`
  - `headshot_rate = 1` with high kill count
  - Players with kills but zero movement (`killswithoutMoving`)

- **Feature Normalization**  
  Adjusted key metrics based on `playersJoined` to make the model robust across matches of different sizes.

- **Encoding & Scaling**  
  - One-hot encoded categorical features like `matchType`
  - Standardized all numerical features using `StandardScaler`

- **Modeling with CatBoost**  
  - Tuned hyperparameters (`depth`, `iterations`, `learning_rate`) via Grid Search  
  - Achieved **R² score of 0.93** and **RMSE of 0.06** on test data

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R²     | 0.93  |
| RMSE   | 0.06  |

---

## 🧪 Tech Stack

- Python (Pandas, NumPy, Seaborn, Matplotlib)
- Scikit-learn
- CatBoost
- Jupyter Notebook


---

## 🧠 What I Learned

- Designing and validating ML features from domain knowledge
- Applying outlier filtering using game logic (not just statistical rules)
- Handling categorical encoding and normalization for regression
- Tuning advanced models using Grid Search
- Building interpretable machine learning pipelines

---

