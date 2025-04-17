# 💼 Bank Subscription Prediction – ML Deployment App

This project predicts whether a bank customer is likely to subscribe to a term deposit (long-term savings plan), based on their personal and financial details. It demonstrates a full machine learning workflow including data preprocessing, model training, handling imbalanced data, and deploying the model via a Flask API with a modern HTML frontend.

---

## 🚀 Features

- 📊 Trained Random Forest model on real-world bank marketing dataset
- ⚖️ Balanced dataset using SMOTE to improve minority class prediction
- 🔍 Real-time predictions via REST API
- 🌐 Frontend web form (HTML + JS) to collect user inputs and show predictions
- 📦 Model serialized with `pickle`
- 💡 Deployed locally via Flask

---

## 🧠 What It Predicts

> Will a customer say **"yes"** to a term deposit offer?

Based on:
- Age
- Account Balance
- Job Type
- Marital Status
- Education Level
- Credit Default
- Housing Loan
- Personal Loan

---

## 📁 Project Structure

```
.
├── app.py                 # Flask API for prediction
├── model.pkl              # Trained Random Forest model
├── scaler.pkl             # Scaler used on numeric features
├── encoder.pkl            # OneHotEncoder for categorical features
├── BankModel_Challenge_TeamActivity.ipynb  # Notebook with data cleaning and model training
├── /templates
│   └── index.html         # HTML frontend interface
└── screenshot.png         # Web app screenshot (for submission)
```

---

## 📦 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/bank-subscription-predictor.git
cd bank-subscription-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> Make sure `Flask`, `scikit-learn`, `imblearn`, `numpy`, and `pandas` are included.

### 3. Run Flask App
```bash
python app.py
```

Go to `http://127.0.0.1:5000` in your browser to test the form.

---

## 📊 Model Training Highlights

- Dataset: [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- Model: `RandomForestClassifier` with `class_weight='balanced'`
- Data Balancing: `SMOTE` from `imblearn`
- Accuracy: ~84% (with increased **recall** for minority class)

---

## 🧠 Future Improvements

- Use `XGBoost` for higher precision/recall
- Deploy online (Replit, Render, Heroku, or EC2)
- Add input validation to the frontend
- Save predictions to a database or CSV for tracking

---

## 🧑‍💻 Author

**Hrishabh Mahaju**  
🧠 Master’s in Data Science @ University of New Haven  
🌐 [hrishabh.com.np](https://hrishabh.com.np/)  
📫 Connect on [LinkedIn](https://www.linkedin.com/in/hrishabh360)

---