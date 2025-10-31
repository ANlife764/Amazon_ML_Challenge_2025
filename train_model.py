import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import joblib

# --- Load Data ---
train_df = pd.read_csv("train.csv")

train_df['catalog_content'] = train_df['catalog_content'].fillna("").astype(str)
y = np.log1p(train_df['price'])

# --- TF-IDF Features ---
tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1,2))
X = tfidf.fit_transform(train_df['catalog_content'])

# --- Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# --- Train LightGBM ---
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "verbosity": -1
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=2000,
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ]
)

# --- Save Model and TF-IDF ---
model.save_model("lgb_price_model.txt")
joblib.dump(tfidf, "tfidf_union.pkl")

print("✅ Model and TF-IDF saved successfully!")
