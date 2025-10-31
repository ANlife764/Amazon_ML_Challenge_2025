import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

# --- Load Model and TF-IDF ---
model = lgb.Booster(model_file="lgb_price_model.txt")
tfidf = joblib.load("tfidf_union.pkl")

# --- Load Test Data ---
test_df = pd.read_csv("test.csv")
test_df['catalog_content'] = test_df['catalog_content'].fillna("").astype(str)

# --- Generate TF-IDF Features ---
X_test = tfidf.transform(test_df['catalog_content'])

# --- Predict ---
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)
preds = np.maximum(preds, 0)  # Ensure positive

# --- Output CSV ---
output = pd.DataFrame({
    "sample_id": test_df["sample_id"],
    "price": preds
})

output.to_csv("test_out.csv", index=False)
print("✅ test_out.csv created successfully!")
