import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import load_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "model_features.pkl")


def build_features(df):
    """Feature engineering on top of raw columns."""
    df = df.copy()

    # Drop columns not useful for prediction
    drop_cols = ['customer_id', 'name', 'email', 'purchase_date', 'high_value']
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(drop_cols, axis=1)

    # Encode categoricals
    X = pd.get_dummies(X, drop_first=True)
    return X


def train_model():
    df = load_data()

    # Create binary target: top 25% spenders = high value
    threshold = df['total_spend'].quantile(0.75)
    df['high_value'] = (df['total_spend'] >= threshold).astype(int)

    X = build_features(df)
    y = df['high_value']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)\
                 .sort_values(ascending=False)\
                 .head(10)

    # Persist model and feature list
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    return model, acc, cv_scores, cm, feat_imp


def load_model():
    """Load persisted model if available."""
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    return None, None


def predict_single(age, total_spend):
    """Predict a single customer using the persisted model."""
    model, features = load_model()
    if model is None:
        return None

    df = load_data()
    threshold = df['total_spend'].quantile(0.75)

    # Build a minimal row matching training features
    row = pd.DataFrame([{
        'age': age,
        'total_spend': total_spend,
        'annual_income': df['annual_income'].median(),
        'purchase_amount': df['purchase_amount'].median(),
        'quantity': df['quantity'].median(),
        'discount_percentage': df['discount_percentage'].median(),
        'purchase_frequency': df['purchase_frequency'].median(),
        'days_since_last_purchase': df['days_since_last_purchase'].median(),
        'customer_lifetime_value': df['customer_lifetime_value'].median(),
    }])

    # One-hot encode to match training columns
    row_encoded = pd.get_dummies(row)
    row_encoded = row_encoded.reindex(columns=features, fill_value=0)

    pred = model.predict(row_encoded)[0]
    prob = model.predict_proba(row_encoded)[0][1]
    return pred, prob, threshold
