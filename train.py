import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data():
    objects = pd.read_csv("data/objects.csv", low_memory=False)
    acquisitions = pd.read_csv("data/acquisitions.csv", low_memory=False)
    funding_rounds = pd.read_csv("data/funding_rounds.csv", low_memory=False)
    ipos = pd.read_csv("data/ipos.csv")
    investments = pd.read_csv("data/investments.csv")
    milestones = pd.read_csv("data/milestones.csv")

    startups = objects[objects["entity_type"] == "Company"].copy()
    startups = startups[startups["status"].notna()]
    startups = startups[startups["country_code"] != "CSS"]
    startups = startups[startups["country_code"] != "FST"]

    startups["founded_at"] = pd.to_datetime(startups["founded_at"])
    startups["age"] = (pd.Timestamp.now() - startups["founded_at"]).dt.days / 365

    acquisition_counts = acquisitions.groupby("acquiring_object_id").size()
    startups["num_acquisitions_made"] = startups["id"].map(acquisition_counts).fillna(0)

    funding_total = funding_rounds.groupby("object_id")["raised_amount_usd"].sum()
    startups["total_funding"] = startups["id"].map(funding_total).fillna(0)
    startups["log_funding"] = np.log1p(startups["total_funding"])

    round_counts = funding_rounds.groupby("object_id").size()
    startups["num_funding_rounds"] = startups["id"].map(round_counts).fillna(0)

    milestone_counts = milestones.groupby("object_id").size()
    startups["num_milestones"] = startups["id"].map(milestone_counts).fillna(0)

    financial_org_investments = (
        investments[
            investments["investor_object_id"].isin(
                objects[objects["entity_type"] == "FinancialOrg"]["id"]
            )
        ]
        .groupby("funded_object_id")
        .size()
    )
    startups["received_financial_investment"] = (
        startups["id"].isin(financial_org_investments.index)
    ).astype(int)

    features = [
        "age",
        "country_code",
        "category_code",
        "num_acquisitions_made",
        "log_funding",
        "num_funding_rounds",
        "num_milestones",
        "investment_rounds",
        "invested_companies",
        "relationships",
        "received_financial_investment",
    ]

    country_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    startups["country_code"] = country_encoder.fit_transform(
        startups["country_code"].fillna("UNKNOWN")
    )
    startups["category_code"] = category_encoder.fit_transform(
        startups["category_code"].fillna("unknown")
    )

    X = startups[features]
    y = startups["status"]

    return X, y, country_encoder, category_encoder


def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    model = HistGradientBoostingClassifier(max_iter=100, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, X_train.columns


def save_model(model, feature_names, country_encoder, category_encoder):
    if not os.path.exists("models"):
        os.makedirs("models")

    with open("models/startup_success_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    with open("models/country_encoder.pkl", "wb") as f:
        pickle.dump(country_encoder, f)

    with open("models/category_encoder.pkl", "wb") as f:
        pickle.dump(category_encoder, f)


if __name__ == "__main__":
    X, y, country_encoder, category_encoder = load_and_preprocess_data()
    model, feature_names = train_and_evaluate_model(X, y)
    save_model(model, feature_names, country_encoder, category_encoder)
