import os
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# FOLDERS
# ============================================================

os.makedirs("model", exist_ok=True)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# FEATURES
# ============================================================

symptoms = [
    "fever",
    "cough",
    "sore_throat",
    "runny_nose",
    "headache",
    "body_ache",
    "fatigue",
    "nausea",
    "vomiting",
    "diarrhea",
    "stomach_pain",
    "chest_pain",
    "shortness_of_breath",
    "dizziness",
    "frequent_urination",
    "excessive_thirst",
    "skin_rash",
    "joint_pain"
]

medical_history = [
    "diabetes_history",
    "asthma_history",
    "heart_disease_history",
    "hypertension_history",
    "allergy_history"
]

health_parameters = [
    "age",
    "temperature",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "bmi"
]

features = (
    symptoms +
    medical_history +
    health_parameters
)


# ============================================================
# DISEASES
# ============================================================

diseases = [
    "Common Cold",
    "Flu-like Illness",
    "Migraine Risk",
    "Gastrointestinal Infection Risk",
    "Allergy-related Symptoms",
    "Respiratory Risk",
    "Urinary Health Risk",
    "Metabolic Health Risk",
    "Skin-related Infection Risk",
    "Joint Health Risk"
]


# ============================================================
# GENERATE ONE SAMPLE
# ============================================================

def generate_sample(disease):

    row = {}

    # --------------------------------------------------------
    # Default symptom values
    # --------------------------------------------------------

    for symptom in symptoms:
        row[symptom] = 0

    # --------------------------------------------------------
    # Default medical history
    # --------------------------------------------------------

    for history in medical_history:
        row[history] = 0

    # --------------------------------------------------------
    # Basic health parameters
    # --------------------------------------------------------

    row["age"] = random.randint(18, 75)

    row["temperature"] = round(
        random.uniform(36.0, 39.5), 1
    )

    row["heart_rate"] = random.randint(
        55, 115
    )

    row["systolic_bp"] = random.randint(
        95, 165
    )

    row["diastolic_bp"] = random.randint(
        60, 105
    )

    row["bmi"] = round(
        random.uniform(17, 35), 1
    )


    # ========================================================
    # DISEASE PATTERNS
    # ========================================================

    if disease == "Common Cold":

        selected = [
            "cough",
            "sore_throat",
            "runny_nose",
            "fatigue"
        ]

        for s in selected:
            row[s] = 1

        row["temperature"] = round(
            random.uniform(36.5, 38.0), 1
        )


    elif disease == "Flu-like Illness":

        selected = [
            "fever",
            "cough",
            "headache",
            "body_ache",
            "fatigue"
        ]

        for s in selected:
            row[s] = 1

        row["temperature"] = round(
            random.uniform(38.0, 39.5), 1
        )


    elif disease == "Migraine Risk":

        selected = [
            "headache",
            "nausea",
            "dizziness",
            "fatigue"
        ]

        for s in selected:
            row[s] = 1


    elif disease == "Gastrointestinal Infection Risk":

        selected = [
            "nausea",
            "vomiting",
            "diarrhea",
            "stomach_pain",
            "fatigue"
        ]

        for s in selected:
            row[s] = 1


    elif disease == "Allergy-related Symptoms":

        selected = [
            "runny_nose",
            "cough",
            "skin_rash"
        ]

        for s in selected:
            row[s] = 1

        row["allergy_history"] = 1


    elif disease == "Respiratory Risk":

        selected = [
            "cough",
            "shortness_of_breath",
            "chest_pain",
            "fatigue"
        ]

        for s in selected:
            row[s] = 1

        row["asthma_history"] = 1


    elif disease == "Urinary Health Risk":

        selected = [
            "frequent_urination",
            "stomach_pain",
            "fatigue"
        ]

        for s in selected:
            row[s] = 1


    elif disease == "Metabolic Health Risk":

        selected = [
            "frequent_urination",
            "excessive_thirst",
            "fatigue"
        ]

        for s in selected:
            row[s] = 1

        row["diabetes_history"] = 1


    elif disease == "Skin-related Infection Risk":

        selected = [
            "skin_rash",
            "fever",
            "fatigue"
        ]

        for s in selected:
            row[s] = 1


    elif disease == "Joint Health Risk":

        selected = [
            "joint_pain",
            "body_ache",
            "fatigue"
        ]

        for s in selected:
            row[s] = 1


    # ========================================================
    # RANDOM VARIATIONS
    # ========================================================

    for symptom in symptoms:

        if random.random() < 0.08:
            row[symptom] = 1 - row[symptom]


    return row


# ============================================================
# CREATE DATASET
# ============================================================

rows = []

samples_per_disease = 250

for disease in diseases:

    for _ in range(samples_per_disease):

        row = generate_sample(disease)

        row["disease"] = disease

        rows.append(row)


df = pd.DataFrame(rows)


# ============================================================
# TRAINING DATA
# ============================================================

X = df[features]

y = df["disease"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST MODEL
# ============================================================

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("=" * 55)
print("AI HEALTH ASSISTANT - MODEL TRAINING")
print("=" * 55)
print("Dataset records :", len(df))
print("Features        :", len(features))
print("Diseases        :", len(diseases))
print("Model accuracy  :", round(accuracy * 100, 2), "%")
print("=" * 55)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {

    "model": model,

    "features": features,

    "symptoms": symptoms,

    "medical_history": medical_history,

    "health_parameters": health_parameters,

    "diseases": diseases
}


joblib.dump(
    model_package,
    "model/health_model.pkl"
)


print()
print("SUCCESS!")
print("Model saved at:")
print("model/health_model.pkl")
print()