from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "model/health_model.pkl"


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "health_model.pkl not found. "
        "Please run: python train_model.py"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model_package = joblib.load(
    MODEL_PATH
)


model = model_package["model"]

features = model_package["features"]

symptoms = model_package["symptoms"]

medical_history = model_package["medical_history"]

health_parameters = model_package["health_parameters"]


# ============================================================
# GENERAL GUIDANCE
# ============================================================

guidance = {

    "Common Cold": [
        "Get adequate rest.",
        "Maintain adequate fluid intake.",
        "Monitor symptoms.",
        "Seek medical advice if symptoms become severe or persistent."
    ],

    "Flu-like Illness": [
        "Rest and maintain hydration.",
        "Monitor temperature and symptoms.",
        "Avoid close contact with others when unwell.",
        "Consult a healthcare professional if symptoms are severe."
    ],

    "Migraine Risk": [
        "Rest in a quiet environment.",
        "Maintain hydration.",
        "Monitor headache patterns.",
        "Seek medical evaluation for severe or unusual headaches."
    ],

    "Gastrointestinal Infection Risk": [
        "Maintain hydration.",
        "Take fluids in small amounts if necessary.",
        "Monitor for signs of dehydration.",
        "Seek medical advice if symptoms are severe or persistent."
    ],

    "Allergy-related Symptoms": [
        "Avoid known triggers where possible.",
        "Keep the environment clean.",
        "Monitor symptoms.",
        "Consult a healthcare professional for persistent symptoms."
    ],

    "Respiratory Risk": [
        "Avoid smoke and respiratory irritants.",
        "Rest and monitor breathing.",
        "Follow any existing healthcare plan.",
        "Seek urgent medical attention for severe breathing difficulty."
    ],

    "Urinary Health Risk": [
        "Maintain adequate hydration.",
        "Monitor symptoms.",
        "Consult a healthcare professional for proper evaluation."
    ],

    "Metabolic Health Risk": [
        "Maintain balanced nutrition.",
        "Stay physically active appropriately.",
        "Regular health screening may be useful.",
        "Discuss persistent symptoms with a healthcare professional."
    ],

    "Skin-related Infection Risk": [
        "Keep affected skin clean.",
        "Avoid irritating the area.",
        "Monitor for worsening symptoms.",
        "Consult a healthcare professional if symptoms persist."
    ],

    "Joint Health Risk": [
        "Maintain appropriate physical activity.",
        "Avoid activities that significantly worsen pain.",
        "Monitor persistent joint symptoms.",
        "Discuss ongoing symptoms with a healthcare professional."
    ]
}


# ============================================================
# CHATBOT KNOWLEDGE
# ============================================================

chatbot_knowledge = {

    "fever": (
        "For a mild fever, rest, maintain adequate fluids, "
        "and monitor your temperature. If fever is persistent, "
        "severe, or accompanied by concerning symptoms, seek medical advice."
    ),

    "cough": (
        "For a mild cough, rest and maintain adequate fluids. "
        "Avoid smoke and other irritants. A persistent or severe "
        "cough should be evaluated by a healthcare professional."
    ),

    "headache": (
        "Rest in a quiet environment and maintain hydration. "
        "If a headache is sudden, unusually severe, or associated "
        "with other concerning symptoms, seek urgent medical evaluation."
    ),

    "vomiting": (
        "Try small amounts of fluid regularly if you can tolerate them. "
        "If vomiting continues or you cannot maintain hydration, "
        "seek medical advice."
    ),

    "diarrhea": (
        "Maintain hydration and consider oral rehydration fluids. "
        "Seek medical advice if symptoms are severe or persistent."
    ),

    "cold": (
        "Common cold symptoms often include a runny nose, cough, "
        "sore throat and fatigue. Rest, hydration and monitoring "
        "of symptoms are generally helpful."
    ),

    "flu": (
        "Flu-like illness can include fever, cough, headache, "
        "body aches and fatigue. Rest and hydration are important, "
        "and severe symptoms should be evaluated by a healthcare professional."
    ),

    "allergy": (
        "Allergy symptoms can include sneezing, runny nose, cough "
        "or skin symptoms. Avoid known triggers and consult a "
        "healthcare professional if symptoms persist."
    ),

    "asthma": (
        "Asthma can cause coughing, wheezing or breathing difficulty. "
        "If you have a prescribed asthma action plan, follow it. "
        "Severe breathing difficulty requires urgent medical attention."
    ),

    "diabetes": (
        "Diabetes can affect blood glucose regulation. Symptoms such "
        "as unusual thirst or frequent urination should be discussed "
        "with a healthcare professional and appropriate testing."
    ),

    "blood pressure": (
        "Blood pressure should be interpreted using appropriate "
        "measurements and clinical context. Persistently abnormal "
        "readings should be discussed with a healthcare professional."
    ),

    "healthy": (
        "Healthy habits generally include balanced nutrition, "
        "appropriate physical activity, adequate sleep, hygiene, "
        "stress management and recommended healthcare checkups."
    ),

    "doctor": (
        "Consider medical evaluation when symptoms are severe, "
        "persistent, worsening, unusual, or significantly affect "
        "normal activities."
    ),

    "emergency": (
        "If there is severe breathing difficulty, severe chest pain, "
        "fainting, sudden neurological symptoms, or another serious "
        "emergency, seek emergency medical care immediately."
    )
}


# ============================================================
# CHATBOT FUNCTION
# ============================================================

def chatbot_response(question):

    question = question.lower().strip()


    if not question:

        return (
            "Please type a health-related question."
        )


    # Emergency keywords first

    emergency_words = [
        "severe chest pain",
        "can't breathe",
        "cannot breathe",
        "severe breathing",
        "fainting",
        "unconscious"
    ]


    for word in emergency_words:

        if word in question:

            return chatbot_knowledge["emergency"]


    # Keyword matching

    for keyword, answer in chatbot_knowledge.items():

        if keyword in question:

            return answer


    return (
        "I can provide general health information about symptoms, "
        "common illnesses, healthy habits, blood pressure, allergies, "
        "diabetes, asthma and when to seek medical care. "
        "For personal medical advice, please consult a qualified healthcare professional."
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        symptoms=symptoms,
        medical_history=medical_history
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        # ----------------------------------------------------
        # Basic information
        # ----------------------------------------------------

        age = float(
            data.get("age", 30)
        )

        temperature = float(
            data.get("temperature", 36.8)
        )

        heart_rate = float(
            data.get("heart_rate", 75)
        )

        systolic_bp = float(
            data.get("systolic_bp", 120)
        )

        diastolic_bp = float(
            data.get("diastolic_bp", 80)
        )

        bmi = float(
            data.get("bmi", 22)
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if age < 1 or age > 120:

            return jsonify({
                "success": False,
                "message": "Please enter a valid age."
            }), 400


        if temperature < 30 or temperature > 45:

            return jsonify({
                "success": False,
                "message": "Please enter a valid temperature."
            }), 400


        if heart_rate < 30 or heart_rate > 220:

            return jsonify({
                "success": False,
                "message": "Please enter a valid heart rate."
            }), 400


        # ----------------------------------------------------
        # Selected symptoms
        # ----------------------------------------------------

        selected_symptoms = data.get(
            "symptoms",
            []
        )


        if not selected_symptoms:

            return jsonify({
                "success": False,
                "message": "Please select at least one symptom."
            }), 400


        # ----------------------------------------------------
        # Medical history
        # ----------------------------------------------------

        selected_history = data.get(
            "medical_history",
            []
        )


        # ----------------------------------------------------
        # Create feature vector
        # ----------------------------------------------------

        values = []


        # Symptoms

        for symptom in symptoms:

            if symptom in selected_symptoms:
                values.append(1)
            else:
                values.append(0)


        # Medical history

        for history in medical_history:

            if history in selected_history:
                values.append(1)
            else:
                values.append(0)


        # Health parameters

        values.extend([
            age,
            temperature,
            heart_rate,
            systolic_bp,
            diastolic_bp,
            bmi
        ])


        # ----------------------------------------------------
        # Data for model
        # ----------------------------------------------------

        input_array = np.array(
            values,
            dtype=float
        ).reshape(
            1,
            -1
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_array
        )[0]


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_array
        )[0]


        class_names = model.classes_


        probability_map = dict(
            zip(
                class_names,
                probabilities
            )
        )


        # ----------------------------------------------------
        # Top 3
        # ----------------------------------------------------

        top_results = sorted(
            probability_map.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]


        top_predictions = []


        for disease, probability in top_results:

            top_predictions.append({

                "disease": disease,

                "probability": round(
                    probability * 100,
                    2
                )

            })


        confidence = round(
            probability_map[prediction] * 100,
            2
        )


        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        readable_symptoms = [

            symptom
            .replace("_", " ")
            .title()

            for symptom in selected_symptoms

        ]


        readable_history = [

            history
            .replace("_history", "")
            .replace("_", " ")
            .title()

            for history in selected_history

        ]


        explanation = (

            "The model generated this estimate from the "
            "symptoms, medical-history selections and basic "
            "health parameters entered by the user. "

        )


        if readable_symptoms:

            explanation += (

                "Selected symptoms include "
                + ", ".join(readable_symptoms)
                + ". "

            )


        if readable_history:

            explanation += (

                "Selected medical-history factors include "
                + ", ".join(readable_history)
                + "."

            )


        # ----------------------------------------------------
        # Guidance
        # ----------------------------------------------------

        preventive_guidance = guidance.get(
            prediction,
            [
                "Monitor your symptoms.",
                "Maintain healthy habits.",
                "Consult a healthcare professional for personal medical advice."
            ]
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "prediction": prediction,

            "confidence": confidence,

            "top_predictions": top_predictions,

            "explanation": explanation,

            "guidance": preventive_guidance

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "message": str(error)

        }), 500


# ============================================================
# CHATBOT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    try:

        data = request.get_json()

        question = data.get(
            "question",
            ""
        )


        answer = chatbot_response(
            question
        )


        return jsonify({

            "success": True,

            "answer": answer

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "message": str(error)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health_check():

    return jsonify({

        "status": "running",

        "project": "AI-Powered Health Assistant",

        "model_loaded": True

    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )