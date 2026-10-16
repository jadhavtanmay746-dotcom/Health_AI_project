import streamlit as st
import joblib
import numpy as np


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

MODEL_PATH = "model/health_model.pkl"

try:
    package = joblib.load(MODEL_PATH)

    model = package["model"]
    symptoms = package["symptoms"]
    medical_history = package["medical_history"]

except Exception as e:
    st.error("Unable to load the ML model.")
    st.code(str(e))
    st.stop()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🩺 AI-Powered Health Assistant")

st.write(
    "Educational AIML prototype that analyzes symptoms, "
    "medical history and basic health parameters."
)

st.warning(
    "This is an educational prototype and is not a medical diagnosis."
)


# ---------------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------------

st.header("1. Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )

with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=30.0,
        max_value=45.0,
        value=36.8,
        step=0.1
    )

with col3:
    heart_rate = st.number_input(
        "Heart Rate (BPM)",
        min_value=30,
        max_value=220,
        value=75
    )


col4, col5, col6 = st.columns(3)

with col4:
    systolic_bp = st.number_input(
        "Systolic BP",
        min_value=60,
        max_value=250,
        value=120
    )

with col5:
    diastolic_bp = st.number_input(
        "Diastolic BP",
        min_value=30,
        max_value=150,
        value=80
    )

with col6:
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=22.0,
        step=0.1
    )


# ---------------------------------------------------------
# SYMPTOMS
# ---------------------------------------------------------

st.header("2. Select Symptoms")

selected_symptoms = st.multiselect(
    "Choose the symptoms:",
    options=symptoms,
    format_func=lambda x: x.replace("_", " ").title()
)


# ---------------------------------------------------------
# MEDICAL HISTORY
# ---------------------------------------------------------

st.header("3. Medical History")

selected_history = st.multiselect(
    "Choose relevant medical history:",
    options=medical_history,
    format_func=lambda x:
        x.replace("_history", "")
         .replace("_", " ")
         .title()
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

st.header("4. Health Prediction")

if st.button(
    "🔍 Analyze Health Information",
    type="primary"
):

    if not selected_symptoms:

        st.error(
            "Please select at least one symptom."
        )

    else:

        # -----------------------------------------------
        # CREATE INPUT VECTOR
        # -----------------------------------------------

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

        input_array = np.array(
            values,
            dtype=float
        ).reshape(1, -1)


        # -----------------------------------------------
        # PREDICTION
        # -----------------------------------------------

        prediction = model.predict(
            input_array
        )[0]


        probabilities = model.predict_proba(
            input_array
        )[0]


        classes = model.classes_


        results = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )


        # -----------------------------------------------
        # MAIN RESULT
        # -----------------------------------------------

        confidence = (
            results[0][1] * 100
        )

        st.success(
            f"Highest model likelihood: {prediction}"
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # -----------------------------------------------
        # TOP 3
        # -----------------------------------------------

        st.subheader(
            "Top 3 Model Estimates"
        )

        for disease, probability in results[:3]:

            st.write(
                f"**{disease}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


        # -----------------------------------------------
        # EXPLANATION
        # -----------------------------------------------

        st.subheader(
            "Why was this result produced?"
        )

        symptom_text = ", ".join(
            s.replace("_", " ").title()
            for s in selected_symptoms
        )

        if selected_history:

            history_text = ", ".join(
                h.replace("_history", "")
                 .replace("_", " ")
                 .title()
                for h in selected_history
            )

        else:

            history_text = "No selected medical-history factors"


        st.info(
            "The model generated this estimate using "
            "the selected symptoms, medical history and "
            "basic health parameters such as age, temperature, "
            "heart rate, blood pressure and BMI."
        )

        st.write(
            "**Selected symptoms:**",
            symptom_text
        )

        st.write(
            "**Medical history:**",
            history_text
        )


        # -----------------------------------------------
        # GENERAL GUIDANCE
        # -----------------------------------------------

        st.subheader(
            "General Preventive Guidance"
        )

        guidance = {

            "Common Cold": [
                "Get adequate rest.",
                "Maintain adequate fluid intake.",
                "Monitor symptoms."
            ],

            "Flu-like Illness": [
                "Rest and maintain hydration.",
                "Monitor temperature and symptoms.",
                "Seek professional medical advice if symptoms are severe."
            ],

            "Migraine Risk": [
                "Rest in a quiet environment.",
                "Maintain hydration.",
                "Seek medical evaluation for severe or unusual headaches."
            ],

            "Gastrointestinal Infection Risk": [
                "Maintain hydration.",
                "Monitor symptoms.",
                "Seek medical advice if symptoms are severe or persistent."
            ],

            "Allergy-related Symptoms": [
                "Avoid known triggers.",
                "Monitor symptoms.",
                "Consult a healthcare professional for persistent symptoms."
            ],

            "Respiratory Risk": [
                "Avoid smoke and respiratory irritants.",
                "Monitor breathing.",
                "Seek urgent medical care for severe breathing difficulty."
            ],

            "Urinary Health Risk": [
                "Maintain adequate hydration.",
                "Monitor symptoms.",
                "Consult a healthcare professional."
            ],

            "Metabolic Health Risk": [
                "Maintain balanced nutrition.",
                "Stay appropriately active.",
                "Discuss persistent symptoms with a healthcare professional."
            ],

            "Skin-related Infection Risk": [
                "Keep affected skin clean.",
                "Monitor symptoms.",
                "Consult a healthcare professional if symptoms persist."
            ],

            "Joint Health Risk": [
                "Maintain appropriate physical activity.",
                "Monitor persistent joint symptoms.",
                "Discuss ongoing symptoms with a healthcare professional."
            ]
        }


        for item in guidance.get(
            prediction,
            [
                "Monitor your symptoms.",
                "Maintain healthy habits.",
                "Consult a healthcare professional for personal advice."
            ]
        ):

            st.write(
                "• " + item
            )


# ---------------------------------------------------------
# CHATBOT
# ---------------------------------------------------------

st.header("5. 💬 AI Health Chatbot")

st.write(
    "Ask a general health-related question."
)


question = st.text_input(
    "Your question:",
    placeholder="Example: What should I do for fever?"
)


def chatbot_response(question):

    question = question.lower().strip()

    if "fever" in question:

        return (
            "For a mild fever, rest, maintain adequate fluids "
            "and monitor your temperature. Persistent or severe "
            "symptoms should be evaluated by a healthcare professional."
        )

    if "cough" in question:

        return (
            "For a mild cough, rest and maintain adequate fluids. "
            "Avoid smoke and other irritants. Persistent or severe "
            "cough should be evaluated by a healthcare professional."
        )

    if "headache" in question:

        return (
            "Rest in a quiet environment and maintain hydration. "
            "A severe or unusual headache should be evaluated "
            "by a healthcare professional."
        )

    if "diabetes" in question:

        return (
            "Diabetes affects blood glucose regulation. Symptoms "
            "such as unusual thirst or frequent urination should "
            "be discussed with a healthcare professional."
        )

    if "asthma" in question:

        return (
            "Asthma can cause coughing and breathing difficulty. "
            "Follow your prescribed healthcare plan if you have one. "
            "Severe breathing difficulty requires urgent medical care."
        )

    if "allergy" in question:

        return (
            "Allergy symptoms can include a runny nose, coughing "
            "or skin symptoms. Avoid known triggers and seek "
            "professional advice for persistent symptoms."
        )

    if "healthy" in question:

        return (
            "Healthy habits generally include balanced nutrition, "
            "appropriate physical activity, adequate sleep, hygiene "
            "and recommended healthcare checkups."
        )

    if "doctor" in question:

        return (
            "Consider medical evaluation when symptoms are severe, "
            "persistent, worsening or unusual."
        )

    return (
        "I can answer general questions about fever, cough, "
        "headache, allergies, asthma, diabetes, healthy habits "
        "and when to seek medical care."
    )


if st.button("💬 Ask AI"):

    if question.strip():

        answer = chatbot_response(
            question
        )

        st.info(
            answer
        )

    else:

        st.warning(
            "Please enter a question."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "AI-Powered Health Assistant | Educational AIML Project | "
    "Not a medical diagnostic tool"
)