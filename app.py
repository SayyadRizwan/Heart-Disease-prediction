import gradio as gr
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load("heart_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Prediction function
def predict_heart_disease(age, bp, cholesterol, max_hr, ekg, pain_type, angina, vessels, sex, slope, thallium):
    input_data = {
        'Age': age,
        'BP': bp,
        'Cholesterol': cholesterol,
        'Max HR': max_hr,
        'EKG results': ekg,
        'Encoded_Pain_Type': pain_type,
        'Exercise angina': angina,
        'Number of vessels fluro': vessels,
        'Sex': sex,
        'Slope of ST': slope,
        'Thallium': thallium
    }

    input_df = pd.DataFrame([input_data])
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)

    return "Presence of Heart Disease 💔" if prediction[0] == 1 else "No Heart Disease ❤️"

# Gradio UI
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Blood Pressure (BP)"),
        gr.Number(label="Cholesterol"),
        gr.Number(label="Maximum Heart Rate (Max HR)"),
        gr.Number(label="EKG results (0 or 1)"),
        gr.Number(label="Encoded Pain Type (0-3)"),
        gr.Number(label="Exercise Angina (0=No, 1=Yes)"),
        gr.Number(label="Number of Vessels Fluro (0-3)"),
        gr.Radio([0, 1], label="Sex (0=Female, 1=Male)"),
        gr.Number(label="Slope of ST (0-2)"),
        gr.Number(label="Thallium Stress Test Result (3, 6, 7)")
    ],
    outputs="text",
    title="Heart Disease Predictor",
    description="Enter patient details to predict presence or absence of heart disease"
)

iface.launch()