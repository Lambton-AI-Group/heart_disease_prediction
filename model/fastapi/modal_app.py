import modal
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

# Modal App Definition
image = modal.Image.debian_slim().pip_install(
    ["fastapi[standard]", "pandas", "joblib", "shap", "scikit-learn", "requests"]
)
app = modal.App(image=image)

# Load Models and Features
@app.function()
def initialize():
    logistic_model = joblib.load(
        'https://lamtonbucket.s3.us-east-2.amazonaws.com/logistic_regression_model.pkl'
    )
    tree_model = joblib.load(
        'https://lamtonbucket.s3.us-east-2.amazonaws.com/decision_tree_explainer.pkl'
    )
    data = pd.read_csv(
        'https://lamtonbucket.s3.us-east-2.amazonaws.com/cleaned_data_numeric.csv'
    )
    data['HadAngina_GeneralHealth'] = data['HadAngina'] * data['GeneralHealth']
    data['AgeCategory_HadStroke'] = data['AgeCategory'] * data['HadStroke']
    data['DifficultyWalking_GeneralHealth'] = data['DifficultyWalking'] * data['GeneralHealth']
    data['HadDiabetes_DifficultyWalking'] = data['HadDiabetes'] * data['DifficultyWalking']

    model_2_features = [
        'HadAngina', 'GeneralHealth', 'HadStroke', 'AgeCategory', 'ChestScan', 'DifficultyWalking',
        'HadAngina_GeneralHealth', 'AgeCategory_HadStroke',
        'DifficultyWalking_GeneralHealth', 'HadDiabetes_DifficultyWalking'
    ]

    return logistic_model, tree_model, model_2_features, data

# Define Input Schema
class PredictionInput(BaseModel):
    HadAngina: int
    GeneralHealth: int
    HadStroke: int
    AgeCategory: int
    ChestScan: int
    DifficultyWalking: int
    HadDiabetes: int

# Define Prediction Endpoint
@app.function()
@modal.web_endpoint(method="POST")
def predict(input_data: PredictionInput):
    # Load dependencies
    logistic_model, tree_model, model_2_features, data = initialize()

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Compute Interaction Terms
    input_df['HadAngina_GeneralHealth'] = input_df['HadAngina'] * input_df['GeneralHealth']
    input_df['AgeCategory_HadStroke'] = input_df['AgeCategory'] * input_df['HadStroke']
    input_df['DifficultyWalking_GeneralHealth'] = input_df['DifficultyWalking'] * input_df['GeneralHealth']
    input_df['HadDiabetes_DifficultyWalking'] = input_df['HadDiabetes'] * input_df['DifficultyWalking']

    # Ensure correct feature order
    model_2_input = input_df[model_2_features]

    # Logistic Regression Prediction
    prediction_probability = logistic_model.predict_proba(model_2_input)[0, 1]

    # Decision Tree Explanation Prediction
    explanation_prediction = tree_model.predict(model_2_input)[0]

    # SHAP Values
    explainer = shap.Explainer(logistic_model, data[model_2_features])
    shap_values = explainer(model_2_input)
    shap_contributions = dict(zip(model_2_features, shap_values.values[0]))

    # Top Feature Contributions
    sorted_contributions = sorted(
        shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    insights = [
        {
            "feature": feature,
            "contribution": round(contribution, 3),
            "impact": "positive" if contribution > 0 else "negative"
        }
        for feature, contribution in sorted_contributions[:5]
    ]

    # Response
    response = {
        "prediction_probability": round(prediction_probability, 3),
        "decision_tree_explanation": round(explanation_prediction, 3),
        "insights": insights
    }

    return JSONResponse(response)
