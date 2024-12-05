from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import shap
import uvicorn
from sklearn.model_selection import train_test_split
import os

# Load the saved models
logistic_model = joblib.load('./fastapi/logistic_regression_model.pkl')
tree_model = joblib.load('./fastapi/decision_tree_explainer.pkl')

# Load dataset and compute interaction terms
data = pd.read_csv('./fastapi/cleaned_data_numeric.csv')
data['HadAngina_GeneralHealth'] = data['HadAngina'] * data['GeneralHealth']
data['AgeCategory_HadStroke'] = data['AgeCategory'] * data['HadStroke']
data['DifficultyWalking_GeneralHealth'] = data['DifficultyWalking'] * data['GeneralHealth']
data['HadDiabetes_DifficultyWalking'] = data['HadDiabetes'] * data['DifficultyWalking']

# Define features used during training
model_1_features = ['HadAngina', 'GeneralHealth', 'HadStroke', 'AgeCategory', 'ChestScan', 'DifficultyWalking']
model_2_features = model_1_features + [
    'HadAngina_GeneralHealth', 'AgeCategory_HadStroke',
    'DifficultyWalking_GeneralHealth', 'HadDiabetes_DifficultyWalking'
]

X = data[model_2_features]
y = data['HeartDisease']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SHAP Explainer Initialization
explainer = shap.Explainer(logistic_model, X_train)

# FastAPI App Initialization
app = FastAPI(title="Heart Disease Prediction API with SHAP")


class InputData(BaseModel):
    HadAngina: int
    GeneralHealth: int
    HadStroke: int
    AgeCategory: int
    ChestScan: int
    DifficultyWalking: int
    HadDiabetes: int


@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running!"}


@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        # Compute interaction terms
        input_df['HadAngina_GeneralHealth'] = input_df['HadAngina'] * input_df['GeneralHealth']
        input_df['AgeCategory_HadStroke'] = input_df['AgeCategory'] * input_df['HadStroke']
        input_df['DifficultyWalking_GeneralHealth'] = input_df['DifficultyWalking'] * input_df['GeneralHealth']
        input_df['HadDiabetes_DifficultyWalking'] = input_df['HadDiabetes'] * input_df['DifficultyWalking']

        # Ensure the input has the correct features
        model_2_input = input_df[model_2_features]

        # Get predictions from the models
        prediction_probability = logistic_model.predict_proba(model_2_input)[:, 1][0]
        explanation_prediction = tree_model.predict(model_2_input)[0]

        # Compute SHAP values for feature contributions
        shap_values = explainer(model_2_input)
        shap_contributions = dict(zip(model_2_features, shap_values.values[0]))

        # Sort by absolute contribution
        sorted_contributions = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        # Format insights
        insights = [
            {
                "feature": feature,
                "contribution": round(contribution, 3),
                "impact": "positive" if contribution > 0 else "negative"
            }
            for feature, contribution in sorted_contributions[:5]  # Top 5 contributions
        ]

        # Prepare response
        response = {
            "prediction_probability": round(prediction_probability, 3),
            "decision_tree_explanation": round(explanation_prediction, 3),
            "insights": insights
        }
        return response

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
