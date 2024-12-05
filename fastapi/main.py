from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


# Load the saved models
logistic_model = joblib.load('logistic_regression_model.pkl')
tree_model = joblib.load('decision_tree_explainer.pkl')

data = pd.read_csv(
    'cleaned_data_numeric.csv')

# Define the features used during training
model_1_features = ['HadAngina', 'GeneralHealth', 'HadStroke',
                    'AgeCategory', 'ChestScan', 'DifficultyWalking']

data['HadAngina_GeneralHealth'] = data['HadAngina'] * data['GeneralHealth']
data['AgeCategory_HadStroke'] = data['AgeCategory'] * data['HadStroke']
data['DifficultyWalking_GeneralHealth'] = data['DifficultyWalking'] * \
    data['GeneralHealth']
data['HadDiabetes_DifficultyWalking'] = data['HadDiabetes'] * \
    data['DifficultyWalking']

model_2_features = model_1_features + [
    'HadAngina_GeneralHealth', 'AgeCategory_HadStroke',
    'DifficultyWalking_GeneralHealth', 'HadDiabetes_DifficultyWalking'
]

X = data[model_2_features]
y = data['HeartDisease']


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize SHAP Explainer for Logistic Regression
explainer = shap.Explainer(logistic_model, X_train)


# Initialize FastAPI
app = FastAPI(title="Heart Disease Prediction API with SHAP")

# Define the input schema


class InputData(BaseModel):
    HadAngina: int
    GeneralHealth: int
    HadStroke: int
    AgeCategory: int
    ChestScan: int
    DifficultyWalking: int
    HadDiabetes: int  # Needed to compute interaction terms

# Route for health check


@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running!"}

# Prediction endpoint


@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    # Compute interaction terms
    input_df['HadAngina_GeneralHealth'] = input_df['HadAngina'] * \
        input_df['GeneralHealth']
    input_df['AgeCategory_HadStroke'] = input_df['AgeCategory'] * \
        input_df['HadStroke']
    input_df['DifficultyWalking_GeneralHealth'] = input_df['DifficultyWalking'] * \
        input_df['GeneralHealth']
    input_df['HadDiabetes_DifficultyWalking'] = input_df['HadDiabetes'] * \
        input_df['DifficultyWalking']

    # Ensure the input has the correct features
    model_2_input = input_df[model_2_features]

    # Get predictions from the models
    prediction_probability = logistic_model.predict_proba(model_2_input)[
        :, 1][0]
    explanation_prediction = tree_model.predict(model_2_input)[0]

    # Compute SHAP values for feature contributions
    shap_values = explainer(model_2_input)
    shap_contributions = dict(zip(model_2_features, shap_values.values[0]))

    # Sort by absolute contribution
    sorted_contributions = sorted(
        shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    # Format insights
    insights = [
        {
            "feature": feature,
            "contribution": round(contribution, 3),
            "impact": "positive" if contribution > 0 else "negative"
        }
        # Top 5 contributions
        for feature, contribution in sorted_contributions[:5]
    ]

    # Prepare response
    response = {
        "prediction_probability": round(prediction_probability, 3),
        "decision_tree_explanation": round(explanation_prediction, 3),
        "insights": insights
    }
    return response
