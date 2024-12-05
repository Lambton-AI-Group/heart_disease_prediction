from joblib import load
import pandas as pd

# Load the re-saved or re-trained model
tree_model = load('decision_tree_explainer.pkl')

# Prepare sample input
sample_data = pd.DataFrame([{
    'HadAngina': 1,
    'GeneralHealth': 3,
    'HadStroke': 0,
    'AgeCategory': 10,
    'ChestScan': 1,
    'DifficultyWalking': 1,
    'HadAngina_GeneralHealth': 3,
    'AgeCategory_HadStroke': 0,
    'DifficultyWalking_GeneralHealth': 3,
    'HadDiabetes_DifficultyWalking': 0
}])

# Make predictions
prediction = tree_model.predict(sample_data)
print("Prediction:", prediction)
