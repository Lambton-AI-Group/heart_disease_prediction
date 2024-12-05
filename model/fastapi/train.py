from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

data = pd.read_csv('/Users/ernestgaisie/Desktop/heart-disease-prediction/model/cleaning/cleaned_data_numeric.csv')


model_1_features = ['HadAngina', 'GeneralHealth', 'HadStroke', 
                    'AgeCategory', 'ChestScan', 'DifficultyWalking']

data['HadAngina_GeneralHealth'] = data['HadAngina'] * data['GeneralHealth']
data['AgeCategory_HadStroke'] = data['AgeCategory'] * data['HadStroke']
data['DifficultyWalking_GeneralHealth'] = data['DifficultyWalking'] * data['GeneralHealth']
data['HadDiabetes_DifficultyWalking'] = data['HadDiabetes'] * data['DifficultyWalking']

# Select features for Model 2
model_2_features = model_1_features + [
    'HadAngina_GeneralHealth', 'AgeCategory_HadStroke', 
    'DifficultyWalking_GeneralHealth', 'HadDiabetes_DifficultyWalking'
]

X = data[model_2_features]
y = data['HeartDisease']


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model_2 = LogisticRegression(max_iter=1000)
model_2.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_2.predict(X_test)
y_pred_prob = model_2.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Model 2 Accuracy: {accuracy:.2f}")
print(f"Model 2 ROC AUC: {roc_auc:.2f}")

# Model 1's probability predictions as the target for Model 2
y_prob_train = model_2.predict_proba(X_train)[:, 1]  # Logistic regression probabilities
y_prob_test = model_2.predict_proba(X_test)[:, 1]



# Train a decision tree regressor on the same features to explain Model 1's predictions
explanation_model = DecisionTreeRegressor(max_depth=3, random_state=42)
explanation_model.fit(X_train, y_prob_train)

# Evaluate the performance of the explanation model
y_explained = explanation_model.predict(X_test)
mse = mean_squared_error(y_prob_test, y_explained)

print(f"Explanation Model MSE: {mse:.4f}")

joblib.dump(model_2, 'logistic_regression_model.pkl')
joblib.dump(explanation_model, 'decision_tree_explainer.pkl')