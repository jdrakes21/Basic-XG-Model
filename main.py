import os
print(os.getcwd())
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings  # Import the warnings module

# Suppress UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)


# Load your dataset with features (e.g., shot_distance, shot_angle, etc.)
# and labels (0 for misses, 1 for goals)
data = pd.read_csv('synthetic_xg_dataset.csv')

# Define features (X) and labels (y)
X = data[['shot_distance', 'shot_angle']].values
y = data['is_goal'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
# Modify the classification report to calculate metrics only for class 0
report = classification_report(y_test, y_pred, labels=[0], target_names=['Class 0'])

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)

# Now, you can use the model to predict xG for new shots
new_shot_distance = 20  # Replace with the actual distance of the shot
new_shot_angle = 30    # Replace with the actual angle of the shot

# Predict the probability of the shot resulting in a goal
xG_probability = model.predict_proba([[new_shot_distance, new_shot_angle]])[:, 1]

print(f"Expected Goals (xG) Probability: {xG_probability[0]}")