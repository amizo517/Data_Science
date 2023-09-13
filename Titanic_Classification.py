import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate some example data (replace this with your real dataset)
data = {
    'Socioeconomic_Status': [1, 2, 3, 2, 1, 3, 2, 1, 3, 2],
    'Age': [30, 45, 25, 50, 60, 35, 40, 55, 28, 48],
    'Gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 1 for male, 0 for female
    'Health': [4, 3, 5, 2, 4, 3, 2, 5, 4, 3],
    'Safe_from_Sinking': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for safe, 0 for not safe
}

df = pd.DataFrame(data)

# Split the data into features and target
X = df[['Socioeconomic_Status', 'Age', 'Gender', 'Health']]
y = df['Safe_from_Sinking']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but often recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Example prediction for a new person
new_person = np.array([[3, 40, 1, 3]])  # Replace with the features of the new person
new_person = scaler.transform(new_person)
prediction = model.predict(new_person)

if prediction[0] == 1:
    print("The person is likely to be safe from sinking.")
else:
    print("The person is not likely to be safe from sinking.")
