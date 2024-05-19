from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Data dictionary of 5 students
students_data = [
    {"name": "John", "cgpa": 8.5, "marks": 850, "percentage": 85},
    {"name": "Alice", "cgpa": 9.2, "marks": 920, "percentage": 92},
    {"name": "Bob", "cgpa": 7.8, "marks": 780, "percentage": 78},
    {"name": "Emily", "cgpa": 8.9, "marks": 890, "percentage": 89},
    {"name": "Tom", "cgpa": 8.0, "marks": 800, "percentage": 80}
]

# Extracting features and target
X = np.array([[student["marks"], student["percentage"]] for student in students_data])
y = np.array([student["cgpa"] for student in students_data])

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
