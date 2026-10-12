print("Student Performance Prediction")
print("Project started successfully")
import pandas as pd

print("Pandas imported successfully")
df = pd.DataFrame({
    "Maths": [70, 80, 65, 90, 75],
    "Science": [75, 85, 70, 95, 80],
    "English": [72, 78, 68, 88, 76],
    "Attendance": [85, 90, 80, 95, 88]
})

print(df)
X = df[["Science", "English", "Attendance"]]
y = df["Maths"]

print("Features and target selected successfully")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training and testing data split successfully")
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully")
from sklearn.metrics import mean_absolute_error

y_pred = model.predict(X_test)

error = mean_absolute_error(y_test, y_pred)

print("Actual Maths Marks:", list(y_test))
print("Predicted Maths Marks:", y_pred.round(2))
print("Mean Absolute Error:", round(error, 2))
new_student = pd.DataFrame({
    "Science": [82],
    "English": [79],
    "Attendance": [90]
})

predicted_marks = model.predict(new_student)

print("New Student Predicted Maths Marks:", round(predicted_marks[0], 2))
