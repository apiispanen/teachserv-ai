import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def generate_random_training():
    # Random seed for reproducibility
    np.random.seed(42)

    # Generate the data
    data = {
        'professor credentials': np.random.randint(0, 4, 20),
        '# Topics Covered': np.random.randint(1, 10, 20),
        'Price': np.random.randint(50, 500, 20),
        'Student Feedback': np.random.uniform(1, 5, 20).round(1),
        'Student Success Rate': np.random.uniform(0, 1, 20).round(2),
        'Difficulty': np.random.uniform(1, 5, 20).round(1),
        'GPA': np.random.uniform(0, 4, 20).round(1),
        '# Awards': np.random.randint(0, 5, 20),
        'Output': np.random.uniform(0, 10, 20).round(1)
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('training_data.csv', index=False)

    # Print the DataFrame
    print(df)

# Load the data
df = pd.read_csv('training_data.csv')

# Split the data into inputs and outputs
X = df.drop('Output', axis=1)
y = df['Output']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Use the model to make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(importances_df)

def predict_output(professor_credentials, topics_covered, price, student_feedback, student_success_rate, difficulty, gpa, awards):
    # Preprocess the input
    input_data = pd.DataFrame({
        'professor credentials': [professor_credentials],
        '# Topics Covered': [topics_covered],
        'Price': [price],
        'Student Feedback': [student_feedback],
        'Student Success Rate': [student_success_rate],
        'Difficulty': [difficulty],
        'GPA': [gpa],
        '# Awards': [awards],
    })

    # Make the prediction
    prediction = model.predict(input_data)
    
    # Return the prediction
    return prediction[0]

# - professor credentials (0,1,2,3)
# - # Topics Covered
# - Price
# - Student Feedback (1-5)
# - Student Success Rate (%)
# - Difficulty (1-5)
# - GPA (0-4)
# - # Awards 

output = predict_output(2, 7, 200, 4.5, 0.85, 3, 3.5, 1)
print(f"Predicted Output: {output}")
