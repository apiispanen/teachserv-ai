import streamlit as st
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
    st.write(df)

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
st.image("ts.png", width=80)
st.title("Teachserv AI Generator")
st.subheader("A Predictive Model for APTI Scoring")
st.write("""
This application uses a Machine Learning (ML) model to predict the scoring points for various classes on Teachserv platform. 

Teachserv, an innovative e-learning platform, is considering implementing a new scoring system to quantitatively evaluate the progress of their students, akin to a GPA. This machine learning model is proposed as a key component of this new scoring system.

The model is trained on a diverse set of course features such as professor credentials, the number of topics covered, the price of the course, student feedback, student success rate, the difficulty of the course, the average GPA of students, and the number of awards the course or professor has received.

Our ML model is a Random Forest Regressor, an ensemble learning method that generates multiple decision trees and outputs the mean prediction of the individual trees. This provides a robust and generalized scoring system.

The Mean Squared Error (MSE) is used as a metric to evaluate the model's accuracy. The lower the MSE, the more accurate the model is in making predictions.

In the 'User Input Parameters' section below, you can input different parameters to see how they influence the predicted scoring points. This tool can help to understand how different aspects of a course might contribute to a student's overall score.

However, please bear in mind that the model's predictions are not absolute. They are advisory and should be used as one of many factors when considering the development of the new scoring system.
""")
st.markdown('***')

st.subheader("Predictive Model Breakdown")
st.write(f"Mean Squared Error: {mse}")

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
st.write(importances_df)

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
st.markdown('***')

# Use Streamlit to create user inputs
st.header('Make a Prediction')
st.subheader('User Input Parameters')
professor_credentials = st.slider('Professor Credentials (Rank)', 0, 3, 2)
topics_covered = st.slider('Topics Covered', 1, 10, 7)
price = st.slider('Price', 50, 500, 200)
student_feedback = st.slider('Student Feedback', 1.0, 5.0, 4.5)
student_success_rate = st.slider('Student Success Rate', 0.0, 1.0, 0.85)
difficulty = st.slider('Difficulty', 1.0, 5.0, 3.0)
gpa = st.slider('GPA', 0.0, 4.0, 3.5)
awards = st.slider('Awards', 0, 5, 1)

output = predict_output(professor_credentials, topics_covered, price, student_feedback, student_success_rate, difficulty, gpa, awards)
st.metric("Predicted Output", output)
