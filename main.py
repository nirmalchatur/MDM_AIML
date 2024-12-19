import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              BaggingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
file_path = 'Diabetes.csv'
data = pd.read_csv(file_path)

# Step 2: Data Preprocessing
# Clean inconsistencies in the 'CLASS' column
data['CLASS'] = data['CLASS'].str.strip()

# Encode categorical columns 'Gender' and 'CLASS'
label_encoders = {}
for col in ['Gender', 'CLASS']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Normalize numerical columns (excluding IDs and target variable 'CLASS')
numerical_cols = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Step 3: Splitting the Dataset
X = data.drop(columns=['ID', 'No_Pation', 'CLASS'])
y = data['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Define Models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'SVM': SVC(class_weight='balanced', probability=True),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Step 5: Train Models and Evaluate Performance
model_metrics = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    model_metrics[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm
    }

# Step 6: Display Model Performance on Streamlit
st.title("Diabetes Prediction Model 🩺")

st.sidebar.header("Enter Your Details 🥼")

age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", ["M", "F"])
urea = st.sidebar.number_input("Urea level (mg/dL)", min_value=0.0, value=30.0)
cr = st.sidebar.number_input("Creatinine level (Cr) (mg/dL)", min_value=0.0, value=0.8)
hba1c = st.sidebar.number_input("HbA1c (%)", min_value=0.0, value=5.7)
chol = st.sidebar.number_input("Cholesterol level (mg/dL)", min_value=0.0, value=200.0)
tg = st.sidebar.number_input("Triglycerides (TG) level (mg/dL)", min_value=0.0, value=150.0)
hdl = st.sidebar.number_input("HDL level (mg/dL)", min_value=0.0, value=50.0)
ldl = st.sidebar.number_input("LDL level (mg/dL)", min_value=0.0, value=100.0)
vldl = st.sidebar.number_input("VLDL level (mg/dL)", min_value=0.0, value=30.0)
bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value=10.0, value=22.5)

# Encode gender input
gender_encoded = label_encoders['Gender'].transform([gender])[0]

# Prepare the input data dictionary
user_data = {
    'AGE': age,
    'Gender': gender_encoded,
    'Urea': urea,
    'Cr': cr,
    'HbA1c': hba1c,
    'Chol': chol,
    'TG': tg,
    'HDL': hdl,
    'LDL': ldl,
    'VLDL': vldl,
    'BMI': bmi
}

# Step 7: Function to Predict Diabetes for User Input
def predict_diabetes(input_data, model):
    # Preprocess user input
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Make a prediction
    prediction_proba = model.predict_proba(input_df)[0]
    prediction = np.argmax(prediction_proba)

    class_decoded = label_encoders['CLASS'].inverse_transform([prediction])[0]
    return class_decoded, prediction_proba

# Step 8: User Input and Prediction
st.sidebar.header("Choose a Model for Prediction 💡")
model_choice = st.sidebar.selectbox("Select a Model for Prediction", list(models.keys()))

if st.sidebar.button("Predict 🩺"):
    model = models[model_choice]
    class_decoded, prediction_proba = predict_diabetes(user_data, model)

    st.write(f"### Predicted Diabetes Class: {class_decoded}")
    st.write(f"Prediction Probabilities: {prediction_proba}")

# Step 9: Display Model Metrics
st.sidebar.header("Choose Model Performance Metrics 📊")
chosen_model = st.sidebar.selectbox("Select Model for Performance Metrics", list(models.keys()))

if chosen_model:
    metrics = model_metrics[chosen_model]
    st.write(f"### Performance Metrics for {chosen_model}")
    st.write(f"**Accuracy:** {metrics['Accuracy']:.2f}")
    st.write(f"**Precision:** {metrics['Precision']:.2f}")
    st.write(f"**Recall:** {metrics['Recall']:.2f}")
    st.write(f"**F1 Score:** {metrics['F1 Score']:.2f}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoders['CLASS'].classes_,
                yticklabels=label_encoders['CLASS'].classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# Step 10: Compare Metrics Across All Models
st.header("Model Performance Comparison 📊")

# Extract metrics for all models
model_names = list(model_metrics.keys())
accuracies = [model_metrics[model]['Accuracy'] for model in model_names]
recalls = [model_metrics[model]['Recall'] for model in model_names]
f1_scores = [model_metrics[model]['F1 Score'] for model in model_names]

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names))  # the label locations
width = 0.25  # the width of the bars

# Create bars for each metric
ax.bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
ax.bar(x, recalls, width, label='Recall', color='lightgreen')
ax.bar(x + width, f1_scores, width, label='F1 Score', color='lightcoral')

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Scores')
ax.set_xlabel('Models')
ax.set_title('Comparison of Model Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()

# Display the bar chart in Streamlit
st.pyplot(fig)
