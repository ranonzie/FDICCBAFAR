import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Performance Metrics Data -----
performance_data = {
    'Model': ['XGBoost', 'Voting', 'SVM', 'RandomForest'],
    'ROC AUC': [0.876475, 0.870923, 0.844930, 0.834949],
    'F1': [0.122905, 0.138614, 0.151955, 0.006944],
    'Precision': [0.289474, 0.229508, 0.110930, 0.166667],
    'Recall': [0.078014, 0.099291, 0.241135, 0.003546],
    'Log Loss': [0.059872, 0.069646, 0.112454, 0.087955]
}
metrics_df = pd.DataFrame(performance_data).set_index('Model')

# ----- Streamlit App Layout -----
st.set_page_config(page_title="Fraud Detection Interactive App", layout="wide")
st.title("Fraud Detection Interactive App")
st.markdown("""
This app allows users to interact with pre-trained fraud detection models and visualize predictions in real time.
Below are the performance metrics of the models used:
""")

# Display Performance Metrics Table in Sidebar
st.sidebar.header("Model Performance Metrics")
st.sidebar.table(metrics_df)

# Model selection from sidebar
model_choice = st.sidebar.selectbox("Select a Model", ["XGBoost", "Voting", "SVM", "RandomForest"])

# ----- Input Transaction Features -----
st.header("Input Transaction Details")
st.markdown("Enter transaction details below to simulate a fraud prediction.")

# Example input fields (in practice these should match your model's features)
income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
credit_risk_score = st.number_input("Credit Risk Score", min_value=300, max_value=850, value=650)
velocity_24h = st.number_input("Transaction Velocity (24h)", min_value=0.0, value=1.0, step=0.1)
proposed_credit_limit = st.number_input("Proposed Credit Limit", min_value=0.0, value=10000.0, step=100.0)
name_email_similarity = st.slider("Name-Email Similarity", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
# Add more fields as needed...

# Bundle inputs into a DataFrame (simulation of feature vector)
input_data = pd.DataFrame({
    'income': [income],
    'credit_risk_score': [credit_risk_score],
    'velocity_24h': [velocity_24h],
    'proposed_credit_limit': [proposed_credit_limit],
    'name_email_similarity': [name_email_similarity]
})
st.write("### Input Features")
st.write(input_data)

# ----- Threshold Adjustment -----
st.header("Decision Threshold Adjustment")
threshold = st.slider("Set Prediction Threshold", 0.0, 1.0, 0.5, step=0.01)
st.write(f"Current threshold is set to {threshold:.2f}")

# ----- Simulated Model Prediction -----
st.header("Model Prediction")
st.markdown(f"Simulated prediction for **{model_choice}**:")

# In a real app, you would load your trained model and call model.predict_proba(input_data) etc.
# Here, we simulate different predictions for each model.
if model_choice == "XGBoost":
    simulated_proba = 0.10  # 10% fraud probability
elif model_choice == "Voting":
    simulated_proba = 0.15  # 15% fraud probability
elif model_choice == "SVM":
    simulated_proba = 0.20  # 20% fraud probability
elif model_choice == "RandomForest":
    simulated_proba = 0.05  # 5% fraud probability

# Apply threshold to get predicted class
predicted_class = int(simulated_proba >= threshold)

st.write(f"**Predicted Fraud Probability:** {simulated_proba:.2%}")
st.write(f"**Predicted Class:** {'Fraud' if predicted_class == 1 else 'Non-Fraud'}")

# ----- Visualization of Prediction -----
fig, ax = plt.subplots(figsize=(6, 4))
categories = ['Fraud', 'Non-Fraud']
probs = [simulated_proba, 1 - simulated_proba]
colors = ['red', 'green']
ax.bar(categories, probs, color=colors)
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("Simulated Prediction Distribution")
st.pyplot(fig)

# ----- Model Performance Visualization -----
st.header("Overall Model Performance Comparison")
st.markdown("Below is a bar chart comparing the key performance metrics of the models.")

fig2, ax2 = plt.subplots(2, 2, figsize=(14, 10))
metrics_to_plot = ['ROC AUC', 'F1', 'Precision', 'Recall']
for i, metric in enumerate(metrics_to_plot):
    row, col = divmod(i, 2)
    sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax2[row][col], palette='coolwarm')
    ax2[row][col].set_title(metric)
    ax2[row][col].set_ylim(0, 1)
    ax2[row][col].set_xlabel("")

plt.tight_layout()
st.pyplot(fig2)

# ----- Additional Interactivity: ROC Curve and Precision-Recall Curve -----
st.header("Model Metric Curves (Simulated)")
st.markdown("""
For demonstration purposes, below are simulated ROC and Precision-Recall curves for the selected model.
In a production system, these curves would be generated from test data predictions.
""")

# Simulated ROC Curve
fpr = np.linspace(0, 1, 100)
tpr = np.sqrt(fpr)  # Dummy function for demonstration
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(fpr, tpr, label=f"{model_choice} (ROC Curve)")
ax3.plot([0, 1], [0, 1], 'k--', label="Chance")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("Simulated ROC Curve")
ax3.legend()
st.pyplot(fig3)

# Simulated Precision-Recall Curve
recall_vals = np.linspace(0, 1, 100)
precision_vals = 1 - recall_vals**2  # Dummy function for demonstration
fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.plot(recall_vals, precision_vals, label=f"{model_choice} (Precision-Recall)")
ax4.set_xlabel("Recall")
ax4.set_ylabel("Precision")
ax4.set_title("Simulated Precision-Recall Curve")
ax4.legend()
st.pyplot(fig4)

# ----- Final Note -----
st.markdown("""
**Note:** This app is a prototype. In a real deployment, the simulated predictions and curves would be replaced by outputs from your trained models and real-time data.
""")
