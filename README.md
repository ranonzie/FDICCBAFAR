# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit][(https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)](https://fidccara.streamlit.app/)

### How to run it on your own machine


```markdown
# Credit Card Fraud Detection Model Comparison - Streamlit App

This repository contains a Streamlit app designed to compare the performance of different machine learning models for a credit card fraud detection problem. The app visualizes and presents key performance metrics such as ROC AUC, F1 score, Precision, Recall, and Log Loss for various models.

## Features
- **Model Selection:** The app allows the user to select which models to compare via a sidebar widget. Users can choose to display multiple models.
- **Data Representation:** The performance metrics are displayed in a stylized table with background gradients to visually indicate higher or lower values.
- **Visualization:** A bar chart compares the selected models' performance based on the chosen metric (e.g., ROC AUC, F1, Precision, Recall, Log Loss).
- **Metric Explanations:** An expandable section explains the meaning and importance of each metric in the context of fraud detection.

## Setup

To run this app, follow the instructions below:

### Prerequisites
- Python 3.x
- Streamlit
- Pandas
- Matplotlib
- Seaborn

You can install the required dependencies by running:

```bash
pip install streamlit pandas matplotlib seaborn
```

### Running the App

After the dependencies are installed, you can run the app by navigating to the directory where the app script (`app.py`) is located and executing:

```bash
streamlit run app.py
```

The app will open in your default web browser, where you can interact with it.

## Code Overview

### 1. Import Libraries
```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
This section imports the necessary libraries: 
- `streamlit` for building the web app,
- `pandas` for data manipulation,
- `matplotlib.pyplot` and `seaborn` for plotting.

### 2. Configure the Streamlit Page
```python
st.set_page_config(page_title="Fraud Detection Model Performance", layout="wide")
st.title("Credit Card Fraud Detection Model Comparison")
```
Here, we configure the page title and layout for a wide view, and set the main page title.

### 3. Hardcoded Performance Metrics
```python
results = {
    'Model': ['LogisticRegression', 'XGBoost', 'SVM', 'Voting',
              'GradientBoosting', 'AdaBoost', 'RandomForest', 'DecisionTree'],
    'ROC AUC': [0.881443, 0.861234, 0.859230, 0.858387,
                0.857378, 0.844937, 0.808874, 0.667226],
    'F1': [0.188679, 0.243902, 0.173913, 0.157895,
           0.205128, 0.226415, 0.108108, 0.121951],
    'Precision': [0.120968, 0.833333, 0.176471, 1.000000,
                  1.000000, 0.333333, 1.000000, 0.106383],
    'Recall': [0.428571, 0.142857, 0.171429, 0.085714,
               0.114286, 0.171429, 0.057143, 0.142857],
    'Log Loss': [0.148347, 0.066429, 0.088901, 0.063087,
                 0.066573, 0.611846, 0.115316, 0.794323]
}
```
The `results` dictionary stores hardcoded performance metrics for eight different models. Each model has performance data for ROC AUC, F1 score, Precision, Recall, and Log Loss.

### 4. DataFrame Creation
```python
results_df = pd.DataFrame(results).set_index('Model')
```
This converts the `results` dictionary into a Pandas DataFrame and sets the "Model" column as the index.

### 5. Sidebar Controls
```python
st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect(
    "Choose models to display:",
    options=results_df.index.tolist(),
    default=results_df.index.tolist()
)
```
A sidebar is created with a multiselect widget that allows the user to select which models to display. The models default to all options being selected.

### 6. Filter Results Based on Selected Models
```python
filtered_df = results_df.loc[selected_models]
```
This filters the results DataFrame to include only the models selected by the user.

### 7. Display Performance Metrics in Table
```python
styled_df = filtered_df.style.format("{:.3f}").background_gradient(
    cmap='coolwarm', subset=['ROC AUC', 'F1', 'Precision', 'Recall']
).background_gradient(
    cmap='coolwarm_r', subset=['Log Loss']
)
st.dataframe(styled_df, use_container_width=True)
```
A styled DataFrame is created to display the metrics in a visually appealing way, with background gradients to indicate high and low values.

### 8. Performance Visualization
```python
metric = st.selectbox(
    "Select metric to visualize:",
    ['ROC AUC', 'F1', 'Precision', 'Recall', 'Log Loss']
)
```
A dropdown widget allows the user to select the metric to visualize.

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=filtered_df.index,
    y=filtered_df[metric],
    palette="coolwarm",
    ax=ax
)
```
A bar plot is created using Seaborn to visualize the performance of the selected models based on the chosen metric.

### 9. Metric Explanations
```python
with st.expander("Understanding the Metrics"):
    st.markdown("""...""")
```
An expandable section provides a brief explanation of the key performance metrics.

## How to Use the App

1. **Model Selection:** Use the sidebar to select which models to display. You can select multiple models or all models by default.
2. **Metric Selection:** Once the models are chosen, select a performance metric (e.g., ROC AUC, F1, Precision, Recall, Log Loss) from the dropdown menu.
3. **View Results:** The app will display a table of the selected models and their corresponding performance metrics. The results are color-coded to highlight high and low values.
4. **Visualize Data:** A bar plot will be generated to compare the performance of the selected models based on the chosen metric.
5. **Metric Explanations:** Expand the section to learn more about each metric's significance in evaluating fraud detection models.

## Conclusion

This Streamlit app provides a simple and interactive way to compare multiple machine learning models based on key performance metrics, making it useful for evaluating and selecting the best model for fraud detection tasks.

---

**Run the app** using the command:

```bash
streamlit run app.py
```

Happy analyzing!
```
