import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(page_title="Fraud Detection Model Performance", layout="wide")
st.title("Credit Card Fraud Detection Model Comparison")

# Hardcoded performance metrics
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

# Create DataFrame
results_df = pd.DataFrame(results).set_index('Model')

# Sidebar controls
st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect(
    "Choose models to display:",
    options=results_df.index.tolist(),
    default=results_df.index.tolist()
)

# Filter results
filtered_df = results_df.loc[selected_models]

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Performance Metrics")
    # Display styled table
    styled_df = filtered_df.style.format("{:.3f}").background_gradient(
        cmap='coolwarm', subset=['ROC AUC', 'F1', 'Precision', 'Recall']
    ).background_gradient(
        cmap='coolwarm_r', subset=['Log Loss']
    )
    st.dataframe(styled_df, use_container_width=True)

with col2:
    st.header("Performance Visualization")
    
    # Metric selection
    metric = st.selectbox(
        "Select metric to visualize:",
        ['ROC AUC', 'F1', 'Precision', 'Recall', 'Log Loss']
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=filtered_df.index,
        y=filtered_df[metric],
        palette="coolwarm",
        ax=ax
    )
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1 if metric != 'Log Loss' else filtered_df['Log Loss'].max()*1.1)
    
    # Annotate values
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 9),
            textcoords='offset points'
        )
    
    st.pyplot(fig)

# Metric explanations
with st.expander("Understanding the Metrics"):
    st.markdown("""
    **ROC AUC (Area Under the ROC Curve):**
    - Measures overall classification performance across all thresholds
    - Higher values indicate better model performance (Range: 0-1)
    
    **F1 Score:**
    - Harmonic mean of precision and recall
    - Balances both false positives and false negatives
    
    **Precision:**
    - Ratio of true positives to all predicted positives
    - Measures model's accuracy in positive predictions
    
    **Recall:**
    - Ratio of true positives to all actual positives
    - Measures model's ability to find all positive instances
    
    **Log Loss:**
    - Measures prediction confidence quality
    - Lower values indicate better calibrated probabilities
    """)

# Run with: streamlit run app.py
