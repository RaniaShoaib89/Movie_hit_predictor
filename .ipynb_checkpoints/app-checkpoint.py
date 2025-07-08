import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and their accuracies
models = {
    "Random Forest": joblib.load("rf_model.pkl"),
    "XGBoost": joblib.load("xgb_model.pkl"),
    "Logistic Regression": joblib.load("logistic_model.pkl")
}

accuracies = {
    "Random Forest": 0.7656,
    "XGBoost": 0.7729,
    "Logistic Regression": 0.7530
}

# Feature importances (Manually stored or precomputed)
importances = {
    "Random Forest": [0.14, 0.48, 0.30, 0.08],
    "XGBoost": [0.11, 0.52, 0.28, 0.09],
    "Logistic Regression": [0.00000002, 0.047, -0.214, 0.0012]
}

features = ['budget', 'popularity', 'vote_average', 'vote_count']

# Styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1716323373551-88465fb4d986?q=80&w=1173&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }
    .block-container {
      background-color: rgba(0, 0, 0, 0.5);
      padding: 2rem;
      border-radius: 15px;
    }
    h1 {
        font-size: 3em;
        font-weight: 900;
        color: white;
        text-shadow: 3px 3px 6px black;
        text-align: center;
    }
    
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stMarkdown {
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 1.2em !important;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8)!important;
    }
    input, .stNumberInput input {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #555;
        border-radius: 6px;
        padding: 8px;
    }
    .feature-title
    {
      color: white;
    }
    div.stButton > button {
        background-color: #ff6347;
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        border: none;
        padding: 0.6em 1.4em;
        border-radius: 8px;
        transition: 0.3s ease-in-out;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        background-color: #ff4500;
        transform: scale(1.05);
    }
    .custom-alert {
        padding: 1rem;
        margin-top: 1rem;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        border-radius: 10px;
        color: white;
    }
    .success { background-color: rgba(0, 128, 0, 0.8); }
    .error { background-color: rgba(255, 0, 0, 0.8); }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎬 Movie Hit Predictor</h1>", unsafe_allow_html=True)

# Model selection
model_choice = st.selectbox("Select a Model", list(models.keys()))
model = models[model_choice]

# Accuracy display
st.markdown(f"<p class='accuracy-text'>✅ Accuracy of {model_choice}: {accuracies[model_choice]:.4f}</p>", unsafe_allow_html=True)

# User inputs
budget = st.number_input("Budget in $", min_value=0.0, step=1.0)
popularity = st.number_input("Popularity", min_value=0.0, step=0.1)
vote_average = st.number_input("Vote Average", min_value=0.0, max_value=10.0, step=0.1)
vote_count = st.number_input("Vote Count", min_value=0.0, step=1.0)




# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'budget': [budget],
        'popularity': [popularity],
        'vote_average': [vote_average],
        'vote_count': [vote_count]
    })

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.markdown("<div class='custom-alert success'>🎉 Likely a HIT!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='custom-alert error'>🚫 Likely a FLOP.</div>", unsafe_allow_html=True)


st.markdown("---")
st.markdown("<h3 class='feature-title'>📁 Upload CSV for Bulk Prediction</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        # Check if all required columns are present
        required_columns = ['budget', 'popularity', 'vote_average', 'vote_count']
        if all(col in uploaded_df.columns for col in required_columns):
            predictions = model.predict(uploaded_df[required_columns])
            uploaded_df['Prediction'] = ['🎯 HIT' if p == 1 else '❌ FLOP' for p in predictions]
            st.write("✅ Predictions:")
            st.dataframe(uploaded_df)

            # Optional: Download button
            csv = uploaded_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='predicted_movies.csv',
                mime='text/csv',
            )

        else:
            st.error("The CSV file must include the following columns: budget, popularity, vote_average, vote_count")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Feature importance plot
st.markdown(f"<h2 class='feature-title'>🔍 Feature Importance - {model_choice}</h2>", unsafe_allow_html=True)
importance_vals = importances[model_choice]
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance_vals
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
ax.set_title(f"Feature Importance ({model_choice})")
st.pyplot(fig)
