import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

import os
import gdown
import joblib

# === Download model if not exists ===
model_path = "random_forest_model.pkl"
if not os.path.exists(model_path):
    print("🔄 Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=14YzrsoIRHMjC8Zk5kG5Yo2EnrcLFHwlT"
    gdown.download(url=url, output=model_path, quiet=False, fuzzy=True)

# === Load model ===
rf_model = joblib.load(model_path)



# -----------------------------
# Load Data & Models
# -----------------------------
data = pd.read_csv("loan_default_cleaned.csv")
model_columns = pickle.load(open("model_columns.pkl", "rb"))
scaler = pickle.load(open("minmax_scaler.pkl", "rb"))

# Models
rf_model = pickle.load(open("random_forest_model.pkl", "rb"))
nb_model = pickle.load(open("naive_bayes_model.pkl", "rb"))
lstm_model = load_model("lstm_model.h5")

# Model Accuracies
accuracy_rf = pickle.load(open("accuracy_rf.pkl", "rb"))
accuracy_nb = pickle.load(open("accuracy_nb.pkl", "rb"))

# Feature Importance
feature_importance = pickle.load(open("feature_importance.pkl", "rb"))
# Confusion Matrices
conf_matrix_rf = pickle.load(open("conf_matrix_rf.pkl", "rb"))
conf_matrix_nb = pickle.load(open("conf_matrix_nb.pkl", "rb"))

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Loan Default Risk Alert System", layout="wide")
st.markdown("""
    <h1 style='text-align: center; background-color: #1c1c1c; padding: 10px; border-radius: 8px;'>
        <span style='color: #f1f1f1;'>💼 Real-Time Loan Default Risk Alert System</span>
    </h1>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='padding: 10px; background-color: #fdfdfd ; border-left: 6px solid #4CAF50; color: black;'>
    This system helps financial institutions assess the likelihood of a loan applicant defaulting using AI. We use:
    <ul>
        <li>🔍 <b>Random Forest & Naive Bayes</b> for classification</li>
        <li>🧠 <b>LSTM</b> for behavioral anomaly detection</li>
        <li>📊 <b>Visualizations</b> to explain patterns and improve decision making</li>
    </ul>
    Use the dropdowns in the sidebar to choose a model and a sample. The system will:
    <ul>
        <li>Predict default probability</li>
        <li>Classify risk level</li>
        <li>Compare model results</li>
        <li>Show explanations & insights</li>
    </ul>
</div>
<br>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Input
# -----------------------------
st.sidebar.header("Choose a Sample and Model")
selected_sample = st.sidebar.selectbox("Choose a sample (first 100 shown)", data["Name"].head(100))
model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest", "Naive Bayes"])

sample_data = data[data["Name"] == selected_sample]
sample_display = sample_data.drop(columns=["loan_default", "Name"])
sample_input_df = pd.get_dummies(sample_display)
missing_cols = set(model_columns) - set(sample_input_df.columns)
for col in missing_cols:
    sample_input_df[col] = 0
sample_input_df = sample_input_df[model_columns]
sample_input_df = sample_input_df.astype(float)

# -----------------------------
# Prediction Logic
# -----------------------------
input_scaled = pd.DataFrame(scaler.transform(sample_input_df), columns=sample_input_df.columns)

if model_choice == "Random Forest":
    model = rf_model
    accuracy = accuracy_rf
    conf_matrix = conf_matrix_rf
elif model_choice == "Naive Bayes":
    model = nb_model
    accuracy = accuracy_nb
    conf_matrix = conf_matrix_nb

# Show Applicant Details Before Prediction
# -----------------------------
st.markdown("### 📋 Applicant Details")
st.dataframe(sample_display.T.rename(columns={sample_display.index[0]: "Value"}), use_container_width=True)
st.markdown("---")


try:
    pred_prob = model.predict_proba(input_scaled)[0][1]
except Exception as e:
    st.error(f"Prediction Failed: {e}")
    st.stop()

pred_class = int(pred_prob >= 0.5)
pred_label = "Default" if pred_class == 1 else "Non-Default"
actual_label = "Default" if int(sample_data["loan_default"].values[0]) == 1 else "Non-Default"

# -----------------------------
# Risk Category
# -----------------------------
if pred_prob <= 0.30:
    risk_category = "Low Risk ✅"
elif pred_prob <= 0.70:
    risk_category = "Medium Risk ⚠️"
else:
    risk_category = "High Risk 🚨"

# -----------------------------
# Display Result
# -----------------------------
st.subheader("🔮 Prediction Result")

if pred_label == "Default":
    st.markdown(f"<h3 style='color: red;'>Prediction: {pred_label}</h3>", unsafe_allow_html=True)
else:
    st.markdown(f"<h3 style='color: green;'>Prediction: {pred_label}</h3>", unsafe_allow_html=True)

st.markdown(f"<h5 style='color: #888;'>→ Actual: {actual_label}</h5>", unsafe_allow_html=True)

# Explanation of what prediction means
if pred_label == "Default":
    st.markdown("""
    <div style='background-color:#fff0f0; padding:10px; border-left:5px solid red; color:black;'>
        ❌ This applicant is likely to default. Default means they may not repay the loan on time. Consider tightening approval conditions or requesting more documentation.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='background-color:#f0fff5; padding:10px; border-left:5px solid green; color:black;'>
        ✅ This applicant is predicted to repay on time. You may proceed with normal approval workflow, though further due diligence is always recommended.
    </div>
    """, unsafe_allow_html=True)

# Risk Score
st.write("#### 📉 Risk Score")
st.markdown(f"<h4>{round(pred_prob * 100, 2)}% (Range: 0% = Safe, 100% = Highly Risky)</h4>", unsafe_allow_html=True)
st.markdown("""
<div style='color: #444; color white'>
Risk Score represents the likelihood (based on historical patterns) that the applicant might default.
A higher score means a higher chance of defaulting.
</div><br>
""", unsafe_allow_html=True)

# Risk Category
st.write("#### 🧠 Risk Category")
st.write(risk_category)

# Summary
if pred_label == "Default":
    st.error("❌ Applicant May Be Risky")
else:
    st.success("✅ Applicant Looks Safe")

# -----------------------------
# Suggestions
# -----------------------------
st.write("#### 💡 Suggested Action")
if pred_label == "Default":
    st.markdown("🔎 Review the application manually, verify income and credit history.")
    st.markdown("📑 Consider asking for collateral or additional references.")
else:
    st.markdown("👍 Proceed with standard loan processing.")

# -----------------------------
# Model Info Below Suggestion
# -----------------------------
st.write("#### 🧠 Model Used")
st.write(f"Model: {model_choice}")
st.write("#### 📊 Model Accuracy")
st.write(f"{round(accuracy * 100, 2)}%")

# -----------------------------
# -----------------------------
# Anomaly Detection (LSTM)
# -----------------------------
st.markdown("---")
st.markdown("""
### 🧠 Behavior Anomaly Detection (LSTM)

This section uses an LSTM model to detect if an applicant's behavior is unusual compared to typical loan applicants.  
A high score may indicate abnormal patterns, which could be linked to risk or fraud.

**Score Range:**  
- **0.00 – 0.30**: Normal ✅  
- **0.31 – 0.70**: Slightly Unusual ⚠️  
- **Above 0.70**: Abnormal 🚨
""")




try:
    sample_scaled = scaler.transform(sample_input_df)
    sample_lstm = sample_scaled.reshape(1, 1, -1)
    anomaly_score = float(lstm_model.predict(sample_lstm).flatten()[0])

    st.write(f"**Anomaly Score**: {round(float(anomaly_score), 2)} (Range: 0 = Normal, 1 = Highly Abnormal)")
    # Anomaly Explanation Based on Score
    if anomaly_score < 0.30:
        st.success("✅ Behavior is completely normal. No anomalies detected.")
    elif 0.30 <= anomaly_score <= 0.70:
        st.warning("⚠️ Some behavioral deviations noticed. May not be risky but worth a quick review.")
    else:
        st.error("🚨 Highly unusual behavior pattern detected. Manual investigation recommended.")

except Exception as e:
    st.warning(f"LSTM Anomaly Detection Failed: {e}")

# -----------------------------
# ------------------------ Explanation Before Visualizations ------------------------
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='background-color:rgba(255, 255, 255, 0.3); padding: 15px; border-radius: 10px; border-left: 6px solid #6c63ff; color: #fdfdfd;'>
    <h4 style='margin-bottom: 10px;'> 📊 Model Insights & 📌 Visual Explanations</h4>
    <div>
        These visualizations help you understand how the AI system makes decisions:
        <ul style='margin-top: 8px;'>
            <li><b>Feature Importance</b>: Highlights the most critical factors in predicting loan default.</li>
            <li><b>Confusion Matrix</b>: Evaluates the accuracy of the model's classification.</li>
            <li><b>LSTM Loss Curve</b>: Shows how well the LSTM model trained over time.</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)



# Feature Importance
# -----------------------------
with st.expander("📊 Feature Importance"):
    st.markdown("---")
    st.markdown("""
This bar chart shows the most influential features the model uses to predict loan default risk.  
It helps identify which factors (like income, loan amount, loan term, etc.) play a major role in deciding if a person might default.  
Understanding this helps us know what the model focuses on when making its prediction.  
👉 **Higher bars mean the feature had more impact on the model's decision**.
""")
    
    try:
        # Load feature names
        with open("model_columns.pkl", "rb") as f:
            feature_names = pickle.load(f)

        # Load feature importance
        with open("feature_importance.pkl", "rb") as f:
            feature_importance = pickle.load(f)

        # Safety check: lengths must match
        if len(feature_importance) != len(feature_names):
            raise ValueError(f"Length mismatch: {len(feature_importance)} importances vs {len(feature_names)} features")

        # Create Series and plot
        feat_imp_series = pd.Series(feature_importance, index=feature_names).sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        feat_imp_series[:15].plot(kind='barh', ax=ax1, color='skyblue')
        ax1.invert_yaxis()
        ax1.set_title("Top Features Influencing Loan Default")
        st.pyplot(fig1)

    except Exception as e:
        st.warning(f"Feature Importance plot failed: {e}")



# -----------------------------
# Confusion Matrix
# -----------------------------
with st.expander("📉 Confusion Matrix"):
    st.markdown("---")
    st.markdown("""
This heatmap shows how well the selected model is performing in terms of actual vs predicted classifications.

- **True Positive (TP)**: Correctly predicted defaults  
- **True Negative (TN)**: Correctly predicted non-defaults  
- **False Positive (FP)**: Incorrectly predicted default  
- **False Negative (FN)**: Missed default cases  
""")
    
    try:
        fig2, ax2 = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        ax2.set_title(f"Confusion Matrix - {model_choice}")
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Confusion Matrix plot failed: {e}")


# -----------------------------
# LSTM Loss Curve
# -----------------------------
with st.expander("📈 LSTM Loss Curve"):
    st.markdown("---")
    st.markdown("""
This plot shows how the loss decreased over time during LSTM training, helping assess model learning performance.
""")
    
    try:
        lstm_loss = pd.read_csv("lstm_loss.csv")
        fig3, ax3 = plt.subplots()
        ax3.plot(lstm_loss['loss'], label='Training Loss', color='orange')
        ax3.set_title("LSTM Training Loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()
        st.pyplot(fig3)
    except Exception as e:
        st.info("LSTM loss curve not available or failed to load.")
