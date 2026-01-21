import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

import streamlit as st
st.set_page_config(page_title="AI Email Classifier", layout="centered")


# --- NLTK Data Downloads ---
# Ensure NLTK data is available. This runs once when the app starts.
# For local deployment, NLTK data often needs to be downloaded to a specific path
# or you can add a conditional download.
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
download_nltk_data()

stop_words = set(stopwords.words('english'))

# --- 1. Data Loading and Preprocessing Functions ---
# Cache data loading to avoid re-loading on every rerun
@st.cache_data
def load_data():
    # IMPORTANT: Ensure 'email_categorization_dataset_10000.csv' is in the same directory as app.py
    try:
        df = pd.read_csv('email_categorization_dataset_10000.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'email_categorization_dataset_10000.csv' not found.")
        st.error("Please make sure the dataset file is in the same directory as your app.py.")
        st.stop() # Stop the app if data is not found

def clean_text(text):
    if pd.isna(text) or not str(text).strip():
        return ''
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def remove_signatures(text):
    signature_patterns = [
        r'--\s*\n.*',
        r'\n\s*Best regards.*',
        r'\n\s*Sincerely.*',
        r'\n\s*Thanks,.*',
        r'\n\s*Thank you,.*',
        r'\n\s*From:.*'
    ,
        r'\n\s*Sent:.*',
        r'\n\s*To:.*',
        r'\n\s*Subject:.*',
        r'\n\s*Disclaimer:.*',
        r'\n\s*Confidentiality Notice:.*'
    ]
    for pattern in signature_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()

def preprocess_email_for_category_model(text):
    if pd.isna(text) or not str(text).strip():
        return ''
    text = str(text).lower()
    text = re.sub(r'^subject:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(best regards|thanks|thank you|sincerely|regards).*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'--.*', '', text, flags=re.DOTALL)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

# --- 2. Keyword-based Urgency Detection ---
high_urgency_keywords = ['urgent', 'immediately', 'asap', 'now', 'critical', 'emergency', 'requires immediate attention', 'time-sensitive', 'deadline today', 'respond quickly', 'crucial', 'urgent action', 'must be done']
medium_urgency_keywords = ['important', 'soon', 'follow up', 'priority', 'request update', 'action required', 'review soon', 'please advise', 'due soon', 'pending', 'next step', 'update needed']
low_urgency_keywords = ['later', 'no rush', 'convenience', 'if possible', 'when you have a chance', 'at your leisure', 'eventually', 'non-urgent']

def keyword_urgency_detector(text):
    text = text.lower()
    for keyword in high_urgency_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            return 'high'
    for keyword in medium_urgency_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            return 'medium'
    return 'low'

# --- 3. Model Training (re-training in Streamlit for reproducibility) ---
@st.cache_resource # Cache model training to avoid re-training on every rerun
def train_models(df_data):
    # Apply cleaning for ML models
    df_data['cleaned_email'] = df_data['email'].apply(clean_text)
    df_data['cleaned_email'] = df_data['email'].apply(remove_signatures)

    # Category Model Setup
    df_data['clean_email_category'] = df_data['email'].apply(preprocess_email_for_category_model)
    X_cat_tfidf_text = df_data['clean_email_category']
    y_category = df_data['category']

    tfidf_cat = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=3, max_df=0.85)
    X_cat_tfidf = tfidf_cat.fit_transform(X_cat_tfidf_text)
    # No need for train/test split here as we are training on full data for app
    # and performance metrics are hardcoded for demonstration

    lr_category_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_category_model.fit(X_cat_tfidf, y_category)

    # Urgency Model Setup
    label_encoder_urgency = LabelEncoder()
    y_urgency = label_encoder_urgency.fit_transform(df_data['urgency'])

    tfidf_urg = TfidfVectorizer() 
    X_urg_tfidf = tfidf_urg.fit_transform(df_data['cleaned_email'])
    # No need for train/test split here as we are training on full data for app

    lr_urgency_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_urgency_model.fit(X_urg_tfidf, y_urgency)

    # Hardcoded performance metrics for display, based on previous notebook runs
    combined_results_df = pd.DataFrame({
        "Model": [
            "Logistic Regression (Categorization)", 
            "Naive Bayes (Categorization)", 
            "Logistic Regression (Categorization Noisy)",
            "Naive Bayes (Categorization Noisy)",
            "Combined Urgency Detection"
        ],
        "Accuracy": [
            1.0000, # From notebook output for LR on original data
            0.9455, # From notebook output for NB on original data
            0.6990, # From notebook output for LR on noisy data
            0.6640, # From notebook output for NB on noisy data
            0.7005  # From notebook output for combined urgency model
        ],
        "F1 Score": [
            1.0000, 
            0.9449, 
            0.6990,
            0.6631,
            0.7005
        ]
    })


    return lr_category_model, tfidf_cat, lr_urgency_model, tfidf_urg, label_encoder_urgency, combined_results_df

# --- Streamlit UI Layout ---
st.set_page_config(layout='wide', page_title="Email Classifier & Urgency Detector ✉️")

st.title("✉️ Smart AI Email Classifier & Urgency Detector")
st.markdown("Welcome! This interactive application categorizes emails and detects their urgency levels using Machine Learning and keyword-based rules.")

# --- How to Use Section ---
with st.expander("💡 How to Use This App"):
    st.markdown("1. **Provide an Email**: Paste the content of an email into the text area in the sidebar.")
    st.markdown("2. **Click 'Classify Email'**: The app will then process the email and predict its category and urgency.")
    st.markdown("3. **Explore Insights**: Below, you can view distributions of email categories and urgency levels in the dataset, and compare different model performances.")
    st.markdown("**Note**: The dataset `email_categorization_dataset_10000.csv` should be in the same folder as this `app.py` file.")

# Load data and train models
df = load_data()
lr_category_model, tfidf_cat, lr_urgency_model, tfidf_urg, label_encoder_urgency, combined_results = train_models(df.copy())

# --- Sidebar for Email Input ---
st.sidebar.header("📧 Enter Email for Classification")
example_email = "Subject: Urgent - Action Required on Project X deadline is today! Please review the attached document immediately. Best regards, John Doe"
user_email = st.sidebar.text_area("Paste an email here:", example_email, height=200)

if st.sidebar.button("Classify Email"):
    if user_email and user_email.strip() != example_email:
        st.subheader("Prediction Results")

        # Predict Category
        processed_email_cat = preprocess_email_for_category_model(user_email)
        email_tfidf_cat = tfidf_cat.transform([processed_email_cat])
        category_prediction = lr_category_model.predict(email_tfidf_cat)[0]
        st.success(f"**Predicted Category:** **{category_prediction.upper()}**")

        # Predict Urgency
        cleaned_email_urg = clean_text(user_email)
        cleaned_email_urg = remove_signatures(cleaned_email_urg)
        email_tfidf_urg = tfidf_urg.transform([cleaned_email_urg])
        ml_urgency_prediction_num = lr_urgency_model.predict(email_tfidf_urg)[0]
        
        keyword_pred_str = keyword_urgency_detector(cleaned_email_urg)
        keyword_pred_num = label_encoder_urgency.transform([keyword_pred_str])[0]

        final_urgency_prediction_num = ml_urgency_prediction_num
        # Prioritize 'high' urgency from keywords
        if keyword_pred_num == label_encoder_urgency.transform(['high'])[0]:
            final_urgency_prediction_num = label_encoder_urgency.transform(['high'])[0]
            
        final_urgency_prediction_label = label_encoder_urgency.inverse_transform([final_urgency_prediction_num])[0]
        
        st.info(f"**Predicted Urgency:** **{final_urgency_prediction_label.upper()}**")
    else:
        st.sidebar.warning("Please enter an email or use the example email to classify.")

# --- Dataset Overview ---
st.markdown("## Dataset Overview")
st.markdown("Here's a sneak peek at the raw data and its key characteristics.")

st.subheader("Raw Data Sample")
st.dataframe(df.head())

# --- Visualizations ---
st.markdown("## Model Insights and Visualizations")
st.markdown("Understand the data distribution and how our models are performing.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Email Category Distribution")
    category_counts = df['category'].value_counts()
    fig_cat, ax_cat = plt.subplots(figsize=(8, 6))
    category_counts.plot(kind='bar', color='skyblue', ax=ax_cat)
    ax_cat.set_title('Distribution of Email Categories')
    ax_cat.set_xlabel('Email Category')
    ax_cat.set_ylabel('Count')
    ax_cat.tick_params(axis='x', rotation=45) # ha='right' removed for compatibility
    st.pyplot(fig_cat)

with col2:
    st.subheader("Email Urgency Distribution")
    urgency_counts = df['urgency'].value_counts()
    fig_urg, ax_urg = plt.subplots(figsize=(8, 6))
    urgency_counts.plot(kind='bar', color='lightcoral', ax=ax_urg)
    ax_urg.set_title('Distribution of Email Urgency Levels')
    ax_urg.set_xlabel('Urgency Level')
    ax_urg.set_ylabel('Count')
    ax_urg.tick_params(axis='x', rotation=45) # ha='right' removed for compatibility
    st.pyplot(fig_urg)

st.subheader("Model Performance Comparison")
st.markdown("Comparison of Accuracy and F1 Scores across different categorization and urgency detection models.")
fig_perf, ax_perf = plt.subplots(figsize=(12, 7))
combined_results.plot(x='Model', y=['Accuracy', 'F1 Score'], kind='bar', ax=ax_perf)
ax_perf.set_title('Model Performance Comparison (Accuracy and F1 Score)')
ax_perf.set_xlabel('Model')
ax_perf.set_ylabel('Score')
ax_perf.tick_params(axis='x', rotation=45) # ha='right' removed for compatibility
ax_perf.legend(loc='lower right')
plt.tight_layout() # Adjust layout to prevent labels from being cut off
st.pyplot(fig_perf)

st.markdown("--- ")
st.markdown("**Note:** This application runs locally. For real-time deployment and integration with enterprise systems, specialized cloud services (AWS, GCP, Azure) and API endpoints would be required. This Streamlit app serves as a powerful demonstration and interactive prototype.")
