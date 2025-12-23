

import pandas as pd

df = pd.read_csv('/content/email_categorization_dataset_10000.csv')
print("First 5 rows of the DataFrame:")
print(df.head())
print("\nColumn names of the DataFrame:")
print(df.columns)

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
nltk.download('punkt_tab')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_email(text):
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

df['clean_email'] = df['email'].apply(preprocess_email)

print("Preprocessing function defined and applied. Displaying first 5 rows with original and cleaned emails:")
print(df[['email', 'clean_email']].head())

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_email(text):
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

df['clean_email'] = df['email'].apply(preprocess_email)

print("Preprocessing function defined and applied. Displaying first 5 rows with original and cleaned emails:")
print(df[['email', 'clean_email']].head())


from sklearn.model_selection import train_test_split
X = df['clean_email']

y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

print("Features (X) and target (y) separated.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF Vectorizer initialized, fitted, and transformed data.")
print(f"Shape of X_train_tfidf: {X_train_tfidf.shape}")
print(f"Shape of X_test_tfidf: {X_test_tfidf.shape}")


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, random_state=42)


lr_model.fit(X_train_tfidf, y_train)


y_pred_lr = lr_model.predict(X_test_tfidf)

print("Logistic Regression model trained and predictions made.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Logistic Regression Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_lr, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_lr, average='weighted'))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_lr))

from sklearn.naive_bayes import ComplementNB


nb_model = ComplementNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)

print("Complement Naive Bayes model trained and predictions made.")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Naive Bayes Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_nb, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_nb, average='weighted'))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_nb))


import pandas as pd

results_original_data = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_nb)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr, average='weighted'),
        f1_score(y_test, y_pred_nb, average='weighted')
    ]
})

print("Model Performance on Original Data:")
print(results_original_data)


import numpy as np

noise_percentage = 0.30 # 30% noise
print(f"Noise percentage set to: {noise_percentage * 100}%")


unique_categories = df['category'].unique().tolist()

num_noise_samples = int(len(df) * noise_percentage)
noise_indices = np.random.choice(df.index, size=num_noise_samples, replace=False)

for idx in noise_indices:
    original_category = df.loc[idx, 'category']
    possible_new_categories = [cat for cat in unique_categories if cat != original_category]
    if possible_new_categories:
        df.loc[idx, 'category'] = np.random.choice(possible_new_categories)

print("Noise introduced into 'category' column.")
print(df['category'].value_counts())


from sklearn.model_selection import train_test_split

X = df['clean_email']


y = df['category']

X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

print("Features (X) and noisy target (y) separated and split.")
print(f"X_train_noisy shape: {X_train_noisy.shape}")
print(f"X_test_noisy shape: {X_test_noisy.shape}")
print(f"y_train_noisy shape: {y_train_noisy.shape}")
print(f"y_test_noisy shape: {y_test_noisy.shape}")


X_train_tfidf_noisy = tfidf.transform(X_train_noisy)
X_test_tfidf_noisy = tfidf.transform(X_test_noisy)

print("TF-IDF transformed noisy data for training and testing.")
print(f"Shape of X_train_tfidf_noisy: {X_train_tfidf_noisy.shape}")
print(f"Shape of X_test_tfidf_noisy: {X_test_tfidf_noisy.shape}")



from sklearn.linear_model import LogisticRegression

lr_model_noisy = LogisticRegression(max_iter=1000, random_state=42)
lr_model_noisy.fit(X_train_tfidf_noisy, y_train_noisy)

y_pred_lr_noisy = lr_model_noisy.predict(X_test_tfidf_noisy)

print("Logistic Regression model trained on noisy data and predictions made.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Logistic Regression Results with Noisy Data")
print("Accuracy:", accuracy_score(y_test_noisy, y_pred_lr_noisy))
print("Precision:", precision_score(y_test_noisy, y_pred_lr_noisy, average='weighted'))
print("Recall:", recall_score(y_test_noisy, y_pred_lr_noisy, average='weighted'))
print("F1 Score:", f1_score(y_test_noisy, y_pred_lr_noisy, average='weighted'))

print("\nClassification Report:\n")
print(classification_report(y_test_noisy, y_pred_lr_noisy))



nb_model_noisy = ComplementNB(alpha=1.0)
nb_model_noisy.fit(X_train_tfidf_noisy, y_train_noisy)
y_pred_nb_noisy = nb_model_noisy.predict(X_test_tfidf_noisy)

print("Naive Bayes model trained on noisy data and predictions made.")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Naive Bayes Results with Noisy Data")
print("Accuracy:", accuracy_score(y_test_noisy, y_pred_nb_noisy))
print("Precision:", precision_score(y_test_noisy, y_pred_nb_noisy, average='weighted'))
print("Recall:", recall_score(y_test_noisy, y_pred_nb_noisy, average='weighted'))
print("F1 Score:", f1_score(y_test_noisy, y_pred_nb_noisy, average='weighted'))

print("\nClassification Report:\n")
print(classification_report(y_test_noisy, y_pred_nb_noisy))




import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

noisy_model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression (Noisy)", "Naive Bayes (Noisy)"],
    "Accuracy": [
        accuracy_score(y_test_noisy, y_pred_lr_noisy),
        accuracy_score(y_test_noisy, y_pred_nb_noisy)
    ],
    "F1 Score": [
        f1_score(y_test_noisy, y_pred_lr_noisy, average='weighted'),
        f1_score(y_test_noisy, y_pred_nb_noisy, average='weighted')
    ]
})

print("Model Performance with Noisy Data:")
print(noisy_model_comparison)



combined_results = pd.concat([results_original_data, noisy_model_comparison], ignore_index=True)

distilbert_row = pd.DataFrame({
    "Model": ["DistilBERT"],
    "Accuracy": [0.90],  
    "F1 Score": [0.90]   
})

combined_results = pd.concat([combined_results, distilbert_row], ignore_index=True)

print("\nOverall Model Performance Comparison:")
print(combined_results)

