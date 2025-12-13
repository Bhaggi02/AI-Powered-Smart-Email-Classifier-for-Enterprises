Week 1 – Data Collection & Preprocessing

AI Powered Smart Email Classifier for Enterprises

**Infosys Internship Project**

---

📌 Week 1 Overview

Week 1 focuses on building a **strong data foundation** for the project. The objective of this week is to collect historical email data, clean and preprocess the text, and create a labeled dataset that can be used for training machine learning models in later stages.

This step is critical because the performance of NLP and ML models heavily depends on the quality of the data.

---

🎯 Week 1 Objectives

* Collect historical or publicly available email datasets
* Clean and normalize raw email text
* Remove noise such as HTML tags, signatures, and stopwords
* Label emails with categories and urgency levels
* Prepare a machine-learning-ready dataset

---

🧩 Module Implemented (Week 1)

🔹 Module 1: Email Data Collection & Preprocessing

1. Email Data Collection

* Collected email data using CSV files
* Used publicly available or sample email datasets
* Loaded datasets into the system using Pandas in Google Colab

```python
df = pd.read_csv("emails.csv")
```

---

 2. Email Preprocessing

The raw email text was cleaned and standardized using Natural Language Processing techniques.

**Preprocessing steps include:**

* Converting text to lowercase
* Removing HTML tags and URLs
* Removing email signatures (e.g., Regards, Thanks)
* Removing special characters and numbers
* Removing stopwords using NLTK
* Applying lemmatization to normalize words

This ensures consistent and meaningful text representation for model training.

---

3. Cleaned Email Generation

* A new column `clean_email` was created containing preprocessed email text
* This column serves as the primary input for machine learning models

```python
df["clean_email"] = df["email_text"].apply(preprocess_email)
```

---

 4. Email Categorization (Labeling)

Emails were labeled into the following categories:

* Complaints
* Requests
* Feedback
* Spam

This labeling enables supervised learning for email classification in future weeks.

```python
df["category"] = df["clean_email"].apply(assign_category)
```

---

5. Urgency Tagging

Each email was assigned an urgency level based on content analysis.

**Urgency levels:**

* High
* Medium
* Low

Keyword-based logic was used to identify urgency indicators such as *urgent*, *asap*, and *immediately*.

```python
df["urgency"] = df["clean_email"].apply(assign_urgency)
```

---

📊 Week 1 Deliverables

* Cleaned and preprocessed email dataset
* Labeled dataset with category and urgency tags
* CSV file ready for machine learning training

```python
df.to_csv("processed_emails.csv", index=False)
```

---

🛠️ Tools & Technologies Used

* **Programming Language:** Python
* **Environment:** Google Colab
* **Libraries:** Pandas, NLTK, Regular Expressions (re)

---

 ✅ Week 1 Outcome

By the end of Week 1:

* A high-quality, labeled email dataset was successfully prepared
* Data is ready for feature extraction and model training
* Foundation is set for building the Email Categorization Engine in Week 2


