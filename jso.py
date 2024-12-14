import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Sample data
data = {
    'skills': [
        "Python", "R", "Julia", "Java", "C++", "C#", "Ruby", "Go", "Rust", "Kotlin",
        "Swift", "JavaScript", "PHP", "Dart", "Haskell", "Clojure", "Scala", "Fortran",
        "MATLAB", "Perl"
    ],
    'experience_years': [
        '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'
    ],
    'projects': [
        'NLP Project, Image Classification', 'Chatbot Development, Data Analysis',
        'NLP Project, SQL Development', 'Java Project, Data Mining',
        'Machine Learning Model, Fraud Detection', 'Web Scraping Tool, Sentiment Analysis',
        'Data Visualization Dashboard, Sales Forecasting', 'Recommendation System, Collaborative Filtering',
        'Blockchain Application, Smart Contracts', 'Computer Vision, Object Detection',
        'Speech Recognition, Voice Commands', 'Text Summarization, News Aggregator',
        'E-commerce Website, Customer Reviews Analysis', 'Predictive Analytics, Stock Market Trends',
        'IoT Application, Sensor Data Monitoring', 'Healthcare Analytics, Patient Data Insights',
        'Time Series Analysis, Weather Prediction', 'Big Data Processing, Hadoop Cluster',
        'Augmented Reality App, 3D Modelling', 'Cybersecurity Tool, Intrusion Detection'
    ],
    'label': [
        '1', '0', '1', '0', '1', '0', '1', '0', '1', '0',
        '1', '0', '1', '0', '1', '0', '1', '0', '1', '0'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Initialize TF-IDF Vectorizer for skills and projects
vectorizer_skills = TfidfVectorizer(stop_words='english')
vectorizer_projects = TfidfVectorizer(stop_words='english')

# Vectorize skills and projects separately
skills_tfidf = vectorizer_skills.fit_transform(df['skills'])
projects_tfidf = vectorizer_projects.fit_transform(df['projects'])

# Encode experience years
label_encoder_exp = LabelEncoder()
experience_encoded = label_encoder_exp.fit_transform(df['experience_years'])

# Combine TF-IDF features and encoded experience years
X = np.hstack([skills_tfidf.toarray(), projects_tfidf.toarray(), experience_encoded.reshape(-1, 1)])
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data (use predict_proba for roc_auc_score)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Evaluate model performance
roc_auc = roc_auc_score(y_test.astype(int), y_pred_proba.astype(float))
f1 = f1_score(y_test.astype(int), y_pred.astype(int))

print(f"ROC-AUC Score: {roc_auc}")
print(f"F1 Score: {f1}")

# Save the model
joblib.dump(model, 'resume_match_model.pkl')

# Dummy values for employer requirements and resume data

dummy_skills = ['Ruby']
dummy_experience_years = ['1']  # String
dummy_projects = ['Data Visualization Dashboard, Sales Forecasting', 'Healthcare Analytics, Patient Data Insights']

# Function to calculate match score
def calculate_match_score(dummy_skills, dummy_experience_years, dummy_projects):
    def match_percentage(dummy_elements, dataset_elements):
        intersection = len(set(dummy_elements) & set(dataset_elements))
        return (intersection / len(dataset_elements)) * 1000

    skills_percentage = match_percentage(dummy_skills, df['skills'])
    experience_percentage = match_percentage(dummy_experience_years, df['experience_years'])
    projects_percentage = match_percentage(dummy_projects, df['projects'])
    print(f"the skills percentage is:",{skills_percentage})
    print(f"the skills percentage is:",{experience_percentage})
    print(f"the skills percentage is:",{projects_percentage})
    # Average the percentages to get an overall match score
    match_score = (skills_percentage + experience_percentage + projects_percentage) / 3
    return match_score

# Example usage
match_score = calculate_match_score(dummy_skills, dummy_experience_years, dummy_projects)
print(f"Match Score: {match_score}%")
