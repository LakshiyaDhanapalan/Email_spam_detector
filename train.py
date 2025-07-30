import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("mail_data_extended.csv")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['Message'].apply(preprocess_text)

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Category'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label_encoded'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))
print("Model and artifacts saved!")
