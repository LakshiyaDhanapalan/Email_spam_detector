from flask import Flask, render_template, request
import pickle
import re

# Load trained artifacts
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

app = Flask(__name__)

# Preprocess input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_text = request.form['email']
        cleaned_text = preprocess_text(email_text)
        vec_text = vectorizer.transform([cleaned_text])
        pred = model.predict(vec_text)
        prediction = label_encoder.inverse_transform(pred)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
