from flask import Flask, render_template, request, flash, jsonify
from scraper import scrape_website, clean, save_to_csv
import pickle
import json

app = Flask(__name__)
app.secret_key = "dolamquan1234"

def classifier(sentences):
    with open('decision_tree_model.pkl','rb') as f:
        model = pickle.load(f)
    
    with open('vectorizer','rb') as f:
        vectorizer = pickle.load(f)

    count = 0
    hate_speech = []
    output_csv = 'scraped.csv'
    for sentence in sentences:
        X = vectorizer.transform([sentence])
        prediction = model.predict(X)
        if prediction == 1:
            hate_speech.append(sentence)
    
    save_to_csv(hate_speech,output_csv)
    return hate_speech

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/', methods = ['GET','POST'])
def analyze():
    if request.method == 'POST':
        link = request.form['link']
        sentences = scrape_website(link)
        cleaned_sentence = clean(sentences)
        hate_speech = classifier(cleaned_sentence)
    
        return render_template('index.html', hate_speech_sentences = hate_speech)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8000)