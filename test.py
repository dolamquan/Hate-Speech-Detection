from scraper import scrape_website
from scraper import save_to_csv
from scraper import clean
import pickle


def classifier(sentences):
    with open('decision_tree_model.pkl','rb') as f:
        model = pickle.load(f)
    
    with open('vectorizer','rb') as f:
        vectorizer = pickle.load(f)

    count = 0
    output_csv = 'scraped.csv'
    hate_speech = []


    for sentence in sentences:
        X = vectorizer.transform([sentence])
        prediction = model.predict(X)
        if prediction == 1:
            hate_speech.append(sentence)
            count +=1
    save_to_csv(hate_speech,output_csv)
    return count


if __name__ == "__main__":
    url = input("Enter the URL to scrape:")
    sentences = scrape_website(url)
    cleaned_sentences = clean(sentences)
    hate_speech_count = classifier(cleaned_sentences)
    print(hate_speech_count)
