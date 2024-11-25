import requests
from bs4 import BeautifulSoup
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # check if the request is succeeded
        html_content = response.text

        #Parse the HTML with Beautifulsoup --> analyze it and break it down
        soup = BeautifulSoup(html_content,'html.parser')

        #Extract text from all paragraph tags
        paragraphs = soup.find_all('p')
        sentences = [para.get_text() for para in paragraphs if para.get_text().strip()]
        return sentences
    except Exception as e:
        print(f"Error occured while scraping: {e}")
        return []

def clean(sentences):
    # Remove special characters, numbers and punctuation
    cleaned_sentences = [
        re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", sentence.lower())
        for sentence in sentences
    ]
    return cleaned_sentences

def save_to_csv(sentences,output_file):
    with open(output_file, 'w', newline='',encoding='utf-8') as file:
         writer = csv.writer(file)
         writer.writerow(['Sentence'])
         for sentence in sentences:
             writer.writerow([sentence])


