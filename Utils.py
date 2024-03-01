import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



def clean_text_file(input_dir, output_dir, filename):
    """Cleans the content of a given file using NLP techniques."""
    # Load the SpaCy model for English
    nlp = spacy.load("en_core_web_sm")

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    # Read the raw text
    with open(input_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    
    # NLP Processing
    doc = nlp(raw_text.lower())  # Normalize text to lowercase
    cleaned_text = []
    
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue  # Skip stop words, punctuation, and whitespace
        cleaned_text.append(token.lemma_)  # Use lemmatized form of the word
    
    # Write the cleaned text to a new file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(" ".join(cleaned_text))


def process_directory(input_dir, output_dir):
    """Processes each file in the input directory with the cleaning function."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'): 
            clean_text_file(input_dir, output_dir, filename)
            print(f"Processed {filename}")



def extract_important_terms(directory, max_features=40):
    """Extracts important terms from cleaned text files in a directory using TF-IDF."""
    texts = []
    file_names = []

    # Read and preprocess text files
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read().lower())  # Ensure text is in lowercase
            file_names.append(file_name)

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Extract feature names and tf-idf scores
    feature_names = vectorizer.get_feature_names_out()
    scores = np.mean(tfidf_matrix, axis=0).tolist()[0]  # Mean tf-idf score for each term across all documents
    terms_scores = list(zip(feature_names, scores))

    # Sort terms by their score and select top 25 to 40
    important_terms = sorted(terms_scores, key=lambda x: x[1], reverse=True)[:max_features]

    return important_terms

# Example usage
directory = "cleaned_content"
important_terms = extract_important_terms(directory)
for term, score in important_terms:
    print(f"{term}: {score}")




# Example usage
input_dir = "scraped_content"
output_dir = "cleaned_content"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

process_directory(input_dir, output_dir)
