import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS



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

# Example usage
input_dir = "scraped_content"
output_dir = "cleaned_content"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

process_directory(input_dir, output_dir)
