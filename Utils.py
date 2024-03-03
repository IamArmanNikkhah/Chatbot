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
#directory = "cleaned_content"
#important_terms = extract_important_terms(directory)
#for term, score in important_terms:
#    print(f"{term}: {score}")

# Example usage
#input_dir = "scraped_content"
#output_dir = "cleaned_content"
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

#process_directory(input_dir, output_dir)


# Import necessary libraries
import requests
from bs4 import BeautifulSoup

def fetch_page_content(url):
    """
    Fetches the content of a webpage.
    
    Args:
        url (str): The URL of the webpage to fetch.
        
    Returns:
        str: The HTML content of the page.
        
    Raises:
        requests.exceptions.RequestException: If an error occurs during the request.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page content: {e}")
        return None

def parse_html(html_content):
    """
    Parses HTML content to create a BeautifulSoup object.
    
    Args:
        html_content (str): The HTML content to parse.
        
    Returns:
        BeautifulSoup: The BeautifulSoup object for parsed HTML content.
    """
    return BeautifulSoup(html_content, 'html.parser')

def extract_links(soup, base_url):
    """
    Extracts and formats Wikipedia links related to US Presidents based on the provided HTML structure.
    
    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the parsed HTML content.
        base_url (str): The base URL to append to relative links for completeness.
        
    Returns:
        list of str: A list of formatted strings containing president names and their Wikipedia links.
    """
    links = []
    # Find all 'td' elements with a 'data-sort-value' attribute, which contains the president's name
    for td in soup.find_all('td', {'data-sort-value': True}):
        # Extracting the name and the relative link
        president_name = td['data-sort-value']
        link_tag = td.find('a', href=True)
        if link_tag and president_name:
            # Construct the full URL
            full_link = base_url + link_tag['href']
            links.append(f"{president_name}: {full_link}")
    return links

def display_links(links):
    """
    Prints each link in the list on a new line.
    
    Args:
        links (list of str): The list of links to display.
    """
    for link in links:
        print(link)

# Main program function
def main():
    url = "https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States"
    base_url = "https://en.wikipedia.org"
    
    html_content = fetch_page_content(url)
    if html_content:
        soup = parse_html(html_content)
        links = extract_links(soup, base_url)
        display_links(links)
    else:
        print("Failed to fetch or parse page content.")



