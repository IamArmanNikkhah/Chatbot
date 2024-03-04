import OpenAIEmbedder
import ChatbotDatabase
import numpy as np
from typing import Optional, List, Tuple
import spacy
import xml.etree.ElementTree as ET


# Initialize NLP model for Named-entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

DATABASE_PATH = "PATH"


class ChatBot:
    def __init__(self):
        self.greetings = ["Hello! I'm here to help you learn about U.S. Presidents.",
                          "Hi there! Ask me anything about U.S. Presidents."]
        self.user_name = ""
        self.favorite_president = ""
        self.embedder = OpenAIEmbedder()
        self.database = ChatbotDatabase(DATABASE_PATH)

    def introduce_and_ask_info(self):
        print(np.random.choice(self.greetings))
        user_input = input("What's your name? ")
        self.user_name = self.extract_name(user_input)
        print(f"Nice to meet you, {self.user_name}! Which U.S. President would you like to learn about?")
        pres_input = input()
        self.favorite_president = self.extract_president(pres_input)
        self.save_preference()

    def extract_name(self, text: str) -> str:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return "there"  # Default if no name is found

    def extract_president(self, text: str) -> str:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return "Unknown"  # Default if no president is found
    
    def extract_questions(self, text: str) -> List[str]:
        """
        Extracts questions from the given text.

        Parameters:
        - text (str): The input text from which to extract questions.

        Returns:
        - List[str]: A list of strings, each a question found in the input text.
        """
        questions = []
        doc = nlp(text)
        for sent in doc.sents:
            if sent.text.strip().endswith('?'):
                questions.append(sent.text.strip())
        return questions

    def save_preference(self):
        root = ET.Element("UserPreferences")
        ET.SubElement(root, "Name").text = self.user_name
        ET.SubElement(root, "FavoritePresident").text = self.favorite_president
        tree = ET.ElementTree(root)
        tree.write("user_preferences.xml")

    
    def handle_user_query(self, query: str):
        questions = self.extract_questions(query)
    
        if not questions:  # If no questions were extracted, treat the whole input as a single question.
            questions = [query]
    
        for question in questions:
            query_embedding = self.embedder.get_embedding(question)
            if query_embedding is not None:
                facts = self.database.retrieve_facts_by_embedding(query_embedding)
                if facts:
                    print(f"For your question: '{question}'")
                    print(f"Here's something interesting: {facts[0][1]}")
                    print("You might also find these questions intriguing:")
                    for fact in facts[1:]:
                        print(f"- {fact[0]}")
                        print("\n")  # Add a newline for better readability between questions
                else:
                    print(f"I'm sorry, I couldn't find an answer for your question: '{question}'")
            else:
                print("I'm sorry, I couldn't process your request.")





if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.introduce_and_ask_info()
    while True:
        user_query = input("What would you like to know? Please format your input like a question and make sure it has ? at the end of that. ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        chatbot.handle_user_query(user_query)
