import OpenAIEmbedder
import ChatbotDatabase
import numpy as np
from typing import Optional, List, Tuple
import spacy
import xml.etree.ElementTree as ET


# Initialize NLP model for Named-entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

DATABASE_PATH = '/content/drive/MyDrive/Colab Notebooks/chatbot_database.db'


class ChatBot:
    def __init__(self):
        self.greetings = ["Hello! I'm here to help you learn about U.S. Presidents.",
                          "Hi there! Ask me anything about U.S. Presidents."]
        self.user_name = ""
        self.favorite_president = ""
        self.embedder = OpenAIEmbedder()
        self.database = ChatbotDatabase(DATABASE_PATH)
        self.similarity_threshold = 0.5

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
      # Extract questions from the query; if none, treat the entire query as a single question.
      extracted_questions = self.extract_questions(query)
    
      if not extracted_questions:
        extracted_questions = [query]
    
      # Process each question separately
      for question in extracted_questions:
        # Generate an embedding for the question
        question_embedding = self.embedder.get_embedding(question)
        
        if question_embedding is not None:
            # Retrieve relevant facts based on the embedding
            retrieved_facts = self.database.retrieve_facts_by_embedding(question_embedding)

            # Check if the top fact meets the similarity threshold
            if retrieved_facts and retrieved_facts[0][2] >= self.similarity_threshold:
                print(f"🔍 For your question: \"{question}\"")
                print(f"✨ Interesting Fact: {retrieved_facts[0][1]}")

                # Suggest additional related questions
                print("🤔 You might also find these questions intriguing:")
                for fact in retrieved_facts[1:]:
                    if fact[2] >= self.similarity_threshold - 0.2:
                        print(f"- {self.database.retrieve_term_by_term_id(fact[0])}")
                        print("\n")  # Enhance readability with a newline between questions
            else:
                 # Handle cases where no satisfying facts were found
                  print(f"🧐 For your question: \"{question}\", I couldn't find enough information.")
        else:
              # Handle cases where the query couldn't be processed
              print("❗ I'm sorry, I couldn't process your request. Please try rephrasing.")


        





if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.introduce_and_ask_info()
    while True:
        user_query = input("What would you like to know? Please format your input like a question and make sure it has ? at the end of that. ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        chatbot.handle_user_query(user_query)
