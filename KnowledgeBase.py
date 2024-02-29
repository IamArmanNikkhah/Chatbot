import sqlite3
from typing import List, Tuple, Any
import numpy as np
from scipy.spatial.distance import cosine

class ChatbotDatabase:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.setup_database()

    def setup_database(self):
        """Initialize the database tables with adjusted schema for unique embeddings."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS Terms (
                term_id INTEGER PRIMARY KEY,
                term TEXT UNIQUE NOT NULL
            );
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS Embeddings (
                term_id INTEGER UNIQUE NOT NULL,
                embedding BLOB UNIQUE NOT NULL,
                FOREIGN KEY (term_id) REFERENCES Terms(term_id)
            );
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS Facts (
                fact_id INTEGER PRIMARY KEY,
                term_id INTEGER NOT NULL,
                fact TEXT NOT NULL,
                FOREIGN KEY (term_id) REFERENCES Terms(term_id)
            );
        ''')
        self.connection.commit()

    def add_term_with_embedding(self, term: str, embedding: bytes):
        """Add a term along with its unique embedding."""
        self.cursor.execute('INSERT OR IGNORE INTO Terms (term) VALUES (?)', (term,))
        term_id = self.cursor.lastrowid
        if term_id:  # If the term was newly added
            self.cursor.execute('INSERT INTO Embeddings (term_id, embedding) VALUES (?, ?)', (term_id, embedding))
        else:  # If the term already existed, update its embedding
            self.cursor.execute('''
                UPDATE Embeddings
                SET embedding = ?
                WHERE term_id = (SELECT term_id FROM Terms WHERE term = ?)
            ''', (embedding, term))
        self.connection.commit()

    def add_fact(self, term: str, fact: str):
        """Add a fact associated with a term, identified by the term text."""
        self.cursor.execute('''
            INSERT INTO Facts (term_id, fact) 
            VALUES ((SELECT term_id FROM Terms WHERE term = ?), ?)
        ''', (term, fact))
        self.connection.commit()

    def retrieve_facts(self, term: str) -> List[str]:
        """Retrieve facts for a given term."""
        self.cursor.execute('''
            SELECT fact FROM Facts
            INNER JOIN Terms ON Facts.term_id = Terms.term_id
            WHERE term = ?
        ''', (term,))
        return [row[0] for row in self.cursor.fetchall()]

    def execute_query(self, query: str, params: Tuple[Any, ...] = ()) -> List[Tuple]:
        """Execute an arbitrary query for flexibility."""
        self.cursor.execute(query, params)
        return self.cursor.fetchall()
   
    def retrieve_facts_by_embedding(self, input_embedding: np.ndarray) -> list:
        """
        Retrieves facts and their similarity scores from the database based on the
        similarity of the input embedding to the stored embeddings. This function
        calculates cosine similarity between embeddings and selects the closest matches.

        Parameters:
        - input_embedding (np.ndarray): The input embedding vector.

        Returns:
        - list of tuples: Each tuple contains a fact and its corresponding similarity score.
        """
        # Fetch all embeddings from the database
        self.cursor.execute('SELECT term_id, embedding FROM Embeddings')
        embeddings = self.cursor.fetchall()

        # Calculate similarities and store them with their term_id
        similarities = []
        for term_id, stored_embedding in embeddings:
            stored_embedding_arr = np.frombuffer(stored_embedding, dtype=np.float32)
            similarity = 1 - cosine(input_embedding, stored_embedding_arr)
            similarities.append((similarity, term_id))

        # Sort based on similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Retrieve facts for the top N matches, N could be adjusted based on requirements
        top_matches = similarities[:5]  # Assuming we take the top 5 matches
        facts_with_scores = []
        for similarity, term_id in top_matches:
            self.cursor.execute('SELECT fact FROM Facts WHERE term_id = ?', (term_id,))
            facts = self.cursor.fetchall()
            for fact in facts:
                # Append both the fact and its similarity score to the result
                facts_with_scores.append((fact[0], similarity))

        return facts_with_scores

    # Additional methods for indexing and searching could be added here


# Example Usage
if __name__ == "__main__":
    db = ChatbotDatabase('chatbot_database.db')
  # Adding a term and its unique embedding
    term = "Python programming"
    embedding = b'some_binary_representation'  # Simplified for illustration
    db.add_term_with_embedding(term, embedding)

    # Adding facts related to the term
    db.add_fact(term, "Python is a high-level, interpreted programming language.")
    db.add_fact(term, "Guido van Rossum created Python.")

    # Retrieving facts for a term
    facts = db.retrieve_facts(term)
    print(facts)

