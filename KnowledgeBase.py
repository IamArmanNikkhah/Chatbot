import sqlite3
from typing import List, Tuple, Any

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
                embedding BLOB NOT NULL,
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

