import sqlite3
from sqlite3 import Error

class ChatbotDatabase:
    def __init__(self, db_file):
        """Initialize the database connection."""
        self.conn = self.create_connection(db_file)
        self.create_table()

    def create_connection(self, db_file):
        """Create a database connection to the SQLite database."""
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)
        return conn

    def create_table(self):
        """Create Terms and Facts tables."""
        create_terms_table_sql = """CREATE TABLE IF NOT EXISTS Terms (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        term TEXT UNIQUE
                                    );"""
        create_facts_table_sql = """CREATE TABLE IF NOT EXISTS Facts (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        term_id INTEGER,
                                        fact TEXT,
                                        FOREIGN KEY (term_id) REFERENCES Terms (id)
                                    );"""
        self.execute_query(create_terms_table_sql)
        self.execute_query(create_facts_table_sql)
        self.execute_query("CREATE VIRTUAL TABLE IF NOT EXISTS FactSearch USING fts5(fact);")

    def execute_query(self, sql, params=()):
        """Execute a generic SQL query."""
        try:
            c = self.conn.cursor()
            c.execute(sql, params)
            self.conn.commit()
        except Error as e:
            print(e)

    def add_term(self, term):
        """Add a term if it's not already in the database."""
        sql = "INSERT OR IGNORE INTO Terms (term) VALUES (?)"
        self.execute_query(sql, (term,))

    def add_fact(self, term, fact):
        """Add a fact related to a term, ensuring no duplicates."""
        self.add_term(term)  # Ensure the term exists
        term_id_query = "SELECT id FROM Terms WHERE term = ?"
        c = self.conn.cursor()
        c.execute(term_id_query, (term,))
        term_id = c.fetchone()[0]

        # Check if the fact already exists for this term
        check_fact_sql = "SELECT id FROM Facts WHERE term_id = ? AND fact = ?"
        c.execute(check_fact_sql, (term_id, fact))
        if c.fetchone() is None:
            # Add the fact if it doesn't exist
            add_fact_sql = "INSERT INTO Facts (term_id, fact) VALUES (?, ?)"
            self.execute_query(add_fact_sql, (term_id, fact))
            # Add to full-text search table
            add_to_search_sql = "INSERT INTO FactSearch (fact) VALUES (?)"
            self.execute_query(add_to_search_sql, (fact,))

    def retrieve_facts(self, term):
        """Retrieve facts related to a term."""
        sql = """SELECT f.fact
                 FROM Facts f
                 JOIN Terms t ON f.term_id = t.id
                 WHERE t.term = ?"""
        c = self.conn.cursor()
        c.execute(sql, (term,))
        return c.fetchall()

    def search_facts(self, search_query):
        """Search facts using full-text search."""
        sql = "SELECT fact FROM FactSearch WHERE FactSearch MATCH ?"
        c = self.conn.cursor()
        c.execute(sql, (search_query,))
        return c.fetchall()

# Example Usage
if __name__ == "__main__":
    db = ChatbotDatabase("chatbot_database.db")
    db.add_fact("Python", "Python is a high-level, interpreted programming language.")
    db.add_fact("Python", "Python supports multiple programming paradigms.")
    print(db.retrieve_facts("Python"))
    print(db.search_facts("interpreted programming"))
