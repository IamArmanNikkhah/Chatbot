import os
import openai
import re


OPENAI_API_KEY = "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"

class PresidentsQA:
    """
    A class to generate questions and answers about U.S. Presidents using the OpenAI API, allowing for a specified number of question-answer pairs.

    Methods:
    generateQA(num_pairs: int) -> dict: Generates a specified number of general question-answer pairs about U.S. Presidents.
    generateQA_withTerm(term: str, num_pairs: int) -> dict: Generates a specified number of question-answer pairs about U.S. Presidents, incorporating a given term.
    """

    def __init__(self):
        """
        Initializes the PresidentsQA class by setting up the OpenAI client with an API key.
        """
        try:
            api_key = OPENAI_API_KEY
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment variables.")
            self.client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
    
    def _api_call(self, prompt):
        """
        Internal method to make a chat completion API call to OpenAI.

        Parameters:
        prompt (str): The prompt to send to the API.

        Returns:
        str: The API's response text.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "As a knowledgeable AI you provide a series of well-structured question-and-answer pairs related to U.S. presidents. Your responses should adhere to the following format:\n Q: [Clearly stated question about a U.S. president] \n A: [Concise and accurate answer to the question] \n Q: Which U.S. President signed the Emancipation Proclamation during the Civil War? \n A: Abraham Lincoln signed the Emancipation Proclamation on January 1, 1863, freeing enslaved individuals in the Confederate states."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed: {e}")
            return ""

    def _parse_qa_pairs(self, text):
      """
      Parses the text to extract question and answer pairs following a specified format.

      Parameters:
      text (str): The text containing question and answer pairs.

      Returns:
      dict: A dictionary of questions and their corresponding answers.
      """
      # Adjusted pattern to match provided output format correctly
      # Note: The pattern now properly handles the spacing and structure observed in the output.
      pattern = r"Q: (.*?)\nA: (.*?)\n?(?=Q:|$)"
      matches = re.findall(pattern, text, re.DOTALL)
      return {question.strip(): answer.strip() for question, answer in matches}


    def generateQA(self, num_pairs):
        """
        Generates a specified number of general question-answer pairs about U.S. Presidents.

        Parameters:
        num_pairs (int): The number of question-answer pairs to generate.
        """
        if num_pairs < 1:
            print("Number of pairs must be at least 1.")
            return {}
        
        prompt = f"Generate {num_pairs} question(s) and answer(s) about U.S. Presidents. Your output should strictly follow the following format:\n Q: Which U.S. President served as the 16th President of the United States and led the country during the American Civil War? \n A: Abraham Lincoln.\n Q: Which U.S. President signed the Emancipation Proclamation during the Civil War? \n A: Abraham Lincoln signed the Emancipation Proclamation on January 1, 1863, freeing enslaved individuals in the Confederate states."
        response = self._api_call(prompt)
        qa_pairs = self._parse_qa_pairs(response)
        #print(response)
        if qa_pairs:
            for q, a in qa_pairs.items():
                print(f"Q: {q}\nA: {a}\n")
            return self._parse_qa_pairs(response)
        else:
            print("Failed to generate question and answer pairs.")

    def generateQA_withTerm(self, term, num_pairs):
        """
        Generates a specified number of question-answer pairs about U.S. Presidents, incorporating a given term.

        Parameters:
        term (str): The term to include in the question or answer.
        num_pairs (int): The number of question-answer pairs to generate.
        """
        if num_pairs < 1:
            print("Number of pairs must be at least 1.")
            return {}
        
        if not term:
            print("Term is required.")
            return {}
        
        prompt = f"Generate {num_pairs} question(s) and answer(s) about U.S. Presidents that include the term '{term}' in the question. Ensure the term '{term}' is included in the question  "
        response = self._api_call(prompt)
        qa_pairs = self._parse_qa_pairs(response)
        #print(response)
        if qa_pairs:
            for q, a in qa_pairs.items():
                print(f"Q: {q}\nA: {a}\n")
            return self._parse_qa_pairs(response)
        
        else:
            print("Failed to generate question and answer pairs with specified term.")

# Example usage
if __name__ == "__main__":
    qa = PresidentsQA()
    
    general_qa_pairs = qa.generateQA(2)  # Generate 2 general Q&A pairs
    term_qa_pairs    = qa.generateQA_withTerm("Car", 5)  # Generate 2 Q&A pairs including the term "Car"
    
    # Example on how to integrate the results into a dictionary or further processing
    all_qa_pairs = {**general_qa_pairs, **term_qa_pairs}
    for question, answer in all_qa_pairs.items():
        print(f"Q: {question}\nA: {answer}\n")