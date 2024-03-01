import os
import openai
import numpy as np
from typing import List, Optional
import pandas as pd

class OpenAIEmbedder:
    """
    A class to interact with the OpenAI API to obtain and process text embeddings.
    
    This class is designed to fetch embeddings from OpenAI's API, reduce the embedding
    dimension to 256, and apply L2 normalization.
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initializes the OpenAIEmbedder with a specific model.
        
        Parameters:
            model (str): The model to be used for text embeddings.
        """
        self.client = openai.OpenAI()
        self.model = model

    @staticmethod
    def normalize_l2(x: np.ndarray) -> np.ndarray:
        """
        Applies L2 normalization to an embedding.
        
        Parameters:
            x (np.ndarray): The embedding to normalize.
        
        Returns:
            np.ndarray: The L2 normalized embedding.
        """
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Obtains and processes the embedding for a given text.
        
        This method fetches the embedding, reduces its dimensionality to 256,
        and applies L2 normalization.
        
        Parameters:
            text (str): The text to get the embedding for.
        
        Returns:
            Optional[np.ndarray]: The processed embedding, or None if an error occurs.
        """
        processed_text = text.replace("\n", " ")
        try:
            response = self.client.embeddings.create(
                input=[processed_text], model=self.model, encoding_format="float"
            )
            embedding = np.array(response.data[0].embedding[:256])
            normalized_embedding = self.normalize_l2(embedding)
            return normalized_embedding
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Please try again later.")
        except openai.error.InvalidRequestError as e:
            print(f"Invalid request: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('your_input_file.csv')  # Make sure there's a 'text' column
    
    embedder = OpenAIEmbedder()
    df['embeddings'] = df['text'].apply(lambda x: list(embedder.get_embedding(x)))
    
    # Save the DataFrame with embeddings
    df.to_csv('output_with_embeddings.csv', index=False)
    print("Embeddings added and saved to output_with_embeddings.csv.")
